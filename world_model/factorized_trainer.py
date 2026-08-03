"""Minimal optimizer orchestration for :class:`FactorizedDreamer`.

The vanilla trainer remains untouched.  This module provides the first complete
factorized update path while its experiment/config integration is developed.
"""

from __future__ import annotations

from typing import Mapping
import time

import torch
from optim import clip_grad_agc_

from factorized_dreamer import FactorizedDreamer


class FactorizedTrainStep:
    """Execute one joint world-model and actor-critic optimization step."""

    def __init__(self, model: FactorizedDreamer, optimizer: torch.optim.Optimizer,
                 *, imag_horizon: int = 15, grad_clip: float = 100.0,
                 loss_scales: Mapping[str, float], agc: float | None = None,
                 pmin: float = 1e-3, amp_device: str | None = None,
                 slow_target_update: int = 1) -> None:
        self.model = model
        self.optimizer = optimizer
        self.imag_horizon = int(imag_horizon)
        self.grad_clip = float(grad_clip)
        self.loss_scales = {str(k): float(v) for k, v in loss_scales.items()}
        self.agc = None if agc is None else float(agc)
        self.pmin = float(pmin)
        self.amp_device = amp_device
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_device is not None and torch.device(amp_device).type == "cuda"
        )
        self.update_count = 0
        self.slow_target_update = max(1, int(slow_target_update))

    def _weighted_world_loss(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        total = next(iter(losses.values())).new_zeros(())
        for name, loss in losses.items():
            if name not in self.loss_scales:
                raise KeyError(f"Missing Hydra loss_scales.{name}; factorized losses are never implicitly weighted")
            total = total + self.loss_scales[name] * loss
        return total

    def __call__(self, batch: dict[str, torch.Tensor], initial=None) -> dict[str, float]:
        update_start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        device_type = torch.device(self.amp_device).type if self.amp_device is not None else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16,
                            enabled=self.scaler.is_enabled()):
            world_losses, states, aux = self.model.world_model_loss(batch, initial)
            self.last_states = states
            prediction_losses, predictions = self.model.prediction_loss(states, batch)
            actor_losses, imag_metrics = self.model.actor_critic_loss(
                states, aux, batch, self.imag_horizon,
            )
            all_world = {**world_losses, **prediction_losses}
            total = self._weighted_world_loss(all_world)
            for name, loss in actor_losses.items():
                if name not in self.loss_scales:
                    raise KeyError(f"Missing Hydra loss_scales.{name}")
                total = total + self.loss_scales[name] * loss
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite factorized training loss")
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        if self.agc is not None:
            clip_grad_agc_(self.model.parameters(), self.agc, self.pmin, foreach=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite factorized gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_count += 1
        if self.update_count % self.slow_target_update == 0:
            self.model.update_slow_value()
        metrics = {f"loss/{key}": float(value.detach()) for key, value in all_world.items()}
        metrics.update({f"loss/{key}": float(value.detach()) for key, value in actor_losses.items()})
        metrics.update({f"imag/{key}": float(value.detach()) for key, value in imag_metrics.items()})
        metrics.update({"loss/total": float(total.detach()), "grad_norm": float(grad_norm.detach()),
                        "opt/grad_scale": float(self.scaler.get_scale())})
        if "truncated_people" in batch:
            metrics["data/truncated_people"] = float(batch["truncated_people"].float().sum().detach())
        human_mask = batch["human_mask"].float()
        people = human_mask.sum(-1)
        metrics.update({
            "data/actual_people_mean": float(people.mean().detach()),
            "data/actual_people_min": float(people.min().detach()),
            "data/actual_people_max": float(people.max().detach()),
            "data/all_human_missing_ratio": float((people == 0).float().mean().detach()),
            "data/valid_joint_ratio": float(batch["joint_mask"].float().mean().detach())
                if "joint_mask" in batch else 0.0,
            "state/ego_latent_norm": float(states["ego"]["deter"].norm(dim=-1).mean().detach()),
            "state/human_latent_norm": float(states["human"]["deter"].norm(dim=-1).mean().detach()),
            "state/human_valid_slot_count": float(people.mean().detach()),
        })
        if "human_ids" in batch and batch["human_ids"].shape[1] > 1:
            ids = batch["human_ids"]
            valid_pair = batch["human_mask"][:, 1:] & batch["human_mask"][:, :-1]
            metrics["data/human_id_switch_count"] = float(
                ((ids[:, 1:] != ids[:, :-1]) & valid_pair).sum().detach()
            )
        if predictions:
            ego_error = torch.linalg.vector_norm(
                predictions["ego"]["next_state"][:, :-1, :3] - batch["ego_state"][:, 1:, :3], dim=-1)
            metrics["prediction/ego_position_error"] = float(ego_error.mean().detach())
            human_valid = batch["human_mask"][:, 1:].bool()
            xyz = batch["skeleton"][..., :3]
            root_true = 0.5 * (xyz[..., 11, :] + xyz[..., 12, :])
            root_error = torch.linalg.vector_norm(
                predictions["human"]["root"][:, :-1] - root_true[:, 1:], dim=-1)
            denom = human_valid.sum().clamp_min(1)
            metrics["prediction/human_root_ade"] = float(
                ((root_error * human_valid).sum() / denom).detach())
            last_valid = human_valid[:, -1]
            metrics["prediction/human_root_fde"] = float(
                ((root_error[:, -1] * last_valid).sum() / last_valid.sum().clamp_min(1)).detach())
            presence = predictions["human"]["presence_logit"][:, :-1] > 0
            tp = (presence & human_valid).sum().float()
            metrics["prediction/human_presence_precision"] = float(
                (tp / presence.sum().clamp_min(1)).detach())
            metrics["prediction/human_presence_recall"] = float(
                (tp / human_valid.sum().clamp_min(1)).detach())
        obs_weights = aux.get("encoded", {}).get("observation_attention_weights")
        if obs_weights is not None:
            prob = obs_weights.float().clamp_min(1e-8)
            metrics["attention/obs_attention_entropy"] = float(
                (-(prob * prob.log()).sum(-1)).mean().detach()
            )
            ego_query = obs_weights[:, :, :, 0]
            metrics.update({
                "attention/obs_attention_ego_to_human": float(
                    ego_query[..., 1:].sum(-1).mean().detach()),
                "attention/obs_attention_modality_mass_ego": float(
                    obs_weights[..., 0].mean().detach()),
                "attention/obs_attention_modality_mass_human": float(
                    obs_weights[..., 1:].sum(-1).mean().detach()),
            })
            total_mask = torch.cat((
                torch.ones((*aux["human_mask"].shape[:2], 1), dtype=torch.bool,
                           device=aux["human_mask"].device),
                aux["human_mask"],
            ), -1)
            metrics["attention/obs_invalid_token_attention_mass"] = float(
                (obs_weights * (~total_mask)[:, :, None, None, :]).sum(-1).mean().detach())
        latent_steps = aux.get("latent_attention", [])
        if latent_steps:
            token = torch.stack([x["latent_tokens"] for x in latent_steps], 1)
            metrics["attention/action_token_norm"] = float(token[:, :, 0].norm(dim=-1).mean().detach())
            attn = torch.stack([x["attention_weights"] for x in latent_steps], 1).float()
            action_weights = attn[:, :, :, 0]
            metrics.update({
                "attention/latent_attention_entropy": float(
                    (-(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(-1)).mean().detach()),
                "attention/action_token_attention_to_goal": float(action_weights[..., 1].mean().detach()),
                "attention/action_token_attention_to_ego": float(action_weights[..., 2].mean().detach()),
                "attention/action_token_attention_to_human": float(
                    action_weights[..., 3:].sum(-1).mean().detach()),
            })
            latent_mask = torch.stack([x["latent_mask"] for x in latent_steps], 1)
            metrics["attention/latent_invalid_token_attention_mass"] = float(
                (attn * (~latent_mask)[:, :, None, None, :]).sum(-1).mean().detach())
        metrics["runtime/world_model_update_ms"] = (time.perf_counter() - update_start) * 1000.0
        return metrics
