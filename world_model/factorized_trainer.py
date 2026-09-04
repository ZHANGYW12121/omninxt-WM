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
from modules.reward_components import REWARD_COMPONENT_KEYS, symexp
from modules.transition_event_head import HumanSafetyCritic


_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_amp_dtype(value: str) -> tuple[str, torch.dtype]:
    """Return one canonical AMP name and dtype, rejecting silent fallbacks."""
    key = str(value).strip().lower()
    if key not in _AMP_DTYPES:
        choices = ", ".join(("bfloat16", "float16", "float32"))
        raise ValueError(f"Unsupported amp_dtype={value!r}; choose {choices}")
    dtype = _AMP_DTYPES[key]
    canonical = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[dtype]
    return canonical, dtype


def actor_std_schedule_can_advance(
    imagination_metrics: Mapping[str, torch.Tensor | float],
    violation_limit: float,
) -> bool:
    """Gate std using rejected complete samples, not averaged dimensions."""
    sample_rejected = float(
        imagination_metrics["support_rejected_sample_ratio"])
    mode_rejected = float(
        imagination_metrics["mode_support_rejected_sample_ratio"])
    return (
        torch.isfinite(torch.tensor(sample_rejected)).item()
        and torch.isfinite(torch.tensor(mode_rejected)).item()
        and max(sample_rejected, mode_rejected) <= float(violation_limit)
    )


class FactorizedTrainStep:
    """Execute one joint world-model and actor-critic optimization step."""

    def __init__(self, model: FactorizedDreamer, optimizer: torch.optim.Optimizer,
                 *, imag_horizon: int = 15, grad_clip: float = 100.0,
                 loss_scales: Mapping[str, float], agc: float | None = None,
                 pmin: float = 1e-3, amp_device: str | None = None,
                 amp_dtype: str = "bfloat16", amp_init_scale: float = 1.0,
                 slow_target_update: int = 1,
                 behavior_warmup_steps: int = 0,
                 policy_ramp_steps: int = 0,
                 behavior_clone_decay_steps: int = 0,
                 behavior_clone_final_scale: float = 1.0,
                 safety_warmup_steps: int = 0,
                 safety_ramp_steps: int = 0,
                 selection_weights: Mapping[str, float] | None = None,
                 store_last_states: bool = True,
                 world_model_only: bool = False,
                 separate_risk_auxiliary: bool = False,
                 risk_optimizer: torch.optim.Optimizer | None = None,
                 risk_positive_weight_max: float = 8.0,
                 risk_calibration_scale: float = 1.0,
                 risk_balanced_scale: float = 0.5,
                 actor_std_schedule_steps: int = 0,
                 module_optimizers: Mapping[
                     str, torch.optim.Optimizer] | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.imag_horizon = int(imag_horizon)
        self.grad_clip = float(grad_clip)
        self.loss_scales = {str(k): float(v) for k, v in loss_scales.items()}
        self.agc = None if agc is None else float(agc)
        self.pmin = float(pmin)
        self.amp_device = amp_device
        self.amp_dtype_name, self.amp_dtype = resolve_amp_dtype(amp_dtype)
        self.amp_init_scale = float(amp_init_scale)
        if not self.amp_init_scale > 0.0:
            raise ValueError("amp_init_scale must be positive")
        device_type = (
            torch.device(amp_device).type if amp_device is not None else "cpu"
        )
        self.autocast_enabled = (
            device_type == "cuda" and self.amp_dtype is not torch.float32
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.autocast_enabled and self.amp_dtype is torch.float16,
            init_scale=self.amp_init_scale,
        )
        self.update_count = 0
        self.actor_local_update_count = 0
        self.actor_std_schedule_steps = max(0, int(actor_std_schedule_steps))
        self.use_actor_local_clock = bool(getattr(model, "v63_enabled", False))
        self.skipped_update_count = 0
        self.last_step_skipped = False
        self.world_optimizer_stepped = False
        self.slow_target_update = max(1, int(slow_target_update))
        self.behavior_warmup_steps = max(0, int(behavior_warmup_steps))
        self.policy_ramp_steps = max(0, int(policy_ramp_steps))
        self.behavior_clone_decay_steps = max(
            0, int(behavior_clone_decay_steps))
        self.behavior_clone_final_scale = float(behavior_clone_final_scale)
        self.behavior_clone_decay_enabled = True
        if not 0.0 <= self.behavior_clone_final_scale <= 1.0:
            raise ValueError(
                "behavior_clone_final_scale must lie in [0, 1]")
        self.safety_warmup_steps = max(0, int(safety_warmup_steps))
        self.safety_ramp_steps = max(0, int(safety_ramp_steps))
        self.selection_weights = {
            "world": 1.0,
            "action_mae": 4.0,
            "forward_mae": 2.0,
            "saturation": 4.0,
            "unsafe_vertical": 8.0,
            "risk_collision": 2.0,
            "risk_clearance": 1.0,
            "false_safe_tolerance": 0.15,
            "false_danger_tolerance": 0.30,
            "false_safe_excess": 8.0,
            "false_danger_excess": 2.0,
            "false_safe_floor": 1.0,
            "false_danger_floor": 0.25,
            "safety_collision_loss": 0.5,
            "lateral_candidate": 0.5,
            "goal_directed": 1.0,
            "imagined_safety": 1.0,
            "imagined_reward": 0.5,
            "human_fde_5": 0.25,
            "human_fde_10": 0.25,
            "clearance_mae": 0.5,
            **({} if selection_weights is None else {
                str(key): float(value) for key, value in selection_weights.items()
            }),
        }
        self.store_last_states = bool(store_last_states)
        self.world_model_only = bool(world_model_only)
        self.separate_risk_auxiliary = bool(separate_risk_auxiliary)
        self.risk_optimizer = (
            optimizer if risk_optimizer is None else risk_optimizer)
        self.risk_positive_weight_max = max(
            1.0, float(risk_positive_weight_max))
        self.risk_calibration_scale = float(risk_calibration_scale)
        self.risk_balanced_scale = float(risk_balanced_scale)
        self.last_states = None
        self.risk_aux_update_count = 0
        self.event_aux_update_count = 0
        self.actor_updates_enabled = True
        self.closed_loop_baseline_observed = False
        self.module_optimizers = (
            None if module_optimizers is None else dict(module_optimizers))
        if self.module_optimizers is not None:
            required = {"world", "actor", "task_critic", "safety_critic"}
            if set(self.module_optimizers) != required:
                raise ValueError(
                    "v6.2 module optimizers must be exactly "
                    f"{sorted(required)}")

    @property
    def core_model(self) -> FactorizedDreamer:
        """Return the underlying model when training through a DDP wrapper."""
        module = getattr(self.model, "module", self.model)
        return module

    def _scaler_found_nonfinite(
        self, optimizer: torch.optim.Optimizer | None = None,
    ) -> bool:
        """Read the overflow flag populated by ``GradScaler.unscale_``.

        PyTorch does not expose this flag publicly.  Keeping this compatibility
        shim in one place lets FP16 skip the unsafe optimizer update and call
        ``update()`` so the dynamic scale can recover.  BF16/FP32 never enter
        this path because they do not use GradScaler.
        """
        if not self.scaler.is_enabled():
            return False
        active_optimizer = self.optimizer if optimizer is None else optimizer
        found_inf = self.scaler._found_inf_per_device(  # noqa: SLF001
            active_optimizer)
        return any(float(value.item()) != 0.0 for value in found_inf.values())

    def _weighted_world_loss(
        self,
        losses: Mapping[str, torch.Tensor],
        *,
        training: bool = False,
    ) -> torch.Tensor:
        total = next(iter(losses.values())).new_zeros(())
        for name, loss in losses.items():
            if name not in self.loss_scales:
                raise KeyError(f"Missing Hydra loss_scales.{name}; factorized losses are never implicitly weighted")
            if (
                training
                and self.separate_risk_auxiliary
                and name.startswith("risk_")
            ):
                continue
            total = total + self.loss_scales[name] * loss
        return total

    def _actor_loss_scale(self, name: str) -> float:
        scale = self.loss_scales[name]
        schedule_step = (
            self.actor_local_update_count
            if getattr(self, "use_actor_local_clock", False)
            else self.update_count
        )
        if name == "behavior_clone":
            if self.behavior_clone_decay_steps <= 0:
                return scale * self.behavior_clone_final_scale
            if not getattr(self, "behavior_clone_decay_enabled", True):
                return scale
            progress = min(
                1.0,
                max(0.0, schedule_step / self.behavior_clone_decay_steps),
            )
            multiplier = (
                1.0
                + progress * (self.behavior_clone_final_scale - 1.0)
            )
            return scale * multiplier
        safety_names = {
            "safety_policy", "collision_risk", "clearance_barrier", "unsafe_speed",
            "replay_unsafe_speed", "vertical_action",
            "lateral_candidate",
            "action_smoothness",
        }
        if name in safety_names:
            if schedule_step < self.safety_warmup_steps:
                return 0.0
            if self.safety_ramp_steps <= 0:
                return scale
            progress = (
                schedule_step - self.safety_warmup_steps + 1
            ) / self.safety_ramp_steps
            return scale * min(1.0, max(0.0, progress))
        if name != "policy":
            return scale
        if schedule_step < self.behavior_warmup_steps:
            return 0.0
        if self.policy_ramp_steps <= 0:
            return scale
        progress = (
            schedule_step - self.behavior_warmup_steps + 1
        ) / self.policy_ramp_steps
        return scale * min(1.0, max(0.0, progress))

    @staticmethod
    def _module_grad_norm(module: torch.nn.Module) -> torch.Tensor | None:
        norms = [
            parameter.grad.detach().float().norm()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return None
        return torch.stack(norms).norm()

    @staticmethod
    def _prediction_metrics(
        predictions: Mapping[str, Mapping[str, torch.Tensor]],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not predictions:
            return metrics
        ego_error = torch.linalg.vector_norm(
            predictions["ego"]["next_state"][:, :-1, :3]
            - batch["ego_state"][:, 1:, :3],
            dim=-1,
        )
        metrics["prediction/ego_position_error"] = float(ego_error.mean().detach())
        human_valid = batch["human_mask"][:, 1:].bool()
        if "human_root" in batch:
            root_true = batch["human_root"][..., :3]
        else:
            from modules.skeleton_topology import hip_joint_indices
            xyz = batch["skeleton"][..., :3]
            hips = hip_joint_indices(xyz.shape[-2])
            if hips is None:
                raise ValueError(
                    "human_root is required for an unsupported joint topology")
            root_true = 0.5 * (xyz[..., hips[0], :] + xyz[..., hips[1], :])
        root_error = torch.linalg.vector_norm(
            predictions["human"]["root"][:, :-1] - root_true[:, 1:], dim=-1)
        denom = human_valid.sum().clamp_min(1)
        metrics["prediction/human_root_ade"] = float(
            ((root_error * human_valid).sum() / denom).detach())
        last_valid = human_valid[:, -1]
        metrics["prediction/human_root_fde"] = float(
            ((root_error[:, -1] * last_valid).sum()
             / last_valid.sum().clamp_min(1)).detach())
        presence = predictions["human"]["presence_logit"][:, :-1] > 0
        tp = (presence & human_valid).sum().float()
        metrics["prediction/human_presence_precision"] = float(
            (tp / presence.sum().clamp_min(1)).detach())
        metrics["prediction/human_presence_recall"] = float(
            (tp / human_valid.sum().clamp_min(1)).detach())
        return metrics

    def risk_auxiliary_step(
        self,
        batch: dict[str, torch.Tensor],
        calibration_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Update only the observational action-risk head on a balanced batch.

        The main Dreamer update must retain the replay distribution seen by the
        environment because its reward model, Critic, and Actor optimize task
        returns under that distribution.  Collision-danger oversampling is an
        auxiliary classification/calibration concern.  We therefore infer the
        posterior without gradients and backpropagate the balanced risk loss
        only into ``action_risk``.  This prevents a deliberately biased danger
        batch from changing RSSM dynamics, reward prediction, or Value targets.
        """
        update_start = time.perf_counter()
        self.last_step_skipped = False
        self.model.train()
        optimizer = self.risk_optimizer
        optimizer.zero_grad(set_to_none=True)
        device_type = (
            torch.device(self.amp_device).type
            if self.amp_device is not None else "cpu"
        )
        with torch.autocast(
            device_type=device_type,
            dtype=self.amp_dtype,
            enabled=self.autocast_enabled,
            cache_enabled=False,
        ):
            with torch.no_grad():
                _, posterior_aux, _ = self.core_model.posterior(batch)
                joint_feature = posterior_aux["joint_feat"].detach()
                # Window balancing still leaves fewer positive transitions.
                # A square-root correction is deliberately milder than the
                # old negative/positive ratio (often 20x+), which collapsed
                # the classifier into predicting danger everywhere.
                action_valid = batch.get("action_valid")
                if action_valid is None:
                    action_valid = torch.ones_like(
                        batch["future_collision"][:, :-1], dtype=torch.bool)
                else:
                    action_valid = action_valid[:, 1:].bool().reshape(
                        *action_valid[:, 1:].shape[:-1], -1).all(
                            -1, keepdim=True)
                valid = action_valid & ~batch["is_last"][:, :-1].bool()
                collision_target = batch["future_collision"][:, :-1].bool()
                positive_count = (valid & collision_target).sum().float()
                negative_count = (valid & ~collision_target).sum().float()
                empirical_positive_weight = torch.sqrt(
                    negative_count / positive_count.clamp_min(1.0)
                )
                collision_positive_weight = torch.clamp(
                    empirical_positive_weight,
                    min=1.0,
                    max=self.risk_positive_weight_max,
                )
            balanced_losses, balanced_diagnostics = (
                self.core_model.action_risk_prediction_loss(
                joint_feature,
                batch,
                collision_positive_weight=collision_positive_weight,
                normalize_collision_weights=True,
            ))
            if calibration_batch is None:
                calibration_losses = balanced_losses
                calibration_diagnostics = balanced_diagnostics
            else:
                with torch.no_grad():
                    _, calibration_aux, _ = self.core_model.posterior(
                        calibration_batch)
                    calibration_feature = calibration_aux[
                        "joint_feat"].detach()
                calibration_losses, calibration_diagnostics = (
                    self.core_model.action_risk_prediction_loss(
                        calibration_feature,
                        calibration_batch,
                        collision_positive_weight=1.0,
                        normalize_collision_weights=True,
                    ))
            names = (
                "risk_collision",
                "risk_clearance",
                "risk_false_safe",
                "risk_false_danger",
            )
            total = joint_feature.new_zeros(())
            for name in names:
                total = total + self.loss_scales[name] * (
                    self.risk_calibration_scale * calibration_losses[name]
                    + self.risk_balanced_scale * balanced_losses[name]
                )
            counterfactual_loss, counterfactual_diagnostics = (
                self.core_model.counterfactual_action_risk_objective(
                    joint_feature, batch)
            )
            total = total + (
                self.core_model.risk_counterfactual_scale
                * counterfactual_loss)
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite balanced risk auxiliary loss")
        self.scaler.scale(total).backward()
        self.scaler.unscale_(optimizer)
        fp16_overflow = self._scaler_found_nonfinite(optimizer)
        if fp16_overflow:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            self.skipped_update_count += 1
            self.last_step_skipped = True
            grad_norm = total.detach().new_zeros(())
            skipped = True
        else:
            risk_parameters = tuple(self.core_model.action_risk.parameters())
            risk_grad_norm = self._module_grad_norm(
                self.core_model.action_risk)
            if risk_grad_norm is None or not torch.isfinite(risk_grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Balanced risk auxiliary head received no finite gradient")
            if self.agc is not None:
                clip_grad_agc_(
                    risk_parameters, self.agc, self.pmin, foreach=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                risk_parameters, self.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite balanced risk auxiliary gradient norm")
            self.scaler.step(optimizer)
            self.scaler.update()
            self.risk_aux_update_count += 1
            skipped = False
        metrics = {
            f"loss/{name}": float(loss.detach())
            for name, loss in balanced_losses.items()
        }
        metrics.update({
            f"calibration_loss/{name}": float(loss.detach())
            for name, loss in calibration_losses.items()
        })
        metrics.update({
            f"diagnostic/{name}": float(value.detach())
            for name, value in balanced_diagnostics.items()
            if not name.startswith("aggregate_")
        })
        metrics.update({
            f"calibration/{name}": float(value.detach())
            for name, value in calibration_diagnostics.items()
            if not name.startswith("aggregate_")
        })
        metrics.update({
            f"counterfactual/{name}": float(value.detach())
            for name, value in counterfactual_diagnostics.items()
        })
        metrics.update({
            "loss/total": float(total.detach()),
            "loss/risk_counterfactual": float(
                counterfactual_loss.detach()),
            "grad_norm": float(grad_norm.detach()),
            "step_skipped": float(skipped),
            "update_count": float(self.risk_aux_update_count),
            "collision_positive_weight": float(
                collision_positive_weight.detach()),
            "runtime/update_ms": (
                time.perf_counter() - update_start) * 1000.0,
        })
        return metrics

    @staticmethod
    def _risk_validation_diagnostics(
        model: FactorizedDreamer,
        aux: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
        *,
        histogram_bins: int = 100,
    ) -> dict[str, float]:
        """Return counterfactual sensitivity and mergeable score histograms."""
        feature = aux["joint_feat"][:, :-1]
        recorded_action = batch["action"][:, 1:].float().clamp(-1.0, 1.0)
        prediction = model.action_risk(feature, recorded_action)
        valid = batch.get("action_valid")
        if valid is None:
            valid = torch.ones_like(recorded_action[..., :1], dtype=torch.bool)
        else:
            valid = valid[:, 1:].bool()
        valid = valid & ~batch["is_last"][:, :-1].bool()
        target = batch.get("future_collision")
        metrics: dict[str, float] = {}
        if target is not None:
            target = target[:, :-1].bool()
            probability = prediction["collision_probability"].float()
            bucket = torch.clamp(
                (probability * histogram_bins).long(),
                min=0, max=histogram_bins - 1)
            for label, mask in (
                ("positive", valid & target),
                ("negative", valid & ~target),
            ):
                counts = torch.bincount(
                    bucket.masked_select(mask), minlength=histogram_bins)
                for index, count in enumerate(counts):
                    metrics[
                        f"risk/aggregate_hist_{label}_{index:03d}"
                    ] = float(count.detach())

        if (
            getattr(model, "v63_enabled", False)
            and model.transition_event is not None
            and batch.get("transition_event_target") is not None
            and batch.get("transition_event_valid") is not None
        ):
            event = model.transition_event(feature, recorded_action)
            event_target = batch["transition_event_target"][:, :-1].long()
            event_valid = (
                valid & batch["transition_event_valid"][:, :-1].bool())
            human_target = event_target.eq(1)
            human_probability = event[
                "human_collision_probability"].float()
            metrics["transition/aggregate_valid_count"] = float(
                event_valid.sum().detach())
            metrics["transition/aggregate_human_count"] = float(
                (event_valid & human_target).sum().detach())
            metrics["transition/aggregate_brier_sum"] = float((
                (human_probability - human_target.float()).square()
                * event_valid.float()
            ).sum().detach())
            event_log_probability = torch.nn.functional.log_softmax(
                event["logits"].float(), dim=-1)
            event_one_hot = torch.nn.functional.one_hot(
                event_target.squeeze(-1),
                num_classes=event_log_probability.shape[-1],
            ).float()
            event_nll = -event_log_probability.gather(
                -1, event_target).squeeze(-1)
            event_brier = (
                event["probability"] - event_one_hot).square().sum(-1)
            metrics["transition/aggregate_multiclass_nll_sum"] = float(
                (event_nll * event_valid.squeeze(-1).float()).sum().detach())
            metrics["transition/aggregate_multiclass_brier_sum"] = float(
                (event_brier * event_valid.squeeze(-1).float()).sum().detach())
            event_prediction = event["probability"].argmax(-1)
            event_names = (
                "continue", "human_collision", "static_collision",
                "reached_goal", "other_task_terminal",
            )
            for actual_index, actual_name in enumerate(event_names):
                actual_mask = event_valid.squeeze(-1) & event_target.squeeze(
                    -1).eq(actual_index)
                metrics[
                    f"transition/aggregate_class_count_{actual_name}"
                ] = float(actual_mask.sum().detach())
                for predicted_index, predicted_name in enumerate(event_names):
                    metrics[
                        "transition/aggregate_confusion_"
                        f"{actual_name}_as_{predicted_name}"
                    ] = float((
                        actual_mask & event_prediction.eq(predicted_index)
                    ).sum().detach())
            bucket = torch.clamp(
                (human_probability * histogram_bins).long(),
                min=0, max=histogram_bins - 1)
            for label, mask in (
                ("positive", event_valid & human_target),
                ("negative", event_valid & ~human_target),
            ):
                counts = torch.bincount(
                    bucket.masked_select(mask), minlength=histogram_bins)
                for index, count in enumerate(counts):
                    metrics[
                        f"transition/aggregate_hist_{label}_{index:03d}"
                    ] = float(count.detach())
            for index in range(10):
                lower = index / 10.0
                upper = (index + 1) / 10.0
                in_bin = event_valid & (human_probability >= lower) & (
                    human_probability < upper if index < 9
                    else human_probability <= upper)
                metrics[f"transition/aggregate_ece_count_{index:02d}"] = float(
                    in_bin.sum().detach())
                metrics[f"transition/aggregate_ece_probability_{index:02d}"] = float(
                    (human_probability * in_bin.float()).sum().detach())
                metrics[f"transition/aggregate_ece_target_{index:02d}"] = float(
                    (human_target.float() * in_bin.float()).sum().detach())

        if (
            getattr(model, "v63_enabled", False)
            and isinstance(model.safety_value, HumanSafetyCritic)
            and batch.get("remaining_human_collision_target") is not None
            and batch.get("remaining_human_collision_valid") is not None
        ):
            safety_probability = model.safety_value(feature).float()
            safety_target = batch[
                "remaining_human_collision_target"][:, :-1].bool()
            safety_valid = (
                ~batch["is_last"][:, :-1].bool()
                & batch["remaining_human_collision_valid"][:, :-1].bool())
            metrics["safety/aggregate_valid_count"] = float(
                safety_valid.sum().detach())
            metrics["safety/aggregate_human_count"] = float(
                (safety_valid & safety_target).sum().detach())
            metrics["safety/aggregate_brier_sum"] = float((
                (safety_probability - safety_target.float()).square()
                * safety_valid.float()
            ).sum().detach())
            bucket = torch.clamp(
                (safety_probability * histogram_bins).long(),
                min=0, max=histogram_bins - 1)
            for label, mask in (
                ("positive", safety_valid & safety_target),
                ("negative", safety_valid & ~safety_target),
            ):
                counts = torch.bincount(
                    bucket.masked_select(mask), minlength=histogram_bins)
                for index, count in enumerate(counts):
                    metrics[
                        f"safety/aggregate_hist_{label}_{index:03d}"
                    ] = float(count.detach())
            for index in range(10):
                lower = index / 10.0
                upper = (index + 1) / 10.0
                in_bin = safety_valid & (safety_probability >= lower) & (
                    safety_probability < upper if index < 9
                    else safety_probability <= upper)
                metrics[f"safety/aggregate_ece_count_{index:02d}"] = float(
                    in_bin.sum().detach())
                metrics[
                    f"safety/aggregate_ece_probability_{index:02d}"
                ] = float(
                    (safety_probability * in_bin.float()).sum().detach())
                metrics[
                    f"safety/aggregate_ece_target_{index:02d}"
                ] = float(
                    (safety_target.float() * in_bin.float()).sum().detach())

        if model.uses_internal_action_adapter:
            actor_feature = aux["actor_feat"][:, :-1]
            policy_action = model._dist_mode(
                model.actor(actor_feature)).float()
            actor_action = model.applied_from_policy_action(
                policy_action, batch["ego_state"][:, :-1].float())
        else:
            actor_action = model._dist_mode(
                model.actor(feature)).float().clamp(-1.0, 1.0)
        masks = {"all": valid}
        clearance = batch.get("future_min_human_clearance_m")
        clearance_valid = batch.get("future_min_human_clearance_valid")
        if clearance is not None and clearance_valid is not None:
            masks["danger"] = (
                valid
                & clearance_valid[:, :-1].bool()
                & (clearance[:, :-1].float() <= model.safe_clearance_m)
            )
        for action_name, dimension in (("vx", 0), ("vy", 1)):
            if actor_action.shape[-1] <= dimension:
                continue
            low = actor_action.clone()
            high = actor_action.clone()
            low[..., dimension] = -1.0
            high[..., dimension] = 1.0
            low_prediction = model.action_risk(feature, low)
            high_prediction = model.action_risk(feature, high)
            probability_span = (
                high_prediction["collision_probability"]
                - low_prediction["collision_probability"]
            ).abs()
            clearance_span = (
                high_prediction["min_human_clearance_m"]
                - low_prediction["min_human_clearance_m"]
            ).abs()
            for subset, mask in masks.items():
                metrics[f"risk/action_probability_span_{action_name}_{subset}"] = float(
                    model._masked_mean(probability_span, mask).detach())
                metrics[f"risk/action_clearance_span_{action_name}_{subset}"] = float(
                    model._masked_mean(clearance_span, mask).detach())
        return metrics

    @staticmethod
    @torch.no_grad()
    def _counterfactual_dynamics_diagnostics(
        model: FactorizedDreamer,
        states,
        aux: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
        *,
        rollout_steps: int = 5,
        maximum_starts: int = 64,
    ) -> dict[str, float]:
        """Measure whether applied actions cause distinct imagined futures."""
        if not model.uses_internal_action_adapter:
            return {}
        flat_state = model._flatten_state_time(states)
        total = flat_state["ego"]["deter"].shape[0]
        count = min(int(maximum_starts), total)
        select = torch.arange(count, device=batch["action"].device)
        state = {
            branch: {
                key: value.index_select(0, select)
                for key, value in fields.items()
            } for branch, fields in flat_state.items()
        }
        people = aux["human_mask"].reshape(total, -1).index_select(
            0, select).bool()
        ego = batch["ego_state"].reshape(total, -1).index_select(
            0, select).float()
        goal = batch["goal_position"]
        if goal.ndim == 2:
            goal = goal[:, None].expand(-1, batch["action"].shape[1], -1)
        goal = goal.reshape(total, 3).index_select(0, select).float()
        root = batch["human_root"].reshape(
            total, batch["human_root"].shape[-2], -1).index_select(
                0, select).float()
        candidates = ego.new_tensor((
            (0.40, 0.00, 0.00),
            (0.10, 0.00, 0.00),
            (0.25, 0.35, 0.00),
            (0.25, -0.35, 0.00),
        ))
        final_ego, final_root, final_distance, final_clearance = [], [], [], []
        applied_candidates = []
        for candidate in candidates:
            imagined_state = {
                branch: {key: value.clone() for key, value in fields.items()}
                for branch, fields in state.items()
            }
            imagined_ego = ego.clone()
            imagined_root = root.clone()
            decoded_ego = model.prediction_heads.decode_ego_state(
                model.rssm.get_branch_feats(imagined_state)["ego"])
            for _ in range(int(rollout_steps)):
                policy_action = candidate[None].expand(count, -1)
                applied = model.applied_from_policy_action(
                    policy_action, imagined_ego)
                if not applied_candidates or len(applied_candidates) < len(final_ego) + 1:
                    applied_candidates.append(applied)
                next_state, _ = model.rssm.img_step(
                    imagined_state, applied, people)
                next_decoded = model.prediction_heads.decode_ego_state(
                    model.rssm.get_branch_feats(next_state)["ego"])
                next_ego = model.anchor_ego_residual(
                    imagined_ego, decoded_ego, next_decoded, applied)
                human_feature = model.rssm.get_branch_feats(next_state)[
                    "human"]
                human_prediction = model.prediction_heads.human(
                    human_feature, imagined_root,
                    current_ego=imagined_ego, next_ego=next_ego)
                imagined_root = torch.cat((
                    human_prediction["root"],
                    human_prediction["root_velocity"],
                    imagined_root[..., 6:],
                ), dim=-1)
                imagined_state = next_state
                imagined_ego = next_ego
                decoded_ego = next_decoded
            final_ego.append(imagined_ego)
            final_root.append(imagined_root[..., :3])
            distance = torch.linalg.vector_norm(
                goal - imagined_ego[..., :3], dim=-1)
            final_distance.append(distance)
            surface = torch.linalg.vector_norm(
                imagined_root[..., :2], dim=-1
            ) - model.lateral_candidate_surface_radius_m
            surface = surface.masked_fill(~people, float("inf"))
            final_clearance.append(surface.amin(dim=-1))
        ego_stack = torch.stack(final_ego, dim=1)
        root_stack = torch.stack(final_root, dim=1)
        distance_stack = torch.stack(final_distance, dim=1)
        clearance_stack = torch.stack(final_clearance, dim=1)
        human_weight = people.to(root_stack.dtype)
        left_right_human = torch.linalg.vector_norm(
            root_stack[:, 2] - root_stack[:, 3], dim=-1)
        human_span = (
            (left_right_human * human_weight).sum()
            / human_weight.sum().clamp_min(1.0))
        initial_distance = torch.linalg.vector_norm(
            goal - ego[..., :3], dim=-1)
        progress = initial_distance[:, None] - distance_stack
        clearance_span = clearance_stack.amax(1) - clearance_stack.amin(1)
        clearance_valid = people.any(dim=-1) & torch.isfinite(clearance_span)
        if model.transition_event is not None and model.next_human_clearance is not None:
            joint = aux["joint_feat"].reshape(total, -1).index_select(0, select)
            applied_stack = torch.stack(applied_candidates[:len(candidates)], dim=1)
            joint_candidates = joint[:, None].expand(-1, len(candidates), -1)
            event = model.transition_event(joint_candidates, applied_stack)
            event_human = event["human_collision_probability"].squeeze(-1)
            event_static = event[
                "event_probability/static_collision"].squeeze(-1)
            event_clearance = model.next_human_clearance(
                joint_candidates, applied_stack).squeeze(-1)
            event_human_span = (
                event_human.amax(1) - event_human.amin(1)).mean()
            event_static_span = (
                event_static.amax(1) - event_static.amin(1)).mean()
            event_clearance_span = (
                event_clearance.amax(1) - event_clearance.amin(1)).mean()
        else:
            event_human_span = clearance_span.new_zeros(())
            event_static_span = clearance_span.new_zeros(())
            event_clearance_span = clearance_span.new_zeros(())
        return {
            "causal/ego_left_right_separation_m": float(
                torch.linalg.vector_norm(
                    ego_stack[:, 2, :2] - ego_stack[:, 3, :2],
                    dim=-1).mean()),
            "causal/human_left_right_separation_m": float(human_span),
            "causal/goal_progress_candidate_span_m": float(
                (progress.amax(1) - progress.amin(1)).mean()),
            "causal/clearance_candidate_span_m": float(
                clearance_span.masked_select(clearance_valid).mean()
                if bool(clearance_valid.any())
                else clearance_span.new_zeros(())),
            "causal/forward_progress_m": float(progress[:, 0].mean()),
            "causal/slow_progress_m": float(progress[:, 1].mean()),
            "causal/event_human_action_span": float(event_human_span),
            # Diagnostic only: the dataset has no static obstacle geometry
            # with which to determine whether this action dependence points
            # in the correct direction. It must never gate Actor unfreezing.
            "diagnostic/event_static_action_span_unverified": float(
                event_static_span),
            "causal/event_clearance_action_span_m": float(event_clearance_span),
        }

    @torch.no_grad()
    def evaluate(self, batch: dict[str, torch.Tensor], initial=None) -> dict[str, float]:
        """Evaluate observed-sequence losses without mutating optimizer state."""
        model = self.core_model
        was_training = model.training
        model.eval()
        device_type = (
            torch.device(self.amp_device).type
            if self.amp_device is not None else "cpu")
        with torch.autocast(
            device_type=device_type, dtype=self.amp_dtype,
            enabled=self.autocast_enabled,
        ):
            world_losses, states, aux = model.world_model_loss(batch, initial)
            prediction_losses, predictions = model.prediction_loss(states, batch)
            all_world = {**world_losses, **prediction_losses}
            world_total = self._weighted_world_loss(all_world)
            behavior_clone, policy_metrics = model.behavior_clone_objective(
                aux, batch)
            mode_support, mode_support_metrics = (
                model.behavior_mode_support_objective(aux, batch)
                if getattr(model, "v63_enabled", False)
                else (behavior_clone.detach() * 0.0, {}))
            lateral_candidate, candidate_metrics = (
                model.lateral_candidate_objective(states, aux, batch)
            )
            goal_directed, goal_directed_metrics = (
                model.goal_directed_objective(aux, batch)
            )
            risk_diagnostics = self._risk_validation_diagnostics(
                model, aux, batch)
            causal_diagnostics = self._counterfactual_dynamics_diagnostics(
                model, states, aux, batch)
            imagination_metrics = model.actor_imagination_diagnostics(
                states, aux, batch, self.imag_horizon)
            reward_component_metrics: dict[str, torch.Tensor] = {}
            if "reward_components" in batch:
                component_prediction = symexp(
                    model.reward_components(aux["joint_feat"]))
                component_target = batch["reward_components"].float()
                component_error = (
                    component_prediction - component_target).abs()
                for index, name in enumerate(REWARD_COMPONENT_KEYS):
                    reward_component_metrics[f"mae_{name}"] = (
                        component_error[..., index].mean())
                reward_component_metrics["task_reward_mae"] = (
                    model.policy_reward(aux["joint_feat"])
                    - model.policy_reward_target(batch)
                ).abs().mean()
        if was_training:
            model.train()
        if not torch.isfinite(world_total):
            raise FloatingPointError("Non-finite factorized validation loss")
        metrics = {
            f"loss/{key}": float(value.detach())
            for key, value in all_world.items()
        }
        metrics["loss/world_total"] = float(world_total.detach())
        metrics["loss/behavior_clone"] = float(behavior_clone.detach())
        metrics["loss/mode_support"] = float(mode_support.detach())
        metrics["loss/lateral_candidate"] = float(
            lateral_candidate.detach())
        metrics["loss/goal_directed"] = float(goal_directed.detach())
        metrics.update({
            f"risk/{key}": float(value.detach())
            for key, value in aux.get("risk_metrics", {}).items()
        })
        metrics.update({
            f"transition/{key}": float(value.detach())
            for key, value in aux.get("transition_metrics", {}).items()
        })
        metrics.update({
            f"overshoot/{key}": float(value.detach())
            for key, value in aux.get("overshoot_metrics", {}).items()
        })
        metrics.update({
            f"policy/{key}": float(value.detach())
            for key, value in policy_metrics.items()
        })
        metrics.update({
            f"policy/mode_support_{key}": float(value.detach())
            for key, value in mode_support_metrics.items()
        })
        metrics.update({
            f"candidate/{key}": float(value.detach())
            for key, value in candidate_metrics.items()
        })
        metrics.update({
            f"goal_directed/{key}": float(value.detach())
            for key, value in goal_directed_metrics.items()
        })
        metrics.update(risk_diagnostics)
        metrics.update(causal_diagnostics)
        metrics.update({
            f"imag/{key}": float(value.detach())
            for key, value in imagination_metrics.items()
        })
        metrics.update({
            f"reward_components/{key}": float(value.detach())
            for key, value in reward_component_metrics.items()
        })
        metrics["selection/score"] = (
            self.selection_weights["world"] * metrics["loss/world_total"]
            + self.selection_weights["action_mae"] * metrics["policy/action_mae"]
            + self.selection_weights["forward_mae"] * metrics["policy/mae_vx"]
            + self.selection_weights["saturation"] * metrics["policy/saturation_ratio"]
            + self.selection_weights["unsafe_vertical"]
            * metrics["policy/unsafe_vertical_ratio"]
            + self.selection_weights["risk_collision"]
            * metrics.get("loss/risk_collision", 0.0)
            + self.selection_weights["risk_clearance"]
            * metrics.get("loss/risk_clearance", 0.0)
            + self.selection_weights["lateral_candidate"]
            * metrics.get("loss/lateral_candidate", 0.0)
            + self.selection_weights["goal_directed"]
            * metrics.get("loss/goal_directed", 0.0)
        )
        # Aggregate world loss preferred the unsafe late-v6 checkpoint even
        # while danger false-safe errors grew. Keep a dedicated, asymmetric
        # safety score so world-model calibration can select an earlier model
        # without changing the Actor-stage score expected by deployment.
        worst_false_safe = max(
            metrics.get("risk/false_safe_clearance_ratio", 0.0),
            metrics.get("overshoot/h5/false_safe_clearance_ratio", 0.0),
            metrics.get("overshoot/h10/false_safe_clearance_ratio", 0.0),
        )
        worst_false_danger = max(
            metrics.get("risk/false_danger_clearance_ratio", 0.0),
            metrics.get("overshoot/h5/false_danger_clearance_ratio", 0.0),
            metrics.get("overshoot/h10/false_danger_clearance_ratio", 0.0),
        )
        metrics["selection/safety_calibration_score"] = (
            self.selection_weights["false_safe_excess"]
            * max(
                0.0,
                worst_false_safe
                - self.selection_weights["false_safe_tolerance"],
            )
            + self.selection_weights["false_danger_excess"]
            * max(
                0.0,
                worst_false_danger
                - self.selection_weights["false_danger_tolerance"],
            )
            + self.selection_weights["false_safe_floor"] * worst_false_safe
            + self.selection_weights["false_danger_floor"] * worst_false_danger
            + self.selection_weights["safety_collision_loss"]
            * metrics.get("loss/risk_collision", 0.0)
        )
        metrics.update(self._prediction_metrics(predictions, batch))
        # World calibration and Actor improvement optimize different modules
        # and therefore require different validation rankings. The former
        # balances safety calibration with multi-step Human prediction; the
        # latter observes deterministic imagined reward/safety in addition to
        # in-distribution action quality.
        metrics["selection/online_world_score"] = (
            metrics["selection/safety_calibration_score"]
            + self.selection_weights["human_fde_5"]
            * metrics.get("overshoot/h5/human_fde_m", 0.0)
            + self.selection_weights["human_fde_10"]
            * metrics.get("overshoot/h10/human_fde_m", 0.0)
            + self.selection_weights["clearance_mae"]
            * metrics.get("risk/clearance_mae_m", 0.0)
        )
        metrics["selection/online_actor_score"] = (
            self.selection_weights["action_mae"]
            * metrics["policy/action_mae"]
            + self.selection_weights["forward_mae"]
            * metrics["policy/mae_vx"]
            + self.selection_weights["saturation"]
            * metrics["policy/saturation_ratio"]
            + self.selection_weights["unsafe_vertical"]
            * metrics["policy/unsafe_vertical_ratio"]
            + self.selection_weights["lateral_candidate"]
            * metrics.get("loss/lateral_candidate", 0.0)
            + self.selection_weights["goal_directed"]
            * metrics.get("loss/goal_directed", 0.0)
            + self.selection_weights["imagined_safety"]
            * metrics["imag/safety_cost_mean"]
            - self.selection_weights["imagined_reward"]
            * metrics["imag/reward_mean"]
        )
        people = batch["human_mask"].float().sum(-1)
        metrics.update({
            "data/actual_people_mean": float(people.mean().detach()),
            "data/all_human_missing_ratio": float(
                (people == 0).float().mean().detach()),
            "data/valid_joint_ratio": float(
                batch["joint_mask"].float().mean().detach())
                if "joint_mask" in batch else 0.0,
        })
        return metrics

    def __call__(self, batch: dict[str, torch.Tensor], initial=None, *,
                 actor_batch: dict[str, torch.Tensor] | None = None,
                 safety_batch: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
        if self.module_optimizers is not None:
            return self._v62_call(
                batch, initial, actor_batch=actor_batch,
                safety_batch=safety_batch)
        update_start = time.perf_counter()
        self.last_step_skipped = False
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        device_type = torch.device(self.amp_device).type if self.amp_device is not None else "cpu"
        # Actor/Value are first used under no_grad() to construct Dreamer
        # targets and then used again with gradients for policy/value losses.
        # Autocast's weight cache would otherwise reuse the no-grad casted
        # weights on the second call and silently disconnect both heads.
        with torch.autocast(
            device_type=device_type, dtype=self.amp_dtype,
            enabled=self.autocast_enabled, cache_enabled=False,
        ):
            (
                world_losses,
                states,
                aux,
                prediction_losses,
                predictions,
                actor_losses,
                imag_metrics,
            ) = self.model(
                batch, initial, self.imag_horizon, self.world_model_only)
            # Replay only needs the numeric posterior state.  Keeping the
            # graph-connected sequence here retains a complete training graph
            # after every optimizer step and raises the next step's peak CUDA
            # memory enough to OOM on long runs.
            if self.store_last_states:
                self.last_states = {
                    branch: {
                        key: value.detach()
                        for key, value in branch_state.items()
                    }
                    for branch, branch_state in states.items()
                }
            else:
                self.last_states = None
            all_world = {**world_losses, **prediction_losses}
            total = self._weighted_world_loss(all_world, training=True)
            for name, loss in actor_losses.items():
                if name not in self.loss_scales:
                    raise KeyError(f"Missing Hydra loss_scales.{name}")
                total = total + self._actor_loss_scale(name) * loss
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite factorized training loss")
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        fp16_overflow = self._scaler_found_nonfinite()
        if fp16_overflow:
            # GradScaler.step() observes the flag set by unscale_ and skips the
            # optimizer mutation. update() then lowers the dynamic scale. The
            # old code raised before these calls, so FP16 could never recover.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.last_step_skipped = True
            self.skipped_update_count += 1
            grad_norm = total.detach().new_zeros(())
        else:
            if self.world_model_only:
                human_grad_norm = self._module_grad_norm(
                    self.core_model.rssm.human_rssm)
                risk_grad_norm = self._module_grad_norm(
                    self.core_model.action_risk)
                if (
                    human_grad_norm is None
                    or not torch.isfinite(human_grad_norm)
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Human RSSM received no finite gradient")
                if risk_grad_norm is None or not torch.isfinite(risk_grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Action-risk head received no finite gradient")
                actor_grad_norm = None
                value_grad_norm = None
            else:
                actor_grad_norm = self._module_grad_norm(
                    self.core_model.actor)
                value_grad_norm = self._module_grad_norm(
                    self.core_model.value)
                if actor_grad_norm is None or not torch.isfinite(actor_grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Actor received no finite gradient")
                if value_grad_norm is None or not torch.isfinite(value_grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Value received no finite gradient")
            if self.agc is not None:
                clip_grad_agc_(self.model.parameters(), self.agc, self.pmin, foreach=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite factorized gradient norm")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_count += 1
            if (
                not self.world_model_only
                and self.update_count % self.slow_target_update == 0
            ):
                self.core_model.update_slow_value()
        metrics = {f"loss/{key}": float(value.detach()) for key, value in all_world.items()}
        metrics.update({f"loss/{key}": float(value.detach()) for key, value in actor_losses.items()})
        metrics.update({f"imag/{key}": float(value.detach()) for key, value in imag_metrics.items()})
        metrics.update({
            f"risk/{key}": float(value.detach())
            for key, value in aux.get("risk_metrics", {}).items()
        })
        metrics.update({
            f"transition/{key}": float(value.detach())
            for key, value in aux.get("transition_metrics", {}).items()
        })
        metrics.update({
            f"overshoot/{key}": float(value.detach())
            for key, value in aux.get("overshoot_metrics", {}).items()
        })
        metrics.update({
            "loss/total": float(total.detach()),
            "loss/world_total": float(
                self._weighted_world_loss(
                    all_world, training=True).detach()),
            "grad_norm": float(grad_norm.detach()),
            "opt/grad_scale": float(self.scaler.get_scale()),
            "opt/step_skipped": float(self.last_step_skipped),
            "opt/skipped_update_count": float(self.skipped_update_count),
        })
        for name in actor_losses:
            metrics[f"loss_scale/{name}"] = self._actor_loss_scale(name)
            metrics[f"loss_weighted/{name}"] = (
                self._actor_loss_scale(name) * float(actor_losses[name].detach())
            )
        if not fp16_overflow:
            if self.world_model_only:
                metrics["grad/human_rssm_norm"] = float(
                    human_grad_norm.detach())
                metrics["grad/action_risk_norm"] = float(
                    risk_grad_norm.detach())
            else:
                metrics["grad/actor_norm"] = float(actor_grad_norm.detach())
                metrics["grad/value_norm"] = float(value_grad_norm.detach())
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
        metrics.update(self._prediction_metrics(predictions, batch))
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

    def _v62_optimizer_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters,
        *,
        label: str,
    ) -> torch.Tensor:
        """Run one isolated optimizer mutation and return its clipped norm."""
        parameters = tuple(parameters)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite v6.2 {label} loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        if self._scaler_found_nonfinite(optimizer):
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            self.skipped_update_count += 1
            self.last_step_skipped = True
            return loss.detach().new_zeros(())
        if self.agc is not None:
            clip_grad_agc_(parameters, self.agc, self.pmin, foreach=True)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite v6.2 {label} gradient norm")
        self.scaler.step(optimizer)
        self.scaler.update()
        return norm

    def _v62_call(
        self, batch: dict[str, torch.Tensor], initial=None, *,
        actor_batch: dict[str, torch.Tensor] | None = None,
        safety_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Independent world/critic/Actor steps with detached posterior input."""
        update_start = time.perf_counter()
        self.last_step_skipped = False
        self.model.train()
        model = self.core_model
        optimizers = self.module_optimizers
        assert optimizers is not None
        device_type = (
            torch.device(self.amp_device).type
            if self.amp_device is not None else "cpu")

        # Encode the detached rare-event view before constructing the much
        # larger natural-replay autograd graph.  Although ``no_grad`` keeps
        # the balanced branch from updating Encoder/RSSM, running this
        # posterior while ``world_total`` is live still makes both activation
        # peaks overlap and can exhaust a 24 GiB learner GPU at the formal
        # batch/sequence size (16 x 64).  Keeping only joint_feat costs a
        # small, fixed tensor and preserves the exact detached-head contract.
        event_joint_feat: torch.Tensor | None = None
        if getattr(model, "v63_enabled", False) and safety_batch is not None:
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=self.amp_dtype,
                enabled=self.autocast_enabled, cache_enabled=False,
            ):
                _, event_aux, _ = model.posterior(safety_batch)
                event_joint_feat = event_aux["joint_feat"].detach()

        # 1) Natural replay updates only representation/dynamics/task heads.
        optimizers["world"].zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device_type, dtype=self.amp_dtype,
            enabled=self.autocast_enabled, cache_enabled=False,
        ):
            world_losses, states, aux = model.world_model_loss(batch, initial)
            prediction_losses, predictions = model.prediction_loss(
                states, batch)
            all_world = {**world_losses, **prediction_losses}
            world_train_losses = {
                name: value for name, value in all_world.items()
                if not name.startswith("risk_")
            }
            world_total = self._weighted_world_loss(
                world_train_losses, training=True)
            event_auxiliary_metrics: dict[str, torch.Tensor] = {}
            if event_joint_feat is not None and safety_batch is not None:
                # The balanced view must not move Encoder/RSSM toward an
                # artificial class prior. Only the event head receives this
                # importance-corrected NLL and ranking gradient.
                event_auxiliary_loss, event_auxiliary_metrics = (
                    model.transition_event_balanced_auxiliary_loss(
                        event_joint_feat, safety_batch))
                world_total = world_total + (
                    self.loss_scales.get(
                        "transition_event_balanced", 0.5)
                    * event_auxiliary_loss)
        world_parameters = tuple(
            parameter
            for group in optimizers["world"].param_groups
            for parameter in group["params"])
        trainable_world_parameters = tuple(
            parameter for parameter in world_parameters
            if parameter.requires_grad)
        if trainable_world_parameters:
            world_norm = self._v62_optimizer_step(
                world_total, optimizers["world"], trainable_world_parameters,
                label="world")
            self.world_optimizer_stepped = not self.last_step_skipped
        else:
            world_norm = world_total.detach().new_zeros(())
            self.world_optimizer_stepped = False

        # 2) Recompute the posterior after the world mutation, but expose only
        # detached numeric states to every downstream optimizer.
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=self.amp_dtype,
            enabled=self.autocast_enabled, cache_enabled=False,
        ):
            posterior_states, posterior_aux, _ = model.posterior(
                batch, initial)
            posterior_states = {
                branch: {
                    key: value.detach() for key, value in fields.items()
                } for branch, fields in posterior_states.items()
            }
            posterior_aux = {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in posterior_aux.items()
            }
        if self.store_last_states:
            self.last_states = posterior_states
        else:
            self.last_states = None

        with torch.autocast(
            device_type=device_type, dtype=self.amp_dtype,
            enabled=self.autocast_enabled, cache_enabled=False,
        ):
            actor_loss_fn = (
                model.actor_critic_loss_v63
                if getattr(model, "v63_enabled", False)
                else model.actor_critic_loss_v62
            )
            actor_losses, imag_metrics = actor_loss_fn(
                posterior_states, posterior_aux, batch, self.imag_horizon)

        auxiliary_batch = (
            actor_batch if actor_batch is not None else safety_batch)
        if getattr(model, "v63_enabled", False):
            # v6.3 Actor/Critics follow the natural replay distribution. A
            # danger-balanced auxiliary view would both bias probability values
            # and update ReturnEMA twice in one optimizer step.
            auxiliary_batch = None
        auxiliary_losses = None
        if auxiliary_batch is not None:
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=self.amp_dtype,
                enabled=self.autocast_enabled, cache_enabled=False,
            ):
                auxiliary_states, auxiliary_aux, _ = model.posterior(
                    auxiliary_batch)
                auxiliary_states = {
                    branch: {
                        key: value.detach() for key, value in fields.items()
                    } for branch, fields in auxiliary_states.items()
                }
                auxiliary_aux = {
                    key: (value.detach() if torch.is_tensor(value) else value)
                    for key, value in auxiliary_aux.items()
                }
            with torch.autocast(
                device_type=device_type, dtype=self.amp_dtype,
                enabled=self.autocast_enabled, cache_enabled=False,
            ):
                auxiliary_losses, auxiliary_metrics = (
                    actor_loss_fn(
                        auxiliary_states, auxiliary_aux, auxiliary_batch,
                        self.imag_horizon))
            imag_metrics.update({
                f"balanced_{name}": value
                for name, value in auxiliary_metrics.items()
            })

        task_loss = self._actor_loss_scale("value") * actor_losses["value"]
        task_norm = self._v62_optimizer_step(
            task_loss, optimizers["task_critic"], model.value.parameters(),
            label="task_critic")

        safety_value_scale = self.loss_scales.get("safety_value", 1.0)
        safety_source = (
            auxiliary_losses
            if safety_batch is not None and auxiliary_losses is not None
            else actor_losses)
        safety_critic_loss = (
            safety_value_scale * safety_source["safety_value"])
        safety_norm = self._v62_optimizer_step(
            safety_critic_loss, optimizers["safety_critic"],
            model.safety_value.parameters(), label="safety_critic")

        actor_terms = (
            (
                "policy", "behavior_clone", "gate_teacher",
                "support_policy", "mode_support",
            )
            if getattr(model, "v63_enabled", False)
            else ("policy", "behavior_clone", "gate_teacher")
        )
        actor_total = (
            self._actor_loss_scale("policy") * actor_losses["policy"])
        auxiliary_actor_source = (
            auxiliary_losses
            if actor_batch is not None and auxiliary_losses is not None
            else actor_losses)
        for name in actor_terms[1:]:
            actor_total = actor_total + (
                self._actor_loss_scale(name)
                * auxiliary_actor_source[name])
        if self.actor_updates_enabled:
            actor_skips_before = self.skipped_update_count
            actor_norm = self._v62_optimizer_step(
                actor_total, optimizers["actor"], model.actor.parameters(),
                label="actor")
            actor_stepped = self.skipped_update_count == actor_skips_before
            if actor_stepped and getattr(model, "v63_enabled", False):
                self.actor_local_update_count += 1
                if hasattr(model, "actor_local_step"):
                    model.actor_local_step.fill_(self.actor_local_update_count)
                if hasattr(model.actor, "set_std_schedule_progress"):
                    if actor_std_schedule_can_advance(
                        imag_metrics,
                        model.imagination_support_violation_limit,
                    ):
                        increment = (
                            1.0 / max(1, self.actor_std_schedule_steps)
                            if self.actor_std_schedule_steps > 0 else 1.0)
                        current_progress = float(
                            model.actor.std_schedule_progress.detach())
                        model.actor.set_std_schedule_progress(
                            min(1.0, current_progress + increment))
        else:
            actor_norm = actor_total.detach().new_zeros(())

        if not self.last_step_skipped:
            self.update_count += 1
            if getattr(model, "v63_enabled", False) and safety_batch is not None:
                self.event_aux_update_count += 1
            if self.update_count % self.slow_target_update == 0:
                model.update_slow_value()

        metrics = {
            f"loss/{key}": float(value.detach())
            for key, value in all_world.items()
        }
        metrics.update({
            f"loss/{key}": float(value.detach())
            for key, value in actor_losses.items()
        })
        metrics.update({
            f"imag/{key}": float(value.detach())
            for key, value in imag_metrics.items()
        })
        metrics.update({
            f"risk/{key}": float(value.detach())
            for key, value in aux.get("risk_metrics", {}).items()
        })
        metrics.update({
            f"transition/{key}": float(value.detach())
            for key, value in aux.get("transition_metrics", {}).items()
        })
        metrics.update({
            f"event_aux/{key}": float(value.detach())
            for key, value in event_auxiliary_metrics.items()
        })
        total = world_total + task_loss + safety_critic_loss + actor_total
        metrics.update({
            "loss/total": float(total.detach()),
            "loss/world_total": float(world_total.detach()),
            "loss/actor_total": float(actor_total.detach()),
            "loss/task_critic_total": float(task_loss.detach()),
            "loss/safety_critic_total": float(
                safety_critic_loss.detach()),
            "grad_norm": float(torch.stack((
                world_norm, task_norm, safety_norm, actor_norm)).norm()),
            "grad/world_norm": float(world_norm),
            "grad/actor_norm": float(actor_norm),
            "grad/task_critic_norm": float(task_norm),
            "grad/safety_critic_norm": float(safety_norm),
            "opt/grad_scale": float(self.scaler.get_scale()),
            "opt/step_skipped": float(self.last_step_skipped),
            "opt/skipped_update_count": float(self.skipped_update_count),
            "opt/actor_updates_enabled": float(
                self.actor_updates_enabled),
            "opt/actor_local_step": float(self.actor_local_update_count),
            "opt/event_aux_update_count": float(
                self.event_aux_update_count),
            "opt/actor_std_schedule_progress": float(
                getattr(model.actor, "std_schedule_progress", torch.zeros(())).detach()),
            "runtime/update_ms": (
                time.perf_counter() - update_start) * 1000.0,
        })
        for name in actor_terms:
            metrics[f"loss_scale/{name}"] = self._actor_loss_scale(name)
        return metrics
