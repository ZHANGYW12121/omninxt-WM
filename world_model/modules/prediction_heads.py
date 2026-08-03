"""Prediction heads for the structured UAV-through-crowd world model.

These heads decode the shared RSSM latent feature into the three observation
branches used by the crowd task:

1. human skeletons and person presence
2. ego-centric BEV occupancy
3. UAV ego state

They are intentionally independent from the encoder implementation.  During
training they consume ``rssm.get_feat(stoch, deter)``; during imagination they
can consume prior/imagination features in exactly the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tools import weight_init_


DEFAULT_EGO_STATE_SCALE = (
    60.0,  # x_local
    30.0,  # y_local
    5.0,  # z_rel
    5.0,  # vx_local
    5.0,  # vy_local
    5.0,  # vz
    10.0,  # ax_local
    10.0,  # ay_local
    10.0,  # az
    math.pi,  # roll
    math.pi,  # pitch
    math.pi,  # yaw_rel
    60.0,  # goal_dx_body
    30.0,  # goal_dy_body
    5.0,  # goal_dz
    70.0,  # goal_distance
    math.pi,  # heading_error
)
DEFAULT_EGO_STATE_MEAN = (0.0,) * 17


@dataclass(frozen=True)
class PredictionHeadConfig:
    """Configuration shared by the three prediction heads."""

    hidden_dim: int = 256
    layers: int = 2
    dropout: float = 0.0
    max_pose_people: int = 16
    num_joints: int = 17
    pose_keypoint_threshold: float = 0.05
    smooth_l1_beta: float = 0.05
    bev_hw: tuple[int, int] = (24, 16)
    env_pos_weight: float = 3.0
    ego_state_dim: int = 17
    ego_state_mean: tuple[float, ...] = DEFAULT_EGO_STATE_MEAN
    ego_state_std: tuple[float, ...] | None = None
    ego_state_scale: tuple[float, ...] = DEFAULT_EGO_STATE_SCALE


def _build_mlp(in_dim: int, hidden_dim: int, layers: int, dropout: float) -> nn.Sequential:
    modules: list[nn.Module] = []
    cur = int(in_dim)
    hidden_dim = int(hidden_dim)
    for i in range(max(1, int(layers))):
        modules.append(nn.RMSNorm(cur, eps=1e-04, dtype=torch.float32))
        modules.append(nn.Linear(cur, hidden_dim, bias=True))
        modules.append(nn.SiLU(inplace=True))
        if float(dropout) > 0.0:
            modules.append(nn.Dropout(float(dropout)))
        cur = hidden_dim
    return nn.Sequential(*modules)


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    weight = weight.to(dtype=loss.dtype, device=loss.device)
    return (loss * weight).sum() / weight.sum().clamp_min(eps)


def _align_people(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the person dimension to the common min size."""

    people = min(int(pred.shape[2]), int(target.shape[2]))
    return pred[:, :, :people], target[:, :, :people]


def _current_pose_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "pose_windows" not in batch:
        raise KeyError("Human prediction loss requires batch['pose_windows'].")
    pose_windows = batch["pose_windows"]
    if pose_windows.dim() != 6:
        raise ValueError(f"Expected pose_windows [B,T,M,W,V,3], got {tuple(pose_windows.shape)}")
    return pose_windows[:, :, :, -1].float()


def _normalize_pose_xy(
    pose_xy: torch.Tensor,
    image_size_hw: torch.Tensor | None,
) -> torch.Tensor:
    """Normalize pixel keypoints to [0, 1] using per-frame image size."""

    if image_size_hw is None:
        denom = pose_xy.detach().amax(dim=(-5, -4, -3, -2), keepdim=True).clamp_min(1.0)
        return (pose_xy / denom).clamp(0.0, 1.0)
    if image_size_hw.dim() != 3 or image_size_hw.shape[-1] != 2:
        raise ValueError(f"Expected image_size_hw [B,T,2], got {tuple(image_size_hw.shape)}")
    size = image_size_hw.to(dtype=pose_xy.dtype, device=pose_xy.device).clamp_min(1.0)
    height = size[..., 0].view(*size.shape[:2], 1, 1, 1)
    width = size[..., 1].view(*size.shape[:2], 1, 1, 1)
    denom = torch.cat([width, height], dim=-1)
    return (pose_xy / denom).clamp(0.0, 1.0)


def _downsample_occupancy(target: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Downsample raw BEV occupancy to the prediction head resolution."""

    if target.dim() != 5:
        raise ValueError(f"Expected occupancy [B,T,1,H,W], got {tuple(target.shape)}")
    b, t, c, h, w = target.shape
    if c != 1:
        raise ValueError(f"Expected single-channel occupancy, got {c} channels.")
    flat = target.float().reshape(b * t, c, h, w)
    pooled = F.adaptive_max_pool2d(flat, output_size=tuple(out_hw))
    return pooled.reshape(b, t, c, *tuple(out_hw))


class HumanPredictionHead(nn.Module):
    """Decode RSSM features into per-person skeletons and presence logits."""

    def __init__(self, feat_dim: int, config: PredictionHeadConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.trunk = _build_mlp(feat_dim, hidden, config.layers, config.dropout)
        self.pose_out = nn.Linear(hidden, int(config.max_pose_people) * int(config.num_joints) * 3)
        self.person_out = nn.Linear(hidden, int(config.max_pose_people))
        self.apply(weight_init_)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.trunk(feat.float())
        leading = x.shape[:-1]
        pose_raw = self.pose_out(x).reshape(
            *leading,
            int(self.config.max_pose_people),
            int(self.config.num_joints),
            3,
        )
        return {
            "xy": torch.sigmoid(pose_raw[..., :2]),
            "keypoint_conf_logits": pose_raw[..., 2],
            "person_logits": self.person_out(x).reshape(*leading, int(self.config.max_pose_people)),
        }

    def loss(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        target_pose = _current_pose_target(batch)
        pred_xy, target_pose = _align_people(pred["xy"], target_pose)
        pred_conf = pred["keypoint_conf_logits"][:, :, : pred_xy.shape[2]]
        pred_person = pred["person_logits"][:, :, : pred_xy.shape[2]]

        target_xy = _normalize_pose_xy(target_pose[..., :2], batch.get("pose_image_size_hw"))
        target_conf = target_pose[..., 2].clamp(0.0, 1.0)

        if "pose_token_mask" in batch:
            person_target = batch["pose_token_mask"][:, :, : pred_xy.shape[2]].to(dtype=torch.bool)
        else:
            person_target = target_conf.amax(dim=-1) >= float(self.config.pose_keypoint_threshold)

        keypoint_visible = (target_conf >= float(self.config.pose_keypoint_threshold)) & person_target[..., None]
        xy_weight = target_conf * keypoint_visible.to(dtype=target_conf.dtype)
        xy_loss = F.smooth_l1_loss(
            pred_xy,
            target_xy,
            reduction="none",
            beta=float(self.config.smooth_l1_beta),
        ).sum(dim=-1)
        xy_loss = _weighted_mean(xy_loss, xy_weight)

        conf_loss_raw = F.binary_cross_entropy_with_logits(
            pred_conf,
            keypoint_visible.to(dtype=pred_conf.dtype),
            reduction="none",
        )
        conf_weight = person_target[..., None].expand_as(conf_loss_raw).to(dtype=pred_conf.dtype)
        conf_loss = _weighted_mean(conf_loss_raw, conf_weight)

        presence_loss = F.binary_cross_entropy_with_logits(
            pred_person,
            person_target.to(dtype=pred_person.dtype),
            reduction="mean",
        )

        with torch.no_grad():
            pred_person_prob = torch.sigmoid(pred_person)
            pred_visible_prob = torch.sigmoid(pred_conf)
            metrics = {
                "pred/human_person_target_frac": person_target.float().mean(),
                "pred/human_person_prob": pred_person_prob.mean(),
                "pred/human_keypoint_visible_frac": keypoint_visible.float().mean(),
                "pred/human_keypoint_prob": pred_visible_prob.mean(),
            }
        return {
            "pred_human_xy": xy_loss,
            "pred_human_keypoint": conf_loss,
            "pred_human_presence": presence_loss,
        }, metrics


class EnvironmentPredictionHead(nn.Module):
    """Decode RSSM features into ego-centric BEV occupancy logits."""

    def __init__(self, feat_dim: int, config: PredictionHeadConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.trunk = _build_mlp(feat_dim, hidden, config.layers, config.dropout)
        h, w = tuple(config.bev_hw)
        self.occupancy_out = nn.Linear(hidden, int(h) * int(w))
        self.apply(weight_init_)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.trunk(feat.float())
        h, w = tuple(self.config.bev_hw)
        logits = self.occupancy_out(x).reshape(*x.shape[:-1], 1, int(h), int(w))
        return {"occupancy_logits": logits}

    def loss(
        self,
        pred: dict[str, torch.Tensor],
        encoded: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if "pillar_occupancy" not in encoded:
            raise KeyError("Environment prediction loss requires encoded['pillar_occupancy'].")
        logits = pred["occupancy_logits"]
        target = _downsample_occupancy(encoded["pillar_occupancy"], tuple(logits.shape[-2:])).to(device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight = 1.0 + target * (float(self.config.env_pos_weight) - 1.0)
        occupancy_loss = (bce * weight).mean()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            metrics = {
                "pred/env_occupancy_target_frac": target.mean(),
                "pred/env_occupancy_prob": prob.mean(),
            }
        return {"pred_env_occupancy": occupancy_loss}, metrics


class EgoStatePredictionHead(nn.Module):
    """Decode RSSM features into normalized UAV ego state."""

    def __init__(self, feat_dim: int, config: PredictionHeadConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        self.trunk = _build_mlp(feat_dim, hidden, config.layers, config.dropout)
        self.state_out = nn.Linear(hidden, int(config.ego_state_dim))
        mean = torch.tensor(config.ego_state_mean, dtype=torch.float32)
        std_values = config.ego_state_std if config.ego_state_std is not None else config.ego_state_scale
        std = torch.tensor(std_values, dtype=torch.float32)
        if mean.numel() != int(config.ego_state_dim):
            raise ValueError(
                f"ego_state_mean length {mean.numel()} does not match ego_state_dim={config.ego_state_dim}."
            )
        if std.numel() != int(config.ego_state_dim):
            raise ValueError(
                f"ego_state_std/scale length {std.numel()} does not match ego_state_dim={config.ego_state_dim}."
            )
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std.clamp_min(1.0e-6))
        self.apply(weight_init_)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        state_norm = self.state_out(self.trunk(feat.float()))
        state = (
            state_norm * self.state_std.to(dtype=state_norm.dtype, device=state_norm.device)
            + self.state_mean.to(dtype=state_norm.dtype, device=state_norm.device)
        )
        return {
            "state_norm": state_norm,
            "state": state,
        }

    def loss(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if "ego_state" not in batch:
            raise KeyError("Ego prediction loss requires batch['ego_state'].")
        target = batch["ego_state"].float().to(device=pred["state_norm"].device)
        mean = self.state_mean.to(dtype=target.dtype, device=target.device)
        std = self.state_std.to(dtype=target.dtype, device=target.device)
        target_norm = (target - mean) / std
        state_loss = F.smooth_l1_loss(
            pred["state_norm"],
            target_norm,
            reduction="mean",
            beta=float(self.config.smooth_l1_beta),
        )
        with torch.no_grad():
            mae = (pred["state"] - target).abs().mean()
            metrics = {
                "pred/ego_state_mae": mae,
                "pred/ego_state_norm_mae": (pred["state_norm"] - target_norm).abs().mean(),
            }
        return {"pred_ego_state": state_loss}, metrics


class CrowdWorldModelPredictionHeads(nn.Module):
    """Container for human, environment and ego prediction heads."""

    def __init__(self, feat_dim: int, config: PredictionHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictionHeadConfig()
        self.human = HumanPredictionHead(feat_dim, self.config)
        self.env = EnvironmentPredictionHead(feat_dim, self.config)
        self.ego = EgoStatePredictionHead(feat_dim, self.config)

    def forward(self, feat: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "human": self.human(feat),
            "env": self.env(feat),
            "ego": self.ego(feat),
        }

    def loss(
        self,
        pred: dict[str, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
        encoded: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        human_losses, human_metrics = self.human.loss(pred["human"], batch)
        env_losses, env_metrics = self.env.loss(pred["env"], encoded)
        ego_losses, ego_metrics = self.ego.loss(pred["ego"], batch)

        losses.update(human_losses)
        losses.update(env_losses)
        losses.update(ego_losses)
        metrics.update(human_metrics)
        metrics.update(env_metrics)
        metrics.update(ego_metrics)
        return losses, metrics

    def forward_loss(
        self,
        feat: torch.Tensor,
        batch: dict[str, torch.Tensor],
        encoded: dict[str, torch.Tensor],
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pred = self(feat)
        losses, metrics = self.loss(pred, batch, encoded)
        return pred, losses, metrics


def predict_crowd_world_model_batch(
    heads: CrowdWorldModelPredictionHeads,
    feat: torch.Tensor,
    batch: dict[str, torch.Tensor],
    encoded: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convenience wrapper mirroring the encoder helper functions."""

    return heads.forward_loss(feat, batch, encoded)
