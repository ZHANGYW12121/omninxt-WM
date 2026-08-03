"""Ego/Human prediction heads for the two-branch world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FactorizedPredictionConfig:
    hidden_dim: int = 256
    ego_dim: int = 14
    num_joints: int = 17
    smooth_l1_beta: float = 0.05
    yaw_sin_index: int = 12
    yaw_cos_index: int = 13
    yaw_unit_weight: float = 0.1


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype, device=value.device)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


class EgoPredictionHead(nn.Module):
    """Decode current Ego14 and predict its next-step delta."""

    def __init__(self, feat_dim: int, config: FactorizedPredictionConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(feat_dim, config.hidden_dim)
        self.current = nn.Linear(config.hidden_dim, config.ego_dim)
        self.delta = nn.Linear(config.hidden_dim, config.ego_dim)

    def _normalize_yaw(self, state: torch.Tensor) -> torch.Tensor:
        sin_i, cos_i = self.config.yaw_sin_index, self.config.yaw_cos_index
        yaw = state[..., [sin_i, cos_i]]
        norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
        normalized = yaw / norm.clamp_min(1e-6)
        fallback = torch.zeros_like(normalized)
        fallback[..., 1] = 1.0
        yaw = torch.where(norm > 1e-6, normalized, fallback)
        state = state.clone()
        state[..., sin_i], state[..., cos_i] = yaw[..., 0], yaw[..., 1]
        return state

    def decode_current(self, feat: torch.Tensor) -> torch.Tensor:
        return self._normalize_yaw(self.current(self.trunk(feat)))

    def forward(self, feat: torch.Tensor, current_state: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(feat)
        reconstruction = self._normalize_yaw(self.current(hidden))
        delta = self.delta(hidden)
        predicted = self._normalize_yaw(current_state.float() + delta)
        return {"current_state": reconstruction, "delta": delta, "next_state": predicted}

    def loss(self, pred, target):
        reconstruction = F.smooth_l1_loss(
            pred["current_state"], target.float(), beta=self.config.smooth_l1_beta,
        )
        next_state = F.smooth_l1_loss(
            pred["next_state"][:, :-1], target[:, 1:].float(),
            beta=self.config.smooth_l1_beta,
        )
        raw_next = target[:, :-1].float() + pred["delta"][:, :-1]
        sin_i, cos_i = self.config.yaw_sin_index, self.config.yaw_cos_index
        yaw_unit = (
            raw_next[..., sin_i].square() + raw_next[..., cos_i].square() - 1.0
        ).square().mean()
        return {"ego_recon": reconstruction, "ego_pred": next_state, "yaw_unit": yaw_unit}


class Human3DPredictionHead(nn.Module):
    def __init__(self, feat_dim: int, config: FactorizedPredictionConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(feat_dim, config.hidden_dim)
        self.root_delta = nn.Linear(config.hidden_dim, 3)
        self.joints = nn.Linear(config.hidden_dim, config.num_joints * 3)
        self.presence = nn.Linear(config.hidden_dim, 1)

    def forward(self, feat, current_root):
        hidden = self.trunk(feat)
        delta = self.root_delta(hidden)
        root = current_root.float() + delta
        relative = self.joints(hidden).reshape(*hidden.shape[:-1], self.config.num_joints, 3)
        return {
            "root_delta": delta, "root": root, "joint_relative": relative,
            "joints": root[..., None, :] + relative,
            "presence_logit": self.presence(hidden).squeeze(-1),
        }

    def loss(self, pred, skeleton, human_mask, joint_mask):
        xyz = skeleton[..., :3].float()
        root_target = 0.5 * (xyz[..., 11, :] + xyz[..., 12, :])
        valid_human = human_mask[:, 1:].bool()
        root_raw = F.smooth_l1_loss(
            pred["root"][:, :-1], root_target[:, 1:], reduction="none",
            beta=self.config.smooth_l1_beta,
        )
        root_loss = _masked_mean(root_raw, valid_human)
        joint_valid = joint_mask[:, 1:].bool() & valid_human[..., None]
        joint_error = torch.linalg.vector_norm(
            pred["joints"][:, :-1] - xyz[:, 1:], dim=-1,
        )
        mpjpe = _masked_mean(joint_error, joint_valid)
        presence = F.binary_cross_entropy_with_logits(
            pred["presence_logit"][:, :-1], valid_human.float(), reduction="mean",
        )
        return {"human_root": root_loss, "human_mpjpe": mpjpe, "human_presence": presence}


class FactorizedPredictionHeads(nn.Module):
    def __init__(self, ego_feat_dim, human_feat_dim,
                 config: FactorizedPredictionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FactorizedPredictionConfig()
        self.ego = EgoPredictionHead(ego_feat_dim, self.config)
        self.human = Human3DPredictionHead(human_feat_dim, self.config)

    def decode_ego_state(self, ego_feat: torch.Tensor) -> torch.Tensor:
        return self.ego.decode_current(ego_feat)

    def forward_loss(self, branch_feats, batch):
        skeleton = batch["skeleton"]
        root = 0.5 * (skeleton[..., 11, :3] + skeleton[..., 12, :3])
        pred = {
            "ego": self.ego(branch_feats["ego"], batch["ego_state"]),
            "human": self.human(branch_feats["human"], root),
        }
        losses = {}
        losses.update(self.ego.loss(pred["ego"], batch["ego_state"]))
        losses.update(self.human.loss(
            pred["human"], skeleton, batch["human_mask"], batch["joint_mask"],
        ))
        return pred, losses
