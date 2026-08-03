"""Auxiliary prediction of interpretable reward components."""

import torch
import torch.nn.functional as F
from torch import nn


REWARD_COMPONENT_KEYS = ("event", "progress", "human_clearance", "smoothness", "height", "time")


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


class RewardComponentHead(nn.Module):
    """Predict six symlog components; their sum is not used as policy reward."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                 nn.SiLU(), nn.Linear(hidden_dim, len(REWARD_COMPONENT_KEYS)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature.float())

    def loss(self, feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape != (*feature.shape[:-1], len(REWARD_COMPONENT_KEYS)):
            raise ValueError(f"reward_components must end in {len(REWARD_COMPONENT_KEYS)}, got {target.shape}")
        return F.smooth_l1_loss(self(feature), symlog(target.float()))
