"""Prediction of the task's interpretable reward components."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


REWARD_COMPONENT_KEYS = ("event", "progress", "human_clearance", "smoothness", "height", "time")


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of symlog for policy reward composition."""
    value = value.float().clamp(-20.0, 20.0)
    return torch.sign(value) * torch.expm1(torch.abs(value))


class RewardComponentHead(nn.Module):
    """Predict six symlog components with optional bounded Event support.

    The Isaac reward contract makes the one-off Event component finite.  A
    plain symlog regressor does not preserve that fact: a small extrapolation
    beyond ``symlog(-120)`` becomes an arbitrarily large negative reward after
    ``symexp``.  When Event bounds are supplied, the first output therefore
    uses a smooth soft clip whose image is the exact physical interval.  The
    transform is effectively the identity well inside that interval and only
    rolls off near its endpoints, so it does not rescale ordinary reward
    learning.  This is part of the learned reward model, not a detached
    policy-time clip, so imagined reward retains the standard Dreamer dynamics
    gradient.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        *,
        event_value_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                 nn.SiLU(), nn.Linear(hidden_dim, len(REWARD_COMPONENT_KEYS)))
        if event_value_bounds is None:
            self.event_value_bounds = None
            self._event_negative_symlog_limit = None
            self._event_positive_symlog_limit = None
        else:
            lower, upper = (float(value) for value in event_value_bounds)
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError("Event reward bounds must be finite")
            if not lower < 0.0 < upper:
                raise ValueError(
                    "Event reward bounds must straddle zero, got "
                    f"{event_value_bounds}")
            self.event_value_bounds = (lower, upper)
            self._event_negative_symlog_limit = math.log1p(-lower)
            self._event_positive_symlog_limit = math.log1p(upper)

    def _bounded_event_symlog(self, raw: torch.Tensor) -> torch.Tensor:
        if self.event_value_bounds is None:
            return raw
        lower = raw.new_tensor(-self._event_negative_symlog_limit)
        upper = raw.new_tensor(self._event_positive_symlog_limit)
        # lower + softplus(x-lower) - softplus(x-upper) is monotonic and has
        # limits [lower, upper]. beta=2 confines the roll-off to roughly half
        # a symlog unit around each known physical endpoint while preserving
        # a unit derivative around the ordinary target zero.
        return (
            lower
            + F.softplus(raw - lower, beta=2.0)
            - F.softplus(raw - upper, beta=2.0)
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.net(feature.float())
        event = self._bounded_event_symlog(raw[..., :1])
        return torch.cat((event, raw[..., 1:]), dim=-1)

    def loss(
        self, feature: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target.shape != (*feature.shape[:-1], len(REWARD_COMPONENT_KEYS)):
            raise ValueError(f"reward_components must end in {len(REWARD_COMPONENT_KEYS)}, got {target.shape}")
        raw = F.smooth_l1_loss(
            self(feature), symlog(target.float()), reduction="none")
        if mask is None:
            return raw.mean()
        weight = mask.to(raw)
        while weight.ndim < raw.ndim:
            weight = weight.unsqueeze(-1)
        return (raw * weight).sum() / weight.expand_as(raw).sum().clamp_min(1)
