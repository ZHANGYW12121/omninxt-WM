"""Action-conditioned short-horizon human-collision risk prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ActionRiskHead(nn.Module):
    """Predict collision logit and minimum surface clearance for one action.

    ``detach_parameters=True`` is used by the Actor objective: gradients still
    flow through the candidate action, but the risk estimator cannot lower the
    Actor penalty by changing its own weights.  The head itself is trained only
    by privileged replay labels.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = int(feature_dim) + int(action_dim)
        hidden_dim = int(hidden_dim)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim, action_dim and hidden_dim must be positive")
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.collision = nn.Linear(hidden_dim, 1)
        self.clearance = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _linear(layer: nn.Linear, value: torch.Tensor, detach: bool) -> torch.Tensor:
        weight = layer.weight.detach() if detach else layer.weight
        bias = layer.bias
        if bias is not None and detach:
            bias = bias.detach()
        return F.linear(value, weight, bias)

    def forward(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> dict[str, torch.Tensor]:
        if feature.shape[:-1] != action.shape[:-1]:
            raise ValueError("feature and action leading dimensions must match")
        value = torch.cat((feature, action), dim=-1)
        value = F.silu(self._linear(self.input, value, detach_parameters))
        value = F.silu(self._linear(self.hidden, value, detach_parameters))
        collision_logit = self._linear(
            self.collision, value, detach_parameters)
        # Surface clearance is non-negative for the predictor. Contact labels
        # can be slightly negative; their supervised target is clamped to zero.
        clearance_m = F.softplus(self._linear(
            self.clearance, value, detach_parameters))
        return {
            "collision_logit": collision_logit,
            "collision_probability": collision_logit.sigmoid(),
            "min_human_clearance_m": clearance_m,
        }
