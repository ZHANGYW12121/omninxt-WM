"""Single-owner horizontal-policy to applied-flight-action contract.

The learned v6.2 policy controls only horizontal navigation and yaw.  This
module deterministically inserts the normalized vertical command before the
action is consumed by either RSSM dynamics or the simulator.  Keeping this in
the model boundary prevents the previous-action latent from seeing a different
command than Isaac/PX4 actually executed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


POLICY_ACTION_KEYS = ("vx_forward", "vy_left", "yaw_rate")
APPLIED_ACTION_KEYS = ("vx_forward", "vy_left", "vz_up", "yaw_rate")


@dataclass(frozen=True)
class ActionAdapterConfig:
    target_altitude_agl_m: float = 1.0
    altitude_kp: float = 0.8
    vertical_velocity_kd: float = 0.15
    maximum_vertical_action: float = 0.20


class HorizontalActionAdapter(nn.Module):
    """Map ``[vx, vy, yaw]`` to the exact applied ``[vx, vy, vz, yaw]``.

    Ego14 indices are part of the dataset-v3 contract: vertical velocity is
    index 5 and altitude AGL is index 9.  The output remains normalized; the
    simulation controller performs only the existing unit conversion.
    """

    policy_action_dim = 3
    applied_action_dim = 4

    def __init__(self, config: ActionAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionAdapterConfig()
        if self.config.target_altitude_agl_m <= 0.0:
            raise ValueError("target_altitude_agl_m must be positive")
        if self.config.altitude_kp < 0.0:
            raise ValueError("altitude_kp must be non-negative")
        if self.config.vertical_velocity_kd < 0.0:
            raise ValueError("vertical_velocity_kd must be non-negative")
        if not 0.0 < self.config.maximum_vertical_action <= 1.0:
            raise ValueError("maximum_vertical_action must lie in (0, 1]")

    def vertical_action(self, ego_state: torch.Tensor) -> torch.Tensor:
        if ego_state.shape[-1] != 14:
            raise ValueError("ActionAdapter requires Ego14 state")
        altitude_error = (
            ego_state.new_tensor(self.config.target_altitude_agl_m)
            - ego_state[..., 9]
        )
        vertical_velocity = ego_state[..., 5]
        action = (
            self.config.altitude_kp * altitude_error
            - self.config.vertical_velocity_kd * vertical_velocity
        )
        return action.clamp(
            -self.config.maximum_vertical_action,
            self.config.maximum_vertical_action,
        )

    def forward(self, policy_action: torch.Tensor,
                ego_state: torch.Tensor) -> torch.Tensor:
        if policy_action.shape[:-1] != ego_state.shape[:-1]:
            raise ValueError("policy action and Ego14 batch shapes must match")
        if policy_action.shape[-1] != self.policy_action_dim:
            raise ValueError("policy action must be [vx, vy, yaw]")
        vertical = self.vertical_action(ego_state)[..., None]
        applied = torch.cat((
            policy_action[..., :2], vertical, policy_action[..., 2:3],
        ), dim=-1)
        # Policy samples are already tanh-bounded.  This assertion-like check
        # catches contract violations without silently changing log-probability.
        if torch.is_grad_enabled() and not torch.isfinite(applied).all():
            raise FloatingPointError("ActionAdapter produced a non-finite action")
        return applied

    @staticmethod
    def policy_from_applied(applied_action: torch.Tensor) -> torch.Tensor:
        if applied_action.shape[-1] != 4:
            raise ValueError("applied action must be [vx, vy, vz, yaw]")
        return torch.cat((
            applied_action[..., :2], applied_action[..., 3:4],
        ), dim=-1)
