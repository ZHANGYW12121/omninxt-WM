"""Goal-coordinate conversion shared by replay and imagination."""

from __future__ import annotations

import numpy as np
import torch


GOAL_FEATURE_DIM = 8


def goal_features_torch(ego_state: torch.Tensor, goal_position: torch.Tensor) -> torch.Tensor:
    """Convert an episode-local 3-D goal to an Ego/body-relative 8-D feature.

    ``ego_state`` follows Ego14: local position at indices 0:3 and relative
    yaw as sin/cos at 12:14.  ``goal_position`` is fixed in the same
    episode-start local coordinate system.  Returned values are
    ``[dx_body, dy_body, dz, distance, unit_xyz, heading_error/pi]``.
    """
    if ego_state.shape[-1] < 14 or goal_position.shape[-1] != 3:
        raise ValueError("expected Ego14 and a 3-D goal position")
    delta = goal_position.to(ego_state) - ego_state[..., :3]
    sin_yaw, cos_yaw = ego_state[..., 12], ego_state[..., 13]
    dx = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
    dy = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
    body_delta = torch.stack((dx, dy, delta[..., 2]), dim=-1)
    distance = torch.linalg.vector_norm(body_delta, dim=-1, keepdim=True)
    unit = body_delta / distance.clamp_min(1.0e-6)
    heading = torch.atan2(dy, dx).unsqueeze(-1) / torch.pi
    return torch.cat((body_delta, distance, unit, heading), dim=-1)


def goal_features_numpy(ego_state: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
    """NumPy equivalent of :func:`goal_features_torch`."""
    ego = np.asarray(ego_state, dtype=np.float32)
    goal = np.asarray(goal_position, dtype=np.float32)
    delta = goal - ego[..., :3]
    sin_yaw, cos_yaw = ego[..., 12], ego[..., 13]
    dx = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
    dy = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
    body_delta = np.stack((dx, dy, delta[..., 2]), axis=-1)
    distance = np.linalg.norm(body_delta, axis=-1, keepdims=True)
    unit = body_delta / np.maximum(distance, 1.0e-6)
    heading = np.arctan2(dy, dx)[..., None] / np.pi
    return np.concatenate((body_delta, distance, unit, heading), axis=-1).astype(np.float32)
