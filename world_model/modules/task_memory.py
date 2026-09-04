"""Deployable Markov memory for recorder reward and task watchdog logic.

Ego14 is not by itself a complete decision state for the Isaac task.  Crash
eligibility depends on elapsed episode time, ``stuck_timeout`` depends on two
history timers and the best goal distance, and the smoothness reward depends
on the previous filtered horizontal acceleration.  The actuator filter also
depends on its previous smoothed policy target.  This module makes all of those
quantities explicit, derives them from replay, updates them in imagination,
and provides the identical stateful update used by deployment.
"""

from __future__ import annotations

import math

import numpy as np
import torch


TASK_MEMORY_DIM = 9
TASK_MEMORY_ORDER = (
    "elapsed_episode_s",
    "best_goal_distance_m",
    "stalled_for_s",
    "low_motion_for_s",
    "filtered_acceleration_episode_x_mps2",
    "filtered_acceleration_episode_y_mps2",
    "previous_smoothed_policy_forward",
    "previous_smoothed_policy_left",
    "previous_smoothed_policy_yaw",
)
TASK_MEMORY_SCALE = (120.0, 50.0, 30.0, 30.0, 8.0, 8.0, 1.0, 1.0, 1.0)
TASK_MEMORY_FEATURE_SCALE = 0.5


def _episode_acceleration_numpy(ego_state: np.ndarray) -> np.ndarray:
    ego = np.asarray(ego_state, np.float32)
    yaw = np.arctan2(ego[..., 12], ego[..., 13])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.stack((
        cosine * ego[..., 6] - sine * ego[..., 7],
        sine * ego[..., 6] + cosine * ego[..., 7],
    ), -1).astype(np.float32)


def derive_task_memory_numpy(
    simulation_time_s: np.ndarray,
    ego_state: np.ndarray,
    goal_position: np.ndarray,
    applied_action: np.ndarray,
    *,
    progress_epsilon_m: float = 0.25,
    stuck_max_horizontal_speed_mps: float = 0.15,
    acceleration_filter_alpha: float = 0.25,
) -> np.ndarray:
    """Reconstruct the causal task memory for every physical replay row."""
    times = np.asarray(simulation_time_s, np.float64).reshape(-1)
    ego = np.asarray(ego_state, np.float32)
    goal = np.asarray(goal_position, np.float32)
    action = np.asarray(applied_action, np.float32)
    if ego.shape != (times.size, 14):
        raise ValueError("task-memory Ego state must be [T,14]")
    if goal.shape == (3,):
        goal = np.broadcast_to(goal, (times.size, 3))
    if goal.shape != (times.size, 3):
        raise ValueError("task-memory goal must be [3] or [T,3]")
    if action.shape != (times.size, 4):
        raise ValueError("task-memory applied action must be [T,4]")
    if times.size and (
        not np.isfinite(times).all()
        or (times.size > 1 and np.any(np.diff(times) <= 0))
    ):
        raise ValueError("task-memory timestamps must be finite/strictly increasing")
    if not 0.0 <= float(acceleration_filter_alpha) <= 1.0:
        raise ValueError("acceleration filter alpha must lie in [0,1]")
    if float(progress_epsilon_m) <= 0.0:
        raise ValueError("progress epsilon must be positive")
    if float(stuck_max_horizontal_speed_mps) <= 0.0:
        raise ValueError("stuck speed threshold must be positive")
    if times.size == 0:
        return np.zeros((0, TASK_MEMORY_DIM), np.float32)

    elapsed = times - times[0]
    distance = np.linalg.norm(goal - ego[:, :3], axis=-1)
    acceleration = _episode_acceleration_numpy(ego)
    result = np.zeros((times.size, TASK_MEMORY_DIM), np.float32)
    best_distance = float(distance[0])
    last_progress_s = 0.0
    low_motion_since_s = 0.0
    filtered_acceleration = acceleration[0].astype(np.float64)
    for index in range(times.size):
        now = float(elapsed[index])
        if index:
            if float(distance[index]) <= (
                best_distance - float(progress_epsilon_m)
            ):
                best_distance = float(distance[index])
                last_progress_s = now
            if float(np.linalg.norm(ego[index, 3:5])) > float(
                stuck_max_horizontal_speed_mps
            ):
                low_motion_since_s = now
            filtered_acceleration = (
                float(acceleration_filter_alpha) * acceleration[index]
                + (1.0 - float(acceleration_filter_alpha))
                * filtered_acceleration
            )
        result[index] = (
            now,
            best_distance,
            now - last_progress_s,
            now - low_motion_since_s,
            float(filtered_acceleration[0]),
            float(filtered_acceleration[1]),
            float(action[index, 0]),
            float(action[index, 1]),
            float(action[index, 3]),
        )
    if not np.isfinite(result).all():
        raise ValueError("derived task memory contains NaN/Inf")
    return result


def task_memory_step_torch(
    memory: torch.Tensor,
    next_ego_state: torch.Tensor,
    goal_position: torch.Tensor,
    next_smoothed_policy_action: torch.Tensor,
    *,
    dt_s: float,
    progress_epsilon_m: float,
    stuck_max_horizontal_speed_mps: float,
    acceleration_filter_alpha: float,
) -> torch.Tensor:
    """Advance task memory by one fixed-rate imagined transition."""
    if memory.shape[-1] != TASK_MEMORY_DIM:
        raise ValueError("task memory must end in nine fields")
    if next_ego_state.shape[:-1] != memory.shape[:-1] \
            or next_ego_state.shape[-1] != 14:
        raise ValueError("next Ego state must match task-memory rows")
    if goal_position.shape != memory.shape[:-1] + (3,):
        raise ValueError("goal position must match task-memory rows")
    if next_smoothed_policy_action.shape != memory.shape[:-1] + (3,):
        raise ValueError(
            "smoothed policy action must match task-memory rows")
    if float(dt_s) <= 0.0:
        raise ValueError("task-memory dt must be positive")
    if not 0.0 <= float(acceleration_filter_alpha) <= 1.0:
        raise ValueError("acceleration filter alpha must lie in [0,1]")

    elapsed = memory[..., 0] + float(dt_s)
    distance = torch.linalg.vector_norm(
        goal_position.to(next_ego_state) - next_ego_state[..., :3], dim=-1)
    progressed = distance <= (
        memory[..., 1] - float(progress_epsilon_m))
    best = torch.where(progressed, distance, memory[..., 1])
    stalled = torch.where(
        progressed, torch.zeros_like(elapsed),
        memory[..., 2] + float(dt_s))
    moving = torch.linalg.vector_norm(
        next_ego_state[..., 3:5], dim=-1
    ) > float(stuck_max_horizontal_speed_mps)
    low_motion = torch.where(
        moving, torch.zeros_like(elapsed),
        memory[..., 3] + float(dt_s))

    yaw = torch.atan2(next_ego_state[..., 12], next_ego_state[..., 13])
    cosine, sine = yaw.cos(), yaw.sin()
    acceleration_episode = torch.stack((
        cosine * next_ego_state[..., 6] - sine * next_ego_state[..., 7],
        sine * next_ego_state[..., 6] + cosine * next_ego_state[..., 7],
    ), -1)
    filtered_acceleration = (
        float(acceleration_filter_alpha) * acceleration_episode
        + (1.0 - float(acceleration_filter_alpha)) * memory[..., 4:6]
    )
    return torch.cat((
        elapsed[..., None], best[..., None], stalled[..., None],
        low_motion[..., None], filtered_acceleration,
        next_smoothed_policy_action.float().clamp(-1.0, 1.0),
    ), -1)


def task_memory_feature(
    memory: torch.Tensor, feature_dim: int,
) -> torch.Tensor:
    """Tile fixed-scale physical memory into an existing decision token."""
    if memory.shape[-1] != TASK_MEMORY_DIM:
        raise ValueError("task memory must end in nine fields")
    scale = memory.new_tensor(TASK_MEMORY_SCALE)
    normalized = (memory.float() / scale).clamp(-5.0, 5.0)
    repeats = (int(feature_dim) + TASK_MEMORY_DIM - 1) // TASK_MEMORY_DIM
    return normalized.repeat_interleave(repeats, dim=-1)[
        ..., :int(feature_dim)
    ] * TASK_MEMORY_FEATURE_SCALE


class TaskMemoryTracker:
    """Stateful NumPy twin used by the policy server between reset and step."""

    def __init__(
        self, *, progress_epsilon_m: float = 0.25,
        stuck_max_horizontal_speed_mps: float = 0.15,
        acceleration_filter_alpha: float = 0.25,
    ) -> None:
        self.progress_epsilon_m = float(progress_epsilon_m)
        self.stuck_max_horizontal_speed_mps = float(
            stuck_max_horizontal_speed_mps)
        self.acceleration_filter_alpha = float(acceleration_filter_alpha)
        self.reset()

    def reset(self) -> None:
        self.origin_timestamp_s: float | None = None
        self.previous_timestamp_s: float | None = None
        self.best_goal_distance_m: float | None = None
        self.last_progress_s = 0.0
        self.low_motion_since_s = 0.0
        self.filtered_acceleration: np.ndarray | None = None

    def update(
        self, timestamp_s: float, ego_state: np.ndarray,
        goal_position: np.ndarray,
        previous_smoothed_policy_action: np.ndarray,
    ) -> np.ndarray:
        timestamp = float(timestamp_s)
        ego = np.asarray(ego_state, np.float32).reshape(14)
        goal = np.asarray(goal_position, np.float32).reshape(3)
        previous_action = np.asarray(
            previous_smoothed_policy_action, np.float32).reshape(3)
        if not math.isfinite(timestamp) or not np.isfinite(ego).all() \
                or not np.isfinite(goal).all() \
                or not np.isfinite(previous_action).all():
            raise ValueError("task-memory tracker inputs must be finite")
        if self.origin_timestamp_s is None \
                or self.previous_timestamp_s is None:
            self.reset()
            self.origin_timestamp_s = timestamp
            self.previous_timestamp_s = timestamp
            self.best_goal_distance_m = float(np.linalg.norm(goal - ego[:3]))
            self.filtered_acceleration = _episode_acceleration_numpy(
                ego[None])[0]
        else:
            if timestamp <= self.previous_timestamp_s:
                raise ValueError(
                    "task-memory timestamps must increase; reset the tracker "
                    "explicitly at an episode boundary")
            elapsed = timestamp - self.origin_timestamp_s
            distance = float(np.linalg.norm(goal - ego[:3]))
            assert self.best_goal_distance_m is not None
            if distance <= (
                self.best_goal_distance_m - self.progress_epsilon_m
            ):
                self.best_goal_distance_m = distance
                self.last_progress_s = elapsed
            if float(np.linalg.norm(ego[3:5])) > (
                self.stuck_max_horizontal_speed_mps
            ):
                self.low_motion_since_s = elapsed
            acceleration = _episode_acceleration_numpy(ego[None])[0]
            assert self.filtered_acceleration is not None
            self.filtered_acceleration = (
                self.acceleration_filter_alpha * acceleration
                + (1.0 - self.acceleration_filter_alpha)
                * self.filtered_acceleration
            )
            self.previous_timestamp_s = timestamp
        assert self.origin_timestamp_s is not None
        assert self.best_goal_distance_m is not None
        assert self.filtered_acceleration is not None
        elapsed = timestamp - self.origin_timestamp_s
        return np.asarray((
            elapsed,
            self.best_goal_distance_m,
            elapsed - self.last_progress_s,
            elapsed - self.low_motion_since_s,
            self.filtered_acceleration[0],
            self.filtered_acceleration[1],
            np.clip(previous_action[0], -1.0, 1.0),
            np.clip(previous_action[1], -1.0, 1.0),
            np.clip(previous_action[2], -1.0, 1.0),
        ), np.float32)
