#!/usr/bin/env python3
"""Serve a trained Factorized Dreamer policy to the Isaac controller.

The service intentionally runs in the training Conda environment.  Isaac Sim
ships its own PyTorch build and does not include Hydra/TensorDict, so importing
the training stack into Isaac's interpreter is unsafe.  The loopback protocol
contains only the versioned v3 observation fields and a normalized FLU action.
"""

from __future__ import annotations

import argparse
import json
import math
import socketserver
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.compact_skeleton_v3 import (
    _body_to_episode_rotation,
    _bounded_joint_velocity_episode,
    _ego_velocity_full_body,
    _risk_scores,
)
from datasets.factorized_schema import human_root_and_relative_joints_numpy
from factorized_agent import (
    FactorizedDreamerAgent, load_compatible_factorized_state,
)
from modules.goal_conditioning import GOAL_FEATURE_DIM, goal_features_numpy
from modules.skeleton_topology import (
    complete_coco12_topology_numpy,
    sanitize_coco12_geometry_numpy,
)
from modules.task_geometry import (
    task_geometry_proximity_numpy,
    task_physical_state_numpy,
)
from modules.task_memory import TaskMemoryTracker


PROTOCOL = "omninxt.factorized_policy.v1"
JOINT_COUNT = 12
FEATURE_DIM = 7
ACTION_DIM = 4
# Checkpoints written before the explicit Event safety-margin field learned
# physical-contact risk using the Isaac joint-marker contactOffset.
LEGACY_ISAAC_JOINT_CONTACT_OFFSET_M = 0.02


def checkpoint_actor_explicit_human_geometry_enabled(
    checkpoint: dict[str, object], architecture_version: str,
) -> bool:
    """Recover the Actor observation contract stored by training.

    New checkpoints persist the flag explicitly.  The objective-name fallback
    is only for already-written checkpoints: v6.9 originally disabled the
    path, while pure-Dreamer objectives whose contract names human geometry
    trained and audited with it enabled.
    """
    if "actor_explicit_human_geometry_enabled" in checkpoint:
        enabled = checkpoint["actor_explicit_human_geometry_enabled"]
        if not isinstance(enabled, bool):
            raise RuntimeError(
                "checkpoint actor_explicit_human_geometry_enabled must be bool")
        return enabled
    if architecture_version != "factorized_dreamer_v6.9":
        return True
    objective = str(checkpoint.get("training_objective_version", ""))
    return objective.startswith((
        "factorized_pure_dreamer_v11_",
        "factorized_pure_dreamer_v12_",
        "factorized_pure_dreamer_v13_",
    )) or objective.endswith("_human_geometry")


def checkpoint_actor_human_reflection_contract(
    checkpoint: dict[str, object],
) -> tuple[bool, int]:
    """Return the Actor's persisted Human reflection architecture contract."""
    enabled = checkpoint.get(
        "actor_human_reflection_equivariant_enabled", False)
    if not isinstance(enabled, bool):
        raise RuntimeError(
            "checkpoint actor_human_reflection_equivariant_enabled must be bool")
    slots = checkpoint.get("actor_human_physical_slots", 0)
    if isinstance(slots, bool) or not isinstance(slots, int):
        raise RuntimeError("checkpoint actor_human_physical_slots must be int")
    if slots < 0 or (enabled and slots <= 0):
        raise RuntimeError(
            "checkpoint Human reflection slots are inconsistent with enablement")
    return enabled, slots


def checkpoint_actor_authoritative_ego_token_enabled(
    checkpoint: dict[str, object],
) -> bool:
    """Recover whether Actor goal/Ego input bypasses Human-coupled RSSM."""
    field = "actor_authoritative_ego_token_enabled"
    if field not in checkpoint:
        # Historical checkpoints were trained with the latent Ego token.
        # Never silently change their deployed policy function.
        return False
    enabled = checkpoint[field]
    if not isinstance(enabled, bool):
        raise RuntimeError(f"checkpoint {field} must be bool")
    return enabled


def _checkpoint_bool(
    checkpoint: dict[str, object], field: str, *, default: bool = False,
) -> bool:
    value = checkpoint.get(field, default)
    if not isinstance(value, bool):
        raise RuntimeError(f"checkpoint {field} must be bool")
    return value


def checkpoint_actor_unified_policy_enabled(
    checkpoint: dict[str, object],
) -> bool:
    """Recover whether one full-state Actor mean was trained."""
    return _checkpoint_bool(
        checkpoint, "actor_unified_policy_enabled", default=False)


def checkpoint_actor_human_conditioned_residual_enabled(
    checkpoint: dict[str, object],
) -> bool:
    """Recover the exact avoidance-residual function used during training."""
    field = "actor_human_conditioned_residual_enabled"
    if field in checkpoint:
        enabled = checkpoint[field]
        if not isinstance(enabled, bool):
            raise RuntimeError(f"checkpoint {field} must be bool")
        return enabled
    # v15 was written before the semantic flag was persisted, but its frozen
    # training config is unambiguous. Earlier objectives used the legacy path.
    return str(checkpoint.get("training_objective_version", "")) == (
        "factorized_pure_dreamer_v15_closed_event_actuator_human_geometry")


def checkpoint_human_geometry_topk_physical_slots(
    checkpoint: dict[str, object],
) -> int:
    """Recover the physical layout of the Actor/Event Human geometry token."""
    field = "human_geometry_topk_physical_slots"
    if field in checkpoint:
        slots = checkpoint[field]
        if isinstance(slots, bool) or not isinstance(slots, int):
            raise RuntimeError(f"checkpoint {field} must be int")
        if slots < 0:
            raise RuntimeError(f"checkpoint {field} must be non-negative")
        return slots
    if str(checkpoint.get("training_objective_version", "")) == (
        "factorized_pure_dreamer_v15_closed_event_actuator_human_geometry"
    ):
        return 10
    return 0

BODY_BONES = (
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (8, 10),
    (7, 9), (9, 11),
)
# The independent Isaac benchmark uses a reduced GT rig whose collision
# capsules span Pelvis->Head, Head->Elbow and Pelvis->Knee.  The policy sees
# COCO12 and therefore has neither a head nor a pelvis joint.  Standard COCO
# edges alone leave the interior of those long GT capsules uncovered (seed
# 308 was contacted by Head->R_ElbowShareBone while every COCO edge was
# outside the shield).  These virtual endpoints conservatively approximate
# the benchmark rig from the information actually available online.  Each
# endpoint is the mean of the listed COCO12 joints.
SAFETY_VIRTUAL_BODY_CAPSULES = (
    ((0, 1), (6, 7)),  # shoulder centre -> hip centre (Head -> Pelvis)
    ((0, 1), (2,)),    # shoulder centre -> left elbow
    ((0, 1), (3,)),    # shoulder centre -> right elbow
    ((6, 7), (8,)),    # hip centre -> left knee
    ((6, 7), (9,)),    # hip centre -> right knee
)
# Conservative single-radius proxy for the online COCO12 body capsules:
# 0.17 m OmniNxt envelope + 0.18 m maximum torso capsule radius. This is the
# physical overlap boundary; target surface clearance is configured separately.
OBSERVED_DRONE_RADIUS_M = 0.17
OBSERVED_MAX_HUMAN_BODY_RADIUS_M = 0.18
OBSERVED_COMBINED_BODY_RADIUS_M = (
    OBSERVED_DRONE_RADIUS_M + OBSERVED_MAX_HUMAN_BODY_RADIUS_M)
ESCAPE_DIRECTION_MAX_HOLD_S = 1.50
ESCAPE_DIRECTION_RELEASE_CLEARANCE_M = 1.80
ESCAPE_DIRECTION_RELEASE_DWELL_S = 0.50
KINEMATIC_CLEARANCE_HORIZON_S = 1.20
KINEMATIC_MAX_RELATIVE_SPEED_MPS = 4.00
ESCAPE_CANDIDATE_COUNT = 24
ESCAPE_CANDIDATE_SPEED_MPS = 1.20
ESCAPE_DIRECTION_MAX_TURN_RATE_RAD_S = 0.5 * math.pi
# Keep the chosen side throughout the release hysteresis band.  v42 changed
# side at 1.98 m and entered a dense group before the former 1.50 m freeze
# threshold could take effect.
# Freeze only in the actual hard-stop band.  The old value equalled the
# 2.4 m release distance, so a direction selected on the first emergency frame
# could never adapt while the vehicle remained near the crowd and often drove
# all the way to a boundary.
ESCAPE_DIRECTION_FREEZE_CLEARANCE_M = 0.50
ESCAPE_MAX_HUMAN_SPEED_MPS = 2.00
ACTION_XY_LIMIT_MPS = 2.15
MULTI_HUMAN_BARRIER_ACTIVATION_CLEARANCE_M = 1.80
# Match the learned safety decision boundary.  Combined geometry already
# includes the person body and rotor envelope; another full metre of surface
# clearance made dense corridors infeasible and prevented useful online data.
MULTI_HUMAN_BARRIER_TARGET_CLEARANCE_M = 0.70
MULTI_HUMAN_BARRIER_GAIN_PER_S = 1.00
MULTI_HUMAN_BARRIER_MAX_OBSTACLE_SPEED_MPS = 3.00
MULTI_HUMAN_FALLBACK_DIRECTION_COUNT = 48
MULTI_HUMAN_FALLBACK_DEVIATION_WEIGHT = 0.20
MULTI_HUMAN_FALLBACK_PROGRESS_WEIGHT = 0.05
# The learned Actor occasionally orbits just outside the 1 m terminal sphere:
# it keeps a useful forward speed but also keeps yawing, so body-forward no
# longer points at the final metre.  Use an observable goal servo only in the
# small terminal region and only while both current and swept human geometry
# are clear.  The resulting velocity still passes through the joint
# human/static projection below, so this helper can never bypass the shield.
GOAL_CAPTURE_ACTIVATION_DISTANCE_M = 4.0
GOAL_CAPTURE_MIN_HUMAN_CLEARANCE_M = 2.5
GOAL_CAPTURE_GAIN_PER_S = 0.75
GOAL_CAPTURE_MAX_SPEED_MPS = 0.85
# Aim slightly inside the benchmark/recording 1 m terminal sphere and slow as
# its edge approaches. This prevents an 0.85 m/s terminal command from
# stepping across the sphere between 10 Hz benchmark samples.
GOAL_CAPTURE_TARGET_RADIUS_M = 0.85
# ``boundary_flight_bounds_xy`` is already inset from the walk polygon by the
# configured vehicle margin (1.5 m X / 0.8 m Y).  The normal 2 m restoration
# band is intentionally conservative at cruise speed, but can make a valid
# near-wall goal sphere unreachable.  Retain another 0.75 m inside the inset
# bound while the slow, human-clear terminal servo is active.
GOAL_CAPTURE_BOUNDARY_ACTIVATION_M = 0.75


def _goal_capture_velocity(
    ego_state: np.ndarray,
    goal_position: np.ndarray,
    human_clearance_m: float,
    *,
    activation_distance_m: float = GOAL_CAPTURE_ACTIVATION_DISTANCE_M,
    minimum_human_clearance_m: float = GOAL_CAPTURE_MIN_HUMAN_CLEARANCE_M,
    gain_per_s: float = GOAL_CAPTURE_GAIN_PER_S,
    maximum_speed_mps: float = GOAL_CAPTURE_MAX_SPEED_MPS,
    target_radius_m: float = GOAL_CAPTURE_TARGET_RADIUS_M,
) -> tuple[np.ndarray, bool, float]:
    """Return a body-frame terminal velocity without bypassing safety.

    ``ego_state`` and ``goal_position`` use the same episode-local frame.  A
    caller must still run the returned velocity through the human/static
    projection; this function deliberately contains no collision fallback.
    """
    ego = np.asarray(ego_state, np.float64).reshape(14)
    goal = np.asarray(goal_position, np.float64).reshape(3)
    if not (np.isfinite(ego).all() and np.isfinite(goal).all()):
        raise ValueError("goal capture state contains NaN/Inf")
    if min(
        activation_distance_m,
        minimum_human_clearance_m,
        gain_per_s,
        maximum_speed_mps,
        target_radius_m,
    ) <= 0.0:
        raise ValueError("goal capture parameters must be positive")
    delta_episode = goal - ego[:3]
    distance = float(np.linalg.norm(delta_episode))
    horizontal_episode = delta_episode[:2]
    horizontal_distance = float(np.linalg.norm(horizontal_episode))
    active = (
        distance <= float(activation_distance_m)
        and float(human_clearance_m) >= float(minimum_human_clearance_m)
        and horizontal_distance > 1.0e-6
    )
    if not active:
        return np.zeros(2, np.float32), False, distance
    rotation = _body_to_episode_rotation(ego)
    direction_body = rotation.T @ np.asarray(
        (horizontal_episode[0], horizontal_episode[1], 0.0), np.float64)
    direction_xy = direction_body[:2]
    direction_norm = float(np.linalg.norm(direction_xy))
    if direction_norm <= 1.0e-6:
        return np.zeros(2, np.float32), False, distance
    distance_to_capture = max(distance - float(target_radius_m), 0.0)
    speed = min(
        float(maximum_speed_mps),
        float(gain_per_s) * distance_to_capture,
    )
    if speed <= 1.0e-6:
        return np.zeros(2, np.float32), False, distance
    return (
        np.asarray(direction_xy / direction_norm * speed, np.float32),
        True,
        distance,
    )


def _boundary_barrier_halfplanes(
    position_world_xy: np.ndarray,
    yaw_rad: float,
    flight_bounds_xy: np.ndarray,
    *,
    activation_m: float,
    return_gain_per_s: float,
    max_return_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build body-frame half-planes that restore the static flight box."""
    position = np.asarray(position_world_xy, np.float64).reshape(2)
    bounds = np.asarray(flight_bounds_xy, np.float64).reshape(4)
    if not np.isfinite(position).all() or not np.isfinite(bounds).all():
        raise ValueError("boundary state contains NaN/Inf")
    if min(activation_m, return_gain_per_s, max_return_speed_mps) <= 0.0:
        raise ValueError("boundary barrier parameters must be positive")
    x_min, x_max, y_min, y_max = bounds.tolist()
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("flight bounds are invalid")
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    world_from_body = np.asarray(
        ((cosine, -sine), (sine, cosine)), np.float64)
    normals_world: list[np.ndarray] = []
    limits: list[float] = []

    def add(normal_world: tuple[float, float], penetration: float) -> None:
        restore = min(
            float(max_return_speed_mps),
            float(return_gain_per_s) * max(float(penetration), 0.0),
        )
        normals_world.append(np.asarray(normal_world, np.float64))
        limits.append(-restore)

    x, y = position.tolist()
    if x <= x_min + activation_m:
        # v_world_x >= restore  ->  -e_x dot v_world <= -restore
        add((-1.0, 0.0), x_min + activation_m - x)
    if x >= x_max - activation_m:
        add((1.0, 0.0), x - (x_max - activation_m))
    if y <= y_min + activation_m:
        add((0.0, -1.0), y_min + activation_m - y)
    if y >= y_max - activation_m:
        add((0.0, 1.0), y - (y_max - activation_m))
    if not normals_world:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    normals_body = np.asarray(normals_world) @ world_from_body
    return normals_body.astype(np.float32), np.asarray(limits, np.float32)


def _all_body_capsule_obstacles(
    xyz: np.ndarray,
    valid: np.ndarray,
    velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the closest point on every valid articulated body capsule.

    The benchmark can report a propeller contact with a swinging elbow even
    while the torso is receding. One closest point per person therefore is
    insufficient for the final shield; retain each arm/leg/torso capsule and
    its local animation velocity for short-horizon swept prediction.
    """
    xyz = np.asarray(xyz, np.float32)
    valid = np.asarray(valid, np.bool_)
    if velocity is None:
        velocity = np.zeros_like(xyz)
    velocity = np.asarray(velocity, np.float32)
    if velocity.shape != xyz.shape:
        raise ValueError("human velocity must match human XYZ shape")
    points: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    clearances: list[float] = []
    def endpoint(
        person_xyz: np.ndarray,
        person_valid: np.ndarray,
        person_velocity: np.ndarray,
        indices: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        index = np.asarray(indices, np.int64)
        if not bool(np.all(person_valid[index])):
            return None
        return (
            np.asarray(person_xyz[index], np.float64).mean(axis=0),
            np.asarray(person_velocity[index], np.float64).mean(axis=0),
        )

    def append_capsule(
        start_position: np.ndarray,
        end_position: np.ndarray,
        start_velocity: np.ndarray,
        end_velocity: np.ndarray,
    ) -> None:
        delta = end_position - start_position
        denominator = float(np.dot(delta, delta))
        fraction = 0.0 if denominator <= 1.0e-12 else float(np.clip(
            -np.dot(start_position, delta) / denominator, 0.0, 1.0))
        point = start_position + fraction * delta
        point_velocity = (
            (1.0 - fraction) * start_velocity
            + fraction * end_velocity
        )
        points.append(np.asarray(point, np.float64).copy())
        velocities.append(np.asarray(point_velocity, np.float64)[:2].copy())
        clearances.append(
            float(np.linalg.norm(point))
            - OBSERVED_COMBINED_BODY_RADIUS_M)

    for person_xyz, person_valid, person_velocity in zip(
        xyz, valid, velocity
    ):
        # Extrapolating an elbow/knee animation velocity as constant whole-body
        # translation produced impossible two-second crossings and made the
        # shield rewrite one third of online actions.  All current articulated
        # capsule positions remain in the hard geometry; only their future
        # translation uses the robust torso motion.  If the torso is too
        # incomplete, zero is safer and less misleading than treating one
        # swinging wrist as pedestrian velocity.
        torso_indices = np.asarray((0, 1, 6, 7), np.int64)
        torso_valid = person_valid[torso_indices]
        if np.count_nonzero(torso_valid) >= 2:
            person_translation_velocity = np.median(
                person_velocity[torso_indices[torso_valid]], axis=0)
        else:
            person_translation_velocity = np.zeros(3, np.float32)
        for start, end in BODY_BONES:
            if not (person_valid[start] and person_valid[end]):
                continue
            append_capsule(
                np.asarray(person_xyz[start], np.float64),
                np.asarray(person_xyz[end], np.float64),
                np.asarray(person_translation_velocity, np.float64),
                np.asarray(person_translation_velocity, np.float64),
            )
        for start_indices, end_indices in SAFETY_VIRTUAL_BODY_CAPSULES:
            start = endpoint(
                person_xyz, person_valid, person_velocity, start_indices)
            end = endpoint(
                person_xyz, person_valid, person_velocity, end_indices)
            if start is None or end is None:
                continue
            append_capsule(
                start[0], end[0],
                person_translation_velocity,
                person_translation_velocity,
            )
    if not points:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32),
        )
    return (
        np.asarray(points, np.float32),
        np.asarray(velocities, np.float32),
        np.asarray(clearances, np.float32),
    )


def _closest_body_capsule_obstacles(
    xyz: np.ndarray,
    valid: np.ndarray,
    velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one closest body-capsule point and velocity per person."""
    xyz = np.asarray(xyz, np.float32)
    valid = np.asarray(valid, np.bool_)
    if velocity is None:
        velocity = np.zeros_like(xyz)
    velocity = np.asarray(velocity, np.float32)
    if velocity.shape != xyz.shape:
        raise ValueError("human velocity must match human XYZ shape")
    points: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    clearances: list[float] = []
    for person_xyz, person_valid, person_velocity in zip(
        xyz, valid, velocity
    ):
        closest_point = None
        closest_velocity = None
        minimum = float("inf")
        for start, end in BODY_BONES:
            if not (person_valid[start] and person_valid[end]):
                continue
            a = person_xyz[start].astype(np.float64, copy=False)
            b = person_xyz[end].astype(np.float64, copy=False)
            delta = b - a
            denominator = float(np.dot(delta, delta))
            fraction = 0.0 if denominator <= 1.0e-12 else float(np.clip(
                -np.dot(a, delta) / denominator, 0.0, 1.0))
            point = a + fraction * delta
            clearance = (
                float(np.linalg.norm(point))
                - OBSERVED_COMBINED_BODY_RADIUS_M
            )
            if clearance < minimum:
                minimum = clearance
                closest_point = point
                closest_velocity = (
                    (1.0 - fraction) * person_velocity[start]
                    + fraction * person_velocity[end]
                )
        if closest_point is not None:
            # Capsule endpoints move with gait/arm animation.  Using their
            # instantaneous endpoint velocity made a stationary torso look
            # as if it crossed at several m/s. Estimate whole-person motion
            # from the robust median of shoulders/hips, falling back to every
            # valid joint only when the torso is incomplete.
            torso_indices = np.asarray((0, 1, 6, 7), np.int64)
            torso_valid = person_valid[torso_indices]
            if np.count_nonzero(torso_valid) >= 2:
                robust_velocity = np.median(
                    person_velocity[torso_indices[torso_valid]], axis=0)
            elif np.any(person_valid):
                robust_velocity = np.median(
                    person_velocity[person_valid], axis=0)
            else:
                robust_velocity = closest_velocity
            points.append(np.asarray(closest_point, np.float64).copy())
            velocities.append(
                np.asarray(robust_velocity, np.float64)[:2].copy())
            clearances.append(minimum)
    if not points:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32),
        )
    return (
        np.asarray(points, np.float32),
        np.asarray(velocities, np.float32),
        np.asarray(clearances, np.float32),
    )


def _project_multi_human_velocity(
    desired_velocity_xy: np.ndarray,
    obstacle_points: np.ndarray,
    obstacle_velocities: np.ndarray,
    obstacle_clearances: np.ndarray,
    *,
    activation_clearance_m: float = (
        MULTI_HUMAN_BARRIER_ACTIVATION_CLEARANCE_M),
    target_clearance_m: float = MULTI_HUMAN_BARRIER_TARGET_CLEARANCE_M,
    gain_per_s: float = MULTI_HUMAN_BARRIER_GAIN_PER_S,
    max_speed_mps: float = ACTION_XY_LIMIT_MPS,
    preferred_direction_xy: np.ndarray | None = None,
    prediction_horizon_s: float = KINEMATIC_CLEARANCE_HORIZON_S,
    static_constraint_normals: np.ndarray | None = None,
    static_constraint_bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Project a desired velocity into all active human barrier half-planes.

    For a human at relative position ``p`` with velocity ``u``, the barrier
    condition is ``n dot v <= n dot u + alpha * (clearance - target)``.
    Unlike a nearest-person tangent, the two-dimensional projection satisfies
    every simultaneously visible close person whenever the bounded feasible
    set is non-empty.
    """
    desired = np.asarray(desired_velocity_xy, np.float64).reshape(2)
    points = np.asarray(obstacle_points, np.float64).reshape(-1, 3)
    velocities = np.asarray(obstacle_velocities, np.float64).reshape(-1, 2)
    clearances = np.asarray(obstacle_clearances, np.float64).reshape(-1)
    if not (
        points.shape[0] == velocities.shape[0] == clearances.shape[0]
    ):
        raise ValueError("barrier obstacle arrays must have equal length")
    if min(
        activation_clearance_m,
        target_clearance_m,
        gain_per_s,
        max_speed_mps,
        prediction_horizon_s,
    ) <= 0.0:
        raise ValueError("barrier parameters must be positive")
    if preferred_direction_xy is None:
        preferred = np.zeros(2, np.float64)
    else:
        preferred = np.asarray(
            preferred_direction_xy, np.float64).reshape(2).copy()
        preferred_norm = float(np.linalg.norm(preferred))
        if preferred_norm > 1.0e-6:
            preferred /= preferred_norm
        else:
            preferred.fill(0.0)
    has_preferred_direction = bool(np.linalg.norm(preferred) > 0.5)
    desired_norm = float(np.linalg.norm(desired))
    if desired_norm > max_speed_mps:
        desired *= max_speed_mps / desired_norm

    normals: list[np.ndarray] = []
    bounds: list[float] = []
    for point, human_velocity, clearance in zip(
        points, velocities, clearances
    ):
        horizontal_norm = float(np.linalg.norm(point[:2]))
        if (
            clearance > activation_clearance_m
            or horizontal_norm <= 1.0e-5
        ):
            continue
        normal = point[:2] / horizontal_norm
        speed = float(np.linalg.norm(human_velocity))
        if speed > MULTI_HUMAN_BARRIER_MAX_OBSTACLE_SPEED_MPS:
            human_velocity = (
                human_velocity
                * MULTI_HUMAN_BARRIER_MAX_OBSTACLE_SPEED_MPS
                / speed)
        bound = (
            float(np.dot(normal, human_velocity))
            + gain_per_s * (float(clearance) - target_clearance_m)
        )
        normals.append(normal)
        bounds.append(bound)

    human_constraint_count = len(normals)
    if static_constraint_normals is None:
        static_matrix = np.zeros((0, 2), np.float64)
        static_limit = np.zeros((0,), np.float64)
    else:
        static_matrix = np.asarray(
            static_constraint_normals, np.float64).reshape(-1, 2)
        static_limit = np.asarray(
            static_constraint_bounds, np.float64).reshape(-1)
        if len(static_matrix) != len(static_limit):
            raise ValueError("static barrier arrays must have equal length")
        if not (np.isfinite(static_matrix).all()
                and np.isfinite(static_limit).all()):
            raise ValueError("static barrier arrays contain NaN/Inf")
        normals.extend(static_matrix)
        bounds.extend(static_limit)
    static_constraint_count = len(static_matrix)

    if not normals and not len(points):
        return desired.astype(np.float32), {
            "active_constraints": 0,
            "human_active_constraints": 0,
            "static_active_constraints": 0,
            "projected": False,
            "feasible": True,
            "predictive_override": False,
            "predicted_min_clearance_before_m": float("inf"),
            "predicted_min_clearance_after_m": float("inf"),
            "minimum_slack_before_mps": float("inf"),
            "minimum_slack_after_mps": float("inf"),
            "goal_progress_before_mps": float(
                np.dot(desired, preferred)),
            "goal_progress_after_mps": float(
                np.dot(desired, preferred)),
            "progress_preserving_selected": False,
        }
    matrix = np.asarray(normals, np.float64).reshape(-1, 2)
    limit = np.asarray(bounds, np.float64)
    tolerance = 1.0e-7

    candidates = [desired.copy(), np.zeros(2, np.float64)]
    for normal, bound in zip(matrix, limit):
        candidates.append(desired - (np.dot(normal, desired) - bound) * normal)
        if abs(bound) <= max_speed_mps:
            tangent = np.asarray((-normal[1], normal[0]), np.float64)
            offset = math.sqrt(max(max_speed_mps**2 - bound**2, 0.0))
            candidates.extend((bound * normal + offset * tangent,
                               bound * normal - offset * tangent))
    # In two dimensions every vertex of the feasible polygon is the
    # intersection of two active half-planes.  Dense articulated crowds can
    # contribute more than one hundred capsule constraints.  Calling
    # np.linalg.det/solve separately for every pair made this exact step take
    # about 100 ms by itself.  The closed-form 2-D solve below evaluates the
    # same lexicographically ordered pairs in one vectorized operation.
    if len(matrix) >= 2:
        first, second = np.triu_indices(len(matrix), k=1)
        first_normal = matrix[first]
        second_normal = matrix[second]
        determinant = (
            first_normal[:, 0] * second_normal[:, 1]
            - first_normal[:, 1] * second_normal[:, 0]
        )
        nonparallel = np.abs(determinant) > 1.0e-8
        if np.any(nonparallel):
            first_normal = first_normal[nonparallel]
            second_normal = second_normal[nonparallel]
            first_bound = limit[first[nonparallel]]
            second_bound = limit[second[nonparallel]]
            determinant = determinant[nonparallel]
            pair_intersections = np.column_stack((
                (
                    first_bound * second_normal[:, 1]
                    - first_normal[:, 1] * second_bound
                ) / determinant,
                (
                    first_normal[:, 0] * second_bound
                    - first_bound * second_normal[:, 0]
                ) / determinant,
            ))
            candidates.extend(pair_intersections)
    angles = np.linspace(-math.pi, math.pi, 144, endpoint=False)
    candidates.extend(
        max_speed_mps * np.asarray((math.cos(angle), math.sin(angle)))
        for angle in angles
    )

    candidate_array = np.asarray(candidates, np.float64).reshape(-1, 2)
    finite_mask = (
        np.isfinite(candidate_array).all(axis=1)
        & (
            np.linalg.norm(candidate_array, axis=1)
            <= max_speed_mps + tolerance
        )
    )
    finite = candidate_array[finite_mask]
    if len(matrix):
        feasible_mask = np.all(
            matrix @ finite.T <= limit[:, None] + tolerance,
            axis=0,
        )
    else:
        feasible_mask = np.ones((len(finite),), np.bool_)
    feasible = finite[feasible_mask]
    exact = bool(len(feasible))
    progress_preserving_selected = False
    if exact:
        selection_pool = feasible
        if has_preferred_direction:
            non_backtracking = (feasible @ preferred) >= -tolerance
            if np.any(non_backtracking):
                selection_pool = feasible[non_backtracking]
                progress_preserving_selected = True
        selected = selection_pool[int(np.argmin(
            np.square(selection_pool - desired[None]).sum(axis=1)
        ))]
    else:
        selected = desired.copy()

    # A conventional CBF only activates after the *current* clearance enters
    # its band. That is too late for a pedestrian crossing at 1--2 m/s: seed
    # 308 had a predicted CPA inside the danger tube while the current range
    # was still over four metres. Evaluate the CBF result over a finite swept
    # horizon and invoke the velocity lattice before contact whenever needed.
    # This also remains the fallback when simultaneous half-planes are empty.
    predicted_before = float("inf")
    predicted_after = float("inf")
    predictive_override = False
    if len(points):
        swept_velocity = velocities.copy()
        swept_speed = np.linalg.norm(
            swept_velocity, axis=-1, keepdims=True)
        swept_velocity *= np.minimum(
            1.0,
            MULTI_HUMAN_BARRIER_MAX_OBSTACLE_SPEED_MPS
            / np.maximum(swept_speed, 1.0e-6),
        )
        base_times = np.asarray(
            (0.0, 0.10, 0.20, 0.40, 0.70, 1.0, 1.2),
            np.float64,
        )
        times = base_times[
            base_times <= float(prediction_horizon_s) + 1.0e-9]
        if not np.isclose(times[-1], float(prediction_horizon_s)):
            times = np.append(times, float(prediction_horizon_s))

        def swept_clearance(candidates: np.ndarray) -> np.ndarray:
            candidate_array = np.asarray(
                candidates, np.float64).reshape(-1, 2)
            relative_velocity = (
                swept_velocity[None, :, None, :]
                - candidate_array[:, None, None, :]
            )
            future_xy = (
                points[None, :, None, :2]
                + relative_velocity * times[None, None, :, None]
            )
            future_z = points[None, :, None, 2:3]
            return np.sqrt(
                np.square(future_xy).sum(axis=-1)
                + np.square(future_z[..., 0])
            ) - OBSERVED_COMBINED_BODY_RADIUS_M

        selected_swept = swept_clearance(selected[None])
        predicted_before = float(selected_swept.min())
        predicted_hazard = predicted_before < target_clearance_m
        if not exact or predicted_hazard:
            # With articulated limbs or people closing from both sides the
            # CBF intersection can be empty. A bounded velocity lattice also
            # handles the feasible-but-future-unsafe crossing case.
            angles = np.linspace(
                -math.pi, math.pi,
                MULTI_HUMAN_FALLBACK_DIRECTION_COUNT,
                endpoint=False,
            )
            fallback_speed_limit = min(
                max_speed_mps,
                max(ESCAPE_CANDIDATE_SPEED_MPS, desired_norm),
            )
            speeds = np.unique(np.clip(
                np.asarray((0.0, 0.4, 0.8, fallback_speed_limit)),
                0.0,
                fallback_speed_limit,
            ))
            directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
            lattice = np.concatenate((
                np.zeros((1, 2), np.float64),
                (speeds[1:, None, None] * directions[None]).reshape(-1, 2),
                desired[None],
                selected[None],
            ), axis=0)
            swept = swept_clearance(lattice)
            worst = swept.min(axis=(1, 2))
            unsafe = np.square(
                np.maximum(target_clearance_m - swept, 0.0)
            ).mean(axis=(1, 2))
            deviation = np.square(lattice - desired[None]).sum(axis=-1)
            lattice_speed = np.linalg.norm(lattice, axis=-1)
            goal_progress = (
                (lattice @ preferred) / np.maximum(lattice_speed, 1.0e-6)
                if has_preferred_direction
                else np.zeros(len(lattice), np.float64)
            )
            # Once the target tube is clear, extra clearance has no safety
            # value and must not pull the UAV into a different crowd at full
            # speed. Below target, worst clearance remains the primary term;
            # the stronger deviation term suppresses direction chatter for
            # centimetre-scale score differences in an infeasible corridor.
            fallback_score = (
                np.minimum(worst, target_clearance_m)
                - 0.20 * unsafe
                - MULTI_HUMAN_FALLBACK_DEVIATION_WEIGHT * deviation
                + MULTI_HUMAN_FALLBACK_PROGRESS_WEIGHT * goal_progress
            )

            admissible = np.ones(len(lattice), np.bool_)
            if exact and len(matrix):
                joint_feasible = np.all(
                    matrix @ lattice.T <= limit[:, None] + tolerance,
                    axis=0,
                )
                # The exact candidate itself is in the lattice, so a
                # feasible CBF solution must never be replaced by one that
                # violates an immediate human or wall constraint.
                if np.any(joint_feasible):
                    admissible &= joint_feasible
            elif static_constraint_count:
                static_feasible = np.all(
                    static_matrix @ lattice.T
                    <= static_limit[:, None] + tolerance,
                    axis=0,
                )
                # Racks cannot move out of the way. If the lattice contains
                # any static-safe action, never trade a wall violation for a
                # small improvement in predicted human clearance.
                if np.any(static_feasible):
                    admissible &= static_feasible
            # When at least one jointly feasible command stays outside the
            # safety tube without losing already completed goal progress,
            # select only among that set. Safety is still lexicographically
            # first: this preference is never applied if every non-retreating
            # command is unsafe.
            if has_preferred_direction:
                safe_progress = (
                    admissible
                    & (worst >= target_clearance_m - tolerance)
                    & (goal_progress >= -tolerance)
                )
                if np.any(safe_progress):
                    admissible = safe_progress
                    progress_preserving_selected = True
            fallback_score = np.where(
                admissible, fallback_score, -np.inf)
            best_index = int(np.argmax(fallback_score))
            replacement = lattice[best_index]
            predictive_override = bool(
                predicted_hazard
                and np.linalg.norm(replacement - selected) > 1.0e-5)
            selected = replacement
            predicted_after = float(worst[best_index])
        else:
            predicted_after = predicted_before
    elif not exact:
        # Only mutually incompatible static constraints remain. Keep the
        # least-violating bounded command from the finite candidate set.
        violation = np.maximum(
            matrix @ finite.T - limit[:, None], 0.0
        ).sum(axis=0)
        selected = finite[int(np.argmin(violation))]
    slack_before = (
        float(np.min(limit - matrix @ desired))
        if len(matrix) else float("inf"))
    slack_after = (
        float(np.min(limit - matrix @ selected))
        if len(matrix) else float("inf"))
    return np.asarray(selected, np.float32), {
        "active_constraints": int(len(matrix)),
        "human_active_constraints": int(human_constraint_count),
        "static_active_constraints": int(static_constraint_count),
        "projected": bool(np.linalg.norm(selected - desired) > 1.0e-5),
        "feasible": exact,
        "predictive_override": predictive_override,
        "predicted_min_clearance_before_m": predicted_before,
        "predicted_min_clearance_after_m": predicted_after,
        "minimum_slack_before_mps": slack_before,
        "minimum_slack_after_mps": slack_after,
        "goal_progress_before_mps": float(np.dot(desired, preferred)),
        "goal_progress_after_mps": float(np.dot(selected, preferred)),
        "progress_preserving_selected": progress_preserving_selected,
    }


def _select_escape_direction(
    obstacle_points: list[np.ndarray],
    obstacle_velocities: list[np.ndarray],
    *,
    goal_direction_body_xy: np.ndarray | None = None,
    horizon_s: float = KINEMATIC_CLEARANCE_HORIZON_S,
) -> np.ndarray:
    """Choose a causal velocity-obstacle escape direction.

    A weighted repulsion vector is not enough in a moving crowd: two people
    can cancel each other, and a direction that was open on the first frame
    can be occupied a moment later.  Score a small deterministic circle of
    candidate velocities against every visible body capsule over the next two
    seconds.  Worst-case clearance dominates; navigation progress is only a
    tie breaker between similarly safe directions.
    """
    if not obstacle_points:
        return np.zeros(2, np.float32)
    if goal_direction_body_xy is None:
        goal = np.asarray((1.0, 0.0), np.float64)
    else:
        goal = np.asarray(goal_direction_body_xy, np.float64).reshape(2)
    goal_norm = float(np.linalg.norm(goal))
    if goal_norm > 1.0e-6:
        goal /= goal_norm
    else:
        goal[:] = (1.0, 0.0)

    angles = np.linspace(
        -math.pi, math.pi, ESCAPE_CANDIDATE_COUNT,
        endpoint=False, dtype=np.float64)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    # Denser near-term samples make a candidate that first cuts through a
    # person lose even if it would be clear at the end of the horizon.
    times = np.asarray((0.20, 0.40, 0.70, 1.00, 1.50, 2.00), np.float64)
    times = times[times <= max(float(horizon_s), 0.20) + 1.0e-9]
    if times.size == 0:
        times = np.asarray((max(float(horizon_s), 0.0),), np.float64)

    points = np.asarray(obstacle_points, np.float64)
    velocities = np.asarray(obstacle_velocities, np.float64)
    velocity_norm = np.linalg.norm(velocities, axis=-1, keepdims=True)
    velocities *= np.minimum(
        1.0,
        ESCAPE_MAX_HUMAN_SPEED_MPS / np.maximum(velocity_norm, 1.0e-6),
    )
    candidate_velocity = directions * ESCAPE_CANDIDATE_SPEED_MPS
    relative_velocity = (
        velocities[None, :, None, :]
        - candidate_velocity[:, None, None, :]
    )
    future_xy = (
        points[None, :, None, :2]
        + relative_velocity * times[None, None, :, None]
    )
    future_z = points[None, :, None, 2:3]
    distances = np.sqrt(
        np.square(future_xy).sum(axis=-1)
        + np.square(future_z[..., 0])
    ) - OBSERVED_COMBINED_BODY_RADIUS_M
    worst_clearance = distances.min(axis=(1, 2))
    # Penalize the whole unsafe tube, not only its single worst sample.  This
    # distinguishes a brief grazing prediction from driving along a crowd.
    unsafe_tube = np.square(np.maximum(1.0 - distances, 0.0)).mean(
        axis=(1, 2))
    progress = directions @ goal

    nearest_index = int(np.argmin(np.linalg.norm(points, axis=-1)))
    repulsion = -points[nearest_index, :2]
    repulsion_norm = float(np.linalg.norm(repulsion))
    if repulsion_norm > 1.0e-6:
        repulsion /= repulsion_norm
        separation = directions @ repulsion
    else:
        separation = np.zeros(len(directions), np.float64)
    score = (
        worst_clearance
        - 0.35 * unsafe_tube
        + 0.18 * progress
        + 0.04 * separation
    )
    # Emergency avoidance may pause goal progress, but it must not repeatedly
    # carry the vehicle back through crowd layers that it already passed.
    # The half-plane always contains two pure side-step candidates, so this
    # constraint does not force forward motion through a person.
    score = np.where(progress >= -1.0e-6, score, -np.inf)
    return directions[int(np.argmax(score))].astype(np.float32)


def _observed_human_geometry(
    xyz: np.ndarray,
    valid: np.ndarray,
    velocity: np.ndarray | None = None,
    ego_velocity_body: np.ndarray | None = None,
    goal_direction_body_xy: np.ndarray | None = None,
    horizon_s: float = KINEMATIC_CLEARANCE_HORIZON_S,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return current/kinematic clearance and multi-person escape direction.

    Clearance still comes from the closest valid body capsule.  The direction,
    however, uses the closest capsule on *each* visible person.  This avoids a
    discontinuous left/right command when two pedestrians exchange which one
    is nearest by a few centimetres in a dense group.  A constant-velocity
    closest-point-of-approach estimate anticipates a pedestrian crossing the
    UAV path; it uses only causal skeleton motion and Ego14 velocity.
    """
    xyz = np.asarray(xyz, np.float32)
    valid = np.asarray(valid, np.bool_)
    if velocity is None:
        velocity = np.zeros_like(xyz)
    velocity = np.asarray(velocity, np.float32)
    if velocity.shape != xyz.shape:
        raise ValueError("human velocity must match human XYZ shape")
    if ego_velocity_body is None:
        ego_velocity_body = np.zeros(3, np.float32)
    ego_velocity_body = np.asarray(
        ego_velocity_body, np.float32).reshape(3)
    if horizon_s < 0.0:
        raise ValueError("kinematic clearance horizon must be non-negative")
    obstacle_points, obstacle_velocities, obstacle_clearances = (
        _closest_body_capsule_obstacles(xyz, valid, velocity))
    if not len(obstacle_points):
        return (
            100.0,
            100.0,
            np.zeros(2, np.float32),
            np.zeros(2, np.float32),
        )
    minimum_index = int(np.argmin(obstacle_clearances))
    minimum = float(obstacle_clearances[minimum_index])
    global_closest_point = obstacle_points[minimum_index]
    kinematic_minimum = float("inf")
    for person_closest_point, person_closest_velocity, person_minimum in zip(
        obstacle_points, obstacle_velocities, obstacle_clearances
    ):
        relative_velocity = (
            np.asarray(person_closest_velocity, np.float64)
            - np.asarray(ego_velocity_body, np.float64)[:2]
        )
        relative_speed = float(np.linalg.norm(relative_velocity))
        if relative_speed > KINEMATIC_MAX_RELATIVE_SPEED_MPS:
            relative_velocity *= (
                KINEMATIC_MAX_RELATIVE_SPEED_MPS / relative_speed)
        horizontal_speed_sq = float(np.dot(
            relative_velocity[:2], relative_velocity[:2]))
        closest_time = 0.0
        if horizontal_speed_sq > 1.0e-6:
            closest_time = float(np.clip(
                -np.dot(person_closest_point[:2], relative_velocity[:2])
                / horizontal_speed_sq,
                0.0,
                horizon_s,
            ))
        future_point = np.asarray(person_closest_point, np.float64).copy()
        future_point[:2] += closest_time * relative_velocity
        future_clearance = (
            float(np.linalg.norm(future_point))
            - OBSERVED_COMBINED_BODY_RADIUS_M
        )
        person_kinematic_minimum = min(person_minimum, future_clearance)
        kinematic_minimum = min(
            kinematic_minimum, person_kinematic_minimum)
    escape_direction = _select_escape_direction(
        list(obstacle_points),
        list(obstacle_velocities),
        goal_direction_body_xy=goal_direction_body_xy,
        horizon_s=horizon_s,
    )
    return (
        float(minimum),
        float(kinematic_minimum),
        escape_direction,
        np.asarray(global_closest_point[:2], np.float32).copy(),
    )


def _observed_human_clearance_m(
    xyz: np.ndarray, valid: np.ndarray
) -> float:
    return _observed_human_geometry(xyz, valid)[0]


class EscapeDirectionLatch:
    """Slew-limit a causal escape direction in the world frame.

    The nearest skeleton capsule can change between people on consecutive
    frames. Directly following the instantaneous optimum makes the UAV reverse
    at 10 Hz, while holding it exactly can steer into a different moving
    person. A bounded angular rate gives continuity and still lets the chosen
    direction adapt when the previously open corridor becomes occupied.
    """

    def __init__(
        self,
        max_hold_s: float = ESCAPE_DIRECTION_MAX_HOLD_S,
        release_clearance_m: float = ESCAPE_DIRECTION_RELEASE_CLEARANCE_M,
        release_dwell_s: float = ESCAPE_DIRECTION_RELEASE_DWELL_S,
    ) -> None:
        self.max_hold_s = float(max_hold_s)
        self.release_clearance_m = float(release_clearance_m)
        self.release_dwell_s = float(release_dwell_s)
        if (
            self.max_hold_s <= 0.0
            or self.release_clearance_m <= 0.0
            or self.release_dwell_s <= 0.0
        ):
            raise ValueError("escape latch parameters must be positive")
        self.reset()

    def reset(self) -> None:
        self._world_direction: np.ndarray | None = None
        self._locked_at_s = float("-inf")
        self._last_update_s: float | None = None
        self._clear_since_s: float | None = None

    @staticmethod
    def _body_world_rotation(ego_state: np.ndarray) -> np.ndarray:
        yaw = math.atan2(float(ego_state[12]), float(ego_state[13]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.asarray(((cosine, -sine), (sine, cosine)), np.float32)

    def update(
        self,
        *,
        timestamp_s: float,
        ego_state: np.ndarray,
        observed_clearance_m: float,
        observed_away_body_xy: np.ndarray,
        emergency_zone: bool,
    ) -> tuple[np.ndarray, bool]:
        rotation = self._body_world_rotation(ego_state)
        current_body = np.asarray(
            observed_away_body_xy, np.float32).reshape(2).copy()
        norm = float(np.linalg.norm(current_body))
        current_world = None
        if norm > 1.0e-6:
            current_world = rotation @ (current_body / norm)

        elapsed = float(timestamp_s) - self._locked_at_s
        hold_expired = (
            self._world_direction is not None
            and not emergency_zone
            and elapsed >= self.max_hold_s
        )
        clear_observation = (
            not emergency_zone
            and (
                float(observed_clearance_m) >= self.release_clearance_m
                or hold_expired
            )
        )
        if clear_observation:
            if self._clear_since_s is None:
                self._clear_since_s = float(timestamp_s)
            elif (
                float(timestamp_s) - self._clear_since_s
                >= self.release_dwell_s
            ):
                self.reset()
        else:
            # A single close observation cancels the release timer.  This
            # hysteresis covers one-frame detector dropouts and momentary
            # nearest-person swaps in a dense crowd.
            self._clear_since_s = None
        if (
            emergency_zone
            and current_world is not None
            and self._world_direction is None
        ):
            self._world_direction = current_world
            self._locked_at_s = float(timestamp_s)
            self._last_update_s = float(timestamp_s)
        elif (
            emergency_zone
            and current_world is not None
            and self._world_direction is not None
            and float(observed_clearance_m)
            > ESCAPE_DIRECTION_FREEZE_CLEARANCE_M
        ):
            previous_angle = math.atan2(
                float(self._world_direction[1]),
                float(self._world_direction[0]))
            desired_angle = math.atan2(
                float(current_world[1]), float(current_world[0]))
            angle_error = math.atan2(
                math.sin(desired_angle - previous_angle),
                math.cos(desired_angle - previous_angle))
            dt = float(np.clip(
                float(timestamp_s) - float(self._last_update_s), 0.02, 0.20))
            maximum_turn = ESCAPE_DIRECTION_MAX_TURN_RATE_RAD_S * dt
            applied_turn = float(np.clip(
                angle_error, -maximum_turn, maximum_turn))
            updated_angle = previous_angle + applied_turn
            self._world_direction = np.asarray((
                math.cos(updated_angle), math.sin(updated_angle)), np.float32)
            self._last_update_s = float(timestamp_s)

        if self._world_direction is None:
            return current_body, False
        locked_body = rotation.T @ self._world_direction
        locked_norm = float(np.linalg.norm(locked_body))
        if locked_norm <= 1.0e-6:
            return current_body, False
        return (locked_body / locked_norm).astype(np.float32), True


class OnlineHumanSlots:
    """Causal online equivalent of the training dataset's slot assignment."""

    def __init__(self, max_people: int, empty_hold_s: float = 0.30) -> None:
        self.max_people = int(max_people)
        self.empty_hold_s = float(empty_hold_s)
        if self.max_people <= 0 or self.empty_hold_s <= 0.0:
            raise ValueError("Human slot count/empty hold must be positive")
        self.preferred_slot: dict[int, int] = {}
        self.previous_active = np.full(self.max_people, -1, np.int64)
        self.previous_tracks: dict[str, object] | None = None

    def reset(self) -> None:
        self.preferred_slot.clear()
        self.previous_active.fill(-1)
        self.previous_tracks = None

    def prospective_active_count(
        self, timestamp_s: float, raw_ids: np.ndarray,
    ) -> int:
        """Count current plus lifecycle-retained tracks before slot truncation.

        ``update`` deliberately keeps a missing track for up to two frames.
        Deployment support checks must include those retained tracks; checking
        only the current detector output can otherwise activate an unseen
        flattened Human rank after the check has already passed.
        """
        raw_ids = np.asarray(raw_ids, np.int64).reshape(-1)
        count = int(raw_ids.size)
        if self.previous_tracks is None:
            return count
        previous = self.previous_tracks
        elapsed = float(timestamp_s) - float(previous["timestamp_s"])
        if not (0.0 < elapsed <= self.empty_hold_s):
            return count
        previous_ids = np.asarray(previous["raw_ids"], np.int64)
        previous_missed = np.asarray(
            previous.get(
                "missed_frames", np.zeros(previous_ids.size, np.int64)),
            np.int64,
        )
        active_ids = set(int(value) for value in raw_ids)
        retained = sum(
            int(track_id) not in active_ids and int(missed) < 2
            for track_id, missed in zip(previous_ids, previous_missed)
        )
        return count + retained

    def update(
        self,
        ego_state: np.ndarray,
        timestamp_s: float,
        raw_ids: np.ndarray,
        xyz: np.ndarray,
        confidence: np.ndarray,
        joint_valid: np.ndarray,
        joint_measured: np.ndarray,
        joint_predicted: np.ndarray,
        root_velocity: np.ndarray,
        velocity_valid: np.ndarray,
        velocity_sigma_mps: np.ndarray,
        measurement_age_s: np.ndarray,
        track_age_frames: np.ndarray,
        prediction_run_frames: np.ndarray,
        identity_confidence: np.ndarray,
        goal_position: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        raw_ids = np.asarray(raw_ids, np.int64).reshape(-1)
        xyz = np.asarray(xyz, np.float32)
        confidence = np.asarray(confidence, np.float32)
        joint_valid = np.asarray(joint_valid, np.bool_)
        joint_measured = np.asarray(joint_measured, np.bool_)
        joint_predicted = np.asarray(joint_predicted, np.bool_)
        root_velocity = np.asarray(root_velocity, np.float32)
        velocity_valid = np.asarray(velocity_valid, np.bool_).reshape(-1)
        velocity_sigma_mps = np.asarray(
            velocity_sigma_mps, np.float32).reshape(-1)
        measurement_age_s = np.asarray(
            measurement_age_s, np.float32).reshape(-1)
        track_age_frames = np.asarray(track_age_frames, np.float32).reshape(-1)
        prediction_run_frames = np.asarray(
            prediction_run_frames, np.float32).reshape(-1)
        identity_confidence = np.asarray(
            identity_confidence, np.float32).reshape(-1)
        count = raw_ids.size
        source_count = count
        if xyz.shape != (count, JOINT_COUNT, 3):
            raise ValueError(f"human_xyz must be [{count},{JOINT_COUNT},3]")
        if confidence.shape != (count, JOINT_COUNT):
            raise ValueError(f"human_confidence must be [{count},{JOINT_COUNT}]")
        if joint_valid.shape != (count, JOINT_COUNT):
            raise ValueError(f"human_joint_valid must be [{count},{JOINT_COUNT}]")
        if joint_measured.shape != (count, JOINT_COUNT):
            raise ValueError(
                f"human_joint_measured must be [{count},{JOINT_COUNT}]")
        if joint_predicted.shape != (count, JOINT_COUNT):
            raise ValueError(
                f"human_joint_predicted must be [{count},{JOINT_COUNT}]")
        joint_measured &= joint_valid
        joint_predicted &= joint_valid & ~joint_measured
        xyz, joint_valid, topology_completed = (
            complete_coco12_topology_numpy(
                xyz, joint_valid, joint_measured))
        confidence = confidence.copy()
        confidence[topology_completed] = 0.0
        joint_measured &= ~topology_completed
        joint_predicted = (
            joint_predicted | topology_completed
        ) & joint_valid & ~joint_measured
        xyz, geometry_adjusted = sanitize_coco12_geometry_numpy(
            xyz, joint_valid)
        joint_measured &= ~geometry_adjusted
        joint_predicted = (
            joint_predicted | geometry_adjusted) & joint_valid & ~joint_measured
        if root_velocity.shape != (count, 3):
            raise ValueError(f"human_root_velocity must be [{count},3]")
        for name, value in (
            ("human_velocity_valid", velocity_valid),
            ("human_velocity_sigma_mps", velocity_sigma_mps),
            ("human_measurement_age_s", measurement_age_s),
            ("human_track_age_frames", track_age_frames),
            ("human_prediction_run_frames", prediction_run_frames),
            ("human_identity_confidence", identity_confidence),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must be [{count}]")
        if count and (raw_ids < 0).any():
            raise ValueError("active human_track_id values must be non-negative")
        if count and np.unique(raw_ids).size != count:
            raise ValueError("human_track_id values must be unique")
        if not (
            np.isfinite(xyz).all() and np.isfinite(confidence).all()
            and np.isfinite(root_velocity).all()
            and np.isfinite(velocity_sigma_mps).all()
            and np.isfinite(measurement_age_s).all()
            and np.isfinite(track_age_frames).all()
            and np.isfinite(prediction_run_frames).all()
            and np.isfinite(identity_confidence).all()
        ):
            raise ValueError("human observations contain NaN/Inf")

        rotation = _body_to_episode_rotation(ego_state)
        ego_position = np.asarray(ego_state[:3], np.float32)
        root_velocity = root_velocity.copy()
        root_velocity[~velocity_valid] = 0.0
        velocity = np.broadcast_to(
            root_velocity[:, None, :],
            (count, JOINT_COUNT, 3),
        ).copy()
        current_episode_xyz = (
            np.einsum("ij,nkj->nki", rotation, xyz)
            + ego_position[None, None, :]
        ).astype(np.float32)
        if self.previous_tracks is not None and count:
            previous = self.previous_tracks
            elapsed = float(timestamp_s) - float(previous["timestamp_s"])
            if elapsed > 1.0e-6:
                previous_ids = np.asarray(previous["raw_ids"], np.int64)
                previous_xyz = np.asarray(
                    previous["xyz_episode"], np.float32)
                previous_valid = np.asarray(
                    previous["joint_valid"], np.bool_)
                for index, track_id in enumerate(raw_ids):
                    matches = np.flatnonzero(previous_ids == int(track_id))
                    if matches.size != 1:
                        continue
                    old = int(matches[0])
                    joint_velocity_episode, supported = (
                        _bounded_joint_velocity_episode(
                            current_episode_xyz[index],
                            previous_xyz[old],
                            joint_valid[index],
                            previous_valid[old],
                            elapsed,
                        )
                    )
                    episode_velocity = np.broadcast_to(
                        (rotation @ root_velocity[index])[None, :],
                        (JOINT_COUNT, 3),
                    ).copy()
                    episode_velocity[supported] = (
                        joint_velocity_episode[supported])
                    velocity[index] = np.einsum(
                        "ij,kj->ki", rotation.T, episode_velocity)
        retained_empty_observation = False
        missed_frames = np.zeros(count, np.int64)
        if self.previous_tracks is not None:
            previous = self.previous_tracks
            elapsed = float(timestamp_s) - float(previous["timestamp_s"])
            if 0.0 < elapsed <= self.empty_hold_s:
                previous_ids = np.asarray(previous["raw_ids"], np.int64)
                previous_missed = np.asarray(
                    previous.get(
                        "missed_frames",
                        np.zeros(previous_ids.size, np.int64)),
                    np.int64,
                )
                active_ids = set(int(value) for value in raw_ids)
                held_indices = np.asarray([
                    index for index, track_id in enumerate(previous_ids)
                    if int(track_id) not in active_ids
                    and int(previous_missed[index]) < 2
                ], np.int64)
                if held_indices.size:
                    retained_empty_observation = source_count == 0
                    held_ids = previous_ids[held_indices]
                    held_root_episode_velocity = np.asarray(
                        previous["root_velocity_episode"],
                        np.float32)[held_indices]
                    held_joint_episode_velocity = np.asarray(
                        previous["joint_velocity_episode"],
                        np.float32)[held_indices]
                    held_episode_xyz = np.asarray(
                        previous["xyz_episode"],
                        np.float32)[held_indices].copy()
                    held_episode_xyz += (
                        held_joint_episode_velocity * elapsed)
                    held_xyz = np.einsum(
                        "ij,nkj->nki", rotation.T,
                        held_episode_xyz - ego_position[None, None, :],
                    ).astype(np.float32)
                    held_root_velocity = np.einsum(
                        "ij,nj->ni", rotation.T,
                        held_root_episode_velocity,
                    ).astype(np.float32)
                    held_joint_velocity = np.einsum(
                        "ij,nkj->nki", rotation.T,
                        held_joint_episode_velocity,
                    ).astype(np.float32)
                    held_joint_valid = np.asarray(
                        previous["joint_valid"], np.bool_)[held_indices]
                    held_count = held_indices.size
                    raw_ids = np.concatenate((raw_ids, held_ids))
                    xyz = np.concatenate((xyz, held_xyz), axis=0)
                    confidence = np.concatenate((
                        confidence,
                        np.zeros((held_count, JOINT_COUNT), np.float32),
                    ), axis=0)
                    joint_valid = np.concatenate((
                        joint_valid, held_joint_valid), axis=0)
                    joint_measured = np.concatenate((
                        joint_measured,
                        np.zeros_like(held_joint_valid),
                    ), axis=0)
                    joint_predicted = np.concatenate((
                        joint_predicted, held_joint_valid.copy()), axis=0)
                    root_velocity = np.concatenate((
                        root_velocity, held_root_velocity), axis=0)
                    velocity = np.concatenate((
                        velocity, held_joint_velocity), axis=0)
                    velocity_valid = np.concatenate((
                        velocity_valid,
                        np.asarray(previous["velocity_valid"], np.bool_)[
                            held_indices],
                    ), axis=0)
                    velocity_sigma_mps = np.concatenate((
                        velocity_sigma_mps,
                        np.asarray(
                            previous["velocity_sigma_mps"],
                            np.float32)[held_indices],
                    ), axis=0)
                    identity_confidence = np.concatenate((
                        identity_confidence,
                        np.asarray(
                            previous["identity_confidence"],
                            np.float32)[held_indices],
                    ), axis=0)
                    measurement_age_s = np.concatenate((
                        measurement_age_s,
                        np.asarray(previous["measurement_age_s"], np.float32)[
                            held_indices] + elapsed,
                    ), axis=0)
                    track_age_frames = np.concatenate((
                        track_age_frames,
                        np.asarray(previous["track_age_frames"], np.float32)[
                            held_indices] + 1.0,
                    ), axis=0)
                    prediction_run_frames = np.concatenate((
                        prediction_run_frames,
                        np.asarray(
                            previous["prediction_run_frames"],
                            np.float32)[held_indices] + 1.0,
                    ), axis=0)
                    missed_frames = np.concatenate((
                        missed_frames, previous_missed[held_indices] + 1))
                    count = raw_ids.size

        xyz, propagated_geometry_adjusted = sanitize_coco12_geometry_numpy(
            xyz, joint_valid)
        joint_measured &= ~propagated_geometry_adjusted
        joint_predicted = (
            joint_predicted | propagated_geometry_adjusted
        ) & joint_valid & ~joint_measured

        # Training reserves -1 for padding, so detector ID zero becomes one.
        ids = raw_ids + 1
        if count > 0:
            self.previous_tracks = {
                "timestamp_s": float(timestamp_s),
                "raw_ids": raw_ids.copy(),
                "xyz_episode": (
                    np.einsum("ij,nkj->nki", rotation, xyz)
                    + ego_position[None, None, :]
                ).astype(np.float32),
                "confidence": confidence.copy(),
                "joint_valid": joint_valid.copy(),
                "root_velocity_episode": np.einsum(
                    "ij,nj->ni", rotation, root_velocity,
                ).astype(np.float32),
                "joint_velocity_episode": np.einsum(
                    "ij,nkj->nki", rotation, velocity,
                ).astype(np.float32),
                "velocity_valid": velocity_valid.copy(),
                "velocity_sigma_mps": velocity_sigma_mps.copy(),
                "measurement_age_s": measurement_age_s.copy(),
                "track_age_frames": track_age_frames.copy(),
                "prediction_run_frames": prediction_run_frames.copy(),
                "identity_confidence": identity_confidence.copy(),
                "missed_frames": missed_frames.copy(),
            }
        else:
            self.previous_tracks = None

        skeleton_detected = np.concatenate(
            (xyz, velocity, confidence[..., None]), axis=-1
        ).astype(np.float32)
        skeleton_detected[~joint_valid] = 0.0
        ego_velocity_body = _ego_velocity_full_body(ego_state)
        risk = _risk_scores(
            skeleton_detected,
            joint_valid,
            ego_velocity_body=ego_velocity_body,
        )
        order = np.lexsort((ids, -risk))[: self.max_people]

        used: set[int] = set()
        assignment: dict[int, int] = {}
        for detection_index in order:
            track_id = int(ids[detection_index])
            slot = self.preferred_slot.get(track_id)
            if slot is not None and slot not in used:
                assignment[int(detection_index)] = slot
                used.add(slot)
        free = iter(slot for slot in range(self.max_people) if slot not in used)
        for detection_index in order:
            index = int(detection_index)
            if index not in assignment:
                assignment[index] = next(free)

        skeleton = np.zeros(
            (self.max_people, JOINT_COUNT, FEATURE_DIM), np.float32)
        human_mask = np.zeros(self.max_people, np.bool_)
        joint_mask = np.zeros((self.max_people, JOINT_COUNT), np.bool_)
        human_ids = np.full(self.max_people, -1, np.int64)
        human_is_first = np.zeros(self.max_people, np.bool_)
        slotted_root_velocity = np.zeros(
            (self.max_people, 3), np.float32)
        observation_quality = np.zeros((self.max_people, 7), np.float32)
        current_active = np.full(self.max_people, -1, np.int64)
        for detection_index, slot in assignment.items():
            track_id = int(ids[detection_index])
            self.preferred_slot[track_id] = slot
            human_mask[slot] = True
            joint_mask[slot] = joint_valid[detection_index]
            human_ids[slot] = track_id
            current_active[slot] = track_id
            human_is_first[slot] = self.previous_active[slot] != track_id
            skeleton[slot] = skeleton_detected[detection_index]
            slotted_root_velocity[slot] = root_velocity[detection_index]
            valid_count = max(1, int(joint_valid[detection_index].sum()))
            measured_ratio = float(
                joint_measured[detection_index].sum()) / valid_count
            predicted_ratio = float(
                joint_predicted[detection_index].sum()) / valid_count
            observation_quality[slot] = (
                measured_ratio,
                predicted_ratio,
                float(track_age_frames[detection_index]) * 0.1,
                float(measurement_age_s[detection_index]),
                float(prediction_run_frames[detection_index]),
                float(velocity_sigma_mps[detection_index])
                if bool(velocity_valid[detection_index]) else 0.0,
                float(bool(velocity_valid[detection_index])) * float(
                    np.clip(identity_confidence[detection_index], 0.0, 1.0)),
            )
        self.previous_active = current_active

        human_root, human_joints = human_root_and_relative_joints_numpy(
            skeleton[None], human_mask[None], joint_mask[None])
        human_root[0, ..., 3:6] = slotted_root_velocity
        human_joints[0, ..., 3:6] = (
            skeleton[..., 3:6] - slotted_root_velocity[:, None, :])
        human_joints[0, ~joint_mask] = 0.0
        (
            observed_clearance,
            kinematic_clearance,
            observed_away,
            nearest_human_xy,
        ) = _observed_human_geometry(
            xyz,
            joint_valid,
            velocity=velocity,
            ego_velocity_body=ego_velocity_body,
            goal_direction_body_xy=(
                None
                if goal_position is None
                else (
                    rotation.T
                    @ (np.asarray(goal_position, np.float32) - ego_position)
                )[:2]
            ),
        )
        (
            safety_obstacle_points,
            safety_obstacle_velocities,
            safety_obstacle_clearances,
        ) = _all_body_capsule_obstacles(xyz, joint_valid, velocity)
        return {
            "skeleton": skeleton,
            "human_root": human_root[0],
            "human_joints": human_joints[0],
            "human_mask": human_mask,
            "joint_mask": joint_mask,
            "human_ids": human_ids,
            "human_is_first": human_is_first,
            "human_observation_quality": observation_quality,
            "truncated_people": np.asarray(max(0, count - len(order)), np.int32),
            "retained_empty_observation": np.asarray(
                retained_empty_observation, np.bool_),
            "observed_human_clearance_m": np.asarray(
                observed_clearance, np.float32),
            "kinematic_human_clearance_m": np.asarray(
                kinematic_clearance, np.float32),
            "observed_human_away_xy": observed_away,
            "observed_nearest_human_xy": nearest_human_xy,
            "safety_obstacle_points": safety_obstacle_points,
            "safety_obstacle_velocities": safety_obstacle_velocities,
            "safety_obstacle_clearances": safety_obstacle_clearances,
        }


class R2CandidateCommitment:
    """Keep one Event-selected candidate direction across replans.

    The planner still evaluates every configured interval, but a different
    candidate may replace the committed one when it materially lowers Event
    risk, or when both actions are inside the safe-risk budget and the new
    action materially improves goal progress. This prevents alternating
    lateral offsets from cancelling each other at 5 Hz without locking the
    vehicle into a safe but goal-diverging action.
    """

    def __init__(
        self, hold_steps: int = 8, switch_risk_improvement: float = 0.03,
        safe_risk_cap: float = 0.08,
        safe_risk_margin: float = 0.02,
        switch_progress_improvement: float = 0.25,
    ) -> None:
        self.hold_steps = int(hold_steps)
        self.switch_risk_improvement = float(switch_risk_improvement)
        self.safe_risk_cap = float(safe_risk_cap)
        self.safe_risk_margin = float(safe_risk_margin)
        self.switch_progress_improvement = float(
            switch_progress_improvement)
        if self.hold_steps <= 0:
            raise ValueError("candidate commitment hold_steps must be positive")
        if (
            not math.isfinite(self.switch_risk_improvement)
            or self.switch_risk_improvement < 0.0
        ):
            raise ValueError("candidate switch margin must be non-negative")
        if not 0.0 < self.safe_risk_cap < 1.0:
            raise ValueError("candidate safe-risk cap must be within (0,1)")
        if (
            not math.isfinite(self.safe_risk_margin)
            or self.safe_risk_margin < 0.0
        ):
            raise ValueError("candidate safe-risk margin must be non-negative")
        if (
            not math.isfinite(self.switch_progress_improvement)
            or self.switch_progress_improvement < 0.0
        ):
            raise ValueError(
                "candidate progress switch margin must be non-negative")
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.remaining_steps = 0

    def select(
        self,
        proposed_index: int,
        candidate_risk: np.ndarray,
        candidate_progress: np.ndarray,
        candidate_supported: np.ndarray,
    ) -> tuple[int, dict[str, object]]:
        risk = np.asarray(candidate_risk, np.float64).reshape(-1)
        progress = np.asarray(candidate_progress, np.float64).reshape(-1)
        supported = np.asarray(candidate_supported, np.bool_).reshape(-1)
        proposed_index = int(proposed_index)
        if (
            risk.shape != supported.shape
            or progress.shape != risk.shape
            or risk.size == 0
        ):
            raise ValueError(
                "candidate commitment risk/progress/support mismatch")
        if not 0 <= proposed_index < risk.size:
            raise ValueError("candidate commitment proposed index is invalid")
        if not np.isfinite(risk).all() or not np.isfinite(progress).all():
            raise ValueError(
                "candidate commitment scores must be finite")
        if not bool(supported[proposed_index]):
            if not bool(supported.any()):
                # Preserve the planner's explicit out-of-support Actor
                # fallback.  A stale committed offset must never leak into a
                # state where none of the candidate actions is supported.
                self.reset()
                return proposed_index, {
                    "active": False,
                    "held": False,
                    "switched": False,
                    "committed_index": 0,
                    "remaining_steps": 0,
                    "switch_risk_improvement": float("nan"),
                    "switch_risk_margin": self.switch_risk_improvement,
                    "switch_progress_improvement": float("nan"),
                    "switch_progress_margin": (
                        self.switch_progress_improvement),
                    "safe_risk_cap": self.safe_risk_cap,
                    "safe_risk_margin": self.safe_risk_margin,
                    "unsupported_fallback": True,
                }
            raise ValueError(
                "candidate commitment proposal is unsupported while another "
                "candidate is available")

        active = (
            self.remaining_steps > 0
            and self.index != 0
            and self.index < risk.size
            and bool(supported[self.index])
        )
        selected = proposed_index
        held = False
        switched = False
        improvement = float("nan")
        progress_improvement = float("nan")
        if active and proposed_index != self.index:
            improvement = float(risk[self.index] - risk[proposed_index])
            progress_improvement = float(
                progress[proposed_index] - progress[self.index])
            minimum_risk = float(np.min(risk[supported]))
            relative_cap = minimum_risk + self.safe_risk_margin
            current_safe = bool(
                risk[self.index] <= self.safe_risk_cap
                and risk[self.index] <= relative_cap)
            proposed_safe = bool(
                risk[proposed_index] <= self.safe_risk_cap
                and risk[proposed_index] <= relative_cap)
            if current_safe and proposed_safe:
                switched = bool(
                    progress_improvement
                    >= self.switch_progress_improvement)
            elif not current_safe and proposed_safe:
                switched = True
            elif current_safe and not proposed_safe:
                switched = False
            else:
                switched = bool(
                    improvement >= self.switch_risk_improvement)
            if not switched:
                selected = self.index
                held = True

        if selected != 0 and (
            not active or switched or selected != self.index
        ):
            self.index = int(selected)
            self.remaining_steps = self.hold_steps
        elif selected == 0 and not held:
            self.reset()

        return int(selected), {
            "active": bool(self.remaining_steps > 0 and self.index != 0),
            "held": held,
            "switched": switched,
            "committed_index": int(self.index),
            "remaining_steps": int(self.remaining_steps),
            "switch_risk_improvement": improvement,
            "switch_risk_margin": self.switch_risk_improvement,
            "switch_progress_improvement": progress_improvement,
            "switch_progress_margin": self.switch_progress_improvement,
            "safe_risk_cap": self.safe_risk_cap,
            "safe_risk_margin": self.safe_risk_margin,
            "unsupported_fallback": False,
        }

    def advance(self) -> None:
        if self.remaining_steps > 0:
            self.remaining_steps -= 1
        if self.remaining_steps == 0:
            self.index = 0


class SmoothRandomPrefillExplorer:
    """Temporally correlated random policy used only during replay prefill.

    A target is held for a short simulated-time interval, while the returned
    command approaches it through a first-order low-pass and a per-second slew
    bound.  This prevents 10-Hz independent samples from turning camera shake
    into the dominant training signal.  The caller records the returned,
    actually applied action rather than the latent target.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.value = np.zeros(3, np.float32)
        self.target = np.zeros(3, np.float32)
        self.target_until_s = float("-inf")
        self.last_timestamp_s: float | None = None
        # Formal pure-Dreamer readiness requires observed applied support beyond
        # +/-0.25 on every policy axis before the first optimizer step.  The old
        # +/-0.22 yaw target made that gate mathematically unreachable on a
        # fresh run, so the Actor could never take over from random prefill.
        # Leave headroom for the low-pass/slew dynamics to actually cross 0.25.
        self.target_limits = np.asarray((0.70, 0.55, 0.40), np.float32)
        self.slew_per_s = np.asarray((1.20, 1.00, 0.45), np.float32)
        self.time_constant_s = 0.35

    def policy_contract(self) -> dict[str, object]:
        """Return the complete, recorder-safe identity of this explorer."""
        return {
            "schema": "omninxt.collection-policy.v1",
            "mode": "smooth_random_training_prefill",
            "actor_action_used": False,
            "checkpoint_state_used_for_action": False,
            "target_sampling": "independent_uniform_per_axis",
            "target_limits": self.target_limits.astype(float).tolist(),
            "target_hold_range_s": [0.5, 1.0],
            "slew_per_s": self.slew_per_s.astype(float).tolist(),
            "time_constant_s": float(self.time_constant_s),
            "timestamp_source": "simulation_time",
            "output": "normalized_policy3_before_fixed_action_adapter",
        }

    def episode_evidence(self) -> dict[str, object]:
        return {"random_seed_state_is_checkpoint_clock_derived": True}

    def reset(self, *, collection_scene_seed: int | None = None) -> None:
        del collection_scene_seed
        self.value.fill(0.0)
        self.target.fill(0.0)
        self.target_until_s = float("-inf")
        self.last_timestamp_s = None

    def _new_target(self, timestamp_s: float) -> None:
        self.target = self.rng.uniform(
            -self.target_limits, self.target_limits).astype(np.float32)
        self.target_until_s = float(timestamp_s) + float(
            self.rng.uniform(0.5, 1.0))

    def step(self, timestamp_s: float) -> np.ndarray:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("prefill explorer timestamp must be finite")
        if self.last_timestamp_s is None or timestamp_s < self.last_timestamp_s:
            self.reset()
            self.last_timestamp_s = timestamp_s
            self._new_target(timestamp_s)
            return self.value.copy()
        dt = float(np.clip(timestamp_s - self.last_timestamp_s, 1.0e-3, 0.25))
        self.last_timestamp_s = timestamp_s
        if timestamp_s >= self.target_until_s:
            self._new_target(timestamp_s)
        alpha = 1.0 - math.exp(-dt / self.time_constant_s)
        desired_delta = alpha * (self.target - self.value)
        bounded_delta = np.clip(
            desired_delta, -self.slew_per_s * dt, self.slew_per_s * dt)
        self.value = np.clip(
            self.value + bounded_delta, -1.0, 1.0).astype(np.float32)
        return self.value.copy()


class ProspectiveValidationExplorer:
    """Actor-independent validation excitation frozen before model fitting.

    The long axis plateaus deliberately reach the real normalized command
    boundary after the same low-pass and slew limits used by random prefill.
    Starting at a different phase on each reset prevents early termination
    from systematically removing one action axis from the held-out replay.
    """

    TARGET_SEQUENCE = np.asarray((
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0),
    ), dtype=np.float32)

    def __init__(self) -> None:
        self.target_limits = np.ones(3, np.float32)
        self.slew_per_s = np.asarray((1.20, 1.00, 0.45), np.float32)
        self.time_constant_s = 0.35
        self.target_hold_s = 5.0
        self.episode_ordinal = -1
        self.initial_target_index = -1
        self.value = np.zeros(3, np.float32)
        self.target = np.zeros(3, np.float32)
        self.target_until_s = float("-inf")
        self.last_timestamp_s: float | None = None
        self.target_index = 0

    def policy_contract(self) -> dict[str, object]:
        return {
            "schema": "omninxt.collection-policy.v1",
            "mode": "prospective_actor_independent_axis_saturation_v1",
            "actor_action_used": False,
            "checkpoint_state_used_for_action": False,
            "randomness": "none",
            "target_sequence": self.TARGET_SEQUENCE.astype(float).tolist(),
            "target_limits": [1.0, 1.0, 1.0],
            "target_hold_s": float(self.target_hold_s),
            "episode_start_phase": "crowd_seed_mod_target_count",
            "slew_per_s": [1.2, 1.0, 0.45],
            "time_constant_s": float(self.time_constant_s),
            "timestamp_source": "simulation_time",
            "output": "normalized_policy3_before_fixed_action_adapter",
        }

    def episode_evidence(self) -> dict[str, object]:
        return {
            "episode_ordinal": int(self.episode_ordinal),
            "collection_scene_seed": int(self.collection_scene_seed),
            "initial_target_index": int(self.initial_target_index),
        }

    def reset(self, *, collection_scene_seed: int | None = None) -> None:
        if (
            not isinstance(collection_scene_seed, int)
            or isinstance(collection_scene_seed, bool)
            or collection_scene_seed < 0
        ):
            raise ValueError(
                "prospective validation collection requires a non-negative "
                "integer scene seed at reset")
        self.episode_ordinal += 1
        self.collection_scene_seed = int(collection_scene_seed)
        self.initial_target_index = (
            self.collection_scene_seed % int(self.TARGET_SEQUENCE.shape[0]))
        self.target_index = self.initial_target_index
        self.value.fill(0.0)
        self.target = self.TARGET_SEQUENCE[self.target_index].copy()
        self.target_until_s = float("-inf")
        self.last_timestamp_s = None

    def _advance_target(self, timestamp_s: float) -> None:
        self.target_index = (
            (self.target_index + 1) % int(self.TARGET_SEQUENCE.shape[0]))
        self.target = self.TARGET_SEQUENCE[self.target_index].copy()
        self.target_until_s = float(timestamp_s) + self.target_hold_s

    def step(self, timestamp_s: float) -> np.ndarray:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("validation explorer timestamp must be finite")
        if self.last_timestamp_s is None:
            self.last_timestamp_s = timestamp_s
            self.target_until_s = timestamp_s + self.target_hold_s
            return self.value.copy()
        if timestamp_s < self.last_timestamp_s:
            raise RuntimeError(
                "validation explorer time moved backwards without reset")
        dt = float(np.clip(timestamp_s - self.last_timestamp_s, 1.0e-3, 0.25))
        self.last_timestamp_s = timestamp_s
        if timestamp_s >= self.target_until_s:
            self._advance_target(timestamp_s)
        alpha = 1.0 - math.exp(-dt / self.time_constant_s)
        desired_delta = alpha * (self.target - self.value)
        bounded_delta = np.clip(
            desired_delta, -self.slew_per_s * dt, self.slew_per_s * dt)
        self.value = np.clip(
            self.value + bounded_delta, -1.0, 1.0).astype(np.float32)
        return self.value.copy()


class FactorizedPolicyRuntime:
    def __init__(self, checkpoint_path: Path, device: str, *,
                 allow_unsafe_policy: bool = False,
                 world_model_planner: bool = False,
                 r2_candidate_planner: bool = False,
                 r2_planner_horizon_steps: int | None = None,
                 r2_planner_interval_steps: int = 2,
                 r2_planner_risk_tolerance: float = 0.002,
                 r2_planner_safe_risk_cap: float = 0.08,
                 r2_planner_safe_risk_margin: float = 0.02,
                 r2_planner_boundary_buffer_m: float = 0.35,
                 r2_planner_commitment_steps: int = 8,
                 r2_planner_switch_risk_improvement: float = 0.03,
                 r2_planner_switch_progress_improvement: float = 0.25,
                 r2_planner_rollout_diagnostics: bool = False,
                 human_empty_hold_s: float = 0.30,
                 shadow_imagination: bool = False,
                 shadow_horizon_steps: int = 15,
                 shadow_interval_steps: int = 10,
                 shadow_step_duration_s: float = 0.1,
                 online_exploration_scale: float = 0.0,
                 exploration_delta_limits=(0.05, 0.12, 0.015, 0.03),
                 exploration_disable_clearance_m: float = 0.55,
                 exploration_full_clearance_m: float = 1.50,
                 stochastic_actor: bool = False,
                 horizontal_altitude_hold: bool = False,
                 altitude_hold_target_agl_m: float = 1.0,
                 altitude_hold_kp: float = 0.8,
                 altitude_hold_max_action: float = 0.20) -> None:
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict

        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False)
        self.policy_validation = checkpoint.get("validation_metrics") or {}
        required_policy_metrics = {
            "policy/action_mae": 0.25,
            "policy/mae_vx": 0.20,
            "policy/mae_vy": 0.20,
            "policy/mae_yaw_rate": 0.10,
            "policy/saturation_ratio": 0.05,
            "policy/unsafe_vertical_ratio": 0.01,
        }
        declared_architecture_version = str(
            checkpoint.get("architecture_version"))
        objective_version = str(
            checkpoint.get("training_objective_version", ""))
        # The pure-Actor entry point binds this class attribute to its single
        # current trainer objective. Generic/historical factorized serving
        # retains the last v8.4 compatibility objective by default.
        v84_pure_objective = str(getattr(
            self,
            "EXPECTED_PURE_OBJECTIVE_VERSION",
            "factorized_pure_dreamer_v42_route_occupancy_actor",
        ))
        current_pure_objective = {
            "factorized_dreamer_v8.3": (
                "factorized_pure_dreamer_v38_actionable_collision_precursors"),
            "factorized_dreamer_v8.4": v84_pure_objective,
            "factorized_dreamer_v8.5": v84_pure_objective,
            "factorized_dreamer_v8.6": v84_pure_objective,
            "factorized_dreamer_v8.7": v84_pure_objective,
        }.get(declared_architecture_version)
        if (
            current_pure_objective is not None
            and objective_version != current_pure_objective
        ):
            raise RuntimeError(
                f"{declared_architecture_version} has an inconsistent pure "
                "Dreamer objective")
        # Current pure-Dreamer versions reuse the audited v8.2 external state,
        # task and execution contracts.  The active pure_dreamer config still
        # constructs their exact current modules; checkpoint/model shape and
        # decision-encoder fingerprints are verified by the pure runtime.
        architecture_version = (
            "factorized_dreamer_v8.2"
            if declared_architecture_version in (
                "factorized_dreamer_v8.3", "factorized_dreamer_v8.4",
                "factorized_dreamer_v8.5", "factorized_dreamer_v8.6",
                "factorized_dreamer_v8.7")
            else declared_architecture_version
        )
        if architecture_version not in (
            "factorized_dreamer_v6.2", "factorized_dreamer_v6.3",
            "factorized_dreamer_v6.4", "factorized_dreamer_v6.5",
            "factorized_dreamer_v6.6", "factorized_dreamer_v6.7",
            "factorized_dreamer_v6.8", "factorized_dreamer_v6.9",
            "factorized_dreamer_v7.0", "factorized_dreamer_v7.1",
            "factorized_dreamer_v7.2", "factorized_dreamer_v7.3",
            "factorized_dreamer_v7.4", "factorized_dreamer_v7.6",
            "factorized_dreamer_v7.7", "factorized_dreamer_v7.8",
            "factorized_dreamer_v7.9", "factorized_dreamer_v8.0",
            "factorized_dreamer_v8.1", "factorized_dreamer_v8.2",
        ):
            required_policy_metrics["policy/mae_vz"] = 0.10
        self.architecture_version = declared_architecture_version
        audit_errors = []
        reflection_enabled, reflection_slots = (
            checkpoint_actor_human_reflection_contract(checkpoint))
        authoritative_ego_token = (
            checkpoint_actor_authoritative_ego_token_enabled(checkpoint))
        direct_ego_task_state = _checkpoint_bool(
            checkpoint, "actor_direct_ego_task_state_enabled", default=False)
        human_conditioned_residual = (
            checkpoint_actor_human_conditioned_residual_enabled(checkpoint))
        unified_policy = checkpoint_actor_unified_policy_enabled(checkpoint)
        permutation_invariant_entities = _checkpoint_bool(
            checkpoint,
            "permutation_invariant_decision_entities_enabled",
            default=False,
        )
        metric_root = _checkpoint_bool(
            checkpoint, "human_root_metric_scaling_enabled", default=False)
        metric_ego = _checkpoint_bool(
            checkpoint, "ego_metric_scaling_enabled", default=False)
        metric_goal = _checkpoint_bool(
            checkpoint, "goal_metric_scaling_enabled", default=False)
        metric_quality = _checkpoint_bool(
            checkpoint, "human_quality_metric_scaling_enabled", default=False)
        critic_full_state = _checkpoint_bool(
            checkpoint, "critic_full_state_enabled", default=False)
        direct_task_geometry = _checkpoint_bool(
            checkpoint, "direct_task_geometry_enabled", default=False)
        task_memory_enabled = _checkpoint_bool(
            checkpoint, "task_memory_enabled", default=False)
        direct_task_memory = _checkpoint_bool(
            checkpoint, "direct_task_memory_enabled", default=False)
        analytic_task_memory_events = _checkpoint_bool(
            checkpoint, "analytic_task_memory_events_enabled", default=False)
        explicit_joint_kinematics = _checkpoint_bool(
            checkpoint, "explicit_joint_kinematics_enabled", default=False)
        physical_fields_per_slot = int(checkpoint.get(
            "actor_human_physical_fields_per_slot", 6))
        actor_human_geometry_dim = int(checkpoint.get(
            "actor_human_geometry_dim", 0))
        actor_input_dim = int(checkpoint.get("actor_input_dim", 0))
        critic_input_dim = int(checkpoint.get("critic_input_dim", 0))
        actor_task_physical_state = _checkpoint_bool(
            checkpoint, "actor_task_physical_state_enabled", default=False)
        actor_task_obstacle_slots = int(checkpoint.get(
            "actor_task_physical_obstacle_slots", 0))
        actor_task_state_dim = int(checkpoint.get(
            "actor_task_physical_state_dim", 0))
        event_actor_full_state_slots = int(checkpoint.get(
            "event_actor_full_state_slots", 0))
        event_actor_full_fields = int(checkpoint.get(
            "event_actor_full_fields_per_slot", 0))
        event_actor_full_learned = _checkpoint_bool(
            checkpoint, "event_actor_full_learned_per_slot_enabled",
            default=False)
        event_actor_full_learned_fields = int(checkpoint.get(
            "event_actor_full_learned_fields_per_slot", 0))
        actor_presence_physical = _checkpoint_bool(
            checkpoint,
            "actor_human_physical_presence_state_enabled",
            default=False,
        )
        internal_action_adapter = _checkpoint_bool(
            checkpoint, "internal_action_adapter_enabled", default=False)
        action_smoother_enabled = _checkpoint_bool(
            checkpoint, "action_smoother_enabled", default=False)
        action_adapter_contract = checkpoint.get("action_adapter_contract")
        event_presence_physical = _checkpoint_bool(
            checkpoint, "human_presence_physical_enabled", default=False)
        event_full_articulated_state = _checkpoint_bool(
            checkpoint, "event_full_articulated_state_enabled", default=False)
        kinematic_joints = _checkpoint_bool(
            checkpoint, "human_joint_kinematic_enabled", default=False)
        conservative_cv = _checkpoint_bool(
            checkpoint, "conservative_cv_clearance_enabled", default=True)
        deterministic_evaluation_state = _checkpoint_bool(
            checkpoint, "deterministic_evaluation_state_enabled",
            default=False)
        root_velocity_residual = float(checkpoint.get(
            "human_root_max_velocity_residual_mps", 0.35))
        joint_velocity_residual = float(checkpoint.get(
            "human_joint_max_velocity_residual_mps", 0.35))
        root_max_speed = checkpoint.get("human_root_max_speed_mps")
        joint_max_speed = checkpoint.get("human_joint_max_speed_mps")
        root_max_speed = (
            None if root_max_speed is None else float(root_max_speed))
        joint_max_speed = (
            None if joint_max_speed is None else float(joint_max_speed))
        if min(root_velocity_residual, joint_velocity_residual) <= 0.0:
            raise RuntimeError(
                "checkpoint Human velocity residual bounds must be positive")
        if any(
            value is not None and (
                not math.isfinite(value) or value <= 0.0)
            for value in (root_max_speed, joint_max_speed)
        ):
            raise RuntimeError(
                "checkpoint Human absolute speed bounds must be positive")
        geometry_topk_slots = (
            checkpoint_human_geometry_topk_physical_slots(checkpoint))
        if architecture_version == "factorized_dreamer_v7.3":
            if objective_version != (
                "factorized_pure_dreamer_v20_complete_markov_task"
            ):
                raise RuntimeError(
                    "v7.3 requires the complete-Markov v20 objective")
            required_v73 = {
                "goal metric scaling": metric_goal,
                "Ego metric scaling": metric_ego,
                "Human root metric scaling": metric_root,
                "Human quality metric scaling": metric_quality,
                "unified Actor": unified_policy,
                "full-state Critic": critic_full_state,
                "injective direct Ego/task state": direct_ego_task_state,
                "direct task geometry": direct_task_geometry,
                "task memory": task_memory_enabled,
                "direct task memory": direct_task_memory,
                "analytic task-memory events": analytic_task_memory_events,
                "explicit joint kinematics": explicit_joint_kinematics,
                "kinematic joints": kinematic_joints,
                "Actor Human lifecycle state": actor_presence_physical,
                "Event Human lifecycle state": event_presence_physical,
                "model-owned ActionAdapter": internal_action_adapter,
                "model-owned action smoother": action_smoother_enabled,
            }
            missing_v73 = [
                name for name, enabled in required_v73.items() if not enabled]
            if missing_v73:
                raise RuntimeError(
                    "incomplete v7.3 checkpoint contract: "
                    + ", ".join(missing_v73))
            if reflection_enabled or human_conditioned_residual or conservative_cv:
                raise RuntimeError(
                    "v7.3 forbids reflection, residual branches, and CV vetoes")
            if physical_fields_per_slot != 12 or geometry_topk_slots != 10:
                raise RuntimeError(
                    "v7.3 explicit Human geometry must be 10x12")
            required_adapter_fields = {
                "target_altitude_agl_m", "altitude_kp",
                "vertical_velocity_kd", "maximum_vertical_action",
            }
            if (
                not isinstance(action_adapter_contract, dict)
                or set(action_adapter_contract) != required_adapter_fields
                or not all(math.isfinite(float(value)) for value in
                           action_adapter_contract.values())
            ):
                raise RuntimeError(
                    "v7.3 lacks an exact model-owned ActionAdapter contract")
            if (
                world_model_planner
                or r2_candidate_planner
                or horizontal_altitude_hold
                or float(online_exploration_scale) > 0.0
            ):
                raise RuntimeError(
                    "v7.3 single-Actor execution forbids planners, external "
                    "altitude hold, and Actor-external exploration noise")
            if (
                int(checkpoint.get("actor_update_count", 0)) > 0
                and not isinstance(checkpoint.get("replay_task_contract"), dict)
            ):
                raise RuntimeError(
                    "trained v7.3 checkpoint lacks an audited replay/task contract")
        if architecture_version in (
            "factorized_dreamer_v7.4", "factorized_dreamer_v7.6",
            "factorized_dreamer_v7.7", "factorized_dreamer_v7.8",
            "factorized_dreamer_v7.9", "factorized_dreamer_v8.0",
            "factorized_dreamer_v8.1", "factorized_dreamer_v8.2",
        ):
            expected_complete_objective = (
                "factorized_pure_dreamer_v22_audited_complete_state"
                if architecture_version == "factorized_dreamer_v7.4" else
                (
                    None
                    if architecture_version in (
                        "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                    else (
                        "factorized_pure_dreamer_v28_actor_gradient_headroom"
                        if architecture_version == "factorized_dreamer_v8.0"
                        else (
                            "factorized_pure_dreamer_v26_bounded_human_kinematics"
                            if architecture_version == "factorized_dreamer_v7.9"
                            else (
                                "factorized_pure_dreamer_v25_event_time_quadrature"
                            )
                        )
                    )
                )
            )
            valid_action_objectives = {
                (
                    "factorized_pure_dreamer_v30_goal_heading_potential",
                    "affine_reachable_interval_slew_bounded_v1",
                    "affine_reachable_interval",
                ),
                (
                    "factorized_pure_dreamer_v33_uncertain_swept_human_contact",
                    "soft_absolute_target_mean_reverting_slew_bounded_v1",
                    "soft_absolute_target",
                ),
            }
            if architecture_version in (
                "factorized_dreamer_v8.1", "factorized_dreamer_v8.2"):
                action_objective = (
                    objective_version,
                    checkpoint.get("action_smoother_contract"),
                    checkpoint.get("action_smoother_parameterization"),
                )
                expected_action_objectives = (
                    {(
                        ({
                            "factorized_dreamer_v8.3": (
                                "factorized_pure_dreamer_v38_"
                                "actionable_collision_precursors"),
                            "factorized_dreamer_v8.4": v84_pure_objective,
                            "factorized_dreamer_v8.5": v84_pure_objective,
                            "factorized_dreamer_v8.6": v84_pure_objective,
                            "factorized_dreamer_v8.7": v84_pure_objective,
                        }.get(
                            declared_architecture_version,
                            "factorized_pure_dreamer_v34_"
                            "unimix_delta_coupling")),
                        "soft_absolute_target_mean_reverting_slew_bounded_v1",
                        "soft_absolute_target",
                    )}
                    if architecture_version == "factorized_dreamer_v8.2"
                    else valid_action_objectives
                )
                if action_objective not in expected_action_objectives:
                    raise RuntimeError(
                        f"{architecture_version} has an inconsistent "
                        "objective/action-smoother contract")
            elif objective_version != expected_complete_objective:
                raise RuntimeError(
                    f"{architecture_version} requires its audited "
                    "complete-state objective")
            if (
                architecture_version == "factorized_dreamer_v7.6"
                and checkpoint.get("control_transition_contract")
                != "acknowledged_physical_control_step_v1"
            ):
                raise RuntimeError(
                    "v7.6 requires the acknowledged physical control-step "
                    "contract")
            if (
                architecture_version == "factorized_dreamer_v7.6"
                and checkpoint.get("rssm_initial_state_contract")
                != "learned_prior_categorical_reset_v1"
            ):
                raise RuntimeError(
                    "v7.6 requires the learned categorical RSSM reset-state "
                    "contract")
            if (
                architecture_version == "factorized_dreamer_v7.7"
                and checkpoint.get("control_transition_contract")
                != "acknowledged_physical_control_step_v1"
            ):
                raise RuntimeError(
                    "v7.7 requires the acknowledged physical control-step "
                    "contract")
            if (
                architecture_version == "factorized_dreamer_v7.7"
                and checkpoint.get("rssm_initial_state_contract")
                != "learned_prior_categorical_reset_deterministic_eval_v2"
            ):
                raise RuntimeError(
                    "v7.7 requires learned categorical RSSM resets and "
                    "deterministic evaluation posterior state")
            if (
                architecture_version == "factorized_dreamer_v7.7"
                and not deterministic_evaluation_state
            ):
                raise RuntimeError(
                    "v7.7 checkpoint disabled deterministic evaluation state")
            if (
                architecture_version in (
                    "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                    "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                    "factorized_dreamer_v8.2")
                and checkpoint.get("control_transition_contract")
                != "acknowledged_physical_control_step_v1"
            ):
                raise RuntimeError(
                    f"{architecture_version} requires the acknowledged "
                    "physical control-step "
                    "contract")
            if (
                architecture_version in (
                    "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                    "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                    "factorized_dreamer_v8.2")
                and checkpoint.get("rssm_initial_state_contract")
                != "learned_prior_categorical_reset_deterministic_eval_v2"
            ):
                raise RuntimeError(
                    f"{architecture_version} requires learned categorical "
                    "RSSM resets and "
                    "deterministic evaluation posterior state")
            if architecture_version in (
                "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                "factorized_dreamer_v8.2",
            ) and (
                not deterministic_evaluation_state
                or not event_actor_full_learned
                or event_actor_full_learned_fields != 116
                or actor_input_dim != (
                    7229 if architecture_version in (
                        "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                    else 6572)
                or critic_input_dim != (
                    7229 if architecture_version in (
                        "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                    else 6572)
            ):
                raise RuntimeError(
                    "v7.8 checkpoint lacks aligned per-person learned Human "
                    "decision state or exact Actor/Critic widths")
            if (
                architecture_version in (
                    "factorized_dreamer_v7.9", "factorized_dreamer_v8.0",
                    "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                and not permutation_invariant_entities
            ):
                raise RuntimeError(
                    "v7.9+ requires permutation-invariant masked Human and "
                    "obstacle encoders")
            if architecture_version == "factorized_dreamer_v8.0" and (
                checkpoint.get("action_smoother_contract")
                != "reachable_interval_fraction_hold_zero_slew_bounded_v1"
                or checkpoint.get("action_smoother_parameterization")
                != "reachable_interval_fraction"
            ):
                raise RuntimeError(
                    f"{architecture_version} requires the reachable-gradient "
                    "action contract")
            required_v74 = {
                "goal metric scaling": metric_goal,
                "Ego metric scaling": metric_ego,
                "Human root metric scaling": metric_root,
                "Human quality metric scaling": metric_quality,
                "unified Actor": unified_policy,
                "full-state Critic": critic_full_state,
                "injective direct Ego/task state": direct_ego_task_state,
                "complete exact task state": actor_task_physical_state,
                "direct task geometry": direct_task_geometry,
                "task memory": task_memory_enabled,
                "direct task memory": direct_task_memory,
                "analytic task-memory events": analytic_task_memory_events,
                "explicit joint kinematics": explicit_joint_kinematics,
                "Event full articulated state": event_full_articulated_state,
                "kinematic joints": kinematic_joints,
                "Actor Human lifecycle state": actor_presence_physical,
                "Event Human lifecycle state": event_presence_physical,
                "model-owned ActionAdapter": internal_action_adapter,
                "model-owned action smoother": action_smoother_enabled,
            }
            missing_v74 = [
                name for name, enabled in required_v74.items() if not enabled]
            if missing_v74:
                raise RuntimeError(
                    f"incomplete {architecture_version} checkpoint contract: "
                    + ", ".join(missing_v74))
            if reflection_enabled or human_conditioned_residual or conservative_cv:
                raise RuntimeError(
                    f"{architecture_version} forbids reflection, residual "
                    "branches, and CV vetoes")
            if (
                reflection_slots != (
                    23 if architecture_version in (
                        "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                    else 20)
                or physical_fields_per_slot != 103
                or actor_human_geometry_dim != (
                    (5037 if architecture_version in (
                        "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                     else 4380)
                    if architecture_version in (
                        "factorized_dreamer_v7.8",
                        "factorized_dreamer_v7.9",
                        "factorized_dreamer_v8.0",
                        "factorized_dreamer_v8.1",
                        "factorized_dreamer_v8.2",
                    )
                    else 2188)
                or event_actor_full_state_slots != (
                    23 if architecture_version in (
                        "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                    else 20)
                or event_actor_full_fields != 103
                or geometry_topk_slots != 10
                or actor_task_obstacle_slots != 256
                or actor_task_state_dim != 1808
            ):
                raise RuntimeError(
                    f"{architecture_version} complete Human/task state "
                    "dimensions are invalid")
            required_adapter_fields = {
                "target_altitude_agl_m", "altitude_kp",
                "vertical_velocity_kd", "maximum_vertical_action",
            }
            if (
                not isinstance(action_adapter_contract, dict)
                or set(action_adapter_contract) != required_adapter_fields
                or not all(math.isfinite(float(value)) for value in
                           action_adapter_contract.values())
            ):
                raise RuntimeError(
                    f"{architecture_version} lacks an exact model-owned "
                    "ActionAdapter contract")
            if (
                world_model_planner
                or r2_candidate_planner
                or horizontal_altitude_hold
                or float(online_exploration_scale) > 0.0
            ):
                raise RuntimeError(
                    f"{architecture_version} single-Actor execution forbids "
                    "planners, external "
                    "altitude hold, and Actor-external exploration noise")
            if (
                int(checkpoint.get("actor_update_count", 0)) > 0
                and not isinstance(checkpoint.get("replay_task_contract"), dict)
            ):
                raise RuntimeError(
                    f"trained {architecture_version} checkpoint lacks an "
                    "audited replay/task contract")
        if reflection_enabled and (
            not human_conditioned_residual
            or reflection_slots != geometry_topk_slots
        ):
            raise RuntimeError(
                "checkpoint Human reflection/conditioning/geometry contracts "
                "are inconsistent")
        training_mode = checkpoint.get("training_mode")
        world_model_only_checkpoint = (
            training_mode == "human_prior_world_model"
            or objective_version.endswith("_human_prior_world_model")
        )
        audited_policy_metrics = {
            "selection/score",
            # Online Actor selection additionally measures deterministic
            # imagined task reward and safety cost while retaining every
            # action/saturation audit enforced below.
            "selection/online_actor_score",
        }
        if (
            checkpoint.get("best_metric_name") not in audited_policy_metrics
            and not world_model_only_checkpoint
        ):
            audit_errors.append(
                "checkpoint was not selected by the safe policy metric")
        for name, maximum in required_policy_metrics.items():
            value = self.policy_validation.get(name)
            if value is None:
                audit_errors.append(f"missing {name}")
            elif not math.isfinite(float(value)) or float(value) > maximum:
                audit_errors.append(
                    f"{name}={float(value):.6f} exceeds {maximum:.6f}")
        if audit_errors and not allow_unsafe_policy:
            raise RuntimeError(
                "Refusing unaudited/unsafe policy checkpoint: "
                + "; ".join(audit_errors)
                + ". Retrain with the offline Actor safeguards."
            )
        runtime_model_config = (
            "pure_dreamer"
            if architecture_version in (
                "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                "factorized_dreamer_v8.2")
            else "factorized_dreamer"
        )
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            config = compose(config_name="configs", overrides=[
                "env=isaac_uav_human",
                f"model={runtime_model_config}",
                f"device={self.device}",
                f"model.device={self.device}",
                f"model.rssm.device={self.device}",
            ])
        with open_dict(config.model.factorized):
            config.model.factorized.max_people = reflection_slots
            config.model.factorized.ego_state_mean = np.asarray(
                checkpoint["ego_state_mean"], np.float32).tolist()
            config.model.factorized.ego_state_std = np.asarray(
                checkpoint["ego_state_std"], np.float32).tolist()
            config.model.factorized.human_root_metric_scaling = metric_root
            config.model.factorized.ego_metric_scaling = metric_ego
            config.model.factorized.goal_metric_scaling = metric_goal
            config.model.factorized.human_quality_metric_scaling = metric_quality
            config.model.factorized.v63.action_smoothing.enabled = (
                architecture_version in (
                    "factorized_dreamer_v6.5", "factorized_dreamer_v6.6",
                    "factorized_dreamer_v6.7", "factorized_dreamer_v6.8",
                    "factorized_dreamer_v6.9", "factorized_dreamer_v7.0",
                    "factorized_dreamer_v7.1", "factorized_dreamer_v7.2",
                    "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
                    "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
                    "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                    "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                    "factorized_dreamer_v8.2"))
            config.model.factorized.v63.action_smoothing.parameterization = (
                str(checkpoint.get("action_smoother_parameterization"))
                if architecture_version in (
                    "factorized_dreamer_v8.1", "factorized_dreamer_v8.2")
                else (
                    "reachable_interval_fraction"
                    if architecture_version == "factorized_dreamer_v8.0"
                    else "legacy_absolute_target")
            )
            config.model.factorized.v63.explicit_human_geometry = (
                architecture_version in (
                    "factorized_dreamer_v6.6", "factorized_dreamer_v6.7",
                    "factorized_dreamer_v6.8", "factorized_dreamer_v6.9",
                    "factorized_dreamer_v7.0", "factorized_dreamer_v7.1",
                    "factorized_dreamer_v7.2", "factorized_dreamer_v7.3",
                    "factorized_dreamer_v7.4", "factorized_dreamer_v7.6",
                    "factorized_dreamer_v7.7", "factorized_dreamer_v7.8",
                    "factorized_dreamer_v7.9", "factorized_dreamer_v8.0",
                    "factorized_dreamer_v8.1", "factorized_dreamer_v8.2"))
            config.model.factorized.v63.deterministic_evaluation_state = (
                deterministic_evaluation_state)
            config.model.factorized.v63.actor_full_learned_per_slot = (
                event_actor_full_learned)
            config.model.factorized.v63.explicit_joint_kinematics = (
                explicit_joint_kinematics)
            config.model.factorized.v63.explicit_human_presence_physical = (
                event_presence_physical)
            config.model.factorized.v63.event_full_articulated_state = (
                event_full_articulated_state)
            config.model.factorized.v63.direct_task_geometry = (
                direct_task_geometry)
            config.model.factorized.v63.task_memory = task_memory_enabled
            config.model.factorized.v63.direct_task_memory = (
                direct_task_memory)
            config.model.factorized.v63.analytic_task_memory_events = (
                analytic_task_memory_events)
            config.model.factorized.v63.geometry_topk_physical_slots = (
                geometry_topk_slots)
            config.model.factorized.human_dynamics = {
                "kinematic_velocity_only": (
                    architecture_version in (
                        "factorized_dreamer_v6.8",
                        "factorized_dreamer_v6.9",
                        "factorized_dreamer_v7.0",
                        "factorized_dreamer_v7.1",
                        "factorized_dreamer_v7.2",
                        "factorized_dreamer_v7.3",
                        "factorized_dreamer_v7.4",
                        "factorized_dreamer_v7.6",
                        "factorized_dreamer_v7.7",
                        "factorized_dreamer_v7.8",
                        "factorized_dreamer_v7.9",
                        "factorized_dreamer_v8.0",
                        "factorized_dreamer_v8.1",
                        "factorized_dreamer_v8.2")),
                "max_velocity_residual_mps": root_velocity_residual,
                "max_root_speed_mps": root_max_speed,
                "kinematic_joint_velocity_only": kinematic_joints,
                "max_joint_velocity_residual_mps": joint_velocity_residual,
                "max_joint_speed_mps": joint_max_speed,
                "conservative_cv_clearance": conservative_cv,
            }
            config.model.factorized.v62.unified_policy = unified_policy
            config.model.factorized.v62.critic_full_state = critic_full_state
            config.model.factorized.v62.permutation_invariant_entities = (
                permutation_invariant_entities)
            config.model.factorized.v62.human_reflection_equivariant = (
                reflection_enabled)
            config.model.factorized.v62.human_physical_slots = reflection_slots
            config.model.factorized.v62.human_physical_fields_per_slot = (
                physical_fields_per_slot)
            config.model.factorized.v62.human_physical_presence_state = (
                actor_presence_physical)
            config.model.factorized.v62.human_conditioned_residual = (
                human_conditioned_residual)
            config.model.factorized.v62.authoritative_ego_token = (
                authoritative_ego_token)
            config.model.factorized.v62.direct_ego_task_state = (
                direct_ego_task_state)
            config.model.factorized.v62.task_physical_obstacle_slots = (
                actor_task_obstacle_slots)
            if architecture_version in (
                "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
                "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
                "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                "factorized_dreamer_v8.2",
            ):
                for key, value in action_adapter_contract.items():
                    config.model.factorized.v62.action_adapter[key] = float(
                        value)
        # v7.6 is the first checkpoint whose state dictionary and reset
        # semantics contain a learned categorical RSSM prior. Older models
        # carried ``initial: learned`` in YAML, but their implementation
        # actually emitted zeros; reconstruct that exact historical state.
        with open_dict(config.model.rssm):
            config.model.rssm.initial = (
                "learned"
                if architecture_version in (
                    "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
                    "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                    "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                    "factorized_dreamer_v8.2")
                else "zeros"
            )
        obs_space = SimpleNamespace(spaces={
            "goal": SimpleNamespace(shape=(GOAL_FEATURE_DIM,)),
        })
        act_space = SimpleNamespace(shape=(ACTION_DIM,))
        self.agent = FactorizedDreamerAgent(
            config.model, obs_space, act_space).to(self.device)
        if architecture_version in (
            "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
            "factorized_dreamer_v8.2"
        ) and (
            checkpoint.get("resolved_training_model_config_hash")
            != self.agent.resolved_config_hash
        ):
            raise RuntimeError(
                f"{architecture_version} deployment model config differs "
                "from the exact training YAML/hash: "
                f"checkpoint={checkpoint.get('resolved_training_model_config_hash')} "
                f"runtime={self.agent.resolved_config_hash}")
        if architecture_version in (
            "factorized_dreamer_v6.5", "factorized_dreamer_v6.6",
            "factorized_dreamer_v6.7", "factorized_dreamer_v6.8",
            "factorized_dreamer_v6.9", "factorized_dreamer_v7.0",
            "factorized_dreamer_v7.1", "factorized_dreamer_v7.2",
            "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
            "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
            "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
            "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
            "factorized_dreamer_v8.2",
        ):
            incompatible = self.agent.load_state_dict(
                checkpoint["agent_state_dict"], strict=False)
            allowed_calibration_suffixes = (
                "transition_event.human_calibration_log_temperature",
                "transition_event.human_calibration_bias",
                "transition_event.human_calibration_count_bias",
                "transition_event.human_calibration_shrinkage_weight",
                "transition_event.human_calibration_prior_probability",
                "transition_event.human_calibration_fitted",
            )
            illegal_missing = tuple(
                key for key in incompatible.missing_keys
                if not key.endswith(allowed_calibration_suffixes)
                and not (
                    architecture_version not in (
                        "factorized_dreamer_v6.9", "factorized_dreamer_v7.0",
                        "factorized_dreamer_v7.1", "factorized_dreamer_v7.2",
                        "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
                        "factorized_dreamer_v7.6",
                        "factorized_dreamer_v7.7",
                        "factorized_dreamer_v7.8",
                        "factorized_dreamer_v7.9",
                        "factorized_dreamer_v8.0",
                        "factorized_dreamer_v8.1",
                        "factorized_dreamer_v8.2")
                    and key.endswith(
                        "model.task_geometry_adapter.weight")
                )
                and not (
                    architecture_version not in (
                        "factorized_dreamer_v7.2", "factorized_dreamer_v7.3",
                        "factorized_dreamer_v7.4",
                        "factorized_dreamer_v7.6",
                        "factorized_dreamer_v7.7",
                        "factorized_dreamer_v7.8",
                        "factorized_dreamer_v7.9",
                        "factorized_dreamer_v8.0",
                        "factorized_dreamer_v8.1",
                        "factorized_dreamer_v8.2")
                    and key.endswith("model.task_memory_adapter.weight")
                ))
            if incompatible.unexpected_keys or illegal_missing:
                raise RuntimeError(
                    "factorized checkpoint compatibility load mismatch: "
                    f"missing={tuple(incompatible.missing_keys)}, "
                    f"unexpected={tuple(incompatible.unexpected_keys)}")
            migration = {
                "initialized_target": tuple(incompatible.missing_keys)}
        elif architecture_version == "factorized_dreamer_v6.4":
            self.agent.load_state_dict(
                checkpoint["agent_state_dict"], strict=True)
            migration = {"initialized_target": ()}
        elif architecture_version == "factorized_dreamer_v6.3":
            incompatible = self.agent.load_state_dict(
                checkpoint["agent_state_dict"], strict=False)
            unexpected = tuple(incompatible.unexpected_keys)
            missing = tuple(incompatible.missing_keys)
            allowed_missing = all(
                ".transition_event.non_goal_network." in key
                for key in missing)
            if unexpected or not allowed_missing:
                raise RuntimeError(
                    "v6.3 compatibility load mismatch: "
                    f"missing={missing}, unexpected={unexpected}")
            migration = {"initialized_target": missing}
        elif architecture_version == "factorized_dreamer_v6.2":
            migration = load_compatible_factorized_state(
                self.agent, checkpoint["agent_state_dict"])
        else:
            raise RuntimeError(
                "Refusing to deploy a legacy checkpoint through the v6.3 "
                "3D Actor: run bounded warm-start first; architecture="
                f"{architecture_version!r}")
        self.agent.eval()
        self.model = self.agent.model
        if architecture_version in (
            "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
            "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
            "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
            "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
            "factorized_dreamer_v8.2",
        ):
            if bool(self.model.uses_internal_action_adapter) != (
                internal_action_adapter
            ) or bool(self.model.action_smoother is not None) != (
                action_smoother_enabled
            ):
                raise RuntimeError(
                    "deployed action adapter/smoother differs from checkpoint")
            deployed_adapter = self.model.action_adapter.config
            for key, expected in action_adapter_contract.items():
                if not math.isclose(
                    float(getattr(deployed_adapter, key)), float(expected),
                    rel_tol=0.0, abs_tol=1.0e-9,
                ):
                    raise RuntimeError(
                        f"deployed ActionAdapter field {key} differs from "
                        "checkpoint")
        self.model.task_geometry_enabled = (
            architecture_version in (
                "factorized_dreamer_v6.9", "factorized_dreamer_v7.0",
                "factorized_dreamer_v7.1", "factorized_dreamer_v7.2",
                "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
                "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
                "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                "factorized_dreamer_v8.2"))
        self.model.actor_explicit_human_geometry_enabled = (
            checkpoint_actor_explicit_human_geometry_enabled(
                checkpoint, architecture_version))
        if bool(self.model.actor_authoritative_ego_token_enabled) != (
            authoritative_ego_token
        ):
            raise RuntimeError(
                "deployed Actor authoritative Ego-token contract differs "
                "from checkpoint")
        if bool(self.model.actor_direct_ego_task_state_enabled) != (
            direct_ego_task_state
        ):
            raise RuntimeError(
                "deployed Actor direct Ego/task-state contract differs "
                "from checkpoint")
        if architecture_version in (
            "factorized_dreamer_v7.4", "factorized_dreamer_v7.6",
            "factorized_dreamer_v7.7", "factorized_dreamer_v7.8",
            "factorized_dreamer_v7.9", "factorized_dreamer_v8.0",
            "factorized_dreamer_v8.1", "factorized_dreamer_v8.2",
        ) and (
            bool(self.model.actor_task_physical_state_enabled)
            != actor_task_physical_state
            or int(self.model.actor_task_physical_obstacle_slots)
            != actor_task_obstacle_slots
            or int(self.model.actor.task_geometry_dim)
            != actor_task_state_dim
            or int(self.model.actor.human_geometry_dim)
            != actor_human_geometry_dim
        ):
            raise RuntimeError(
                "deployed Actor complete Human/task-state contract differs "
                "from checkpoint")
        if bool(self.model.actor.config.human_reflection_equivariant) != (
            reflection_enabled
        ) or int(self.model.actor.config.human_physical_slots) != reflection_slots:
            raise RuntimeError(
                "deployed Actor Human reflection contract differs from checkpoint")
        if bool(self.model.actor.config.human_conditioned_residual) != (
            human_conditioned_residual
        ):
            raise RuntimeError(
                "deployed Actor Human-conditioned residual differs from checkpoint")
        if bool(self.model.actor.config.unified_policy) != unified_policy:
            raise RuntimeError(
                "deployed Actor unified-policy contract differs from checkpoint")
        if bool(
            self.model.actor.config.permutation_invariant_entities
        ) != permutation_invariant_entities:
            raise RuntimeError(
                "deployed entity-set encoder contract differs from checkpoint")
        if bool(
            self.model.rssm.latent_policy_attention.goal_metric_scaling
        ) != metric_goal:
            raise RuntimeError(
                "deployed Goal metric-scaling contract differs from checkpoint")
        if bool(self.model.critic_full_state_enabled) != critic_full_state:
            raise RuntimeError(
                "deployed Critic state contract differs from checkpoint")
        if bool(
            self.model.deterministic_evaluation_state_enabled
        ) != deterministic_evaluation_state:
            raise RuntimeError(
                "deployed deterministic evaluation-state contract differs "
                "from checkpoint")
        if bool(
            self.model.transition_event.actor_full_learned_per_slot
        ) != event_actor_full_learned or int(
            self.model.transition_event.actor_full_learned_fields_per_slot
        ) != event_actor_full_learned_fields:
            raise RuntimeError(
                "deployed per-person learned Human-state contract differs "
                "from checkpoint")
        deployed_critic_input_dim = int(getattr(
            self.model.value, "input_dim", -1))
        if deployed_critic_input_dim <= 0:
            critic_input_layer = next((
                module for module in self.model.value.modules()
                if isinstance(module, torch.nn.Linear)
            ), None)
            deployed_critic_input_dim = (
                -1 if critic_input_layer is None
                else int(critic_input_layer.in_features))
        if (
            int(self.model.actor.input_dim) != actor_input_dim
            or deployed_critic_input_dim != critic_input_dim
        ):
            raise RuntimeError(
                "deployed Actor/Critic decision-state widths differ from "
                "checkpoint")
        if bool(self.model.direct_task_geometry_enabled) != direct_task_geometry:
            raise RuntimeError(
                "deployed direct task-geometry contract differs from checkpoint")
        if bool(self.model.task_memory_enabled) != task_memory_enabled \
                or bool(self.model.direct_task_memory_enabled) != (
                    direct_task_memory):
            raise RuntimeError(
                "deployed task-memory contract differs from checkpoint")
        if bool(
            self.model.transition_event.analytic_task_memory_events
        ) != analytic_task_memory_events:
            raise RuntimeError(
                "deployed analytic task-event contract differs from checkpoint")
        if bool(
            self.model.transition_event.explicit_joint_kinematics
        ) != explicit_joint_kinematics:
            raise RuntimeError(
                "deployed directional joint-geometry contract differs from checkpoint")
        if bool(
            self.model.transition_event.event_full_articulated_state
        ) != event_full_articulated_state:
            raise RuntimeError(
                "deployed Event articulated-state contract differs from checkpoint")
        if bool(
            self.model.transition_event.explicit_human_presence_physical
        ) != event_presence_physical or bool(
            self.model.actor.config.human_physical_presence_state
        ) != actor_presence_physical:
            raise RuntimeError(
                "deployed Human lifecycle-state contract differs from checkpoint")
        if bool(
            self.model.prediction_heads.config.kinematic_joint_velocity_only
        ) != kinematic_joints:
            raise RuntimeError(
                "deployed Human joint dynamics differs from checkpoint")
        if bool(self.model.conservative_human_clearance_enabled) != conservative_cv:
            raise RuntimeError(
                "deployed learned/CV return contract differs from checkpoint")
        if int(
            self.model.transition_event.geometry_topk_physical_slots
        ) != geometry_topk_slots:
            raise RuntimeError(
                "deployed Human geometry layout differs from checkpoint")
        self.max_people = self.agent.max_people
        self.pose_history = int(config.model.factorized.pose_history)
        self.amp_enabled = self.device.type == "cuda"
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = int(checkpoint.get("step", -1))
        self.world_update_count = int(checkpoint.get(
            "world_update_count", self.checkpoint_step))
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.environment_prefill_frames = int(checkpoint.get(
            "environment_prefill_frames", 0))
        checkpoint_phase = str(checkpoint.get("training_phase", ""))
        actor_update_count = int(checkpoint.get("actor_update_count", 0))
        # ``--allow-unsafe-policy`` is the explicit isolation-canary escape
        # hatch. Once an Actor exists, do not silently replace it with smooth
        # random prefill merely because a downstream review is still open.
        # The checkpoint phase remains unchanged, and formal deployment still
        # rejects the unreviewed policy above unless this opt-in is supplied.
        self.unsafe_canary_actor_enabled = bool(
            allow_unsafe_policy
            and actor_update_count > 0
            and checkpoint_phase in (
                "event_review", "critic_review", "actor_candidate"))
        self.prefill_active = (
            self.environment_steps < self.environment_prefill_frames
            or self.world_update_count < int(checkpoint.get(
                "world_only_updates", 0))
            or (
                checkpoint_phase in (
                    "world_model_only", "human_adaptation", "human_review",
                    "event_repair", "event_review",
                    "critic_warmup", "critic_review",
                    "actor_candidate",
                )
                and not self.unsafe_canary_actor_enabled
            )
            or actor_update_count <= 0)
        self.prefill_explorer = SmoothRandomPrefillExplorer(
            seed=max(0, self.environment_steps + self.checkpoint_step))
        self.best_validation = float(checkpoint.get("best_validation", math.nan))
        self.human_empty_hold_s = float(human_empty_hold_s)
        self.slots = OnlineHumanSlots(
            self.max_people, empty_hold_s=self.human_empty_hold_s)
        pure_state = checkpoint.get("pure_dreamer_training_state") or {}
        pure_config = pure_state.get("config") or {}
        self.task_memory_tracker = TaskMemoryTracker(
            progress_epsilon_m=float(pure_config.get(
                "watchdog_progress_epsilon_m", 0.25)),
            stuck_max_horizontal_speed_mps=float(pure_config.get(
                "watchdog_stuck_max_horizontal_speed_mps", 0.15)),
            acceleration_filter_alpha=float(pure_config.get(
                "acceleration_filter_alpha", 0.25)),
        )
        self.escape_direction = EscapeDirectionLatch()
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=self.pose_history)
        self.agent_state = None
        self.first = True
        self.lock = threading.Lock()
        self.world_model_planner = bool(world_model_planner)
        self.r2_candidate_planner = bool(r2_candidate_planner)
        pure_r2_state = checkpoint.get("pure_r2dreamer_training_state") or {}
        saved_event_clearance = pure_r2_state.get(
            "event_collision_clearance_m")
        if saved_event_clearance is None:
            saved_boundaries = pure_r2_state.get(
                "event_geometry_boundaries_m") or {}
            saved_event_clearance = max(
                (float(value[0]) for value in saved_boundaries.values()),
                default=LEGACY_ISAAC_JOINT_CONTACT_OFFSET_M,
            )
        self.r2_planner_event_collision_clearance_m = float(
            saved_event_clearance)
        if (
            not math.isfinite(
                self.r2_planner_event_collision_clearance_m)
            or self.r2_planner_event_collision_clearance_m < 0.0
        ):
            raise ValueError(
                "checkpoint Event collision-clearance contract is invalid")
        self.r2_planner_behavior_support = pure_r2_state.get(
            "event_behavior_support")
        release_horizons = tuple(int(value) for value in checkpoint.get(
            "event_release_horizons",
            pure_r2_state.get("event_release_horizons", (10, 15))))
        configured_r2_horizon = max(release_horizons or (15,))
        self.r2_planner_horizon_steps = int(
            configured_r2_horizon
            if r2_planner_horizon_steps is None
            else r2_planner_horizon_steps)
        self.r2_planner_interval_steps = int(r2_planner_interval_steps)
        self.r2_planner_forward_delta = float(pure_r2_state.get(
            "actor_directional_forward_delta", 0.25))
        self.r2_planner_lateral_delta = float(pure_r2_state.get(
            "actor_directional_lateral_delta", 0.35))
        # Runtime selection is intentionally stricter than older Actor
        # checkpoints, which commonly stored 0.02.  Keep this independently
        # configurable so loading an old checkpoint cannot silently restore
        # the broad progress tie band that masked meaningful Event gaps.
        self.r2_planner_risk_tolerance = float(
            r2_planner_risk_tolerance)
        self.r2_planner_safe_risk_cap = float(
            r2_planner_safe_risk_cap)
        self.r2_planner_safe_risk_margin = float(
            r2_planner_safe_risk_margin)
        self.r2_planner_minimum_risk_improvement = float(pure_r2_state.get(
            "actor_directional_min_risk_improvement", 0.01))
        self.r2_planner_minimum_progress_improvement = float(
            pure_r2_state.get(
                "actor_directional_min_progress_improvement", 0.005))
        self.r2_planner_boundary_buffer_m = float(
            r2_planner_boundary_buffer_m)
        self.r2_planner_commitment_steps = int(
            r2_planner_commitment_steps)
        self.r2_planner_switch_risk_improvement = float(
            r2_planner_switch_risk_improvement)
        self.r2_planner_switch_progress_improvement = float(
            r2_planner_switch_progress_improvement)
        self.r2_commitment = R2CandidateCommitment(
            hold_steps=self.r2_planner_commitment_steps,
            switch_risk_improvement=(
                self.r2_planner_switch_risk_improvement),
            safe_risk_cap=self.r2_planner_safe_risk_cap,
            safe_risk_margin=self.r2_planner_safe_risk_margin,
            switch_progress_improvement=(
                self.r2_planner_switch_progress_improvement),
        )
        self.r2_planner_rollout_diagnostics = bool(
            r2_planner_rollout_diagnostics)
        self.r2_planner_support_snapshot_version = 0
        self.r2_planner_support_snapshot_sha256 = ""
        if self.r2_candidate_planner:
            if architecture_version != "factorized_dreamer_v6.8":
                raise ValueError(
                    "pure R2 candidate planning requires a v6.8 checkpoint")
            if int(checkpoint.get("actor_update_count", 0)) <= 0:
                raise ValueError(
                    "pure R2 candidate planning requires a trained Actor")
            if not self.model.uses_internal_action_adapter:
                raise ValueError(
                    "pure R2 candidate planning requires ActionAdapter")
            if self.r2_planner_behavior_support is None:
                raise ValueError(
                    "checkpoint has no frozen Event behavior support")
            support = self.r2_planner_behavior_support
            self.r2_planner_support_snapshot_version = int(
                support.get("snapshot_version", 0))
            self.r2_planner_support_snapshot_sha256 = str(
                support.get("snapshot_sha256", ""))
            if (
                self.r2_planner_support_snapshot_version <= 0
                or len(self.r2_planner_support_snapshot_sha256) != 64
            ):
                raise ValueError(
                    "checkpoint Event behavior support is not bound to a "
                    "committed snapshot")
            self.r2_planner_behavior_support = {
                key: (
                    value.to(self.device)
                    if torch.is_tensor(value) else value
                )
                for key, value in support.items()
            }
        self.shadow_imagination_enabled = bool(shadow_imagination)
        self.shadow_horizon_steps = int(shadow_horizon_steps)
        self.shadow_interval_steps = int(shadow_interval_steps)
        self.shadow_step_duration_s = float(shadow_step_duration_s)
        self.shadow_risk_horizon_s = float(
            config.model.factorized.offline_actor.safety.risk_horizon_s)
        self.shadow_geometry_warning_buffer_m = float(
            (checkpoint.get("pure_r2dreamer_training_state") or {}).get(
                "conservative_geometry_warning_buffer_m", 0.1))
        self.online_exploration_scale = float(online_exploration_scale)
        self.exploration_delta_limits = torch.as_tensor(
            exploration_delta_limits, dtype=torch.float32,
            device=self.device).reshape(-1)
        if self.model.uses_internal_action_adapter:
            if self.exploration_delta_limits.shape == (ACTION_DIM,):
                self.exploration_delta_limits = self.exploration_delta_limits[
                    torch.tensor((0, 1, 3), device=self.device)]
        self.exploration_disable_clearance_m = float(
            exploration_disable_clearance_m)
        self.exploration_full_clearance_m = float(
            exploration_full_clearance_m)
        self.stochastic_actor = bool(stochastic_actor)
        self.horizontal_altitude_hold = bool(horizontal_altitude_hold)
        self.altitude_hold_target_agl_m = float(altitude_hold_target_agl_m)
        self.altitude_hold_kp = float(altitude_hold_kp)
        self.altitude_hold_max_action = float(altitude_hold_max_action)
        if self.shadow_horizon_steps <= 0:
            raise ValueError("shadow_horizon_steps must be positive")
        if self.shadow_interval_steps <= 0:
            raise ValueError("shadow_interval_steps must be positive")
        if self.shadow_step_duration_s <= 0.0:
            raise ValueError("shadow_step_duration_s must be positive")
        if self.world_model_planner and self.shadow_imagination_enabled:
            raise ValueError(
                "shadow imagination must evaluate the raw Actor; disable "
                "world_model_planner")
        if self.world_model_planner and self.model.uses_internal_action_adapter:
            raise ValueError(
                "world_model_planner uses the disabled legacy action_risk "
                "contract and is unavailable for pure R2-Dreamer checkpoints")
        if self.world_model_planner and self.r2_candidate_planner:
            raise ValueError(
                "legacy and pure R2 candidate planners are mutually exclusive")
        if self.r2_planner_horizon_steps <= 0:
            raise ValueError("r2_planner_horizon_steps must be positive")
        if self.r2_planner_interval_steps <= 0:
            raise ValueError("r2_planner_interval_steps must be positive")
        if self.r2_planner_commitment_steps <= 0:
            raise ValueError("r2_planner_commitment_steps must be positive")
        if (
            not math.isfinite(self.r2_planner_risk_tolerance)
            or self.r2_planner_risk_tolerance < 0.0
        ):
            raise ValueError(
                "r2_planner_risk_tolerance must be finite and non-negative")
        if not 0.0 < self.r2_planner_safe_risk_cap < 1.0:
            raise ValueError(
                "r2_planner_safe_risk_cap must be within (0,1)")
        if (
            not math.isfinite(self.r2_planner_safe_risk_margin)
            or self.r2_planner_safe_risk_margin < 0.0
        ):
            raise ValueError(
                "r2_planner_safe_risk_margin must be finite and non-negative")
        if (
            not math.isfinite(
                self.r2_planner_switch_progress_improvement)
            or self.r2_planner_switch_progress_improvement < 0.0
        ):
            raise ValueError(
                "r2 planner progress switch margin must be non-negative")
        if self.r2_planner_boundary_buffer_m <= 0.0:
            raise ValueError("r2_planner_boundary_buffer_m must be positive")
        if self.human_empty_hold_s <= 0.0:
            raise ValueError("human_empty_hold_s must be positive")
        if self.online_exploration_scale < 0.0:
            raise ValueError("online_exploration_scale must be non-negative")
        expected_exploration_dim = (
            self.agent.policy_act_dim
            if self.model.uses_internal_action_adapter else ACTION_DIM)
        if self.exploration_delta_limits.shape != (expected_exploration_dim,):
            raise ValueError(
                "exploration_delta_limits does not match policy action size")
        if bool((self.exploration_delta_limits < 0.0).any()):
            raise ValueError("exploration delta limits must be non-negative")
        if not (
            0.0 <= self.exploration_disable_clearance_m
            < self.exploration_full_clearance_m
        ):
            raise ValueError("invalid exploration clearance interval")
        if self.altitude_hold_target_agl_m <= 0.0:
            raise ValueError("altitude hold target must be positive")
        if self.altitude_hold_kp < 0.0 or self.altitude_hold_max_action <= 0.0:
            raise ValueError("invalid altitude hold gain/action limit")
        self.reset()

    def _commit_r2_planner_selection(
        self, planner: dict[str, object],
    ) -> dict[str, object]:
        """Apply temporal Event hysteresis to one batch-one plan result."""
        risk_tensor = planner["candidate_risk"]
        support_tensor = planner["candidate_supported"]
        selected_tensor = planner["selected_index"]
        if not (
            torch.is_tensor(risk_tensor)
            and torch.is_tensor(support_tensor)
            and torch.is_tensor(selected_tensor)
            and risk_tensor.shape[0] == 1
        ):
            raise RuntimeError("live candidate commitment requires batch one")
        raw_selected = int(selected_tensor[0].item())
        selected, commitment = self.r2_commitment.select(
            raw_selected,
            risk_tensor[0].detach().float().cpu().numpy(),
            planner["candidate_progress"][0].detach().float().cpu().numpy(),
            support_tensor[0].detach().bool().cpu().numpy(),
        )
        planner["raw_selected_index"] = selected_tensor.detach().clone()
        if selected != raw_selected:
            index = torch.as_tensor(
                [selected], dtype=selected_tensor.dtype,
                device=selected_tensor.device)
            row = torch.zeros(1, dtype=torch.long, device=index.device)
            planner["selected_index"] = index
            planner["policy_action"] = planner[
                "candidate_policy_action"][row, index]
            planner["smoothed_policy_action"] = planner[
                "candidate_smoothed_policy_action"][row, index]
            planner["action"] = planner["candidate_action"][row, index]
            available = planner["available"].bool()
            planner["intervened"] = available & index.ne(0)
            planner["risk_improvement"] = (
                planner["candidate_risk"][:, 0]
                - planner["candidate_risk"][row, index]
            )
            planner["progress_improvement"] = (
                planner["candidate_progress"][row, index]
                - planner["candidate_progress"][:, 0]
            )
        planner["commitment"] = commitment
        return planner

    @staticmethod
    def _cached_r2_policy_target(
        actor_policy: torch.Tensor,
        cached_target: torch.Tensor,
        selected_index: int,
    ) -> torch.Tensor:
        """Preserve absolute structured intent between 5 Hz replans."""
        if actor_policy.shape != cached_target.shape:
            raise ValueError("cached R2 policy target shape mismatch")
        return (
            actor_policy if int(selected_index) == 0 else cached_target
        ).clamp(-1.0, 1.0)

    @staticmethod
    def _r2_rollout_diagnostics_payload(
        planner: dict[str, object],
        human_mask: np.ndarray,
        human_ids: np.ndarray,
        horizon_steps: int,
    ) -> dict[str, object]:
        """Serialize compact h5/h10/h15 candidate rollout diagnostics.

        This payload is observational only.  It samples the already-computed
        candidate rollout and never feeds a value back into candidate ranking
        or the executed action.  Padded Human slots are omitted to keep the
        live JSON response bounded.
        """
        horizon_steps = int(horizon_steps)
        diagnostic_steps = tuple(sorted({
            step for step in (5, 10, 15, horizon_steps)
            if 0 < step <= horizon_steps
        }))
        if not diagnostic_steps:
            raise ValueError("R2 rollout diagnostics require a positive horizon")
        active_slots = np.flatnonzero(
            np.asarray(human_mask, dtype=np.bool_).reshape(-1))
        model_ids = np.asarray(human_ids, dtype=np.int64).reshape(-1)
        if model_ids.shape != np.asarray(human_mask).reshape(-1).shape:
            raise ValueError("R2 diagnostic Human ID/mask shape mismatch")

        def tensor(name: str) -> torch.Tensor:
            value = planner.get(name)
            if not torch.is_tensor(value) or value.shape[0] != 1:
                raise RuntimeError(
                    f"R2 rollout diagnostic tensor {name!r} is unavailable")
            return value[0].detach().float().cpu()

        event_step = tensor("candidate_event_step_probability")
        ego = tensor("candidate_predicted_ego_state")
        root = tensor("candidate_predicted_human_root")
        clearance = tensor(
            "candidate_predicted_human_joint_clearance")
        origin_root = tensor("diagnostic_origin_human_root")
        origin_clearance = tensor(
            "diagnostic_origin_human_joint_clearance")
        if (
            event_step.ndim != 2
            or event_step.shape[1] != horizon_steps
            or ego.shape[:2] != event_step.shape
            or root.shape[:2] != event_step.shape
            or clearance.shape[:2] != event_step.shape
        ):
            raise RuntimeError("R2 rollout diagnostic tensor shapes disagree")

        step_indices = torch.as_tensor(
            [step - 1 for step in diagnostic_steps], dtype=torch.long)
        active_index = torch.as_tensor(active_slots, dtype=torch.long)
        cumulative_risk = 1.0 - torch.cumprod(
            1.0 - event_step.clamp(0.0, 1.0), dim=1)
        sampled_root = root.index_select(1, step_indices)
        sampled_clearance = clearance.index_select(1, step_indices)
        if active_slots.size:
            sampled_root = sampled_root.index_select(2, active_index)
            sampled_clearance = sampled_clearance.index_select(
                2, active_index)
            minimum_clearance = sampled_clearance.amin(dim=2)
        else:
            sampled_root = sampled_root[:, :, :0]
            sampled_clearance = sampled_clearance[:, :, :0]
            minimum_clearance = torch.full(
                (event_step.shape[0], len(diagnostic_steps)),
                float("inf"), dtype=torch.float32)
        return {
            "schema_version": 1,
            "model_step_duration_s": 0.1,
            "sampled_future_steps": list(diagnostic_steps),
            "ego_state_order": [
                "local_x", "local_y", "local_z", "body_vx", "body_vy",
                "world_vz", "body_ax", "body_ay", "world_az",
                "altitude_agl", "roll", "pitch", "sin_relative_yaw",
                "cos_relative_yaw",
            ],
            "human_root_order": [
                "body_x", "body_y", "body_z", "body_vx", "body_vy",
                "body_vz", "extent_x", "extent_y", "extent_z",
                "confidence",
            ],
            "active_human_slot_indices": active_slots.astype(int).tolist(),
            # Training/model IDs reserve zero for padding, whereas the live
            # tracker IDs in the request are zero based.
            "active_human_model_ids": model_ids[active_slots].astype(int).tolist(),
            "active_human_track_ids": (
                model_ids[active_slots] - 1).astype(int).tolist(),
            "origin_human_root": (
                origin_root.index_select(0, active_index).tolist()
                if active_slots.size else []),
            "origin_human_joint_clearance_m": (
                origin_clearance.index_select(0, active_index).tolist()
                if active_slots.size else []),
            "candidate_event_step_probability": event_step.tolist(),
            "candidate_event_cumulative_risk_at_steps": (
                cumulative_risk.index_select(1, step_indices).tolist()),
            "candidate_predicted_ego_state_at_steps": (
                ego.index_select(1, step_indices).tolist()),
            "candidate_predicted_human_root_at_steps": sampled_root.tolist(),
            "candidate_predicted_human_joint_clearance_m_at_steps": (
                sampled_clearance.tolist()),
            "candidate_predicted_minimum_human_joint_clearance_m_at_steps": (
                minimum_clearance.tolist()),
        }

    def reset(
        self, *, collection_scene_seed: int | None = None,
    ) -> dict[str, object]:
        self.slots.reset()
        self.task_memory_tracker.reset()
        self.escape_direction.reset()
        self.history.clear()
        self.agent_state = self.model.initial_agent_state(
            1, self.max_people, ACTION_DIM)
        self.first = True
        self.shadow_step_count = 0
        self.r2_planner_step_count = 0
        self.r2_cached_policy_target = None
        self.r2_cached_planner = None
        self.r2_commitment.reset()
        self.barrier_projection_count = 0
        self.barrier_infeasible_count = 0
        self.barrier_active_count = 0
        self.prefill_explorer.reset(
            collection_scene_seed=collection_scene_seed)
        return {
            "ok": True,
            "protocol": PROTOCOL,
            "checkpoint_step": self.checkpoint_step,
            "collector_policy_step": (
                0 if self.prefill_active else self.checkpoint_step),
            "collection_policy_contract": (
                self.prefill_explorer.policy_contract()
                if self.prefill_active else {
                    "schema": "omninxt.collection-policy.v1",
                    "mode": (
                        "smoothed_stochastic_actor"
                        if self.stochastic_actor else "actor_mode"),
                    "actor_action_used": True,
                    "checkpoint_state_used_for_action": True,
                }
            ),
            "collection_policy_episode": (
                self.prefill_explorer.episode_evidence()
                if self.prefill_active else {}
            ),
            "collection_policy_source": {
                "checkpoint_path": str(self.checkpoint_path),
                "loaded_checkpoint_step": int(self.checkpoint_step),
                "evaluated_checkpoint_independent": False,
            },
        }

    @staticmethod
    def _array(request: dict, key: str, dtype, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(request.get(key), dtype=dtype)
        if value.shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN/Inf")
        return value

    @staticmethod
    def _require_valid_ego_altitude(
        request: dict, ego_state: np.ndarray,
    ) -> None:
        valid = request.get("ego_altitude_valid")
        if not isinstance(valid, bool) or not valid:
            raise ValueError(
                "ego_altitude_valid must be the boolean true; missing AGL "
                "cannot be imputed as a physical zero")
        altitude = float(ego_state[9])
        if not math.isfinite(altitude) or altitude < 0.0:
            raise ValueError("ego_state altitude_agl must be finite/non-negative")

    @staticmethod
    def _human_model_observation(human: dict[str, object]) -> dict[str, object]:
        """Preserve the complete articulated Human state for ``act_step``.

        ``human_joints`` is the root-relative encoder input, whereas
        ``skeleton`` contains the absolute body-frame joint positions and
        causal per-joint velocities used by the deployable geometry/Event and
        full-state Actor paths.  They are not interchangeable.
        """
        return {
            "skeleton": human["skeleton"],
            "human_root": human["human_root"],
            "human_joints": human["human_joints"],
            "human_mask": human["human_mask"],
            "joint_mask": human["joint_mask"],
            "human_is_first": human["human_is_first"],
            "human_observation_quality": human[
                "human_observation_quality"],
        }

    def step(self, request: dict) -> dict[str, object]:
        started = time.perf_counter()
        ego_state = self._array(request, "ego_state", np.float32, (14,))
        self._require_valid_ego_altitude(request, ego_state)
        goal_position = self._array(
            request, "goal_position", np.float32, (3,))
        raw_ids = np.asarray(request.get("human_track_id", ()), np.int64).reshape(-1)
        count = raw_ids.size
        if count:
            xyz = self._array(
                request, "human_xyz", np.float32, (count, JOINT_COUNT, 3))
            confidence = self._array(
                request, "human_confidence", np.float32, (count, JOINT_COUNT))
            joint_valid = self._array(
                request, "human_joint_valid", np.bool_, (count, JOINT_COUNT))
            joint_measured = self._array(
                request, "human_joint_measured", np.bool_,
                (count, JOINT_COUNT))
            joint_predicted = self._array(
                request, "human_joint_predicted", np.bool_,
                (count, JOINT_COUNT))
            root_velocity = self._array(
                request, "human_root_velocity", np.float32, (count, 3))
            velocity_valid = self._array(
                request, "human_velocity_valid", np.bool_, (count,))
            velocity_sigma_mps = self._array(
                request, "human_velocity_sigma_mps", np.float32, (count,))
            measurement_age_s = self._array(
                request, "human_measurement_age_s", np.float32, (count,))
            track_age_frames = self._array(
                request, "human_track_age_frames", np.float32, (count,))
            prediction_run_frames = self._array(
                request, "human_prediction_run_frames", np.float32, (count,))
            identity_confidence = self._array(
                request, "human_identity_confidence", np.float32, (count,))
        else:
            # JSON has no shape metadata for an empty nested array.
            xyz = np.zeros((0, JOINT_COUNT, 3), np.float32)
            confidence = np.zeros((0, JOINT_COUNT), np.float32)
            joint_valid = np.zeros((0, JOINT_COUNT), np.bool_)
            joint_measured = np.zeros((0, JOINT_COUNT), np.bool_)
            joint_predicted = np.zeros((0, JOINT_COUNT), np.bool_)
            root_velocity = np.zeros((0, 3), np.float32)
            velocity_valid = np.zeros((0,), np.bool_)
            velocity_sigma_mps = np.zeros((0,), np.float32)
            measurement_age_s = np.zeros((0,), np.float32)
            track_age_frames = np.zeros((0,), np.float32)
            prediction_run_frames = np.zeros((0,), np.float32)
            identity_confidence = np.zeros((0,), np.float32)
        timestamp_s = float(request["timestamp_s"])
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        human = self.slots.update(
            ego_state, timestamp_s, raw_ids, xyz, confidence, joint_valid,
            joint_measured, joint_predicted,
            root_velocity, velocity_valid, velocity_sigma_mps,
            measurement_age_s, track_age_frames, prediction_run_frames,
            identity_confidence,
            goal_position=goal_position)
        previous_smoothed_policy = self.agent_state.get(
            "smoothed_policy_action")
        if previous_smoothed_policy is None:
            previous_smoothed_policy_np = np.zeros(3, np.float32)
        else:
            previous_smoothed_policy_np = (
                previous_smoothed_policy.detach().float().cpu().numpy()
                .reshape(3))
        task_memory = self.task_memory_tracker.update(
            timestamp_s, ego_state, goal_position,
            previous_smoothed_policy_np)
        observed_clearance = float(human["observed_human_clearance_m"])
        kinematic_clearance = float(human["kinematic_human_clearance_m"])
        planner_clearance = (
            observed_clearance
            if self.r2_candidate_planner
            else min(observed_clearance, kinematic_clearance)
        )
        if not math.isfinite(planner_clearance):
            clearance_exploration_factor = 1.0
        else:
            clearance_exploration_factor = float(np.clip(
                (
                    planner_clearance
                    - self.exploration_disable_clearance_m
                ) / (
                    self.exploration_full_clearance_m
                    - self.exploration_disable_clearance_m
                ),
                0.0,
                1.0,
            ))
        effective_exploration_scale = (
            self.online_exploration_scale * clearance_exploration_factor)
        emergency_zone = (
            planner_clearance
            <= self.model.planner_observed_emergency_clearance_m
        )
        observed_away, escape_direction_locked = self.escape_direction.update(
            timestamp_s=timestamp_s,
            ego_state=ego_state,
            observed_clearance_m=observed_clearance,
            observed_away_body_xy=human["observed_human_away_xy"],
            emergency_zone=emergency_zone,
        )
        task_geometry = None
        if self.model.task_geometry_enabled:
            required_task_geometry = (
                "boundary_position_world_xy",
                "boundary_position_world_z",
                "boundary_yaw_rad",
                "boundary_flight_bounds_xy",
                "maximum_altitude_m",
                "static_obstacle_aabbs_world",
                "static_obstacle_geometry_use",
            )
            missing_task_geometry = tuple(
                key for key in required_task_geometry if key not in request)
            if missing_task_geometry:
                raise ValueError(
                    "v6.9 pure Actor requires task geometry fields: "
                    f"{missing_task_geometry}")
            static_aabbs = np.asarray(
                request["static_obstacle_aabbs_world"],
                np.float32,
            ).reshape(-1, 4)
            if static_aabbs.shape[0] > 256 or not np.isfinite(
                static_aabbs).all():
                raise ValueError("invalid static_obstacle_aabbs_world")
            task_yaw = float(request["boundary_yaw_rad"])
            if not math.isfinite(task_yaw):
                raise ValueError("boundary_yaw_rad must be finite")
            static_geometry_use = str(
                request["static_obstacle_geometry_use"])
            if static_geometry_use not in (
                "counterfactual_censor_and_actor_proximity_cost",
                "analytic_task_terminal_and_actor_proximity_cost",
            ):
                raise ValueError("invalid static_obstacle_geometry_use")
            task_geometry = task_geometry_proximity_numpy(
                self._array(
                    request, "boundary_position_world_xy", np.float32, (2,)),
                task_yaw,
                self._array(
                    request, "boundary_flight_bounds_xy", np.float32, (4,)),
                static_aabbs,
            )
            task_physical_state = None
            if self.model.actor_task_physical_state_enabled:
                task_physical_state = task_physical_state_numpy(
                    np.concatenate((
                        self._array(
                            request, "boundary_position_world_xy",
                            np.float32, (2,)),
                        np.asarray([
                            float(request["boundary_position_world_z"])
                        ], np.float32),
                    )),
                    task_yaw,
                    self._array(
                        request, "boundary_flight_bounds_xy", np.float32,
                        (4,)),
                    static_aabbs,
                    maximum_altitude_m=float(
                        request["maximum_altitude_m"]),
                    bounds_valid=True,
                    maximum_altitude_valid=True,
                    static_terminal_valid=(
                        static_geometry_use
                        == "analytic_task_terminal_and_actor_proximity_cost"),
                    maximum_obstacles=int(
                        self.model.actor_task_physical_obstacle_slots),
                )
        else:
            task_physical_state = None
        observation = {
            "ego_state": ego_state,
            "goal": goal_features_numpy(ego_state, goal_position),
            "goal_position": goal_position,
            **self._human_model_observation(human),
            "observed_human_clearance_m": np.asarray(
                planner_clearance, np.float32),
            "current_observed_human_clearance_m": np.asarray(
                observed_clearance, np.float32),
            "observed_human_away_xy": observed_away,
            "observed_nearest_human_xy": human[
                "observed_nearest_human_xy"],
            "is_first": np.asarray([self.first], np.bool_),
        }
        if task_geometry is not None:
            observation["task_geometry"] = task_geometry
        if task_physical_state is not None:
            observation["task_physical_state"] = task_physical_state
        if self.model.task_memory_enabled:
            observation["task_memory"] = task_memory
        self.history.append(observation)
        stacked = {
            key: torch.as_tensor(
                np.stack([item[key] for item in self.history], axis=0),
                device=self.device,
            ).unsqueeze(0)
            for key in observation
        }
        # ``human_joints`` is root-relative model input.  Shadow's analytic
        # clearance needs the measured joints in the UAV body frame instead.
        current_human_joints_body = torch.as_tensor(
            human["skeleton"][..., :3],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        boundary_keys = (
            "boundary_position_world_xy",
            "boundary_yaw_rad",
            "boundary_flight_bounds_xy",
        )
        boundary_fields_present = tuple(
            key in request for key in boundary_keys)
        if any(boundary_fields_present) and not all(boundary_fields_present):
            raise ValueError(
                "R2 boundary scoring requires position, yaw, and bounds "
                "together")
        if all(boundary_fields_present):
            boundary_position = torch.as_tensor(
                self._array(
                    request, "boundary_position_world_xy", np.float32, (2,)),
                device=self.device,
            ).unsqueeze(0)
            boundary_yaw = torch.as_tensor(
                [float(request["boundary_yaw_rad"])],
                dtype=torch.float32,
                device=self.device,
            )
            boundary_bounds = torch.as_tensor(
                self._array(
                    request, "boundary_flight_bounds_xy", np.float32, (4,)),
                device=self.device,
            ).unsqueeze(0)
            if not torch.isfinite(boundary_yaw).all():
                raise ValueError("boundary_yaw_rad must be finite")
        else:
            boundary_position = None
            boundary_yaw = None
            boundary_bounds = None
        shadow_tensor = None
        shadow_started = None
        r2_planner_tensor = None
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.amp_enabled,
            cache_enabled=False,
        ):
            previous_applied_action = self.agent_state[
                "prev_action"].detach().clone()
            if self.prefill_active:
                # Prefill is a behavior policy, not an Actor override. Do not
                # execute RSSM/Actor and then discard their command: that made
                # collection availability and latency depend on a checkpoint
                # which the validation protocol claims cannot affect data.
                prefill_policy_np = self.prefill_explorer.step(timestamp_s)
                prefill_policy = torch.as_tensor(
                    prefill_policy_np, device=self.device,
                    dtype=previous_applied_action.dtype).reshape(1, -1)
                current_ego = stacked["ego_state"][:, -1].float()
                action = self.model.applied_from_policy_action(
                    prefill_policy, current_ego)
                model_applied_action = action.detach().clone()
                self.agent_state["policy_action"] = prefill_policy
                self.agent_state["smoothed_policy_action"] = prefill_policy
                self.agent_state["prev_action"] = action
            else:
                action, self.agent_state = self.model.act_step(
                    stacked,
                    self.agent_state,
                    evaluation=(
                        not self.stochastic_actor
                        and self.online_exploration_scale <= 0.0),
                    use_planner=self.world_model_planner,
                    exploration_scale=effective_exploration_scale,
                    exploration_delta_limits=self.exploration_delta_limits,
                    sample_actor_distribution=self.stochastic_actor,
                )
                model_applied_action = action.detach().clone()
            if not self.prefill_active and self.r2_candidate_planner:
                assert self.r2_planner_behavior_support is not None
                actor_action = action.detach().clone()
                actor_policy = self.agent_state[
                    "policy_action"].float().detach().clone()
                actor_smoothed = self.agent_state[
                    "smoothed_policy_action"].float().detach().clone()
                planner_evaluated = (
                    self.r2_cached_planner is None
                    or self.r2_planner_step_count
                    % self.r2_planner_interval_steps == 0
                )
                if planner_evaluated:
                    r2_planner_tensor = self.model.plan_pure_r2_candidates(
                        self.agent_state["latent"],
                        stacked["human_mask"][:, -1].bool(),
                        actor_policy,
                        previous_applied_action.float(),
                        stacked["ego_state"][:, -1].float(),
                        stacked["goal_position"][:, -1].float(),
                        stacked["human_root"][:, -1].float(),
                        stacked["human_observation_quality"][:, -1].float(),
                        stacked["joint_mask"][:, -1].bool(),
                        current_human_joints_body,
                        self.r2_planner_behavior_support,
                        horizon_steps=self.r2_planner_horizon_steps,
                        forward_delta=self.r2_planner_forward_delta,
                        lateral_delta=self.r2_planner_lateral_delta,
                        risk_tolerance=self.r2_planner_risk_tolerance,
                        safe_risk_cap=self.r2_planner_safe_risk_cap,
                        safe_risk_margin=self.r2_planner_safe_risk_margin,
                        minimum_risk_improvement=(
                            self.r2_planner_minimum_risk_improvement),
                        minimum_progress_improvement=(
                            self.r2_planner_minimum_progress_improvement),
                        commitment_steps=(
                            self.r2_planner_commitment_steps),
                        boundary_position_world_xy=boundary_position,
                        boundary_yaw_rad=boundary_yaw,
                        boundary_flight_bounds_xy=boundary_bounds,
                        boundary_buffer_m=self.r2_planner_boundary_buffer_m,
                        return_rollout_diagnostics=(
                            self.r2_planner_rollout_diagnostics),
                    )
                    r2_planner_tensor = self._commit_r2_planner_selection(
                        r2_planner_tensor)
                    self.r2_cached_policy_target = r2_planner_tensor[
                        "policy_action"].detach().clone()
                    r2_planner_tensor["planner_evaluated"] = torch.ones(
                        actor_policy.shape[0], dtype=torch.bool,
                        device=self.device)
                    r2_planner_tensor["cache_age_steps"] = torch.zeros(
                        actor_policy.shape[0], dtype=torch.long,
                        device=self.device)
                    self.r2_cached_planner = {
                        key: value.detach().clone()
                        if torch.is_tensor(value) else value
                        for key, value in r2_planner_tensor.items()
                    }
                else:
                    assert self.r2_cached_planner is not None
                    assert self.r2_cached_policy_target is not None
                    r2_planner_tensor = {
                        key: value.detach().clone()
                        if torch.is_tensor(value) else value
                        for key, value in self.r2_cached_planner.items()
                    }
                    cached_index = int(
                        r2_planner_tensor["selected_index"][0].item())
                    # Candidate zero means the live Actor and should keep
                    # evolving. Structured candidates are absolute temporal
                    # targets, so preserving an old Actor-relative delta would
                    # silently change brake/reverse semantics on cached ticks.
                    cached_policy = self._cached_r2_policy_target(
                        actor_policy,
                        self.r2_cached_policy_target,
                        cached_index,
                    )
                    previous_policy = self.model.policy_from_applied_action(
                        previous_applied_action.float().clamp(-1.0, 1.0))
                    if self.model.action_smoother is None:
                        cached_smoothed = cached_policy
                    else:
                        cached_smoothed = self.model.action_smoother(
                            cached_policy, previous_policy)
                    action = self.model.applied_from_policy_action(
                        cached_smoothed,
                        stacked["ego_state"][:, -1].float())
                    r2_planner_tensor.update({
                        "action": action,
                        "policy_action": cached_policy,
                        "smoothed_policy_action": cached_smoothed,
                        "actor_action": actor_action,
                        "actor_policy_action": actor_policy,
                        "actor_smoothed_policy_action": actor_smoothed,
                        "planner_evaluated": torch.zeros(
                            actor_policy.shape[0], dtype=torch.bool,
                            device=self.device),
                        "cache_age_steps": torch.full(
                            (actor_policy.shape[0],),
                            self.r2_planner_step_count
                            % self.r2_planner_interval_steps,
                            dtype=torch.long, device=self.device),
                    })
                    commitment = dict(
                        r2_planner_tensor.get("commitment") or {})
                    commitment["remaining_steps"] = int(
                        self.r2_commitment.remaining_steps)
                    commitment["committed_index"] = int(
                        self.r2_commitment.index)
                    r2_planner_tensor["commitment"] = commitment
                action = r2_planner_tensor["action"]
                self.agent_state["policy_action"] = r2_planner_tensor[
                    "policy_action"]
                self.agent_state["smoothed_policy_action"] = (
                    r2_planner_tensor["smoothed_policy_action"])
                self.agent_state["prev_action"] = action
                self.agent_state["r2_candidate_planner"] = r2_planner_tensor
                self.r2_planner_step_count += 1
                self.r2_commitment.advance()
            shadow_ready = (
                not self.prefill_active
                and
                self.shadow_imagination_enabled
                and len(self.history) >= self.pose_history
                and (
                    self.shadow_step_count - (self.pose_history - 1)
                ) % self.shadow_interval_steps == 0
            )
            if shadow_ready:
                shadow_started = time.perf_counter()
                shadow_tensor = self.model.shadow_imagination(
                    self.agent_state["latent"],
                    stacked["human_mask"][:, -1].bool(),
                    stacked["goal_position"][:, -1].float(),
                    stacked["ego_state"][:, -1].float(),
                    action,
                    horizon_steps=self.shadow_horizon_steps,
                    human_root=stacked["human_root"][:, -1].float(),
                    human_quality=stacked[
                        "human_observation_quality"][:, -1].float(),
                    human_joint_mask=stacked[
                        "joint_mask"][:, -1].bool(),
                    human_joints_body=current_human_joints_body,
                    geometry_warning_buffer_m=(
                        self.shadow_geometry_warning_buffer_m),
                )
        self.first = False
        self.shadow_step_count += 1
        action_np = action.detach().float().cpu().numpy().reshape(-1)
        model_applied_action_np = (
            model_applied_action.detach().float().cpu().numpy().reshape(-1))
        if action_np.shape != (ACTION_DIM,) or not np.isfinite(action_np).all():
            raise RuntimeError("policy returned an invalid action")
        action_np = np.clip(action_np, -1.0, 1.0)
        policy_action_np = self.agent_state.get("policy_action")
        if policy_action_np is None:
            policy_action_np = self.model.policy_from_applied_action(action)
        policy_action_np = (
            policy_action_np.detach().float().cpu().numpy().reshape(-1))
        smoothed_policy_action = self.agent_state.get(
            "smoothed_policy_action")
        if smoothed_policy_action is None:
            smoothed_policy_action = self.model.policy_from_applied_action(
                action)
        smoothed_policy_action_np = (
            smoothed_policy_action.detach().float().cpu().numpy().reshape(-1))
        actor_vertical_action = float(action_np[2])
        altitude_hold_action = actor_vertical_action
        if self.horizontal_altitude_hold and not self.model.uses_internal_action_adapter:
            altitude_agl = float(ego_state[9])
            if math.isfinite(altitude_agl):
                altitude_hold_action = float(np.clip(
                    self.altitude_hold_kp
                    * (self.altitude_hold_target_agl_m - altitude_agl),
                    -self.altitude_hold_max_action,
                    self.altitude_hold_max_action,
                ))
                action_np[2] = altitude_hold_action
        planner = self.agent_state.get("planner")
        r2_planner = self.agent_state.get("r2_candidate_planner")
        planner_action_np = action_np.copy()
        actor_action_np = planner_action_np.copy()
        action_source = r2_planner if r2_planner is not None else planner
        if action_source is not None and "actor_action" in action_source:
            actor_action_np = (
                action_source["actor_action"].detach().float().cpu().numpy()
                .reshape(-1)
            )
            if (
                actor_action_np.shape != (ACTION_DIM,)
                or not np.isfinite(actor_action_np).all()
            ):
                raise RuntimeError("planner returned an invalid Actor action")
            actor_action_np = np.clip(actor_action_np, -1.0, 1.0)
        barrier = {
            "active_constraints": 0,
            "human_active_constraints": 0,
            "static_active_constraints": 0,
            "projected": False,
            "feasible": True,
            "predictive_override": False,
            "predicted_min_clearance_before_m": float("inf"),
            "predicted_min_clearance_after_m": float("inf"),
            "minimum_slack_before_mps": float("inf"),
            "minimum_slack_after_mps": float("inf"),
            "goal_progress_before_mps": float("nan"),
            "goal_progress_after_mps": float("nan"),
            "progress_preserving_selected": False,
        }
        goal_capture_velocity = np.zeros(2, np.float32)
        goal_capture_active = False
        goal_capture_distance_m = float(np.linalg.norm(
            goal_position - np.asarray(ego_state[:3], np.float32)))
        if self.world_model_planner:
            goal_capture_velocity, goal_capture_active, (
                goal_capture_distance_m
            ) = _goal_capture_velocity(
                ego_state,
                goal_position,
                planner_clearance,
            )
            if "boundary_position_world_xy" in request:
                boundary_activation_m = float(request[
                    "boundary_activation_m"])
                if goal_capture_active:
                    boundary_activation_m = min(
                        boundary_activation_m,
                        GOAL_CAPTURE_BOUNDARY_ACTIVATION_M,
                    )
                boundary_normals, boundary_bounds = (
                    _boundary_barrier_halfplanes(
                        self._array(
                            request, "boundary_position_world_xy",
                            np.float32, (2,)),
                        float(request["boundary_yaw_rad"]),
                        self._array(
                            request, "boundary_flight_bounds_xy",
                            np.float32, (4,)),
                        activation_m=boundary_activation_m,
                        return_gain_per_s=float(request[
                            "boundary_return_gain_per_s"]),
                        max_return_speed_mps=float(request[
                            "boundary_max_return_speed_mps"]),
                    )
                )
            else:
                boundary_normals = np.zeros((0, 2), np.float32)
                boundary_bounds = np.zeros((0,), np.float32)
                boundary_activation_m = float("nan")
            if goal_capture_active:
                action_np[:2] = np.clip(
                    goal_capture_velocity / ACTION_XY_LIMIT_MPS,
                    -1.0,
                    1.0,
                )
                # Body-frame velocity already points at the goal. Continuing
                # the Actor's yaw command recreates the terminal orbit on the
                # next control tick, so hold yaw during safe goal capture.
                action_np[3] = 0.0
            desired_velocity_xy = action_np[:2] * ACTION_XY_LIMIT_MPS
            goal_delta_episode = (
                goal_position - np.asarray(ego_state[:3], np.float32))
            goal_direction_body_xy = (
                _body_to_episode_rotation(ego_state).T
                @ goal_delta_episode
            )[:2]
            safe_velocity_xy, barrier = _project_multi_human_velocity(
                desired_velocity_xy,
                human["safety_obstacle_points"],
                human["safety_obstacle_velocities"],
                human["safety_obstacle_clearances"],
                preferred_direction_xy=goal_direction_body_xy,
                static_constraint_normals=boundary_normals,
                static_constraint_bounds=boundary_bounds,
            )
            boundary_projection_magnitude = float(
                np.linalg.norm(safe_velocity_xy - desired_velocity_xy)
                if len(boundary_normals) else 0.0)
            action_np[:2] = np.clip(
                safe_velocity_xy / ACTION_XY_LIMIT_MPS, -1.0, 1.0)
            # act_step stored the pre-projection action. The RSSM transition
            # on the next control tick must receive the action that Isaac
            # actually executes, otherwise online latent state drifts exactly
            # in the safety-critical frames that matter most.
            self.agent_state["prev_action"] = torch.as_tensor(
                action_np,
                device=self.device,
                dtype=self.agent_state["prev_action"].dtype,
            ).reshape_as(self.agent_state["prev_action"])
            self.barrier_projection_count += int(barrier["projected"])
            self.barrier_infeasible_count += int(not barrier["feasible"])
            self.barrier_active_count += int(
                int(barrier["active_constraints"]) > 0)
        if (
            self.architecture_version in (
                "factorized_dreamer_v7.3", "factorized_dreamer_v7.4",
                "factorized_dreamer_v7.6", "factorized_dreamer_v7.7",
                "factorized_dreamer_v7.8", "factorized_dreamer_v7.9",
                "factorized_dreamer_v8.0", "factorized_dreamer_v8.1",
                "factorized_dreamer_v8.2", "factorized_dreamer_v8.3",
                "factorized_dreamer_v8.4")
            and not self.prefill_active
            and not np.array_equal(action_np, model_applied_action_np)
        ):
            maximum_change = float(np.max(np.abs(
                action_np - model_applied_action_np)))
            raise RuntimeError(
                f"{self.architecture_version} single-Actor deployment changed "
                "the model-owned "
                f"applied action after inference (max delta {maximum_change:.9g})")
        planner_response = None
        if r2_planner is not None:
            def planner_list(name: str, *, nullable: bool = False):
                value = (
                    r2_planner[name][0].detach().float().cpu().numpy())
                if not nullable:
                    return value.tolist()
                return [
                    None if not math.isfinite(float(item)) else float(item)
                    for item in value.reshape(-1)
                ]

            selected_index = int(
                r2_planner["selected_index"][0].item())
            raw_selected_index = int(
                r2_planner.get(
                    "raw_selected_index", r2_planner["selected_index"]
                )[0].item())
            proposed_index = int(
                r2_planner["proposed_index"][0].item())
            candidate_risk = planner_list("candidate_risk")
            candidate_fused_risk = planner_list("candidate_fused_risk")
            candidate_learned_risk = planner_list(
                "candidate_learned_risk")
            candidate_progress = planner_list("candidate_progress")
            state_supported = bool(
                r2_planner["state_supported"][0].item())
            planner_response = {
                "type": "pure_r2_candidate_event_v2",
                "candidate_scheme": str(r2_planner.get(
                    "candidate_scheme", "structured_temporal_v2")),
                "candidate_names": list(r2_planner.get(
                    "candidate_names", ())),
                "origin_timestamp_s": timestamp_s,
                "uses_legacy_action_risk": False,
                "human_safety_source": "world_model_event_only",
                "event_used_for_selection": True,
                "swept_geometry_enabled": False,
                "kinematic_cpa_enabled": False,
                "planner_evaluated": bool(
                    r2_planner["planner_evaluated"][0].item()),
                "cache_age_steps": int(
                    r2_planner["cache_age_steps"][0].item()),
                "planner_interval_steps": self.r2_planner_interval_steps,
                "triggered": bool(
                    r2_planner["intervened"][0].item()),
                "emergency": False,
                "intervened": bool(
                    r2_planner["intervened"][0].item()),
                "best_index": selected_index,
                "raw_best_index": raw_selected_index,
                "proposed_index": proposed_index,
                "candidate_count": len(candidate_risk),
                "score_improvement": float(
                    r2_planner["progress_improvement"][0].item()),
                "risk_improvement": float(
                    r2_planner["risk_improvement"][0].item()),
                "collision_probability": float(
                    candidate_learned_risk[selected_index]),
                "predicted_clearance_m": float("nan"),
                "observed_clearance_m": float(planner_clearance),
                "current_observed_clearance_m": observed_clearance,
                "kinematic_clearance_m": float("nan"),
                "available": bool(r2_planner["available"][0].item()),
                "base_supported": bool(
                    r2_planner["base_supported"][0].item()),
                "base_in_safe_band": bool(
                    r2_planner["base_in_safe_band"][0].item()),
                "base_within_safe_risk_budget": bool(
                    r2_planner[
                        "base_within_safe_risk_budget"][0].item()),
                "safe_candidate_available": bool(
                    r2_planner[
                        "safe_candidate_available"][0].item()),
                "candidate_within_safe_risk_budget": (
                    r2_planner[
                        "candidate_within_safe_risk_budget"][0].detach()
                    .cpu().bool().tolist()),
                "state_supported": state_supported,
                "selection_mode": (
                    f"h{self.r2_planner_horizon_steps}_event_probability"),
                "state_support_distance": float(
                    r2_planner["state_support_distance"][0].item()),
                "candidate_supported": (
                    r2_planner["candidate_supported"][0].detach().cpu()
                    .bool().tolist()),
                "candidate_event_supported": (
                    r2_planner["candidate_event_supported"][0].detach().cpu()
                    .bool().tolist()),
                "candidate_action_support_distance": planner_list(
                    "candidate_action_support_distance"),
                "candidate_risk": candidate_risk,
                "candidate_fused_risk": candidate_fused_risk,
                "candidate_human_risk": planner_list(
                    "candidate_human_risk"),
                "candidate_learned_risk": candidate_learned_risk,
                "candidate_boundary_risk": planner_list(
                    "candidate_boundary_risk"),
                "candidate_progress": candidate_progress,
                "candidate_minimum_boundary_clearance_m": planner_list(
                    "candidate_minimum_boundary_clearance_m"),
                "candidate_policy_action": planner_list(
                    "candidate_policy_action"),
                "candidate_smoothed_policy_action": planner_list(
                    "candidate_smoothed_policy_action"),
                "candidate_action": planner_list("candidate_action"),
                "boundary_available": bool(
                    r2_planner["boundary_available"][0].item()),
                "boundary_buffer_m": self.r2_planner_boundary_buffer_m,
                "risk_tolerance": self.r2_planner_risk_tolerance,
                "safe_risk_cap": self.r2_planner_safe_risk_cap,
                "safe_risk_margin": self.r2_planner_safe_risk_margin,
                "event_collision_clearance_m": (
                    self.r2_planner_event_collision_clearance_m),
                "horizon_steps": self.r2_planner_horizon_steps,
                "commitment": dict(
                    r2_planner.get("commitment") or {}),
                "support_snapshot_version": (
                    self.r2_planner_support_snapshot_version),
                "support_snapshot_sha256": (
                    self.r2_planner_support_snapshot_sha256),
                "diagnostics_enabled": (
                    self.r2_planner_rollout_diagnostics),
                "actor_policy_action": planner_list(
                    "actor_policy_action"),
                "actor_smoothed_policy_action": planner_list(
                    "actor_smoothed_policy_action"),
                "actor_action": actor_action_np.tolist(),
                "planner_action": planner_action_np.tolist(),
                "projected_action": action_np.tolist(),
            }
            if (
                self.r2_planner_rollout_diagnostics
                and bool(r2_planner["planner_evaluated"][0].item())
            ):
                try:
                    planner_response["rollout_diagnostics"] = (
                        self._r2_rollout_diagnostics_payload(
                            r2_planner,
                            human["human_mask"],
                            human["human_ids"],
                            self.r2_planner_horizon_steps,
                        )
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    # The trace is explicitly observational: a serialization
                    # defect must not replace the already-selected action.
                    planner_response["diagnostics_error"] = (
                        f"{type(error).__name__}: {error}")
        elif planner is not None:
            planner_response = {
                "type": "legacy_action_risk",
                "uses_legacy_action_risk": True,
                "triggered": bool(planner["triggered"][0].item()),
                "emergency": bool(planner["emergency"][0].item()),
                "intervened": bool(planner["intervened"][0].item()),
                "best_index": int(planner["best_index"][0].item()),
                "score_improvement": float(
                    planner["score_improvement"][0].item()),
                "collision_probability": float(
                    planner["collision_probability"][0].item()),
                "predicted_clearance_m": float(
                    planner["predicted_clearance_m"][0].item()),
                "observed_clearance_m": float(
                    planner["observed_clearance_m"][0].item()),
                "current_observed_clearance_m": observed_clearance,
                "kinematic_clearance_m": kinematic_clearance,
                "escape_direction_locked": bool(escape_direction_locked),
                "hard_stop": bool(planner["hard_stop"][0].item()),
                "forward_cap_action": float(
                    planner["forward_cap_action"][0].item()),
                "multi_human_projected": bool(barrier["projected"]),
                "multi_human_constraint_count": int(
                    barrier["active_constraints"]),
                "multi_human_constraint_count_only": int(
                    barrier["human_active_constraints"]),
                "static_boundary_constraint_count": int(
                    barrier["static_active_constraints"]),
                "joint_boundary_applied": True,
                "boundary_projection_magnitude_mps": (
                    boundary_projection_magnitude),
                "multi_human_feasible": bool(barrier["feasible"]),
                "multi_human_predictive_override": bool(
                    barrier["predictive_override"]),
                "multi_human_predicted_clearance_before_m": float(
                    barrier["predicted_min_clearance_before_m"]),
                "multi_human_predicted_clearance_after_m": float(
                    barrier["predicted_min_clearance_after_m"]),
                "multi_human_slack_before_mps": float(
                    barrier["minimum_slack_before_mps"]),
                "multi_human_slack_after_mps": float(
                    barrier["minimum_slack_after_mps"]),
                "goal_progress_before_mps": float(
                    barrier["goal_progress_before_mps"]),
                "goal_progress_after_mps": float(
                    barrier["goal_progress_after_mps"]),
                "progress_preserving_selected": bool(
                    barrier["progress_preserving_selected"]),
                "multi_human_projection_count": int(
                    self.barrier_projection_count),
                "multi_human_infeasible_count": int(
                    self.barrier_infeasible_count),
                "multi_human_active_count": int(
                    self.barrier_active_count),
                "goal_capture_active": bool(goal_capture_active),
                "goal_capture_distance_m": float(
                    goal_capture_distance_m),
                "goal_capture_desired_speed_mps": float(
                    np.linalg.norm(goal_capture_velocity)),
                "boundary_activation_m": float(boundary_activation_m),
                "actor_action": actor_action_np.tolist(),
                "planner_action": planner_action_np.tolist(),
                "projected_action": action_np.tolist(),
            }
        shadow_response = None
        if shadow_tensor is not None:
            def shadow_array(name):
                return (
                    shadow_tensor[name][0].detach().float().cpu().numpy()
                )

            future_steps = shadow_tensor["future_steps"].detach().cpu().numpy()
            ego_future = shadow_array("ego_state")
            collision_future = shadow_array("collision_probability")
            clearance_future = shadow_array(
                "predicted_min_human_clearance_m")
            shadow_response = {
                "origin_timestamp_s": timestamp_s,
                "step_duration_s": self.shadow_step_duration_s,
                "risk_horizon_s": self.shadow_risk_horizon_s,
                "horizon_steps": self.shadow_horizon_steps,
                "future_steps": future_steps.astype(int).tolist(),
                "ego_position_episode_local": ego_future[:, :3].tolist(),
                "goal_distance_m": shadow_array("goal_distance_m").tolist(),
                "action": shadow_array("action").tolist(),
                "collision_probability": collision_future.tolist(),
                "learned_collision_probability": shadow_array(
                    "learned_collision_probability").tolist(),
                "geometry_collision_probability": shadow_array(
                    "geometry_collision_probability").tolist(),
                "predicted_min_human_clearance_m": clearance_future.tolist(),
                "raw_min_human_clearance_m": shadow_array(
                    "raw_min_human_clearance_m").tolist(),
                "reward": shadow_array("reward").tolist(),
                "critic_value": shadow_array("critic_value").tolist(),
                "continuation_probability": shadow_array(
                    "continuation_probability").tolist(),
                "origin_collision_probability": float(
                    shadow_array("origin_collision_probability")),
                "origin_learned_collision_probability": float(
                    shadow_array(
                        "origin_learned_collision_probability")),
                "origin_geometry_collision_probability": float(
                    shadow_array(
                        "origin_geometry_collision_probability")),
                "origin_action": shadow_array("origin_action").tolist(),
                "origin_predicted_min_human_clearance_m": float(
                    shadow_array(
                        "origin_predicted_min_human_clearance_m")),
                "origin_raw_min_human_clearance_m": float(
                    shadow_array("origin_raw_min_human_clearance_m")),
                "origin_critic_value": float(
                    shadow_array("origin_critic_value")),
                "maximum_collision_probability": float(
                    np.max(collision_future)),
                "minimum_predicted_human_clearance_m": float(
                    np.min(clearance_future)),
                "compute_latency_ms": (
                    (time.perf_counter() - shadow_started) * 1000.0
                    if shadow_started is not None else float("nan")
                ),
            }
            if "one_step_human_hazard" in shadow_tensor:
                shadow_response.update({
                    "transition_event_probability": shadow_array(
                        "transition_event_probability").tolist(),
                    "one_step_human_hazard": shadow_array(
                        "one_step_human_hazard").tolist(),
                    "one_step_continue_probability": shadow_array(
                        "one_step_continue_probability").tolist(),
                })
        return {
            "ok": True,
            "protocol": PROTOCOL,
            # Pure Dreamer uses this echo as an execution acknowledgement on
            # the next physical control step. Historical runtimes ignore the
            # additive request/response field.
            "control_step_index": request.get("control_step_index"),
            "action": action_np.tolist(),
            "policy_action": policy_action_np.tolist(),
            "smoothed_policy_action": smoothed_policy_action_np.tolist(),
            "action_adapter": {
                "applied_inside_model": bool(
                    self.model.uses_internal_action_adapter),
                "applied_action": action_np.tolist(),
            },
            "action_smoothing": {
                "enabled": self.model.action_smoother is not None,
                "raw_policy_target": policy_action_np.tolist(),
                "smoothed_policy_target": (
                    smoothed_policy_action_np.tolist()),
                "maximum_step_delta": (
                    [] if self.model.action_smoother is None else
                    self.model.action_smoother.maximum_step_delta.detach()
                    .cpu().tolist()),
            },
            "people": int(human["human_mask"].sum()),
            "truncated_people": int(human["truncated_people"]),
            "perception": {
                "retained_empty_observation": bool(
                    human["retained_empty_observation"]),
                "empty_hold_s": self.human_empty_hold_s,
            },
            "history_frames": len(self.history),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "checkpoint_step": self.checkpoint_step,
            # Name the generator of the recorded policy3 command, not merely
            # the update clock of the checkpoint loaded for observation
            # packing. A prefill explorer remains policy step zero even when
            # a World-only checkpoint has already advanced.
            "collector_policy_step": (
                0 if self.prefill_active else self.checkpoint_step),
            "exploration": {
                "enabled": (
                    self.prefill_active
                    or self.online_exploration_scale > 0.0),
                "mode": (
                    "smooth_random_prefill" if self.prefill_active
                    else "smoothed_stochastic_actor" if self.stochastic_actor
                    else "actor_mode"),
                "prefill_active": self.prefill_active,
                "environment_steps": self.environment_steps,
                "environment_prefill_frames": (
                    self.environment_prefill_frames),
                "world_update_count": self.world_update_count,
                "configured_scale": self.online_exploration_scale,
                "effective_scale": effective_exploration_scale,
                "clearance_factor": clearance_exploration_factor,
                "delta_limits": self.exploration_delta_limits.detach()
                .cpu().tolist(),
                "macro_index": self.agent_state.get(
                    "exploration_macro_index"),
                "macro_remaining_steps": self.agent_state.get(
                    "exploration_macro_remaining"),
            },
            "altitude_hold": {
                "enabled": self.horizontal_altitude_hold,
                "target_agl_m": self.altitude_hold_target_agl_m,
                "actor_vertical_action": actor_vertical_action,
                "applied_vertical_action": altitude_hold_action,
            },
            "planner": planner_response,
            "shadow_imagination": shadow_response,
        }

    def handle(self, request: dict) -> dict[str, object]:
        if request.get("protocol") not in (None, PROTOCOL):
            raise ValueError("unsupported protocol")
        kind = str(request.get("type", ""))
        with self.lock:
            if kind == "reset":
                return self.reset(
                    collection_scene_seed=request.get(
                        "collection_scene_seed"))
            if kind == "status":
                return {
                    "ok": True,
                    "protocol": PROTOCOL,
                    "checkpoint": str(self.checkpoint_path),
                    "checkpoint_step": self.checkpoint_step,
                    "best_validation": self.best_validation,
                    "device": str(self.device),
                    "max_people": self.max_people,
                    "pose_history": self.pose_history,
                    "actor_explicit_human_geometry_enabled": bool(
                        self.model.actor_explicit_human_geometry_enabled),
                    "actor_authoritative_ego_token_enabled": bool(
                        self.model.actor_authoritative_ego_token_enabled),
                    "actor_direct_ego_task_state_enabled": bool(
                        self.model.actor_direct_ego_task_state_enabled),
                    "deterministic_evaluation_state_enabled": bool(
                        self.model.deterministic_evaluation_state_enabled),
                    "actor_unified_policy_enabled": bool(
                        self.model.actor.config.unified_policy),
                    "actor_human_reflection_equivariant_enabled": bool(
                        self.model.actor.config.human_reflection_equivariant),
                    "actor_human_conditioned_residual_enabled": bool(
                        self.model.actor.config.human_conditioned_residual),
                    "actor_human_physical_slots": int(
                        self.model.actor.config.human_physical_slots),
                    "actor_human_physical_fields_per_slot": int(
                        self.model.actor.config.human_physical_fields_per_slot),
                    "critic_full_state_enabled": bool(
                        self.model.critic_full_state_enabled),
                    "direct_task_geometry_enabled": bool(
                        self.model.direct_task_geometry_enabled),
                    "task_memory_enabled": bool(
                        self.model.task_memory_enabled),
                    "direct_task_memory_enabled": bool(
                        self.model.direct_task_memory_enabled),
                    "analytic_task_memory_events_enabled": bool(
                        self.model.transition_event.analytic_task_memory_events),
                    "explicit_joint_kinematics_enabled": bool(
                        self.model.transition_event.explicit_joint_kinematics),
                    "human_geometry_topk_physical_slots": int(
                        self.model.transition_event.geometry_topk_physical_slots),
                    "human_joint_kinematic_enabled": bool(
                        self.model.prediction_heads.config.kinematic_joint_velocity_only),
                    "conservative_cv_clearance_enabled": bool(
                        self.model.conservative_human_clearance_enabled),
                    "policy_validation": dict(self.policy_validation),
                    "prefill_active": self.prefill_active,
                    "unsafe_canary_actor_enabled": (
                        self.unsafe_canary_actor_enabled),
                    "world_model_planner": self.world_model_planner,
                    "r2_candidate_planner": self.r2_candidate_planner,
                    "r2_planner_horizon_steps": (
                        self.r2_planner_horizon_steps),
                    "r2_planner_interval_steps": (
                        self.r2_planner_interval_steps),
                    "r2_planner_event_collision_clearance_m": (
                        self.r2_planner_event_collision_clearance_m),
                    "r2_planner_boundary_buffer_m": (
                        self.r2_planner_boundary_buffer_m),
                    "r2_planner_commitment_steps": (
                        self.r2_planner_commitment_steps),
                    "r2_planner_switch_risk_improvement": (
                        self.r2_planner_switch_risk_improvement),
                    "r2_planner_rollout_diagnostics": (
                        self.r2_planner_rollout_diagnostics),
                    "r2_planner_support_snapshot_version": (
                        self.r2_planner_support_snapshot_version),
                    "r2_planner_support_snapshot_sha256": (
                        self.r2_planner_support_snapshot_sha256),
                    "shadow_imagination": self.shadow_imagination_enabled,
                    "shadow_horizon_steps": self.shadow_horizon_steps,
                    "shadow_interval_steps": self.shadow_interval_steps,
                    "shadow_step_duration_s": self.shadow_step_duration_s,
                    "shadow_risk_horizon_s": self.shadow_risk_horizon_s,
                    "online_exploration_scale": self.online_exploration_scale,
                    "stochastic_actor": self.stochastic_actor,
                    "human_empty_hold_s": self.human_empty_hold_s,
                    "action_smoothing": (
                        self.model.action_smoother is not None),
                    "action_smoothing_maximum_step_delta": (
                        [] if self.model.action_smoother is None else
                        self.model.action_smoother.maximum_step_delta.detach()
                        .cpu().tolist()),
                    "exploration_delta_limits": (
                        self.exploration_delta_limits.detach().cpu().tolist()),
                }
            if kind == "set_exploration_scale":
                scale = float(request.get("scale", 0.0))
                if not math.isfinite(scale) or scale < 0.0:
                    raise ValueError(
                        "exploration scale must be finite and non-negative")
                self.online_exploration_scale = scale
                return {
                    "ok": True,
                    "protocol": PROTOCOL,
                    "online_exploration_scale": scale,
                }
            if kind == "step":
                return self.step(request)
        raise ValueError(f"unsupported request type {kind!r}")


class ReloadableFactorizedPolicyRuntime:
    """Atomically adopt a published policy only at an episode boundary.

    The learner replaces ``checkpoint_path`` with ``os.replace``. Isaac keeps
    using the current runtime throughout an episode; the controller's reset
    request is the sole place where a changed checkpoint may be constructed
    and swapped in. A failed audit/load retains the previous known-good model.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        *,
        runtime_factory=FactorizedPolicyRuntime,
        **runtime_kwargs,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.runtime_factory = runtime_factory
        self.runtime_kwargs = dict(runtime_kwargs)
        self.lock = threading.Lock()
        self.runtime = self.runtime_factory(
            self.checkpoint_path, self.device, **self.runtime_kwargs)
        self.checkpoint_fingerprint = self._fingerprint()
        self.reload_count = 0
        self.last_reload_error: str | None = None

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def _fingerprint(self) -> tuple[int, int, int]:
        stat = self.checkpoint_path.stat()
        return int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns)

    def _reload_if_published(self) -> bool:
        fingerprint = self._fingerprint()
        if fingerprint == self.checkpoint_fingerprint:
            return False
        try:
            replacement = self.runtime_factory(
                self.checkpoint_path, self.device, **self.runtime_kwargs)
        except Exception as error:
            self.last_reload_error = f"{type(error).__name__}: {error}"
            print(
                "FACTORIZED_POLICY_RELOAD_REJECTED "
                f"checkpoint={self.checkpoint_path} "
                f"error={self.last_reload_error}",
                flush=True,
            )
            # Remember neither the new fingerprint nor the failed runtime: a
            # corrected atomic publication should be retried next episode.
            return False
        self.runtime = replacement
        self.checkpoint_fingerprint = fingerprint
        self.reload_count += 1
        self.last_reload_error = None
        print(
            "FACTORIZED_POLICY_RELOADED "
            f"checkpoint={self.checkpoint_path} "
            f"step={replacement.checkpoint_step} "
            f"reload_count={self.reload_count}",
            flush=True,
        )
        return True

    def handle(self, request: dict) -> dict[str, object]:
        kind = str(request.get("type", ""))
        # ``PolicyServer`` is threaded.  Keep the publication lock for the
        # complete stateful request, not merely while copying ``self.runtime``.
        # Otherwise an old-runtime ``step`` can still be executing while a
        # concurrent reset installs and resets a new runtime, so one recorded
        # episode no longer has an atomic policy-version boundary.
        with self.lock:
            reloaded = self._reload_if_published() if kind == "reset" else False
            runtime = self.runtime
            response = runtime.handle(request)
            if kind in ("reset", "status"):
                response["reload_on_reset"] = True
                response["reload_count"] = self.reload_count
                response["last_reload_error"] = self.last_reload_error
                if kind == "reset":
                    response["checkpoint_reloaded"] = reloaded
            return response


class PolicyHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline(4 * 1024 * 1024)
            if not line:
                return
            try:
                request = json.loads(line)
                response = self.server.runtime.handle(request)  # type: ignore[attr-defined]
            except Exception as error:  # keep the controller connection alive
                response = {
                    "ok": False,
                    "protocol": PROTOCOL,
                    "error": f"{type(error).__name__}: {error}",
                }
            self.wfile.write(
                (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))
            self.wfile.flush()


class PolicyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9775)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--allow-unsafe-policy", action="store_true",
        help="Bypass policy validation gates for offline diagnostics only.",
    )
    parser.add_argument(
        "--reload-checkpoint-on-reset",
        action="store_true",
        help=(
            "Reload an atomically replaced --checkpoint only when the "
            "controller starts a new episode."
        ),
    )
    parser.add_argument(
        "--world-model-planner",
        action="store_true",
        help=(
            "Enable short-horizon Actor/slow/left/right imagination for "
            "conservative online data collection."
        ),
    )
    parser.add_argument(
        "--r2-candidate-planner",
        action="store_true",
        help=(
            "Execute the best supported Actor-centred candidate after frozen "
            "World/Event, predicted progress and boundary scoring. Human "
            "safety selection is Event-only; swept geometry and CPA are off. "
            "This does not use the disabled legacy action_risk head."
        ),
    )
    parser.add_argument(
        "--r2-planner-horizon-steps",
        type=int,
        default=None,
        help=(
            "Override the Event candidate horizon. By default the longest "
            "checkpoint Event release horizon is used (normally h15)."
        ),
    )
    parser.add_argument(
        "--r2-planner-interval-steps",
        type=int,
        default=2,
        help=(
            "Run the full Event candidate rollout every N policy steps and "
            "reuse its selected policy offset between evaluations. At 10 Hz, "
            "N=2 runs the planner at 5 Hz."
        ),
    )
    parser.add_argument(
        "--r2-planner-risk-tolerance",
        type=float,
        default=0.002,
        help=(
            "Maximum Event-risk gap admitted to the progress tie band. "
            "This runtime value overrides the value stored in old Actor "
            "checkpoints."
        ),
    )
    parser.add_argument(
        "--r2-planner-safe-risk-cap",
        type=float,
        default=0.08,
        help=(
            "Absolute cumulative h15 Event-risk budget. Supported candidates "
            "below this cap compete on predicted goal progress; when none "
            "qualify, the planner returns to strict minimum-risk selection."
        ),
    )
    parser.add_argument(
        "--r2-planner-safe-risk-margin",
        type=float,
        default=0.02,
        help=(
            "Maximum Event-risk gap above the safest supported candidate "
            "that may still compete on goal progress inside the absolute "
            "safe-risk cap."
        ),
    )
    parser.add_argument(
        "--r2-planner-boundary-buffer-m", type=float, default=0.35,
        help="Minimum desired clearance inside the supplied flight bounds.",
    )
    parser.add_argument(
        "--r2-planner-commitment-steps", type=int, default=8,
        help=(
            "Keep an Event-selected candidate direction for this many "
            "10 Hz policy steps unless another candidate reduces Event risk "
            "by the configured switch margin (default: 8 = 0.8 s)."),
    )
    parser.add_argument(
        "--r2-planner-switch-risk-improvement", type=float, default=0.03,
        help="Absolute Event-risk improvement required to break commitment.",
    )
    parser.add_argument(
        "--r2-planner-switch-progress-improvement", type=float, default=0.25,
        help=(
            "Predicted progress improvement required to break commitment "
            "when both candidates are inside the safe-risk budget."),
    )
    parser.add_argument(
        "--r2-planner-rollout-diagnostics",
        action="store_true",
        help=(
            "Expose observational h5/h10/h15 candidate rollout tensors for "
            "per-run JSONL tracing. This never changes candidate selection."
        ),
    )
    parser.add_argument(
        "--human-empty-hold-s", type=float, default=0.30,
        help=(
            "Causally propagate the last non-empty Human packet across a "
            "short fresh-but-empty detector dropout."),
    )
    parser.add_argument(
        "--shadow-imagination",
        action="store_true",
        help=(
            "Run deterministic RSSM/Actor futures for diagnostics without "
            "changing the raw Actor action."
        ),
    )
    parser.add_argument("--shadow-horizon-steps", type=int, default=15)
    parser.add_argument("--shadow-interval-steps", type=int, default=10)
    parser.add_argument("--shadow-step-duration-s", type=float, default=0.1)
    parser.add_argument(
        "--online-exploration-scale", type=float, default=0.0,
        help=(
            "Blend a bounded Actor sample around its mode during online data "
            "collection. Zero keeps deterministic evaluation behavior."),
    )
    parser.add_argument(
        "--stochastic-actor",
        action="store_true",
        help=(
            "Sample the exact Actor distribution used by latent imagination. "
            "This is the pure Dreamer collection policy and bypasses the "
            "historical online macro-exploration branch."),
    )
    parser.add_argument(
        "--exploration-delta-limits", default="0.05,0.12,0.015,0.03",
        help="Per-action maximum sampled delta for vx,vy,vz,yaw.",
    )
    parser.add_argument(
        "--exploration-disable-clearance-m", type=float, default=0.55)
    parser.add_argument(
        "--exploration-full-clearance-m", type=float, default=1.50)
    parser.add_argument(
        "--horizontal-altitude-hold", action="store_true",
        help=(
            "Replace only vz with a bounded AGL feedback command so Actor "
            "training can focus on horizontal navigation."),
    )
    parser.add_argument("--altitude-hold-target-agl-m", type=float, default=1.0)
    parser.add_argument("--altitude-hold-kp", type=float, default=0.8)
    parser.add_argument("--altitude-hold-max-action", type=float, default=0.20)
    args = parser.parse_args()
    if args.world_model_planner and args.shadow_imagination:
        parser.error(
            "--shadow-imagination evaluates the raw Actor and cannot be "
            "combined with --world-model-planner")
    if args.world_model_planner and args.r2_candidate_planner:
        parser.error(
            "--world-model-planner and --r2-candidate-planner are mutually "
            "exclusive")
    if (
        args.r2_planner_horizon_steps is not None
        and args.r2_planner_horizon_steps <= 0
    ):
        parser.error("--r2-planner-horizon-steps must be positive")
    if args.r2_planner_interval_steps <= 0:
        parser.error("--r2-planner-interval-steps must be positive")
    if args.r2_planner_commitment_steps <= 0:
        parser.error("--r2-planner-commitment-steps must be positive")
    if (
        args.r2_planner_rollout_diagnostics
        and not args.r2_candidate_planner
    ):
        parser.error(
            "--r2-planner-rollout-diagnostics requires "
            "--r2-candidate-planner")
    if (
        not math.isfinite(args.r2_planner_risk_tolerance)
        or args.r2_planner_risk_tolerance < 0.0
    ):
        parser.error(
            "--r2-planner-risk-tolerance must be finite and non-negative")
    if (
        not math.isfinite(args.r2_planner_safe_risk_cap)
        or not 0.0 < args.r2_planner_safe_risk_cap < 1.0
    ):
        parser.error("--r2-planner-safe-risk-cap must be within (0,1)")
    if (
        not math.isfinite(args.r2_planner_safe_risk_margin)
        or args.r2_planner_safe_risk_margin < 0.0
    ):
        parser.error(
            "--r2-planner-safe-risk-margin must be finite and non-negative")
    if args.r2_planner_boundary_buffer_m <= 0.0:
        parser.error("--r2-planner-boundary-buffer-m must be positive")
    if (
        not math.isfinite(args.r2_planner_switch_risk_improvement)
        or args.r2_planner_switch_risk_improvement < 0.0
    ):
        parser.error(
            "--r2-planner-switch-risk-improvement must be non-negative")
    if (
        not math.isfinite(args.r2_planner_switch_progress_improvement)
        or args.r2_planner_switch_progress_improvement < 0.0
    ):
        parser.error(
            "--r2-planner-switch-progress-improvement must be non-negative")
    if args.human_empty_hold_s <= 0.0:
        parser.error("--human-empty-hold-s must be positive")
    if args.shadow_horizon_steps <= 0:
        parser.error("--shadow-horizon-steps must be positive")
    if args.shadow_interval_steps <= 0:
        parser.error("--shadow-interval-steps must be positive")
    if args.shadow_step_duration_s <= 0.0:
        parser.error("--shadow-step-duration-s must be positive")
    if args.online_exploration_scale < 0.0:
        parser.error("--online-exploration-scale must be non-negative")
    try:
        args.exploration_delta_limits = tuple(
            float(item) for item in args.exploration_delta_limits.split(","))
    except ValueError:
        parser.error("--exploration-delta-limits must be comma-separated floats")
    if (
        len(args.exploration_delta_limits) != ACTION_DIM
        or any(value < 0.0 for value in args.exploration_delta_limits)
    ):
        parser.error("--exploration-delta-limits requires four non-negative values")
    if not (
        0.0 <= args.exploration_disable_clearance_m
        < args.exploration_full_clearance_m
    ):
        parser.error("invalid exploration clearance interval")
    if args.altitude_hold_target_agl_m <= 0.0:
        parser.error("--altitude-hold-target-agl-m must be positive")
    if args.altitude_hold_kp < 0.0 or args.altitude_hold_max_action <= 0.0:
        parser.error("invalid altitude hold gain/action limit")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    runtime_kwargs = {
        "allow_unsafe_policy": args.allow_unsafe_policy,
        "world_model_planner": args.world_model_planner,
        "r2_candidate_planner": args.r2_candidate_planner,
        "r2_planner_horizon_steps": args.r2_planner_horizon_steps,
        "r2_planner_interval_steps": args.r2_planner_interval_steps,
        "r2_planner_risk_tolerance": args.r2_planner_risk_tolerance,
        "r2_planner_safe_risk_cap": args.r2_planner_safe_risk_cap,
        "r2_planner_safe_risk_margin": args.r2_planner_safe_risk_margin,
        "r2_planner_boundary_buffer_m": (
            args.r2_planner_boundary_buffer_m),
        "r2_planner_commitment_steps": (
            args.r2_planner_commitment_steps),
        "r2_planner_switch_risk_improvement": (
            args.r2_planner_switch_risk_improvement),
        "r2_planner_switch_progress_improvement": (
            args.r2_planner_switch_progress_improvement),
        "r2_planner_rollout_diagnostics": (
            args.r2_planner_rollout_diagnostics),
        "human_empty_hold_s": args.human_empty_hold_s,
        "shadow_imagination": args.shadow_imagination,
        "shadow_horizon_steps": args.shadow_horizon_steps,
        "shadow_interval_steps": args.shadow_interval_steps,
        "shadow_step_duration_s": args.shadow_step_duration_s,
        "online_exploration_scale": args.online_exploration_scale,
        "exploration_delta_limits": args.exploration_delta_limits,
        "exploration_disable_clearance_m": (
            args.exploration_disable_clearance_m),
        "exploration_full_clearance_m": args.exploration_full_clearance_m,
        "stochastic_actor": args.stochastic_actor,
        "horizontal_altitude_hold": args.horizontal_altitude_hold,
        "altitude_hold_target_agl_m": args.altitude_hold_target_agl_m,
        "altitude_hold_kp": args.altitude_hold_kp,
        "altitude_hold_max_action": args.altitude_hold_max_action,
    }
    if args.reload_checkpoint_on_reset:
        runtime = ReloadableFactorizedPolicyRuntime(
            checkpoint, args.device, **runtime_kwargs)
    else:
        runtime = FactorizedPolicyRuntime(
            checkpoint, args.device, **runtime_kwargs)
    with PolicyServer((args.host, args.port), PolicyHandler) as server:
        server.runtime = runtime  # type: ignore[attr-defined]
        print(
            f"FACTORIZED_POLICY_READY protocol={PROTOCOL} host={args.host} "
            f"port={args.port} device={args.device} checkpoint={checkpoint} "
            f"step={runtime.checkpoint_step} "
            f"prefill_active={int(runtime.prefill_active)} "
            "unsafe_canary_actor_enabled="
            f"{int(runtime.unsafe_canary_actor_enabled)} "
            f"world_model_planner={int(runtime.world_model_planner)} "
            f"r2_candidate_planner={int(runtime.r2_candidate_planner)} "
            f"r2_planner_horizon_steps={runtime.r2_planner_horizon_steps} "
            f"r2_planner_interval_steps={runtime.r2_planner_interval_steps} "
            f"r2_planner_risk_tolerance="
            f"{runtime.r2_planner_risk_tolerance} "
            f"r2_planner_safe_risk_cap="
            f"{runtime.r2_planner_safe_risk_cap} "
            f"r2_planner_safe_risk_margin="
            f"{runtime.r2_planner_safe_risk_margin} "
            f"r2_planner_event_collision_clearance_m="
            f"{runtime.r2_planner_event_collision_clearance_m} "
            f"r2_planner_commitment_steps="
            f"{runtime.r2_planner_commitment_steps} "
            f"r2_planner_switch_risk_improvement="
            f"{runtime.r2_planner_switch_risk_improvement} "
            f"r2_planner_switch_progress_improvement="
            f"{runtime.r2_planner_switch_progress_improvement} "
            f"r2_planner_rollout_diagnostics="
            f"{int(runtime.r2_planner_rollout_diagnostics)} "
            f"human_empty_hold_s={runtime.human_empty_hold_s} "
            f"reload_on_reset={int(args.reload_checkpoint_on_reset)} "
            f"shadow_imagination={int(runtime.shadow_imagination_enabled)} "
            f"shadow_horizon_steps={runtime.shadow_horizon_steps} "
            f"shadow_interval_steps={runtime.shadow_interval_steps} "
            f"online_exploration_scale={runtime.online_exploration_scale}",
            f"stochastic_actor={int(runtime.stochastic_actor)}",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
