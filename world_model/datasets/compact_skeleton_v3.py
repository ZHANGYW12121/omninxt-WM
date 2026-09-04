"""Direct reader for compact Isaac skeleton/state dataset v3.

The recorder keeps all scene people and COCO12_BODY positions (COCO17 source
indices 5..16). This loader
derives relative joint velocity, risk-selects model slots, and emits the exact
factorized world-model batch without a separate pose-cache conversion step.
"""

from __future__ import annotations

import copy
import json
import math
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - training environment ships SciPy
    linear_sum_assignment = None

from modules.goal_conditioning import goal_features_numpy
from modules.skeleton_topology import (
    COCO12_BODY_JOINT_COUNT, complete_coco12_topology_numpy,
    hip_joint_indices,
    sanitize_coco12_geometry_numpy,
)
from modules.task_geometry import (
    WAREHOUSE_V2_KLT134_STATIC_AABB_WORLD,
    WAREHOUSE_V2_KLT134_STATIC_GEOMETRY_MIGRATION,
    migrate_warehouse_v2_klt134_static_geometry,
    task_geometry_proximity_numpy,
)
from modules.task_memory import derive_task_memory_numpy
from .factorized_schema import (
    HumanDetections,
    assign_stable_human_slots,
    human_root_and_relative_joints_numpy,
    validate_factorized_batch,
)

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    Dataset = object  # type: ignore[misc,assignment]


SCHEMA = "omninxt.crowd_skeleton_state.v3"
JOINT_COUNT = COCO12_BODY_JOINT_COUNT
SKELETON_FEATURE_DIM = 7
# The upstream per-joint Kalman filter caps articulated velocity at 2.5 m/s.
# Compact-v3 did not persist that state, so replay/deployment reconstruct the
# same causal velocity from successive filtered joint positions and retain a
# small margin for finite-difference error.  This prevents a depth/bone-fit
# discontinuity from becoming a tens-of-m/s imagined Human state.
CAUSAL_JOINT_SPEED_LIMIT_MPS = 3.0
# A navigation episode is recorded only after takeoff has settled. These are
# the simulator's versioned pre-navigation gates; episodes outside them came
# from stale PX4/PhysX reset state and are not samples from the deployed MDP.
PRE_NAVIGATION_MAX_SPAWN_XY_ERROR_M = 0.75
PRE_NAVIGATION_MAX_ALTITUDE_ERROR_M = 0.25
PRE_NAVIGATION_MAX_SPEED_MPS = 0.75
PRE_NAVIGATION_MAX_TILT_DEG = 10.0
PRE_NAVIGATION_EXPECTED_ALTITUDE_M = 1.0
DEFAULT_EXCLUDED_TERMINATION_REASONS = ("time_limit",)
DEFAULT_RISK_HORIZON_S = 2.5
DEFAULT_DENSE_CLEARANCE_M = 0.7
GT_MATCH_MIN_GATE_M = 0.25
GT_MATCH_MAX_GATE_M = 1.00
# Privileged collision geometry is an audit/target channel, not a policy
# observation.  It must losslessly retain every person in the configured
# 17--40-person scene while remaining independent of the model's 23 local
# observable slots.
PRIVILEGED_GT_COLLISION_MAX_PEOPLE = 40
PRIVILEGED_COLLISION_JOINT_NAMES = (
    "Pelvis", "R_Hand", "L_Hand", "R_Foot", "L_Foot",
    "R_KneeShareBone", "L_KneeShareBone",
    "R_ElbowShareBone", "L_ElbowShareBone", "Head",
)
# These are the actual Isaac collision-sphere radii used by both
# skeleton_tracker and data_recorder for the versioned compact-v3 schema.
PRIVILEGED_COLLISION_JOINT_RADII_M = {
    "Pelvis": 0.14,
    "Head": 0.12,
    "R_Hand": 0.08,
    "L_Hand": 0.08,
    "R_Foot": 0.09,
    "L_Foot": 0.09,
    "R_KneeShareBone": 0.09,
    "L_KneeShareBone": 0.09,
    "R_ElbowShareBone": 0.08,
    "L_ElbowShareBone": 0.08,
}
LEGACY_TASK_GEOMETRY_CONTRACT = (
    "omninxt.factorized-task-geometry.clean-replay.v1")
CLEAN_TASK_GEOMETRY_CONTRACT = (
    "omninxt.factorized-task-geometry.actor-proximity.v2")
STATIC_TERMINAL_TASK_GEOMETRY_CONTRACT = (
    "omninxt.factorized-task-geometry.analytic-static-terminal.v3")
STATIC_GROUND_AGL_CONTRACT = "omninxt.static-ground-plane-agl.v1"
# Historical recordings used a 0.20 m-high FixedCuboid centered at z=0 as a
# spawn support.  Its top was therefore exactly 0.10 m above the warehouse
# floor.  The closest-hit AGL ray alternated between those two surfaces and
# could also hit a Human.  This constant identifies that recorded geometry;
# it is not an estimated correction parameter.
LEGACY_RAISED_SUPPORT_TOP_OFFSET_M = 0.10
REWARD_COMPONENT_KEYS = (
    "event", "progress", "human_clearance", "smoothness", "height", "time",
)

TRANSITION_EVENT_CONTINUE = 0
TRANSITION_EVENT_HUMAN_COLLISION = 1
TRANSITION_EVENT_STATIC_COLLISION = 2
TRANSITION_EVENT_REACHED_GOAL = 3
TRANSITION_EVENT_OTHER_TERMINAL = 4
NON_GOAL_EVENT_CONTINUE = 0
NON_GOAL_EVENT_HUMAN_COLLISION = 1
NON_GOAL_EVENT_STATIC_COLLISION = 2
NON_GOAL_EVENT_HARD_FAILURE = 3
NON_GOAL_EVENT_STUCK_TIMEOUT = 4
TRANSITION_EVENT_KEYS = (
    "continue",
    "human_collision",
    "static_collision",
    "reached_goal",
    "other_task_terminal",
)
_TERMINATION_TO_TRANSITION_EVENT = {
    0: TRANSITION_EVENT_CONTINUE,
    1: TRANSITION_EVENT_REACHED_GOAL,
    2: TRANSITION_EVENT_HUMAN_COLLISION,
    3: TRANSITION_EVENT_STATIC_COLLISION,
    4: TRANSITION_EVENT_OTHER_TERMINAL,
    5: TRANSITION_EVENT_OTHER_TERMINAL,
    6: TRANSITION_EVENT_OTHER_TERMINAL,
}
_CENSORED_TERMINATION_CODES = frozenset((7, 8, 9, 10, 11))
_TERMINATION_TO_NON_GOAL_EVENT = {
    0: NON_GOAL_EVENT_CONTINUE,
    # Goal is not a learned event in the fresh objective. It means no
    # non-goal terminal occurred and is composed analytically from next Ego.
    1: NON_GOAL_EVENT_CONTINUE,
    2: NON_GOAL_EVENT_HUMAN_COLLISION,
    3: NON_GOAL_EVENT_STATIC_COLLISION,
    4: NON_GOAL_EVENT_HARD_FAILURE,
    5: NON_GOAL_EVENT_HARD_FAILURE,
    6: NON_GOAL_EVENT_STUCK_TIMEOUT,
}
_TASK_TERMINATION_REASONS = frozenset((
    "reached_goal", "human_collision", "static_collision",
    "out_of_bounds", "crash", "stuck_timeout",
))


def canonicalize_static_ground_altitude(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    task_static_ground_z_world: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Normalize replay AGL/reward to the versioned static task floor.

    New recordings declare and already satisfy the exact static-ground
    contract.  Legacy warehouse recordings are repaired in memory: the old
    closest-hit ray saw either the z=0 floor, the raised z=0.1 spawn support,
    or (on audited Human-contact rows) a dynamic obstacle.  Local/world Z was
    recorded independently and therefore provides the exact intended AGL.
    The immutable replay files are never edited.
    """
    ground_z = float(task_static_ground_z_world)
    if not math.isfinite(ground_z):
        raise ValueError("task_static_ground_z_world must be finite")
    output = {key: np.asarray(value) for key, value in arrays.items()}
    ego = np.asarray(output.get("ego_state"), np.float32)
    if ego.ndim != 2 or ego.shape[1] != 14 or not np.isfinite(ego).all():
        raise ValueError("ego_state must be finite [T,14] for AGL canonicalization")
    length = int(ego.shape[0])
    try:
        origin_z = float(np.asarray(
            metadata["ego_reference_origin_xyz"], np.float64).reshape(3)[2])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "ego_reference_origin_xyz is required for static-ground AGL"
        ) from exc
    if not math.isfinite(origin_z):
        raise ValueError("ego_reference_origin_xyz Z must be finite")

    world_z = origin_z + ego[:, 2].astype(np.float64)
    raw_agl = ego[:, 9].astype(np.float64)
    if not np.isfinite(raw_agl).all() or bool((raw_agl < 0.0).any()):
        raise ValueError("recorded AGL must be finite and non-negative")
    canonical_agl = world_z - ground_z
    if bool((canonical_agl < -1.0e-3).any()):
        raise ValueError("replay vehicle center lies below the static task floor")

    episode = metadata.get("episode") or {}
    declared_contract = episode.get("altitude_agl_contract")
    tolerance = 1.0e-3
    inferred_hit_surface = world_z - raw_agl
    static_floor_rows = np.isclose(
        inferred_hit_surface, ground_z, rtol=0.0, atol=tolerance)
    raised_support_rows = np.isclose(
        inferred_hit_surface,
        ground_z + LEGACY_RAISED_SUPPORT_TOP_OFFSET_M,
        rtol=0.0,
        atol=tolerance,
    )

    if declared_contract is not None:
        if declared_contract != STATIC_GROUND_AGL_CONTRACT:
            raise ValueError(
                f"unsupported altitude_agl_contract {declared_contract!r}")
        try:
            declared_ground = float(episode["static_ground_z_world"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "declared static-ground AGL lacks static_ground_z_world"
            ) from exc
        if not math.isclose(
            declared_ground, ground_z, rel_tol=0.0, abs_tol=1.0e-6,
        ):
            raise ValueError(
                "recorded and configured static_ground_z_world differ")
        if episode.get("altitude_agl_dynamic_obstacles_excluded") is not True:
            raise ValueError(
                "static-ground AGL must explicitly exclude dynamic obstacles")
        mismatch = np.abs(raw_agl - canonical_agl)
        if bool((mismatch > tolerance).any()):
            raise ValueError(
                "declared static-ground AGL disagrees with recorded world Z")
        mode = "declared_static_ground"
        non_task_surface_rows = np.zeros(length, np.bool_)
    else:
        recognized = static_floor_rows | raised_support_rows
        if length == 0 or float(recognized.mean()) < 0.50:
            raise ValueError(
                "legacy AGL cannot be identified as the audited warehouse "
                "floor/raised-support recording contract")
        # A downward closest-hit ray may report a surface above the task floor
        # but cannot honestly report one above its own origin/vehicle center.
        if bool((inferred_hit_surface < ground_z - tolerance).any()) or bool(
            (inferred_hit_surface > world_z + tolerance).any()
        ):
            raise ValueError("legacy AGL ray-hit geometry is physically invalid")
        mode = "legacy_raised_support_and_closest_hit_repaired"
        non_task_surface_rows = ~recognized

    correction = canonical_agl - raw_agl
    corrected_ego = ego.copy()
    corrected_ego[:, 9] = canonical_agl.astype(np.float32)
    output["ego_state"] = corrected_ego

    # The factual reward head is diagnostic for Actor optimization, but its
    # scalar and component targets must still describe the corrected state.
    # Recompute only the AGL-derived height component and preserve every other
    # recorded component exactly.
    component = np.asarray(output.get("reward_components"), np.float32)
    reward = np.asarray(output.get("reward"), np.float32)
    times = np.asarray(output.get("simulation_time_s"), np.float64).reshape(-1)
    if (
        component.shape != (length, len(REWARD_COMPONENT_KEYS))
        or reward.reshape(-1).size != length
        or times.size != length
        or not np.isfinite(times).all()
    ):
        raise ValueError("reward/time arrays are invalid for AGL canonicalization")
    reward_cfg = metadata.get("reward_config") or {}
    try:
        cruise = float(reward_cfg["cruise_height_m"])
        tolerance_m = float(reward_cfg["height_tolerance_m"])
        scale_m = float(reward_cfg["height_scale_m"])
        weight_per_s = float(reward_cfg["height_weight_per_sec"])
        term_clip = float(reward_cfg["smooth_term_clip"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("height reward contract is incomplete") from exc
    if not all(math.isfinite(value) for value in (
        cruise, tolerance_m, scale_m, weight_per_s, term_clip,
    )) or min(tolerance_m, weight_per_s) < 0.0 or min(scale_m, term_clip) <= 0.0:
        raise ValueError("height reward contract is invalid")
    dt = np.zeros(length, np.float64)
    if length > 1:
        delta = np.diff(times)
        if bool((delta < -1.0e-9).any()):
            raise ValueError("simulation_time_s must be monotonic")
        dt[1:] = np.maximum(delta, 0.0)
    excess = np.maximum(np.abs(canonical_agl - cruise) - tolerance_m, 0.0)
    height_term = np.minimum(np.square(excess / scale_m), term_clip)
    corrected_height = (-weight_per_s * dt * height_term).astype(np.float32)
    height_index = REWARD_COMPONENT_KEYS.index("height")
    corrected_component = component.copy()
    old_height = corrected_component[:, height_index].copy()
    corrected_component[:, height_index] = corrected_height
    corrected_reward = reward.reshape(-1).copy()
    corrected_reward += corrected_height - old_height
    output["reward_components"] = corrected_component
    output["reward"] = corrected_reward.reshape(reward.shape).astype(np.float32)

    status = {
        "contract": STATIC_GROUND_AGL_CONTRACT,
        "mode": mode,
        "task_static_ground_z_world": ground_z,
        "row_count": length,
        "static_floor_row_count": int(static_floor_rows.sum()),
        "legacy_raised_support_row_count": int(raised_support_rows.sum()),
        "non_task_surface_hit_row_count": int(non_task_surface_rows.sum()),
        "corrected_row_count": int((np.abs(correction) > 1.0e-6).sum()),
        "maximum_abs_agl_correction_m": (
            float(np.max(np.abs(correction))) if length else 0.0),
        "height_reward_corrected_row_count": int(
            (np.abs(corrected_height - old_height) > 1.0e-7).sum()),
    }
    return output, status


def resolve_collision_human_id(
    metadata: Mapping[str, Any], summary: Mapping[str, Any],
) -> int | None:
    """Resolve a PhysX collider name to the recorder's numeric Human ID.

    Isaac names characters ``person1``, ``person2``, ... while privileged
    arrays deliberately store their zero-based semantic IDs ``0, 1, ...``.
    Parsing the decimal suffix as the ID shifts every contact to the next
    person and makes the last person appear absent.  Resolve through the
    episode's explicit ``[{id, name}]`` table instead; an unresolved or
    ambiguous contact remains unknown and must be handled by the Event hidden
    cause rather than guessed.
    """
    if str(summary.get("termination_reason", "")) != "human_collision":
        return None
    details = summary.get("event_details")
    contact_name = str(
        details.get("pedestrian_id", "")
        if isinstance(details, Mapping) else "")
    episode = metadata.get("episode")
    people = episode.get("people", ()) if isinstance(episode, Mapping) else ()
    matches: list[int] = []
    for person in people if isinstance(people, Sequence) else ():
        if not isinstance(person, Mapping):
            continue
        if str(person.get("name", "")) != contact_name:
            continue
        try:
            identity = int(person["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if identity >= 0:
            matches.append(identity)
    if len(matches) != 1:
        return None
    return matches[0]


def clean_task_geometry_status(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect deployable task geometry without treating sentinels as data.

    Older compact-v3 recordings are still readable for representation and
    factual Human-event training.  They are not valid posterior starts for
    the clean Replay Actor/Critic objective unless the recorder explicitly
    wrote this versioned contract.
    """
    episode = metadata.get("episode") or {}
    reasons: list[str] = []
    geometry_contract = episode.get("factorized_task_geometry_contract")
    if geometry_contract not in (
        LEGACY_TASK_GEOMETRY_CONTRACT,
        CLEAN_TASK_GEOMETRY_CONTRACT,
        STATIC_TERMINAL_TASK_GEOMETRY_CONTRACT,
    ):
        reasons.append("factorized_task_geometry_contract")

    bounds_valid = False
    try:
        bounds = np.asarray(
            episode["factorized_flight_bounds_xy"], np.float64).reshape(4)
        bounds_valid = bool(
            np.isfinite(bounds).all()
            and bounds[0] < bounds[1]
            and bounds[2] < bounds[3]
        )
    except (KeyError, TypeError, ValueError):
        bounds_valid = False
    if not bounds_valid:
        reasons.append("factorized_flight_bounds_xy")

    altitude_valid = False
    try:
        maximum_altitude = float(episode["maximum_altitude_m"])
        altitude_valid = math.isfinite(maximum_altitude)
    except (KeyError, TypeError, ValueError):
        altitude_valid = False
    if not altitude_valid:
        reasons.append("maximum_altitude_m")

    static_proxy_valid = False
    try:
        obstacle_values = np.asarray(
            episode["factorized_static_obstacle_aabbs_xy"],
            np.float64,
        ).reshape(-1, 4)
        margin = float(episode["factorized_static_obstacle_preinflation_m"])
        static_proxy_valid = bool(
            np.isfinite(obstacle_values).all()
            and (
                obstacle_values.size == 0
                or bool(np.all(obstacle_values[:, 0] < obstacle_values[:, 2]))
                and bool(np.all(obstacle_values[:, 1] < obstacle_values[:, 3]))
            )
            and math.isfinite(margin)
            and margin >= 0.0
            and episode.get("factorized_static_geometry_type")
            == "pedestrian_2d_conservative_proxy"
            and episode.get("factorized_static_geometry_exact") is False
            and (
                (
                    geometry_contract == LEGACY_TASK_GEOMETRY_CONTRACT
                    and episode.get("factorized_static_geometry_use")
                    == "counterfactual_censor_only"
                )
                or (
                    geometry_contract == CLEAN_TASK_GEOMETRY_CONTRACT
                    and episode.get("factorized_static_geometry_use")
                    == "counterfactual_censor_and_actor_proximity_cost"
                )
                or (
                    geometry_contract
                    == STATIC_TERMINAL_TASK_GEOMETRY_CONTRACT
                    and episode.get("factorized_static_geometry_use")
                    == "analytic_task_terminal_and_actor_proximity_cost"
                )
            )
        )
    except (KeyError, TypeError, ValueError):
        static_proxy_valid = False
    if not static_proxy_valid:
        reasons.append("factorized_static_geometry_proxy")

    return {
        "valid": not reasons,
        "bounds_valid": bounds_valid,
        "maximum_altitude_valid": altitude_valid,
        "static_proxy_valid": static_proxy_valid,
        "static_terminal_valid": bool(
            static_proxy_valid
            and geometry_contract == STATIC_TERMINAL_TASK_GEOMETRY_CONTRACT
            and episode.get("factorized_static_geometry_use")
            == "analytic_task_terminal_and_actor_proximity_cost"),
        "reasons": tuple(reasons),
    }


def require_clean_task_geometry(
    metadata: Mapping[str, Any], *, source: str = "episode",
) -> None:
    """Reject a new-run episode whose analytic task contract is incomplete."""
    status = clean_task_geometry_status(metadata)
    if not bool(status["valid"]):
        raise ValueError(
            f"{source}: clean Replay task geometry is missing or invalid: "
            + ", ".join(str(value) for value in status["reasons"])
        )


def fixed_privileged_collision_geometry(
    joints_world: np.ndarray,
    joint_valid: np.ndarray,
    *,
    capacity: int = PRIVILEGED_GT_COLLISION_MAX_PEOPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad GT collision people to a strict audit-only capacity.

    Silently truncating this channel could remove the person that actually
    collides with the UAV and turn a collision candidate into a false-safe
    target.  Capacity overflow is therefore a hard data-contract failure.
    """
    joints = np.asarray(joints_world, np.float32)
    valid = np.asarray(joint_valid, np.bool_)
    if joints.ndim != 4 or joints.shape[-1] != 3:
        raise ValueError(
            "privileged collision joints must have shape [T,N,J,3], got "
            f"{joints.shape}")
    if valid.shape != joints.shape[:-1]:
        raise ValueError(
            "privileged collision validity must match [T,N,J]: "
            f"{valid.shape} vs {joints.shape[:-1]}")
    people = int(joints.shape[1])
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("privileged GT collision capacity must be positive")
    if people > capacity:
        raise ValueError(
            "privileged GT collision people exceed the lossless capacity: "
            f"{people} > {capacity}; increase the audit capacity instead of "
            "truncating collision targets")
    padded_joints = np.zeros(
        (joints.shape[0], capacity, joints.shape[2], 3), np.float32)
    padded_valid = np.zeros(
        (valid.shape[0], capacity, valid.shape[2]), np.bool_)
    padded_joints[:, :people] = joints
    padded_valid[:, :people] = valid
    return padded_joints, padded_valid


def _episode_to_world_rotation(origin_yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(origin_yaw)), math.sin(float(origin_yaw))
    return np.asarray(((cosine, -sine, 0.0),
                       (sine, cosine, 0.0),
                       (0.0, 0.0, 1.0)), np.float64)


def _swept_segments_intersect_aabbs_xy(
    points_xy: np.ndarray,
    aabbs_xy: np.ndarray,
) -> np.ndarray:
    """Return whether each consecutive point segment touches any XY AABB."""
    points = np.asarray(points_xy, np.float64).reshape(-1, 2)
    boxes = np.asarray(aabbs_xy, np.float64).reshape(-1, 4)
    if points.shape[0] < 2 or boxes.shape[0] == 0:
        return np.zeros(max(points.shape[0] - 1, 0), np.bool_)

    start = points[:-1, None, :]
    delta = points[1:, None, :] - start
    lower = boxes[None, :, :2]
    upper = boxes[None, :, 2:]
    parallel = np.abs(delta) <= np.finfo(np.float64).eps
    parallel_inside = (~parallel) | ((start >= lower) & (start <= upper))
    safe_delta = np.where(parallel, 1.0, delta)
    first = (lower - start) / safe_delta
    second = (upper - start) / safe_delta
    axis_enter = np.where(parallel, -np.inf, np.minimum(first, second))
    axis_exit = np.where(parallel, np.inf, np.maximum(first, second))
    enter = np.maximum(axis_enter.max(axis=-1), 0.0)
    leave = np.minimum(axis_exit.min(axis=-1), 1.0)
    intersects = parallel_inside.all(axis=-1) & (enter <= leave)
    return intersects.any(axis=1)


def static_terminal_promotion_status(
    metadata: Mapping[str, Any],
    ego_state: np.ndarray,
    is_last: np.ndarray,
) -> dict[str, Any]:
    """Prove that old v2 geometry can use the current v3 terminal MDP.

    The v2 recorder serialized the same conservative world-frame AABBs but
    labelled them as proximity/counterfactual geometry only. Current Isaac
    ends an episode on entry into those AABBs. Promotion is therefore safe
    only when every recorded non-final state is outside every AABB *and* no
    non-terminal transition sweeps through an AABB between sampled states.
    A v3 final transition is allowed to enter because its destination is the
    terminal state produced by that exact rule.
    """
    geometry = clean_task_geometry_status(metadata)
    episode = metadata.get("episode")
    episode = episode if isinstance(episode, Mapping) else {}
    contract = episode.get("factorized_task_geometry_contract")
    reasons: list[str] = []
    migration = episode.get("factorized_static_geometry_migration")
    migrated_terminal_lag_rows = 0
    if not bool(geometry["static_proxy_valid"]):
        reasons.append("invalid static geometry proxy")
    if contract not in (
        CLEAN_TASK_GEOMETRY_CONTRACT,
        STATIC_TERMINAL_TASK_GEOMETRY_CONTRACT,
    ):
        reasons.append("only v2/v3 task geometry can be promoted")

    ego = np.asarray(ego_state, np.float64)
    last = np.asarray(is_last, np.bool_).reshape(-1)
    if ego.ndim != 2 or ego.shape[-1] != 14 or ego.shape[0] != last.size:
        reasons.append("Ego14/is_last shape mismatch")
    elif not np.isfinite(ego).all():
        reasons.append("Ego14 contains non-finite values")
    else:
        try:
            origin = np.asarray(
                metadata["ego_reference_origin_xyz"], np.float64,
            ).reshape(3)
            origin_yaw = float(metadata["ego_reference_origin_yaw"])
            obstacles = np.asarray(
                episode["factorized_static_obstacle_aabbs_xy"], np.float64,
            ).reshape(-1, 4)
            if not np.isfinite(origin).all() or not math.isfinite(origin_yaw):
                raise ValueError("non-finite episode transform")
            world = (
                np.einsum(
                    "ij,nj->ni", _episode_to_world_rotation(origin_yaw),
                    ego[:, :3],
                )
                + origin[None]
            )
            if obstacles.size:
                inside = (
                    (world[:, None, 0] >= obstacles[None, :, 0])
                    & (world[:, None, 0] <= obstacles[None, :, 2])
                    & (world[:, None, 1] >= obstacles[None, :, 1])
                    & (world[:, None, 1] <= obstacles[None, :, 3])
                ).any(axis=1)
            else:
                inside = np.zeros(last.shape, np.bool_)
            inconsistent = inside & ~last
            if inconsistent.any():
                rows = np.flatnonzero(inconsistent)
                # Before KLT134 entered the analytic scene contract, PhysX
                # reported its three recorded contacts 1--3 control frames
                # after the conservative envelope was entered.  Those are the
                # only historical replay rows in the added AABB.  Accept only
                # a short contiguous terminal suffix; an interior traversal or
                # a successful episode can never pass this proof.
                lag_allowed = (
                    migration
                    == WAREHOUSE_V2_KLT134_STATIC_GEOMETRY_MIGRATION
                    and bool(last[-1])
                    and bool(inside[-1])
                    and rows.size <= 3
                    and int(rows[-1]) == inside.size - 2
                    and bool(np.all(np.diff(rows) == 1))
                )
                if lag_allowed:
                    migrated_terminal_lag_rows += int(rows.size)
                else:
                    reasons.append(
                        "non-terminal Ego state lies inside static terminal "
                        f"AABB at rows {rows[:8].tolist()}"
                    )
            swept = _swept_segments_intersect_aabbs_xy(
                world[:, :2], obstacles)
            inconsistent_swept = swept & ~last[1:]
            if inconsistent_swept.any():
                rows = np.flatnonzero(inconsistent_swept)
                lag_allowed = (
                    migration
                    == WAREHOUSE_V2_KLT134_STATIC_GEOMETRY_MIGRATION
                    and bool(last[-1])
                    and bool(inside[-1])
                    and rows.size <= 4
                    and int(rows[-1]) == swept.size - 2
                    and bool(np.all(np.diff(rows) == 1))
                )
                if lag_allowed:
                    migrated_terminal_lag_rows += int(rows.size)
                else:
                    reasons.append(
                        "non-terminal Ego transition crosses static terminal "
                        f"AABB at source rows {rows[:8].tolist()}"
                    )
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"invalid world transform/static AABBs: {error}")

    return {
        "valid": not reasons,
        "already_v3": bool(geometry["static_terminal_valid"]),
        "migrated_terminal_lag_rows": int(migrated_terminal_lag_rows),
        "reasons": tuple(reasons),
    }


def _match_detected_slots_to_gt(
    measured_root_body: np.ndarray,
    measured_root_valid: np.ndarray,
    human_ids: np.ndarray,
    ego_state: np.ndarray,
    gt_ids: np.ndarray,
    gt_mask: np.ndarray,
    gt_pelvis_world: np.ndarray,
    gt_pelvis_valid: np.ndarray,
    *,
    origin_xyz: np.ndarray,
    origin_yaw: float,
) -> dict[str, np.ndarray]:
    """Match perception tracks to simulator people without leaking GT input.

    Assignment is one-to-one per frame.  A small continuity preference keeps
    an already matched detector track on the same simulator person, while the
    validity gate is estimated robustly from the episode's own synchronous
    pelvis residuals and capped by a physical one-metre sanity bound.
    """
    if linear_sum_assignment is None:
        raise RuntimeError("GT matching requires scipy.optimize")
    length, slots = human_ids.shape
    output_id = np.full((length, slots), -1, np.int64)
    output_valid = np.zeros((length, slots), np.bool_)
    output_error = np.zeros((length, slots), np.float32)
    output_confidence = np.zeros((length, slots), np.float32)
    output_pelvis_body = np.zeros((length, slots, 3), np.float32)
    output_pelvis_episode = np.zeros((length, slots, 3), np.float32)

    world_from_episode = _episode_to_world_rotation(origin_yaw)
    detected_world = np.zeros((length, slots, 3), np.float64)
    provisional_distances: list[float] = []
    for frame in range(length):
        rotation = _body_to_episode_rotation(ego_state[frame]).astype(np.float64)
        episode = ego_state[frame, :3] + np.einsum(
            "ij,nj->ni", rotation, measured_root_body[frame, :, :3])
        detected_world[frame] = origin_xyz + np.einsum(
            "ij,nj->ni", world_from_episode, episode)
        detected = np.flatnonzero(measured_root_valid[frame] & (human_ids[frame] >= 0))
        truth = np.flatnonzero(
            gt_mask[frame] & gt_pelvis_valid[frame] & (gt_ids[frame] >= 0))
        if detected.size and truth.size:
            distance = np.linalg.norm(
                detected_world[frame, detected][:, None, :]
                - gt_pelvis_world[frame, truth][None, :, :], axis=-1)
            provisional_distances.extend(distance.min(axis=1).tolist())

    if provisional_distances:
        values = np.asarray(provisional_distances, np.float64)
        centre = float(np.median(values))
        mad = float(np.median(np.abs(values - centre))) * 1.4826
        gate = float(np.clip(
            centre + 6.0 * max(mad, 0.01),
            GT_MATCH_MIN_GATE_M, GT_MATCH_MAX_GATE_M))
        ambiguity_margin = float(np.clip(2.0 * max(mad, 0.01), 0.05, 0.25))
    else:
        gate, ambiguity_margin = GT_MATCH_MIN_GATE_M, 0.10

    previous_by_track: dict[int, int] = {}
    for frame in range(length):
        detected = np.flatnonzero(measured_root_valid[frame] & (human_ids[frame] >= 0))
        truth = np.flatnonzero(
            gt_mask[frame] & gt_pelvis_valid[frame] & (gt_ids[frame] >= 0))
        if not detected.size or not truth.size:
            continue
        raw_cost = np.linalg.norm(
            detected_world[frame, detected][:, None, :]
            - gt_pelvis_world[frame, truth][None, :, :], axis=-1)
        assignment_cost = raw_cost.copy()
        for row, slot in enumerate(detected):
            prior = previous_by_track.get(int(human_ids[frame, slot]))
            if prior is None:
                continue
            assignment_cost[row, gt_ids[frame, truth] == prior] -= min(
                0.15, 0.25 * gate)
        rows, columns = linear_sum_assignment(assignment_cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            slot, gt_slot = int(detected[row]), int(truth[column])
            error = float(raw_cost[row, column])
            alternatives = np.delete(raw_cost[row], column)
            separation = (
                float(alternatives.min() - error) if alternatives.size
                else float("inf"))
            track_id = int(human_ids[frame, slot])
            gt_id = int(gt_ids[frame, gt_slot])
            continuity = previous_by_track.get(track_id) == gt_id
            valid = error <= gate and (
                separation >= ambiguity_margin or continuity)
            output_error[frame, slot] = error
            output_confidence[frame, slot] = float(np.clip(
                (gate - error) / max(gate, 1.0e-6), 0.0, 1.0))
            if not valid:
                continue
            output_id[frame, slot] = gt_id
            output_valid[frame, slot] = True
            previous_by_track[track_id] = gt_id
            pelvis_world = gt_pelvis_world[frame, gt_slot].astype(np.float64)
            pelvis_episode = np.einsum(
                "ij,j->i", world_from_episode.T, pelvis_world - origin_xyz)
            output_pelvis_episode[frame, slot] = pelvis_episode
            rotation = _body_to_episode_rotation(ego_state[frame]).astype(np.float64)
            output_pelvis_body[frame, slot] = np.einsum(
                "ij,j->i", rotation.T,
                pelvis_episode - ego_state[frame, :3])
    return {
        "human_gt_id": output_id,
        "human_gt_match_valid": output_valid,
        "human_gt_match_error_m": output_error,
        "human_gt_match_confidence": output_confidence,
        "human_gt_pelvis_body": output_pelvis_body,
        "human_gt_pelvis_episode": output_pelvis_episode,
        "human_gt_match_gate_m": np.full(
            (length, 1), gate, np.float32),
    }


def _carry_gt_identity_through_causal_track(
    human_ids: np.ndarray,
    human_mask: np.ndarray,
    human_is_first: np.ndarray,
    direct_gt_id: np.ndarray,
    direct_gt_valid: np.ndarray,
    simulation_time_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Carry privileged target identity through an Actor-visible track hold.

    The carried value is a training-only correspondence, never an observation.
    A detector track that was matched while measured remains the same physical
    person during its causal prediction/empty-observation hold. Clearing on
    ``human_is_first`` prevents a recycled slot or tracker identity from
    inheriting a previous person's simulator ID. Direct matches own a GT ID
    before carried rows are assigned, preserving one-to-one attribution.
    """
    track_id = np.asarray(human_ids, np.int64)
    mask = np.asarray(human_mask, np.bool_)
    first = np.asarray(human_is_first, np.bool_)
    direct_id = np.asarray(direct_gt_id, np.int64)
    direct_valid = np.asarray(direct_gt_valid, np.bool_)
    times = np.asarray(simulation_time_s, np.float64).reshape(-1)
    if not (
        track_id.shape == mask.shape == first.shape
        == direct_id.shape == direct_valid.shape
        and track_id.shape[0] == times.size
    ):
        raise ValueError("GT identity carry tensors have inconsistent shapes")
    identity_id = np.full_like(direct_id, -1)
    identity_valid = np.zeros_like(direct_valid)
    identity_age_s = np.zeros(direct_id.shape, np.float32)
    known: dict[int, tuple[int, float]] = {}
    for frame in range(track_id.shape[0]):
        for slot in range(track_id.shape[1]):
            detector_id = int(track_id[frame, slot])
            if detector_id >= 0 and bool(first[frame, slot]):
                known.pop(detector_id, None)
        used_gt: set[int] = set()
        for slot in range(track_id.shape[1]):
            detector_id = int(track_id[frame, slot])
            if (
                not bool(mask[frame, slot])
                or detector_id < 0
                or not bool(direct_valid[frame, slot])
            ):
                continue
            simulator_id = int(direct_id[frame, slot])
            if simulator_id < 0 or simulator_id in used_gt:
                continue
            identity_id[frame, slot] = simulator_id
            identity_valid[frame, slot] = True
            used_gt.add(simulator_id)
            known[detector_id] = (simulator_id, float(times[frame]))
        for slot in range(track_id.shape[1]):
            detector_id = int(track_id[frame, slot])
            if (
                identity_valid[frame, slot]
                or not bool(mask[frame, slot])
                or detector_id < 0
            ):
                continue
            prior = known.get(detector_id)
            if prior is None or prior[0] in used_gt:
                continue
            simulator_id, matched_at = prior
            identity_id[frame, slot] = simulator_id
            identity_valid[frame, slot] = True
            identity_age_s[frame, slot] = max(
                0.0, float(times[frame]) - matched_at)
            used_gt.add(simulator_id)
    return {
        "human_gt_identity_id": identity_id,
        "human_gt_identity_valid": identity_valid,
        "human_gt_identity_age_s": identity_age_s,
    }


def derive_one_step_transition_targets(
    termination_code: np.ndarray,
    human_clearance_m: np.ndarray | None = None,
    human_clearance_valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Derive v6.3 source-aligned one-step event and clearance labels.

    Recorder row ``k`` stores the action applied from observation ``k-1`` to
    observation ``k``. Therefore source row ``t`` is supervised by the event
    and privileged clearance recorded at row ``t+1``. Truncations and invalid
    controller stops are censored rather than forced into an event class.
    """
    termination = np.asarray(termination_code, np.uint8).reshape(-1)
    length = int(termination.size)
    event = np.zeros(length, np.int64)
    event_valid = np.zeros(length, np.bool_)
    event_censored = np.zeros(length, np.bool_)
    non_goal_event = np.zeros(length, np.int64)
    for source in range(max(0, length - 1)):
        code = int(termination[source + 1])
        if code in _TERMINATION_TO_TRANSITION_EVENT:
            event[source] = _TERMINATION_TO_TRANSITION_EVENT[code]
            event_valid[source] = True
            non_goal_event[source] = _TERMINATION_TO_NON_GOAL_EVENT[code]
        elif code in _CENSORED_TERMINATION_CODES or code == 255:
            event_censored[source] = True
        else:
            # Unknown recorder codes are invalid/censored by construction. A
            # future schema extension must explicitly map them before use.
            event_censored[source] = True

    next_clearance = np.zeros(length, np.float32)
    next_clearance_valid = np.zeros(length, np.bool_)
    if human_clearance_m is not None and human_clearance_valid is not None:
        clearance = np.asarray(human_clearance_m, np.float32).reshape(-1)
        valid = np.asarray(human_clearance_valid, np.bool_).reshape(-1)
        if clearance.size != length or valid.size != length:
            raise ValueError("one-step clearance arrays must match termination length")
        if length > 1:
            next_clearance[:-1] = clearance[1:]
            next_clearance_valid[:-1] = valid[1:] & np.isfinite(clearance[1:])

    return {
        "transition_event_target": event[:, None],
        "transition_event_valid": event_valid[:, None],
        "transition_event_censored": event_censored[:, None],
        "non_goal_transition_event_target": non_goal_event[:, None],
        "next_human_clearance_m": next_clearance[:, None],
        "next_human_clearance_valid": next_clearance_valid[:, None],
    }


@dataclass(frozen=True)
class CompactEpisode:
    name: str
    directory: Path
    chunks: tuple[Path, ...]
    length: int
    metadata: Mapping[str, Any]
    summary: Mapping[str, Any]
    # Source row t is paired with the destination-row code/action at t+1.
    # These compact tuples are populated while discovery already has each NPZ
    # open, avoiding a second full replay read for event-prior construction.
    event_source_code: tuple[int, ...] = ()
    event_source_valid: tuple[bool, ...] = ()
    # True only when this immutable episode was recorded with the audited
    # legacy nine-AABB warehouse scene and upgraded in memory to KLT134.  It
    # is deliberately distinct from merely carrying the current ten-AABB
    # scene so future malformed episodes can never enter a legacy quarantine.
    static_geometry_migration_applied: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _migrate_fixed_warehouse_static_geometry(
    metadata: dict[str, Any], *, source: Path,
) -> bool:
    """Apply the hash-gated KLT134 scene correction without editing replay.

    Historical compact episodes remain immutable on disk.  Every dataset
    view, including incremental live refreshes, materializes the same corrected
    ten-AABB scene in memory.  A current ``warehouse_v2`` episode is
    fail-closed if it advertises some third geometry identity; older imported
    episodes without the template tag are upgraded only when their scene hash
    exactly matches the audited legacy identity.
    """
    episode = metadata.get("episode")
    if not isinstance(episode, dict):
        return False
    required = (
        "factorized_flight_bounds_xy",
        "factorized_static_obstacle_aabbs_xy",
        "maximum_altitude_m",
    )
    if any(name not in episode for name in required):
        return False
    template = episode.get("crowd_template_version")
    try:
        obstacles, applied = migrate_warehouse_v2_klt134_static_geometry(
            episode["factorized_flight_bounds_xy"],
            episode["factorized_static_obstacle_aabbs_xy"],
            maximum_altitude_m=float(episode["maximum_altitude_m"]),
        )
    except (TypeError, ValueError) as error:
        if template == "warehouse_v2":
            raise ValueError(
                f"{source}: warehouse_v2 fixed geometry is outside the "
                "audited KLT134 migration") from error
        return False
    episode["factorized_static_obstacle_aabbs_xy"] = obstacles.tolist()
    episode["factorized_static_geometry_migration"] = (
        WAREHOUSE_V2_KLT134_STATIC_GEOMETRY_MIGRATION)
    return bool(applied)


def initial_navigation_state_status(
    metadata: Mapping[str, Any], first_ego_state: np.ndarray,
) -> dict[str, Any]:
    """Audit whether an episode begins from the actual navigation reset set.

    The recorder reference origin is established immediately before the first
    observation. Combining it with the first local Ego position recovers the
    measured world pose. Legacy replay predates explicit reset-gate metadata,
    so the documented v1 gate values above are its only accepted defaults.
    """
    episode = metadata.get("episode")
    episode = episode if isinstance(episode, Mapping) else {}
    ego = np.asarray(first_ego_state, np.float64).reshape(-1)
    reasons: list[str] = []
    if ego.shape != (14,) or not np.isfinite(ego).all():
        return {"valid": False, "reasons": ("first Ego14 is invalid",)}
    try:
        origin = np.asarray(
            metadata["ego_reference_origin_xyz"], np.float64).reshape(3)
        spawn = np.asarray(
            episode["drone_spawn_world"], np.float64).reshape(3)
    except (KeyError, TypeError, ValueError):
        return {
            "valid": False,
            "reasons": ("spawn/reference origin contract is missing",),
        }
    actual = origin + ego[:3]
    if not np.isfinite(origin).all() or not np.isfinite(spawn).all() \
            or not np.isfinite(actual).all():
        return {"valid": False, "reasons": ("initial world pose is invalid",)}

    def limit(name: str, default: float) -> float:
        try:
            value = float(episode.get(name, default))
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0.0:
            reasons.append(f"{name} is invalid")
            return default
        return value

    xy_limit = limit(
        "pre_navigation_max_spawn_xy_error_m",
        PRE_NAVIGATION_MAX_SPAWN_XY_ERROR_M)
    altitude_limit = limit(
        "pre_navigation_max_altitude_error_m",
        PRE_NAVIGATION_MAX_ALTITUDE_ERROR_M)
    speed_limit = limit(
        "pre_navigation_max_speed_mps",
        PRE_NAVIGATION_MAX_SPEED_MPS)
    tilt_limit = limit(
        "pre_navigation_max_tilt_deg",
        PRE_NAVIGATION_MAX_TILT_DEG)
    try:
        expected_altitude = float(episode.get(
            "pre_navigation_expected_altitude_m",
            metadata.get("reward_config", {}).get(
                "cruise_height_m", PRE_NAVIGATION_EXPECTED_ALTITUDE_M),
        ))
    except (TypeError, ValueError):
        expected_altitude = float("nan")
    if not math.isfinite(expected_altitude):
        reasons.append("pre_navigation_expected_altitude_m is invalid")
        expected_altitude = PRE_NAVIGATION_EXPECTED_ALTITUDE_M

    xy_error = float(np.linalg.norm(actual[:2] - spawn[:2]))
    altitude_error = abs(float(actual[2]) - expected_altitude)
    speed = float(np.linalg.norm(ego[3:6]))
    tilt_deg = float(np.rad2deg(np.max(np.abs(ego[10:12]))))
    if xy_error > xy_limit:
        reasons.append(f"spawn xy error {xy_error:.3f}m exceeds {xy_limit:.3f}m")
    if altitude_error > altitude_limit:
        reasons.append(
            f"altitude error {altitude_error:.3f}m exceeds "
            f"{altitude_limit:.3f}m")
    if speed > speed_limit:
        reasons.append(f"speed {speed:.3f}m/s exceeds {speed_limit:.3f}m/s")
    if tilt_deg > tilt_limit:
        reasons.append(f"tilt {tilt_deg:.3f}deg exceeds {tilt_limit:.3f}deg")
    return {
        "valid": not reasons,
        "reasons": tuple(reasons),
        "actual_world_xyz": actual.astype(np.float32),
        "spawn_xy_error_m": xy_error,
        "altitude_error_m": altitude_error,
        "speed_mps": speed,
        "tilt_deg": tilt_deg,
    }


def compact_episode_initial_state_status(
    episode: CompactEpisode,
) -> dict[str, Any]:
    """Read only the first immutable row needed by the reset audit."""
    with np.load(episode.chunks[0], allow_pickle=False) as arrays:
        ego = np.asarray(arrays["ego_state"], np.float32)
        if ego.ndim != 2 or ego.shape[0] == 0:
            return {"valid": False, "reasons": ("first chunk lacks Ego14",)}
        first = ego[0]
    return initial_navigation_state_status(episode.metadata, first)


def compact_episode_matches_recording_run(
    metadata: Mapping[str, Any], directory: Path, recording_run_id: str,
) -> bool:
    """Accept native episodes and explicitly retargeted replay imports."""
    episode_metadata = metadata.get("episode") or {}
    native_match = str(episode_metadata.get(
        "recording_run_id", "")) == str(recording_run_id)
    import_contract = metadata.get("v16_import_provenance") or {}
    import_sidecar = directory / "v16_import_provenance.json"
    if import_sidecar.is_file():
        import_contract = _read_json(import_sidecar)
    imported_match = (
        import_contract.get("schema") in {
            "omninxt.v16-replay-import-provenance.v1",
            "omninxt.pure-dreamer-replay-import-provenance.v2",
        }
        and str(import_contract.get("target_run_id", ""))
        == str(recording_run_id)
    )
    return bool(native_match or imported_match)


def discover_compact_episodes(
    root: str | Path,
    *,
    include_incomplete: bool = False,
    recording_run_id: str | None = None,
    include_episode_names: Sequence[str] | None = None,
    migrate_warehouse_klt134_static_geometry: bool = False,
) -> list[CompactEpisode]:
    root = Path(root).expanduser()
    episodes_root = root / "episodes" if (root / "episodes").is_dir() else root
    selected_names = (
        None if include_episode_names is None
        else frozenset(str(name) for name in include_episode_names)
    )
    episodes: list[CompactEpisode] = []
    for directory in sorted(path for path in episodes_root.glob("*") if path.is_dir()):
        # Live replay refreshes pass only newly finalized episode directories.
        # Filter before opening metadata or NPZ chunks so an incremental
        # refresh never re-reads historical transitions.
        if selected_names is not None and directory.name not in selected_names:
            continue
        metadata_path = directory / "metadata.json"
        summary_path = directory / "summary.json"
        chunks = tuple(sorted((directory / "chunks").glob("chunk_*.npz")))
        # Offline training accepts only finalized episodes.  Live Dreamer may
        # consume atomically committed chunks before summary.json exists; the
        # caller must additionally guard the censored risk-target tail.
        if not metadata_path.is_file() or not chunks:
            continue
        metadata = _read_json(metadata_path)
        if recording_run_id is not None:
            if not compact_episode_matches_recording_run(
                metadata, directory, str(recording_run_id),
            ):
                continue
        if summary_path.is_file():
            summary = _read_json(summary_path)
        elif include_incomplete:
            summary = {
                "schema": SCHEMA,
                "episode_id": directory.name,
                "termination_reason": "recording",
                "success": False,
                "incomplete": True,
            }
        else:
            # Abruptly interrupted episodes can contain valid atomic chunks
            # but no summary. They never silently enter offline training.
            continue
        if metadata.get("schema") != SCHEMA:
            continue
        if metadata.get("joint_topology") != "COCO12_BODY":
            raise ValueError(f"{metadata_path}: expected COCO12_BODY")
        if int(metadata.get("recorded_joint_count", -1)) != JOINT_COUNT:
            raise ValueError(f"{metadata_path}: expected {JOINT_COUNT} recorded joints")
        static_geometry_migration_applied = False
        if migrate_warehouse_klt134_static_geometry:
            static_geometry_migration_applied = (
                _migrate_fixed_warehouse_static_geometry(
                    metadata, source=metadata_path))
        length = 0
        previous_frame: int | None = None
        expected_people = int(metadata["stored_people_count"])
        source_codes: list[np.ndarray] = []
        source_action_valid: list[np.ndarray] = []
        source_is_last: list[np.ndarray] = []
        source_skeleton_fresh: list[np.ndarray] = []
        source_human_active: list[np.ndarray] = []
        for chunk in chunks:
            with np.load(chunk, allow_pickle=False) as arrays:
                frame = np.asarray(arrays["frame_index"], np.int64)
                if frame.ndim != 1 or frame.size == 0:
                    raise ValueError(f"{chunk}: invalid frame_index")
                if previous_frame is not None and int(frame[0]) != previous_frame + 1:
                    raise ValueError(f"{chunk}: episode frame_index is not contiguous")
                if np.asarray(arrays["human_xyz"]).shape[1] != expected_people:
                    raise ValueError(f"{chunk}: people dimension differs from metadata")
                quality_keys = {
                    "human_joint_measured", "human_joint_predicted"}
                present_quality = quality_keys.intersection(arrays.files)
                if present_quality and present_quality != quality_keys:
                    raise ValueError(
                        f"{chunk}: measured/predicted fields must appear together")
                if present_quality:
                    valid = np.asarray(arrays["human_joint_valid"], np.bool_)
                    measured = np.asarray(
                        arrays["human_joint_measured"], np.bool_)
                    predicted = np.asarray(
                        arrays["human_joint_predicted"], np.bool_)
                    if measured.shape != valid.shape or predicted.shape != valid.shape:
                        raise ValueError(
                            f"{chunk}: skeleton quality field shape mismatch")
                    if np.any(measured & predicted):
                        raise ValueError(
                            f"{chunk}: joint cannot be measured and predicted")
                    if np.any(valid != (measured | predicted)):
                        raise ValueError(
                            f"{chunk}: valid joints need one measurement source")
                previous_frame = int(frame[-1])
                length += int(frame.size)
                source_codes.append(np.asarray(
                    arrays["termination_code"], np.uint8).reshape(-1))
                source_action_valid.append(np.asarray(
                    arrays["action_valid"], np.bool_).reshape(-1))
                source_is_last.append(np.asarray(
                    arrays["is_last"], np.bool_).reshape(-1))
                source_skeleton_fresh.append(np.asarray(
                    arrays.get(
                        "skeleton_fresh",
                        np.ones(frame.size, np.bool_),
                    ),
                    np.bool_,
                ).reshape(-1))
                source_human_active.append(np.any(np.asarray(
                    arrays["human_joint_valid"], np.bool_), axis=(1, 2)))
        termination = np.concatenate(source_codes)
        action_valid = np.concatenate(source_action_valid)
        is_last = np.concatenate(source_is_last)
        skeleton_fresh = np.concatenate(source_skeleton_fresh)
        human_active = np.concatenate(source_human_active)
        # Recorder-stale rows are still usable when the Dataset can construct
        # a causal one/two-row persistence state from a recently observed
        # person. A stale empty row with no such state cannot be treated as a
        # trustworthy human-risk negative.
        source_observation_reliable = skeleton_fresh.copy()
        for source in np.flatnonzero(~skeleton_fresh):
            begin = max(0, int(source) - 2)
            source_observation_reliable[source] = bool(
                np.any(human_active[begin:int(source)]))
        destination_code = termination[1:]
        event_valid = (
            action_valid[1:]
            & ~is_last[:-1]
            & source_observation_reliable[:-1]
            & np.asarray([
                int(code) in _TERMINATION_TO_TRANSITION_EVENT
                for code in destination_code
            ], np.bool_)
        )
        episodes.append(CompactEpisode(
            directory.name, directory, chunks, length, metadata, summary,
            tuple(int(value) for value in destination_code),
            tuple(bool(value) for value in event_valid),
            static_geometry_migration_applied,
        ))
    return episodes


def _compact_episode_ego_and_last(
    episode: CompactEpisode,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the two physical arrays needed by static-terminal auditing."""
    ego_chunks: list[np.ndarray] = []
    last_chunks: list[np.ndarray] = []
    for chunk in episode.chunks:
        with np.load(chunk, allow_pickle=False) as arrays:
            try:
                ego_chunks.append(np.asarray(arrays["ego_state"], np.float32))
                last_chunks.append(np.asarray(arrays["is_last"], np.bool_))
            except KeyError as error:
                raise ValueError(
                    f"{episode.directory}: static-terminal promotion "
                    "requires ego_state and is_last"
                ) from error
    if not ego_chunks:
        raise ValueError(
            f"{episode.directory}: static-terminal promotion has no chunks")
    return (
        np.concatenate(ego_chunks, axis=0),
        np.concatenate(last_chunks, axis=0),
    )


def _legacy_klt134_promotion_status(
    episode: CompactEpisode,
    ego_state: np.ndarray,
    is_last: np.ndarray,
) -> dict[str, Any]:
    """Re-run the terminal proof on the exact pre-migration scene.

    This is the fail-closed proof used before quarantining an immutable legacy
    trajectory.  Removing anything except the one hash-gated, audited KLT134
    suffix is rejected, and the remaining nine-AABB scene must independently
    pass the complete static-terminal audit.
    """
    if not episode.static_geometry_migration_applied:
        return {
            "valid": False,
            "reasons": ("episode was not migrated from the legacy scene",),
        }
    metadata = copy.deepcopy(dict(episode.metadata))
    episode_metadata = metadata.get("episode")
    if not isinstance(episode_metadata, dict):
        return {"valid": False, "reasons": ("episode metadata is missing",)}
    try:
        obstacles = np.asarray(
            episode_metadata["factorized_static_obstacle_aabbs_xy"],
            np.float64,
        ).reshape(-1, 4)
    except (KeyError, TypeError, ValueError):
        return {
            "valid": False,
            "reasons": ("migrated static geometry is malformed",),
        }
    expected = np.asarray(
        WAREHOUSE_V2_KLT134_STATIC_AABB_WORLD, np.float64).reshape(4)
    if (
        obstacles.shape[0] < 1
        or not np.allclose(
            obstacles[-1], expected, rtol=0.0, atol=5.0e-6)
    ):
        return {
            "valid": False,
            "reasons": ("migrated geometry lacks the audited KLT134 suffix",),
        }
    episode_metadata["factorized_static_obstacle_aabbs_xy"] = (
        obstacles[:-1].tolist())
    episode_metadata.pop("factorized_static_geometry_migration", None)
    return static_terminal_promotion_status(metadata, ego_state, is_last)


def _body_to_episode_rotation(ego_state: np.ndarray) -> np.ndarray:
    """Return R_episode_from_body from Ego14 roll/pitch/relative-yaw."""
    roll, pitch = float(ego_state[10]), float(ego_state[11])
    yaw = math.atan2(float(ego_state[12]), float(ego_state[13]))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), np.float32)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), np.float32)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), np.float32)
    return rz @ ry @ rx


def _ego_velocity_full_body(ego_state: np.ndarray) -> np.ndarray:
    """Express Ego14 world velocity in skeleton ``base_link`` axes.

    Ego14 stores horizontal velocity in the yaw-heading frame and vertical
    velocity in the episode/world frame.  Compact-v3 Human velocity is Human
    world velocity expressed in the full roll/pitch/yaw body frame.  TTC and
    slot-priority calculations must therefore use this conversion before
    subtracting the two velocities.
    """
    ego = np.asarray(ego_state, np.float32)
    if ego.shape != (14,):
        raise ValueError("Ego velocity conversion requires Ego14")
    yaw = math.atan2(float(ego[12]), float(ego[13]))
    cy, sy = math.cos(yaw), math.sin(yaw)
    velocity_episode = np.asarray((
        cy * float(ego[3]) - sy * float(ego[4]),
        sy * float(ego[3]) + cy * float(ego[4]),
        float(ego[5]),
    ), np.float32)
    return (_body_to_episode_rotation(ego).T @ velocity_episode).astype(
        np.float32)


def derive_relative_joint_velocity(
    xyz: np.ndarray,
    joint_valid: np.ndarray,
    track_ids: np.ndarray,
    ego_state: np.ndarray,
    simulation_time_s: np.ndarray,
) -> np.ndarray:
    """Diagnostic finite-difference motion expressed in current body axes.

    The previous point is first lifted into the episode frame and then
    transformed into the current body frame. A stationary world point has zero
    human-motion velocity even when the UAV translates or yaws; relative
    position remains represented by the recorded XYZ channels. Production
    model input does not call this helper: it consumes the causal tracker
    velocity recorded with compact-v3 instead.
    """
    xyz = np.asarray(xyz, np.float32)
    valid = np.asarray(joint_valid, np.bool_)
    ids = np.asarray(track_ids, np.int64)
    ego = np.asarray(ego_state, np.float32)
    times = np.asarray(simulation_time_s, np.float64)
    velocity = np.zeros_like(xyz, np.float32)
    previous: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in range(xyz.shape[0]):
        rotation = _body_to_episode_rotation(ego[t])
        ego_position = ego[t, :3]
        dt = 0.0 if t == 0 else float(times[t] - times[t - 1])
        current: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for raw_slot, track_id_value in enumerate(ids[t]):
            track_id = int(track_id_value)
            if track_id < 0:
                continue
            current_episode = ego_position + np.einsum(
                "ij,kj->ki", rotation, xyz[t, raw_slot])
            current_valid = valid[t, raw_slot]
            current[track_id] = (current_episode, current_valid.copy())
            old = previous.get(track_id)
            if old is None or dt <= 1.0e-6:
                continue
            previous_episode, previous_valid = old
            usable = current_valid & previous_valid
            previous_in_current_body = np.einsum(
                "ij,kj->ki", rotation.T, previous_episode - ego_position)
            velocity[t, raw_slot, usable] = (
                xyz[t, raw_slot, usable] - previous_in_current_body[usable]) / dt
        previous = current
    velocity[~valid] = 0.0
    return velocity


def _bounded_joint_velocity_episode(
    current_episode_position: np.ndarray,
    previous_episode_position: np.ndarray,
    current_valid: np.ndarray,
    previous_valid: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal per-joint world motion and the joints that support it.

    This is deliberately a two-frame, past-only reconstruction. Both the
    replay loader and online service call the same helper, so detailed joint
    velocity cannot be present during training but silently replaced by a
    broadcast pelvis velocity at deployment.
    """
    current = np.asarray(current_episode_position, np.float32)
    previous = np.asarray(previous_episode_position, np.float32)
    current_mask = np.asarray(current_valid, np.bool_)
    previous_mask = np.asarray(previous_valid, np.bool_)
    if current.shape != previous.shape or current.shape[-1] != 3:
        raise ValueError("joint position histories must have matching [...,3] shape")
    if current_mask.shape != current.shape[:-1] \
            or previous_mask.shape != current_mask.shape:
        raise ValueError("joint velocity masks must match joint positions")
    usable = current_mask & previous_mask
    velocity = np.zeros_like(current, np.float32)
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 1.0e-6:
        return velocity, np.zeros_like(usable)
    raw = (current - previous) / float(dt_s)
    speed = np.linalg.norm(raw, axis=-1, keepdims=True)
    scale = np.minimum(
        1.0,
        CAUSAL_JOINT_SPEED_LIMIT_MPS / np.maximum(speed, 1.0e-6),
    )
    velocity[usable] = (raw * scale)[usable]
    return velocity, usable


def _risk_scores(
    skeleton: np.ndarray,
    joint_mask: np.ndarray,
    *,
    ego_velocity_body: np.ndarray | None = None,
) -> np.ndarray:
    """Rank people by current and closing risk in one physical frame.

    Skeleton velocity is Human world velocity in ``base_link`` axes, so it is
    not a relative velocity until Ego velocity in the same axes is subtracted.
    """
    if ego_velocity_body is None:
        ego_velocity_body = np.zeros(3, np.float32)
    ego_velocity_body = np.asarray(
        ego_velocity_body, np.float32).reshape(3)
    count = skeleton.shape[0]
    risk = np.zeros(count, np.float32)
    for index in range(count):
        valid = joint_mask[index]
        if not np.any(valid):
            continue
        position = skeleton[index, valid, :3].mean(axis=0)
        velocity = (
            skeleton[index, valid, 3:6].mean(axis=0)
            - ego_velocity_body
        )
        distance = float(np.linalg.norm(position))
        closing_speed = 0.0 if distance < 1e-6 else float(
            -np.dot(position, velocity) / distance)
        ttc = distance / closing_speed if closing_speed > 1e-3 else math.inf
        in_forward_corridor = position[0] > 0.0 and abs(position[1]) < 1.0
        risk[index] = (
            4.0 / (distance + 0.1)
            + (3.0 / (ttc + 0.1) if ttc < 5.0 else 0.0)
            + (2.0 if in_forward_corridor else 0.0)
            + 0.1 * float(skeleton[index, valid, 6].mean())
        )
    return risk


class CompactSkeletonV3Dataset(Dataset):  # type: ignore[misc]
    """Fixed-length factorized-model sequences from chunked v3 episodes."""

    def __init__(self, root: str | Path, *, sequence_length: int = 64,
                 stride: int | None = None, max_people: int = 20,
                 cache_episodes: int = 2,
                 human_cache_episodes: int | None = None,
                 risk_horizon_s: float = DEFAULT_RISK_HORIZON_S,
                 dense_clearance_m: float = DEFAULT_DENSE_CLEARANCE_M,
                 include_incomplete: bool = False,
                 incomplete_tail_guard_frames: int = 0,
                 recording_run_id: str | None = None,
                 include_episode_names: Sequence[str] | None = None,
                 include_crowd_seeds: Sequence[int] | None = None,
                 exclude_crowd_seeds: Sequence[int] = (),
                 allow_event_only_without_windows: bool = False,
                 transition_unique_windows: bool = False,
                 require_clean_task_geometry_contract: bool = False,
                 promote_static_proxy_to_task_terminal: bool = False,
                 migrate_warehouse_klt134_static_geometry: bool = False,
                 exclude_invalid_initial_state: bool = False,
                 allow_empty_dataset: bool = False,
                 task_static_ground_z_world: float | None = None,
                 exclude_termination_reasons: Sequence[str] =
                 DEFAULT_EXCLUDED_TERMINATION_REASONS) -> None:
        self.root = Path(root).expanduser()
        self.sequence_length = int(sequence_length)
        self.stride = int(stride or sequence_length)
        self.max_people = int(max_people)
        self.cache_episodes = max(1, int(cache_episodes))
        self.human_cache_episodes = max(
            1,
            int(cache_episodes if human_cache_episodes is None
                else human_cache_episodes),
        )
        self.risk_horizon_s = float(risk_horizon_s)
        self.dense_clearance_m = float(dense_clearance_m)
        self.include_incomplete = bool(include_incomplete)
        self.incomplete_tail_guard_frames = int(incomplete_tail_guard_frames)
        self.recording_run_id = recording_run_id
        self.include_episode_names = (
            None if include_episode_names is None
            else frozenset(str(name) for name in include_episode_names)
        )
        self.include_crowd_seeds = (
            None if include_crowd_seeds is None
            else frozenset(int(seed) for seed in include_crowd_seeds)
        )
        self.exclude_crowd_seeds = frozenset(
            int(seed) for seed in exclude_crowd_seeds)
        self.allow_event_only_without_windows = bool(
            allow_event_only_without_windows)
        self.transition_unique_windows = bool(transition_unique_windows)
        self.require_clean_task_geometry_contract = bool(
            require_clean_task_geometry_contract)
        self.promote_static_proxy_to_task_terminal = bool(
            promote_static_proxy_to_task_terminal)
        self.migrate_warehouse_klt134_static_geometry = bool(
            migrate_warehouse_klt134_static_geometry)
        self.exclude_invalid_initial_state = bool(
            exclude_invalid_initial_state)
        self.allow_empty_dataset = bool(allow_empty_dataset)
        self.task_static_ground_z_world = (
            None if task_static_ground_z_world is None
            else float(task_static_ground_z_world)
        )
        if (
            self.task_static_ground_z_world is not None
            and not math.isfinite(self.task_static_ground_z_world)
        ):
            raise ValueError("task_static_ground_z_world must be finite")
        if self.sequence_length <= 0 or self.stride <= 0 or self.max_people <= 0:
            raise ValueError("sequence_length, stride and max_people must be positive")
        if self.risk_horizon_s <= 0.0 or self.dense_clearance_m <= 0.0:
            raise ValueError("risk_horizon_s and dense_clearance_m must be positive")
        if self.incomplete_tail_guard_frames < 0:
            raise ValueError("incomplete_tail_guard_frames must be non-negative")
        discovered = discover_compact_episodes(
            self.root,
            include_incomplete=self.include_incomplete,
            recording_run_id=self.recording_run_id,
            include_episode_names=self.include_episode_names,
            migrate_warehouse_klt134_static_geometry=(
                self.migrate_warehouse_klt134_static_geometry),
        )
        if not discovered and not self.allow_empty_dataset:
            raise ValueError(f"No {SCHEMA} episodes found under {self.root}")
        self.excluded_termination_reasons = frozenset(
            str(reason) for reason in exclude_termination_reasons)
        self.excluded_episodes = tuple(
            episode for episode in discovered
            if str(episode.summary.get("termination_reason", ""))
            in self.excluded_termination_reasons)
        eligible = [
            episode for episode in discovered
            if str(episode.summary.get("termination_reason", ""))
            not in self.excluded_termination_reasons
            and (
                self.include_crowd_seeds is None
                or int(episode.metadata.get("episode", {}).get(
                    "crowd_seed", -1)) in self.include_crowd_seeds)
            and int(episode.metadata.get("episode", {}).get(
                "crowd_seed", -1)) not in self.exclude_crowd_seeds]
        initial_status = tuple(
            (episode, compact_episode_initial_state_status(episode))
            for episode in eligible)
        self.invalid_initial_state_episodes = tuple(
            (episode, status)
            for episode, status in initial_status
            if not bool(status["valid"])
        )
        self.episodes = [
            episode for episode, status in initial_status
            if not self.exclude_invalid_initial_state or bool(status["valid"])
        ]
        self.initial_state_status = tuple(
            status for _, status in initial_status
            if not self.exclude_invalid_initial_state or bool(status["valid"])
        )
        if not self.episodes and not self.allow_empty_dataset:
            excluded = ", ".join(sorted(self.excluded_termination_reasons)) or "none"
            raise ValueError(
                f"No eligible {SCHEMA} episodes found under {self.root}; "
                f"excluded termination reasons: {excluded}")
        self.task_geometry_status = tuple(
            clean_task_geometry_status(episode.metadata)
            for episode in self.episodes
        )
        if self.require_clean_task_geometry_contract:
            for episode, status in zip(
                self.episodes, self.task_geometry_status, strict=True,
            ):
                if not bool(status["valid"]):
                    raise ValueError(
                        f"{episode.directory}: clean Replay task geometry is "
                        "missing or invalid: "
                        + ", ".join(str(value) for value in status["reasons"])
                    )
        self.static_terminal_promotion_status: tuple[dict[str, Any], ...] = ()
        self.static_geometry_migration_quarantined_episodes: tuple[
            tuple[CompactEpisode, dict[str, Any]], ...
        ] = ()
        if self.promote_static_proxy_to_task_terminal:
            kept_episodes: list[CompactEpisode] = []
            kept_initial_status: list[dict[str, Any]] = []
            kept_geometry_status: list[dict[str, Any]] = []
            promotion: list[dict[str, Any]] = []
            quarantined: list[tuple[CompactEpisode, dict[str, Any]]] = []
            for episode, initial, geometry in zip(
                self.episodes,
                self.initial_state_status,
                self.task_geometry_status,
                strict=True,
            ):
                ego_state, is_last = _compact_episode_ego_and_last(episode)
                status = static_terminal_promotion_status(
                    episode.metadata, ego_state, is_last)
                if bool(status["valid"]):
                    kept_episodes.append(episode)
                    kept_initial_status.append(initial)
                    kept_geometry_status.append(geometry)
                    promotion.append(status)
                    continue

                # A trajectory may be quarantined only when discovery proved
                # that its immutable metadata was the audited legacy scene
                # and the exact same trajectory passes the full proof after
                # removing solely the newly appended KLT134 AABB. Current
                # ten-AABB episodes remain fail-closed.
                legacy_status = _legacy_klt134_promotion_status(
                    episode, ego_state, is_last)
                if bool(legacy_status["valid"]):
                    quarantine_status = dict(status)
                    quarantine_status.update({
                        "quarantine_reason": (
                            "legacy trajectory conflicts only with the "
                            "new KLT134 analytic terminal"),
                        "legacy_scene_valid": True,
                        "legacy_scene_reasons": tuple(
                            legacy_status.get("reasons", ())),
                    })
                    quarantined.append((episode, quarantine_status))
                    continue
                raise ValueError(
                    f"{episode.directory}: replay cannot be promoted to "
                    "the current analytic static-terminal task: "
                    + "; ".join(str(value) for value in status["reasons"])
                )
            self.episodes = kept_episodes
            self.initial_state_status = tuple(kept_initial_status)
            self.task_geometry_status = tuple(kept_geometry_status)
            self.static_terminal_promotion_status = tuple(promotion)
            self.static_geometry_migration_quarantined_episodes = tuple(
                quarantined)
            if not self.episodes and not self.allow_empty_dataset:
                raise ValueError(
                    "No replay episode remains after the audited KLT134 "
                    "static-geometry migration quarantine")
        self.windows: list[tuple[int, int]] = []
        self.usable_episode_lengths: list[int] = []
        for episode_index, episode in enumerate(self.episodes):
            usable_length = episode.length
            if bool(episode.summary.get("incomplete", False)):
                usable_length -= self.incomplete_tail_guard_frames
            usable_length = max(0, int(usable_length))
            self.usable_episode_lengths.append(usable_length)
            if usable_length <= 0:
                continue
            if self.transition_unique_windows:
                # Partition physical transitions, not rows.  Adjacent windows
                # share exactly their boundary row, so every transition is
                # supervised once while terminal transitions remain present.
                # A short prefix is left-padded and encoded by a negative
                # start whose magnitude is its real row count.
                if usable_length <= self.sequence_length:
                    if usable_length == self.sequence_length:
                        self.windows.append((episode_index, 0))
                    else:
                        self.windows.append((episode_index, -usable_length))
                    continue
                transition_span = self.sequence_length - 1
                prefix_transitions = (usable_length - 1) % transition_span
                if prefix_transitions:
                    prefix_length = prefix_transitions + 1
                    self.windows.append((episode_index, -prefix_length))
                    first_full_start = prefix_transitions
                else:
                    first_full_start = 0
                self.windows.extend(
                    (episode_index, start)
                    for start in range(
                        first_full_start,
                        usable_length - self.sequence_length + 1,
                        transition_span,
                    )
                )
                continue
            if usable_length < self.sequence_length:
                continue
            starts = list(range(
                0, usable_length - self.sequence_length + 1, self.stride))
            final_start = usable_length - self.sequence_length
            if starts[-1] != final_start:
                starts.append(final_start)
            self.windows.extend((episode_index, start) for start in starts)
        if (
            not self.windows
            and not self.allow_event_only_without_windows
            and not self.allow_empty_dataset
        ):
            raise ValueError("No episode is long enough for the requested sequence_length")
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.altitude_contract_status: dict[int, dict[str, Any]] = {}
        self._risk_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._human_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def preload_human_targets(self) -> dict[str, int]:
        """Materialize the expensive causal Human reconstruction before fork.

        Offline training uses shuffled windows.  With an episode cache of two,
        that access pattern rebuilt the full causal tracker state for nearly
        every sample.  Preloading in the parent process lets Linux DataLoader
        workers share the immutable NumPy pages through copy-on-write.  Raw
        episode/risk caches remain small and are not retained by this method.
        """
        if self.human_cache_episodes < len(self.episodes):
            raise ValueError(
                "human_cache_episodes must cover every episode before "
                "preloading")
        for episode_index in range(len(self.episodes)):
            self._episode_human_targets(episode_index)
        byte_count = sum(
            int(value.nbytes)
            for episode in self._human_cache.values()
            for value in episode.values()
            if isinstance(value, np.ndarray)
        )
        return {
            "episodes": len(self._human_cache),
            "bytes": byte_count,
        }

    def _episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached
        episode = self.episodes[episode_index]
        chunks: list[dict[str, np.ndarray]] = []
        for path in episode.chunks:
            with np.load(path, allow_pickle=False) as loaded:
                chunks.append({key: np.asarray(loaded[key]) for key in loaded.files})
        keys = set(chunks[0])
        if any(set(chunk) != keys for chunk in chunks[1:]):
            raise ValueError(f"{episode.directory}: chunk fields differ")
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
                  for key in sorted(keys)}
        if self.task_static_ground_z_world is not None:
            arrays, altitude_status = canonicalize_static_ground_altitude(
                arrays,
                episode.metadata,
                task_static_ground_z_world=self.task_static_ground_z_world,
            )
            self.altitude_contract_status[episode_index] = altitude_status
        self._cache[episode_index] = arrays
        self._cache.move_to_end(episode_index)
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return arrays

    def _episode_human_targets(self, episode_index: int) -> dict[str, np.ndarray]:
        """Build slot-stable Human inputs and measured-only targets once per episode.

        Slot assignment, track age, measurement age, and finite differences must
        be computed before sequence slicing.  Reinitialising any of them at a
        64-frame window boundary changes the supervision target without a real
        observation event and was a major source of Human-RSSM label noise.
        """
        cached = self._human_cache.get(episode_index)
        if cached is not None:
            self._human_cache.move_to_end(episode_index)
            return cached

        arrays = self._episode_arrays(episode_index)
        xyz = np.asarray(arrays["human_xyz"], np.float32)
        confidence = np.asarray(arrays["human_confidence"], np.float32)
        joint_valid = np.asarray(arrays["human_joint_valid"], np.bool_)
        track_ids = np.asarray(arrays["human_track_id"], np.int64)
        length, raw_people, joint_count = joint_valid.shape
        skeleton_fresh = np.asarray(
            arrays.get("skeleton_fresh", np.ones(length, np.bool_)),
            np.bool_,
        ).reshape(-1)
        if skeleton_fresh.size != length:
            raise ValueError("skeleton_fresh must match episode length")
        measured_raw = np.asarray(
            arrays.get("human_joint_measured", joint_valid), np.bool_)
        predicted_raw = np.asarray(
            arrays.get("human_joint_predicted", joint_valid & ~measured_raw),
            np.bool_,
        )
        measured_raw &= joint_valid
        predicted_raw &= joint_valid & ~measured_raw
        xyz, joint_valid, topology_completed = (
            complete_coco12_topology_numpy(
                xyz, joint_valid, measured_raw))
        confidence = confidence.copy()
        confidence[topology_completed] = 0.0
        measured_raw &= ~topology_completed
        predicted_raw = (
            predicted_raw | topology_completed
        ) & joint_valid & ~measured_raw
        xyz, geometry_adjusted = sanitize_coco12_geometry_numpy(
            xyz, joint_valid)
        # A projected coordinate remains a causal physical obstacle but is no
        # longer a direct measurement target.  Treating it as measured would
        # train the Human decoder to reproduce the sanitizer boundary and let
        # a handful of corrupt poses dominate articulated losses.
        measured_raw &= ~geometry_adjusted
        predicted_raw = (
            predicted_raw | geometry_adjusted) & joint_valid & ~measured_raw
        times = np.asarray(arrays["simulation_time_s"], np.float64).reshape(-1)
        ego_state = np.asarray(arrays["ego_state"], np.float32)
        required_velocity_fields = {
            "human_root_velocity", "human_velocity_valid",
            "human_velocity_sigma_mps", "human_measurement_age_s",
            "human_track_age_frames", "human_prediction_run_frames",
            "human_identity_confidence",
        }
        missing_velocity = required_velocity_fields.difference(arrays)
        if missing_velocity:
            raise ValueError(
                "fresh tracker-velocity contract is missing fields: "
                f"{sorted(missing_velocity)}")
        root_velocity_raw = np.asarray(
            arrays["human_root_velocity"], np.float32)
        velocity_valid_raw = np.asarray(
            arrays["human_velocity_valid"], np.bool_)
        velocity_sigma_raw = np.asarray(
            arrays["human_velocity_sigma_mps"], np.float32)
        measurement_age_raw = np.asarray(
            arrays["human_measurement_age_s"], np.float32)
        track_age_frames_raw = np.asarray(
            arrays["human_track_age_frames"], np.float32)
        prediction_run_raw = np.asarray(
            arrays["human_prediction_run_frames"], np.float32)
        identity_confidence_raw = np.asarray(
            arrays["human_identity_confidence"], np.float32)
        if root_velocity_raw.shape != (length, raw_people, 3):
            raise ValueError("human_root_velocity must be [T,N,3]")

        detections: list[HumanDetections] = []
        provenance_by_frame: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
        measured_velocity_by_frame: list[dict[int, np.ndarray]] = []
        root_velocity_by_frame: list[dict[int, np.ndarray]] = []
        tracker_quality_by_frame: list[dict[int, tuple[Any, ...]]] = []
        persistence_by_frame: list[set[int]] = []
        # One or two dropped 10-Hz rows must not destroy a person latent that
        # is otherwise continuously tracked.  Held detections are propagated
        # causally in the episode frame; no future frame is inspected.
        hold_grace_frames = 2
        held: dict[int, dict[str, Any]] = {}
        for frame in range(length):
            active = np.any(joint_valid[frame], axis=1) & (track_ids[frame] >= 0)
            ids = (track_ids[frame, active] + 1).astype(np.int64)
            active_raw_slots = np.flatnonzero(active)
            active_velocity = root_velocity_raw[frame, active_raw_slots].copy()
            active_velocity_valid = velocity_valid_raw[
                frame, active_raw_slots]
            active_velocity[~active_velocity_valid] = 0.0
            velocity = np.broadcast_to(
                active_velocity[:, None, :],
                (active_velocity.shape[0], joint_count, 3),
            ).copy()
            mask = joint_valid[frame, active]
            provenance: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            measured_velocity_map: dict[int, np.ndarray] = {}
            root_velocity_map: dict[int, np.ndarray] = {}
            rotation = _body_to_episode_rotation(ego_state[frame])
            ego_position = ego_state[frame, :3]
            active_ids: set[int] = set()
            tracker_quality: dict[int, tuple[Any, ...]] = {}
            for local, raw_slot in enumerate(active_raw_slots):
                track_id = int(ids[local])
                active_ids.add(track_id)
                valid = mask[local]
                episode_position = ego_position + np.einsum(
                    "ij,kj->ki", rotation, xyz[frame, raw_slot])
                episode_root_velocity = rotation @ active_velocity[local]
                episode_velocity = np.broadcast_to(
                    episode_root_velocity[None, :],
                    (joint_count, 3),
                ).copy()
                previous_track = held.get(track_id)
                if previous_track is not None:
                    joint_velocity, joint_velocity_valid = (
                        _bounded_joint_velocity_episode(
                            episode_position,
                            previous_track["episode_position"],
                            valid,
                            previous_track["joint_mask"],
                            float(times[frame]) - float(previous_track["time"]),
                        )
                    )
                    episode_velocity[joint_velocity_valid] = (
                        joint_velocity[joint_velocity_valid])
                velocity[local] = np.einsum(
                    "ij,kj->ki", rotation.T, episode_velocity)
                held[track_id] = {
                    "episode_position": episode_position,
                    "episode_velocity": episode_velocity,
                    "root_episode_velocity": episode_root_velocity,
                    "joint_mask": valid.copy(),
                    "missed": 0,
                    "time": float(times[frame]),
                    "velocity_valid": bool(velocity_valid_raw[frame, raw_slot]),
                    "velocity_sigma": float(velocity_sigma_raw[frame, raw_slot]),
                    "measurement_age": float(measurement_age_raw[frame, raw_slot]),
                    "track_age_frames": float(track_age_frames_raw[frame, raw_slot]),
                    "prediction_run": float(prediction_run_raw[frame, raw_slot]),
                    "identity_confidence": float(
                        identity_confidence_raw[frame, raw_slot]),
                }
                provenance[track_id] = (
                    measured_raw[frame, raw_slot].copy(),
                    predicted_raw[frame, raw_slot].copy(),
                )
                if bool(velocity_valid_raw[frame, raw_slot]):
                    measured_velocity_map[track_id] = np.broadcast_to(
                        root_velocity_raw[frame, raw_slot][None, :],
                        (joint_count, 3),
                    ).copy()
                root_velocity_map[track_id] = active_velocity[local].copy()
                tracker_quality[track_id] = (
                    bool(velocity_valid_raw[frame, raw_slot]),
                    float(velocity_sigma_raw[frame, raw_slot]),
                    float(measurement_age_raw[frame, raw_slot]),
                    float(track_age_frames_raw[frame, raw_slot]),
                    float(prediction_run_raw[frame, raw_slot]),
                    float(identity_confidence_raw[frame, raw_slot]),
                )

            skeleton = np.concatenate((
                xyz[frame, active], velocity,
                confidence[frame, active, :, None]), axis=-1,
            ).astype(np.float32)

            held_ids, held_skeletons, held_masks = [], [], []
            for track_id, track in list(held.items()):
                if track_id in active_ids:
                    continue
                track["missed"] = int(track["missed"]) + 1
                if int(track["missed"]) > hold_grace_frames:
                    held.pop(track_id, None)
                    continue
                dt = max(0.0, float(times[frame]) - float(track["time"]))
                track["episode_position"] = (
                    track["episode_position"] + track["episode_velocity"] * dt)
                track["time"] = float(times[frame])
                track["measurement_age"] = (
                    float(track["measurement_age"]) + dt)
                track["track_age_frames"] = (
                    float(track["track_age_frames"]) + 1.0)
                track["prediction_run"] = (
                    float(track["prediction_run"]) + 1.0)
                current_position = np.einsum(
                    "ij,kj->ki", rotation.T,
                    track["episode_position"] - ego_position)
                current_velocity = np.einsum(
                    "ij,kj->ki", rotation.T, track["episode_velocity"])
                current_root_velocity = (
                    rotation.T @ track["root_episode_velocity"]
                ).astype(np.float32)
                held_confidence = np.zeros((joint_count, 1), np.float32)
                held_skeletons.append(np.concatenate((
                    current_position, current_velocity, held_confidence), -1))
                held_masks.append(track["joint_mask"].copy())
                held_ids.append(track_id)
                provenance[track_id] = (
                    np.zeros(joint_count, np.bool_),
                    track["joint_mask"].copy(),
                )
                tracker_quality[track_id] = (
                    bool(track["velocity_valid"]),
                    float(track["velocity_sigma"]),
                    float(track["measurement_age"]),
                    float(track["track_age_frames"]),
                    float(track["prediction_run"]),
                    float(track["identity_confidence"]),
                )
                root_velocity_map[track_id] = current_root_velocity
            if held_ids:
                ids = np.concatenate((ids, np.asarray(held_ids, np.int64)))
                skeleton = np.concatenate((
                    skeleton, np.asarray(held_skeletons, np.float32)), axis=0)
                mask = np.concatenate((
                    mask, np.asarray(held_masks, np.bool_)), axis=0)
            sanitized_position, held_geometry_adjusted = (
                sanitize_coco12_geometry_numpy(skeleton[..., :3], mask))
            skeleton[..., :3] = sanitized_position
            for local, track_id in enumerate(ids):
                adjusted = held_geometry_adjusted[local]
                if np.any(adjusted):
                    measured, predicted = provenance[int(track_id)]
                    measured = measured & ~adjusted
                    predicted = (
                        predicted | adjusted) & mask[local] & ~measured
                    provenance[int(track_id)] = (measured, predicted)
                if int(track_id) in held:
                    held[int(track_id)]["episode_position"] = (
                        ego_position
                        + np.einsum(
                            "ij,kj->ki", rotation, sanitized_position[local])
                    )
            detections.append(HumanDetections(
                ids, skeleton, mask,
                _risk_scores(
                    skeleton,
                    mask,
                    ego_velocity_body=_ego_velocity_full_body(
                        ego_state[frame]),
                ),
            ))
            provenance_by_frame.append(provenance)
            measured_velocity_by_frame.append(measured_velocity_map)
            root_velocity_by_frame.append(root_velocity_map)
            tracker_quality_by_frame.append(tracker_quality)
            persistence_by_frame.append(set(int(value) for value in held_ids))
        slots = assign_stable_human_slots(
            detections, max_people=self.max_people,
            joint_count=JOINT_COUNT, feat_dim=SKELETON_FEATURE_DIM,
        )

        slotted_measured = np.zeros(
            (length, self.max_people, joint_count), np.bool_)
        slotted_predicted = np.zeros_like(slotted_measured)
        slotted_measured_velocity = np.zeros(
            (length, self.max_people, joint_count, 3), np.float32)
        slotted_root_velocity = np.zeros(
            (length, self.max_people, 3), np.float32)
        slotted_persistence = np.zeros(
            (length, self.max_people), np.bool_)
        for frame in range(length):
            for slot in np.flatnonzero(slots.human_mask[frame]):
                track_id = int(slots.human_ids[frame, slot])
                provenance = provenance_by_frame[frame].get(track_id)
                if provenance is None:
                    continue
                slotted_measured[frame, slot] = provenance[0]
                slotted_predicted[frame, slot] = provenance[1]
                measured_value = measured_velocity_by_frame[frame].get(track_id)
                if measured_value is not None:
                    slotted_measured_velocity[frame, slot] = measured_value
                root_value = root_velocity_by_frame[frame].get(track_id)
                if root_value is not None:
                    slotted_root_velocity[frame, slot] = root_value
                slotted_persistence[frame, slot] = (
                    track_id in persistence_by_frame[frame])
        slotted_measured &= slots.joint_mask
        slotted_predicted &= slots.joint_mask & ~slotted_measured

        measured_skeleton = slots.skeleton.copy()
        measured_skeleton[..., 3:6] = slotted_measured_velocity
        measured_skeleton[~slotted_measured] = 0.0
        measured_root, _ = human_root_and_relative_joints_numpy(
            measured_skeleton, slots.human_mask, slotted_measured)
        hips = hip_joint_indices(joint_count)
        if hips is None:
            measured_root_valid = slotted_measured.sum(-1) >= 2
        else:
            measured_root_valid = (
                slotted_measured[..., hips[0]] & slotted_measured[..., hips[1]])
        measured_root_valid &= slots.human_mask

        transition_valid = np.ones(max(length - 1, 0), np.bool_)
        if length > 1 and "is_last" in arrays:
            transition_valid &= ~np.asarray(arrays["is_last"], np.bool_).reshape(-1)[:-1]
        same_id = (
            slots.human_mask[:-1] & slots.human_mask[1:]
            & (slots.human_ids[:-1] == slots.human_ids[1:])
            & ~slots.human_is_first[1:]
        )
        human_motion_valid = np.zeros_like(slots.human_mask)
        human_motion_valid[:-1] = same_id & transition_valid[:, None]
        survival_valid = np.zeros_like(slots.human_mask)
        survival_target = np.zeros_like(slots.human_mask)
        lifecycle_transition_valid = (
            transition_valid & skeleton_fresh[:-1] & skeleton_fresh[1:])
        survival_valid[:-1] = (
            slots.human_mask[:-1] & lifecycle_transition_valid[:, None])
        survival_target[:-1] = same_id
        birth_valid = np.zeros_like(slots.human_mask)
        birth_valid[1:] = (
            ~slots.human_mask[:-1]
            & lifecycle_transition_valid[:, None])
        measured_velocity_valid = np.zeros_like(measured_root_valid)

        # Reliability is observational input, not a target.  All temporal
        # counters are episode-global so overlapping windows agree exactly.
        quality = np.zeros((length, self.max_people, 7), np.float32)
        for frame in range(length):
            for slot in range(self.max_people):
                if not slots.human_mask[frame, slot]:
                    continue
                track_id = int(slots.human_ids[frame, slot])
                tracker_quality = tracker_quality_by_frame[frame].get(track_id)
                if tracker_quality is None:
                    continue
                (velocity_valid, velocity_sigma, measurement_age,
                 track_age_frames, prediction_run, identity_confidence) = (
                    tracker_quality)
                valid_count = max(1, int(slots.joint_mask[frame, slot].sum()))
                measured_count = int(slotted_measured[frame, slot].sum())
                predicted_count = int(slotted_predicted[frame, slot].sum())
                quality[frame, slot] = (
                    measured_count / valid_count,
                    predicted_count / valid_count, track_age_frames * 0.1,
                    measurement_age, prediction_run,
                    velocity_sigma if velocity_valid else 0.0,
                    float(velocity_valid) * identity_confidence,
                )
                measured_velocity_valid[frame, slot] = (
                    bool(velocity_valid) and measured_root_valid[frame, slot])

        root, joints = human_root_and_relative_joints_numpy(
            slots.skeleton, slots.human_mask, slots.joint_mask)
        # Root translation uses the robust tracker velocity; articulated joint
        # channels use the causal per-joint reconstruction above. Keeping the
        # decomposition explicit prevents hip finite-difference jitter from
        # redefining the root state while preserving directional limb motion.
        root[..., 3:6] = slotted_root_velocity
        joints[..., 3:6] = (
            slots.skeleton[..., 3:6]
            - slotted_root_velocity[..., None, :]
        )
        joints[~slots.joint_mask] = 0.0
        result = {
            "skeleton": slots.skeleton,
            "human_root": root,
            "human_joints": joints,
            "human_mask": slots.human_mask,
            "joint_mask": slots.joint_mask,
            "human_ids": slots.human_ids,
            "human_is_first": slots.human_is_first,
            "truncated_people": slots.truncated_people,
            "human_joint_measured": slotted_measured,
            "human_joint_predicted": slotted_predicted,
            "human_persistence_mask": slotted_persistence,
            "measured_joint_target": slots.skeleton[..., :3].copy(),
            "measured_joint_target_valid": slotted_measured,
            "measured_root_target": measured_root[..., :6].copy(),
            "measured_root_target_valid": measured_root_valid,
            "measured_velocity_target_valid": measured_velocity_valid,
            "human_motion_valid": human_motion_valid,
            "human_survival_valid": survival_valid,
            "human_survival_target": survival_target,
            "human_birth_target": slots.human_is_first.copy(),
            "human_birth_valid": birth_valid,
            "human_observation_quality": quality,
            "skeleton_fresh": skeleton_fresh[:, None],
            "event_source_observation_reliable": (
                skeleton_fresh | np.any(slotted_persistence, axis=1)
            )[:, None],
        }
        gt_fields = {
            "priv_human_id", "priv_human_mask",
            "priv_human_pelvis_world", "priv_human_pelvis_valid",
        }
        if gt_fields.issubset(arrays):
            episode = self.episodes[episode_index]
            origin_xyz = np.asarray(
                episode.metadata.get("ego_reference_origin_xyz", (0, 0, 0)),
                np.float64).reshape(3)
            origin_yaw = float(
                episode.metadata.get("ego_reference_origin_yaw", 0.0))
            gt_match = _match_detected_slots_to_gt(
                measured_root[..., :3], measured_root_valid,
                slots.human_ids, ego_state,
                np.asarray(arrays["priv_human_id"], np.int64),
                np.asarray(arrays["priv_human_mask"], np.bool_),
                np.asarray(arrays["priv_human_pelvis_world"], np.float32),
                np.asarray(arrays["priv_human_pelvis_valid"], np.bool_),
                origin_xyz=origin_xyz, origin_yaw=origin_yaw,
            )
            result.update(gt_match)
            result.update(_carry_gt_identity_through_causal_track(
                slots.human_ids,
                slots.human_mask,
                slots.human_is_first,
                gt_match["human_gt_id"],
                gt_match["human_gt_match_valid"],
                np.asarray(arrays["simulation_time_s"], np.float64),
            ))
        self._human_cache[episode_index] = result
        self._human_cache.move_to_end(episode_index)
        while len(self._human_cache) > self.human_cache_episodes:
            self._human_cache.popitem(last=False)
        return result

    def _episode_risk_targets(self, episode_index: int) -> dict[str, np.ndarray]:
        """Build causal action-risk targets from recorder privileged truth.

        Targets at row ``t`` describe the interval ``[t, t + horizon]``.  They
        are training-only labels and never enter the observation encoder or
        deployment input. Surface clearance uses the UAV collision radius
        stored in each episode's recorder metadata plus human joint radii,
        rather than a center-to-center skeleton proxy. The binary collision
        target itself comes from the recorded PhysX contact event.
        """
        cached = self._risk_cache.get(episode_index)
        if cached is not None:
            self._risk_cache.move_to_end(episode_index)
            return cached
        arrays = self._episode_arrays(episode_index)
        times = np.asarray(arrays["simulation_time_s"], np.float64).reshape(-1)
        length = int(times.size)
        future_collision = np.zeros(length, np.bool_)
        # Training-only attribution for prioritized imagination starts.  The
        # identity labels which currently visible Human ultimately generated
        # the recorded PhysX contact; it is never part of the observation or
        # deployment state.  Unknown/ambiguous recorder identities stay -1 so
        # sampling cannot silently assign the event to the wrong person.
        future_collision_human_id = np.full(length, -1, np.int64)
        time_to_collision = np.zeros(length, np.float32)
        time_to_collision_valid = np.zeros(length, np.bool_)
        collision = np.asarray(
            arrays.get("priv_collision", np.zeros(length, np.bool_)),
            np.bool_,
        ).reshape(-1)
        # Legacy collision recordings may expose only the terminal code.
        termination = np.asarray(
            arrays.get("termination_code", np.zeros(length, np.uint8)),
            np.uint8,
        ).reshape(-1)
        collision = collision | (termination == 2)
        collision_indices = np.flatnonzero(collision)
        collision_human_id = resolve_collision_human_id(
            self.episodes[episode_index].metadata,
            self.episodes[episode_index].summary,
        )

        clearance = np.asarray(
            arrays.get("priv_min_human_clearance_m", np.zeros(length)),
            np.float32,
        ).reshape(-1)
        clearance_valid = np.asarray(
            arrays.get("priv_min_human_clearance_valid", np.zeros(length)),
            np.bool_,
        ).reshape(-1)
        future_clearance = np.zeros(length, np.float32)
        future_clearance_valid = np.zeros(length, np.bool_)

        ttc = np.asarray(
            arrays.get("priv_min_human_ttc_s", np.zeros(length)),
            np.float32,
        ).reshape(-1)
        ttc_valid = np.asarray(
            arrays.get("priv_min_human_ttc_valid", np.zeros(length)),
            np.bool_,
        ).reshape(-1)
        future_ttc = np.zeros(length, np.float32)
        future_ttc_valid = np.zeros(length, np.bool_)

        stops = np.searchsorted(
            times, times + self.risk_horizon_s, side="right")
        for index, stop in enumerate(stops):
            collision_pos = np.searchsorted(collision_indices, index, side="left")
            if (
                collision_pos < collision_indices.size
                and int(collision_indices[collision_pos]) < int(stop)
            ):
                event_index = int(collision_indices[collision_pos])
                future_collision[index] = True
                time_to_collision[index] = max(
                    0.0, float(times[event_index] - times[index]))
                time_to_collision_valid[index] = True
                if collision_human_id is not None:
                    future_collision_human_id[index] = collision_human_id
            window_clearance_valid = clearance_valid[index:stop]
            if np.any(window_clearance_valid):
                future_clearance[index] = float(np.min(
                    clearance[index:stop][window_clearance_valid]))
                future_clearance_valid[index] = True
            window_ttc_valid = ttc_valid[index:stop]
            if np.any(window_ttc_valid):
                future_ttc[index] = float(np.min(
                    ttc[index:stop][window_ttc_valid]))
                future_ttc_valid[index] = True

        targets = {
            "future_collision": future_collision[:, None],
            "future_collision_human_id": (
                future_collision_human_id[:, None]),
            "time_to_collision_s": time_to_collision[:, None],
            "time_to_collision_valid": time_to_collision_valid[:, None],
            "future_min_human_clearance_m": future_clearance[:, None],
            "future_min_human_clearance_valid": future_clearance_valid[:, None],
            "future_min_human_ttc_s": future_ttc[:, None],
            "future_min_human_ttc_valid": future_ttc_valid[:, None],
            **derive_one_step_transition_targets(
                termination,
                clearance,
                clearance_valid,
            ),
        }
        reason = str(
            self.episodes[episode_index].summary.get(
                "termination_reason", ""))
        remaining_human_collision = np.full(
            (length, 1), reason == "human_collision", np.bool_)
        remaining_human_collision_valid = np.full(
            (length, 1), reason in _TASK_TERMINATION_REASONS, np.bool_)
        if length:
            # A terminal state has no remaining policy outcome. Excluding it
            # also prevents terminal posteriors from entering Safety Critic
            # calibration through a window that ends exactly on the event.
            remaining_human_collision_valid[-1, 0] = False
        targets.update({
            "remaining_human_collision_target": remaining_human_collision,
            "remaining_human_collision_valid": (
                remaining_human_collision_valid),
        })
        self._risk_cache[episode_index] = targets
        self._risk_cache.move_to_end(episode_index)
        while len(self._risk_cache) > self.cache_episodes:
            self._risk_cache.popitem(last=False)
        return targets

    def sampling_category(self, index: int) -> str:
        """Classify a window for outcome-balanced offline sampling."""
        episode_index, start = self.windows[int(index)]
        episode = self.episodes[episode_index]
        stop = start + self.sequence_length
        reason = str(episode.summary.get("termination_reason", ""))
        if reason == "human_collision":
            risk = self._episode_risk_targets(episode_index)["future_collision"]
            if bool(np.any(risk[start:stop])):
                return "collision_danger"
            return "other"
        if reason == "time_limit":
            # Online no-collision timeouts are not demonstrations, but they
            # are valuable OOD posterior starts for teaching Value/Actor to
            # recover progress after a safety intervention.  They remain
            # excluded by default and enter only with --include-time-limit.
            return "timeout_stall"
        if reason != "reached_goal":
            return "other"
        arrays = self._episode_arrays(episode_index)
        clearance = arrays.get("priv_min_human_clearance_m")
        valid = arrays.get("priv_min_human_clearance_valid")
        if clearance is not None and valid is not None:
            window_valid = np.asarray(valid[start:stop], np.bool_)
            if (
                np.any(window_valid)
                and float(np.min(np.asarray(clearance[start:stop])[window_valid]))
                < self.dense_clearance_m
            ):
                return "success_dense"
        return "success_ordinary"

    def sampling_categories(self, indices: Sequence[int]) -> list[str]:
        return [self.sampling_category(int(index)) for index in indices]

    def transition_event_category(self, index: int) -> str:
        """Return the single transition class selected for event balancing.

        Only the final source row of a completed task episode carries a
        terminal event. Every other eligible window supplies one continue
        transition. This avoids oversampling the ordinary transitions that
        merely share a sequence with a rare terminal event.
        """
        episode_index, start = self.windows[int(index)]
        episode = self.episodes[episode_index]
        terminal_source = int(episode.length) - 2
        contains_terminal_source = (
            start <= terminal_source <= start + self.sequence_length - 2)
        if not contains_terminal_source:
            return "continue"
        reason = str(episode.summary.get("termination_reason", ""))
        return {
            "human_collision": "human_collision",
            "static_collision": "static_collision",
            "reached_goal": "reached_goal",
            "out_of_bounds": "hard_failure",
            "crash": "hard_failure",
            "stuck_timeout": "stuck_timeout",
        }.get(reason, "continue")

    def transition_event_source_offset(self, index: int) -> int:
        """Select one source row for an event-balanced auxiliary sample."""
        episode_index, start = self.windows[int(index)]
        episode = self.episodes[episode_index]
        terminal_source = int(episode.length) - 2
        if (
            start <= terminal_source <= start + self.sequence_length - 2
            and self.transition_event_category(index) != "continue"
        ):
            return terminal_source - start
        return int(index) % max(1, self.sequence_length - 1)

    def event_auxiliary_transitions(
        self, *, non_goal_only: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Index every eligible physical transition exactly once.

        Training windows overlap heavily, so a window-category histogram is
        not an event prior.  This index first deduplicates by
        ``(episode, source row)`` and only then counts event classes.  Each
        transition is attached to one deterministic containing window for
        feature reconstruction.
        """
        windows_by_episode: dict[int, list[tuple[int, int]]] = {}
        for window_index, (episode_index, start) in enumerate(self.windows):
            if int(start) < 0:
                continue
            windows_by_episode.setdefault(int(episode_index), []).append(
                (int(start), int(window_index)))
        centre = 0.5 * max(0, self.sequence_length - 2)
        containing: dict[tuple[int, int], tuple[float, int, int]] = {}
        for episode_index, values in windows_by_episode.items():
            values.sort()
            starts = [value[0] for value in values]
            first_source = starts[0]
            final_source = starts[-1] + self.sequence_length - 2
            for source in range(first_source, final_source + 1):
                desired = float(source) - centre
                insertion = bisect_left(starts, desired)
                candidates = []
                for position in (insertion - 1, insertion):
                    if not 0 <= position < len(values):
                        continue
                    start, window_index = values[position]
                    offset = source - start
                    if 0 <= offset <= self.sequence_length - 2:
                        candidates.append((
                            abs(float(offset) - centre), window_index, offset))
                if candidates:
                    containing[(episode_index, source)] = min(candidates)

        # Transition-unique replay represents a short episode/prefix with one
        # left-padded window.  Its physical source offsets begin at the first
        # real row, not at padding row zero.
        for window_index, (episode_index, start) in enumerate(self.windows):
            if int(start) >= 0:
                continue
            real_length = -int(start)
            left_padding = self.sequence_length - real_length
            for source in range(max(0, real_length - 1)):
                containing[(int(episode_index), source)] = (
                    abs(float(source) - centre), int(window_index),
                    left_padding + source,
                )

        full_keys = TRANSITION_EVENT_KEYS
        non_goal_keys = (
            "continue", "human_collision", "static_collision",
            "hard_failure", "stuck_timeout",
        )
        records: list[dict[str, Any]] = []
        for (episode_index, source), (_, window_index, offset) in sorted(
            containing.items()
        ):
            episode = self.episodes[episode_index]
            if source >= len(episode.event_source_code):
                continue
            if not bool(episode.event_source_valid[source]):
                continue
            code = int(episode.event_source_code[source])
            if non_goal_only:
                target = int(_TERMINATION_TO_NON_GOAL_EVENT[code])
                category = non_goal_keys[target]
            else:
                target = int(_TERMINATION_TO_TRANSITION_EVENT[code])
                category = full_keys[target]
            records.append({
                "window_index": int(window_index),
                "source_offset": int(offset),
                "category": category,
                "episode_index": int(episode_index),
                "episode_id": episode.name,
                "source_transition": int(source),
                "left_padding": 0,
            })

        # Ordinary fixed-length replay deliberately excludes short episodes,
        # but their terminal transition is often the most useful collision
        # evidence.  Expose every valid physical transition through an
        # Event-only, left-padded sequence.  ``sequence_valid`` prevents the
        # synthetic prefix from ever becoming a reconstruction/transition
        # target; the posterior is reset again at the first real row.
        represented_episodes = {
            int(episode_index) for episode_index, _ in self.windows}
        for episode_index, episode in enumerate(self.episodes):
            if episode_index in represented_episodes or episode.length < 2:
                continue
            if episode.length >= self.sequence_length:
                continue
            left_padding = self.sequence_length - int(episode.length)
            for source in range(min(
                int(episode.length) - 1, len(episode.event_source_code),
            )):
                if not bool(episode.event_source_valid[source]):
                    continue
                code = int(episode.event_source_code[source])
                if non_goal_only:
                    target = int(_TERMINATION_TO_NON_GOAL_EVENT[code])
                    category = non_goal_keys[target]
                else:
                    target = int(_TERMINATION_TO_TRANSITION_EVENT[code])
                    category = full_keys[target]
                records.append({
                    "window_index": -1,
                    "source_offset": left_padding + int(source),
                    "category": category,
                    "episode_index": int(episode_index),
                    "episode_id": episode.name,
                    "source_transition": int(source),
                    "left_padding": int(left_padding),
                })
        if not records:
            raise ValueError("no valid unique event transitions are available")
        return tuple(records)

    def _build_item(
        self, episode_index: int, start: int, length: int,
    ) -> dict[str, Any]:
        episode = self.episodes[episode_index]
        arrays = self._episode_arrays(episode_index)
        stop = start + int(length)
        raw = {key: value[start:stop] for key, value in arrays.items()}
        risk_targets = {
            key: value[start:stop]
            for key, value in self._episode_risk_targets(episode_index).items()
        }
        human_episode = self._episode_human_targets(episode_index)
        human = {
            key: np.asarray(value[start:stop]).copy()
            for key, value in human_episode.items()
        }
        source_event_valid = np.zeros(
            (length, 1), np.bool_)
        episode_event_valid = np.asarray(
            episode.event_source_valid, np.bool_)
        source_stop = min(stop, episode_event_valid.size)
        if source_stop > start:
            source_event_valid[:source_stop - start, 0] = (
                episode_event_valid[start:source_stop])
        risk_targets["transition_event_valid"] &= source_event_valid
        # A sampled window has no recurrent history before row zero even when
        # the physical track predates the window.  This reset is a model-state
        # boundary, not a birth label; human_birth_target remains unchanged.
        human["human_is_first"][0] = human["human_mask"][0]
        ego_state = np.asarray(raw["ego_state"], np.float32)
        goal_fixed = np.asarray(
            episode.metadata["goal_position_episode_local"], np.float32).reshape(3)
        goal_position = np.broadcast_to(
            goal_fixed, (length, 3)).copy()
        episode_contract = episode.metadata.get("episode", {})
        reward_config = episode.metadata.get("reward_config", {})
        task_memory_all = derive_task_memory_numpy(
            np.asarray(arrays["simulation_time_s"], np.float64),
            np.asarray(arrays["ego_state"], np.float32),
            goal_fixed,
            np.asarray(arrays["action"], np.float32),
            progress_epsilon_m=float(episode_contract.get(
                "watchdog_progress_epsilon_m", 0.25)),
            stuck_max_horizontal_speed_mps=float(episode_contract.get(
                "watchdog_stuck_max_horizontal_speed_mps", 0.15)),
            acceleration_filter_alpha=float(reward_config.get(
                "acceleration_filter_alpha", 0.25)),
        )
        task_memory = task_memory_all[start:stop].copy()
        full_simulation_time = np.asarray(
            arrays["simulation_time_s"], np.float64).reshape(-1)
        simulation_time = full_simulation_time[start:stop].copy()
        episode_time = (
            simulation_time
            - float(full_simulation_time[0])
        ).astype(np.float32)
        full_dt = np.zeros(full_simulation_time.size, np.float32)
        if full_simulation_time.size > 1:
            full_dt[1:] = np.diff(full_simulation_time).astype(np.float32)
        dt = full_dt[start:stop].copy()
        origin_xyz = np.asarray(
            episode.metadata.get("ego_reference_origin_xyz", (0, 0, 0)),
            np.float64).reshape(3)
        origin_yaw = float(
            episode.metadata.get("ego_reference_origin_yaw", 0.0))
        world_from_episode = _episode_to_world_rotation(origin_yaw)
        is_first = np.zeros((length, 1), np.bool_)
        is_first[0, 0] = True
        physical_is_first = np.asarray(
            raw.get("is_first", np.zeros(length, np.bool_)),
            np.bool_,
        ).reshape(-1, 1)
        is_last = np.asarray(raw["is_last"], np.bool_).reshape(-1, 1)
        is_terminal = np.asarray(raw["is_terminal"], np.bool_).reshape(-1, 1)
        item: dict[str, Any] = {
            "record_name": episode.name,
            "frame_ids": np.asarray(raw["frame_index"], np.int64),
            "ego_state": ego_state,
            "goal_position": goal_position,
            "goal": goal_features_numpy(ego_state, goal_position),
            "simulation_time_s": simulation_time,
            "episode_time_s": episode_time[:, None],
            "dt_s": dt[:, None],
            "task_memory": task_memory,
            **human,
            "action": np.asarray(raw["action"], np.float32),
            "action_valid": np.asarray(raw["action_valid"], np.bool_).reshape(-1, 1),
            "reward": np.asarray(raw["reward"], np.float32).reshape(-1, 1),
            "reward_components": np.asarray(raw["reward_components"], np.float32),
            "is_first": is_first,
            # Distinguish a physical episode reset from the artificial RSSM
            # reset introduced at a sampled window boundary.
            "physical_is_first": physical_is_first,
            # Outcome labels are training/control targets and sampling metadata;
            # they are intentionally not consumed by the observation encoder.
            "success": np.asarray(raw["success"], np.bool_).reshape(-1, 1),
            "episode_success": np.full(
                (length, 1),
                str(episode.summary.get("termination_reason", "")) == "reached_goal",
                np.bool_,
            ),
            "termination_code": np.asarray(
                raw["termination_code"], np.uint8).reshape(-1, 1),
            "is_last": is_last,
            "is_terminal": is_terminal,
            **risk_targets,
        }
        collision_human_id = np.full((length, 1), -1, np.int64)
        resolved_collision_human_id = resolve_collision_human_id(
            episode.metadata, episode.summary)
        if resolved_collision_human_id is not None:
            terminal_human = (
                np.asarray(item["termination_code"]).reshape(-1) == 2)
            collision_human_id[
                terminal_human, 0] = resolved_collision_human_id
        item["priv_collision_human_id"] = collision_human_id
        if {
            "priv_collision_joints_world", "priv_collision_joint_valid",
        }.issubset(raw):
            privileged_contract = episode.metadata.get(
                "privileged_geometry_contract", {})
            contact_contract = str(privileged_contract.get(
                "physx_contact_event_contract", ""))
            if contact_contract != (
                "joint_surface_gap_m <= joint_contact_offset_m"
            ):
                raise ValueError(
                    f"{episode.directory}: unsupported privileged collision "
                    f"contract {contact_contract!r}")
            try:
                joint_contact_offset_m = float(
                    privileged_contract["joint_contact_offset_m"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{episode.directory}: privileged collision geometry "
                    "requires a numeric joint_contact_offset_m") from error
            if (
                not np.isfinite(joint_contact_offset_m)
                or joint_contact_offset_m < 0.0
            ):
                raise ValueError(
                    f"{episode.directory}: joint_contact_offset_m must be "
                    "finite and non-negative")
            joints_world = np.asarray(
                raw["priv_collision_joints_world"], np.float32)
            joints_episode = np.einsum(
                "ij,tpqj->tpqi", world_from_episode.T,
                joints_world.astype(np.float64) - origin_xyz,
            ).astype(np.float32)
            (
                item["priv_collision_joints_episode"],
                item["priv_collision_joint_valid"],
            ) = fixed_privileged_collision_geometry(
                joints_episode,
                np.asarray(raw["priv_collision_joint_valid"], np.bool_),
            )
            item["privileged_geometry_available"] = np.ones(
                (length, 1), np.bool_)
            reward_config = episode.metadata.get("reward_config", {})
            drone_radius = float(reward_config.get(
                "drone_collision_radius_m",
                episode.metadata.get("episode", {}).get(
                    "drone_collision_geometry", {}).get("radius_m", 0.17),
            ))
            configured_joint_radii = reward_config.get(
                "joint_collision_radii", {})
            surface_radii = np.asarray([
                drone_radius + float(configured_joint_radii.get(
                    name, PRIVILEGED_COLLISION_JOINT_RADII_M[name]))
                for name in PRIVILEGED_COLLISION_JOINT_NAMES
            ], np.float32)
            if item["priv_collision_joints_episode"].shape[-2] != (
                surface_radii.size
            ):
                raise ValueError(
                    "privileged collision joint/radius topology mismatch")
            item["counterfactual_collision_surface_radii_m"] = (
                np.broadcast_to(surface_radii, (
                    length, surface_radii.size)).copy())
            item["counterfactual_collision_contact_offset_m"] = np.full(
                (length, 1), joint_contact_offset_m, np.float32)
        else:
            item["privileged_geometry_available"] = np.zeros(
                (length, 1), np.bool_)
        geometry_status = self.task_geometry_status[episode_index]
        bounds = np.asarray(
            episode_contract.get(
                "factorized_flight_bounds_xy",
                (-1.0e6, 1.0e6, -1.0e6, 1.0e6)),
            np.float32).reshape(4)
        item["counterfactual_flight_bounds_world"] = np.broadcast_to(
            bounds, (length, 4)).copy()
        item["counterfactual_bounds_valid"] = np.full(
            (length, 1), bool(geometry_status["bounds_valid"]), np.bool_)
        # Fixed padding keeps the normal PyTorch collate path valid even when
        # a future scene contains a different number of obstacle prims.
        maximum_obstacles = 256
        obstacle_values = np.asarray(
            episode_contract.get("factorized_static_obstacle_aabbs_xy", ()),
            np.float32).reshape(-1, 4)
        if obstacle_values.shape[0] > maximum_obstacles:
            raise ValueError(
                f"episode has {obstacle_values.shape[0]} static AABBs; "
                f"maximum is {maximum_obstacles}")
        obstacle_aabbs = np.zeros((maximum_obstacles, 4), np.float32)
        obstacle_valid = np.zeros((maximum_obstacles,), np.bool_)
        obstacle_aabbs[:obstacle_values.shape[0]] = obstacle_values
        obstacle_valid[:obstacle_values.shape[0]] = True
        item["counterfactual_static_obstacle_aabbs_world"] = np.broadcast_to(
            obstacle_aabbs, (length, maximum_obstacles, 4)).copy()
        item["counterfactual_static_obstacle_valid"] = np.broadcast_to(
            obstacle_valid, (length, maximum_obstacles)).copy()
        item["counterfactual_static_obstacle_preinflation_m"] = np.full(
            (length, 1), float(episode_contract.get(
                "factorized_static_obstacle_preinflation_m", 0.0)),
            np.float32)
        item["counterfactual_static_proxy_valid"] = np.full(
            (length, 1), bool(geometry_status["static_proxy_valid"]), np.bool_)
        item["counterfactual_static_terminal_valid"] = np.full(
            (length, 1),
            bool(
                geometry_status["static_terminal_valid"]
                or self.promote_static_proxy_to_task_terminal
            ),
            np.bool_,
        )
        item["counterfactual_origin_xyz"] = np.broadcast_to(
            origin_xyz.astype(np.float32), (length, 3)).copy()
        item["counterfactual_origin_yaw"] = np.full(
            (length, 1), origin_yaw, np.float32)
        item["counterfactual_goal_radius_m"] = np.full(
            (length, 1),
            float(episode_contract.get("goal_radius_3d_m", 1.0)),
            np.float32)
        item["counterfactual_maximum_altitude_m"] = np.full(
            (length, 1),
            float(episode_contract.get("maximum_altitude_m", 1.0e6)),
            np.float32)
        item["counterfactual_maximum_altitude_valid"] = np.full(
            (length, 1), bool(geometry_status["maximum_altitude_valid"]),
            np.bool_)
        item["counterfactual_task_geometry_valid"] = np.full(
            (length, 1), bool(geometry_status["valid"]), np.bool_)
        item["counterfactual_crash_altitude_m"] = np.full(
            (length, 1),
            float(episode_contract.get("crash_altitude_m", -1.0e6)),
            np.float32)
        item["counterfactual_crash_min_elapsed_s"] = np.full(
            (length, 1), float(episode_contract.get(
                "crash_min_elapsed_s", 0.0)), np.float32)
        item["counterfactual_watchdog_min_elapsed_s"] = np.full(
            (length, 1), float(episode_contract.get(
                "watchdog_min_elapsed_s", 5.0)), np.float32)
        item["counterfactual_watchdog_no_progress_timeout_s"] = np.full(
            (length, 1), float(episode_contract.get(
                "watchdog_no_progress_timeout_s", 30.0)), np.float32)
        item["counterfactual_watchdog_progress_epsilon_m"] = np.full(
            (length, 1), float(episode_contract.get(
                "watchdog_progress_epsilon_m", 0.25)), np.float32)
        item["counterfactual_watchdog_stuck_max_horizontal_speed_mps"] = (
            np.full((length, 1), float(episode_contract.get(
                "watchdog_stuck_max_horizontal_speed_mps", 0.15)),
                np.float32))
        relative_yaw = np.arctan2(ego_state[:, 12], ego_state[:, 13])
        cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
        position_world_xy = np.stack((
            origin_xyz[0] + cosine * ego_state[:, 0]
            - sine * ego_state[:, 1],
            origin_xyz[1] + sine * ego_state[:, 0]
            + cosine * ego_state[:, 1],
        ), -1).astype(np.float32)
        item["task_geometry"] = task_geometry_proximity_numpy(
            position_world_xy,
            origin_yaw + relative_yaw,
            bounds,
            obstacle_aabbs,
            obstacle_valid,
        )
        privileged_person_fields = {
            "priv_human_id", "priv_human_mask", "priv_human_position_world",
            "priv_human_velocity_world", "priv_human_pelvis_world",
            "priv_human_pelvis_valid",
        }
        for key in (
            "priv_human_id", "priv_human_mask", "priv_human_position_world",
            "priv_human_velocity_world", "priv_min_human_clearance_m",
            "priv_min_human_clearance_valid", "priv_min_human_ttc_s",
            "priv_min_human_ttc_valid", "priv_collision", "priv_goal_distance_m",
            "priv_human_pelvis_world", "priv_human_pelvis_valid",
        ):
            if key in raw:
                value = np.asarray(raw[key])
                if key in privileged_person_fields:
                    people = int(value.shape[1])
                    if people > PRIVILEGED_GT_COLLISION_MAX_PEOPLE:
                        raise ValueError(
                            f"{key} has {people} people; fixed privileged "
                            "capacity is "
                            f"{PRIVILEGED_GT_COLLISION_MAX_PEOPLE}")
                    shape = (
                        value.shape[0], PRIVILEGED_GT_COLLISION_MAX_PEOPLE,
                        *value.shape[2:])
                    fill_value = -1 if key == "priv_human_id" else 0
                    padded = np.full(shape, fill_value, dtype=value.dtype)
                    padded[:, :people] = value
                    value = padded
                item[key] = value
        validate_factorized_batch(item)
        return item

    def _left_padded_item(
        self, episode_index: int, real_length: int,
    ) -> dict[str, Any]:
        """Return a fixed sequence whose suffix contains real episode rows."""
        real_length = int(real_length)
        left_padding = self.sequence_length - real_length
        if left_padding <= 0:
            raise ValueError("left-padded window must be shorter than sequence")
        real = self._build_item(episode_index, 0, real_length)
        padded: dict[str, Any] = {}
        for key, value in real.items():
            if not isinstance(value, np.ndarray) or value.ndim == 0 \
                    or value.shape[0] != real_length:
                padded[key] = value
                continue
            shape = (self.sequence_length, *value.shape[1:])
            fill = np.zeros(shape, dtype=value.dtype)
            if key in (
                "human_ids", "human_gt_id", "human_gt_identity_id",
                "priv_human_id", "priv_collision_human_id",
                "future_collision_human_id",
            ):
                fill.fill(-1)
            elif key == "frame_ids":
                fill.fill(-1)
            elif key == "ego_state":
                fill[:, 13] = 1.0
            fill[left_padding:] = value
            padded[key] = fill
        padded["sequence_valid"] = np.zeros(
            (self.sequence_length, 1), np.bool_)
        padded["sequence_valid"][left_padding:] = True
        padded["is_first"][:] = False
        padded["is_first"][0, 0] = True
        padded["is_first"][left_padding, 0] = True
        padded["physical_is_first"][:left_padding] = False
        padded["human_is_first"][left_padding] = padded[
            "human_mask"][left_padding]
        # Goal inputs in padding are harmless because the recurrent state is
        # reset at the first real row, but keep them finite and normalized.
        padded["goal_position"][:left_padding] = padded[
            "goal_position"][left_padding]
        padded["goal"][:left_padding] = goal_features_numpy(
            padded["ego_state"][:left_padding],
            padded["goal_position"][:left_padding],
        )
        return padded

    def event_auxiliary_item(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return a normal or Event-only left-padded sequence."""
        window_index = int(record["window_index"])
        if window_index >= 0:
            return dict(self[window_index])

        episode_index = int(record["episode_index"])
        episode = self.episodes[episode_index]
        return self._left_padded_item(episode_index, int(episode.length))

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, start = self.windows[index]
        if int(start) < 0:
            return self._left_padded_item(int(episode_index), -int(start))
        item = self._build_item(
            int(episode_index), int(start), self.sequence_length)
        item["sequence_valid"] = np.ones(
            (self.sequence_length, 1), np.bool_)
        return item
