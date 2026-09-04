"""Shared local static-task geometry for replay, imagination and deployment.

The pure Actor cannot infer world-fixed bounds or obstacles from an
episode-local Ego pose.  Eight body-frame ray proximities are retained as a
compact learned-token input, but they are not a complete decision state: an
AABB corner can miss every ray while still determining the analytic terminal
and reward.  The full-state helpers below therefore expose all boundary
planes, the altitude ceiling and every configured AABB directly to the Actor
and Critic, with identical NumPy/Torch semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import torch


TASK_GEOMETRY_DIM = 8
TASK_GEOMETRY_MAX_RANGE_M = 12.0
TASK_GEOMETRY_FEATURE_SCALE = 0.5
TASK_PHYSICAL_BASE_DIM = 16
TASK_PHYSICAL_OBSTACLE_FIELDS = 7
TASK_SCENE_GEOMETRY_CONTRACT_VERSION = "fixed_world_xy_scene_v1"
TASK_SCENE_GEOMETRY_TRANSPORT_ATOL_M = 5.0e-6
PURE_DREAMER_TASK_SUPPORT_CONTRACT_VERSION = (
    "fixed_warehouse_traversal_support_v2")
PURE_DREAMER_EMPIRICAL_TASK_SUPPORT_VERSION = "observed_replay_support_v1"

# The Simple Warehouse contains one small KLT label whose collision mesh cuts
# into the UAV flight region at the fixed 1 m task altitude.  The original
# obstacle scan was intentionally a *pedestrian-ground* scan
# (bottom_z <= 0.35 m), so this suspended collider was absent from the static
# state given to the Actor even though PhysX terminated three replay episodes
# on it.  The values below were measured from the loaded Isaac USD, not from a
# collision trajectory.  They include the same 0.30 m conservative inflation
# used by the versioned analytic-static-terminal contract.
WAREHOUSE_V2_KLT134_STATIC_GEOMETRY_MIGRATION = (
    "warehouse_v2_cruise_height_klt134_v1")
WAREHOUSE_V2_LEGACY_SCENE_SHA256 = (
    "6ebb1eba68401d573cd09db3db24abdefd4c6a8ee4372fb18743091c31df8548")
WAREHOUSE_V2_KLT134_SCENE_SHA256 = (
    "953d73256dc7814053fda6d5f072f14357e898c8170a5f639ed95292b1b25481")
WAREHOUSE_V2_KLT134_RAW_AABB_WORLD = (
    -6.558715939040021,
    13.717232275746056,
    -6.531031558312147,
    13.798022093770816,
)
WAREHOUSE_V2_KLT134_STATIC_AABB_WORLD = (
    -6.858715939040021,
    13.417232275746056,
    -6.231031558312147,
    14.098022093770817,
)


def migrate_warehouse_v2_klt134_static_geometry(
    flight_bounds_xy: np.ndarray | list[float],
    static_obstacle_aabbs_world: np.ndarray | list[list[float]],
    *,
    maximum_altitude_m: float,
) -> tuple[np.ndarray, bool]:
    """Upgrade only the audited legacy warehouse scene to the KLT-aware MDP.

    This is deliberately hash-gated.  It must never append a warehouse prop
    to an unknown layout merely because an episode happens to use the same
    schema strings.  The returned array is canonical scene content; ``True``
    means that the legacy nine-AABB identity was upgraded in this call.
    """
    obstacles = np.asarray(static_obstacle_aabbs_world, np.float64)
    if obstacles.size == 0:
        obstacles = np.zeros((0, 4), np.float64)
    else:
        obstacles = obstacles.reshape(-1, 4)
    # Replay materializes metadata through float32 tensors before the fixed
    # scene identity is computed.  Canonicalize through that same transport so
    # raw Isaac JSON and an already-rounded checkpoint cannot become two
    # migrations because of one-micron decimal ties.
    obstacles = obstacles.astype(np.float32).astype(np.float64)
    current = fixed_task_scene_geometry_contract(
        flight_bounds_xy,
        obstacles,
        maximum_altitude_m=maximum_altitude_m,
    )
    digest = str(current["sha256"])
    if digest == WAREHOUSE_V2_KLT134_SCENE_SHA256:
        return obstacles.copy(), False
    if digest != WAREHOUSE_V2_LEGACY_SCENE_SHA256:
        raise ValueError(
            "warehouse KLT134 migration received an unknown fixed scene: "
            f"{digest}")
    extended = np.concatenate((
        obstacles,
        np.asarray(
            WAREHOUSE_V2_KLT134_STATIC_AABB_WORLD,
            np.float64,
        ).reshape(1, 4),
    ), axis=0).astype(np.float32).astype(np.float64)
    migrated = fixed_task_scene_geometry_contract(
        flight_bounds_xy,
        extended,
        maximum_altitude_m=maximum_altitude_m,
    )
    if migrated["sha256"] != WAREHOUSE_V2_KLT134_SCENE_SHA256:
        raise RuntimeError(
            "warehouse KLT134 migration did not produce its audited scene")
    return extended, True


def fixed_task_scene_geometry_contract(
    flight_bounds_xy: np.ndarray | list[float],
    static_obstacle_aabbs_world: np.ndarray | list[list[float]],
    static_obstacle_valid: np.ndarray | list[bool] | None = None,
    *,
    maximum_altitude_m: float,
) -> dict[str, object]:
    """Canonical identity of the request-visible fixed task scene.

    Entity order and invalid padding are representation details, not scene
    identity.  Active AABBs are therefore lexicographically sorted before a
    six-decimal world-coordinate digest is produced.  The quantization is
    substantially tighter than simulator geometry accuracy while remaining
    stable across JSON and float32 request transport.
    """
    bounds = np.asarray(flight_bounds_xy, np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("fixed scene flight bounds must be finite [4]")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("fixed scene flight bounds are reversed")
    obstacles = np.asarray(static_obstacle_aabbs_world, np.float64)
    if obstacles.size == 0:
        obstacles = np.zeros((0, 4), np.float64)
    elif obstacles.ndim != 2 or obstacles.shape[1] != 4:
        raise ValueError("fixed scene static AABBs must have shape [K,4]")
    if not np.isfinite(obstacles).all():
        raise ValueError("fixed scene static AABBs must be finite")
    if static_obstacle_valid is None:
        valid = np.ones(obstacles.shape[0], np.bool_)
    else:
        valid = np.asarray(static_obstacle_valid, np.bool_)
        if valid.shape != (obstacles.shape[0],):
            raise ValueError("fixed scene static validity must have shape [K]")
    active = obstacles[valid]
    if active.size and bool((active[:, :2] > active[:, 2:]).any()):
        raise ValueError("fixed scene static AABB bounds are reversed")
    # Quantize before sorting.  Several warehouse fixtures deliberately share
    # the same x extent.  Their raw simulator float64 endpoints can differ by
    # a few ulps while the replay path materializes the same values through
    # float32.  Sorting the raw values therefore made those tied fixtures swap
    # order across transport boundaries, turning one physical scene into two
    # different ordered arrays even though every coordinate was equivalent.
    active = np.round(active, decimals=6)
    active[np.abs(active) < 0.5e-6] = 0.0
    if active.shape[0]:
        order = np.lexsort(tuple(active[:, index] for index in (3, 2, 1, 0)))
        active = active[order]
    maximum_altitude = float(maximum_altitude_m)
    if not math.isfinite(maximum_altitude):
        raise ValueError("fixed scene maximum altitude must be finite")

    def quantized(value: np.ndarray) -> list[object]:
        rounded = np.round(value, decimals=6)
        rounded[np.abs(rounded) < 0.5e-6] = 0.0
        return rounded.tolist()

    payload: dict[str, object] = {
        "version": TASK_SCENE_GEOMETRY_CONTRACT_VERSION,
        "coordinate_frame": "world_xy",
        "flight_bounds_xy": quantized(bounds),
        "maximum_altitude_m": round(maximum_altitude, 6),
        "static_obstacle_aabbs_world": quantized(active),
        "active_static_obstacle_count": int(active.shape[0]),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def fixed_task_scene_geometry_equivalent(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    absolute_tolerance_m: float = TASK_SCENE_GEOMETRY_TRANSPORT_ATOL_M,
) -> bool:
    """Compare one fixed scene across float32/JSON transport boundaries.

    The SHA is an immutable replay/checkpoint identity, but it is too strict
    for a live request: simulator AABBs are emitted as float64 while replay
    validation intentionally materializes them as float32.  At warehouse
    coordinates near 10--30 m that round trip can move a six-decimal rounded
    endpoint by one unit in its final decimal place.  Such micron-scale drift
    is representation noise, not a different MDP.  Compare canonical,
    order-independent geometry with a small absolute tolerance while still
    rejecting any physically meaningful layout change.
    """
    tolerance = float(absolute_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("scene geometry comparison tolerance must be finite")
    try:
        if (
            first.get("version") != TASK_SCENE_GEOMETRY_CONTRACT_VERSION
            or second.get("version") != TASK_SCENE_GEOMETRY_CONTRACT_VERSION
            or int(first["active_static_obstacle_count"])
            != int(second["active_static_obstacle_count"])
        ):
            return False
        first_bounds = np.asarray(first["flight_bounds_xy"], np.float64)
        second_bounds = np.asarray(second["flight_bounds_xy"], np.float64)
        first_obstacles = np.asarray(
            first["static_obstacle_aabbs_world"], np.float64).reshape(-1, 4)
        second_obstacles = np.asarray(
            second["static_obstacle_aabbs_world"], np.float64).reshape(-1, 4)
        first_altitude = float(first["maximum_altitude_m"])
        second_altitude = float(second["maximum_altitude_m"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        first_bounds.shape != (4,)
        or second_bounds.shape != (4,)
        or first_obstacles.shape != second_obstacles.shape
        or not np.isfinite(first_bounds).all()
        or not np.isfinite(second_bounds).all()
        or not np.isfinite(first_obstacles).all()
        or not np.isfinite(second_obstacles).all()
        or not math.isfinite(first_altitude)
        or not math.isfinite(second_altitude)
    ):
        return False
    return bool(
        np.allclose(
            first_bounds, second_bounds, rtol=0.0, atol=tolerance)
        and np.allclose(
            first_obstacles, second_obstacles,
            rtol=0.0, atol=tolerance)
        and math.isclose(
            first_altitude, second_altitude,
            rel_tol=0.0, abs_tol=tolerance)
    )


def pure_dreamer_task_support_contract(
    replay_task_contract: Mapping[str, Any],
    *,
    require_formal_action_support: bool = True,
) -> dict[str, object]:
    """Extract the immutable task identity used by release evidence.

    Replay population counts and observed extrema can grow during online
    collection.  They are deliberately excluded.  The returned fields are
    precisely those that change the MDP, the deployable support envelope, or
    the validity of action-effect identification.
    """
    if not isinstance(replay_task_contract, Mapping):
        raise ValueError("replay task contract must be a mapping")

    def finite_number(name: str) -> float:
        value = replay_task_contract.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"replay task contract lacks numeric {name}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"replay task contract has non-finite {name}")
        return result

    def nonnegative_integer(name: str) -> int:
        value = replay_task_contract.get(name)
        if (
            not isinstance(value, int) or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"replay task contract lacks non-negative integer {name}")
        return int(value)

    def declared_region(name: str) -> dict[str, list[float]]:
        value = replay_task_contract.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"replay task contract lacks {name}")
        result: dict[str, list[float]] = {}
        for axis in ("x_range_m", "y_range_m"):
            raw = value.get(axis)
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"{name}.{axis} must contain two bounds")
            bounds = [float(raw[0]), float(raw[1])]
            if not all(math.isfinite(item) for item in bounds) \
                    or bounds[0] >= bounds[1]:
                raise ValueError(f"{name}.{axis} bounds are invalid")
            result[axis] = bounds
        return result

    scene = replay_task_contract.get("fixed_scene_geometry_contract")
    if not isinstance(scene, Mapping):
        raise ValueError("replay task contract lacks fixed scene geometry")
    try:
        reconstructed_scene = fixed_task_scene_geometry_contract(
            scene["flight_bounds_xy"],
            scene["static_obstacle_aabbs_world"],
            maximum_altitude_m=float(scene["maximum_altitude_m"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fixed scene geometry is invalid") from error
    if (
        scene.get("version") != TASK_SCENE_GEOMETRY_CONTRACT_VERSION
        or scene.get("sha256") != reconstructed_scene["sha256"]
        or scene.get("active_static_obstacle_count")
        != reconstructed_scene["active_static_obstacle_count"]
    ):
        raise ValueError("fixed scene geometry identity is inconsistent")

    sample_rate_hz = finite_number("sample_rate_hz")
    minimum_interval_s = finite_number(
        "minimum_nonterminal_control_interval_s")
    maximum_interval_s = finite_number(
        "maximum_nonterminal_control_interval_s")
    interval_count = nonnegative_integer("ordinary_control_interval_count")
    if (
        sample_rate_hz <= 0.0
        or minimum_interval_s <= 0.0
        or minimum_interval_s > maximum_interval_s
        or interval_count <= 0
        or minimum_interval_s < 0.85 / sample_rate_hz
        or maximum_interval_s > 1.15 / sample_rate_hz
    ):
        raise ValueError("empirical control-period support is invalid")

    fixed_goal_altitude = finite_number("fixed_goal_altitude_agl_m")
    minimum_goal_altitude = finite_number("minimum_goal_altitude_agl_m")
    maximum_goal_altitude = finite_number("maximum_goal_altitude_agl_m")
    if not (
        math.isclose(
            fixed_goal_altitude, minimum_goal_altitude,
            rel_tol=0.0, abs_tol=1.0e-6,
        )
        and math.isclose(
            fixed_goal_altitude, maximum_goal_altitude,
            rel_tol=0.0, abs_tol=1.0e-6,
        )
    ):
        raise ValueError("goal altitude is not fixed in replay support")

    action_support = replay_task_contract.get("world_dynamics_action_support")
    if require_formal_action_support and (
        not isinstance(action_support, Mapping)
        or action_support.get("formal_ready") is not True
    ):
        raise ValueError("formal joint action support is not established")
    if replay_task_contract.get("learned_static_event_in_actor_return") is not False:
        raise ValueError("learned static Event still enters the Actor return")
    if replay_task_contract.get("privileged_geometry_policy_visible") is not False:
        raise ValueError("privileged geometry is marked policy-visible")

    maximum_static = nonnegative_integer("maximum_static_proxy_count")
    if maximum_static != int(
        reconstructed_scene["active_static_obstacle_count"]
    ):
        raise ValueError("fixed scene and static support counts disagree")

    maximum_humans = nonnegative_integer(
        "maximum_simultaneous_actor_humans")
    exact_human_capacity = nonnegative_integer(
        "exact_physical_human_capacity")
    if maximum_humans > exact_human_capacity:
        raise ValueError(
            "observed simultaneous Humans exceed exact physical capacity")

    result: dict[str, object] = {
        "version": PURE_DREAMER_TASK_SUPPORT_CONTRACT_VERSION,
        "sample_rate_hz": sample_rate_hz,
        "goal_radius_m": finite_number("goal_radius_m"),
        "fixed_goal_altitude_agl_m": fixed_goal_altitude,
        "goal_world_xy_region": declared_region(
            "task_goal_world_xy_support"),
        "initial_world_xy_region": declared_region(
            "task_initial_world_xy_support"),
        "crash_altitude_world_m": finite_number(
            "crash_altitude_world_m"),
        "maximum_horizontal_speed_mps": finite_number(
            "maximum_horizontal_speed_mps"),
        "static_ground_z_world": finite_number("static_ground_z_world"),
        "fixed_scene_geometry": {
            "version": reconstructed_scene["version"],
            "sha256": reconstructed_scene["sha256"],
            "active_static_obstacle_count": reconstructed_scene[
                "active_static_obstacle_count"],
        },
        "exact_physical_human_capacity": exact_human_capacity,
        "maximum_static_proxy_count": maximum_static,
        "static_geometry_contract": replay_task_contract.get(
            "static_geometry_contract"),
        "learned_static_event_in_actor_return": False,
        "privileged_geometry_policy_visible": False,
    }
    if require_formal_action_support:
        result["formal_joint_action_support"] = True
    return result


def pure_dreamer_empirical_task_support_contract(
    replay_task_contract: Mapping[str, Any],
    *,
    require_formal_action_support: bool = True,
) -> dict[str, object]:
    """Return mutable replay coverage separately from immutable task identity."""
    # Reuse the authoritative range, capacity, scene and action-support
    # validation above before exposing any empirical extrema to release tools.
    immutable = pure_dreamer_task_support_contract(
        replay_task_contract,
        require_formal_action_support=require_formal_action_support,
    )
    minimum_interval = float(
        replay_task_contract["minimum_nonterminal_control_interval_s"])
    maximum_interval = float(
        replay_task_contract["maximum_nonterminal_control_interval_s"])
    maximum_humans = replay_task_contract[
        "maximum_simultaneous_actor_humans"]
    if (
        not isinstance(maximum_humans, int)
        or isinstance(maximum_humans, bool)
        or maximum_humans < 0
        or maximum_humans > int(immutable["exact_physical_human_capacity"])
    ):
        raise ValueError("empirical Human support is outside Actor capacity")
    return {
        "version": PURE_DREAMER_EMPIRICAL_TASK_SUPPORT_VERSION,
        "ordinary_control_interval_range_s": [
            minimum_interval, maximum_interval],
        "maximum_simultaneous_actor_humans": int(maximum_humans),
    }


def task_physical_state_dim(maximum_obstacles: int) -> int:
    maximum_obstacles = int(maximum_obstacles)
    if maximum_obstacles < 0:
        raise ValueError("maximum obstacle count must be non-negative")
    return (
        TASK_PHYSICAL_BASE_DIM
        + maximum_obstacles * TASK_PHYSICAL_OBSTACLE_FIELDS
    )


def task_physical_state_numpy(
    position_world_xyz: np.ndarray,
    yaw_world_rad: np.ndarray | float,
    flight_bounds_xy: np.ndarray,
    static_obstacle_aabbs_xy: np.ndarray,
    static_obstacle_valid: np.ndarray | None = None,
    *,
    maximum_altitude_m: np.ndarray | float = 0.0,
    bounds_valid: np.ndarray | bool = True,
    maximum_altitude_valid: np.ndarray | bool = False,
    static_terminal_valid: np.ndarray | bool = False,
    maximum_obstacles: int = 256,
    metric_scale_m: float = TASK_GEOMETRY_MAX_RANGE_M,
) -> np.ndarray:
    """Complete fixed-width body-frame task state for deployment.

    Layout: boundary-valid, four ``[inward_normal_x_body,
    inward_normal_y_body,signed_clearance]`` planes, altitude-valid and signed
    ceiling clearance, a static-terminal-semantics bit, followed by
    distance-sorted AABBs as
    ``[valid,center_x_body,center_y_body,half_x,half_y,cos(-yaw),sin(-yaw)]``.
    Metric fields are divided by ``metric_scale_m``.  No valid obstacle is
    discarded; callers must version a larger model if the capacity is
    exceeded.
    """
    position = np.asarray(position_world_xyz, np.float32)
    if position.shape[-1] != 3:
        raise ValueError("position_world_xyz must end in three coordinates")
    leading = position.shape[:-1]
    yaw = np.broadcast_to(np.asarray(yaw_world_rad, np.float32), leading)
    bounds = np.broadcast_to(
        np.asarray(flight_bounds_xy, np.float32), (*leading, 4))
    obstacles = np.asarray(static_obstacle_aabbs_xy, np.float32)
    if obstacles.ndim == 2:
        obstacles = np.broadcast_to(obstacles, (*leading, *obstacles.shape))
    if obstacles.shape[:-2] != leading or obstacles.shape[-1] != 4:
        raise ValueError("static obstacle AABBs must be [...,K,4]")
    capacity = int(maximum_obstacles)
    if capacity < 0 or obstacles.shape[-2] > capacity:
        raise ValueError("static obstacle count exceeds Actor capacity")
    valid = (
        np.ones(obstacles.shape[:-1], np.bool_)
        if static_obstacle_valid is None else
        np.broadcast_to(
            np.asarray(static_obstacle_valid, np.bool_),
            obstacles.shape[:-1])
    )
    def scalar_field(value: object, dtype: np.dtype) -> np.ndarray:
        field = np.asarray(value, dtype)
        if field.shape == (*leading, 1):
            field = field[..., 0]
        return np.broadcast_to(field, leading)

    bound_valid = scalar_field(bounds_valid, np.bool_)
    altitude_valid = scalar_field(maximum_altitude_valid, np.bool_)
    static_terminal = scalar_field(static_terminal_valid, np.bool_)
    maximum_altitude = scalar_field(maximum_altitude_m, np.float32)
    scale = float(metric_scale_m)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("task physical metric scale must be positive")

    cosine, sine = np.cos(yaw), np.sin(yaw)
    normals_world = np.asarray(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        np.float32,
    )
    normal_x = (
        cosine[..., None] * normals_world[:, 0]
        + sine[..., None] * normals_world[:, 1]
    )
    normal_y = (
        -sine[..., None] * normals_world[:, 0]
        + cosine[..., None] * normals_world[:, 1]
    )
    edge_clearance = np.stack((
        position[..., 0] - bounds[..., 0],
        bounds[..., 1] - position[..., 0],
        position[..., 1] - bounds[..., 2],
        bounds[..., 3] - position[..., 1],
    ), -1) / scale
    planes = np.stack((normal_x, normal_y, edge_clearance), -1)
    planes = np.where(bound_valid[..., None, None], planes, 0.0)
    altitude_clearance = np.where(
        altitude_valid,
        (maximum_altitude - position[..., 2]) / scale,
        0.0,
    )
    base = np.concatenate((
        bound_valid.astype(np.float32)[..., None],
        np.clip(planes, -5.0, 5.0).reshape(*leading, 12),
        altitude_valid.astype(np.float32)[..., None],
        np.clip(altitude_clearance, -5.0, 5.0)[..., None],
        static_terminal.astype(np.float32)[..., None],
    ), -1)

    count = obstacles.shape[-2]
    if count:
        center = 0.5 * (obstacles[..., :2] + obstacles[..., 2:])
        half = 0.5 * (obstacles[..., 2:] - obstacles[..., :2])
        if np.any(valid & np.any(half < 0.0, axis=-1)):
            raise ValueError("static obstacle AABB bounds are reversed")
        delta = center - position[..., None, :2]
        center_body = np.stack((
            cosine[..., None] * delta[..., 0]
            + sine[..., None] * delta[..., 1],
            -sine[..., None] * delta[..., 0]
            + cosine[..., None] * delta[..., 1],
        ), -1) / scale
        lower, upper = obstacles[..., :2], obstacles[..., 2:]
        outside = np.maximum(
            np.maximum(lower - position[..., None, :2],
                       position[..., None, :2] - upper),
            0.0,
        )
        clearance = np.linalg.norm(outside, axis=-1)
        priority = np.where(valid, clearance, np.inf)
        order = np.argsort(priority, axis=-1, kind="stable")
        obstacle_state = np.concatenate((
            valid.astype(np.float32)[..., None],
            np.clip(center_body, -5.0, 5.0),
            np.clip(half / scale, 0.0, 5.0),
            np.broadcast_to(cosine[..., None, None], (*leading, count, 1)),
            np.broadcast_to(-sine[..., None, None], (*leading, count, 1)),
        ), -1)
        obstacle_state = np.take_along_axis(
            obstacle_state, order[..., None], axis=-2)
        ordered_valid = np.take_along_axis(valid, order, axis=-1)
        obstacle_state = np.where(
            ordered_valid[..., None], obstacle_state, 0.0)
    else:
        obstacle_state = np.zeros(
            (*leading, 0, TASK_PHYSICAL_OBSTACLE_FIELDS), np.float32)
    if count < capacity:
        obstacle_state = np.pad(
            obstacle_state,
            ((0, 0),) * len(leading)
            + ((0, capacity - count), (0, 0)),
        )
    result = np.concatenate((
        base, obstacle_state.reshape(
            *leading, capacity * TASK_PHYSICAL_OBSTACLE_FIELDS),
    ), -1).astype(np.float32, copy=False)
    expected = task_physical_state_dim(capacity)
    if result.shape[-1] != expected:
        raise RuntimeError("task physical state width is inconsistent")
    return result


def task_physical_state_torch(
    position_world_xyz: torch.Tensor,
    yaw_world_rad: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor | None = None,
    *,
    maximum_altitude_m: torch.Tensor | float = 0.0,
    bounds_valid: torch.Tensor | bool = True,
    maximum_altitude_valid: torch.Tensor | bool = False,
    static_terminal_valid: torch.Tensor | bool = False,
    maximum_obstacles: int = 256,
    metric_scale_m: float = TASK_GEOMETRY_MAX_RANGE_M,
) -> torch.Tensor:
    """Differentiable counterpart of :func:`task_physical_state_numpy`."""
    position = position_world_xyz.float()
    if position.shape[-1] != 3:
        raise ValueError("position_world_xyz must end in three coordinates")
    leading = position.shape[:-1]
    yaw = torch.broadcast_to(yaw_world_rad.to(position), leading)
    bounds = torch.broadcast_to(
        flight_bounds_xy.to(position), (*leading, 4))
    obstacles = static_obstacle_aabbs_xy.to(position)
    if obstacles.ndim == 2:
        obstacles = torch.broadcast_to(
            obstacles, (*leading, *obstacles.shape))
    if obstacles.shape[:-2] != leading or obstacles.shape[-1] != 4:
        raise ValueError("static obstacle AABBs must be [...,K,4]")
    capacity = int(maximum_obstacles)
    if capacity < 0 or obstacles.shape[-2] > capacity:
        raise ValueError("static obstacle count exceeds Actor capacity")
    valid = (
        torch.ones(obstacles.shape[:-1], dtype=torch.bool,
                   device=obstacles.device)
        if static_obstacle_valid is None else
        torch.broadcast_to(
            torch.as_tensor(static_obstacle_valid, device=position.device).bool(),
            obstacles.shape[:-1])
    )
    def scalar_field(value: object, *, boolean: bool) -> torch.Tensor:
        field = torch.as_tensor(
            value, device=position.device,
            dtype=None if boolean else position.dtype)
        field = field.bool() if boolean else field.to(position.dtype)
        if field.shape == (*leading, 1):
            field = field[..., 0]
        return torch.broadcast_to(field, leading)

    bound_valid = scalar_field(bounds_valid, boolean=True)
    altitude_valid = scalar_field(maximum_altitude_valid, boolean=True)
    static_terminal = scalar_field(static_terminal_valid, boolean=True)
    maximum_altitude = scalar_field(maximum_altitude_m, boolean=False)
    scale = float(metric_scale_m)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("task physical metric scale must be positive")

    cosine, sine = yaw.cos(), yaw.sin()
    normals_world = position.new_tensor(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))
    normal_x = (
        cosine[..., None] * normals_world[:, 0]
        + sine[..., None] * normals_world[:, 1])
    normal_y = (
        -sine[..., None] * normals_world[:, 0]
        + cosine[..., None] * normals_world[:, 1])
    edge_clearance = torch.stack((
        position[..., 0] - bounds[..., 0],
        bounds[..., 1] - position[..., 0],
        position[..., 1] - bounds[..., 2],
        bounds[..., 3] - position[..., 1],
    ), -1) / scale
    planes = torch.stack((normal_x, normal_y, edge_clearance), -1)
    planes = torch.where(
        bound_valid[..., None, None], planes, torch.zeros_like(planes))
    altitude_clearance = torch.where(
        altitude_valid,
        (maximum_altitude - position[..., 2]) / scale,
        torch.zeros_like(maximum_altitude),
    )
    base = torch.cat((
        bound_valid.to(position.dtype)[..., None],
        planes.clamp(-5.0, 5.0).flatten(-2),
        altitude_valid.to(position.dtype)[..., None],
        altitude_clearance.clamp(-5.0, 5.0)[..., None],
        static_terminal.to(position.dtype)[..., None],
    ), -1)

    count = obstacles.shape[-2]
    if count:
        center = 0.5 * (obstacles[..., :2] + obstacles[..., 2:])
        half = 0.5 * (obstacles[..., 2:] - obstacles[..., :2])
        if bool((valid & (half < 0.0).any(-1)).any()):
            raise ValueError("static obstacle AABB bounds are reversed")
        delta = center - position[..., None, :2]
        center_body = torch.stack((
            cosine[..., None] * delta[..., 0]
            + sine[..., None] * delta[..., 1],
            -sine[..., None] * delta[..., 0]
            + cosine[..., None] * delta[..., 1],
        ), -1) / scale
        lower, upper = obstacles[..., :2], obstacles[..., 2:]
        outside = torch.maximum(
            torch.maximum(lower - position[..., None, :2],
                          position[..., None, :2] - upper),
            torch.zeros_like(lower),
        )
        clearance = torch.linalg.vector_norm(outside, dim=-1)
        priority = clearance.masked_fill(~valid, torch.inf)
        order = torch.argsort(priority, dim=-1, stable=True)
        obstacle_state = torch.cat((
            valid.to(position.dtype)[..., None],
            center_body.clamp(-5.0, 5.0),
            (half / scale).clamp(0.0, 5.0),
            cosine[..., None, None].expand(*leading, count, 1),
            (-sine)[..., None, None].expand(*leading, count, 1),
        ), -1)
        obstacle_state = obstacle_state.gather(
            -2, order[..., None].expand(
                *order.shape, TASK_PHYSICAL_OBSTACLE_FIELDS))
        ordered_valid = valid.gather(-1, order)
        obstacle_state = (
            obstacle_state * ordered_valid[..., None].to(obstacle_state))
    else:
        obstacle_state = position.new_zeros(
            (*leading, 0, TASK_PHYSICAL_OBSTACLE_FIELDS))
    if count < capacity:
        obstacle_state = torch.nn.functional.pad(
            obstacle_state, (0, 0, 0, capacity - count))
    result = torch.cat((base, obstacle_state.flatten(-2)), -1)
    if result.shape[-1] != task_physical_state_dim(capacity):
        raise RuntimeError("task physical state width is inconsistent")
    return result


def _ray_angles_numpy(yaw_world: np.ndarray) -> np.ndarray:
    offsets = np.arange(TASK_GEOMETRY_DIM, dtype=np.float32) * (
        2.0 * np.pi / TASK_GEOMETRY_DIM)
    angles = np.asarray(yaw_world, np.float32)[..., None] + offsets
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def task_geometry_proximity_numpy(
    position_world_xy: np.ndarray,
    yaw_world_rad: np.ndarray | float,
    flight_bounds_xy: np.ndarray,
    static_obstacle_aabbs_xy: np.ndarray,
    static_obstacle_valid: np.ndarray | None = None,
    *,
    maximum_range_m: float = TASK_GEOMETRY_MAX_RANGE_M,
) -> np.ndarray:
    """Return eight body-ray proximities in ``[0, 1]``.

    Zero means no task geometry within ``maximum_range_m`` and one means the
    ray origin is on/outside a boundary or inside a static AABB. Flight bounds
    are ``[xmin,xmax,ymin,ymax]``; static AABBs are recorder-standard
    ``[xmin,ymin,xmax,ymax]``.
    """
    position = np.asarray(position_world_xy, np.float32)
    if position.shape[-1] != 2:
        raise ValueError("position_world_xy must end in two coordinates")
    leading = position.shape[:-1]
    yaw = np.broadcast_to(np.asarray(yaw_world_rad, np.float32), leading)
    bounds = np.broadcast_to(
        np.asarray(flight_bounds_xy, np.float32), (*leading, 4))
    obstacles = np.asarray(static_obstacle_aabbs_xy, np.float32)
    if obstacles.ndim == 2:
        obstacles = np.broadcast_to(obstacles, (*leading, *obstacles.shape))
    if obstacles.shape[:-2] != leading or obstacles.shape[-1] != 4:
        raise ValueError("static obstacle AABBs must be [...,K,4]")
    if static_obstacle_valid is None:
        obstacle_valid = np.ones(obstacles.shape[:-1], np.bool_)
    else:
        obstacle_valid = np.broadcast_to(
            np.asarray(static_obstacle_valid, np.bool_), obstacles.shape[:-1])

    directions = _ray_angles_numpy(yaw)
    flat_position = position.reshape(-1, 2)
    flat_directions = directions.reshape(-1, TASK_GEOMETRY_DIM, 2)
    flat_bounds = bounds.reshape(-1, 4)
    # NumPy cannot infer ``-1`` when the known obstacle dimension is zero
    # (``reshape(-1, 0, 4)``).  An empty map is a valid real-deployment case
    # when only a conservative virtual flight boundary is available.
    flat_rows = flat_position.shape[0]
    flat_obstacles = obstacles.reshape(
        flat_rows, obstacles.shape[-2], 4)
    flat_valid = obstacle_valid.reshape(flat_rows, obstacles.shape[-2])
    result = np.empty(
        (flat_position.shape[0], TASK_GEOMETRY_DIM), np.float32)
    eps = 1.0e-8
    for row, (origin, rays, box, aabbs, valid) in enumerate(zip(
        flat_position, flat_directions, flat_bounds,
        flat_obstacles, flat_valid, strict=True,
    )):
        aabbs = aabbs[valid]
        x_min, x_max, y_min, y_max = (float(value) for value in box)
        inside_bounds = (
            x_min <= float(origin[0]) <= x_max
            and y_min <= float(origin[1]) <= y_max)
        for ray_index, direction in enumerate(rays):
            if not inside_bounds:
                nearest = 0.0
            else:
                tx = (
                    (x_max - float(origin[0])) / float(direction[0])
                    if direction[0] > eps else
                    (x_min - float(origin[0])) / float(direction[0])
                    if direction[0] < -eps else math.inf)
                ty = (
                    (y_max - float(origin[1])) / float(direction[1])
                    if direction[1] > eps else
                    (y_min - float(origin[1])) / float(direction[1])
                    if direction[1] < -eps else math.inf)
                nearest = max(0.0, min(tx, ty))
            for obstacle in aabbs:
                # Recorder/Isaac obstacle order is the standard XYXY layout:
                # xmin, ymin, xmax, ymax.  Flight bounds intentionally use a
                # different xmin, xmax, ymin, ymax contract above.
                lower = obstacle[:2]
                upper = obstacle[2:]
                t_enter = -math.inf
                t_exit = math.inf
                hit = True
                for axis in range(2):
                    component = float(direction[axis])
                    coordinate = float(origin[axis])
                    if abs(component) <= eps:
                        if coordinate < lower[axis] or coordinate > upper[axis]:
                            hit = False
                            break
                        continue
                    first = (float(lower[axis]) - coordinate) / component
                    second = (float(upper[axis]) - coordinate) / component
                    t_enter = max(t_enter, min(first, second))
                    t_exit = min(t_exit, max(first, second))
                if hit and t_exit >= max(t_enter, 0.0):
                    nearest = min(nearest, max(0.0, t_enter))
            clearance = min(maximum_range_m, max(0.0, nearest))
            result[row, ray_index] = 1.0 - clearance / maximum_range_m
    return result.reshape(*leading, TASK_GEOMETRY_DIM)


def task_geometry_proximity_torch(
    position_world_xy: torch.Tensor,
    yaw_world_rad: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor | None = None,
    *,
    maximum_range_m: float = TASK_GEOMETRY_MAX_RANGE_M,
) -> torch.Tensor:
    """Vectorized differentiable counterpart of the NumPy representation.

    Flight bounds are ``[xmin,xmax,ymin,ymax]`` and static AABBs are
    ``[xmin,ymin,xmax,ymax]``.
    """
    position = position_world_xy.float()
    if position.shape[-1] != 2:
        raise ValueError("position_world_xy must end in two coordinates")
    leading = position.shape[:-1]
    yaw = torch.broadcast_to(yaw_world_rad.float(), leading)
    bounds = torch.broadcast_to(
        flight_bounds_xy.float(), (*leading, 4))
    obstacles = static_obstacle_aabbs_xy.float()
    if obstacles.ndim == 2:
        obstacles = torch.broadcast_to(
            obstacles, (*leading, *obstacles.shape))
    if obstacles.shape[:-2] != leading or obstacles.shape[-1] != 4:
        raise ValueError("static obstacle AABBs must be [...,K,4]")
    obstacle_valid = (
        torch.ones(obstacles.shape[:-1], dtype=torch.bool,
                   device=obstacles.device)
        if static_obstacle_valid is None else
        torch.broadcast_to(static_obstacle_valid.bool(), obstacles.shape[:-1])
    )

    offsets = torch.arange(
        TASK_GEOMETRY_DIM, device=position.device, dtype=position.dtype,
    ) * (2.0 * torch.pi / TASK_GEOMETRY_DIM)
    angles = yaw[..., None] + offsets
    directions = torch.stack((angles.cos(), angles.sin()), -1)
    eps = 1.0e-8

    x, y = position[..., 0, None], position[..., 1, None]
    dx, dy = directions[..., 0], directions[..., 1]
    x_min, x_max = bounds[..., 0, None], bounds[..., 1, None]
    y_min, y_max = bounds[..., 2, None], bounds[..., 3, None]
    inf = torch.full_like(dx, torch.inf)
    tx = torch.where(
        dx > eps, (x_max - x) / dx.clamp_min(eps),
        torch.where(
            dx < -eps, (x_min - x) / dx.clamp_max(-eps), inf))
    ty = torch.where(
        dy > eps, (y_max - y) / dy.clamp_min(eps),
        torch.where(
            dy < -eps, (y_min - y) / dy.clamp_max(-eps), inf))
    inside_bounds = (
        (position[..., 0] >= bounds[..., 0])
        & (position[..., 0] <= bounds[..., 1])
        & (position[..., 1] >= bounds[..., 2])
        & (position[..., 1] <= bounds[..., 3]))
    nearest = torch.minimum(tx, ty).clamp_min(0.0)
    nearest = torch.where(inside_bounds[..., None], nearest,
                          torch.zeros_like(nearest))

    if obstacles.shape[-2] > 0:
        origin = position[..., None, None, :]
        ray = directions[..., :, None, :]
        lower = obstacles[..., None, :, :][..., :2]
        upper = obstacles[..., None, :, :][..., 2:]
        parallel = ray.abs() <= eps
        parallel_inside = (origin >= lower) & (origin <= upper)
        safe_ray = torch.where(parallel, torch.ones_like(ray), ray)
        first = (lower - origin) / safe_ray
        second = (upper - origin) / safe_ray
        near_axis = torch.minimum(first, second)
        far_axis = torch.maximum(first, second)
        near_axis = torch.where(
            parallel & parallel_inside,
            torch.full_like(near_axis, -torch.inf), near_axis)
        far_axis = torch.where(
            parallel & parallel_inside,
            torch.full_like(far_axis, torch.inf), far_axis)
        parallel_outside = (parallel & ~parallel_inside).any(-1)
        enter = near_axis.amax(-1)
        leave = far_axis.amin(-1)
        hit = (
            ~parallel_outside
            & (leave >= torch.maximum(enter, torch.zeros_like(enter)))
            & obstacle_valid[..., None, :])
        distance = enter.clamp_min(0.0).masked_fill(~hit, torch.inf)
        nearest = torch.minimum(nearest, distance.amin(-1))
    clearance = nearest.clamp(0.0, float(maximum_range_m))
    return 1.0 - clearance / float(maximum_range_m)


def task_geometry_clearance_torch(
    position_world_xy: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor | None = None,
    *,
    maximum_range_m: float = TASK_GEOMETRY_MAX_RANGE_M,
) -> torch.Tensor:
    """Exact nearest 2-D clearance to bounds or conservative static AABBs.

    Bounds use ``[xmin,xmax,ymin,ymax]`` while recorder AABBs use the standard
    ``[xmin,ymin,xmax,ymax]`` layout.  A point outside the flight box or inside
    an obstacle has zero clearance.
    """
    position = position_world_xy.float()
    if position.shape[-1] != 2:
        raise ValueError("position_world_xy must end in two coordinates")
    leading = position.shape[:-1]
    bounds = torch.broadcast_to(flight_bounds_xy.float(), (*leading, 4))
    obstacles = static_obstacle_aabbs_xy.float()
    if obstacles.ndim == 2:
        obstacles = torch.broadcast_to(
            obstacles, (*leading, *obstacles.shape))
    if obstacles.shape[:-2] != leading or obstacles.shape[-1] != 4:
        raise ValueError("static obstacle AABBs must be [...,K,4]")
    valid = (
        torch.ones(obstacles.shape[:-1], dtype=torch.bool,
                   device=obstacles.device)
        if static_obstacle_valid is None else
        torch.broadcast_to(static_obstacle_valid.bool(), obstacles.shape[:-1])
    )

    x, y = position[..., 0], position[..., 1]
    inside_bounds = (
        (x >= bounds[..., 0]) & (x <= bounds[..., 1])
        & (y >= bounds[..., 2]) & (y <= bounds[..., 3])
    )
    bound_clearance = torch.stack((
        x - bounds[..., 0], bounds[..., 1] - x,
        y - bounds[..., 2], bounds[..., 3] - y,
    ), -1).amin(-1).clamp_min(0.0)
    bound_clearance = torch.where(
        inside_bounds, bound_clearance, torch.zeros_like(bound_clearance))

    if obstacles.shape[-2] == 0:
        obstacle_clearance = torch.full_like(bound_clearance, torch.inf)
    else:
        lower = obstacles[..., :2]
        upper = obstacles[..., 2:]
        delta = torch.maximum(
            torch.maximum(
                lower - position[..., None, :],
                position[..., None, :] - upper,
            ),
            torch.zeros_like(obstacles[..., :2]),
        )
        distance = torch.linalg.vector_norm(delta, dim=-1)
        obstacle_clearance = distance.masked_fill(~valid, torch.inf).amin(-1)
    return torch.minimum(bound_clearance, obstacle_clearance).clamp(
        0.0, float(maximum_range_m))


def static_obstacle_clearance_torch(
    position_world_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor | None = None,
    *,
    maximum_range_m: float = TASK_GEOMETRY_MAX_RANGE_M,
) -> torch.Tensor:
    """Nearest 2-D clearance to the recorded conservative static proxies."""
    position = position_world_xy.float()
    if position.shape[-1] != 2:
        raise ValueError("position_world_xy must end in two coordinates")
    leading = position.shape[:-1]
    obstacles = static_obstacle_aabbs_xy.float()
    if obstacles.ndim == 2:
        obstacles = torch.broadcast_to(
            obstacles, (*leading, *obstacles.shape))
    if obstacles.shape[:-2] != leading or obstacles.shape[-1] != 4:
        raise ValueError("static obstacle AABBs must be [...,K,4]")
    valid = (
        torch.ones(obstacles.shape[:-1], dtype=torch.bool,
                   device=obstacles.device)
        if static_obstacle_valid is None else
        torch.broadcast_to(static_obstacle_valid.bool(), obstacles.shape[:-1])
    )
    if obstacles.shape[-2] == 0:
        return torch.full(
            leading, float(maximum_range_m),
            dtype=position.dtype, device=position.device)
    lower, upper = obstacles[..., :2], obstacles[..., 2:]
    delta = torch.maximum(
        torch.maximum(
            lower - position[..., None, :],
            position[..., None, :] - upper,
        ),
        torch.zeros_like(obstacles[..., :2]),
    )
    distance = torch.linalg.vector_norm(delta, dim=-1)
    minimum = distance.masked_fill(~valid, torch.inf).amin(-1)
    return torch.where(
        valid.any(-1), minimum.clamp(0.0, float(maximum_range_m)),
        minimum.new_full(minimum.shape, float(maximum_range_m)),
    )


def episode_to_world_static_clearance_torch(
    ego_state: torch.Tensor,
    origin_xyz: torch.Tensor,
    origin_yaw: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor,
) -> torch.Tensor:
    """Static-proxy clearance for an episode-local Ego trajectory."""
    origin_xyz = origin_xyz.to(ego_state).expand(*ego_state.shape[:-1], 3)
    origin_yaw = origin_yaw.to(ego_state).reshape(*ego_state.shape[:-1])
    cosine, sine = origin_yaw.cos(), origin_yaw.sin()
    local_x, local_y = ego_state[..., 0], ego_state[..., 1]
    world_position = torch.stack((
        origin_xyz[..., 0] + cosine * local_x - sine * local_y,
        origin_xyz[..., 1] + sine * local_x + cosine * local_y,
    ), -1)
    return static_obstacle_clearance_torch(
        world_position,
        static_obstacle_aabbs_xy,
        static_obstacle_valid,
    )


def flight_bounds_signed_clearance_torch(
    position_world_xy: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
) -> torch.Tensor:
    """Exact signed clearance to an axis-aligned flight rectangle.

    The value is the minimum of ``x-xmin``, ``xmax-x``, ``y-ymin`` and
    ``ymax-y``.  It is non-negative exactly where the simulator watchdog
    considers the point in bounds and negative exactly where it terminates an
    episode as out of bounds.  Unlike task-geometry clearance, this function
    deliberately does not clamp away the outside sign or mix in approximate
    static-obstacle AABBs.
    """
    position = position_world_xy.float()
    if position.shape[-1] != 2:
        raise ValueError("position_world_xy must end in two coordinates")
    bounds = torch.broadcast_to(
        flight_bounds_xy.float(), (*position.shape[:-1], 4))
    x, y = position[..., 0], position[..., 1]
    return torch.stack((
        x - bounds[..., 0],
        bounds[..., 1] - x,
        y - bounds[..., 2],
        bounds[..., 3] - y,
    ), -1).amin(-1)


def episode_to_world_flight_bounds_signed_clearance_torch(
    ego_state: torch.Tensor,
    origin_xyz: torch.Tensor,
    origin_yaw: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
) -> torch.Tensor:
    """Exact flight-boundary clearance for episode-local imagined Ego14."""
    leading = ego_state.shape[:-1]
    origin = torch.broadcast_to(origin_xyz.to(ego_state), (*leading, 3))
    yaw = torch.broadcast_to(origin_yaw.to(ego_state), leading)
    cosine, sine = yaw.cos(), yaw.sin()
    local_x, local_y = ego_state[..., 0], ego_state[..., 1]
    world_position = torch.stack((
        origin[..., 0] + cosine * local_x - sine * local_y,
        origin[..., 1] + sine * local_x + cosine * local_y,
    ), -1)
    return flight_bounds_signed_clearance_torch(
        world_position, flight_bounds_xy)


def episode_to_world_geometry_torch(
    ego_state: torch.Tensor,
    origin_xyz: torch.Tensor,
    origin_yaw: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor,
) -> torch.Tensor:
    """Build task proximity from episode-local Ego14 and world metadata."""
    origin_xyz = origin_xyz.to(ego_state).expand(*ego_state.shape[:-1], 3)
    origin_yaw = origin_yaw.to(ego_state).reshape(
        *ego_state.shape[:-1])
    cosine, sine = origin_yaw.cos(), origin_yaw.sin()
    local_x, local_y = ego_state[..., 0], ego_state[..., 1]
    world_position = torch.stack((
        origin_xyz[..., 0] + cosine * local_x - sine * local_y,
        origin_xyz[..., 1] + sine * local_x + cosine * local_y,
    ), -1)
    relative_yaw = torch.atan2(ego_state[..., 12], ego_state[..., 13])
    return task_geometry_proximity_torch(
        world_position, origin_yaw + relative_yaw,
        flight_bounds_xy, static_obstacle_aabbs_xy,
        static_obstacle_valid,
    )


def episode_to_world_task_physical_state_torch(
    ego_state: torch.Tensor,
    origin_xyz: torch.Tensor,
    origin_yaw: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor,
    *,
    maximum_altitude_m: torch.Tensor | float,
    bounds_valid: torch.Tensor | bool,
    maximum_altitude_valid: torch.Tensor | bool,
    static_terminal_valid: torch.Tensor | bool,
    maximum_obstacles: int = 256,
) -> torch.Tensor:
    """Build the complete Actor/Critic task state from episode-local Ego14."""
    leading = ego_state.shape[:-1]
    origin = torch.broadcast_to(origin_xyz.to(ego_state), (*leading, 3))
    origin_angle = origin_yaw.to(ego_state)
    if origin_angle.shape == (*leading, 1):
        origin_angle = origin_angle[..., 0]
    origin_angle = torch.broadcast_to(origin_angle, leading)
    cosine, sine = origin_angle.cos(), origin_angle.sin()
    local_x, local_y = ego_state[..., 0], ego_state[..., 1]
    world_position = torch.stack((
        origin[..., 0] + cosine * local_x - sine * local_y,
        origin[..., 1] + sine * local_x + cosine * local_y,
        origin[..., 2] + ego_state[..., 2],
    ), -1)
    relative_yaw = torch.atan2(ego_state[..., 12], ego_state[..., 13])
    return task_physical_state_torch(
        world_position,
        origin_angle + relative_yaw,
        flight_bounds_xy,
        static_obstacle_aabbs_xy,
        static_obstacle_valid,
        maximum_altitude_m=maximum_altitude_m,
        bounds_valid=bounds_valid,
        maximum_altitude_valid=maximum_altitude_valid,
        static_terminal_valid=static_terminal_valid,
        maximum_obstacles=maximum_obstacles,
    )


def episode_to_world_task_clearance_torch(
    ego_state: torch.Tensor,
    origin_xyz: torch.Tensor,
    origin_yaw: torch.Tensor,
    flight_bounds_xy: torch.Tensor,
    static_obstacle_aabbs_xy: torch.Tensor,
    static_obstacle_valid: torch.Tensor,
) -> torch.Tensor:
    """Exact task clearance for an episode-local imagined Ego state."""
    origin_xyz = origin_xyz.to(ego_state).expand(*ego_state.shape[:-1], 3)
    origin_yaw = origin_yaw.to(ego_state).reshape(*ego_state.shape[:-1])
    cosine, sine = origin_yaw.cos(), origin_yaw.sin()
    local_x, local_y = ego_state[..., 0], ego_state[..., 1]
    world_position = torch.stack((
        origin_xyz[..., 0] + cosine * local_x - sine * local_y,
        origin_xyz[..., 1] + sine * local_x + cosine * local_y,
    ), -1)
    return task_geometry_clearance_torch(
        world_position,
        flight_bounds_xy,
        static_obstacle_aabbs_xy,
        static_obstacle_valid,
    )


def task_geometry_feature(
    proximity: torch.Tensor, feature_dim: int,
) -> torch.Tensor:
    """Deterministically tile local geometry into an existing feature width."""
    if proximity.shape[-1] != TASK_GEOMETRY_DIM:
        raise ValueError(
            f"task geometry must end in {TASK_GEOMETRY_DIM} rays")
    repeats = (int(feature_dim) + TASK_GEOMETRY_DIM - 1) // TASK_GEOMETRY_DIM
    return proximity.float().repeat_interleave(repeats, dim=-1)[
        ..., :int(feature_dim)
    ] * TASK_GEOMETRY_FEATURE_SCALE
