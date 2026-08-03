"""Pure-Python schema helpers for skeleton/state dataset v2.

This module deliberately has no Isaac Sim or carb imports so chunks can be
validated and converted on training machines without an Isaac installation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


SCHEMA = "omninxt.crowd_skeleton_state.v2"
SKELETON_SCHEMA = "omninxt.skeleton3d.v1"
JOINT_COUNT = 17
JOINT_FIELDS = (
    "x_m", "y_m", "z_m", "pose_score", "confidence",
    "coordinate_valid", "measured", "predicted",
    "measurement_sigma_m", "measurement_age_ms", "source_code",
)
EGO_STATE_KEYS = (
    "local_x", "local_y", "local_z", "body_vx", "body_vy", "world_vz",
    "body_ax", "body_ay", "world_az", "altitude_agl", "roll", "pitch",
    "sin_relative_yaw", "cos_relative_yaw",
)
ACTION_KEYS = (
    "vx_body_mps", "vy_body_mps", "vz_world_mps", "yaw_rate_rps",
)
REWARD_COMPONENT_KEYS = (
    "event", "progress", "human_clearance", "smoothness", "height", "time",
)
TERMINATION_CODES = {
    "recording": 0,
    "manual_stop": 1,
    "collision": 2,
    "reached_goal": 3,
    "stuck_timeout": 4,
    "record_size_limit": 5,
    "shutdown": 6,
}
STATE_SOURCE_CODES = {
    "invalid": 0,
    "mavsdk_px4_telemetry": 1,
    "isaac_ground_truth": 2,
    "isaac_ground_truth_fallback": 2,
}


def _vec(mapping: Mapping[str, Any], vector_key: str,
         scalar_keys: Sequence[str], size: int = 3) -> np.ndarray:
    value = mapping.get(vector_key)
    if value is None:
        value = [mapping.get(key, 0.0) for key in scalar_keys]
    source = np.asarray(value, dtype=np.float64).reshape(-1)
    output = np.zeros(size, dtype=np.float64)
    output[:min(size, source.size)] = source[:size]
    output[~np.isfinite(output)] = 0.0
    return output


def _world_xy_to_heading(vector: Sequence[float], yaw: float) -> np.ndarray:
    x, y = float(vector[0]), float(vector[1])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray((c * x + s * y, -s * x + c * y), dtype=np.float64)


def ego_reference(drone_state: Mapping[str, Any]) -> dict[str, Any]:
    position = _vec(drone_state, "position", ("x", "y", "z"))
    rpy = _vec(
        drone_state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    return {
        "origin_xyz": position.astype(float).tolist(),
        "origin_yaw": float(rpy[2]),
    }


def build_ego_state(drone_state: Mapping[str, Any],
                    reference: Mapping[str, Any]) -> np.ndarray:
    position = _vec(drone_state, "position", ("x", "y", "z"))
    velocity = _vec(drone_state, "velocity", ("vx", "vy", "vz"))
    acceleration = _vec(drone_state, "acceleration", ("ax", "ay", "az"))
    rpy = _vec(
        drone_state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    origin = np.asarray(reference["origin_xyz"], dtype=np.float64)
    origin_yaw = float(reference["origin_yaw"])
    delta = position - origin
    local_xy = _world_xy_to_heading(delta[:2], origin_yaw)
    body_velocity_xy = _world_xy_to_heading(velocity[:2], rpy[2])
    body_acceleration_xy = _world_xy_to_heading(acceleration[:2], rpy[2])
    relative_yaw = (float(rpy[2]) - origin_yaw + math.pi) % (
        2.0 * math.pi) - math.pi
    altitude = drone_state.get("altitude_agl")
    altitude = 0.0 if altitude is None else float(altitude)
    output = np.asarray((
        local_xy[0], local_xy[1], delta[2],
        body_velocity_xy[0], body_velocity_xy[1], velocity[2],
        body_acceleration_xy[0], body_acceleration_xy[1], acceleration[2],
        altitude, rpy[0], rpy[1],
        math.sin(relative_yaw), math.cos(relative_yaw),
    ), dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite ego state")
    return output


def goal_relative_body(drone_state: Mapping[str, Any],
                       target_point: Sequence[float]) -> np.ndarray:
    position = _vec(drone_state, "position", ("x", "y", "z"))
    rpy = _vec(
        drone_state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    relative = np.asarray(target_point, dtype=np.float64).reshape(3) - position
    body_xy = _world_xy_to_heading(relative[:2], rpy[2])
    return np.asarray((body_xy[0], body_xy[1], relative[2],
                       np.linalg.norm(relative)), dtype=np.float32)


def empty_skeleton_packet(timestamp_ns: int, sequence: int = -1) -> dict[str, Any]:
    return {
        "schema": SKELETON_SCHEMA,
        "sequence": int(sequence),
        "timestamp_ns": int(timestamp_ns),
        "frame_id": "base_link",
        "joint_names": ["joint_{}".format(index) for index in range(JOINT_COUNT)],
        "joint_fields": list(JOINT_FIELDS),
        "people": [],
    }


def skeleton_to_arrays(packet: Mapping[str, Any], max_people: int,
                       last_seen: MutableMapping[int, int],
                       frame_index: int) -> dict[str, np.ndarray]:
    if packet.get("schema") != SKELETON_SCHEMA:
        raise ValueError("Unexpected skeleton schema {}".format(packet.get("schema")))
    if tuple(packet.get("joint_fields", ())) != JOINT_FIELDS:
        raise ValueError("Unexpected skeleton joint fields")
    if len(packet.get("joint_names", ())) != JOINT_COUNT:
        raise ValueError("Expected COCO17 skeleton")
    count = max(1, int(max_people))
    shape = (count, JOINT_COUNT)
    arrays = {
        "human_xyz": np.zeros(shape + (3,), np.float32),
        "human_pose_score": np.zeros(shape, np.float32),
        "human_confidence": np.zeros(shape, np.float32),
        "human_joint_valid": np.zeros(shape, np.bool_),
        "human_joint_measured": np.zeros(shape, np.bool_),
        "human_joint_predicted": np.zeros(shape, np.bool_),
        "human_measurement_sigma_m": np.zeros(shape, np.float32),
        "human_measurement_age_ms": np.zeros(shape, np.float32),
        "human_source_code": np.zeros(shape, np.uint8),
        "human_track_id": np.full(count, -1, np.int32),
        "human_mask": np.zeros(count, np.bool_),
        "human_is_first": np.zeros(count, np.bool_),
    }
    people = sorted(packet.get("people", ()), key=lambda item: int(item["person_id"]))
    for slot, person in enumerate(people[:count]):
        track_id = int(person["person_id"])
        if track_id <= 0:
            raise ValueError("Skeleton track IDs must be positive")
        rows = np.asarray(person.get("joints", ()), dtype=np.float32)
        if rows.shape != (JOINT_COUNT, len(JOINT_FIELDS)):
            raise ValueError("Invalid joint array for track {}".format(track_id))
        finite = np.isfinite(rows).all(axis=1)
        valid = rows[:, 5].astype(bool) & finite
        arrays["human_xyz"][slot] = rows[:, :3]
        arrays["human_xyz"][slot, ~valid] = 0.0
        arrays["human_pose_score"][slot] = rows[:, 3]
        arrays["human_confidence"][slot] = rows[:, 4]
        arrays["human_joint_valid"][slot] = valid
        arrays["human_joint_measured"][slot] = rows[:, 6].astype(bool) & valid
        arrays["human_joint_predicted"][slot] = rows[:, 7].astype(bool) & valid
        arrays["human_measurement_sigma_m"][slot] = np.maximum(rows[:, 8], 0.0)
        arrays["human_measurement_age_ms"][slot] = np.maximum(rows[:, 9], 0.0)
        arrays["human_source_code"][slot] = np.clip(rows[:, 10], 0, 255).astype(np.uint8)
        arrays["human_track_id"][slot] = track_id
        arrays["human_mask"][slot] = bool(np.any(valid))
        arrays["human_is_first"][slot] = last_seen.get(track_id) != frame_index - 1
        last_seen[track_id] = frame_index
    arrays["human_count"] = np.asarray(min(len(people), count), np.int16)
    arrays["human_truncated_count"] = np.asarray(
        max(0, len(people) - count), np.int16)
    return arrays


def normalize_action(action: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    action = action if isinstance(action, Mapping) else {}
    normalized = np.asarray(action.get("normalized", np.zeros(4)), np.float32).reshape(-1)
    requested = action.get("requested") if isinstance(action.get("requested"), Mapping) else {}
    requested_values = np.asarray(
        [requested.get(key, 0.0) for key in ACTION_KEYS], dtype=np.float32)
    applied = action.get("applied_body_flu")
    applied_valid = isinstance(applied, Mapping)
    applied_values = np.asarray(
        [applied.get(key, 0.0) for key in ACTION_KEYS] if applied_valid else np.zeros(4),
        dtype=np.float32,
    )
    fixed_normalized = np.zeros(4, np.float32)
    fixed_normalized[:min(4, normalized.size)] = normalized[:4]
    return {
        "prev_action_requested_norm": fixed_normalized,
        "prev_action_requested_physical": requested_values,
        "prev_action_applied": applied_values,
        "prev_action_applied_valid": np.asarray(applied_valid, np.bool_),
    }


def privileged_to_arrays(snapshot: Mapping[str, Any] | None, max_people: int,
                         joint_count: int) -> dict[str, np.ndarray]:
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    drone = snapshot.get("drone") if isinstance(snapshot.get("drone"), Mapping) else {}
    count, joints = max(1, int(max_people)), max(1, int(joint_count))
    output = {
        "priv_drone_position_world": _vec(drone, "position", ("x", "y", "z")).astype(np.float32),
        "priv_drone_velocity_world": _vec(drone, "velocity", ("vx", "vy", "vz")).astype(np.float32),
        "priv_drone_acceleration_world": _vec(drone, "acceleration", ("ax", "ay", "az")).astype(np.float32),
        "priv_drone_quaternion_xyzw": _vec(drone, "quaternion_xyzw", ("qx", "qy", "qz", "qw"), 4).astype(np.float32),
        "priv_human_id": np.full(count, -1, np.int32),
        "priv_human_group_id": np.full(count, -1, np.int32),
        "priv_human_mask": np.zeros(count, np.bool_),
        "priv_human_position_world": np.zeros((count, 3), np.float32),
        "priv_human_velocity_world": np.zeros((count, 3), np.float32),
        "priv_human_pelvis_world": np.zeros((count, 3), np.float32),
        "priv_human_pelvis_valid": np.zeros(count, np.bool_),
        "priv_collision_joints_world": np.zeros((count, joints, 3), np.float32),
        "priv_collision_joint_valid": np.zeros((count, joints), np.bool_),
    }
    people = snapshot.get("people") if isinstance(snapshot.get("people"), Sequence) else ()
    for slot, person in enumerate(people[:count]):
        if not isinstance(person, Mapping):
            continue
        output["priv_human_id"][slot] = int(person.get("id", slot))
        group_id = person.get("group_id")
        output["priv_human_group_id"][slot] = -1 if group_id is None else int(group_id)
        output["priv_human_mask"][slot] = True
        output["priv_human_position_world"][slot] = _vec(
            person, "position", ("x", "y", "z"))
        output["priv_human_velocity_world"][slot] = _vec(
            person, "velocity", ("vx", "vy", "vz"))
        pelvis = person.get("pelvis")
        if pelvis is not None:
            pelvis_array = np.asarray(pelvis, np.float32).reshape(-1)
            if pelvis_array.size >= 3 and np.isfinite(pelvis_array[:3]).all():
                output["priv_human_pelvis_world"][slot] = pelvis_array[:3]
                output["priv_human_pelvis_valid"][slot] = True
        joint_values = np.asarray(person.get("collision_joints", ()), np.float32)
        joint_valid = np.asarray(person.get("collision_joint_valid", ()), np.bool_)
        if joint_values.shape == (joints, 3):
            valid = (joint_valid if joint_valid.shape == (joints,)
                     else np.isfinite(joint_values).all(axis=1))
            output["priv_collision_joints_world"][slot] = np.nan_to_num(joint_values)
            output["priv_collision_joint_valid"][slot] = valid
    return output


def stack_samples(samples: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("Cannot stack an empty sample list")
    keys = tuple(samples[0].keys())
    if any(tuple(sample.keys()) != keys for sample in samples):
        raise ValueError("Chunk samples have inconsistent keys or ordering")
    arrays = {key: np.stack([np.asarray(sample[key]) for sample in samples], axis=0)
              for key in keys}
    validate_chunk(arrays)
    return arrays


def validate_chunk(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "frame_index", "simulation_time_s", "episode_time_s", "dt_s",
        "skeleton_timestamp_ns", "skeleton_sequence", "skeleton_time_offset_ms",
        "ego_state", "ego_quaternion_xyzw", "human_xyz", "human_track_id",
        "human_mask", "human_joint_valid", "prev_action_applied", "reward",
        "reward_components", "is_first", "is_terminal", "discount",
        "priv_human_position_world", "priv_collision_joints_world",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError("Missing v2 chunk arrays: {}".format(sorted(missing)))
    length = int(np.asarray(arrays["frame_index"]).shape[0])
    if length <= 0:
        raise ValueError("Empty dataset chunk")
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.shape[0] != length:
            raise ValueError("{} has inconsistent time dimension".format(name))
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError("{} contains NaN/Inf".format(name))
    if np.asarray(arrays["ego_state"]).shape != (length, len(EGO_STATE_KEYS)):
        raise ValueError("ego_state must be [T,14]")
    if np.asarray(arrays["human_xyz"]).shape[2:] != (JOINT_COUNT, 3):
        raise ValueError("human_xyz must be [T,N,17,3]")
    if np.any(np.diff(np.asarray(arrays["frame_index"], np.int64)) != 1):
        raise ValueError("frame_index must be contiguous within each chunk")


def write_npz_atomic(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        # Skeleton-only chunks are small; uncompressed NPZ minimizes simulator
        # CPU spikes and compression can be done offline if desired.
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
