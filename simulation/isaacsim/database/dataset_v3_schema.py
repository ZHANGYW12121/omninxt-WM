"""Pure-Python helpers for compact skeleton/state dataset v3.

The simulator records only non-derivable model inputs plus reward/debug truth.
Training-side velocity, stable slots and goal features are derived offline.
This module intentionally has no Isaac Sim or carb imports.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "omninxt.crowd_skeleton_state.v3"
SKELETON_SCHEMA = "omninxt.skeleton3d.v1"
SOURCE_JOINT_COUNT = 17
# COCO17 indices 0..4 are Nose/Eyes/Ears. The recorded/model topology keeps
# only shoulders, elbows, wrists, hips, knees and ankles (source indices 5..16).
BODY_JOINT_INDICES = tuple(range(5, 17))
JOINT_COUNT = len(BODY_JOINT_INDICES)
BODY_JOINT_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
PRIVILEGED_COLLISION_JOINT_NAMES = (
    "Pelvis", "R_Hand", "L_Hand", "R_Foot", "L_Foot",
    "R_KneeShareBone", "L_KneeShareBone",
    "R_ElbowShareBone", "L_ElbowShareBone", "Head",
)
PRIVILEGED_COLLISION_JOINT_COUNT = len(PRIVILEGED_COLLISION_JOINT_NAMES)
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
STATE_SOURCE_CODES = {
    "invalid": 0,
    "mavsdk_px4_telemetry": 1,
    "isaac_ground_truth": 2,
    "isaac_ground_truth_fallback": 2,
}
TERMINATION_CODES = {
    "recording": 0,
    "reached_goal": 1,
    "human_collision": 2,
    "static_collision": 3,
    "out_of_bounds": 4,
    "crash": 5,
    "stuck_timeout": 6,
    "time_limit": 7,
    "controller_error": 8,
    "manual_stop": 9,
    "shutdown": 10,
    "record_size_limit": 11,
}
TASK_TERMINAL_REASONS = frozenset({
    "reached_goal", "human_collision", "static_collision",
    "out_of_bounds", "crash", "stuck_timeout",
})
TASK_FAILURE_REASONS = TASK_TERMINAL_REASONS.difference({"reached_goal"})
TRUNCATION_REASONS = frozenset({
    "time_limit", "manual_stop", "shutdown", "record_size_limit",
})


def outcome_for_reason(reason: str) -> str:
    reason = str(reason)
    if reason == "reached_goal":
        return "success"
    if reason in TASK_FAILURE_REASONS:
        return "task_failure"
    if reason in TRUNCATION_REASONS:
        return "truncated"
    if reason == "recording":
        return "ongoing"
    return "invalid"


def transition_flags(reason: str) -> dict[str, np.ndarray]:
    reason = str(reason)
    is_last = reason != "recording"
    return {
        "success": np.asarray(reason == "reached_goal", np.bool_),
        "termination_code": np.asarray(TERMINATION_CODES.get(reason, 255), np.uint8),
        "is_last": np.asarray(is_last, np.bool_),
        "is_terminal": np.asarray(is_last and reason in TASK_TERMINAL_REASONS, np.bool_),
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
    rpy = _vec(drone_state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    return {"origin_xyz": position.tolist(), "origin_yaw": float(rpy[2])}


def goal_position_episode_local(target_point: Sequence[float],
                                reference: Mapping[str, Any]) -> np.ndarray:
    target = np.asarray(target_point, dtype=np.float64).reshape(3)
    origin = np.asarray(reference["origin_xyz"], dtype=np.float64).reshape(3)
    delta = target - origin
    local_xy = _world_xy_to_heading(delta[:2], float(reference["origin_yaw"]))
    return np.asarray((local_xy[0], local_xy[1], delta[2]), np.float32)


def build_ego_state(drone_state: Mapping[str, Any],
                    reference: Mapping[str, Any]) -> np.ndarray:
    position = _vec(drone_state, "position", ("x", "y", "z"))
    velocity = _vec(drone_state, "velocity", ("vx", "vy", "vz"))
    acceleration = _vec(drone_state, "acceleration", ("ax", "ay", "az"))
    rpy = _vec(drone_state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    origin = np.asarray(reference["origin_xyz"], dtype=np.float64)
    origin_yaw = float(reference["origin_yaw"])
    delta = position - origin
    local_xy = _world_xy_to_heading(delta[:2], origin_yaw)
    body_velocity_xy = _world_xy_to_heading(velocity[:2], rpy[2])
    body_acceleration_xy = _world_xy_to_heading(acceleration[:2], rpy[2])
    relative_yaw = (float(rpy[2]) - origin_yaw + math.pi) % (2 * math.pi) - math.pi
    altitude = drone_state.get("altitude_agl")
    altitude = 0.0 if altitude is None else float(altitude)
    output = np.asarray((
        local_xy[0], local_xy[1], delta[2],
        body_velocity_xy[0], body_velocity_xy[1], velocity[2],
        body_acceleration_xy[0], body_acceleration_xy[1], acceleration[2],
        altitude, rpy[0], rpy[1], math.sin(relative_yaw), math.cos(relative_yaw),
    ), dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite ego state")
    return output


def empty_skeleton_packet(timestamp_ns: int, sequence: int = -1) -> dict[str, Any]:
    return {
        "schema": SKELETON_SCHEMA,
        "sequence": int(sequence),
        "timestamp_ns": int(timestamp_ns),
        "frame_id": "base_link",
        "joint_names": [f"joint_{index}" for index in range(SOURCE_JOINT_COUNT)],
        "joint_fields": list(JOINT_FIELDS),
        "people": [],
    }


def skeleton_to_arrays(packet: Mapping[str, Any], people_capacity: int) -> dict[str, np.ndarray]:
    if packet.get("schema") != SKELETON_SCHEMA:
        raise ValueError(f"Unexpected skeleton schema {packet.get('schema')}")
    if tuple(packet.get("joint_fields", ())) != JOINT_FIELDS:
        raise ValueError("Unexpected skeleton joint fields")
    if len(packet.get("joint_names", ())) != SOURCE_JOINT_COUNT:
        raise ValueError("Expected COCO17 skeleton")
    count = max(1, int(people_capacity))
    shape = (count, JOINT_COUNT)
    arrays = {
        "human_xyz": np.zeros(shape + (3,), np.float32),
        "human_confidence": np.zeros(shape, np.float32),
        "human_joint_valid": np.zeros(shape, np.bool_),
        "human_joint_measured": np.zeros(shape, np.bool_),
        "human_joint_predicted": np.zeros(shape, np.bool_),
        "human_track_id": np.full(count, -1, np.int32),
        "human_root_velocity": np.zeros((count, 3), np.float32),
        "human_velocity_valid": np.zeros(count, np.bool_),
        "human_velocity_sigma_mps": np.zeros(count, np.float32),
        "human_measurement_age_s": np.zeros(count, np.float32),
        "human_track_age_frames": np.zeros(count, np.int32),
        "human_prediction_run_frames": np.zeros(count, np.int32),
        "human_identity_confidence": np.zeros(count, np.float32),
    }
    people = sorted(packet.get("people", ()), key=lambda item: int(item["person_id"]))
    for slot, person in enumerate(people[:count]):
        track_id = int(person["person_id"])
        if track_id < 0:
            raise ValueError("Skeleton track IDs must be non-negative")
        rows = np.asarray(person.get("joints", ()), dtype=np.float32)
        if rows.shape != (SOURCE_JOINT_COUNT, len(JOINT_FIELDS)):
            raise ValueError(f"Invalid joint array for track {track_id}")
        rows = rows[np.asarray(BODY_JOINT_INDICES)]
        finite_xyz = np.isfinite(rows[:, :3]).all(axis=1)
        valid = rows[:, 5].astype(bool) & finite_xyz
        arrays["human_xyz"][slot] = np.where(valid[:, None], rows[:, :3], 0.0)
        arrays["human_confidence"][slot] = np.where(
            valid, np.clip(np.nan_to_num(rows[:, 4]), 0.0, 1.0), 0.0)
        arrays["human_joint_valid"][slot] = valid
        arrays["human_joint_measured"][slot] = (
            rows[:, 6].astype(bool) & valid)
        arrays["human_joint_predicted"][slot] = (
            rows[:, 7].astype(bool) & valid)
        arrays["human_track_id"][slot] = track_id
        velocity = np.asarray(
            person.get("root_velocity_base_link_mps", (0.0, 0.0, 0.0)),
            np.float32).reshape(-1)
        velocity_valid = bool(person.get("velocity_valid", False))
        if velocity.shape == (3,) and np.isfinite(velocity).all():
            arrays["human_root_velocity"][slot] = velocity
        else:
            velocity_valid = False
        sigma = person.get("velocity_sigma_mps")
        sigma = 0.0 if sigma is None else float(sigma)
        if not np.isfinite(sigma) or sigma < 0.0:
            sigma = 0.0
            velocity_valid = False
        arrays["human_velocity_valid"][slot] = velocity_valid
        arrays["human_velocity_sigma_mps"][slot] = sigma
        arrays["human_measurement_age_s"][slot] = max(
            0.0, float(person.get("time_since_observation_ms", 0.0)) * 1.0e-3)
        arrays["human_track_age_frames"][slot] = max(
            0, int(person.get("track_age_frames", 0)))
        arrays["human_prediction_run_frames"][slot] = max(
            0, int(person.get("consecutive_prediction_frames", 0)))
        arrays["human_identity_confidence"][slot] = np.clip(
            float(person.get("identity_confidence", 0.0)), 0.0, 1.0)
    return arrays


def skeleton_source_counts(packet: Mapping[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for person in packet.get("people", ()):
        rows = np.asarray(person.get("joints", ()), dtype=np.float32)
        if rows.shape != (SOURCE_JOINT_COUNT, len(JOINT_FIELDS)):
            continue
        rows = rows[np.asarray(BODY_JOINT_INDICES)]
        valid = rows[:, 5].astype(bool) & np.isfinite(rows[:, :3]).all(axis=1)
        for source in rows[valid, 10].astype(np.int64):
            key = int(source)
            counts[key] = counts.get(key, 0) + 1
    return counts


def normalized_applied_action(action: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    action = action if isinstance(action, Mapping) else {}
    applied = action.get("applied_body_flu")
    valid = isinstance(applied, Mapping)
    values = np.asarray(
        [applied.get(key, 0.0) for key in ACTION_KEYS] if valid else np.zeros(4),
        dtype=np.float32,
    )
    raw_limits = action.get("normalization_limits")
    limits_map = raw_limits if isinstance(raw_limits, Mapping) else {}
    limits = np.asarray([limits_map.get(key, 1.0) for key in ACTION_KEYS], np.float32)
    if not np.isfinite(values).all() or not np.isfinite(limits).all() or np.any(limits <= 0):
        valid = False
        values = np.zeros(4, np.float32)
        limits = np.ones(4, np.float32)
    metadata = action.get("policy_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    policy_action = np.asarray(
        metadata.get("policy_action", np.zeros(3)), np.float32).reshape(-1)
    policy_valid = bool(metadata.get("policy_action_valid", False))
    if policy_action.shape != (3,) or not np.isfinite(policy_action).all():
        policy_action = np.zeros(3, np.float32)
        policy_valid = False
    macro_index = metadata.get("exploration_macro_index")
    macro_remaining = metadata.get("exploration_macro_remaining")
    return {
        "action": np.clip(values / limits, -1.0, 1.0).astype(np.float32),
        "action_valid": np.asarray(valid, np.bool_),
        "policy_action": np.clip(policy_action, -1.0, 1.0),
        "policy_action_valid": np.asarray(policy_valid, np.bool_),
        "exploration_macro_index": np.asarray(
            -1 if macro_index is None else int(macro_index), np.int16),
        "exploration_macro_remaining": np.asarray(
            0 if macro_remaining is None else int(macro_remaining), np.uint8),
        "exploration_scale": np.asarray(
            float(metadata.get("exploration_scale", 0.0)), np.float32),
        "collector_policy_step": np.asarray(
            int(metadata.get("collector_policy_step", -1)), np.int64),
    }


def privileged_to_arrays(snapshot: Mapping[str, Any] | None,
                         people_capacity: int) -> dict[str, np.ndarray]:
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    count = max(1, int(people_capacity))
    output = {
        "priv_human_id": np.full(count, -1, np.int32),
        "priv_human_mask": np.zeros(count, np.bool_),
        "priv_human_position_world": np.zeros((count, 3), np.float32),
        "priv_human_velocity_world": np.zeros((count, 3), np.float32),
        # Simulator-only animation-skeleton truth. These audit targets must
        # never enter the deployment observation encoder.
        "priv_human_pelvis_world": np.zeros((count, 3), np.float32),
        "priv_human_pelvis_valid": np.zeros(count, np.bool_),
        "priv_collision_joints_world": np.zeros(
            (count, PRIVILEGED_COLLISION_JOINT_COUNT, 3), np.float32),
        "priv_collision_joint_valid": np.zeros(
            (count, PRIVILEGED_COLLISION_JOINT_COUNT), np.bool_),
    }
    people = snapshot.get("people") if isinstance(snapshot.get("people"), Sequence) else ()
    for slot, person in enumerate(people[:count]):
        if not isinstance(person, Mapping):
            continue
        output["priv_human_id"][slot] = int(person.get("id", slot))
        output["priv_human_mask"][slot] = True
        output["priv_human_position_world"][slot] = _vec(
            person, "position", ("x", "y", "z"))
        output["priv_human_velocity_world"][slot] = _vec(
            person, "velocity", ("vx", "vy", "vz"))
        pelvis = np.asarray(person.get("pelvis", ()), np.float32).reshape(-1)
        if pelvis.size >= 3 and np.isfinite(pelvis[:3]).all():
            output["priv_human_pelvis_world"][slot] = pelvis[:3]
            output["priv_human_pelvis_valid"][slot] = True
        joints = np.asarray(person.get("collision_joints", ()), np.float32)
        joint_valid = np.asarray(
            person.get("collision_joint_valid", ()), np.bool_)
        if joints.shape == (PRIVILEGED_COLLISION_JOINT_COUNT, 3):
            if joint_valid.shape != (PRIVILEGED_COLLISION_JOINT_COUNT,):
                joint_valid = np.isfinite(joints).all(axis=-1)
            joint_valid &= np.isfinite(joints).all(axis=-1)
            output["priv_collision_joints_world"][slot] = np.nan_to_num(joints)
            output["priv_collision_joint_valid"][slot] = joint_valid
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
        "frame_index", "simulation_time_s", "skeleton_timestamp_ns", "skeleton_fresh",
        "ego_state", "ego_altitude_valid", "ego_state_source", "human_xyz",
        "human_confidence", "human_joint_valid", "human_track_id", "action",
        "human_root_velocity", "human_velocity_valid",
        "human_velocity_sigma_mps", "human_measurement_age_s",
        "human_track_age_frames", "human_prediction_run_frames",
        "human_identity_confidence",
        "action_valid", "reward", "reward_components", "is_first", "is_terminal",
        "success", "termination_code", "is_last",
        "priv_human_id", "priv_human_mask", "priv_human_position_world",
        "priv_human_velocity_world", "priv_min_human_clearance_m",
        "priv_min_human_clearance_valid", "priv_min_human_ttc_s",
        "priv_min_human_ttc_valid", "priv_collision", "priv_goal_distance_m",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Missing v3 chunk arrays: {sorted(missing)}")
    privileged_geometry = {
        "priv_human_pelvis_world", "priv_human_pelvis_valid",
        "priv_collision_joints_world", "priv_collision_joint_valid",
    }
    present_privileged_geometry = privileged_geometry.intersection(arrays)
    if present_privileged_geometry and present_privileged_geometry != privileged_geometry:
        raise ValueError(
            "privileged pelvis/collision-joint fields must appear together")
    length = int(np.asarray(arrays["frame_index"]).shape[0])
    if length <= 0:
        raise ValueError("Empty dataset chunk")
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.shape[0] != length:
            raise ValueError(f"{name} has inconsistent time dimension")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/Inf")
    people = np.asarray(arrays["human_xyz"]).shape[1]
    if np.asarray(arrays["ego_state"]).shape != (length, len(EGO_STATE_KEYS)):
        raise ValueError("ego_state must be [T,14]")
    if np.asarray(arrays["human_xyz"]).shape != (length, people, JOINT_COUNT, 3):
        raise ValueError("human_xyz must be [T,N,12,3]")
    if np.asarray(arrays["human_confidence"]).shape != (length, people, JOINT_COUNT):
        raise ValueError("human_confidence must be [T,N,12]")
    if np.asarray(arrays["human_joint_valid"]).shape != (length, people, JOINT_COUNT):
        raise ValueError("human_joint_valid must be [T,N,12]")
    if present_privileged_geometry:
        privileged_people = np.asarray(arrays["priv_human_id"]).shape[1]
        if np.asarray(arrays["priv_human_pelvis_world"]).shape != (
            length, privileged_people, 3
        ):
            raise ValueError("priv_human_pelvis_world must be [T,P,3]")
        if np.asarray(arrays["priv_human_pelvis_valid"]).shape != (
            length, privileged_people
        ):
            raise ValueError("priv_human_pelvis_valid must be [T,P]")
        if np.asarray(arrays["priv_collision_joints_world"]).shape != (
            length, privileged_people, PRIVILEGED_COLLISION_JOINT_COUNT, 3
        ):
            raise ValueError("priv_collision_joints_world has invalid shape")
        if np.asarray(arrays["priv_collision_joint_valid"]).shape != (
            length, privileged_people, PRIVILEGED_COLLISION_JOINT_COUNT
        ):
            raise ValueError("priv_collision_joint_valid has invalid shape")
    quality_fields = {"human_joint_measured", "human_joint_predicted"}
    present_quality = quality_fields.intersection(arrays)
    if present_quality and present_quality != quality_fields:
        raise ValueError("measured/predicted joint fields must appear together")
    if present_quality:
        measured = np.asarray(arrays["human_joint_measured"], np.bool_)
        predicted = np.asarray(arrays["human_joint_predicted"], np.bool_)
        valid = np.asarray(arrays["human_joint_valid"], np.bool_)
        if measured.shape != valid.shape or predicted.shape != valid.shape:
            raise ValueError("joint quality fields must be [T,N,12]")
        if np.any(measured & predicted):
            raise ValueError("a joint cannot be measured and predicted")
        if np.any(valid != (measured | predicted)):
            raise ValueError("every valid joint needs one measurement source")
    if np.asarray(arrays["human_track_id"]).shape != (length, people):
        raise ValueError("human_track_id must be [T,N]")
    if np.asarray(arrays["human_root_velocity"]).shape != (length, people, 3):
        raise ValueError("human_root_velocity must be [T,N,3]")
    for name in (
        "human_velocity_valid", "human_velocity_sigma_mps",
        "human_measurement_age_s", "human_track_age_frames",
        "human_prediction_run_frames", "human_identity_confidence",
    ):
        if np.asarray(arrays[name]).shape != (length, people):
            raise ValueError(f"{name} must be [T,N]")
    if np.asarray(arrays["action"]).shape != (length, len(ACTION_KEYS)):
        raise ValueError("action must be [T,4]")
    if np.any(np.abs(np.asarray(arrays["action"])) > 1.00001):
        raise ValueError("normalized action must lie in [-1,1]")
    if (
        "policy_action" in arrays
        and np.asarray(arrays["policy_action"]).shape != (length, 3)
    ):
        raise ValueError("policy_action must be [T,3]")
    for name in (
        "policy_action_valid", "exploration_macro_index",
        "exploration_macro_remaining", "exploration_scale",
        "collector_policy_step",
    ):
        if name in arrays and np.asarray(arrays[name]).shape != (length,):
            raise ValueError(f"{name} must be [T]")
    for name in (
        "action_valid", "reward", "is_first", "success",
        "termination_code", "is_last", "is_terminal",
    ):
        if np.asarray(arrays[name]).shape != (length,):
            raise ValueError(f"{name} must be [T]")
    if np.asarray(arrays["reward_components"]).shape != (
            length, len(REWARD_COMPONENT_KEYS)):
        raise ValueError("reward_components must be [T,6]")
    if np.any(np.diff(np.asarray(arrays["frame_index"], np.int64)) != 1):
        raise ValueError("frame_index must be contiguous within each chunk")


def write_npz_atomic(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
