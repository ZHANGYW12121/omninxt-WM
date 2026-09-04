"""Dataset-side schema helpers for the factorized Dreamer world model.

This module intentionally does not reinterpret legacy 2D RTMPose coordinates
as 3D.  Production human observations must provide ``keypoints_xyz`` (and may
optionally provide velocities/confidence); the old ``keypoints_xyc`` cache can
continue to be consumed by :mod:`datasets.isaac_crowd`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from modules.goal_conditioning import GOAL_FEATURE_DIM, goal_features_numpy
from modules.skeleton_topology import (
    COCO12_BODY_JOINT_COUNT, coco12_robust_root_numpy, hip_joint_indices,
)

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    Dataset = object  # type: ignore[misc,assignment]


EGO14_KEYS = (
    "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
    "altitude", "roll", "pitch", "sin_yaw", "cos_yaw",
)


def compute_ego_state_statistics(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute training-split-only Ego14 mean/std for encoder configuration."""
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1, 14) for value in sequences]
    if not arrays or sum(value.shape[0] for value in arrays) == 0:
        raise ValueError("At least one ego_state row is required")
    merged = np.concatenate(arrays, axis=0)
    if not np.isfinite(merged).all():
        raise ValueError("ego_state statistics input contains NaN/Inf")
    return merged.mean(0).astype(np.float32), merged.std(0).clip(1e-6).astype(np.float32)


@dataclass(frozen=True)
class Ego14Reference:
    """Episode-local origin plus optional world-Z ground height."""

    origin_xyz: tuple[float, float, float]
    origin_yaw: float
    ground_z: float | None = None


def _vec3(state: Mapping[str, Any], vector_key: str, scalar_keys: Sequence[str]) -> np.ndarray:
    value = state.get(vector_key)
    if value is None:
        value = [state.get(key, 0.0) for key in scalar_keys]
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    fixed = np.zeros(3, dtype=np.float32)
    fixed[: min(3, out.size)] = out[:3]
    fixed[~np.isfinite(fixed)] = 0.0
    return fixed


def ego14_reference_from_frame(
    frame: Mapping[str, Any], *, ground_z: float | None = None,
) -> Ego14Reference:
    state = frame.get("drone_state") or {}
    if not isinstance(state, Mapping):
        state = {}
    pos = _vec3(state, "position", ("x", "y", "z"))
    rpy = _vec3(state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    return Ego14Reference(tuple(float(x) for x in pos), float(rpy[2]), ground_z)


def _rotate_world_xy_to_heading(vector: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world XY vector into a yaw-aligned local/body frame."""

    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    x, y = float(vector[0]), float(vector[1])
    return np.asarray((c * x + s * y, -s * x + c * y), dtype=np.float32)


def _altitude_agl(state: Mapping[str, Any], pos_z: float,
                  reference: Ego14Reference, strict: bool) -> float:
    for key in ("altitude_agl", "height_above_ground", "altitude"):
        if key in state and state[key] is not None:
            return float(state[key])
    if "ground_z" in state and state["ground_z"] is not None:
        return float(pos_z - float(state["ground_z"]))
    if reference.ground_z is not None:
        return float(pos_z - reference.ground_z)
    if strict:
        raise ValueError(
            "Missing altitude AGL: record altitude_agl/ground_z or provide Ego14Reference.ground_z"
        )
    # Compatibility placeholder only. Unlike the old fallback this is not a
    # duplicate of local z: it means height relative to the episode start Z.
    return float(pos_z - reference.origin_xyz[2])


def ego14_from_frame(
    frame: Mapping[str, Any], reference: Ego14Reference | None = None, *,
    strict_altitude: bool = False,
) -> np.ndarray:
    """Build the guide-specified 14D UAV state from one Isaac frame.

    Position is expressed in the episode-start frame. Horizontal velocity and
    acceleration are expressed in the current yaw/body-heading frame. Yaw is
    relative to episode-start yaw. ``altitude`` means height above local ground
    when available; compatibility mode falls back to height relative to the
    episode start and never duplicates world/local Z silently.
    """

    state = frame.get("drone_state") or {}
    if not isinstance(state, Mapping):
        state = {}
    pos = _vec3(state, "position", ("x", "y", "z"))
    vel = _vec3(state, "velocity", ("vx", "vy", "vz"))
    acc = _vec3(state, "acceleration", ("ax", "ay", "az"))
    rpy = _vec3(state, "roll_pitch_yaw_rad", ("roll", "pitch", "yaw"))
    yaw = float(rpy[2])
    if reference is None:
        reference = Ego14Reference((0.0, 0.0, 0.0), 0.0, None)
    origin = np.asarray(reference.origin_xyz, dtype=np.float32)
    delta = pos - origin
    local_xy = _rotate_world_xy_to_heading(delta[:2], reference.origin_yaw)
    body_vel_xy = _rotate_world_xy_to_heading(vel[:2], yaw)
    body_acc_xy = _rotate_world_xy_to_heading(acc[:2], yaw)
    yaw_rel = (yaw - reference.origin_yaw + math.pi) % (2.0 * math.pi) - math.pi
    altitude = _altitude_agl(state, float(pos[2]), reference, strict_altitude)
    out = np.asarray([
        float(local_xy[0]), float(local_xy[1]), float(delta[2]),
        float(body_vel_xy[0]), float(body_vel_xy[1]), float(vel[2]),
        float(body_acc_xy[0]), float(body_acc_xy[1]), float(acc[2]),
        altitude, float(rpy[0]), float(rpy[1]),
        math.sin(yaw_rel), math.cos(yaw_rel),
    ], dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite value in drone_state while building ego_state[14]")
    return out


def _target_world_xyz(frame: Mapping[str, Any]) -> np.ndarray:
    """Read the task target without silently replacing a missing target."""
    for key in ("target_point", "goal_point", "goal", "target"):
        value = frame.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key in ("position", "point", "xyz"):
                if sub_key in value:
                    value = value[sub_key]
                    break
            else:
                continue
        target = np.asarray(value, dtype=np.float32).reshape(-1)
        if target.size >= 3 and np.isfinite(target[:3]).all():
            return target[:3].copy()
    raise ValueError("Frame is missing a finite 3-D target/goal point")


def goal_position_from_frame(frame: Mapping[str, Any],
                             reference: Ego14Reference) -> np.ndarray:
    """Return the fixed target in the episode-start local coordinate frame."""
    target = _target_world_xyz(frame)
    delta = target - np.asarray(reference.origin_xyz, dtype=np.float32)
    xy = _rotate_world_xy_to_heading(delta[:2], reference.origin_yaw)
    return np.asarray((xy[0], xy[1], delta[2]), dtype=np.float32)


def human_root_and_relative_joints_numpy(
    skeleton: np.ndarray, human_mask: np.ndarray, joint_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dataset equivalent of the Human root/pose split used by the encoder."""
    skel = np.asarray(skeleton, dtype=np.float32)
    hmask = np.asarray(human_mask, dtype=np.bool_)
    jmask = np.asarray(joint_mask, dtype=np.bool_) & hmask[..., None]
    xyz, velocity, confidence = skel[..., :3], skel[..., 3:6], skel[..., 6]
    weight = jmask[..., None].astype(np.float32)
    denom = np.maximum(weight.sum(axis=-2), 1.0)
    mean_pos = (xyz * weight).sum(axis=-2) / denom
    mean_vel = (velocity * weight).sum(axis=-2) / denom
    hips = hip_joint_indices(skel.shape[-2])
    if skel.shape[-2] == COCO12_BODY_JOINT_COUNT:
        root_pos = coco12_robust_root_numpy(xyz, jmask)
        left_hip, right_hip = 6, 7
        hips_valid = jmask[..., left_hip] & jmask[..., right_hip]
        hip_pos = 0.5 * (xyz[..., left_hip, :] + xyz[..., right_hip, :])
        hip_vel = 0.5 * (
            velocity[..., left_hip, :] + velocity[..., right_hip, :])
        robust_hips = hips_valid & (
            np.linalg.norm(root_pos - hip_pos, axis=-1) <= 1.0e-5)
        root_vel = np.where(robust_hips[..., None], hip_vel, mean_vel)
    elif hips is not None:
        left_hip, right_hip = hips
        hips_valid = jmask[..., left_hip] & jmask[..., right_hip]
        hip_pos = 0.5 * (xyz[..., left_hip, :] + xyz[..., right_hip, :])
        hip_vel = 0.5 * (velocity[..., left_hip, :] + velocity[..., right_hip, :])
        root_pos = np.where(hips_valid[..., None], hip_pos, mean_pos)
        root_vel = np.where(hips_valid[..., None], hip_vel, mean_vel)
    else:
        root_pos, root_vel = mean_pos, mean_vel
    low = np.where(jmask[..., None], xyz, np.inf).min(axis=-2)
    high = np.where(jmask[..., None], xyz, -np.inf).max(axis=-2)
    any_joint = jmask.any(axis=-1)
    extent = np.where(any_joint[..., None], high - low, 0.0)
    root_confidence = (
        (confidence * jmask.astype(np.float32)).sum(axis=-1)
        / np.maximum(jmask.sum(axis=-1), 1)
    )
    root = np.concatenate((root_pos, root_vel, extent, root_confidence[..., None]), -1)
    root[~hmask] = 0.0
    relative = skel.copy()
    relative[..., :3] -= root_pos[..., None, :]
    relative[..., 3:6] -= root_vel[..., None, :]
    relative[~jmask] = 0.0
    return root.astype(np.float32), relative.astype(np.float32)


@dataclass(frozen=True)
class HumanDetections:
    """Unslotted people detected in one frame, in arbitrary detector order."""

    ids: np.ndarray                 # [M], positive stable track IDs
    skeleton: np.ndarray            # [M,J,F]
    joint_mask: np.ndarray          # [M,J]
    risk_score: np.ndarray          # [M], larger means retain first


@dataclass(frozen=True)
class StableHumanSlots:
    skeleton: np.ndarray            # [T,N,J,F]
    human_mask: np.ndarray           # [T,N]
    joint_mask: np.ndarray           # [T,N,J]
    human_ids: np.ndarray            # [T,N], -1 for padding
    human_is_first: np.ndarray       # [T,N]
    truncated_people: np.ndarray     # [T]


def _validate_detections(detection: HumanDetections, joint_count: int, feat_dim: int) -> None:
    ids = np.asarray(detection.ids).reshape(-1)
    skeleton = np.asarray(detection.skeleton)
    mask = np.asarray(detection.joint_mask)
    risk = np.asarray(detection.risk_score).reshape(-1)
    m = ids.size
    if skeleton.shape != (m, joint_count, feat_dim):
        raise ValueError(f"skeleton must be [{m},{joint_count},{feat_dim}], got {skeleton.shape}")
    if mask.shape != (m, joint_count) or risk.shape != (m,):
        raise ValueError("joint_mask/risk_score shape does not match ids")
    if m and ((ids <= 0).any() or np.unique(ids).size != m):
        raise ValueError("Each frame requires unique positive human track IDs")
    if not np.isfinite(skeleton).all() or not np.isfinite(risk).all():
        raise ValueError("Human detections contain NaN/Inf")


def assign_stable_human_slots(
    frames: Sequence[HumanDetections], *, max_people: int, joint_count: int,
    feat_dim: int,
) -> StableHumanSlots:
    """Assign stable sequence slots and generate reset masks.

    Candidates are deterministically ranked by descending risk.  A retained ID
    reuses its previous/preferred slot whenever available; detector ordering
    therefore cannot permute Human RSSM state.  A slot reset is emitted after
    absence or whenever its ID changes.
    """

    t_len, n = len(frames), int(max_people)
    if n <= 0 or joint_count <= 0 or feat_dim <= 0:
        raise ValueError("max_people, joint_count and feat_dim must be positive")
    skeleton = np.zeros((t_len, n, joint_count, feat_dim), np.float32)
    human_mask = np.zeros((t_len, n), np.bool_)
    joint_mask = np.zeros((t_len, n, joint_count), np.bool_)
    human_ids = np.full((t_len, n), -1, np.int64)
    human_is_first = np.zeros((t_len, n), np.bool_)
    truncated = np.zeros(t_len, np.int32)
    preferred_slot: dict[int, int] = {}

    for t, detection in enumerate(frames):
        _validate_detections(detection, joint_count, feat_dim)
        ids = np.asarray(detection.ids, dtype=np.int64).reshape(-1)
        risk = np.asarray(detection.risk_score, dtype=np.float64).reshape(-1)
        # ID is the tie breaker, making selection independent of detector order.
        order = np.lexsort((ids, -risk))[:n]
        truncated[t] = max(0, ids.size - order.size)
        selected = [int(i) for i in order]
        used: set[int] = set()
        assignment: dict[int, int] = {}

        # Preserve known locations first, then fill free slots by risk rank.
        for i in selected:
            track_id = int(ids[i])
            slot = preferred_slot.get(track_id)
            if slot is not None and slot not in used:
                assignment[i] = slot
                used.add(slot)
        free = iter(slot for slot in range(n) if slot not in used)
        for i in selected:
            if i not in assignment:
                assignment[i] = next(free)

        for i in selected:
            slot = assignment[i]
            track_id = int(ids[i])
            preferred_slot[track_id] = slot
            human_mask[t, slot] = True
            human_ids[t, slot] = track_id
            joint_mask[t, slot] = np.asarray(detection.joint_mask[i], dtype=np.bool_)
            skeleton[t, slot] = np.asarray(detection.skeleton[i], dtype=np.float32)
            skeleton[t, slot, ~joint_mask[t, slot]] = 0.0
            # Reappearance after a missing frame is a new recurrent segment.
            human_is_first[t, slot] = t == 0 or not human_mask[t - 1, slot] or human_ids[t - 1, slot] != track_id

    result = StableHumanSlots(skeleton, human_mask, joint_mask, human_ids,
                              human_is_first, truncated)
    validate_factorized_humans(result)
    return result


def detections_from_3d_pose_cache(path: str | Path) -> HumanDetections:
    """Read a strict 3D pose-cache NPZ without accepting legacy 2D coordinates.

    Required keys: ``keypoints_xyz [M,J,3]`` and ``track_ids [M]``.
    Optional keys: ``velocities_xyz``, ``confidence``, ``joint_mask``, and
    ``risk_score``.  The resulting per-joint feature is always
    ``[x,y,z,vx,vy,vz,confidence]``.
    """

    with np.load(Path(path), allow_pickle=False) as pose:
        if "keypoints_xyz" not in pose:
            if "keypoints_xyc" in pose:
                raise ValueError("Legacy 2D keypoints_xyc cache cannot be used as a 3D skeleton")
            raise KeyError("3D pose cache requires keypoints_xyz")
        xyz = np.asarray(pose["keypoints_xyz"], dtype=np.float32)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"keypoints_xyz must be [M,J,3], got {xyz.shape}")
        m, j, _ = xyz.shape
        ids = np.asarray(pose["track_ids"], dtype=np.int64).reshape(-1)
        velocity = np.asarray(pose["velocities_xyz"], dtype=np.float32) if "velocities_xyz" in pose else np.zeros_like(xyz)
        confidence = np.asarray(pose["confidence"], dtype=np.float32) if "confidence" in pose else np.ones((m, j), np.float32)
        mask = np.asarray(pose["joint_mask"], dtype=np.bool_) if "joint_mask" in pose else confidence > 0.0
        risk = np.asarray(pose["risk_score"], dtype=np.float32).reshape(-1) if "risk_score" in pose else confidence.mean(1)
    if ids.shape != (m,) or velocity.shape != xyz.shape or confidence.shape != (m, j) or mask.shape != (m, j):
        raise ValueError("3D pose-cache arrays have inconsistent shapes")
    return HumanDetections(ids, np.concatenate((xyz, velocity, confidence[..., None]), -1), mask, risk)


def validate_factorized_humans(data: StableHumanSlots) -> None:
    """Fail early on schema errors that would leak or mix Human RSSM slots."""

    skel, mask, joints = data.skeleton, data.human_mask, data.joint_mask
    ids, first = data.human_ids, data.human_is_first
    if skel.ndim != 4 or mask.shape != skel.shape[:2] or joints.shape != skel.shape[:3]:
        raise ValueError("Invalid factorized human tensor shapes")
    if ids.shape != mask.shape or first.shape != mask.shape:
        raise ValueError("human_ids/human_is_first must match human_mask")
    if not np.isfinite(skel).all():
        raise ValueError("skeleton contains NaN/Inf")
    if (joints & ~mask[..., None]).any() or (first & ~mask).any():
        raise ValueError("joint_mask/human_is_first cannot be true for padding")
    if (ids[mask] <= 0).any() or (ids[~mask] != -1).any():
        raise ValueError("Active humans need positive IDs; padding IDs must be -1")
    if (skel[~joints] != 0).any():
        raise ValueError("Invalid/padded joints must be exactly zero")
    if mask.shape[0] > 1:
        continuation = mask[1:] & mask[:-1] & (ids[1:] == ids[:-1])
        if (first[1:] & continuation).any():
            raise ValueError("Continuous same-ID slots must not reset")
        requires_reset = mask[1:] & ~continuation
        if (requires_reset & ~first[1:]).any():
            raise ValueError("New/reappearing/reassigned IDs must reset human state")


def validate_factorized_batch(batch: Mapping[str, np.ndarray]) -> None:
    """Validate core observation fields before conversion to torch tensors."""

    ego = np.asarray(batch["ego_state"])
    if ego.ndim != 2 or ego.shape[-1] != 14 or not np.isfinite(ego).all():
        raise ValueError("ego_state must be finite [T,14]")
    yaw_norm = np.linalg.norm(ego[:, 12:14], axis=-1)
    if not np.allclose(yaw_norm, 1.0, atol=1e-4):
        raise ValueError("ego_state sin_yaw/cos_yaw must lie on the unit circle")
    goal = np.asarray(batch["goal"])
    goal_position = np.asarray(batch["goal_position"])
    if goal.shape != (ego.shape[0], GOAL_FEATURE_DIM) or not np.isfinite(goal).all():
        raise ValueError(f"goal must be finite [T,{GOAL_FEATURE_DIM}]")
    if goal_position.shape != (ego.shape[0], 3) or not np.isfinite(goal_position).all():
        raise ValueError("goal_position must be finite [T,3]")
    data = StableHumanSlots(
        np.asarray(batch["skeleton"]), np.asarray(batch["human_mask"]),
        np.asarray(batch["joint_mask"]), np.asarray(batch["human_ids"]),
        np.asarray(batch["human_is_first"]),
        np.asarray(batch.get("truncated_people", np.zeros(ego.shape[0], np.int32))),
    )
    if data.skeleton.shape[0] != ego.shape[0]:
        raise ValueError("Ego and human sequences must have the same T")
    validate_factorized_humans(data)
    t_len = ego.shape[0]
    for key in ("action", "reward", "is_first", "is_last", "is_terminal"):
        if key in batch and np.asarray(batch[key]).shape[0] != t_len:
            raise ValueError(f"{key} must have the same T={t_len} as ego_state")
    for key in ("action", "reward"):
        if key in batch and not np.isfinite(np.asarray(batch[key])).all():
            raise ValueError(f"{key} contains NaN/Inf")
    if "priv_collision_human_id" in batch:
        collision_human_id = np.asarray(batch["priv_collision_human_id"])
        if collision_human_id.shape != (t_len, 1):
            raise ValueError("priv_collision_human_id must have shape [T,1]")
    optional_shapes = {
        "human_joint_measured": data.joint_mask.shape,
        "human_joint_predicted": data.joint_mask.shape,
        "measured_joint_target_valid": data.joint_mask.shape,
        "measured_root_target_valid": data.human_mask.shape,
        "measured_velocity_target_valid": data.human_mask.shape,
        "human_motion_valid": data.human_mask.shape,
        "human_survival_valid": data.human_mask.shape,
        "human_survival_target": data.human_mask.shape,
        "human_birth_target": data.human_mask.shape,
        "human_birth_valid": data.human_mask.shape,
        "human_persistence_mask": data.human_mask.shape,
        "human_gt_id": data.human_mask.shape,
        "human_gt_match_valid": data.human_mask.shape,
        "human_gt_match_error_m": data.human_mask.shape,
        "human_gt_match_confidence": data.human_mask.shape,
        "human_gt_identity_id": data.human_mask.shape,
        "human_gt_identity_valid": data.human_mask.shape,
        "human_gt_identity_age_s": data.human_mask.shape,
    }
    for key, expected in optional_shapes.items():
        if key in batch and np.asarray(batch[key]).shape != expected:
            raise ValueError(f"{key} must have shape {expected}")
    if "human_gt_identity_age_s" in batch:
        identity_age = np.asarray(batch["human_gt_identity_age_s"])
        if not np.isfinite(identity_age).all() or np.any(identity_age < 0.0):
            raise ValueError(
                "human_gt_identity_age_s must be finite and non-negative")
    if "human_observation_quality" in batch:
        quality = np.asarray(batch["human_observation_quality"])
        if quality.shape != (*data.human_mask.shape, 7) or not np.isfinite(quality).all():
            raise ValueError("human_observation_quality must be finite [T,N,7]")
    for key in ("human_gt_pelvis_body", "human_gt_pelvis_episode"):
        if key in batch:
            value = np.asarray(batch[key])
            if value.shape != (*data.human_mask.shape, 3) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite [T,N,3]")
    if "measured_root_target" in batch:
        root = np.asarray(batch["measured_root_target"])
        if root.shape != (*data.human_mask.shape, 6) or not np.isfinite(root).all():
            raise ValueError("measured_root_target must be finite [T,N,6]")
    if "measured_joint_target" in batch:
        joints_target = np.asarray(batch["measured_joint_target"])
        if joints_target.shape != (*data.joint_mask.shape, 3) or not np.isfinite(joints_target).all():
            raise ValueError("measured_joint_target must be finite [T,N,J,3]")
    if "human_joint_measured" in batch and "human_joint_predicted" in batch:
        measured = np.asarray(batch["human_joint_measured"], np.bool_)
        predicted = np.asarray(batch["human_joint_predicted"], np.bool_)
        if np.any(measured & predicted):
            raise ValueError("a Human joint cannot be measured and predicted simultaneously")
        if np.any((measured | predicted) & ~data.joint_mask):
            raise ValueError("measurement provenance cannot mark an invalid joint")


class FactorizedIsaacAdapter(Dataset):  # type: ignore[misc]
    """Add factorized-model fields to an existing Isaac sequence dataset.

    The wrapped dataset remains the owner of sequence slicing, actions,
    rewards and termination flags.  It must return ``record_name`` and
    ``frame_ids``.  This adapter reads raw frame JSON to construct Ego14 and
    Goal features, and reads a separate strict 3D pose cache.
    """

    def __init__(self, base_dataset: Any, pose_3d_cache_root: str | Path, *,
                 max_people: int = 20, num_joints: int = 12,
                 require_pose_cache: bool = True,
                 strict_altitude: bool = False,
                 ground_z_by_record: Mapping[str, float] | None = None) -> None:
        self.base_dataset = base_dataset
        self.pose_3d_cache_root = Path(pose_3d_cache_root).expanduser()
        self.max_people = int(max_people)
        self.num_joints = int(num_joints)
        self.require_pose_cache = bool(require_pose_cache)
        self.strict_altitude = bool(strict_altitude)
        self.ground_z_by_record = dict(ground_z_by_record or {})
        episodes = getattr(base_dataset, "episodes", None)
        if episodes is None:
            raise ValueError("base_dataset must expose its episodes")
        self.record_dirs = {str(ep.record_name): Path(ep.record_dir) for ep in episodes}
        self.ego_references: dict[str, Ego14Reference] = {}
        import json
        for record_name, record_dir in self.record_dirs.items():
            episode = next(ep for ep in episodes if str(ep.record_name) == record_name)
            first_id = int(getattr(episode, "first_frame", 0))
            path = record_dir / "frames" / f"frame_{first_id:06d}.json"
            with path.open("r", encoding="utf-8") as file:
                first_frame = json.load(file)
            self.ego_references[record_name] = ego14_reference_from_frame(
                first_frame, ground_z=self.ground_z_by_record.get(record_name),
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _empty_detection(self) -> HumanDetections:
        return HumanDetections(
            np.zeros(0, np.int64), np.zeros((0, self.num_joints, 7), np.float32),
            np.zeros((0, self.num_joints), np.bool_), np.zeros(0, np.float32),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        record_name = str(item["record_name"])
        record_dir = self.record_dirs[record_name]
        frame_ids = np.asarray(item["frame_ids"], dtype=np.int64)
        ego_states: list[np.ndarray] = []
        goal_positions: list[np.ndarray] = []
        detections: list[HumanDetections] = []
        for frame_id in frame_ids:
            stem = f"frame_{int(frame_id):06d}"
            import json
            with (record_dir / "frames" / f"{stem}.json").open("r", encoding="utf-8") as file:
                frame = json.load(file)
                ego_states.append(ego14_from_frame(
                    frame, self.ego_references[record_name],
                    strict_altitude=self.strict_altitude,
                ))
                goal_positions.append(goal_position_from_frame(
                    frame, self.ego_references[record_name],
                ))
            pose_path = self.pose_3d_cache_root / record_name / f"{stem}_pose.npz"
            if pose_path.is_file():
                detection = detections_from_3d_pose_cache(pose_path)
                if detection.skeleton.shape[1] != self.num_joints:
                    raise ValueError(
                        f"{pose_path}: expected {self.num_joints} joints, got {detection.skeleton.shape[1]}"
                    )
                detections.append(detection)
            elif self.require_pose_cache:
                raise FileNotFoundError(f"Missing 3D pose cache: {pose_path}")
            else:
                detections.append(self._empty_detection())

        slots = assign_stable_human_slots(
            detections, max_people=self.max_people,
            joint_count=self.num_joints, feat_dim=7,
        )
        item["ego_state"] = np.stack(ego_states).astype(np.float32)
        item.pop("lidar", None)
        item.pop("lidar_mask", None)
        item.pop("dense_bev", None)
        goal_position = np.stack(goal_positions).astype(np.float32)
        item["goal_position"] = goal_position
        item["goal"] = goal_features_numpy(item["ego_state"], goal_position)
        human_root, human_joints = human_root_and_relative_joints_numpy(
            slots.skeleton, slots.human_mask, slots.joint_mask,
        )
        item.update({
            "skeleton": slots.skeleton,
            "human_root": human_root,
            "human_joints": human_joints,
            "human_mask": slots.human_mask,
            "joint_mask": slots.joint_mask,
            "human_ids": slots.human_ids,
            "human_is_first": slots.human_is_first,
            "truncated_people": slots.truncated_people,
        })
        validate_factorized_batch(item)
        return item
