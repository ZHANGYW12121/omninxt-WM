"""Read-only adapter for Isaac Sim UAV-through-crowd offline records.

The recorder stores each episode as::

    record_xxx/
      metadata.json
      summary.json
      frames/
        frame_000000.json
        frame_000000_camera.jpg
        frame_000000_lidar.npy
        ...

This module keeps those raw files untouched and exposes fixed-length Dreamer/RSSM
training sequences.  Expensive derived data, such as RTMPose keypoints, is read
from a separate cache directory if provided.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - keeps index utilities usable without torch
    torch = None
    Dataset = object  # type: ignore[misc,assignment]


FRAME_JSON_RE = re.compile(r"^frame_(\d{6})\.json$")
ACTION_KEYS = ("vx_body_mps", "vy_body_mps", "vz_world_mps", "yaw_rate_rps")
DEFAULT_EGO_KEYS = (
    "x_local",
    "y_local",
    "z_rel",
    "vx_local",
    "vy_local",
    "vz",
    "ax_local",
    "ay_local",
    "az",
    "roll",
    "pitch",
    "yaw_rel",
    "goal_dx_body",
    "goal_dy_body",
    "goal_dz",
    "goal_distance",
    "heading_error",
)
RAW_EGO_KEYS = ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "roll", "pitch", "yaw")
REWARD_COMPONENT_KEYS = ("event", "progress", "human_clearance", "smoothness", "height", "time")


@dataclass(frozen=True)
class EpisodeInfo:
    """Small, JSON-serializable description of one usable episode."""

    record_name: str
    record_dir: str
    num_frames: int
    first_frame: int
    last_frame: int
    duration_sec: float | None
    episode_return: float | None
    termination_reason: str | None
    reached_goal: bool
    collision: bool
    sample_rate_hz: float | None
    control_rate_hz: float | None

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "EpisodeInfo":
        return cls(
            record_name=str(item["record_name"]),
            record_dir=str(item["record_dir"]),
            num_frames=int(item["num_frames"]),
            first_frame=int(item.get("first_frame", 0)),
            last_frame=int(item.get("last_frame", int(item["num_frames"]) - 1)),
            duration_sec=_optional_float(item.get("duration_sec")),
            episode_return=_optional_float(item.get("episode_return")),
            termination_reason=_optional_str(item.get("termination_reason")),
            reached_goal=bool(item.get("reached_goal", False)),
            collision=bool(item.get("collision", False)),
            sample_rate_hz=_optional_float(item.get("sample_rate_hz")),
            control_rate_hz=_optional_float(item.get("control_rate_hz")),
        )


@dataclass(frozen=True)
class EgoStateReference:
    """Episode-local reference frame for the 17D UAV ego state.

    ``origin_xyz`` and ``origin_yaw`` are taken from the first frame of the
    episode.  Position, horizontal velocity and horizontal acceleration are
    expressed in this episode-local frame.  Goal vectors are expressed in the
    current UAV body/yaw frame.
    """

    origin_xyz: tuple[float, float, float]
    origin_yaw: float


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_frame_json_ids(frames_dir: Path) -> list[int]:
    if not frames_dir.is_dir():
        return []
    ids: list[int] = []
    for name in os.listdir(frames_dir):
        match = FRAME_JSON_RE.match(name)
        if match:
            ids.append(int(match.group(1)))
    ids.sort()
    return ids


def _complete_frame_ids(frames_dir: Path) -> list[int]:
    ids = _list_frame_json_ids(frames_dir)
    complete: list[int] = []
    for frame_id in ids:
        stem = f"frame_{frame_id:06d}"
        if (
            (frames_dir / f"{stem}.json").is_file()
            and (frames_dir / f"{stem}_camera.jpg").is_file()
            and (frames_dir / f"{stem}_lidar.npy").is_file()
        ):
            complete.append(frame_id)
    return complete


def _is_contiguous(ids: Sequence[int]) -> bool:
    if not ids:
        return False
    return list(ids) == list(range(ids[0], ids[-1] + 1))


def build_episode_index(
    data_root: str | Path,
    *,
    min_frames: int = 30,
    exclude_termination: Iterable[str] = ("stuck_timeout",),
    require_complete_triplets: bool = True,
    require_contiguous: bool = True,
    max_records: int | None = None,
    return_rejected: bool = False,
) -> list[EpisodeInfo] | tuple[list[EpisodeInfo], list[dict[str, Any]]]:
    """Scan a raw dataset root and return usable episodes.

    This function only reads the raw dataset.  It does not create, remove, or
    modify files under ``data_root``.
    """

    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    excluded = {str(x) for x in exclude_termination}
    episodes: list[EpisodeInfo] = []
    rejected: list[dict[str, Any]] = []
    records = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("record_"))
    if max_records is not None:
        records = records[: int(max_records)]

    for record_dir in records:
        reason = None
        metadata_path = record_dir / "metadata.json"
        summary_path = record_dir / "summary.json"
        frames_dir = record_dir / "frames"
        if not metadata_path.is_file():
            reason = "missing_metadata"
        elif not summary_path.is_file():
            reason = "missing_summary"
        elif not frames_dir.is_dir():
            reason = "missing_frames_dir"

        metadata: dict[str, Any] = {}
        summary: dict[str, Any] = {}
        frame_ids: list[int] = []
        if reason is None:
            try:
                metadata = _load_json(metadata_path)
                summary = _load_json(summary_path)
            except Exception as exc:  # noqa: BLE001 - report malformed record
                reason = f"json_error:{exc}"

        if reason is None:
            frame_ids = _complete_frame_ids(frames_dir) if require_complete_triplets else _list_frame_json_ids(frames_dir)
            if len(frame_ids) < int(min_frames):
                reason = "too_short"
            elif require_contiguous and not _is_contiguous(frame_ids):
                reason = "non_contiguous_frames"
            elif str(summary.get("termination_reason")) in excluded:
                reason = "excluded_termination"

        if reason is not None:
            rejected.append(
                {
                    "record_name": record_dir.name,
                    "record_dir": str(record_dir),
                    "reason": reason,
                    "num_frames": len(frame_ids),
                }
            )
            continue

        episodes.append(
            EpisodeInfo(
                record_name=record_dir.name,
                record_dir=str(record_dir),
                num_frames=len(frame_ids),
                first_frame=int(frame_ids[0]),
                last_frame=int(frame_ids[-1]),
                duration_sec=_optional_float(summary.get("duration_sec")),
                episode_return=_optional_float(summary.get("episode_return")),
                termination_reason=_optional_str(summary.get("termination_reason")),
                reached_goal=bool(summary.get("reached_goal", False)),
                collision=bool(summary.get("collision", False)),
                sample_rate_hz=_optional_float(metadata.get("sample_rate_hz")),
                control_rate_hz=_optional_float(metadata.get("control_rate_hz")),
            )
        )

    if return_rejected:
        return episodes, rejected
    return episodes


def save_episode_index(index: Sequence[EpisodeInfo], path: str | Path, *, rejected: Sequence[Mapping[str, Any]] = ()) -> None:
    """Save an index file outside or inside the repo, never into raw records unless asked."""

    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": [asdict(item) for item in index],
        "rejected": [dict(item) for item in rejected],
        "schema": "isaac_crowd_episode_index_v1",
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_episode_index(path: str | Path) -> list[EpisodeInfo]:
    payload = _load_json(Path(path).expanduser())
    items = payload["episodes"] if isinstance(payload, dict) and "episodes" in payload else payload
    return [EpisodeInfo.from_dict(item) for item in items]


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _as_float_array(value: Any, length: int, default: Sequence[float] | None = None) -> np.ndarray:
    if default is None:
        out = np.zeros((length,), dtype=np.float32)
    else:
        out = np.asarray(default, dtype=np.float32).reshape(-1)[:length].copy()
        if out.shape[0] < length:
            fixed = np.zeros((length,), dtype=np.float32)
            fixed[: out.shape[0]] = out
            out = fixed
    if value is None:
        return out
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return out
    if arr.shape[0] == 0:
        return out
    out[: min(length, arr.shape[0])] = arr[:length]
    out[~np.isfinite(out)] = 0.0
    return out


def _drone_state_mapping(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    state = frame.get("drone_state") or {}
    return state if isinstance(state, Mapping) else {}


def _state_vec3(state: Mapping[str, Any], vector_key: str, scalar_keys: Sequence[str]) -> np.ndarray:
    if vector_key in state:
        return _as_float_array(state.get(vector_key), 3)
    return np.asarray([float(state.get(key, 0.0) or 0.0) for key in scalar_keys], dtype=np.float32)


def _state_yaw(state: Mapping[str, Any]) -> float:
    if "yaw" in state:
        return float(state.get("yaw", 0.0) or 0.0)
    rpy = _as_float_array(state.get("roll_pitch_yaw_rad"), 3)
    return float(rpy[2])


def _state_roll_pitch(state: Mapping[str, Any]) -> tuple[float, float]:
    roll = state.get("roll")
    pitch = state.get("pitch")
    if roll is not None and pitch is not None:
        return float(roll or 0.0), float(pitch or 0.0)
    rpy = _as_float_array(state.get("roll_pitch_yaw_rad"), 3)
    return float(rpy[0]), float(rpy[1])


def _target_xyz(frame: Mapping[str, Any], fallback_xyz: np.ndarray) -> np.ndarray:
    for key in ("target_point", "goal_point", "goal", "target"):
        value = frame.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key in ("position", "point", "xyz"):
                if sub_key in value:
                    return _as_float_array(value.get(sub_key), 3, default=fallback_xyz)
            continue
        return _as_float_array(value, 3, default=fallback_xyz)
    return np.asarray(fallback_xyz, dtype=np.float32).reshape(3)


def _rotate_world_xy_to_frame(vec_xy: Sequence[float], frame_yaw: float) -> tuple[float, float]:
    """Rotate a world-frame XY vector into a frame with heading ``frame_yaw``."""

    x = float(vec_xy[0])
    y = float(vec_xy[1])
    c = math.cos(float(frame_yaw))
    s = math.sin(float(frame_yaw))
    return c * x + s * y, -s * x + c * y


def ego_state_reference_from_frame(frame: Mapping[str, Any]) -> EgoStateReference:
    """Create an episode-local ego-state reference from the first frame."""

    state = _drone_state_mapping(frame)
    pos = _state_vec3(state, "position", ("x", "y", "z"))
    yaw = _state_yaw(state)
    return EgoStateReference(
        origin_xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
        origin_yaw=float(yaw),
    )


def applied_action_to_normalized(frame: Mapping[str, Any], *, clip: bool = True) -> np.ndarray:
    """Return the action that actually acted on the drone, normalized to [-1, 1].

    ``action.normalized`` in the raw JSON describes the controller-requested
    command.  For dynamics learning, we use ``action.applied`` because it is the
    slew-limited velocity reference that was actually sent downstream.
    """

    action = frame.get("action") or {}
    if not isinstance(action, Mapping):
        return np.zeros((4,), dtype=np.float32)
    applied = action.get("applied") or {}
    limits = action.get("normalization_limits") or {}
    values: list[float] = []
    for key in ACTION_KEYS:
        raw = float(applied.get(key, 0.0) or 0.0)
        limit = abs(float(limits.get(key, 1.0) or 1.0))
        if limit <= 1.0e-8:
            norm = 0.0
        else:
            norm = raw / limit
        values.append(norm)
    out = np.asarray(values, dtype=np.float32)
    if clip:
        out = np.clip(out, -1.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def requested_action_to_normalized(frame: Mapping[str, Any], *, clip: bool = True) -> np.ndarray:
    """Return the original controller request normalized to [-1, 1]."""

    action = frame.get("action") or {}
    normalized = action.get("normalized") if isinstance(action, Mapping) else None
    if normalized is not None:
        out = np.asarray(normalized, dtype=np.float32)
    else:
        requested = action.get("requested") if isinstance(action, Mapping) else {}
        limits = action.get("normalization_limits") if isinstance(action, Mapping) else {}
        values: list[float] = []
        for key in ACTION_KEYS:
            raw = float((requested or {}).get(key, 0.0) or 0.0)
            limit = abs(float((limits or {}).get(key, 1.0) or 1.0))
            values.append(0.0 if limit <= 1.0e-8 else raw / limit)
        out = np.asarray(values, dtype=np.float32)
    if out.shape != (4,):
        fixed = np.zeros((4,), dtype=np.float32)
        fixed[: min(4, out.size)] = out.reshape(-1)[:4]
        out = fixed
    if clip:
        out = np.clip(out, -1.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def ego_state_to_vector(
    frame: Mapping[str, Any],
    reference: EgoStateReference | None = None,
    *,
    raw_keys: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the 17D episode-local UAV state used by the crowd world model.

    The default state is:

    ``x_local, y_local, z_rel, vx_local, vy_local, vz, ax_local, ay_local, az,
    roll, pitch, yaw_rel, goal_dx_body, goal_dy_body, goal_dz, goal_distance,
    heading_error``.

    ``reference`` should be created from the first frame of the episode.  If it
    is omitted, the current frame becomes the reference, which is mainly useful
    for quick single-frame utilities.
    """

    state = _drone_state_mapping(frame)
    if raw_keys is not None:
        values = [float(state.get(key, 0.0) or 0.0) for key in raw_keys]
        out = np.asarray(values, dtype=np.float32)
        out[~np.isfinite(out)] = 0.0
        return out

    pos = _state_vec3(state, "position", ("x", "y", "z"))
    vel = _state_vec3(state, "velocity", ("vx", "vy", "vz"))
    acc = _state_vec3(state, "acceleration", ("ax", "ay", "az"))
    roll, pitch = _state_roll_pitch(state)
    yaw = _state_yaw(state)
    if reference is None:
        reference = ego_state_reference_from_frame(frame)

    origin = np.asarray(reference.origin_xyz, dtype=np.float32)
    origin_yaw = float(reference.origin_yaw)
    dpos = pos - origin
    x_local, y_local = _rotate_world_xy_to_frame(dpos[:2], origin_yaw)
    vx_local, vy_local = _rotate_world_xy_to_frame(vel[:2], origin_yaw)
    ax_local, ay_local = _rotate_world_xy_to_frame(acc[:2], origin_yaw)
    z_rel = float(dpos[2])
    yaw_rel = _wrap_pi(float(yaw) - origin_yaw)

    target = _target_xyz(frame, fallback_xyz=pos)
    goal_vec = target - pos
    goal_dx_body, goal_dy_body = _rotate_world_xy_to_frame(goal_vec[:2], float(yaw))
    goal_dz = float(goal_vec[2])
    goal_distance = float(np.linalg.norm(goal_vec.astype(np.float64)))
    if float(np.linalg.norm(goal_vec[:2].astype(np.float64))) <= 1.0e-6:
        heading_error = 0.0
    else:
        heading_to_goal = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
        heading_error = _wrap_pi(heading_to_goal - float(yaw))

    out = np.asarray(
        [
            x_local,
            y_local,
            z_rel,
            vx_local,
            vy_local,
            float(vel[2]),
            ax_local,
            ay_local,
            float(acc[2]),
            float(roll),
            float(pitch),
            yaw_rel,
            goal_dx_body,
            goal_dy_body,
            goal_dz,
            goal_distance,
            heading_error,
        ],
        dtype=np.float32,
    )
    out[~np.isfinite(out)] = 0.0
    return out


def reward_components_to_vector(frame: Mapping[str, Any], keys: Sequence[str] = REWARD_COMPONENT_KEYS) -> np.ndarray:
    comps = frame.get("reward_components") or {}
    values = [float((comps if isinstance(comps, Mapping) else {}).get(key, 0.0) or 0.0) for key in keys]
    out = np.asarray(values, dtype=np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _is_terminal_frame(frame: Mapping[str, Any]) -> bool:
    reason = str(frame.get("termination_reason", "recording"))
    return bool(frame.get("collision", False) or frame.get("reached_goal", False) or reason != "recording")


def _load_image(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        width, height = int(image_size[0]), int(image_size[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _load_lidar(path: Path, max_points: int | None, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    points = np.load(path).astype(np.float32, copy=False)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected lidar points [N, >=3], got {points.shape} at {path}")
    points = points[:, :3]
    if max_points is None:
        return points
    max_points = int(max_points)
    mask = np.zeros((max_points,), dtype=np.bool_)
    out = np.zeros((max_points, 3), dtype=np.float32)
    if len(points) == 0:
        return out, mask
    if len(points) > max_points:
        if rng is None:
            choice = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        else:
            choice = rng.choice(len(points), size=max_points, replace=False)
        points = points[choice]
    out[: len(points)] = points
    mask[: len(points)] = True
    return out, mask


def _pose_cache_path(pose_cache_root: Path, record_name: str, frame_id: int) -> Path:
    return pose_cache_root / record_name / f"frame_{frame_id:06d}_pose.npz"


def _pose_scores(pose: Mapping[str, np.ndarray]) -> np.ndarray:
    if "person_scores" in pose:
        scores = np.asarray(pose["person_scores"], dtype=np.float32)
    else:
        keypoints = np.asarray(pose.get("keypoints_xyc", np.zeros((0, 17, 3), dtype=np.float32)))
        scores = keypoints[:, :, 2].mean(axis=1).astype(np.float32) if keypoints.size else np.zeros((0,), dtype=np.float32)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _pose_valid_mask(pose: Mapping[str, np.ndarray], min_score: float) -> np.ndarray:
    keypoints = np.asarray(pose.get("keypoints_xyc", np.zeros((0, 17, 3), dtype=np.float32)))
    num_people = int(keypoints.shape[0]) if keypoints.ndim == 3 else 0
    if num_people == 0:
        return np.zeros((0,), dtype=np.bool_)
    if "valid_mask" in pose:
        valid = np.asarray(pose["valid_mask"], dtype=np.bool_).reshape(-1)[:num_people].copy()
        if valid.shape[0] < num_people:
            fixed = np.zeros((num_people,), dtype=np.bool_)
            fixed[: valid.shape[0]] = valid
            valid = fixed
    else:
        valid = keypoints[:, :, 2].max(axis=1) > 0.0
    scores = _pose_scores(pose)
    if scores.shape[0] < num_people:
        fixed_scores = np.zeros((num_people,), dtype=np.float32)
        fixed_scores[: scores.shape[0]] = scores
        scores = fixed_scores
    valid &= scores[:num_people] >= float(min_score)
    return valid


def _pose_track_ids(pose: Mapping[str, np.ndarray], num_people: int) -> np.ndarray:
    if "track_ids" not in pose:
        return -np.ones((num_people,), dtype=np.int32)
    track_ids = np.asarray(pose["track_ids"], dtype=np.int32).reshape(-1)
    if track_ids.shape[0] >= num_people:
        return track_ids[:num_people]
    fixed = -np.ones((num_people,), dtype=np.int32)
    fixed[: track_ids.shape[0]] = track_ids
    return fixed


def _pose_bboxes(pose: Mapping[str, np.ndarray], num_people: int) -> np.ndarray:
    if "bboxes_xyxy" not in pose:
        return np.zeros((num_people, 4), dtype=np.float32)
    bboxes = np.asarray(pose["bboxes_xyxy"], dtype=np.float32)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        return np.zeros((num_people, 4), dtype=np.float32)
    if bboxes.shape[0] >= num_people:
        return bboxes[:num_people]
    fixed = np.zeros((num_people, 4), dtype=np.float32)
    fixed[: bboxes.shape[0]] = bboxes
    return fixed


def _pose_image_size_hw(pose: Mapping[str, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if "image_size_hw" not in pose:
        return fallback
    size = np.asarray(pose["image_size_hw"], dtype=np.float32).reshape(-1)
    if size.shape[0] < 2 or not np.isfinite(size[:2]).all() or (size[:2] <= 0).any():
        return fallback
    return size[:2].astype(np.float32)


def _select_pose_detections(
    pose: Mapping[str, np.ndarray],
    *,
    max_people: int,
    min_score: float,
) -> np.ndarray:
    keypoints = np.asarray(pose.get("keypoints_xyc", np.zeros((0, 17, 3), dtype=np.float32)))
    if keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    num_people = keypoints.shape[0]
    valid = _pose_valid_mask(pose, min_score=min_score)
    scores = _pose_scores(pose)
    if scores.shape[0] < num_people:
        fixed_scores = np.zeros((num_people,), dtype=np.float32)
        fixed_scores[: scores.shape[0]] = scores
        scores = fixed_scores
    candidate = np.flatnonzero(valid[:num_people])
    if candidate.size == 0:
        return candidate.astype(np.int64)
    # Highest confidence people become the first tokens.  This keeps the token
    # order deterministic and independent of filesystem order.
    order = np.argsort(-scores[candidate], kind="stable")
    return candidate[order][: int(max_people)].astype(np.int64)


def _find_track_detection(
    pose: Mapping[str, np.ndarray],
    track_id: int,
    *,
    min_score: float,
) -> int | None:
    keypoints = np.asarray(pose.get("keypoints_xyc", np.zeros((0, 17, 3), dtype=np.float32)))
    if keypoints.ndim != 3 or keypoints.shape[0] == 0 or track_id <= 0:
        return None
    num_people = keypoints.shape[0]
    track_ids = _pose_track_ids(pose, num_people)
    valid = _pose_valid_mask(pose, min_score=min_score)
    matches = np.flatnonzero((track_ids == int(track_id)) & valid[:num_people])
    if matches.size == 0:
        return None
    return int(matches[0])


def _build_causal_pose_windows(
    pose_cache_root: Path,
    episode: EpisodeInfo,
    frame_ids: np.ndarray,
    *,
    window_size: int,
    max_people: int,
    min_score: float,
) -> dict[str, np.ndarray]:
    """Build causal per-person skeleton windows from pose cache files.

    For each current frame ``t`` and selected current person, this matches the
    same ``track_id`` in previous frames and creates ``[t-W+1, ..., t]``.  Missing
    history is zero-padded and marked invalid in ``pose_window_mask``.
    """

    window_size = int(window_size)
    max_people = int(max_people)
    if window_size <= 0:
        raise ValueError("pose_window_size must be positive.")
    if max_people <= 0:
        raise ValueError("max_pose_people must be positive.")

    t_len = int(frame_ids.shape[0])
    pose_windows = np.zeros((t_len, max_people, window_size, 17, 3), dtype=np.float32)
    pose_window_mask = np.zeros((t_len, max_people, window_size), dtype=np.bool_)
    pose_token_mask = np.zeros((t_len, max_people), dtype=np.bool_)
    pose_track_ids = -np.ones((t_len, max_people), dtype=np.int32)
    pose_person_scores = np.zeros((t_len, max_people), dtype=np.float32)
    pose_bboxes_xyxy = np.zeros((t_len, max_people, 4), dtype=np.float32)
    pose_image_size_hw = np.ones((t_len, 2), dtype=np.float32)

    if t_len == 0:
        return {
            "pose_windows": pose_windows,
            "pose_window_mask": pose_window_mask,
            "pose_token_mask": pose_token_mask,
            "pose_track_ids": pose_track_ids,
            "pose_person_scores": pose_person_scores,
            "pose_bboxes_xyxy": pose_bboxes_xyxy,
            "pose_image_size_hw": pose_image_size_hw,
        }

    first_needed = max(int(episode.first_frame), int(frame_ids[0]) - window_size + 1)
    last_needed = int(frame_ids[-1])
    poses_by_frame: dict[int, dict[str, np.ndarray]] = {
        frame_id: _load_pose_npz(_pose_cache_path(pose_cache_root, episode.record_name, frame_id))
        for frame_id in range(first_needed, last_needed + 1)
    }

    fallback_hw = np.ones((2,), dtype=np.float32)
    for ti, frame_id_np in enumerate(frame_ids):
        frame_id = int(frame_id_np)
        current_pose = poses_by_frame.get(frame_id)
        if current_pose is None:
            current_pose = _load_pose_npz(_pose_cache_path(pose_cache_root, episode.record_name, frame_id))
            poses_by_frame[frame_id] = current_pose

        fallback_hw = _pose_image_size_hw(current_pose, fallback_hw)
        pose_image_size_hw[ti] = fallback_hw

        current_keypoints = np.asarray(current_pose.get("keypoints_xyc", np.zeros((0, 17, 3), dtype=np.float32)))
        if current_keypoints.ndim != 3 or current_keypoints.shape[0] == 0:
            continue
        num_current = current_keypoints.shape[0]
        selected = _select_pose_detections(current_pose, max_people=max_people, min_score=min_score)
        if selected.size == 0:
            continue

        current_track_ids = _pose_track_ids(current_pose, num_current)
        current_scores = _pose_scores(current_pose)
        current_bboxes = _pose_bboxes(current_pose, num_current)

        for slot, det_idx_np in enumerate(selected):
            det_idx = int(det_idx_np)
            if det_idx >= num_current:
                continue
            track_id = int(current_track_ids[det_idx])
            pose_token_mask[ti, slot] = True
            pose_track_ids[ti, slot] = track_id
            if det_idx < current_scores.shape[0]:
                pose_person_scores[ti, slot] = float(current_scores[det_idx])
            pose_bboxes_xyxy[ti, slot] = current_bboxes[det_idx]

            for wi, hist_frame in enumerate(range(frame_id - window_size + 1, frame_id + 1)):
                if hist_frame < int(episode.first_frame):
                    continue
                hist_pose = poses_by_frame.get(hist_frame)
                if hist_pose is None:
                    hist_pose = _load_pose_npz(_pose_cache_path(pose_cache_root, episode.record_name, hist_frame))
                    poses_by_frame[hist_frame] = hist_pose

                hist_det_idx: int | None
                if hist_frame == frame_id:
                    hist_det_idx = det_idx
                else:
                    hist_det_idx = _find_track_detection(hist_pose, track_id, min_score=min_score)
                if hist_det_idx is None:
                    continue

                hist_keypoints = np.asarray(
                    hist_pose.get("keypoints_xyc", np.zeros((0, 17, 3), dtype=np.float32)),
                    dtype=np.float32,
                )
                if hist_keypoints.ndim != 3 or hist_det_idx >= hist_keypoints.shape[0]:
                    continue
                skeleton = hist_keypoints[hist_det_idx]
                if skeleton.shape != (17, 3):
                    continue
                pose_windows[ti, slot, wi] = skeleton
                pose_window_mask[ti, slot, wi] = True

    pose_windows[~np.isfinite(pose_windows)] = 0.0
    pose_person_scores[~np.isfinite(pose_person_scores)] = 0.0
    pose_bboxes_xyxy[~np.isfinite(pose_bboxes_xyxy)] = 0.0
    return {
        "pose_windows": pose_windows,
        "pose_window_mask": pose_window_mask,
        "pose_token_mask": pose_token_mask,
        "pose_track_ids": pose_track_ids,
        "pose_person_scores": pose_person_scores,
        "pose_bboxes_xyxy": pose_bboxes_xyxy,
        "pose_image_size_hw": pose_image_size_hw,
    }


class IsaacCrowdSequenceDataset(Dataset):  # type: ignore[misc]
    """Fixed-length sequence dataset for Dreamer/RSSM pretraining.

    By default this returns numeric low-dimensional tensors and file paths for
    images/lidar/pose cache.  Set ``load_images`` or ``load_lidar`` only when the
    downstream model/collate can afford loading those large arrays immediately.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        *,
        index_path: str | Path | None = None,
        episodes: Sequence[EpisodeInfo] | None = None,
        sequence_length: int = 64,
        stride: int = 1,
        min_frames: int = 30,
        exclude_termination: Iterable[str] = ("stuck_timeout",),
        chunk_resets: bool = True,
        load_images: bool = False,
        image_size: tuple[int, int] | None = None,
        load_lidar: bool = False,
        max_lidar_points: int | None = None,
        pose_cache_root: str | Path | None = None,
        load_pose: bool = False,
        load_pose_windows: bool = False,
        pose_window_size: int = 8,
        max_pose_people: int = 16,
        pose_min_score: float = 0.05,
        include_reward_components: bool = True,
        missing_reward_policy: str = "error",
        reward_cache_root: str | Path | None = None,
        include_paths: bool = True,
        action_source: str = "applied",
        clip_actions: bool = True,
        deterministic_lidar_subsample: bool = True,
    ) -> None:
        if episodes is not None:
            self.episodes = list(episodes)
        elif index_path is not None:
            self.episodes = load_episode_index(index_path)
        elif data_root is not None:
            self.episodes = build_episode_index(
                data_root,
                min_frames=min_frames,
                exclude_termination=exclude_termination,
            )
        else:
            raise ValueError("Provide one of data_root, index_path, or episodes.")

        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        self.stride = max(1, int(stride))
        self.chunk_resets = bool(chunk_resets)
        self.load_images = bool(load_images)
        self.image_size = image_size
        self.load_lidar = bool(load_lidar)
        self.max_lidar_points = None if max_lidar_points is None else int(max_lidar_points)
        self.pose_cache_root = None if pose_cache_root is None else Path(pose_cache_root).expanduser()
        self.load_pose = bool(load_pose)
        self.load_pose_windows = bool(load_pose_windows)
        self.pose_window_size = int(pose_window_size)
        self.max_pose_people = int(max_pose_people)
        self.pose_min_score = float(pose_min_score)
        if self.load_pose_windows and self.pose_cache_root is None:
            raise ValueError("load_pose_windows=True requires pose_cache_root.")
        self.include_reward_components = bool(include_reward_components)
        if missing_reward_policy not in {"error", "zero"}:
            raise ValueError("missing_reward_policy must be 'error' or explicit legacy 'zero'")
        self.missing_reward_policy = str(missing_reward_policy)
        self.reward_cache_root = None if reward_cache_root is None else Path(reward_cache_root).expanduser()
        self.include_paths = bool(include_paths)
        if action_source not in {"applied", "requested"}:
            raise ValueError("action_source must be 'applied' or 'requested'.")
        self.action_source = action_source
        self.clip_actions = bool(clip_actions)
        self.deterministic_lidar_subsample = bool(deterministic_lidar_subsample)

        self._chunks: list[tuple[int, int]] = []
        for ep_idx, episode in enumerate(self.episodes):
            if episode.num_frames < self.sequence_length:
                continue
            max_start = episode.num_frames - self.sequence_length
            for start in range(0, max_start + 1, self.stride):
                self._chunks.append((ep_idx, start))
        if not self._chunks:
            raise ValueError(
                f"No chunks available: {len(self.episodes)} episodes, "
                f"sequence_length={self.sequence_length}, stride={self.stride}."
            )

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep_idx, local_start = self._chunks[int(index)]
        episode = self.episodes[ep_idx]
        record_dir = Path(episode.record_dir)
        frames_dir = record_dir / "frames"
        frame_ids = np.arange(
            episode.first_frame + local_start,
            episode.first_frame + local_start + self.sequence_length,
            dtype=np.int32,
        )
        reference_frame = _load_json(frames_dir / f"frame_{int(episode.first_frame):06d}.json")
        ego_reference = ego_state_reference_from_frame(reference_frame)

        timestamps: list[float] = []
        actions: list[np.ndarray] = []
        ego_states: list[np.ndarray] = []
        rewards: list[float] = []
        is_first: list[bool] = []
        is_last: list[bool] = []
        is_terminal: list[bool] = []
        reward_components: list[np.ndarray] = []
        image_paths: list[str] = []
        lidar_paths: list[str] = []
        pose_paths: list[str] = []
        images: list[np.ndarray] = []
        lidars: list[np.ndarray] = []
        lidar_masks: list[np.ndarray] = []
        poses: list[dict[str, np.ndarray]] = []

        rng = None
        if self.max_lidar_points is not None and not self.deterministic_lidar_subsample:
            rng = np.random.default_rng()

        for offset, frame_id in enumerate(frame_ids):
            stem = f"frame_{int(frame_id):06d}"
            frame_json = _load_json(frames_dir / f"{stem}.json")
            reward_record = frame_json
            if "reward" not in reward_record and self.reward_cache_root is not None:
                reward_path = self.reward_cache_root / episode.record_name / f"{stem}_reward.json"
                if reward_path.is_file():
                    reward_record = {**frame_json, **_load_json(reward_path)}
            timestamps.append(float(frame_json.get("timestamp", 0.0) or 0.0))
            ego_states.append(ego_state_to_vector(frame_json, reference=ego_reference))
            if self.action_source == "applied":
                actions.append(applied_action_to_normalized(frame_json, clip=self.clip_actions))
            else:
                actions.append(requested_action_to_normalized(frame_json, clip=self.clip_actions))
            if "reward" not in reward_record:
                if self.missing_reward_policy == "error":
                    raise KeyError(
                        f"Missing reward in {frames_dir / f'{stem}.json'}; "
                        "generate a reward sidecar or explicitly set missing_reward_policy='zero'"
                    )
                rewards.append(0.0)
            else:
                rewards.append(float(reward_record["reward"] or 0.0))
            terminal = _is_terminal_frame(frame_json)
            is_terminal.append(terminal)
            is_last.append(bool(terminal or int(frame_id) == episode.last_frame))
            if self.chunk_resets:
                is_first.append(offset == 0)
            else:
                is_first.append(int(frame_id) == episode.first_frame)
            if self.include_reward_components:
                reward_components.append(reward_components_to_vector(reward_record))

            img_path = frames_dir / f"{stem}_camera.jpg"
            lidar_path = frames_dir / f"{stem}_lidar.npy"
            if self.include_paths:
                image_paths.append(str(img_path))
                lidar_paths.append(str(lidar_path))
            if self.load_images:
                images.append(_load_image(img_path, self.image_size))
            if self.load_lidar:
                loaded = _load_lidar(lidar_path, self.max_lidar_points, rng)
                if isinstance(loaded, tuple):
                    points, mask = loaded
                    lidars.append(points)
                    lidar_masks.append(mask)
                else:
                    lidars.append(loaded)

            if self.pose_cache_root is not None:
                pose_path = _pose_cache_path(self.pose_cache_root, episode.record_name, int(frame_id))
                if self.include_paths:
                    pose_paths.append(str(pose_path))
                if self.load_pose:
                    poses.append(_load_pose_npz(pose_path))

        item: dict[str, Any] = {
            "record_name": episode.record_name,
            "frame_ids": frame_ids,
            "timestamp": np.asarray(timestamps, dtype=np.float32),
            "ego_state": np.stack(ego_states, axis=0).astype(np.float32),
            "action": np.stack(actions, axis=0).astype(np.float32),
            "reward": np.asarray(rewards, dtype=np.float32)[:, None],
            "is_first": np.asarray(is_first, dtype=np.bool_)[:, None],
            "is_last": np.asarray(is_last, dtype=np.bool_)[:, None],
            "is_terminal": np.asarray(is_terminal, dtype=np.bool_)[:, None],
        }
        if self.include_reward_components:
            item["reward_components"] = np.stack(reward_components, axis=0).astype(np.float32)
            item["reward_component_keys"] = REWARD_COMPONENT_KEYS
        if self.include_paths:
            item["image_path"] = image_paths
            item["lidar_path"] = lidar_paths
            if self.pose_cache_root is not None:
                item["pose_path"] = pose_paths
        if self.load_images:
            item["image"] = np.stack(images, axis=0).astype(np.float32)
        if self.load_lidar:
            if self.max_lidar_points is None:
                item["lidar"] = lidars
            else:
                item["lidar"] = np.stack(lidars, axis=0).astype(np.float32)
                item["lidar_mask"] = np.stack(lidar_masks, axis=0)
        if self.load_pose:
            item["pose"] = poses
        if self.load_pose_windows:
            assert self.pose_cache_root is not None
            item.update(
                _build_causal_pose_windows(
                    self.pose_cache_root,
                    episode,
                    frame_ids,
                    window_size=self.pose_window_size,
                    max_people=self.max_pose_people,
                    min_score=self.pose_min_score,
                )
            )
        return item


def _load_pose_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {
            "keypoints_xyc": np.zeros((0, 17, 3), dtype=np.float32),
            "track_ids": np.zeros((0,), dtype=np.int32),
            "valid_mask": np.zeros((0,), dtype=np.bool_),
            "bboxes_xyxy": np.zeros((0, 4), dtype=np.float32),
        }
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _to_tensor(value: Any) -> Any:
    if torch is None:
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return torch.from_numpy(value.astype(np.bool_, copy=False))
        if np.issubdtype(value.dtype, np.integer):
            return torch.from_numpy(value.astype(np.int64, copy=False))
        return torch.from_numpy(value.astype(np.float32, copy=False))
    return value


def crowd_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate numeric sequence fields into tensors and keep paths as lists."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if isinstance(first, np.ndarray):
            out[key] = _to_tensor(np.stack(values, axis=0))
        elif isinstance(first, (float, int, bool, np.number)):
            out[key] = _to_tensor(np.asarray(values))
        else:
            out[key] = values
    return out
