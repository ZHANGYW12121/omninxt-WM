"""Direct reader for compact Isaac skeleton/state dataset v3.

The recorder keeps all scene people and COCO12_BODY positions (COCO17 source
indices 5..16). This loader
derives relative joint velocity, risk-selects model slots, and emits the exact
factorized world-model batch without a separate pose-cache conversion step.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from modules.goal_conditioning import goal_features_numpy
from modules.skeleton_topology import COCO12_BODY_JOINT_COUNT
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


@dataclass(frozen=True)
class CompactEpisode:
    name: str
    directory: Path
    chunks: tuple[Path, ...]
    length: int
    metadata: Mapping[str, Any]
    summary: Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def discover_compact_episodes(root: str | Path) -> list[CompactEpisode]:
    root = Path(root).expanduser()
    episodes_root = root / "episodes" if (root / "episodes").is_dir() else root
    episodes: list[CompactEpisode] = []
    for directory in sorted(path for path in episodes_root.glob("*") if path.is_dir()):
        metadata_path = directory / "metadata.json"
        summary_path = directory / "summary.json"
        chunks = tuple(sorted((directory / "chunks").glob("chunk_*.npz")))
        # Abruptly interrupted episodes can contain valid atomic chunks but no
        # summary. They are retained on disk for audit and the seed is retried,
        # but they must not silently enter offline training.
        if not metadata_path.is_file() or not summary_path.is_file() or not chunks:
            continue
        metadata = _read_json(metadata_path)
        summary = _read_json(summary_path)
        if metadata.get("schema") != SCHEMA:
            continue
        if metadata.get("joint_topology") != "COCO12_BODY":
            raise ValueError(f"{metadata_path}: expected COCO12_BODY")
        if int(metadata.get("recorded_joint_count", -1)) != JOINT_COUNT:
            raise ValueError(f"{metadata_path}: expected {JOINT_COUNT} recorded joints")
        length = 0
        previous_frame: int | None = None
        expected_people = int(metadata["stored_people_count"])
        for chunk in chunks:
            with np.load(chunk, allow_pickle=False) as arrays:
                frame = np.asarray(arrays["frame_index"], np.int64)
                if frame.ndim != 1 or frame.size == 0:
                    raise ValueError(f"{chunk}: invalid frame_index")
                if previous_frame is not None and int(frame[0]) != previous_frame + 1:
                    raise ValueError(f"{chunk}: episode frame_index is not contiguous")
                if np.asarray(arrays["human_xyz"]).shape[1] != expected_people:
                    raise ValueError(f"{chunk}: people dimension differs from metadata")
                previous_frame = int(frame[-1])
                length += int(frame.size)
        episodes.append(CompactEpisode(
            directory.name, directory, chunks, length, metadata, summary))
    return episodes


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


def derive_relative_joint_velocity(
    xyz: np.ndarray,
    joint_valid: np.ndarray,
    track_ids: np.ndarray,
    ego_state: np.ndarray,
    simulation_time_s: np.ndarray,
) -> np.ndarray:
    """Finite-difference human motion expressed in current base_link axes.

    The previous point is first lifted into the episode frame and then
    transformed into the current body frame. A stationary world point has zero
    human-motion velocity even when the UAV translates or yaws; relative
    position remains represented by the recorded XYZ channels.
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


def _risk_scores(skeleton: np.ndarray, joint_mask: np.ndarray) -> np.ndarray:
    count = skeleton.shape[0]
    risk = np.zeros(count, np.float32)
    for index in range(count):
        valid = joint_mask[index]
        if not np.any(valid):
            continue
        position = skeleton[index, valid, :3].mean(axis=0)
        velocity = skeleton[index, valid, 3:6].mean(axis=0)
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
                 cache_episodes: int = 2) -> None:
        self.root = Path(root).expanduser()
        self.sequence_length = int(sequence_length)
        self.stride = int(stride or sequence_length)
        self.max_people = int(max_people)
        self.cache_episodes = max(1, int(cache_episodes))
        if self.sequence_length <= 0 or self.stride <= 0 or self.max_people <= 0:
            raise ValueError("sequence_length, stride and max_people must be positive")
        self.episodes = discover_compact_episodes(self.root)
        if not self.episodes:
            raise ValueError(f"No {SCHEMA} episodes found under {self.root}")
        self.windows: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            if episode.length < self.sequence_length:
                continue
            starts = list(range(
                0, episode.length - self.sequence_length + 1, self.stride))
            final_start = episode.length - self.sequence_length
            if starts[-1] != final_start:
                starts.append(final_start)
            self.windows.extend((episode_index, start) for start in starts)
        if not self.windows:
            raise ValueError("No episode is long enough for the requested sequence_length")
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

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
        self._cache[episode_index] = arrays
        self._cache.move_to_end(episode_index)
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        arrays = self._episode_arrays(episode_index)
        stop = start + self.sequence_length
        raw = {key: value[start:stop] for key, value in arrays.items()}
        xyz = np.asarray(raw["human_xyz"], np.float32)
        confidence = np.asarray(raw["human_confidence"], np.float32)
        joint_valid = np.asarray(raw["human_joint_valid"], np.bool_)
        track_ids = np.asarray(raw["human_track_id"], np.int64)
        # One-frame pre-roll preserves the first velocity of a window sampled
        # from the middle of an episode while recurrent state still resets.
        history_start = max(0, start - 1)
        history_offset = start - history_start
        velocity = derive_relative_joint_velocity(
            arrays["human_xyz"][history_start:stop],
            arrays["human_joint_valid"][history_start:stop],
            arrays["human_track_id"][history_start:stop],
            arrays["ego_state"][history_start:stop],
            arrays["simulation_time_s"][history_start:stop],
        )[history_offset:]
        detections: list[HumanDetections] = []
        for t in range(self.sequence_length):
            active = np.any(joint_valid[t], axis=1) & (track_ids[t] >= 0)
            ids = track_ids[t, active] + 1  # reserve -1 padding; detector IDs may start at zero
            skeleton = np.concatenate((
                xyz[t, active], velocity[t, active], confidence[t, active, :, None]),
                axis=-1).astype(np.float32)
            mask = joint_valid[t, active]
            detections.append(HumanDetections(
                ids.astype(np.int64), skeleton, mask, _risk_scores(skeleton, mask)))
        slots = assign_stable_human_slots(
            detections, max_people=self.max_people,
            joint_count=JOINT_COUNT, feat_dim=SKELETON_FEATURE_DIM)
        human_root, human_joints = human_root_and_relative_joints_numpy(
            slots.skeleton, slots.human_mask, slots.joint_mask)
        ego_state = np.asarray(raw["ego_state"], np.float32)
        goal_fixed = np.asarray(
            episode.metadata["goal_position_episode_local"], np.float32).reshape(3)
        goal_position = np.broadcast_to(
            goal_fixed, (self.sequence_length, 3)).copy()
        is_first = np.zeros((self.sequence_length, 1), np.bool_)
        is_first[0, 0] = True
        is_last = np.asarray(raw["is_last"], np.bool_).reshape(-1, 1)
        is_terminal = np.asarray(raw["is_terminal"], np.bool_).reshape(-1, 1)
        item: dict[str, Any] = {
            "record_name": episode.name,
            "frame_ids": np.asarray(raw["frame_index"], np.int64),
            "ego_state": ego_state,
            "goal_position": goal_position,
            "goal": goal_features_numpy(ego_state, goal_position),
            "skeleton": slots.skeleton,
            "human_root": human_root,
            "human_joints": human_joints,
            "human_mask": slots.human_mask,
            "joint_mask": slots.joint_mask,
            "human_ids": slots.human_ids,
            "human_is_first": slots.human_is_first,
            "truncated_people": slots.truncated_people,
            "action": np.asarray(raw["action"], np.float32),
            "action_valid": np.asarray(raw["action_valid"], np.bool_).reshape(-1, 1),
            "reward": np.asarray(raw["reward"], np.float32).reshape(-1, 1),
            "reward_components": np.asarray(raw["reward_components"], np.float32),
            "is_first": is_first,
            # Outcome labels are training/control targets and sampling metadata;
            # they are intentionally not consumed by the observation encoder.
            "success": np.asarray(raw["success"], np.bool_).reshape(-1, 1),
            "termination_code": np.asarray(
                raw["termination_code"], np.uint8).reshape(-1, 1),
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
        for key in (
            "priv_human_id", "priv_human_mask", "priv_human_position_world",
            "priv_human_velocity_world", "priv_min_human_clearance_m",
            "priv_min_human_clearance_valid", "priv_min_human_ttc_s",
            "priv_min_human_ttc_valid", "priv_collision", "priv_goal_distance_m",
        ):
            if key in raw:
                item[key] = np.asarray(raw[key])
        validate_factorized_batch(item)
        return item
