"""Chunked skeleton/state-only dataset recorder for crowd navigation."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import carb
import numpy as np

from data_recorder import DatasetRecorder
from dataset_v3_schema import (
    ACTION_KEYS,
    BODY_JOINT_INDICES,
    BODY_JOINT_NAMES,
    EGO_STATE_KEYS,
    REWARD_COMPONENT_KEYS,
    SCHEMA,
    STATE_SOURCE_CODES,
    TERMINATION_CODES,
    outcome_for_reason,
    build_ego_state,
    ego_reference,
    empty_skeleton_packet,
    goal_position_episode_local,
    normalized_applied_action,
    privileged_to_arrays,
    skeleton_source_counts,
    skeleton_to_arrays,
    stack_samples,
    transition_flags,
    write_npz_atomic,
)


class SkeletonStateDatasetRecorder(DatasetRecorder):
    """Record delayed perception packets against timestamped simulator state.

    The recorder samples vehicle/action/privileged state on the simulation
    clock, buffers it, and joins incoming skeleton packets by their original
    Isaac timestamp. Disk latency and perception latency therefore do not
    change the controller cadence or silently misalign observations.
    """

    def __init__(self, *args, skeleton_receiver=None, privileged_provider=None,
                 episode_metadata_provider=None, people_count_provider=None,
                 storage_max_people=60,
                 privileged_max_people=60, chunk_frames=256,
                 sync_tolerance_sec=0.075, skeleton_wait_wall_sec=0.75,
                 **kwargs):
        super().__init__(*args, camera_sensor=None, **kwargs)
        self.skeleton_receiver = skeleton_receiver
        self.privileged_provider = privileged_provider
        self.episode_metadata_provider = episode_metadata_provider
        self.people_count_provider = people_count_provider
        self.storage_max_people = int(storage_max_people)
        self.privileged_max_people = int(privileged_max_people)
        self.chunk_frames = int(chunk_frames)
        self.sync_tolerance_sec = float(sync_tolerance_sec)
        self.skeleton_wait_wall_sec = float(skeleton_wait_wall_sec)
        self.chunks_dir = None
        self.events_path = None
        self._state_buffer = deque(maxlen=512)
        self._last_packet_arrival_index = -1
        self._episode_people_capacity = max(1, self.storage_max_people)
        self._ego_reference = None
        self._last_sample_written = False
        self._chunk_count = 0
        self._unmatched_skeleton_count = 0
        self._missing_skeleton_count = 0
        self._capture_count = 0
        self._skeleton_source_counts = {}

    def start(self):
        if self.is_recording:
            return
        now = self._now()
        if now is None:
            carb.log_warn("[REC][V3] Cannot start without simulation time.")
            return
        root = self.dataset_root
        episodes_root = root / "episodes"
        episodes_root.mkdir(parents=True, exist_ok=True)
        episode_name = datetime.now().strftime("episode_%Y%m%d_%H%M%S")
        self.record_dir = episodes_root / episode_name
        suffix = 1
        while self.record_dir.exists():
            self.record_dir = episodes_root / "{}_{}".format(episode_name, suffix)
            suffix += 1
        self.chunks_dir = self.record_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.chunks_dir  # compatibility with base cleanup
        self.events_path = self.record_dir / "events.jsonl"

        self.start_time = float(now)
        self.wall_start_time = time.perf_counter()
        self.next_sample_time = 0.0
        self.frame_index = 0
        self.written_frame_count = 0
        self.dropped_frame_count = 0
        self.collision = False
        self.reached_goal = False
        self.termination_reason = "recording"
        self.event_details = {}
        self.episode_return = 0.0
        self.reward_frame_count = 0
        self.reward_component_sums = self._empty_reward_component_sums()
        self._reward_prev_context = None
        self.reward_calculator.reset()
        self._state_buffer.clear()
        self._last_packet_arrival_index = -1
        actual_people = self._safe_provider(self.people_count_provider)
        try:
            actual_people = max(1, int(actual_people))
        except (TypeError, ValueError):
            actual_people = max(1, self.storage_max_people)
        self._episode_people_capacity = min(
            actual_people, max(1, self.storage_max_people),
            max(1, self.privileged_max_people))
        initial_drone_state = self._drone_state()
        self._ego_reference = ego_reference(initial_drone_state)
        self._last_sample_written = False
        self._chunk_count = 0
        self._unmatched_skeleton_count = 0
        self._missing_skeleton_count = 0
        self._capture_count = 0
        self._skeleton_source_counts = {}
        self.write_queue = queue.Queue(maxsize=self.max_queue_size)
        self.writer_thread = threading.Thread(
            target=self._writer_loop, name="SkeletonDatasetV3Writer", daemon=True)
        self.writer_thread.start()
        self.is_recording = True

        manifest = {
            "schema": SCHEMA,
            "storage": "chunked_npz",
            "frames_per_chunk": self.chunk_frames,
            "sample_rate_hz": self.sample_rate_hz,
            "observation": ["human_skeleton_3d_base_link", "ego_state"],
            "excluded": ["rgb", "depth", "point_cloud", "skeleton_2d"],
            "source_joint_topology": "COCO17",
            "recorded_joint_topology": "COCO12_BODY",
            "recorded_joint_count": len(BODY_JOINT_INDICES),
            "termination_codes": dict(TERMINATION_CODES),
        }
        manifest_path = root / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                existing_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing_manifest = {}
            if existing_manifest.get("schema") != SCHEMA:
                manifest_path = root / "dataset_manifest_v3.json"
        if not manifest_path.exists():
            self._write_json(manifest_path, manifest)

        metadata_extra = self._safe_provider(self.episode_metadata_provider) or {}
        initial_action = self._read_action() or {}
        action_limits = initial_action.get("normalization_limits", {})
        metadata = {
            "schema": SCHEMA,
            "episode_id": self.record_dir.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": self._git_commit(),
            "sample_rate_hz": self.sample_rate_hz,
            "control_rate_hz": self.control_rate_hz,
            "time_source": self.time_source_name,
            "chunk_frames": self.chunk_frames,
            "actual_people_count": int(actual_people),
            "stored_people_count": int(self._episode_people_capacity),
            "source_joint_topology": "COCO17",
            "joint_topology": "COCO12_BODY",
            "source_joint_count": 17,
            "recorded_joint_count": len(BODY_JOINT_INDICES),
            "recorded_source_joint_indices": list(BODY_JOINT_INDICES),
            "joint_names": list(BODY_JOINT_NAMES),
            "recorded_skeleton_fields": [
                "human_xyz", "human_confidence", "human_joint_valid",
                "human_track_id",
            ],
            "skeleton_frame": "base_link",
            "coordinate_convention": "ROS_FLU: +X forward, +Y left, +Z up",
            "simulation_skeleton_depth_source": "isaac_gt_depth",
            "ego_state_order": list(EGO_STATE_KEYS),
            "action_order": list(ACTION_KEYS),
            "action_representation": "normalized_applied_body_flu",
            "action_normalization_limits": {
                key: float(action_limits.get(key, 1.0)) for key in ACTION_KEYS
            },
            "reward_component_order": list(REWARD_COMPONENT_KEYS),
            "termination_codes": dict(TERMINATION_CODES),
            "transition_alignment": (
                "row t stores observation_t, the action applied over "
                "observation_(t-1)->observation_t, and reward_t"
            ),
            "target_point": self.target_point.tolist(),
            "ego_reference_origin_xyz": list(self._ego_reference["origin_xyz"]),
            "ego_reference_origin_yaw": float(self._ego_reference["origin_yaw"]),
            "goal_position_episode_local": goal_position_episode_local(
                self.target_point, self._ego_reference).astype(float).tolist(),
            "goal_region": self.goal_region,
            "reward_config": self.reward_config,
            "privileged_policy_visible": False,
            "synchronization_tolerance_sec": self.sync_tolerance_sec,
            "skeleton_wait_wall_sec": self.skeleton_wait_wall_sec,
            "episode": metadata_extra,
        }
        self._write_json(self.record_dir / "metadata.json", metadata)
        carb.log_warn("[REC][V3] Recording started: {}".format(self.record_dir))

    def update(self, force=False):
        if not self.is_recording:
            return False
        episode_time = self._elapsed_time()
        capture_due = force or episode_time + 1e-9 >= self.next_sample_time
        if capture_due:
            self._advance_next_sample_time(episode_time, force=force)
            self._state_buffer.append(self._capture_state_snapshot())
            self._capture_count += 1

        packets = [] if self.skeleton_receiver is None else \
            self.skeleton_receiver.packets_after(self._last_packet_arrival_index)
        for item in packets:
            packet = item["packet"]
            arrival_index = int(item["arrival_index"])
            packet_time = float(packet["timestamp_ns"]) / 1e9
            snapshot = self._match_snapshot(packet_time)
            if snapshot is None:
                if self._state_buffer and packet_time > (
                        self._state_buffer[-1]["simulation_time_s"] +
                        self.sync_tolerance_sec):
                    break
                self._last_packet_arrival_index = arrival_index
                self._unmatched_skeleton_count += 1
                continue
            self._last_packet_arrival_index = arrival_index
            snapshot["skeleton_packet"] = packet
        return self._flush_ready_snapshots(force=force)

    def stop(self, reason="manual_stop", collision=None, reached_goal=None,
             event_details=None):
        if not self.is_recording:
            return
        self.set_event_status(
            collision=collision, reached_goal=reached_goal,
            termination_reason=reason, event_details=event_details)
        if reason != "recording" and not self._last_sample_written:
            self.update(force=True)
        self.is_recording = False
        duration = self._elapsed_time()
        wall_duration = self._elapsed_wall_time()
        self.write_queue.put(None)
        self.writer_thread.join()
        event = {
            "simulation_time_s": self._now(),
            "episode_time_s": duration,
            "reason": self.termination_reason,
            "collision": self.collision,
            "reached_goal": self.reached_goal,
            "details": self.event_details,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
        summary = {
            "schema": SCHEMA,
            "episode_id": self.record_dir.name,
            "duration_sec": duration,
            "wall_duration_sec": wall_duration,
            "captured_state_count": self._capture_count,
            "queued_frame_count": self.frame_index,
            "written_frame_count": self.written_frame_count,
            "chunk_count": self._chunk_count,
            "dropped_frame_count": self.dropped_frame_count,
            "unmatched_skeleton_count": self._unmatched_skeleton_count,
            "missing_skeleton_count": self._missing_skeleton_count,
            "collision": self.collision,
            "collision_human": bool(
                self.collision and self.event_details.get("category") == "human"),
            "collision_static": bool(
                self.collision and self.event_details.get("category") == "environment"),
            "reached_goal": self.reached_goal,
            "success": self.termination_reason == "reached_goal",
            "outcome_class": outcome_for_reason(self.termination_reason),
            "termination_reason": self.termination_reason,
            "event_details": self.event_details,
            "episode_return": self.episode_return,
            "reward_component_sums": self.reward_component_sums,
            "skeleton_valid_joint_source_counts": {
                str(key): int(value)
                for key, value in sorted(self._skeleton_source_counts.items())
            },
            "skeleton_receiver": (
                None if self.skeleton_receiver is None
                else self.skeleton_receiver.status()),
        }
        self._write_json(self.record_dir / "summary.json", summary)
        carb.log_warn(
            "[REC][V3] Recording stopped: {}, reason={}, frames={}, chunks={}".format(
                self.record_dir, reason, self.written_frame_count, self._chunk_count))
        self._reset_paths()

    def discard(self, reason="discarded", event_details=None):
        if not self.is_recording:
            return None
        record_dir = self.record_dir
        self.is_recording = False
        self.write_queue.put(None)
        self.writer_thread.join()
        self._reset_paths()
        try:
            shutil.rmtree(record_dir)
        except OSError as error:
            carb.log_warn("[REC][V3] Failed to discard {}: {}".format(
                record_dir, error))
        return record_dir

    def _reset_paths(self):
        self.record_dir = None
        self.frames_dir = None
        self.chunks_dir = None
        self.events_path = None
        self.start_time = None
        self.wall_start_time = None
        self.next_sample_time = 0.0
        self.write_queue = None
        self.writer_thread = None
        self.termination_reason = "none"

    def _capture_state_snapshot(self):
        simulation_time = float(self._now())
        drone_state = self._drone_state()
        if self._ego_reference is None:
            self._ego_reference = ego_reference(drone_state)
        drone_position = np.asarray(drone_state["position"], dtype=float)
        joint_distances = self.skeleton_tracker.get_joint_distances(drone_position)
        pelvis = self._pelvis_relative_positions(drone_state)
        return {
            "simulation_time_s": simulation_time,
            "capture_wall_monotonic": time.monotonic(),
            "drone_state": drone_state,
            "action": self._read_action(),
            "privileged": self._safe_provider(self.privileged_provider) or {},
            "joint_distances": joint_distances,
            "pelvis": pelvis,
            "skeleton_packet": None,
            "collision": bool(self.collision),
            "reached_goal": bool(self.reached_goal),
            "termination_reason": str(self.termination_reason),
        }

    def _match_snapshot(self, packet_time):
        candidates = [snapshot for snapshot in self._state_buffer
                      if snapshot["skeleton_packet"] is None]
        if not candidates:
            return None
        snapshot = min(candidates, key=lambda item: abs(
            item["simulation_time_s"] - packet_time))
        if abs(snapshot["simulation_time_s"] - packet_time) > self.sync_tolerance_sec:
            return None
        return snapshot

    def _flush_ready_snapshots(self, force=False):
        queued = False
        now = time.monotonic()
        while self._state_buffer:
            snapshot = self._state_buffer[0]
            packet = snapshot["skeleton_packet"]
            if packet is None and not force and now - float(
                    snapshot["capture_wall_monotonic"]) < self.skeleton_wait_wall_sec:
                break
            self._state_buffer.popleft()
            fresh = packet is not None
            if packet is None:
                packet = empty_skeleton_packet(
                    int(round(snapshot["simulation_time_s"] * 1e9)))
                self._missing_skeleton_count += 1
            queued = self._queue_sample(
                snapshot, packet, skeleton_fresh=fresh, force=force) or queued
        return queued

    def _queue_sample(self, snapshot, packet, skeleton_fresh, force=False):
        sample = self._build_sample(snapshot, packet, skeleton_fresh)
        queued = self._enqueue_frame(
            sample, block=force or not self.drop_when_writer_busy)
        if not queued:
            return False
        self.frame_index += 1
        reward = float(sample["reward"])
        self.episode_return += reward
        self.reward_frame_count += 1
        for index, name in enumerate(REWARD_COMPONENT_KEYS):
            self.reward_component_sums[name] += float(
                sample["reward_components"][index])
        if bool(sample["is_last"]):
            self._last_sample_written = True
        return True

    def _build_sample(self, snapshot, packet, skeleton_fresh):
        simulation_time = float(snapshot["simulation_time_s"])
        drone_state = snapshot["drone_state"]
        reward, components, diagnostics = self.reward_calculator.compute(
            simulation_time, drone_state,
            snapshot["joint_distances"], snapshot["pelvis"],
            collision=snapshot["collision"],
            reached_goal=snapshot["reached_goal"],
            termination_reason=snapshot["termination_reason"],
        )
        end_flags = transition_flags(snapshot["termination_reason"])
        first = self.frame_index == 0
        if first and not bool(end_flags["is_last"]):
            reward = 0.0
            components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
        skeleton = skeleton_to_arrays(packet, self._episode_people_capacity)
        for source, count in skeleton_source_counts(packet).items():
            self._skeleton_source_counts[source] = (
                self._skeleton_source_counts.get(source, 0) + count)
        action = normalized_applied_action(snapshot["action"])
        if first:
            action["action"] = np.zeros(4, np.float32)
            action["action_valid"] = np.asarray(False, np.bool_)
        privileged = privileged_to_arrays(
            snapshot["privileged"], self._episode_people_capacity)
        source = str(drone_state.get("source", "invalid"))
        sample = {
            "frame_index": np.asarray(self.frame_index, np.int64),
            "simulation_time_s": np.asarray(simulation_time, np.float64),
            "skeleton_timestamp_ns": np.asarray(packet["timestamp_ns"], np.int64),
            "skeleton_fresh": np.asarray(skeleton_fresh, np.bool_),
            "ego_state": build_ego_state(drone_state, self._ego_reference),
            "ego_altitude_valid": np.asarray(
                drone_state.get("altitude_agl") is not None, np.bool_),
            "ego_state_source": np.asarray(
                STATE_SOURCE_CODES.get(source, 0), np.uint8),
        }
        sample.update(skeleton)
        sample.update(action)
        sample.update({
            "reward": np.asarray(reward, np.float32),
            "reward_components": np.asarray(
                [components[name] for name in REWARD_COMPONENT_KEYS], np.float32),
            "is_first": np.asarray(first, np.bool_),
        })
        sample.update(end_flags)
        sample.update(privileged)
        sample.update({
            "priv_min_human_clearance_m": np.asarray(
                diagnostics.get("human_min_clearance_m") or 0.0, np.float32),
            "priv_min_human_clearance_valid": np.asarray(
                diagnostics.get("human_min_clearance_m") is not None, np.bool_),
            "priv_min_human_ttc_s": np.asarray(
                diagnostics.get("human_min_ttc_sec") or 0.0, np.float32),
            "priv_min_human_ttc_valid": np.asarray(
                diagnostics.get("human_min_ttc_sec") is not None, np.bool_),
            "priv_collision": np.asarray(snapshot["collision"], np.bool_),
            "priv_goal_distance_m": np.asarray(
                diagnostics.get("goal_distance_3d_m", 0.0), np.float32),
        })
        return sample

    def _writer_loop(self):
        pending = []
        while True:
            item = self.write_queue.get()
            stopping = item is None
            try:
                if stopping:
                    if pending:
                        self._write_chunk(pending)
                else:
                    pending.append(item)
                    if len(pending) >= self.chunk_frames:
                        self._write_chunk(pending)
                        pending = []
            except Exception as error:
                carb.log_warn("[REC][V3] Chunk write failed: {}".format(error))
            finally:
                self.write_queue.task_done()
            if stopping:
                return

    def _write_chunk(self, samples):
        arrays = stack_samples(samples)
        path = self.chunks_dir / "chunk_{:06d}.npz".format(self._chunk_count)
        write_npz_atomic(path, arrays)
        self._chunk_count += 1
        self.written_frame_count += len(samples)

    @staticmethod
    def _safe_provider(provider):
        if provider is None:
            return None
        try:
            return provider()
        except Exception as error:
            carb.log_warn("[REC][V3] Provider failed: {}".format(error))
            return None

    @staticmethod
    def _git_commit():
        repository = Path(__file__).resolve().parents[3]
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository,
                text=True, timeout=2.0).strip()
        except Exception:
            return None
