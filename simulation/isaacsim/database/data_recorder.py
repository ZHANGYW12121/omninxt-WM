#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import carb
import numpy as np
from scipy.spatial.transform import Rotation

from reward_model import CrowdRewardCalculator, REWARD_COMPONENT_KEYS


DEFAULT_REWARD_CONFIG = {
    "progress_weight_per_m": 2.0,
    "max_progress_speed_mps": 3.0,
    "goal_height_tolerance_m": 0.20,
    "human_clearance_weight_per_sec": 4.0,
    "human_hard_clearance_m": 0.10,
    "human_safe_clearance_m": 0.70,
    "drone_collision_radius_m": 0.32,
    "human_ttc_horizon_sec": 2.5,
    "human_predictive_safe_clearance_m": 0.90,
    "flight_corridor_half_width_m": 0.75,
    "flight_corridor_lookahead_m": 4.0,
    "acceleration_weight_per_sec": 0.15,
    "acceleration_scale_mps2": 3.0,
    "jerk_weight_per_sec": 0.10,
    "jerk_scale_mps3": 10.0,
    "yaw_rate_weight_per_sec": 0.05,
    "yaw_rate_scale_rps": 1.5,
    "smooth_term_clip": 4.0,
    "acceleration_filter_alpha": 0.25,
    "height_weight_per_sec": 0.5,
    "cruise_height_m": 1.0,
    "height_tolerance_m": 0.15,
    "height_scale_m": 0.50,
    "time_cost_per_sec": 0.10,
    "event_rewards": {
        "reached_goal": 100.0,
        "collision": -120.0,
        "human_collision": -120.0,
        "static_collision": -120.0,
        "out_of_bounds": -100.0,
        "crash": -100.0,
        "stuck_timeout": -20.0,
        "record_size_limit": 0.0,
        "time_limit": 0.0,
        "controller_error": 0.0,
        "manual_stop": 0.0,
        "shutdown": 0.0,
    },
}

JOINT_COLLISION_RADII = {
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


class DatasetRecorder:
    def __init__(
        self,
        drone,
        camera_sensor,
        skeleton_tracker,
        target_point=None,
        goal_region=None,
        dataset_root=None,
        sample_rate_hz=10.0,
        max_queue_size=80,
        image_format="jpg",
        jpeg_quality=85,
        json_indent=None,
        drop_when_writer_busy=True,
        time_source=None,
        time_source_name="wall_time",
        action_provider=None,
        state_provider=None,
        altitude_agl_provider=None,
        reward_config=None,
        control_rate_hz=None,
    ):
        self.drone = drone
        self.camera_sensor = camera_sensor
        self.skeleton_tracker = skeleton_tracker
        self.target_point = np.array(
            target_point if target_point is not None else [0.0, 122.0, 1.0],
            dtype=float,
        )
        self.goal_region = self._normalize_goal_region(goal_region)
        self.dataset_root = Path(dataset_root or Path(__file__).resolve().parent / "database" / "scene1")
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_period = 1.0 / self.sample_rate_hz
        self.max_queue_size = int(max_queue_size)
        self.image_format = image_format.lower()
        self.jpeg_quality = int(jpeg_quality)
        self.json_indent = json_indent
        self.drop_when_writer_busy = bool(drop_when_writer_busy)
        self.time_source = time_source
        self.time_source_name = str(time_source_name)
        self.action_provider = action_provider
        self.state_provider = state_provider
        self.altitude_agl_provider = altitude_agl_provider
        self.reward_config = self._normalize_reward_config(reward_config)
        calculator_config = dict(self.reward_config)
        calculator_config["joint_collision_radii"] = dict(JOINT_COLLISION_RADII)
        self.reward_calculator = CrowdRewardCalculator(
            calculator_config, self.target_point, self.goal_region,
        )
        self.control_rate_hz = None if control_rate_hz is None else float(control_rate_hz)

        self.is_recording = False
        self.record_dir = None
        self.frames_dir = None
        self.start_time = None
        self.wall_start_time = None
        self.next_sample_time = 0.0
        self.frame_index = 0
        self.written_frame_count = 0
        self.dropped_frame_count = 0
        self.write_queue = None
        self.writer_thread = None
        self.collision = False
        self.reached_goal = False
        self.termination_reason = "none"
        self.event_details = {}
        self.episode_return = 0.0
        self.reward_frame_count = 0
        self.reward_component_sums = self._empty_reward_component_sums()
        self._reward_prev_context = None

    def toggle_recording(self):
        if self.is_recording:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.is_recording:
            return

        record_name = datetime.now().strftime("record_%Y%m%d_%H%M%S")
        self.record_dir = self.dataset_root / record_name
        suffix = 1
        while self.record_dir.exists():
            self.record_dir = self.dataset_root / f"{record_name}_{suffix:02d}"
            suffix += 1

        self.frames_dir = self.record_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = self._now()
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
        self.write_queue = queue.Queue(maxsize=self.max_queue_size)
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="DatasetRecorderWriter",
            daemon=True,
        )
        self.writer_thread.start()
        self.is_recording = True

        metadata = {
            "record_name": self.record_dir.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "target_point": self.target_point.tolist(),
            "goal_region": self.goal_region,
            "sample_rate_hz": self.sample_rate_hz,
            "control_rate_hz": self.control_rate_hz,
            "time_source": self.time_source_name,
            "max_queue_size": self.max_queue_size,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "json_indent": self.json_indent,
            "drop_when_writer_busy": self.drop_when_writer_busy,
            "transition_alignment": (
                "action and reward describe the transition ending at this frame; "
                "the first frame has zero dense transition reward"
            ),
            "reward_config": self.reward_config,
            "frame_schema": {
                "timestamp": "seconds from recording start measured by time_source",
                "wall_timestamp": "real seconds from recording start",
                "image_path": "camera image saved as .png or .npy when available",
                "action": "requested normalized/physical velocity command and applied slew-limited reference",
                "drone_state": "position, velocity, acceleration, roll/pitch/yaw, and state source",
                "pedestrian_joint_distances": "pedestrian id, joint name, distance to drone",
                "pedestrian_pelvis_relative_positions": (
                    "Pelvis world position plus position relative to drone in world/body frames"
                ),
                "goal_region": "success region for the drone, when configured",
                "collision": "whether an Isaac Sim physics contact has occurred between drone and pedestrian skeleton colliders",
                "reached_goal": "whether drone has reached dataset goal",
                "termination_reason": (
                    "recording, reached_goal, human_collision, static_collision, "
                    "out_of_bounds, crash, stuck_timeout, time_limit, controller_error, "
                    "manual_stop, record_size_limit, or shutdown"
                ),
                "event_details": "collision or reached-goal details when an event has occurred",
                "reward": "environment task reward for the transition ending at this frame",
                "reward_components": "separately saved event/progress/safety/smoothness/height/time terms",
                "reward_diagnostics": "distances, risks, motion magnitudes, and dt used to compute reward",
            },
        }
        self._write_json(self.record_dir / "metadata.json", metadata)
        carb.log_warn(f"[REC] Recording started: {self.record_dir}")

    def stop(self, reason="manual_stop", collision=None, reached_goal=None, event_details=None):
        if not self.is_recording:
            return

        self.set_event_status(
            collision=collision,
            reached_goal=reached_goal,
            termination_reason=reason,
            event_details=event_details,
        )
        self.is_recording = False
        duration = self._elapsed_time()
        wall_duration = self._elapsed_wall_time()
        if self.write_queue is not None:
            self.write_queue.put(None)
        if self.writer_thread is not None:
            self.writer_thread.join()

        summary = {
            "record_name": self.record_dir.name,
            "duration_sec": duration,
            "wall_duration_sec": wall_duration,
            "time_source": self.time_source_name,
            "queued_frame_count": self.frame_index,
            "written_frame_count": self.written_frame_count,
            "dropped_frame_count": self.dropped_frame_count,
            "collision": self.collision,
            "reached_goal": self.reached_goal,
            "termination_reason": self.termination_reason,
            "event_details": self.event_details,
            "episode_return": self.episode_return,
            "reward_frame_count": self.reward_frame_count,
            "reward_component_sums": self.reward_component_sums,
        }
        self._write_json(self.record_dir / "summary.json", summary)
        carb.log_warn(
            f"[REC] Recording stopped: {self.record_dir}, reason={self.termination_reason}, "
            f"collision={self.collision}, reached_goal={self.reached_goal}, "
            f"written={self.written_frame_count}, duration={duration:.2f}s "
            f"({self.time_source_name}), wall_duration={wall_duration:.2f}s"
        )

        self.record_dir = None
        self.frames_dir = None
        self.start_time = None
        self.wall_start_time = None
        self.next_sample_time = 0.0
        self.frame_index = 0
        self.write_queue = None
        self.writer_thread = None
        self.termination_reason = "none"

    def discard(self, reason="discarded", event_details=None):
        if not self.is_recording:
            return None

        record_dir = self.record_dir
        self.set_event_status(
            collision=False,
            reached_goal=False,
            termination_reason=reason,
            event_details=event_details,
        )
        self.is_recording = False
        duration = self._elapsed_time()
        wall_duration = self._elapsed_wall_time()

        if self.write_queue is not None:
            self.write_queue.put(None)
        if self.writer_thread is not None:
            self.writer_thread.join()

        carb.log_warn(
            f"[REC] Discarding recording: {record_dir}, reason={self.termination_reason}, "
            f"queued={self.frame_index}, written={self.written_frame_count}, "
            f"duration={duration:.2f}s ({self.time_source_name}), "
            f"wall_duration={wall_duration:.2f}s"
        )

        self.record_dir = None
        self.frames_dir = None
        self.start_time = None
        self.wall_start_time = None
        self.next_sample_time = 0.0
        self.frame_index = 0
        self.write_queue = None
        self.writer_thread = None
        self.termination_reason = "none"

        if record_dir is None:
            return None

        try:
            shutil.rmtree(record_dir)
            carb.log_warn(f"[REC] Deleted discarded recording directory: {record_dir}")
        except Exception as exc:
            carb.log_warn(f"[REC] Failed to delete discarded recording directory {record_dir}: {exc}")
        return record_dir

    def set_event_status(
        self,
        collision=None,
        reached_goal=None,
        termination_reason=None,
        event_details=None,
    ):
        if collision is not None:
            self.collision = bool(collision)
        if reached_goal is not None:
            self.reached_goal = bool(reached_goal)
        if termination_reason is not None:
            self.termination_reason = str(termination_reason)
        if event_details:
            self.event_details.update(event_details)

    def update(self, force=False):
        if not self.is_recording:
            return False

        timestamp = self._elapsed_time()
        if not force and timestamp + 1e-9 < self.next_sample_time:
            return False

        if (
            not force
            and self.drop_when_writer_busy
            and self.write_queue is not None
            and self.write_queue.full()
        ):
            self._advance_next_sample_time(timestamp, force=force)
            self.dropped_frame_count += 1
            self.frame_index += 1
            return False

        self._advance_next_sample_time(timestamp, force=force)
        frame_stem = f"frame_{self.frame_index:06d}"
        drone_state = self._drone_state()
        image = self._read_camera_image()
        drone_position = np.array(drone_state["position"], dtype=float)
        joint_distances = self.skeleton_tracker.get_joint_distances(drone_position)
        pelvis_relative_positions = self._pelvis_relative_positions(drone_state)
        action = self._read_action()
        (
            reward,
            reward_components,
            reward_diagnostics,
            reward_next_context,
        ) = self._compute_reward(
            timestamp=timestamp,
            drone_state=drone_state,
            joint_distances=joint_distances,
            pelvis_relative_positions=pelvis_relative_positions,
        )

        frame_record = {
            "frame_index": self.frame_index,
            "frame_stem": frame_stem,
            "timestamp": timestamp,
            "wall_timestamp": self._elapsed_wall_time(),
            "image": image,
            "action": action,
            "drone_state": drone_state,
            "pedestrian_joint_distances": joint_distances,
            "pedestrian_pelvis_relative_positions": pelvis_relative_positions,
            "target_point": self.target_point.tolist(),
            "goal_region": self.goal_region,
            "collision": self.collision,
            "reached_goal": self.reached_goal,
            "termination_reason": self.termination_reason,
            "event_details": dict(self.event_details),
            "reward": reward,
            "reward_components": reward_components,
            "reward_diagnostics": reward_diagnostics,
        }
        queued = self._enqueue_frame(
            frame_record,
            block=force or not self.drop_when_writer_busy,
        )
        if queued:
            self._reward_prev_context = reward_next_context
            self.episode_return += float(reward)
            self.reward_frame_count += 1
            for name, value in reward_components.items():
                self.reward_component_sums[name] += float(value)
        self.frame_index += 1
        return queued

    def _advance_next_sample_time(self, timestamp, force=False):
        if force:
            self.next_sample_time = max(self.next_sample_time, timestamp + self.sample_period)
        elif self.next_sample_time == 0.0:
            # Anchor the sampling grid to the first successfully sampled
            # observation. The controller uses the same clock and period, so
            # subsequent observation and action boundaries stay aligned.
            self.next_sample_time = timestamp + self.sample_period
        else:
            while self.next_sample_time <= timestamp:
                self.next_sample_time += self.sample_period

    def _enqueue_frame(self, frame_record, block=False):
        try:
            if block:
                self.write_queue.put(frame_record)
            else:
                self.write_queue.put_nowait(frame_record)
            return True
        except queue.Full:
            self.dropped_frame_count += 1
            return False

    def _writer_loop(self):
        while True:
            item = self.write_queue.get()
            try:
                if item is None:
                    return
                self._write_frame_record(item)
                self.written_frame_count += 1
            except Exception as exc:
                carb.log_warn(
                    f"[REC] Failed to write frame {item.get('frame_stem', '<unknown>')}: {exc}"
                )
            finally:
                self.write_queue.task_done()

    def _write_frame_record(self, frame_record):
        frame_stem = frame_record["frame_stem"]
        frame_json_path = self.frames_dir / f"{frame_stem}.json"
        image_path = self._write_camera_image(frame_stem, frame_record["image"])

        record = {
            "frame_index": frame_record["frame_index"],
            "timestamp": frame_record["timestamp"],
            "wall_timestamp": frame_record["wall_timestamp"],
            "image_path": self._relative_path(image_path),
            "action": frame_record["action"],
            "drone_state": frame_record["drone_state"],
            "pedestrian_joint_distances": frame_record["pedestrian_joint_distances"],
            "pedestrian_pelvis_relative_positions": frame_record[
                "pedestrian_pelvis_relative_positions"
            ],
            "target_point": frame_record["target_point"],
            "goal_region": frame_record["goal_region"],
            "collision": frame_record["collision"],
            "reached_goal": frame_record["reached_goal"],
            "termination_reason": frame_record["termination_reason"],
            "event_details": frame_record["event_details"],
            "reward": frame_record["reward"],
            "reward_components": frame_record["reward_components"],
            "reward_diagnostics": frame_record["reward_diagnostics"],
        }
        self._write_json(frame_json_path, record)

    def _read_action(self):
        if self.action_provider is None:
            return None
        try:
            action = self.action_provider()
        except Exception as exc:
            carb.log_warn(f"[REC] Failed to read control action: {exc}")
            return None
        return action

    def _pelvis_relative_positions(self, drone_state):
        drone_position = np.asarray(drone_state.get("position", []), dtype=float)
        if drone_position.shape != (3,) or not np.all(np.isfinite(drone_position)):
            return []

        body_from_world = None
        quat_xyzw = np.asarray(drone_state.get("quaternion_xyzw", []), dtype=float)
        if quat_xyzw.shape == (4,) and np.all(np.isfinite(quat_xyzw)):
            try:
                body_from_world = Rotation.from_quat(quat_xyzw).inv()
            except Exception:
                body_from_world = None

        pelvis_records = []
        marker_positions = getattr(self.skeleton_tracker, "marker_positions", {}) or {}
        for pedestrian_id, joints in marker_positions.items():
            pelvis_position = joints.get("Pelvis") if joints is not None else None
            if pelvis_position is None:
                continue

            pelvis_world = np.asarray(pelvis_position, dtype=float)
            if pelvis_world.shape != (3,) or not np.all(np.isfinite(pelvis_world)):
                continue

            relative_world = pelvis_world - drone_position
            record = {
                "pedestrian_id": str(pedestrian_id),
                "joint_name": "Pelvis",
                "position_world_m": pelvis_world.astype(float).tolist(),
                "relative_position_world_m": relative_world.astype(float).tolist(),
                "distance": float(np.linalg.norm(relative_world)),
            }

            if body_from_world is not None:
                relative_body = body_from_world.apply(relative_world)
                record["relative_position_body_m"] = relative_body.astype(float).tolist()

            pelvis_records.append(record)

        return pelvis_records

    def _compute_reward(self, timestamp, drone_state, joint_distances,
                        pelvis_relative_positions=None):
        reward, components, diagnostics = self.reward_calculator.compute(
            timestamp, drone_state, joint_distances, pelvis_relative_positions,
            collision=self.collision, reached_goal=self.reached_goal,
            termination_reason=self.termination_reason,
        )
        return reward, components, diagnostics, dict(self.reward_calculator.previous)

    def _compute_reward_v1_legacy(self, timestamp, drone_state, joint_distances):
        """Original v1 formula retained only for old-dataset A/B comparison."""
        cfg = self.reward_config
        previous = self._reward_prev_context
        dt = 0.0 if previous is None else max(0.0, float(timestamp) - previous["timestamp"])

        position = np.asarray(drone_state["position"], dtype=float)
        goal_distance = self._goal_distance(position)
        raw_progress = 0.0 if previous is None else previous["goal_distance"] - goal_distance
        max_progress = float(cfg["max_progress_speed_mps"]) * dt
        used_progress = float(np.clip(raw_progress, -max_progress, max_progress)) if dt > 0.0 else 0.0
        progress_reward = float(cfg["progress_weight_per_m"]) * used_progress

        human_min_center_distance = None
        human_min_clearance = None
        human_risk = 0.0
        drone_radius = float(cfg["drone_collision_radius_m"])
        for item in joint_distances or ():
            try:
                distance = float(item["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(distance):
                continue
            joint_radius = JOINT_COLLISION_RADII.get(str(item.get("joint_name", "")), 0.08)
            clearance = distance - drone_radius - joint_radius
            risk = self._clearance_risk(
                clearance,
                float(cfg["human_hard_clearance_m"]),
                float(cfg["human_safe_clearance_m"]),
            )
            human_risk = max(human_risk, risk)
            if human_min_center_distance is None or distance < human_min_center_distance:
                human_min_center_distance = distance
            if human_min_clearance is None or clearance < human_min_clearance:
                human_min_clearance = clearance
        human_reward = -float(cfg["human_clearance_weight_per_sec"]) * dt * human_risk

        acceleration_xy = np.asarray(drone_state["acceleration"][:2], dtype=float)
        alpha = float(np.clip(cfg["acceleration_filter_alpha"], 0.0, 1.0))
        if previous is None:
            filtered_acceleration_xy = acceleration_xy
            jerk_xy = np.zeros(2, dtype=float)
            yaw_rate = 0.0
        else:
            filtered_acceleration_xy = (
                alpha * acceleration_xy
                + (1.0 - alpha) * previous["filtered_acceleration_xy"]
            )
            if dt > 1e-9:
                jerk_xy = (
                    filtered_acceleration_xy - previous["filtered_acceleration_xy"]
                ) / dt
                yaw_delta = self._wrap_pi(float(drone_state["yaw"]) - previous["yaw"])
                yaw_rate = yaw_delta / dt
            else:
                jerk_xy = np.zeros(2, dtype=float)
                yaw_rate = 0.0

        acceleration_norm = float(np.linalg.norm(filtered_acceleration_xy))
        jerk_norm = float(np.linalg.norm(jerk_xy))
        term_clip = float(cfg["smooth_term_clip"])
        acceleration_term = min(
            (acceleration_norm / max(float(cfg["acceleration_scale_mps2"]), 1e-6)) ** 2,
            term_clip,
        )
        jerk_term = min(
            (jerk_norm / max(float(cfg["jerk_scale_mps3"]), 1e-6)) ** 2,
            term_clip,
        )
        yaw_rate_term = min(
            (abs(yaw_rate) / max(float(cfg["yaw_rate_scale_rps"]), 1e-6)) ** 2,
            term_clip,
        )
        smooth_reward = -dt * (
            float(cfg["acceleration_weight_per_sec"]) * acceleration_term
            + float(cfg["jerk_weight_per_sec"]) * jerk_term
            + float(cfg["yaw_rate_weight_per_sec"]) * yaw_rate_term
        )

        height_error = abs(float(position[2]) - float(cfg["cruise_height_m"]))
        height_excess = max(0.0, height_error - float(cfg["height_tolerance_m"]))
        height_term = min(
            (height_excess / max(float(cfg["height_scale_m"]), 1e-6)) ** 2,
            term_clip,
        )
        height_reward = -float(cfg["height_weight_per_sec"]) * dt * height_term
        time_reward = -float(cfg["time_cost_per_sec"]) * dt

        reward_reason = self.termination_reason
        if self.collision:
            reward_reason = "collision"
        elif self.reached_goal:
            reward_reason = "reached_goal"
        event_reward = float(cfg["event_rewards"].get(reward_reason, 0.0))

        components = {
            "event": event_reward,
            "progress": float(progress_reward),
            "human_clearance": float(human_reward),
            "smoothness": float(smooth_reward),
            "height": float(height_reward),
            "time": float(time_reward),
        }
        reward = float(sum(components.values()))
        diagnostics = {
            "dt_sec": float(dt),
            "goal_distance_m": float(goal_distance),
            "goal_progress_raw_m": float(raw_progress),
            "goal_progress_used_m": float(used_progress),
            "human_min_joint_center_distance_m": self._optional_float(human_min_center_distance),
            "human_min_clearance_m": self._optional_float(human_min_clearance),
            "human_risk": float(human_risk),
            "filtered_horizontal_acceleration_mps2": acceleration_norm,
            "filtered_horizontal_jerk_mps3": jerk_norm,
            "yaw_rate_rps": float(yaw_rate),
            "height_error_m": float(height_error),
            "event_reason": str(reward_reason),
        }
        next_context = {
            "timestamp": float(timestamp),
            "goal_distance": float(goal_distance),
            "filtered_acceleration_xy": filtered_acceleration_xy,
            "yaw": float(drone_state["yaw"]),
        }
        return reward, components, diagnostics, next_context

    def _goal_distance(self, position):
        x, y = float(position[0]), float(position[1])
        if self.goal_region is None:
            return float(np.linalg.norm(np.asarray([x, y]) - self.target_point[:2]))
        x_min, x_max = self.goal_region["x_range"]
        dx = max(float(x_min) - x, 0.0, x - float(x_max))
        dy = max(float(self.goal_region["y_min"]) - y, 0.0)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _clearance_risk(clearance, hard_clearance, safe_clearance):
        if clearance >= safe_clearance:
            return 0.0
        if clearance <= hard_clearance:
            return 1.0
        span = max(safe_clearance - hard_clearance, 1e-6)
        return float(((safe_clearance - clearance) / span) ** 2)

    @staticmethod
    def _wrap_pi(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _optional_float(value):
        return None if value is None else float(value)

    @staticmethod
    def _empty_reward_component_sums():
        return {key: 0.0 for key in REWARD_COMPONENT_KEYS}

    @staticmethod
    def _normalize_reward_config(reward_config):
        config = dict(DEFAULT_REWARD_CONFIG)
        config["event_rewards"] = dict(DEFAULT_REWARD_CONFIG["event_rewards"])
        if reward_config:
            supplied = dict(reward_config)
            supplied_events = supplied.pop("event_rewards", None)
            config.update(supplied)
            if supplied_events:
                config["event_rewards"].update(dict(supplied_events))
        if float(config["human_safe_clearance_m"]) <= float(config["human_hard_clearance_m"]):
            raise ValueError("human_safe_clearance_m must exceed human_hard_clearance_m")
        return config

    def _drone_state(self):
        state = None
        source = "isaac_ground_truth"
        if self.state_provider is not None:
            try:
                state = self.state_provider()
            except Exception as exc:
                carb.log_warn(f"[REC] Failed to read state provider: {exc}")
                state = None
            if state is not None:
                source = "mavsdk_px4_telemetry"
        if state is None:
            state = self.drone.state
            source = "isaac_ground_truth_fallback"
        position = np.array(state.position, dtype=float)
        velocity = np.array(state.linear_velocity, dtype=float)
        acceleration = np.array(state.linear_acceleration, dtype=float)
        quat_xyzw = np.array(state.attitude, dtype=float)
        roll, pitch, yaw = Rotation.from_quat(quat_xyzw).as_euler("XYZ", degrees=False)

        return {
            "position": position.tolist(),
            "velocity": velocity.tolist(),
            "acceleration": acceleration.tolist(),
            "roll_pitch_yaw_rad": [float(roll), float(pitch), float(yaw)],
            "roll_pitch_yaw_deg": np.degrees([roll, pitch, yaw]).astype(float).tolist(),
            "quaternion_xyzw": quat_xyzw.tolist(),
            "source": source,
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "vz": float(velocity[2]),
            "ax": float(acceleration[0]),
            "ay": float(acceleration[1]),
            "az": float(acceleration[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "altitude_agl": self._read_altitude_agl(position),
        }

    def _read_altitude_agl(self, position):
        """Return real height above ground or None; never substitute world Z."""
        if self.altitude_agl_provider is None:
            return None
        try:
            value = self.altitude_agl_provider(np.asarray(position, dtype=float))
            value = None if value is None else float(value)
            return value if value is not None and np.isfinite(value) and value >= 0.0 else None
        except Exception as exc:
            carb.log_warn(f"[REC] Failed to read altitude AGL: {exc}")
            return None

    @staticmethod
    def _normalize_goal_region(goal_region):
        if goal_region is None:
            return None
        return {
            "x_range": [float(goal_region["x_range"][0]), float(goal_region["x_range"][1])],
            "y_min": float(goal_region["y_min"]),
        }

    def _write_camera_image(self, frame_stem, image):
        if image is None:
            return None

        image = self._image_to_uint8(np.asarray(image))
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.image_format in ("jpg", "jpeg"):
            image_path = self.frames_dir / f"{frame_stem}_camera.jpg"
        elif self.image_format == "png":
            image_path = self.frames_dir / f"{frame_stem}_camera.png"
        else:
            image_path = self.frames_dir / f"{frame_stem}_camera.ppm"

        try:
            from PIL import Image

            pil_image = Image.fromarray(image)
            if self.image_format in ("jpg", "jpeg"):
                pil_image.save(image_path, quality=self.jpeg_quality, optimize=False)
            elif self.image_format == "png":
                pil_image.save(image_path)
            else:
                raise ValueError("Use PPM fallback")
            return image_path
        except Exception:
            ppm_path = self.frames_dir / f"{frame_stem}_camera.ppm"
            if self._save_ppm(ppm_path, image):
                return ppm_path
            npy_path = self.frames_dir / f"{frame_stem}_camera.npy"
            np.save(npy_path, image)
            return npy_path

    def _read_camera_image(self):
        camera = None
        try:
            if self.camera_sensor is not None:
                if getattr(self.camera_sensor, "state", None) and "camera" in self.camera_sensor.state:
                    camera = self.camera_sensor.state["camera"]
                else:
                    camera = getattr(self.camera_sensor, "_camera", None)
        except Exception:
            camera = None

        if camera is None:
            return None

        for method_name in ("get_rgb_image", "get_rgb", "get_rgba"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                image = method()
                if image is not None:
                    return np.asarray(image).copy()
            except Exception:
                continue
        return None

    @staticmethod
    def _image_to_uint8(image):
        if image.dtype == np.uint8:
            return image
        image = np.nan_to_num(image)
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.max(image)) if image.size else 1.0
            if max_value <= 1.0:
                image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _save_ppm(path, image):
        try:
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=2)
            if image.ndim != 3:
                return False
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            height, width = image.shape[:2]
            with open(path, "wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                f.write(np.ascontiguousarray(image[:, :, :3]).tobytes())
            return True
        except Exception:
            return False

    def _now(self):
        if self.time_source is not None:
            try:
                value = self.time_source()
                if value is not None and np.isfinite(float(value)):
                    return float(value)
            except Exception:
                pass
        return time.perf_counter()

    def _elapsed_time(self):
        if self.start_time is None:
            return 0.0
        return max(0.0, self._now() - float(self.start_time))

    def _elapsed_wall_time(self):
        if self.wall_start_time is None:
            return 0.0
        return max(0.0, time.perf_counter() - float(self.wall_start_time))

    def elapsed_time(self):
        return self._elapsed_time()

    def elapsed_wall_time(self):
        return self._elapsed_wall_time()

    def current_record_size_bytes(self):
        if self.record_dir is None or not self.record_dir.exists():
            return 0

        total_size = 0
        for path in self.record_dir.rglob("*"):
            try:
                if path.is_file():
                    total_size += path.stat().st_size
            except OSError:
                continue
        return total_size

    def _relative_path(self, path):
        if path is None:
            return None
        return str(Path(path).relative_to(self.record_dir))

    def _write_json(self, path, data):
        separators = (",", ":") if self.json_indent is None else None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=self.json_indent,
                separators=separators,
            )
