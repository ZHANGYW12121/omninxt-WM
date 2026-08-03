#!/usr/bin/env python3
"""Independent evaluator for one isolated EGO-Planner or NavRL trial."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import carb
import numpy as np

from app_config import CLASSIC_DRONE_RADIUS


_CAPSULE_LINKS = (
    ("Pelvis", "Head", 0.18),
    ("Pelvis", "R_KneeShareBone", 0.12),
    ("Pelvis", "L_KneeShareBone", 0.12),
    ("R_KneeShareBone", "R_Foot", 0.10),
    ("L_KneeShareBone", "L_Foot", 0.10),
    ("Head", "R_ElbowShareBone", 0.10),
    ("Head", "L_ElbowShareBone", 0.10),
    ("R_ElbowShareBone", "R_Hand", 0.08),
    ("L_ElbowShareBone", "L_Hand", 0.08),
)

DEFAULT_HUMAN_INTRUSION_THRESHOLD_M = 0.50
DEFAULT_HUMAN_INTRUSION_EXIT_THRESHOLD_M = 0.60


def _finite_vector(value, size=3):
    try:
        vector = np.asarray(value, dtype=float).reshape(size)
    except Exception:
        return None
    return vector if np.all(np.isfinite(vector)) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _point_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, segment) / length_sq)
    ratio = min(1.0, max(0.0, ratio))
    closest = start + ratio * segment
    return float(np.linalg.norm(point - closest))


class NavigationBenchmarkEvaluator:
    """Collect navigation metrics without feeding them back to either policy."""

    def __init__(self, app, config):
        self.app = app
        self.config = dict(config or {})
        self.seed = int(self.config["seed"])
        self.repeat_index = int(self.config.get("repeat_index", 1))
        self.run_id = str(self.config.get("run_id") or "")
        self.evaluation_hz = float(self.config.get("evaluation_hz", 10.0))
        self.timeout_sec = float(self.config.get("timeout_sec", 120.0))
        self.goal_radius_m = float(self.config.get("goal_radius_m", 0.80))
        self.human_intrusion_threshold_m = float(
            self.config.get(
                "human_intrusion_threshold_m",
                DEFAULT_HUMAN_INTRUSION_THRESHOLD_M,
            )
        )
        self.human_intrusion_exit_threshold_m = float(
            self.config.get(
                "human_intrusion_exit_threshold_m",
                DEFAULT_HUMAN_INTRUSION_EXIT_THRESHOLD_M,
            )
        )
        if not 0.0 < self.evaluation_hz <= 25.0:
            raise ValueError("benchmark evaluation_hz must be in (0, 25]")
        if self.human_intrusion_threshold_m <= 0.0:
            raise ValueError("human intrusion threshold must be positive")
        if (
            self.human_intrusion_exit_threshold_m
            <= self.human_intrusion_threshold_m
        ):
            raise ValueError(
                "human intrusion exit threshold must exceed entry threshold"
            )
        self.period = 1.0 / self.evaluation_hz

        goal = _finite_vector(getattr(app.crowd_scene, "goal_point", None))
        if goal is None:
            raise RuntimeError(
                "benchmark requires server_v2 crowd_scene.goal_point"
            )
        self.goal = goal
        self.controller = app.classic_controller
        if self.controller is None:
            raise RuntimeError("benchmark requires an autonomous controller")
        self.controller.configure_benchmark(self.goal, self.goal_radius_m)

        default_output = (
            Path(__file__).resolve().parent
            / "benchmark_results"
            / f"{app.control_mode}_seed_{self.seed:03d}_repeat_{self.repeat_index:02d}.json"
        )
        self.output_path = Path(
            self.config.get("output_path") or default_output
        ).expanduser().resolve()

        self.started = False
        self.finished = False
        self.start_sim_time = None
        self.start_wall_time = None
        self.start_position = None
        self.end_position = None
        self._last_sample_sim = None
        self._last_sample_position = None
        self._last_human_markers = None
        self._last_update_sim = None
        self.path_length_3d = 0.0
        self.speed_integral = 0.0
        self.speed_duration = 0.0
        self.intrusion_duration = 0.0
        self.intrusion_event_count = 0
        self._intrusion_active = False
        self.clearance_valid_duration = 0.0
        self.clearance_valid_samples = 0
        self.min_human_clearance_m = float("inf")
        self.sample_count = 0
        self.latencies_ms = []
        self.latency_records = []
        self.reference = None
        self._pending_collision = None
        self._pending_failure = None
        self._event_lock = threading.Lock()

        carb.log_warn(
            f"[BENCH] Prepared run={self.run_id or 'single'}, "
            f"algorithm={self.algorithm}, seed={self.seed}, repeat={self.repeat_index}, "
            f"goal=({self.goal[0]:.3f},{self.goal[1]:.3f},{self.goal[2]:.3f}), "
            f"success_radius_3d={self.goal_radius_m:.2f}m, "
            f"intrusion_entry={self.human_intrusion_threshold_m:.2f}m, "
            f"intrusion_exit={self.human_intrusion_exit_threshold_m:.2f}m, "
            f"eval_hz={self.evaluation_hz:.1f}."
        )

    @property
    def algorithm(self):
        configured = str(self.config.get("algorithm") or "").strip().lower()
        if configured:
            return configured
        if "ego" in self.app.control_mode:
            return "ego"
        if "dpmpc" in self.app.control_mode:
            return "dpmpc"
        return "navrl"

    def record_collision(self, simulation_time, details):
        """Called from the PhysX contact stream; takeoff contacts are ignored."""
        if (
            self.finished
            or not self.started
            or getattr(self.controller, "state", None) != "navigate"
        ):
            return
        event = {
            "simulation_time": float(simulation_time),
            "details": _json_safe(details),
        }
        with self._event_lock:
            if self._pending_collision is None:
                self._pending_collision = event

    def fail(self, reason, details=None):
        if self.finished:
            return
        with self._event_lock:
            if self._pending_failure is None:
                self._pending_failure = {
                    "reason": str(reason),
                    "details": _json_safe(details or {}),
                }

    def update(self):
        if self.finished:
            return
        now = float(self.app._simulation_time())
        self._last_update_sim = now
        self._consume_latency_records()

        if not self.started:
            takeoff_start = getattr(self.controller, "takeoff_start_time", None)
            if takeoff_start is None:
                return
            position = self._position()
            if position is None:
                return
            captured_start = _finite_vector(
                getattr(self.controller, "takeoff_start_position", None)
            )
            self._start(
                float(takeoff_start),
                captured_start if captured_start is not None else position,
            )

        collision, failure = self._pending_events()
        position = self._position()
        if position is None:
            self.fail("invalid_vehicle_state", {"simulation_time": now})
            collision, failure = self._pending_events()

        distance = (
            float(np.linalg.norm(position - self.goal))
            if position is not None
            else float("inf")
        )
        timed_out = now - self.start_sim_time >= self.timeout_sec
        terminating = collision is not None or failure is not None or distance <= self.goal_radius_m or timed_out
        self._sample(now, force=terminating)
        # Swept human collision detection runs inside _sample(). Re-read the
        # pending events so an analytic collision wins over goal arrival in the
        # same update, just like a PhysX contact.
        collision, failure = self._pending_events()

        # A physical collision has priority over reaching the goal in the same
        # render/update interval.
        if collision is not None:
            details = dict(collision.get("details") or {})
            details["contact_simulation_time"] = collision.get("simulation_time")
            self._finish("collision", success=False, details=details)
        elif failure is not None:
            self._finish(
                str(failure.get("reason") or "controller_error"),
                success=False,
                details=failure.get("details") or {},
            )
        elif distance <= self.goal_radius_m:
            self._finish(
                "reached_goal",
                success=True,
                details={"goal_distance_3d_m": distance},
            )
        elif timed_out:
            self._finish(
                "timeout",
                success=False,
                details={"limit_sim_sec": self.timeout_sec},
            )

    def _start(self, start_sim_time, position):
        self.started = True
        self.start_sim_time = float(start_sim_time)
        self.start_wall_time = time.perf_counter()
        self.start_position = position.copy()
        self._last_sample_sim = self.start_sim_time
        self._last_sample_position = position.copy()
        self.reference = self._static_reference_path(position, self.goal)
        carb.log_warn(
            f"[BENCH] Takeoff timer started at sim={self.start_sim_time:.3f}s, "
            f"start=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})."
        )

    def _position(self):
        try:
            return _finite_vector(self.app.drone.state.position)
        except Exception:
            return None

    def _velocity(self):
        try:
            return _finite_vector(self.app.drone.state.linear_velocity)
        except Exception:
            return None

    def _pending_events(self):
        with self._event_lock:
            return self._pending_collision, self._pending_failure

    def _sample(self, now, force=False):
        if not self.started:
            return
        if (
            not force
            and self._last_sample_sim is not None
            and now - self._last_sample_sim < self.period - 1e-9
        ):
            return
        position = self._position()
        velocity = self._velocity()
        if position is None or velocity is None:
            return

        previous_time = self._last_sample_sim
        dt = 0.0 if previous_time is None else max(0.0, now - previous_time)
        if self._last_sample_position is not None:
            self.path_length_3d += float(
                np.linalg.norm(position - self._last_sample_position)
            )
        self.speed_integral += float(np.linalg.norm(velocity)) * dt
        self.speed_duration += dt

        current_markers = self._human_marker_snapshot()
        clearance_result = self._human_clearance_over_interval(
            self._last_sample_position,
            position,
            self._last_human_markers,
            current_markers,
            dt,
        )
        if clearance_result is not None:
            clearance = float(clearance_result["minimum_clearance_m"])
            self.min_human_clearance_m = min(
                self.min_human_clearance_m, clearance
            )
            self.intrusion_duration += (
                dt * float(clearance_result["intrusion_fraction"])
            )
            self._update_intrusion_events(
                clearance_result["clearance_samples_m"]
            )
            self.clearance_valid_duration += dt
            self.clearance_valid_samples += 1
            collision_details = clearance_result.get("collision_details")
            if collision_details is not None:
                self.record_collision(
                    now,
                    {
                        "category": "human",
                        "source": "analytic_swept_joint_capsule_clearance",
                        "minimum_surface_clearance_m": clearance,
                        **collision_details,
                    },
                )

        self.sample_count += 1
        self._last_sample_sim = now
        self._last_sample_position = position.copy()
        self._last_human_markers = current_markers

    def _update_intrusion_events(self, clearance_samples):
        """Count outside-to-inside transitions with exit hysteresis."""
        for clearance in clearance_samples:
            clearance = float(clearance)
            if not math.isfinite(clearance):
                continue
            if self._intrusion_active:
                if clearance >= self.human_intrusion_exit_threshold_m:
                    self._intrusion_active = False
            elif clearance < self.human_intrusion_threshold_m:
                self.intrusion_event_count += 1
                self._intrusion_active = True

    def _human_marker_snapshot(self):
        tracker = self.app.skeleton_tracker
        snapshot = {}
        for person_name, joints in tracker.marker_positions.items():
            valid = {}
            for joint_name, position in joints.items():
                vector = _finite_vector(position)
                if vector is not None:
                    valid[joint_name] = vector.copy()
            if valid:
                snapshot[str(person_name)] = valid
        return snapshot

    def _human_clearance_over_interval(
        self,
        previous_drone,
        current_drone,
        previous_markers,
        current_markers,
        dt,
    ):
        if not current_markers:
            return None
        if previous_drone is None or not previous_markers or dt <= 1e-9:
            alphas = (1.0,)
        else:
            # Interpolate within the normal 10 Hz evaluation interval so a
            # short crossing cannot pass completely between two samples.
            alphas = tuple(np.linspace(0.0, 1.0, 5))

        minimum = float("inf")
        intrusion_hits = 0
        valid_count = 0
        clearance_samples = []
        collision_details = None
        for alpha in alphas:
            drone_position = (
                current_drone
                if previous_drone is None
                else (1.0 - alpha) * previous_drone + alpha * current_drone
            )
            markers = self._interpolate_markers(
                previous_markers, current_markers, alpha
            )
            clearance, details = self._human_surface_clearance(
                drone_position, markers
            )
            if clearance is None:
                continue
            valid_count += 1
            clearance = float(clearance)
            clearance_samples.append(clearance)
            minimum = min(minimum, clearance)
            if clearance < self.human_intrusion_threshold_m:
                intrusion_hits += 1
            if clearance <= 0.0 and collision_details is None:
                collision_details = dict(details or {})
                collision_details["sweep_fraction"] = float(alpha)

        if valid_count == 0:
            return None
        return {
            "minimum_clearance_m": minimum,
            "intrusion_fraction": intrusion_hits / valid_count,
            "clearance_samples_m": clearance_samples,
            "collision_details": collision_details,
        }

    @staticmethod
    def _interpolate_markers(previous, current, alpha):
        previous = previous or {}
        result = {}
        for person_name, current_joints in current.items():
            previous_joints = previous.get(person_name, {})
            interpolated = {}
            for joint_name, current_position in current_joints.items():
                previous_position = previous_joints.get(joint_name)
                interpolated[joint_name] = (
                    current_position
                    if previous_position is None
                    else (1.0 - alpha) * previous_position
                    + alpha * current_position
                )
            if interpolated:
                result[person_name] = interpolated
        return result

    def _human_surface_clearance(self, drone_position, marker_positions):
        tracker = self.app.skeleton_tracker
        minimum = float("inf")
        details = None
        for person_name, valid in marker_positions.items():
            for joint_name, joint_position in valid.items():
                radius = float(tracker.DEFAULT_COLLISION_RADII.get(joint_name, 0.08))
                clearance = (
                    float(np.linalg.norm(drone_position - joint_position))
                    - float(CLASSIC_DRONE_RADIUS)
                    - radius
                )
                if clearance < minimum:
                    minimum = clearance
                    details = {
                        "pedestrian_id": person_name,
                        "human_component": "joint_sphere",
                        "joint_name": joint_name,
                    }
            for start_name, end_name, radius in _CAPSULE_LINKS:
                if start_name not in valid or end_name not in valid:
                    continue
                clearance = (
                    _point_segment_distance(
                        drone_position, valid[start_name], valid[end_name]
                    )
                    - float(CLASSIC_DRONE_RADIUS)
                    - float(radius)
                )
                if clearance < minimum:
                    minimum = clearance
                    details = {
                        "pedestrian_id": person_name,
                        "human_component": "bone_capsule",
                        "capsule_start_joint": start_name,
                        "capsule_end_joint": end_name,
                    }
        return (minimum, details) if details is not None else (None, None)

    def _static_reference_path(self, start, goal):
        delta = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
        return {
            "available": True,
            "method": "direct_3d_euclidean_static_open_space",
            "length_xy_m": float(np.linalg.norm(delta[:2])),
            "delta_z_m": float(delta[2]),
            "length_3d_m": float(np.linalg.norm(delta)),
            "start_xyz": np.asarray(start, dtype=float).tolist(),
            "goal_xyz": np.asarray(goal, dtype=float).tolist(),
            "assumption": (
                "the seeded Warehouse start-goal corridor is statically clear "
                "when pedestrians are removed"
            ),
        }

    def _finish(self, reason, success, details):
        if self.finished:
            return
        now = float(self._last_update_sim)
        if (
            self._last_sample_sim is None
            or now - self._last_sample_sim > 1e-9
        ):
            self._sample(now, force=True)
        self.finished = True
        self._consume_latency_records()
        self.end_position = self._position()
        elapsed_sim = max(0.0, now - float(self.start_sim_time))
        elapsed_wall = max(0.0, time.perf_counter() - float(self.start_wall_time))
        intrusion_ratio = (
            self.intrusion_duration / elapsed_sim if elapsed_sim > 1e-9 else 0.0
        )
        clearance_valid_ratio = (
            self.clearance_valid_duration / elapsed_sim
            if elapsed_sim > 1e-9
            else 0.0
        )
        intrusion_ratio_valid = (
            self.intrusion_duration / self.clearance_valid_duration
            if self.clearance_valid_duration > 1e-9
            else None
        )
        average_speed = (
            self.speed_integral / self.speed_duration
            if self.speed_duration > 1e-9
            else 0.0
        )
        p95 = (
            float(np.percentile(np.asarray(self.latencies_ms, dtype=float), 95))
            if self.latencies_ms
            else None
        )
        latency_by_source = {}
        for record in self.latency_records:
            latency_by_source.setdefault(str(record["source"]), []).append(
                float(record["latency_ms"])
            )
        latency_source_summary = {
            source: {
                "sample_count": len(values),
                "p95_ms": float(np.percentile(values, 95)) if values else None,
                "mean_ms": float(np.mean(values)) if values else None,
            }
            for source, values in sorted(latency_by_source.items())
        }
        reference_length = (
            self.reference.get("length_3d_m")
            if self.reference and self.reference.get("available")
            else None
        )
        efficiency = (
            float(reference_length) / self.path_length_3d
            if success
            and reference_length is not None
            and self.path_length_3d > 1e-9
            else None
        )
        collision_details = details if reason == "collision" else {}
        collision_category = collision_details.get("category")
        scene = self.app.crowd_scene
        scene_fingerprint, scene_fingerprint_payload = self._scene_fingerprint()
        latency_metadata = self.controller.decision_latency_metadata()
        result = {
            "schema_version": 3,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "safety_layers_enabled": self.config.get(
                "safety_layers_enabled"
            ),
            "control_mode": self.app.control_mode,
            "seed": self.seed,
            "repeat_index": self.repeat_index,
            "success": bool(success),
            "termination_reason": str(reason),
            "collision": reason == "collision",
            "collision_human": reason == "collision" and collision_category == "human",
            "collision_static": reason == "collision" and collision_category == "environment",
            "collision_details": collision_details,
            "task_time_sim_sec": elapsed_sim if success else None,
            "task_time_sim_sec_raw": elapsed_sim,
            "task_time_wall_sec_raw": elapsed_wall,
            "start_position": self.start_position,
            "goal_position": self.goal,
            "end_position": self.end_position,
            "goal_radius_3d_m": self.goal_radius_m,
            "final_goal_distance_3d_m": (
                float(np.linalg.norm(self.end_position - self.goal))
                if self.end_position is not None
                else None
            ),
            "path_length_3d_m": self.path_length_3d if success else None,
            "path_length_3d_m_raw": self.path_length_3d,
            "average_speed_3d_mps": average_speed if success else None,
            "average_speed_3d_mps_raw": average_speed,
            "reference_path": self.reference,
            "reference_length_3d_m": reference_length,
            "efficiency": efficiency,
            "human_intrusion_threshold_m": self.human_intrusion_threshold_m,
            "human_intrusion_exit_threshold_m": (
                self.human_intrusion_exit_threshold_m
            ),
            "human_intrusion_ratio": intrusion_ratio,
            "human_intrusion_ratio_valid_coverage": intrusion_ratio_valid,
            "human_intrusion_event_count": self.intrusion_event_count,
            "human_intrusion_event_definition": (
                "one event starts when global minimum human surface clearance "
                "crosses below the entry threshold; it must reach the exit "
                "threshold before another event can start"
            ),
            "human_clearance_valid_duration_sec": self.clearance_valid_duration,
            "human_clearance_valid_ratio": clearance_valid_ratio,
            "human_clearance_valid_samples": self.clearance_valid_samples,
            "min_human_surface_clearance_m": (
                self.min_human_clearance_m
                if math.isfinite(self.min_human_clearance_m)
                else None
            ),
            "evaluation_hz": self.evaluation_hz,
            "evaluation_samples": self.sample_count,
            "decision_latency_definition": latency_metadata["definition"],
            "decision_latency_causal_pairing": latency_metadata["causal_pairing"],
            "decision_latency_note": latency_metadata.get("note"),
            "decision_latency_samples_ms": self.latencies_ms,
            "decision_latency_records": self.latency_records,
            "decision_latency_by_source": latency_source_summary,
            "decision_latency_policy_p95_ms": (
                latency_source_summary.get("policy", {}).get("p95_ms")
            ),
            "decision_latency_direct_goal_p95_ms": (
                latency_source_summary.get("direct_goal", {}).get("p95_ms")
            ),
            "decision_latency_p95_ms": p95 if success else None,
            "decision_latency_p95_ms_raw": p95,
            "crowd_count": int(scene.num_people),
            "crowd_layout": os.environ.get(
                "WAREHOUSE_CROWD_LAYOUT", "sparse"
            ),
            "dense_profile": os.environ.get(
                "WAREHOUSE_DENSE_PROFILE", "transverse40"
            ),
            "crowd_key": str(scene.key),
            "crowd_spawn": list(map(float, scene.drone_spawn)),
            "crowd_goal": list(map(float, scene.goal_point)),
            "scene_fingerprint_sha256": scene_fingerprint,
            "scene_fingerprint_inputs": scene_fingerprint_payload,
            "details": details,
        }
        self._write_result(result)
        self.controller.hold_after_episode(reason)
        self.controller.quit = True
        carb.log_warn(
            f"[BENCH][RESULT] algorithm={self.algorithm}, seed={self.seed}, "
            f"repeat={self.repeat_index}, success={success}, reason={reason}, "
            f"sim_time={elapsed_sim:.3f}s, path3d={self.path_length_3d:.3f}m, "
            f"intrusion={intrusion_ratio:.6f}, "
            f"intrusion_events={self.intrusion_event_count}, "
            f"output={self.output_path}"
        )

    def _consume_latency_records(self):
        records = self.controller.consume_decision_latency_records()
        self.latency_records.extend(records)
        self.latencies_ms.extend(
            float(record["latency_ms"]) for record in records
        )

    def _scene_fingerprint(self):
        scene = self.app.crowd_scene
        scene_definition = (
            asdict(scene)
            if is_dataclass(scene)
            else {
                "key": getattr(scene, "key", None),
                "seed": getattr(scene, "seed", None),
                "drone_spawn": getattr(scene, "drone_spawn", None),
                "goal_point": getattr(scene, "goal_point", None),
            }
        )
        actual_positions = {
            str(name): list(map(float, position))
            for name, position in sorted(
                getattr(self.app, "person_initial_positions", {}).items()
            )
        }
        payload = _json_safe(
            {
                "scene_definition": scene_definition,
                "actual_person_initial_positions": actual_positions,
                "goal_position": self.goal,
            }
        )
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), payload

    def _write_result(self, result):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(
            f".{self.output_path.name}.tmp.{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(result), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, self.output_path)
