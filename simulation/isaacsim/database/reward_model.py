"""Versioned, simulator-independent crowd-navigation reward computation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


REWARD_COMPONENT_KEYS = ("event", "progress", "human_clearance", "smoothness", "height", "time")
REWARD_SCHEMA = "ego_human_reward_v2_agl_ttc_3d"


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def clearance_risk(clearance: float, hard: float, safe: float) -> float:
    if clearance >= safe:
        return 0.0
    if clearance <= hard:
        return 1.0
    return float(((safe - clearance) / max(safe - hard, 1e-6)) ** 2)


class CrowdRewardCalculator:
    """Stateful transition reward shared by online and offline pipelines."""

    def __init__(self, config: Mapping[str, Any], target_point: Sequence[float],
                 goal_region: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(config)
        self.target = np.asarray(target_point, dtype=np.float64).reshape(3)
        self.goal_region = goal_region
        self.previous: dict[str, Any] | None = None

    def reset(self) -> None:
        self.previous = None

    def goal_distance(self, position: np.ndarray) -> float:
        if self.goal_region is None:
            return float(np.linalg.norm(position - self.target))
        x_min, x_max = self.goal_region["x_range"]
        dx = max(float(x_min) - position[0], 0.0, position[0] - float(x_max))
        dy = max(float(self.goal_region["y_min"]) - position[1], 0.0)
        tolerance = float(self.cfg.get("goal_height_tolerance_m", 0.20))
        dz = max(abs(float(position[2] - self.target[2])) - tolerance, 0.0)
        return float(np.linalg.norm((dx, dy, dz)))

    @staticmethod
    def _pelvis_map(records: Sequence[Mapping[str, Any]] | None) -> dict[str, np.ndarray]:
        output = {}
        for item in records or ():
            value = item.get("position_world_m")
            if value is None:
                continue
            position = np.asarray(value, dtype=np.float64)
            if position.shape == (3,) and np.isfinite(position).all():
                output[str(item.get("pedestrian_id"))] = position
        return output

    def _human_risk(self, position, velocity, yaw, dt, joint_distances, pelvis_records):
        cfg = self.cfg
        hard, safe = float(cfg["human_hard_clearance_m"]), float(cfg["human_safe_clearance_m"])
        drone_radius = float(cfg["drone_collision_radius_m"])
        joint_radii = cfg.get("joint_collision_radii", {})
        instant_risk, min_center, min_clearance = 0.0, None, None
        for item in joint_distances or ():
            try:
                distance = float(item["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(distance):
                continue
            radius = float(joint_radii.get(str(item.get("joint_name", "")), 0.08))
            clearance = distance - drone_radius - radius
            instant_risk = max(instant_risk, clearance_risk(clearance, hard, safe))
            min_center = distance if min_center is None else min(min_center, distance)
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)

        predictive_risk, min_ttc, corridor_risk = 0.0, None, 0.0
        current_pelvis = self._pelvis_map(pelvis_records)
        previous_pelvis = {} if self.previous is None else self.previous.get("pelvis_world", {})
        horizon = float(cfg.get("human_ttc_horizon_sec", 2.5))
        predictive_safe = float(cfg.get("human_predictive_safe_clearance_m", safe))
        corridor_half_width = float(cfg.get("flight_corridor_half_width_m", 0.75))
        corridor_lookahead = float(cfg.get("flight_corridor_lookahead_m", 4.0))
        c, s = math.cos(yaw), math.sin(yaw)
        for track_id, pelvis in current_pelvis.items():
            rel = pelvis - position
            rel_body = np.asarray((c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]))
            if 0.0 < rel_body[0] < corridor_lookahead and abs(rel_body[1]) < corridor_half_width:
                lateral = abs(rel_body[1]) / max(corridor_half_width, 1e-6)
                forward = rel_body[0] / max(corridor_lookahead, 1e-6)
                corridor_risk = max(corridor_risk, (1.0 - lateral) ** 2 * (1.0 - 0.5 * forward))
            if dt <= 1e-6 or track_id not in previous_pelvis:
                continue
            human_velocity = (pelvis - previous_pelvis[track_id]) / dt
            relative_velocity = human_velocity - velocity
            speed_sq = float(np.dot(relative_velocity, relative_velocity))
            if speed_sq <= 1e-8:
                continue
            ttc = float(np.clip(-np.dot(rel, relative_velocity) / speed_sq, 0.0, horizon))
            closest = rel + relative_velocity * ttc
            closest_clearance = float(np.linalg.norm(closest)) - drone_radius - 0.14
            if 0.0 < ttc < horizon and closest_clearance < predictive_safe:
                spatial = clearance_risk(closest_clearance, hard, predictive_safe)
                temporal = (1.0 - ttc / horizon) ** 2
                predictive_risk = max(predictive_risk, spatial * temporal)
                min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
        combined = max(instant_risk, predictive_risk, corridor_risk)
        return combined, instant_risk, predictive_risk, corridor_risk, min_center, min_clearance, min_ttc, current_pelvis

    def compute(self, timestamp: float, drone_state: Mapping[str, Any],
                joint_distances=None, pelvis_records=None,
                *, collision=False, reached_goal=False,
                termination_reason="recording"):
        cfg, previous = self.cfg, self.previous
        dt = 0.0 if previous is None else max(0.0, float(timestamp) - previous["timestamp"])
        position = np.asarray(drone_state["position"], dtype=np.float64)
        velocity = np.asarray(drone_state.get("velocity", (0, 0, 0)), dtype=np.float64)
        acceleration = np.asarray(drone_state.get("acceleration", (0, 0, 0)), dtype=np.float64)
        yaw = float(drone_state.get("yaw", drone_state.get("roll_pitch_yaw_rad", (0, 0, 0))[2]))
        goal_distance = self.goal_distance(position)
        raw_progress = 0.0 if previous is None else previous["goal_distance"] - goal_distance
        max_progress = float(cfg["max_progress_speed_mps"]) * dt
        used_progress = float(np.clip(raw_progress, -max_progress, max_progress)) if dt else 0.0
        progress_reward = float(cfg["progress_weight_per_m"]) * used_progress

        (human_risk, instant_risk, predictive_risk, corridor_risk, min_center,
         min_clearance, min_ttc, pelvis_world) = self._human_risk(
            position, velocity, yaw, dt, joint_distances, pelvis_records)
        human_reward = -float(cfg["human_clearance_weight_per_sec"]) * dt * human_risk

        acc_xy = acceleration[:2]
        alpha = float(np.clip(cfg["acceleration_filter_alpha"], 0.0, 1.0))
        if previous is None:
            filtered_acc = acc_xy
            jerk, yaw_rate = np.zeros(2), 0.0
        else:
            filtered_acc = alpha * acc_xy + (1.0 - alpha) * previous["filtered_acceleration_xy"]
            jerk = (filtered_acc - previous["filtered_acceleration_xy"]) / dt if dt > 1e-9 else np.zeros(2)
            yaw_rate = wrap_pi(yaw - previous["yaw"]) / dt if dt > 1e-9 else 0.0
        clip = float(cfg["smooth_term_clip"])
        acc_term = min((np.linalg.norm(filtered_acc) / float(cfg["acceleration_scale_mps2"])) ** 2, clip)
        jerk_term = min((np.linalg.norm(jerk) / float(cfg["jerk_scale_mps3"])) ** 2, clip)
        yaw_term = min((abs(yaw_rate) / float(cfg["yaw_rate_scale_rps"])) ** 2, clip)
        smooth = -dt * (float(cfg["acceleration_weight_per_sec"]) * acc_term
                        + float(cfg["jerk_weight_per_sec"]) * jerk_term
                        + float(cfg["yaw_rate_weight_per_sec"]) * yaw_term)

        altitude = drone_state.get("altitude_agl")
        altitude_valid = altitude is not None and np.isfinite(float(altitude))
        if altitude_valid:
            height_error = abs(float(altitude) - float(cfg["cruise_height_m"]))
            excess = max(0.0, height_error - float(cfg["height_tolerance_m"]))
            height_term = min((excess / float(cfg["height_scale_m"])) ** 2, clip)
            height_reward = -float(cfg["height_weight_per_sec"]) * dt * height_term
        else:
            height_error, height_reward = None, 0.0
        time_reward = -float(cfg["time_cost_per_sec"]) * dt
        reason = "collision" if collision else "reached_goal" if reached_goal else str(termination_reason)
        event = float(cfg["event_rewards"].get(reason, 0.0))
        components = {"event": event, "progress": progress_reward,
                      "human_clearance": human_reward, "smoothness": float(smooth),
                      "height": float(height_reward), "time": float(time_reward)}
        diagnostics = {"reward_schema": REWARD_SCHEMA, "dt_sec": dt,
            "goal_distance_3d_m": goal_distance, "goal_progress_raw_m": raw_progress,
            "goal_progress_used_m": used_progress, "human_min_joint_center_distance_m": min_center,
            "human_min_clearance_m": min_clearance, "human_risk": human_risk,
            "human_instant_risk": instant_risk, "human_predictive_risk": predictive_risk,
            "human_corridor_risk": corridor_risk, "human_min_ttc_sec": min_ttc,
            "filtered_horizontal_acceleration_mps2": float(np.linalg.norm(filtered_acc)),
            "filtered_horizontal_jerk_mps3": float(np.linalg.norm(jerk)),
            "yaw_rate_rps": yaw_rate, "altitude_agl_m": float(altitude) if altitude_valid else None,
            "altitude_valid": bool(altitude_valid), "height_error_m": height_error,
            "event_reason": reason}
        self.previous = {"timestamp": float(timestamp), "goal_distance": goal_distance,
                         "filtered_acceleration_xy": filtered_acc, "yaw": yaw,
                         "pelvis_world": pelvis_world}
        return float(sum(components.values())), components, diagnostics
