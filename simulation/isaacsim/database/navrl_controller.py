"""Official pretrained NavRL policy driven by Isaac GT Warehouse state."""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

import carb
import numpy as np
from omni.physx import get_physx_scene_query_interface

from app_config import (
    CLASSIC_CRUISE_HEIGHT,
    DATA_RECORD_ENABLED,
    DATA_RECORD_GOAL_RADIUS_M,
    EGO_CLOUD_PERSON_HEIGHT_M,
    EGO_CLOUD_PERSON_RADIUS_M,
    NAVRL_CHECKPOINT,
    NAVRL_DEVICE,
    NAVRL_DIAGNOSTICS_DIR,
    NAVRL_DIAGNOSTICS_ENABLED,
    NAVRL_GOAL_HOLD_RADIUS_M,
    NAVRL_LOG_INTERVAL_SEC,
    NAVRL_SAFETY_AGENT_RADIUS_M,
    NAVRL_SAFETY_DISTANCE_M,
    NAVRL_SAFETY_SHIELD_ENABLED,
    NAVRL_SAFETY_TIME_HORIZON_SEC,
    NAVRL_SAFETY_TIME_STEP_SEC,
    NAVRL_STATE_SOURCE,
)
from classic_controller import ClassicAlgorithmController
from navrl_policy import NavRLPolicy
from navrl_safety_shield import NavRLSafetyShield


class NavRLController(ClassicAlgorithmController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = NavRLPolicy(NAVRL_CHECKPOINT, NAVRL_DEVICE)
        self._mission_goal_reached = False
        self._goal_capture_active = False
        self._fixed_goal = None
        self._fixed_goal_direction_2d = None
        self._last_navrl_log = -1e9
        self._scene_query = get_physx_scene_query_interface()
        self._lidar_query_error_logged = False
        self._drone_prim_prefix = "/World/quadrotor1"
        self.safety_shield = NavRLSafetyShield(
            time_horizon=NAVRL_SAFETY_TIME_HORIZON_SEC,
            time_step=NAVRL_SAFETY_TIME_STEP_SEC,
            safety_distance=NAVRL_SAFETY_DISTANCE_M,
            agent_radius=NAVRL_SAFETY_AGENT_RADIUS_M,
            # Official shield converts the pedestrian XY box to its enclosing
            # circle: sqrt(width^2 + depth^2) / 2.
            obstacle_radius=math.sqrt(2.0) * EGO_CLOUD_PERSON_RADIUS_M,
            # Official ROS navigation_runner sends
            # sqrt(2 * vel_limit^2) to the safe_action service.
            max_speed=math.sqrt(2.0) * NavRLPolicy.ACTION_LIMIT_MPS,
        )
        self._shield_intervention_count = 0
        self._last_shield_active = 0
        self._last_static_guard_active = 0
        self._diag_left_duration = 0.0
        self._diag_wall_stall_duration = 0.0
        self._diag_saved_events = set()
        self._diag_sequence = 0
        self._diag_run_dir = None
        if NAVRL_DIAGNOSTICS_ENABLED:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self._diag_run_dir = Path(NAVRL_DIAGNOSTICS_DIR) / run_name
            self._diag_run_dir.mkdir(parents=True, exist_ok=True)
            carb.log_warn(f"[NAVRL][DIAG] Snapshot directory: {self._diag_run_dir}")
        carb.log_warn(
            f"[NAVRL] Official pretrained checkpoint loaded from {NAVRL_CHECKPOINT}; "
            "perception source=Isaac PhysX raycast GT, "
            f"state source={NAVRL_STATE_SOURCE}, "
            "safety layers (VO shield + static LiDAR guard)="
            f"{'enabled' if NAVRL_SAFETY_SHIELD_ENABLED else 'disabled'}."
        )

    def reset_after_episode(self, reason="reset"):
        super().reset_after_episode(reason)
        self._mission_goal_reached = False
        self._goal_capture_active = False
        self._fixed_goal = None
        self._fixed_goal_direction_2d = None
        self._last_navrl_log = -1e9
        self._shield_intervention_count = 0
        self._last_shield_active = 0
        self._last_static_guard_active = 0
        self._diag_left_duration = 0.0
        self._diag_wall_stall_duration = 0.0
        self._diag_saved_events.clear()

    def _plan_path(self, force=False):
        # NavRL is a reactive goal-conditioned policy and has no global path.
        self.path = []
        self.path_index = 0
        self.last_plan_time = self._now()
        position = self._drone_position(prefer_sim=NAVRL_STATE_SOURCE == "isaac")
        if self._fixed_goal is None and position is not None:
            self._lock_fixed_goal(position)
        if force:
            carb.log_warn("[NAVRL] Takeoff complete; pretrained policy is active.")

    def _lock_fixed_goal(self, position):
        """Select the episode goal once; never move it with the vehicle."""
        position = np.asarray(position, dtype=float)
        # PegasusApp owns the episode target and updates the controller and
        # recorder together whenever the seed changes.  Always lock that
        # shared target here so policy actions, goal features, rewards and
        # terminal labels describe the same task.
        self._fixed_goal = np.asarray(self.target_point, dtype=float).copy()
        direction = self._fixed_goal - position
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-6:
            direction[:] = (1.0, 0.0, 0.0)
        else:
            direction /= norm
        self._fixed_goal_direction_2d = direction
        carb.log_warn(
            f"[NAVRL][MISSION] Fixed episode goal locked: "
            f"start=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}), "
            f"goal=({self._fixed_goal[0]:.2f},{self._fixed_goal[1]:.2f},"
            f"{self._fixed_goal[2]:.2f})."
        )

    def _goal_hold_radius(self):
        if self._benchmark_mode:
            return float(self._benchmark_goal_radius_m)
        radius = float(NAVRL_GOAL_HOLD_RADIUS_M)
        if DATA_RECORD_ENABLED:
            # Do not stop outside the recorder's success sphere. Otherwise
            # the policy can hold forever while the same episode is labelled
            # as a time-limit truncation.
            radius = min(radius, float(DATA_RECORD_GOAL_RADIUS_M))
        return radius

    def _update_navigation(self, now):
        dt = self.control_period if self._last_control_time is None else max(
            1e-3, min(0.25, now - self._last_control_time)
        )
        self._last_control_time = now
        self.last_action_update_time = now
        prefer_sim = NAVRL_STATE_SOURCE == "isaac"
        position = self._drone_position(prefer_sim=prefer_sim)
        velocity = self._drone_velocity(prefer_sim=prefer_sim)
        if position is None or velocity is None:
            self._set_motion(0.0, 0.0, 0.0, 0.0)
            return

        if self._fixed_goal is None:
            self._lock_fixed_goal(position)
        goal = self._fixed_goal
        rpos = goal - position
        distance = float(np.linalg.norm(rpos))
        horizontal_distance = float(np.linalg.norm(rpos[:2]))
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        # Match the official ROS2 navigation runner: once the 3-D distance is
        # within its configured capture radius, zero the commanded velocity
        # immediately. There is no speed-settling prerequisite.
        goal_radius = self._goal_hold_radius()
        if not self._mission_goal_reached and distance <= goal_radius:
            self._mission_goal_reached = True
            self._last_safe_velocity = np.zeros(2, dtype=float)
            carb.log_warn(
                f"[NAVRL][MISSION] Goal reached: distance={distance:.2f}m, "
                f"speed={horizontal_speed:.2f}m/s; holding position."
            )

        if self._mission_goal_reached:
            self._set_motion(0.0, 0.0, self._height_velocity(prefer_sim=prefer_sim), 0.0)
            return

        goal_direction = rpos / max(distance, 1e-6)
        # The policy was trained in an episode-fixed start-to-goal frame.
        # Keep that frame fixed even when obstacle avoidance moves the vehicle
        # laterally or temporarily past the goal.
        goal_direction_2d = self._fixed_goal_direction_2d
        state8 = self._state_input(goal_direction, horizontal_distance, rpos[2], velocity,
                                   goal_direction_2d)
        lidar = self._gt_lidar(position, goal_direction_2d)
        people = self._pedestrian_agents(dt)
        dynamic = self._dynamic_input(position, people, goal_direction_2d)
        observation_ready_wall = time.perf_counter()
        if self._has_obstacle_in_policy_sector(lidar, dynamic):
            command_world = self.policy.infer(
                state8, lidar, dynamic, goal_direction_2d
            )
            action_source = "policy"
        else:
            # This is the upstream navigation_runner behavior: in open space
            # fly directly to the goal and reserve the learned policy for
            # obstacle avoidance. Always running the policy caused a persistent
            # lateral bias that could carry the vehicle across the warehouse.
            command_world = (
                rpos / max(distance, 1e-6) * NavRLPolicy.ACTION_LIMIT_MPS
            )
            self.policy.last_local_velocity = np.zeros(3, dtype=float)
            action_source = "direct_goal"

        # Match the official ROS navigation runner: the policy/direct-goal
        # velocity is passed straight to get_safe_action.  Do not reuse the
        # Classic controller's acceleration limiter or EMA here; that added
        # latency is not part of NavRL and can preserve an obsolete lateral
        # command after the policy has already changed direction.
        command_xy = np.asarray(command_world[:2], dtype=float).copy()

        self._last_shield_active = 0
        self._last_static_guard_active = 0
        if NAVRL_SAFETY_SHIELD_ENABLED:
            preferred_xy = command_xy.copy()
            command_xy, self._last_shield_active = self.safety_shield.apply(
                position[:2], preferred_xy, self._shield_obstacles(position, people)
            )
            if not np.allclose(command_xy, preferred_xy, atol=1e-5):
                self._shield_intervention_count += 1

            guarded_xy, self._last_static_guard_active = self._apply_static_lidar_guard(
                command_xy, lidar, goal_direction_2d
            )
            if not np.allclose(guarded_xy, command_xy, atol=1e-5):
                self._shield_intervention_count += 1
            command_xy = guarded_xy

        # Keep the base-controller state coherent for reset/hold behavior, but
        # NavRL does not smooth against this value on the next policy update.
        self._last_safe_velocity = np.asarray(command_xy, dtype=float).copy()

        self._update_policy_diagnostics(
            dt=dt,
            action_source=action_source,
            state8=state8,
            lidar=lidar,
            dynamic=dynamic,
            goal_direction=goal_direction_2d,
            position=position,
            goal=goal,
            goal_distance=horizontal_distance,
            policy_world=command_world,
            final_world_xy=command_xy,
        )

        # Keep the established altitude loop for the first planar Warehouse test.
        command_world[2] = self._height_velocity(prefer_sim=prefer_sim)
        vx_body, vy_body = self._world_velocity_to_body(command_xy)
        yaw_rate = self._yaw_rate_for_goal(command_xy)
        self._set_motion(vx_body, vy_body, float(command_world[2]), yaw_rate)
        self._record_decision_latency(
            time.perf_counter() - observation_ready_wall,
            source=action_source,
        )
        if now - self._last_navrl_log >= max(0.1, NAVRL_LOG_INTERVAL_SEC):
            self._last_navrl_log = now
            carb.log_warn(
                f"[NAVRL] pos=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) "
                f"goal_dist={horizontal_distance:.2f}m v_world=({command_xy[0]:.2f},"
                f"{command_xy[1]:.2f}) people_in={int(np.count_nonzero(np.linalg.norm(dynamic[:, :3], axis=1)))} "
                f"lidar_nearest={NavRLPolicy.LIDAR_RANGE_M - float(lidar.max()):.2f}m "
                f"action_source={action_source} "
                f"local_action=({self.policy.last_local_velocity[0]:.2f},"
                f"{self.policy.last_local_velocity[1]:.2f}) "
                f"shield_active={self._last_shield_active} "
                f"static_guard={self._last_static_guard_active} "
                f"shield_total={self._shield_intervention_count}"
            )

    def decision_latency_metadata(self):
        return {
            "definition": (
                "synchronous_observation_ready_to_final_velocity_output_"
                "wall_clock"
            ),
            "causal_pairing": True,
        }

    def _update_policy_diagnostics(
        self, *, dt, action_source, state8, lidar, dynamic, goal_direction,
        position, goal, goal_distance, policy_world, final_world_xy,
    ):
        """Capture representative policy inputs without changing its action."""
        if not NAVRL_DIAGNOSTICS_ENABLED or self._diag_run_dir is None:
            return

        local_xy = np.asarray(self.policy.last_local_velocity[:2], dtype=float)
        lidar_nearest = NavRLPolicy.LIDAR_RANGE_M - float(np.max(lidar))
        policy_active = action_source == "policy"

        if policy_active and local_xy[1] >= 0.25:
            self._diag_left_duration += float(dt)
        else:
            self._diag_left_duration = 0.0

        if (
            policy_active
            and goal_distance > 1.0
            and lidar_nearest <= 1.60
            and float(np.linalg.norm(local_xy)) <= 0.15
        ):
            self._diag_wall_stall_duration += float(dt)
        else:
            self._diag_wall_stall_duration = 0.0

        if self._diag_left_duration >= 1.0:
            self._save_policy_snapshot(
                "sustained_left_bias", state8, lidar, dynamic, goal_direction,
                position, goal, goal_distance, policy_world, final_world_xy,
                lidar_nearest,
            )
        if self._diag_wall_stall_duration >= 1.0:
            self._save_policy_snapshot(
                "near_wall_stall", state8, lidar, dynamic, goal_direction,
                position, goal, goal_distance, policy_world, final_world_xy,
                lidar_nearest,
            )

    def _save_policy_snapshot(
        self, label, state8, lidar, dynamic, goal_direction, position, goal,
        goal_distance, policy_world, final_world_xy, lidar_nearest,
    ):
        if label in self._diag_saved_events:
            return
        self._diag_saved_events.add(label)
        self._diag_sequence += 1
        path = self._diag_run_dir / f"{self._diag_sequence:02d}_{label}.npz"
        np.savez_compressed(
            path,
            label=np.asarray(label),
            state8=np.asarray(state8, dtype=np.float32),
            lidar36x4=np.asarray(lidar, dtype=np.float32),
            dynamic5x10=np.asarray(dynamic, dtype=np.float32),
            goal_direction_world=np.asarray(goal_direction, dtype=np.float32),
            position_world=np.asarray(position, dtype=np.float32),
            goal_world=np.asarray(goal, dtype=np.float32),
            goal_distance=np.asarray(goal_distance, dtype=np.float32),
            lidar_nearest=np.asarray(lidar_nearest, dtype=np.float32),
            beta_alpha=np.asarray(self.policy.last_alpha, dtype=np.float32),
            beta_beta=np.asarray(self.policy.last_beta, dtype=np.float32),
            normalized_action=np.asarray(
                self.policy.last_normalized_action, dtype=np.float32
            ),
            local_action=np.asarray(
                self.policy.last_local_velocity, dtype=np.float32
            ),
            policy_world_action=np.asarray(policy_world, dtype=np.float32),
            final_world_xy=np.asarray(final_world_xy, dtype=np.float32),
            people_in=np.asarray(
                np.count_nonzero(np.linalg.norm(dynamic[:, :3], axis=1)),
                dtype=np.int32,
            ),
        )
        carb.log_warn(
            f"[NAVRL][DIAG] Captured {label}: {path}; "
            f"local=({self.policy.last_local_velocity[0]:.3f},"
            f"{self.policy.last_local_velocity[1]:.3f}), "
            f"goal_dist={goal_distance:.2f}m, lidar_nearest={lidar_nearest:.2f}m"
        )

    @staticmethod
    def _has_obstacle_in_policy_sector(lidar, dynamic):
        """Match upstream NavRL's policy/direct-goal routing decision."""
        quarter = lidar.shape[0] // 4
        # lidar stores max_range - hit_distance. Upstream ignores the lowest
        # vertical beam and treats >=0.2 m response as a nearby static object.
        has_static = bool(
            np.any(lidar[:quarter, 1:] >= 0.2)
            or np.any(lidar[-quarter:, 1:] >= 0.2)
        )
        has_dynamic = bool(np.any(dynamic != 0.0))
        return has_static or has_dynamic

    @staticmethod
    def _apply_static_lidar_guard(preferred_xy, lidar, goal_direction):
        """Project velocity away from nearby horizontal LiDAR returns.

        This is the dependency-light counterpart of the upstream safe-action
        service's static laser-point constraints.  It does not plan a path; it
        only prevents a policy command from continuing into a visible wall.
        """
        safe = np.asarray(preferred_xy, dtype=float).copy()
        # Use the closest of all four vertical beams. Restricting the guard to
        # the nominally horizontal beam missed Warehouse wall sections that
        # were hit by a neighboring pitched ray.
        ranges = np.min(
            NavRLPolicy.LIDAR_RANGE_M - np.asarray(lidar, dtype=float),
            axis=1,
        )
        yaw0 = math.atan2(goal_direction[1], goal_direction[0])
        clearance = NAVRL_SAFETY_AGENT_RADIUS_M + NAVRL_SAFETY_DISTANCE_M
        activation_distance = clearance + 0.9
        active = 0

        constraints = []
        for index, hit_distance in enumerate(ranges):
            if not np.isfinite(hit_distance) or hit_distance >= activation_distance:
                continue
            yaw = yaw0 + math.radians(10.0 * index)
            toward_obstacle = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
            max_approach = max(
                0.0,
                (float(hit_distance) - clearance)
                / max(NAVRL_SAFETY_TIME_HORIZON_SEC, 1e-3),
            )
            constraints.append((float(hit_distance), toward_obstacle, max_approach))

        # Closest surfaces take precedence. Repeated projection handles corners
        # where two wall planes constrain the command simultaneously.
        constraints.sort(key=lambda item: item[0])
        for _ in range(2):
            for _, direction, max_approach in constraints:
                approach = float(np.dot(safe, direction))
                if approach > max_approach:
                    safe -= (approach - max_approach) * direction
                    active += 1
        return safe, active

    @staticmethod
    def _goal_frame(vector, goal_direction):
        x_axis = goal_direction
        y_axis = np.array([-x_axis[1], x_axis[0], 0.0])
        return np.array([np.dot(vector, x_axis), np.dot(vector, y_axis), vector[2]], dtype=float)

    def _state_input(self, unit_rpos, distance_2d, distance_z, velocity, goal_direction):
        return np.concatenate((
            self._goal_frame(unit_rpos, goal_direction),
            [distance_2d, distance_z], self._goal_frame(velocity, goal_direction),
        )).astype(np.float32)

    def _dynamic_input(self, position, agents, goal_direction):
        candidates = []
        for agent in agents:
            rel = np.array([agent["position"][0] - position[0],
                            agent["position"][1] - position[1], 0.0], dtype=float)
            distance_2d = float(np.linalg.norm(rel[:2]))
            if distance_2d > NavRLPolicy.LIDAR_RANGE_M:
                continue
            rel_goal = self._goal_frame(rel, goal_direction)
            rel_norm = rel_goal / max(float(np.linalg.norm(rel)), 1e-6)
            vel3 = np.array([agent["velocity"][0], agent["velocity"][1], 0.0])
            vel_goal = self._goal_frame(vel3, goal_direction)
            # Match ros2/navigation_runner/scripts/navigation.py: expand the
            # detected obstacle width by the robot diameter, then quantize it
            # into the four 0.25 m bins. Tall/non-overflyable obstacles use 0.
            effective_width = (
                2.0 * EGO_CLOUD_PERSON_RADIUS_M
                + 2.0 * NAVRL_SAFETY_AGENT_RADIUS_M
            )
            width_category = float(np.clip(
                math.ceil(effective_width / 0.25) - 1,
                0,
                3,
            ))
            height = 0.0 if EGO_CLOUD_PERSON_HEIGHT_M > 1.0 else EGO_CLOUD_PERSON_HEIGHT_M
            feature = np.concatenate((rel_norm, [distance_2d, 0.0], vel_goal,
                                      [width_category, height]))
            candidates.append((distance_2d, feature))
        candidates.sort(key=lambda item: item[0])
        output = np.zeros((NavRLPolicy.MAX_DYNAMIC_OBSTACLES, 10), dtype=np.float32)
        for index, (_, feature) in enumerate(candidates[:NavRLPolicy.MAX_DYNAMIC_OBSTACLES]):
            output[index] = feature
        return output

    @staticmethod
    def _shield_obstacles(position, agents):
        candidates = []
        for agent in agents:
            distance = float(np.linalg.norm(
                np.asarray(agent["position"], dtype=float)[:2]
                - np.asarray(position, dtype=float)[:2]
            ))
            if distance <= NavRLPolicy.LIDAR_RANGE_M:
                candidates.append((distance, agent))
        candidates.sort(key=lambda item: item[0])
        return [agent for _, agent in candidates[:NavRLPolicy.MAX_DYNAMIC_OBSTACLES]]

    def _gt_lidar(self, origin, goal_direction):
        ranges = np.full((NavRLPolicy.HORIZONTAL_BEAMS, 4), NavRLPolicy.LIDAR_RANGE_M,
                         dtype=np.float32)
        yaw0 = math.atan2(goal_direction[1], goal_direction[0])
        for hi in range(NavRLPolicy.HORIZONTAL_BEAMS):
            yaw = yaw0 + math.radians(10.0 * hi)
            for vi, pitch_deg in enumerate(NavRLPolicy.VERTICAL_ANGLES_DEG):
                pitch = math.radians(pitch_deg)
                direction = np.array([math.cos(pitch) * math.cos(yaw),
                                      math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
                ranges[hi, vi] = self._physx_raycast_distance(origin, direction)
        # Official observation stores lidar_range - hit_distance.
        return NavRLPolicy.LIDAR_RANGE_M - ranges

    def _physx_raycast_distance(self, origin, direction):
        """Return the closest real collider hit while excluding the drone itself."""
        best = NavRLPolicy.LIDAR_RANGE_M

        def report_hit(hit):
            nonlocal best
            rigid_body = str(hit.rigid_body or "")
            collision = str(hit.collision or "")
            if self._is_drone_path(rigid_body) or self._is_drone_path(collision):
                return True
            distance = float(hit.distance)
            if np.isfinite(distance) and 0.0 <= distance < best:
                best = distance
            return True

        try:
            self._scene_query.raycast_all(
                tuple(float(value) for value in origin),
                tuple(float(value) for value in direction),
                NavRLPolicy.LIDAR_RANGE_M,
                report_hit,
            )
        except Exception as exc:
            if not self._lidar_query_error_logged:
                self._lidar_query_error_logged = True
                carb.log_error(f"[NAVRL][LIDAR] PhysX raycast failed: {exc}")
        return best

    def _is_drone_path(self, path):
        return path == self._drone_prim_prefix or path.startswith(
            f"{self._drone_prim_prefix}/"
        )
