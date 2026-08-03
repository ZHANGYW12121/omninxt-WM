#!/usr/bin/env python3
"""EGO-Planner PositionCommand tracker for the existing Pegasus/PX4 stack."""

import math
import time
from collections import deque

import carb
import numpy as np
from scipy.spatial.transform import Rotation

from app_config import (
    CLASSIC_CRUISE_HEIGHT,
    EGO_CLOUD_HZ,
    EGO_COMMAND_TIMEOUT_SEC,
    EGO_GOAL_HZ,
    EGO_GOAL_HOLD_MAX_SPEED_MPS,
    EGO_GOAL_HOLD_RADIUS_M,
    EGO_GOAL_MIN_APPROACH_SPEED_MPS,
    EGO_GOAL_SLOW_RADIUS_M,
    EGO_MAX_ACCEL,
    EGO_MAX_SPEED_XY,
    EGO_MAX_SPEED_Z,
    EGO_MAX_YAW_RATE,
    EGO_ODOM_HZ,
    EGO_POSITION_KP_XY,
    EGO_POSITION_KP_Z,
    EGO_UDP_HOST,
    EGO_UDP_ISAAC_PORT,
    EGO_UDP_ROS_PORT,
    EGO_YAW_KP,
)
from classic_controller import ClassicAlgorithmController
from ego_planner_bridge import EgoPlannerUdpBridge


class EgoPlannerController(ClassicAlgorithmController):
    def __init__(self, *args, point_cloud_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_cloud_provider = point_cloud_provider
        self.ego_bridge = EgoPlannerUdpBridge(
            EGO_UDP_HOST, EGO_UDP_ROS_PORT, EGO_UDP_ISAAC_PORT
        )
        self._last_odom_time = -1e9
        self._last_cloud_time = -1e9
        self._last_goal_time = -1e9
        self._odom_packets = 0
        self._goal_accepted = False
        self._mission_goal_reached = False
        self._mission_complete_sent = False
        self._last_world_velocity = np.zeros(3, dtype=float)
        self._last_missing_log_wall = 0.0
        self._pending_observation_latencies = deque(maxlen=32)
        self._last_latency_trajectory_id = None
        carb.log_warn(
            "[EGO] Single-drone bridge ready: "
            f"ROS UDP={EGO_UDP_HOST}:{EGO_UDP_ROS_PORT}, "
            f"command RX=0.0.0.0:{EGO_UDP_ISAAC_PORT}."
        )

    def shutdown(self):
        try:
            self.ego_bridge.close()
        finally:
            super().shutdown()

    def update(self):
        # Drive ROS /clock from Isaac simulation time even before takeoff. EGO
        # trajectory durations must not advance on wall time when Isaac is
        # running below real time.
        self.ego_bridge.send_clock(self._now())
        super().update()

    def reset_after_episode(self, reason="reset"):
        super().reset_after_episode(reason)
        self.ego_bridge.reset_after_episode()
        self._last_world_velocity[:] = 0.0
        self._odom_packets = 0
        self._goal_accepted = False
        self._mission_goal_reached = False
        self._mission_complete_sent = False
        self._pending_observation_latencies.clear()
        self._last_latency_trajectory_id = None

    def _begin_takeoff(self):
        """Start EGO flight without the classic episode-reset/disarm pulse."""
        # The established px4_classic/OmniDepth path remains untouched.  EGO
        # uses a persistent external planner, so starting its first mission
        # clears only the held command and must not request an episode reset.
        self.cmd.reset()
        if self.command_sink is not None:
            self.command_sink.set_motion(0.0, 0.0, 0.0, 0.0)
        takeoff_position = self._drone_position(
            prefer_sim=self.command_sink is not None
        )
        self.takeoff_start_position = (
            None
            if takeoff_position is None
            else np.asarray(takeoff_position, dtype=float).copy()
        )
        self._trigger_takeoff()
        self.takeoff_start_time = self._now()
        self.last_takeoff_log_time = self.takeoff_start_time
        self._last_control_time = None
        self.last_action_update_time = None
        self._last_safe_velocity = np.zeros(2, dtype=float)
        self.state = "takeoff"
        self.path = []
        self.path_index = 0
        self.last_plan_time = 0.0
        if self.command_sink is None:
            carb.log_warn("[EGO] Local Pegasus takeoff requested; stabilizing.")
        else:
            carb.log_warn(
                "[EGO] Takeoff requested; waiting for PX4 offboard, "
                "z>=0.90m and stabilization."
            )

    def _plan_path(self, force=False):
        # Global/local trajectory generation belongs to EGO-Planner.  Keep this
        # hook because the inherited takeoff state calls it before navigation.
        self.path = []
        self.path_index = 0
        self.last_plan_time = self._now()
        if force:
            carb.log_warn("[EGO] Takeoff complete; waiting for ROS PositionCommand.")

    def _update_navigation(self, now):
        dt = self.control_period if self._last_control_time is None else max(
            1e-3, min(0.25, now - self._last_control_time)
        )
        self._last_control_time = now
        self.last_action_update_time = now

        position = self._drone_position()
        velocity = self._drone_velocity()
        attitude = self._drone_attitude()
        if position is None or velocity is None or attitude is None:
            self._set_motion(0.0, 0.0, 0.0, 0.0)
            return

        # EGO's trajectory server can keep publishing its final sampled
        # PositionCommand after its FSM has entered WAIT_TARGET.  Do not keep
        # applying a residual endpoint velocity: once the actual vehicle has
        # settled inside the mission goal, latch a zero-XY hover command.
        goal_xy = np.asarray(self._effective_goal_xy(position[:2]), dtype=float)
        if self._benchmark_mode:
            goal_distance = float(np.linalg.norm(position - self.target_point))
            goal_reached = goal_distance <= float(self._benchmark_goal_radius_m)
        else:
            goal_distance = float(np.linalg.norm(position[:2] - goal_xy))
            goal_reached = (
                goal_distance <= EGO_GOAL_HOLD_RADIUS_M
                and float(np.linalg.norm(velocity[:2])) <= EGO_GOAL_HOLD_MAX_SPEED_MPS
            )
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if self._mission_goal_reached or goal_reached:
            if not self._mission_goal_reached:
                self._mission_goal_reached = True
                carb.log_warn(
                    f"[EGO] Mission goal reached: distance={goal_distance:.2f}m, "
                    "latching zero-XY hover and ignoring residual trajectory commands."
                )
            if not self._benchmark_mode and not self._mission_complete_sent:
                self.ego_bridge.send_mission_complete(now, position, horizontal_speed)
                self._mission_complete_sent = True
            self._last_world_velocity[:] = 0.0
            vz = float(np.clip(
                EGO_POSITION_KP_Z * (CLASSIC_CRUISE_HEIGHT - float(position[2])),
                -EGO_MAX_SPEED_Z,
                EGO_MAX_SPEED_Z,
            ))
            self._set_motion(0.0, 0.0, vz, 0.0)
            return

        if now - self._last_odom_time >= 1.0 / max(EGO_ODOM_HZ, 1e-3):
            self.ego_bridge.send_odometry(now, position, velocity, attitude)
            self._last_odom_time = now
            self._odom_packets += 1
        if (
            not self._goal_accepted
            and self._odom_packets >= 5
            and now - self._last_goal_time >= 1.0 / max(EGO_GOAL_HZ, 1e-3)
        ):
            goal_z = float(self.target_point[2]) if self._benchmark_mode else CLASSIC_CRUISE_HEIGHT
            self.ego_bridge.send_goal(now, [goal_xy[0], goal_xy[1], goal_z])
            self._last_goal_time = now
        if (
            self.point_cloud_provider is not None
            and now - self._last_cloud_time >= 1.0 / max(EGO_CLOUD_HZ, 1e-3)
        ):
            cloud = self.point_cloud_provider()
            if cloud is not None and len(cloud):
                observation_ready_wall = time.perf_counter()
                self.ego_bridge.send_point_cloud(now, cloud)
                if self._benchmark_mode:
                    self._pending_observation_latencies.append(
                        (float(now), observation_ready_wall)
                    )
            self._last_cloud_time = now

        command = self.ego_bridge.poll_position_command()
        if command is None or not self.ego_bridge.command_is_fresh(
            EGO_COMMAND_TIMEOUT_SEC, simulation_now=now
        ):
            self._hold_without_planner(position)
            return
        latency_origin_wall = (
            self._consume_new_trajectory_latency_origin(command)
            if self._benchmark_mode
            else None
        )
        self._goal_accepted = True

        desired = np.asarray(command["velocity"], dtype=float).copy()
        error = np.asarray(command["position"], dtype=float) - position
        desired[:2] += EGO_POSITION_KP_XY * error[:2]
        desired[2] += EGO_POSITION_KP_Z * error[2]
        horizontal = float(np.linalg.norm(desired[:2]))
        max_speed_xy = float(EGO_MAX_SPEED_XY)
        if goal_distance < EGO_GOAL_SLOW_RADIUS_M:
            approach_scale = max(
                0.0,
                goal_distance / max(EGO_GOAL_SLOW_RADIUS_M, 1e-3),
            )
            max_speed_xy = max(
                float(EGO_GOAL_MIN_APPROACH_SPEED_MPS),
                float(EGO_MAX_SPEED_XY) * approach_scale,
            )
        if horizontal > max_speed_xy:
            desired[:2] *= max_speed_xy / horizontal
        desired[2] = float(np.clip(desired[2], -EGO_MAX_SPEED_Z, EGO_MAX_SPEED_Z))

        delta = desired - self._last_world_velocity
        max_delta = max(1e-4, EGO_MAX_ACCEL * dt)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            desired = self._last_world_velocity + delta * (max_delta / delta_norm)
        self._last_world_velocity = desired

        vx_body, vy_body = self._world_velocity_to_body(desired[:2])
        yaw = Rotation.from_quat(attitude).as_euler("XYZ", degrees=False)[2]
        yaw_rate = command["yaw_dot"] + EGO_YAW_KP * self._wrap_pi(command["yaw"] - yaw)
        yaw_rate = float(np.clip(yaw_rate, -EGO_MAX_YAW_RATE, EGO_MAX_YAW_RATE))
        self._set_motion(vx_body, vy_body, desired[2], yaw_rate)
        if latency_origin_wall is not None:
            self._record_decision_latency(
                time.perf_counter() - latency_origin_wall,
                source="new_trajectory",
            )

    def _consume_new_trajectory_latency_origin(self, command):
        """Pair only the first final command of each newly planned trajectory.

        EGO-Planner is asynchronous and does not propagate a cloud sequence
        through the occupancy map and optimizer.  The trajectory id lets us
        avoid treating every periodic trajectory sample as a new decision.
        The origin is therefore the latest pending observation no newer than
        the first command of the new trajectory.
        """
        try:
            command_stamp = float(command.get("stamp", 0.0))
            trajectory_id = int(command.get("trajectory_id", -1))
        except (TypeError, ValueError):
            return None
        if trajectory_id < 0:
            return None
        if self._last_latency_trajectory_id == trajectory_id:
            return None
        self._last_latency_trajectory_id = trajectory_id
        candidates = [
            item for item in self._pending_observation_latencies
            if item[0] <= command_stamp + 1e-6
        ]
        if not candidates:
            return None
        observation_stamp, observation_wall = candidates[-1]
        while (
            self._pending_observation_latencies
            and self._pending_observation_latencies[0][0] <= observation_stamp
        ):
            self._pending_observation_latencies.popleft()
        return observation_wall

    def decision_latency_metadata(self):
        return {
            "definition": (
                "latest_pending_observation_to_first_final_velocity_output_"
                "of_new_trajectory_wall_clock"
            ),
            "causal_pairing": False,
            "note": (
                "EGO-Planner does not propagate an observation sequence through "
                "its asynchronous map and optimizer; trajectory_id removes "
                "periodic PositionCommand duplicates, but the observation pair "
                "is the latest eligible pending observation."
            ),
        }

    def _hold_without_planner(self, position):
        self._last_world_velocity[:] = 0.0
        vz = float(np.clip(
            EGO_POSITION_KP_Z * (CLASSIC_CRUISE_HEIGHT - float(position[2])),
            -EGO_MAX_SPEED_Z,
            EGO_MAX_SPEED_Z,
        ))
        self._set_motion(0.0, 0.0, vz, 0.0)
        wall = time.monotonic()
        if wall - self._last_missing_log_wall >= 2.0:
            self._last_missing_log_wall = wall
            carb.log_warn("[EGO] No fresh PositionCommand; holding current XY.")
