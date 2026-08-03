#!/usr/bin/env python3
"""Official DPMPC setpoint adapter for the existing Pegasus/PX4 stack."""

import time

import carb
import numpy as np
from scipy.spatial.transform import Rotation

from app_config import (
    CLASSIC_CRUISE_HEIGHT,
    DPMPC_COMMAND_TIMEOUT_SEC,
    DPMPC_MAX_SPEED_XY,
    DPMPC_MAX_SPEED_Z,
    DPMPC_OBSERVATION_HZ,
    DPMPC_POSITION_KP_XY,
    DPMPC_POSITION_KP_Z,
    DPMPC_STATIC_MAP_RETRY_SEC,
    DPMPC_UDP_HOST,
    DPMPC_UDP_ISAAC_PORT,
    DPMPC_UDP_PLANNER_PORT,
    DPMPC_YAW_KP,
    EGO_MAX_YAW_RATE,
)
from classic_controller import ClassicAlgorithmController
from dpmpc_bridge import DpmpcUdpBridge


class DpmpcController(ClassicAlgorithmController):
    """Tracks the position setpoint selected by official ``forward_idx=10``.

    The upstream MAVROS demo asks PX4's position controller to track that
    setpoint and explicitly ignores velocity.  This adapter performs only the
    equivalent position-to-velocity conversion required by the existing
    MAVSDK velocity-offboard transport.  It adds no navigation heuristic,
    smoothing, safety guard, or obstacle rule.
    """

    def __init__(
        self,
        *args,
        static_point_cloud_provider=None,
        obstacle_state_provider=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 < float(DPMPC_OBSERVATION_HZ) <= 25.0:
            raise ValueError("DPMPC_OBSERVATION_HZ must be in (0, 25]")
        self.static_point_cloud_provider = static_point_cloud_provider
        self.obstacle_state_provider = obstacle_state_provider
        self.dpmpc_bridge = DpmpcUdpBridge(
            DPMPC_UDP_HOST,
            DPMPC_UDP_PLANNER_PORT,
            DPMPC_UDP_ISAAC_PORT,
        )
        self._last_observation_time = -1e9
        self._last_map_send_time = -1e9
        self._map_acknowledged = False
        self._goal_sent = False
        self._last_command_observation_id = None
        self._observation_wall_times = {}
        self._mission_goal_reached = False
        self._last_missing_log_wall = 0.0
        carb.log_warn(
            "[DPMPC] Official planner bridge ready: "
            f"planner UDP={DPMPC_UDP_HOST}:{DPMPC_UDP_PLANNER_PORT}, "
            f"command RX=0.0.0.0:{DPMPC_UDP_ISAAC_PORT}."
        )

    def shutdown(self):
        try:
            self.dpmpc_bridge.close()
        finally:
            super().shutdown()

    def reset_after_episode(self, reason="reset"):
        super().reset_after_episode(reason)
        self.dpmpc_bridge.reset_after_episode(reason)
        self._last_observation_time = -1e9
        self._last_map_send_time = -1e9
        self._map_acknowledged = False
        self._goal_sent = False
        self._last_command_observation_id = None
        self._observation_wall_times.clear()
        self._mission_goal_reached = False

    def _begin_takeoff(self):
        # Match the proven EGO/PX4 start sequence without resetting PX4 after
        # the external planner has already initialized.
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
        self.state = "takeoff"
        self.path = []
        self.path_index = 0
        self.last_plan_time = 0.0
        carb.log_warn(
            "[DPMPC] Takeoff requested; waiting for PX4 offboard, "
            "z>=0.90m and stabilization."
        )

    def _plan_path(self, force=False):
        # Static minimum-snap planning is performed by the official sidecar.
        self.path = []
        self.path_index = 0
        self.last_plan_time = self._now()
        if force:
            carb.log_warn(
                "[DPMPC] Takeoff complete; sending the fixed goal and static map."
            )

    def _update_navigation(self, now):
        self.last_action_update_time = now
        position = self._drone_position()
        velocity = self._drone_velocity()
        attitude = self._drone_attitude()
        if position is None or velocity is None or attitude is None:
            self._set_motion(0.0, 0.0, 0.0, 0.0)
            return

        goal_distance = float(np.linalg.norm(position - self.target_point))
        if self._mission_goal_reached or (
            self._benchmark_mode
            and goal_distance <= float(self._benchmark_goal_radius_m)
        ):
            if not self._mission_goal_reached:
                self._mission_goal_reached = True
                carb.log_warn(
                    f"[DPMPC] Mission goal reached: distance={goal_distance:.2f}m; "
                    "latching hover."
                )
            vz = float(np.clip(
                DPMPC_POSITION_KP_Z
                * (float(self.target_point[2]) - float(position[2])),
                -DPMPC_MAX_SPEED_Z,
                DPMPC_MAX_SPEED_Z,
            ))
            self._set_motion(0.0, 0.0, vz, 0.0)
            return

        if not self._goal_sent:
            self.dpmpc_bridge.send_goal(now, self.target_point)
            self._goal_sent = True

        if (
            not self._map_acknowledged
            and now - self._last_map_send_time >= DPMPC_STATIC_MAP_RETRY_SEC
            and self.static_point_cloud_provider is not None
        ):
            # Goal and chunked map use UDP. Re-send both until the first
            # causally paired command acknowledges complete initialization.
            self.dpmpc_bridge.send_goal(now, self.target_point)
            self._goal_sent = True
            cloud = self.static_point_cloud_provider()
            if cloud is not None:
                self.dpmpc_bridge.send_static_cloud(now, cloud)
                self._last_map_send_time = now

        if (
            now - self._last_observation_time
            >= 1.0 / max(DPMPC_OBSERVATION_HZ, 1e-3)
        ):
            obstacles = (
                []
                if self.obstacle_state_provider is None
                else self.obstacle_state_provider()
            )
            observation_wall = time.perf_counter()
            observation_id = self.dpmpc_bridge.send_observation(
                now, position, velocity, attitude, obstacles
            )
            if self._benchmark_mode:
                self._observation_wall_times[observation_id] = observation_wall
                # At 10 Hz this retains 6.4 seconds, well above command timeout.
                while len(self._observation_wall_times) > 64:
                    oldest = min(self._observation_wall_times)
                    self._observation_wall_times.pop(oldest, None)
            self._last_observation_time = now

        command = self.dpmpc_bridge.poll_command()
        if command is None or not self.dpmpc_bridge.command_is_fresh(
            DPMPC_COMMAND_TIMEOUT_SEC, simulation_now=now
        ):
            self._hold_without_planner(position)
            return

        self._map_acknowledged = True
        desired = np.zeros(3, dtype=float)
        error = np.asarray(command["position"], dtype=float) - position
        desired[:2] = DPMPC_POSITION_KP_XY * error[:2]
        desired[2] = DPMPC_POSITION_KP_Z * error[2]
        speed_xy = float(np.linalg.norm(desired[:2]))
        if speed_xy > DPMPC_MAX_SPEED_XY:
            desired[:2] *= DPMPC_MAX_SPEED_XY / speed_xy
        desired[2] = float(np.clip(
            desired[2], -DPMPC_MAX_SPEED_Z, DPMPC_MAX_SPEED_Z
        ))

        vx_body, vy_body = self._world_velocity_to_body(desired[:2])
        yaw = Rotation.from_quat(attitude).as_euler("XYZ", degrees=False)[2]
        yaw_rate = DPMPC_YAW_KP * self._wrap_pi(command["yaw"] - yaw)
        yaw_rate = float(np.clip(
            yaw_rate, -EGO_MAX_YAW_RATE, EGO_MAX_YAW_RATE
        ))
        self._set_motion(vx_body, vy_body, desired[2], yaw_rate)
        self._record_command_latency(command)

    def _record_command_latency(self, command):
        if not self._benchmark_mode:
            return
        observation_id = int(command.get("observation_id", -1))
        if observation_id < 0 or observation_id == self._last_command_observation_id:
            return
        self._last_command_observation_id = observation_id
        origin = self._observation_wall_times.pop(observation_id, None)
        if origin is not None:
            self._record_decision_latency(
                time.perf_counter() - origin,
                source="official_dpmpc_solve_and_setpoint_adapter",
            )

    def _hold_without_planner(self, position):
        vz = float(np.clip(
            DPMPC_POSITION_KP_Z
            * (CLASSIC_CRUISE_HEIGHT - float(position[2])),
            -DPMPC_MAX_SPEED_Z,
            DPMPC_MAX_SPEED_Z,
        ))
        self._set_motion(0.0, 0.0, vz, 0.0)
        wall_now = time.monotonic()
        if wall_now - self._last_missing_log_wall >= 2.0:
            self._last_missing_log_wall = wall_now
            carb.log_warn("[DPMPC] No fresh planner setpoint; holding current XY.")

    def decision_latency_metadata(self):
        return {
            "definition": (
                "atomic_state_and_obstacles_ready_to_final_px4_velocity_output_"
                "wall_clock"
            ),
            "causal_pairing": True,
            "note": (
                "observation_id is propagated through the official DPMPC solve; "
                "the interval includes MPC and the required position-setpoint "
                "to MAVSDK velocity-offboard adapter."
            ),
        }
