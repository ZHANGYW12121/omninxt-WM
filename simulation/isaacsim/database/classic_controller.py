#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math
import time

import carb
import carb.input
import numpy as np
import omni.appwindow
from scipy.spatial.transform import Rotation

from app_config import (
    CLASSIC_AUTO_START,
    CLASSIC_AVOIDANCE_MARGIN,
    CLASSIC_BACKTRACK_PENALTY,
    CLASSIC_CRUISE_HEIGHT,
    CLASSIC_DESIRED_VELOCITY_WEIGHT,
    CLASSIC_DRONE_RADIUS,
    CLASSIC_COMMAND_MAX_ACCEL_MPS2,
    CLASSIC_COMMAND_SMOOTHING_ALPHA,
    CLASSIC_FINAL_APPROACH_DISTANCE,
    CLASSIC_FINAL_COMMAND_MAX_ACCEL_MPS2,
    CLASSIC_FINAL_LATERAL_SPEED,
    CLASSIC_FINAL_MAX_SPEED,
    CLASSIC_FINAL_VELOCITY_CHANGE_WEIGHT,
    CLASSIC_FINAL_YAW_RATE_SCALE,
    CLASSIC_GOAL_PROGRESS_WEIGHT,
    CLASSIC_GOAL_REGION_X_GUARD,
    CLASSIC_GROUND_MAX_WAIT_SEC,
    CLASSIC_GROUND_SETTLE_SEC,
    CLASSIC_GROUND_VZ_THRESHOLD,
    CLASSIC_GROUND_Z_THRESHOLD,
    CLASSIC_GRID_RESOLUTION,
    CLASSIC_HARD_AVOIDANCE_MARGIN,
    CLASSIC_LATERAL_DETOUR_WEIGHT,
    CLASSIC_MAX_PERSON_SPEED_ESTIMATE,
    CLASSIC_MAX_SPEED,
    CLASSIC_MAX_Z_SPEED,
    CLASSIC_MIN_SPEED,
    CLASSIC_NAV_LOG_INTERVAL_SEC,
    CLASSIC_ORCA_NEIGHBOR_RADIUS,
    CLASSIC_ORCA_TIME_HORIZON,
    CLASSIC_PATH_LOOKAHEAD_DISTANCE,
    CLASSIC_PATH_REPLAN_INTERVAL_SEC,
    CLASSIC_PEDESTRIAN_RADIUS,
    CLASSIC_PERSON_VEL_FILTER,
    CLASSIC_SLOW_RADIUS,
    CLASSIC_SOFT_AVOIDANCE_MARGIN,
    CLASSIC_SOFT_CLEARANCE_WEIGHT,
    CLASSIC_START_KEY,
    CLASSIC_STATIC_LOOKAHEAD_SEC,
    CLASSIC_STATIC_OBSTACLE_CLEARANCE,
    CLASSIC_TAKEOFF_LOG_INTERVAL_SEC,
    CLASSIC_TAKEOFF_MAX_WAIT_SEC,
    CLASSIC_TAKEOFF_RECORD_MIN_Z,
    CLASSIC_TAKEOFF_SETTLE_SEC,
    CLASSIC_WAIT_FOR_GROUND_BEFORE_TAKEOFF,
    CLASSIC_WAYPOINT_REACH_DISTANCE,
    CLASSIC_VELOCITY_CHANGE_WEIGHT,
    CLASSIC_YAW_KP,
    CLASSIC_Z_KP,
    DATASET_GOAL_X_RANGE,
    DATASET_GOAL_Y_MIN,
    YAW_RATE,
)
from geometry_utils import (
    inflate_aabb_2d,
    point_in_any_aabb_2d,
    point_in_polygon_2d,
    segment_is_clear_2d,
)


def _log(message):
    carb.log_warn(f"[CLASSIC] {message}")


class GridAStarPlanner:
    def __init__(self, polygon, resolution):
        self.polygon = list(polygon)
        self.resolution = float(resolution)
        xs = [point[0] for point in self.polygon]
        ys = [point[1] for point in self.polygon]
        self.x_min = min(xs)
        self.y_min = min(ys)
        self.x_max = max(xs)
        self.y_max = max(ys)
        self.x_count = int(math.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.y_count = int(math.ceil((self.y_max - self.y_min) / self.resolution)) + 1

    def plan(self, start_xy, goal_xy, obstacle_aabbs):
        start_cell = self._nearest_walkable_cell(start_xy, obstacle_aabbs)
        goal_cell = self._nearest_walkable_cell(goal_xy, obstacle_aabbs)
        if start_cell is None or goal_cell is None:
            return [tuple(start_xy), tuple(goal_xy)]

        if self._segment_is_valid(start_xy, goal_xy, obstacle_aabbs):
            return [tuple(start_xy), tuple(goal_xy)]

        frontier = [(0.0, start_cell)]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break

            for neighbor, step_cost in self._neighbors(current, obstacle_aabbs):
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal_cell)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal_cell not in came_from:
            return [tuple(start_xy), tuple(goal_xy)]

        cells = []
        current = goal_cell
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()

        points = [self._cell_center(cell) for cell in cells]
        points[0] = tuple(start_xy)
        points[-1] = tuple(goal_xy)
        return self._simplify(points, obstacle_aabbs)

    def _nearest_walkable_cell(self, xy, obstacle_aabbs):
        base = self._world_to_cell(xy)
        max_radius = max(self.x_count, self.y_count)
        for radius in range(max_radius + 1):
            for ix in range(base[0] - radius, base[0] + radius + 1):
                for iy in range(base[1] - radius, base[1] + radius + 1):
                    if radius > 0 and abs(ix - base[0]) != radius and abs(iy - base[1]) != radius:
                        continue
                    cell = (ix, iy)
                    if self._cell_is_walkable(cell, obstacle_aabbs):
                        return cell
        return None

    def _neighbors(self, cell, obstacle_aabbs):
        cx, cy = cell
        current_xy = self._cell_center(cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cx + dx, cy + dy)
                if not self._cell_is_walkable(neighbor, obstacle_aabbs):
                    continue
                neighbor_xy = self._cell_center(neighbor)
                if not self._segment_is_valid(current_xy, neighbor_xy, obstacle_aabbs):
                    continue
                yield neighbor, math.hypot(dx, dy) * self.resolution

    def _cell_is_walkable(self, cell, obstacle_aabbs):
        ix, iy = cell
        if ix < 0 or iy < 0 or ix >= self.x_count or iy >= self.y_count:
            return False
        x, y = self._cell_center(cell)
        if not point_in_polygon_2d(x, y, self.polygon):
            return False
        return not point_in_any_aabb_2d(x, y, obstacle_aabbs)

    def _segment_is_valid(self, p0, p1, obstacle_aabbs):
        if not segment_is_clear_2d(p0, p1, obstacle_aabbs):
            return False

        distance = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
        steps = max(2, int(math.ceil(distance / max(self.resolution * 0.5, 1e-3))))
        for idx in range(steps + 1):
            t = idx / steps
            x = float(p0[0]) + (float(p1[0]) - float(p0[0])) * t
            y = float(p0[1]) + (float(p1[1]) - float(p0[1])) * t
            if not point_in_polygon_2d(x, y, self.polygon):
                return False
        return True

    def _simplify(self, points, obstacle_aabbs):
        if len(points) <= 2:
            return points

        simplified = [points[0]]
        index = 0
        while index < len(points) - 1:
            next_index = index + 1
            for candidate in range(len(points) - 1, index, -1):
                if self._segment_is_valid(points[index], points[candidate], obstacle_aabbs):
                    next_index = candidate
                    break
            simplified.append(points[next_index])
            index = next_index
        return simplified

    def _world_to_cell(self, xy):
        ix = int(round((float(xy[0]) - self.x_min) / self.resolution))
        iy = int(round((float(xy[1]) - self.y_min) / self.resolution))
        return (
            max(0, min(self.x_count - 1, ix)),
            max(0, min(self.y_count - 1, iy)),
        )

    def _cell_center(self, cell):
        return (
            self.x_min + cell[0] * self.resolution,
            self.y_min + cell[1] * self.resolution,
        )

    @staticmethod
    def _heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])


class ClassicAlgorithmController:
    def __init__(
        self,
        shared_cmd,
        drone,
        people,
        target_point,
        walk_polygon,
        obstacle_aabbs_getter,
        start_recording_callback=None,
        abort_episode_callback=None,
        time_source=None,
        control_rate_hz=20.0,
        command_sink=None,
        state_provider=None,
    ):
        self.cmd = shared_cmd
        self.drone = drone
        self.people = list(people)
        self.target_point = np.array(target_point, dtype=float)
        self.walk_polygon = list(walk_polygon)
        self.obstacle_aabbs_getter = obstacle_aabbs_getter
        self.start_recording_callback = start_recording_callback
        self.abort_episode_callback = abort_episode_callback
        self.time_source = time_source
        self.command_sink = command_sink
        self.state_provider = state_provider
        self.control_rate_hz = float(control_rate_hz)
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        self.control_period = 1.0 / self.control_rate_hz

        self.planner = GridAStarPlanner(self.walk_polygon, CLASSIC_GRID_RESOLUTION)
        self.state = "wait_ground" if CLASSIC_AUTO_START else "idle"
        self.quit = False
        self.path = []
        self.path_index = 0
        self.last_plan_time = 0.0
        self.takeoff_start_time = None
        self.takeoff_start_position = None
        self.last_takeoff_log_time = 0.0
        self.ground_wait_start_time = self._now()
        self.ground_ready_since = None
        self.last_px4_ready_wait_log_time = -1e9
        self._last_control_time = None
        self.last_action_update_time = None
        self.last_nav_log_time = 0.0
        self._last_safe_velocity = np.zeros(2, dtype=float)
        self._person_positions = {}
        self._person_velocities = {}
        self._benchmark_mode = False
        self._benchmark_goal_radius_m = None
        self._decision_latencies_ms = []
        self._decision_latency_records = []

        self._pressed_keys = set()
        self.input_iface = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard() if app_window is not None else None
        if self.keyboard is None:
            self.keyboard_sub = None
            _log("No keyboard found; manual mission start is unavailable.")
        else:
            self.keyboard_sub = self.input_iface.subscribe_to_keyboard_events(
                self.keyboard, self._on_keyboard_event
            )

        _log(
            "Classic mode ready. "
            f"auto_start={CLASSIC_AUTO_START}, "
            f"control_rate={self.control_rate_hz:.1f}Hz, "
            f"press {CLASSIC_START_KEY} to start manually."
        )

    def shutdown(self):
        if self.keyboard_sub is not None and self.keyboard is not None:
            self.input_iface.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_sub)
            self.keyboard_sub = None

    def reset_after_episode(self, reason="reset"):
        self.state = "wait_ground" if CLASSIC_AUTO_START else "idle"
        self.path = []
        self.path_index = 0
        self.last_plan_time = 0.0
        self.takeoff_start_time = None
        self.takeoff_start_position = None
        self.last_takeoff_log_time = 0.0
        self.ground_wait_start_time = self._now()
        self.ground_ready_since = None
        self.last_px4_ready_wait_log_time = -1e9
        self._last_control_time = None
        self.last_action_update_time = None
        self.last_nav_log_time = 0.0
        self._last_safe_velocity = np.zeros(2, dtype=float)
        self._person_positions = {}
        self._person_velocities = {}
        self._decision_latencies_ms = []
        self._decision_latency_records = []
        self._reset_motion()
        if CLASSIC_AUTO_START:
            _log(f"Episode reset after {reason}. Waiting for ground contact before auto takeoff.")
        else:
            _log(f"Episode reset after {reason}. Press {CLASSIC_START_KEY} to start another run.")

    def configure_benchmark(self, goal_point, goal_radius_m=1.00):
        """Use one exact 3-D map goal without changing the navigation policy."""
        goal = np.asarray(goal_point, dtype=float).reshape(3)
        if not np.all(np.isfinite(goal)):
            raise ValueError("benchmark goal must be a finite 3-D point")
        self.target_point = goal.copy()
        self._benchmark_mode = True
        self._benchmark_goal_radius_m = float(goal_radius_m)
        self.path = []
        self.path_index = 0

    def _record_decision_latency(self, elapsed_sec, source="controller"):
        elapsed_sec = float(elapsed_sec)
        if self._benchmark_mode and math.isfinite(elapsed_sec) and elapsed_sec >= 0.0:
            latency_ms = elapsed_sec * 1000.0
            self._decision_latencies_ms.append(latency_ms)
            self._decision_latency_records.append(
                {"latency_ms": latency_ms, "source": str(source)}
            )

    def consume_decision_latencies_ms(self):
        values = list(self._decision_latencies_ms)
        self._decision_latencies_ms.clear()
        self._decision_latency_records.clear()
        return values

    def consume_decision_latency_records(self):
        records = [dict(item) for item in self._decision_latency_records]
        self._decision_latency_records.clear()
        self._decision_latencies_ms.clear()
        return records

    def decision_latency_metadata(self):
        """Describe the wall-clock interval recorded by this controller."""
        return {
            "definition": "observation_ready_to_final_velocity_output_wall_clock",
            "causal_pairing": True,
        }

    def request_start(self):
        if self.state not in ("idle", "wait_ground"):
            _log(f"Start ignored because mission state is {self.state}.")
            return
        self._begin_takeoff()

    def hold_after_episode(self, reason="hold"):
        self.state = "idle"
        self.path = []
        self.path_index = 0
        self.last_plan_time = 0.0
        self.takeoff_start_time = None
        self.takeoff_start_position = None
        self.last_takeoff_log_time = 0.0
        self.ground_ready_since = None
        self._last_control_time = None
        self.last_action_update_time = None
        self.last_nav_log_time = 0.0
        self._last_safe_velocity = np.zeros(2, dtype=float)
        self._person_positions = {}
        self._person_velocities = {}
        self.cmd.reset()
        if self.command_sink is not None:
            self.command_sink.set_motion(0.0, 0.0, 0.0, 0.0)
        _log(f"Episode hold after {reason}. Waiting for PX4 landing before reset.")

    def _begin_takeoff(self):
        self._reset_motion()
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
            _log(
                "Takeoff requested. Navigation will start after "
                f"{CLASSIC_TAKEOFF_SETTLE_SEC:.1f}s stabilization."
            )
        else:
            _log(
                "Takeoff requested. Navigation will wait for PX4 offboard, "
                f"z>={CLASSIC_TAKEOFF_RECORD_MIN_Z:.2f}m, and "
                f"{CLASSIC_TAKEOFF_SETTLE_SEC:.1f}s stabilization."
            )

    def update(self):
        now = self._now()

        if self.state == "idle":
            self._set_motion(0.0, 0.0, 0.0, 0.0)
            return

        if self.state == "wait_ground":
            self._set_motion(0.0, 0.0, 0.0, 0.0)
            self._update_wait_ground(now)
            return

        if self.state == "takeoff":
            self._update_takeoff(now)
            return

        if self.state == "navigate":
            self._update_navigation(now)

    def _update_wait_ground(self, now):
        if not CLASSIC_WAIT_FOR_GROUND_BEFORE_TAKEOFF:
            self._begin_takeoff()
            return

        position = self._drone_position(prefer_sim=self.command_sink is not None)
        if position is None:
            return

        velocity = self._drone_velocity(prefer_sim=self.command_sink is not None)
        vz = 0.0 if velocity is None else float(velocity[2])
        low_enough = float(position[2]) <= CLASSIC_GROUND_Z_THRESHOLD
        settled = low_enough and abs(vz) <= CLASSIC_GROUND_VZ_THRESHOLD

        if settled:
            if self.ground_ready_since is None:
                self.ground_ready_since = now
            if now - self.ground_ready_since >= CLASSIC_GROUND_SETTLE_SEC:
                if not self._command_sink_ready_for_takeoff():
                    if now - self.last_px4_ready_wait_log_time >= 1.0:
                        self.last_px4_ready_wait_log_time = now
                        _log(
                            "Ground contact settled; waiting for PX4 "
                            "Ready for takeoff before automatic arm/start."
                        )
                    return
                _log(
                    "Ground contact settled and PX4 is Ready for takeoff. "
                    "Auto takeoff starts now."
                )
                self._begin_takeoff()
            return

        self.ground_ready_since = None
        waited = 0.0 if self.ground_wait_start_time is None else now - self.ground_wait_start_time
        if low_enough and waited >= CLASSIC_GROUND_MAX_WAIT_SEC:
            if not self._command_sink_ready_for_takeoff():
                if now - self.last_px4_ready_wait_log_time >= 1.0:
                    self.last_px4_ready_wait_log_time = now
                    _log(
                        "Ground wait elapsed, but PX4 is not Ready for "
                        "takeoff; automatic arm remains blocked."
                    )
                return
            _log("Ground wait timed out after low altitude. Auto takeoff starts now.")
            self._begin_takeoff()

    def _update_takeoff(self, now):
        if self.command_sink is None:
            self._set_motion(0.0, 0.0, 0.0, 0.0)
        else:
            self._set_motion(0.0, 0.0, self._height_velocity(prefer_sim=True), 0.0)
        elapsed = 0.0 if self.takeoff_start_time is None else now - self.takeoff_start_time

        if now - self.last_takeoff_log_time >= CLASSIC_TAKEOFF_LOG_INTERVAL_SEC:
            self.last_takeoff_log_time = now
            position = self._drone_position(prefer_sim=self.command_sink is not None)
            z = None if position is None else float(position[2])
            z_text = "unknown" if z is None else f"{z:.2f}"
            offboard = self._command_sink_offboard_started()
            _log(
                f"Takeoff stabilizing: elapsed={elapsed:.1f}/"
                f"{CLASSIC_TAKEOFF_SETTLE_SEC:.1f}s, "
                f"offboard={offboard}, z={z_text}"
            )

        if self.command_sink is not None:
            if not self._command_sink_offboard_started():
                self._trigger_takeoff()
                if elapsed >= CLASSIC_TAKEOFF_MAX_WAIT_SEC:
                    _log(
                        "Takeoff timed out while waiting for PX4 offboard. "
                        "Resetting episode without recording."
                    )
                    if self.abort_episode_callback is not None:
                        self.abort_episode_callback()
                return

            position = self._drone_position(prefer_sim=True)
            if position is None or float(position[2]) < float(CLASSIC_TAKEOFF_RECORD_MIN_Z):
                if elapsed >= CLASSIC_TAKEOFF_MAX_WAIT_SEC:
                    _log(
                        "Takeoff timed out before reaching recording altitude. "
                        "Resetting episode without recording."
                    )
                    if self.abort_episode_callback is not None:
                        self.abort_episode_callback()
                return

        if elapsed < CLASSIC_TAKEOFF_SETTLE_SEC:
            return

        self._plan_path(force=True)
        recording_started = False
        if self.start_recording_callback is not None:
            recording_started = bool(self.start_recording_callback())
        self.state = "navigate"
        if recording_started:
            _log("Recording started; autonomous navigation is active.")
        else:
            _log("Autonomous navigation is active; dataset recording is disabled.")

    def _update_navigation(self, now):
        # PegasusApp calls navigation at the configured control rate. During
        # recording this follows saved observations; flight-only mode uses an
        # independent simulation-time clock at the same rate.
        if self._last_control_time is None:
            control_dt = self.control_period
        else:
            control_dt = max(1e-3, min(0.25, now - self._last_control_time))
        self._last_control_time = now
        self.last_action_update_time = now

        if now - self.last_plan_time >= CLASSIC_PATH_REPLAN_INTERVAL_SEC or not self.path:
            self._plan_path(force=not self.path)

        current_xy = self._current_xy()
        if current_xy is None:
            self._set_motion(0.0, 0.0, 0.0, 0.0)
            return

        static_obstacles = self._inflated_static_obstacles()
        desired_velocity = self._desired_velocity_world(current_xy, static_obstacles)
        agents = self._pedestrian_agents(control_dt)
        safe_velocity = self._select_orca_style_velocity(
            current_xy,
            desired_velocity,
            agents,
            static_obstacles,
        )
        safe_velocity = self._shape_final_approach_velocity(current_xy, safe_velocity)
        safe_velocity = self._smooth_velocity_command(safe_velocity, control_dt, current_xy)

        vx_body, vy_body = self._world_velocity_to_body(safe_velocity)
        vz_world = self._height_velocity()
        yaw_rate = self._yaw_rate_for_goal(safe_velocity)
        self._log_navigation_command(
            now,
            current_xy,
            safe_velocity,
            vx_body,
            vy_body,
            vz_world,
            yaw_rate,
        )
        self._set_motion(vx_body, vy_body, vz_world, yaw_rate)

    def _log_navigation_command(
        self,
        now,
        current_xy,
        velocity_world_xy,
        vx_body,
        vy_body,
        vz_world,
        yaw_rate,
    ):
        interval = max(0.0, float(CLASSIC_NAV_LOG_INTERVAL_SEC))
        if interval <= 0.0 or now - self.last_nav_log_time < interval:
            return

        self.last_nav_log_time = now
        attitude = self._drone_attitude()
        yaw_deg = None
        if attitude is not None:
            try:
                yaw_deg = math.degrees(
                    Rotation.from_quat(np.array(attitude, dtype=float)).as_euler(
                        "XYZ",
                        degrees=False,
                    )[2]
                )
            except Exception:
                yaw_deg = None

        goal_xy = self._effective_goal_xy(current_xy)
        yaw_text = "unknown" if yaw_deg is None else f"{yaw_deg:.1f}"
        _log(
            "Nav command: "
            f"pos=({current_xy[0]:.2f},{current_xy[1]:.2f}), "
            f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), "
            f"yaw={yaw_text}deg, "
            f"v_world=({float(velocity_world_xy[0]):.2f},{float(velocity_world_xy[1]):.2f}), "
            f"v_body=({float(vx_body):.2f},{float(vy_body):.2f}), "
            f"vz={float(vz_world):.2f}, yaw_rate={float(yaw_rate):.2f}"
        )

    def _plan_path(self, force=False):
        current_xy = self._current_xy()
        if current_xy is None:
            return

        static_obstacles = self._inflated_static_obstacles()
        goal_xy = self._effective_goal_xy(current_xy)
        self.path = self.planner.plan(current_xy, goal_xy, static_obstacles)
        self.path_index = 0
        self.last_plan_time = self._now()
        if force:
            _log(f"Planned path with {len(self.path)} waypoint(s).")

    def _desired_velocity_world(self, current_xy, static_obstacles):
        self._advance_path_index(current_xy, static_obstacles)
        target_xy = self._lookahead_target(current_xy, static_obstacles)
        delta = np.array([target_xy[0] - current_xy[0], target_xy[1] - current_xy[1]], dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return np.zeros(2, dtype=float)

        final_delta = self._effective_goal_delta(current_xy)
        final_distance = float(np.linalg.norm(final_delta))
        speed_scale = min(1.0, max(0.0, final_distance / CLASSIC_SLOW_RADIUS))
        speed = CLASSIC_MIN_SPEED + (CLASSIC_MAX_SPEED - CLASSIC_MIN_SPEED) * speed_scale
        speed = min(speed, CLASSIC_MAX_SPEED, distance / 0.6)
        if final_distance <= CLASSIC_FINAL_APPROACH_DISTANCE:
            speed = min(speed, CLASSIC_FINAL_MAX_SPEED, final_distance / 1.4)
        return delta / distance * speed

    def _advance_path_index(self, current_xy, static_obstacles):
        while self.path_index < len(self.path) - 1:
            waypoint = self.path[self.path_index]
            distance = math.hypot(waypoint[0] - current_xy[0], waypoint[1] - current_xy[1])
            if distance > CLASSIC_WAYPOINT_REACH_DISTANCE:
                break
            self.path_index += 1

        for candidate in range(len(self.path) - 1, self.path_index, -1):
            if self.planner._segment_is_valid(current_xy, self.path[candidate], static_obstacles):
                self.path_index = candidate
                return

    def _lookahead_target(self, current_xy, static_obstacles):
        effective_goal = self._effective_goal_xy(current_xy)
        if self._effective_goal_distance(current_xy) <= CLASSIC_PATH_LOOKAHEAD_DISTANCE:
            return effective_goal

        if not self.path:
            return effective_goal

        target = self.path[min(self.path_index, len(self.path) - 1)]
        if math.hypot(target[0] - current_xy[0], target[1] - current_xy[1]) >= CLASSIC_PATH_LOOKAHEAD_DISTANCE:
            return target

        for candidate in range(self.path_index + 1, len(self.path)):
            point = self.path[candidate]
            if not self.planner._segment_is_valid(current_xy, point, static_obstacles):
                break
            target = point
            if math.hypot(point[0] - current_xy[0], point[1] - current_xy[1]) >= CLASSIC_PATH_LOOKAHEAD_DISTANCE:
                break
        return target

    def _select_orca_style_velocity(self, current_xy, desired_velocity, agents, static_obstacles):
        candidates = self._velocity_candidates(desired_velocity, current_xy)
        best_velocity = candidates[0]
        best_score = None

        for candidate in candidates:
            score = self._velocity_score(
                current_xy,
                candidate,
                desired_velocity,
                agents,
                static_obstacles,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_velocity = candidate

        return best_velocity

    def _velocity_candidates(self, desired_velocity, current_xy):
        desired_speed = float(np.linalg.norm(desired_velocity))
        if desired_speed < 1e-6:
            return [np.zeros(2, dtype=float)]

        goal_dir = self._goal_direction(current_xy)
        desired_heading = math.atan2(desired_velocity[1], desired_velocity[0])
        goal_heading = desired_heading
        if goal_dir is not None:
            goal_heading = math.atan2(goal_dir[1], goal_dir[0])

        angle_offsets_deg = (
            0,
            5,
            -5,
            10,
            -10,
            18,
            -18,
            28,
            -28,
            42,
            -42,
            60,
            -60,
            85,
            -85,
            120,
            -120,
        )
        speed_scales = (1.0, 0.92, 0.78, 0.60, 0.40, 0.20, 0.0)
        goal_distance = self._effective_goal_distance(current_xy)
        base_speed = desired_speed
        if goal_distance > CLASSIC_SLOW_RADIUS:
            base_speed = max(CLASSIC_MIN_SPEED, desired_speed)

        candidates = [desired_velocity.copy()]
        for base_heading in self._unique_headings((desired_heading, goal_heading)):
            for scale in speed_scales:
                speed = min(CLASSIC_MAX_SPEED, base_speed * scale)
                if scale == 0.0:
                    candidates.append(np.zeros(2, dtype=float))
                    continue
                for angle_offset in angle_offsets_deg:
                    heading = base_heading + math.radians(angle_offset)
                    candidates.append(np.array([math.cos(heading) * speed, math.sin(heading) * speed]))

        deduped = []
        seen = set()
        for candidate in candidates:
            key = (round(float(candidate[0]), 3), round(float(candidate[1]), 3))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _velocity_score(self, current_xy, candidate, desired_velocity, agents, static_obstacles):
        goal_dir = self._goal_direction(current_xy)
        score = CLASSIC_DESIRED_VELOCITY_WEIGHT * float(np.linalg.norm(candidate - desired_velocity) ** 2)
        if goal_dir is not None:
            forward_progress = float(np.dot(candidate, goal_dir))
            lateral_speed = abs(float(goal_dir[0] * candidate[1] - goal_dir[1] * candidate[0]))
            goal_distance = self._effective_goal_distance(current_xy)
            progress_weight_scale = min(1.0, max(0.15, goal_distance / CLASSIC_SLOW_RADIUS))
            score -= CLASSIC_GOAL_PROGRESS_WEIGHT * progress_weight_scale * forward_progress
            score += CLASSIC_LATERAL_DETOUR_WEIGHT * lateral_speed * lateral_speed
            change_weight = CLASSIC_VELOCITY_CHANGE_WEIGHT
            if goal_distance <= CLASSIC_FINAL_APPROACH_DISTANCE:
                change_weight = CLASSIC_FINAL_VELOCITY_CHANGE_WEIGHT
            score += change_weight * float(np.linalg.norm(candidate - self._last_safe_velocity) ** 2)
            if forward_progress < 0.0:
                score += CLASSIC_BACKTRACK_PENALTY * forward_progress * forward_progress

        predicted_xy = (
            current_xy[0] + float(candidate[0]) * CLASSIC_STATIC_LOOKAHEAD_SEC,
            current_xy[1] + float(candidate[1]) * CLASSIC_STATIC_LOOKAHEAD_SEC,
        )
        if (
            not point_in_polygon_2d(predicted_xy[0], predicted_xy[1], self.walk_polygon)
            or point_in_any_aabb_2d(predicted_xy[0], predicted_xy[1], static_obstacles)
            or not segment_is_clear_2d(current_xy, predicted_xy, static_obstacles)
        ):
            score += 10000.0

        hard_radius = (
            CLASSIC_DRONE_RADIUS
            + CLASSIC_PEDESTRIAN_RADIUS
            + min(CLASSIC_AVOIDANCE_MARGIN, CLASSIC_HARD_AVOIDANCE_MARGIN)
        )
        soft_radius = hard_radius + CLASSIC_SOFT_AVOIDANCE_MARGIN
        for agent in agents:
            rel_pos = agent["position"] - np.array(current_xy, dtype=float)
            current_distance = float(np.linalg.norm(rel_pos))
            if current_distance > CLASSIC_ORCA_NEIGHBOR_RADIUS:
                continue

            rel_velocity = candidate - agent["velocity"]
            rel_speed_sq = float(np.dot(rel_velocity, rel_velocity))
            if rel_speed_sq < 1e-8:
                closest_time = 0.0
                closest_distance = current_distance
            else:
                closest_time = float(np.dot(rel_pos, rel_velocity) / rel_speed_sq)
                closest_time = max(0.0, min(CLASSIC_ORCA_TIME_HORIZON, closest_time))
                closest_vector = rel_pos - rel_velocity * closest_time
                closest_distance = float(np.linalg.norm(closest_vector))

            time_weight = 0.25 + 0.75 * (1.0 - closest_time / CLASSIC_ORCA_TIME_HORIZON)
            if current_distance < hard_radius:
                score += 1000000.0 + (hard_radius - current_distance) * 100000.0
            elif closest_distance < hard_radius:
                penetration = hard_radius - closest_distance
                score += 250000.0 + penetration * 120000.0 * time_weight
            elif closest_distance < soft_radius:
                soft_ratio = (soft_radius - closest_distance) / max(
                    soft_radius - hard_radius,
                    1e-6,
                )
                score += CLASSIC_SOFT_CLEARANCE_WEIGHT * soft_ratio * soft_ratio * time_weight
            elif closest_time < CLASSIC_ORCA_TIME_HORIZON * 0.55:
                clearance = closest_distance - soft_radius
                score += 0.05 * time_weight / max(clearance, 0.1)

        return score

    def _shape_final_approach_velocity(self, current_xy, velocity_xy):
        goal_distance = self._effective_goal_distance(current_xy)
        if goal_distance > CLASSIC_FINAL_APPROACH_DISTANCE:
            return np.array(velocity_xy, dtype=float)

        goal_dir = self._goal_direction(current_xy)
        if goal_dir is None:
            return np.zeros(2, dtype=float)

        velocity = np.array(velocity_xy, dtype=float)
        lateral_axis = np.array([-goal_dir[1], goal_dir[0]], dtype=float)
        forward = float(np.dot(velocity, goal_dir))
        lateral = float(np.dot(velocity, lateral_axis))

        speed_scale = min(1.0, max(0.0, goal_distance / max(CLASSIC_FINAL_APPROACH_DISTANCE, 1e-6)))
        max_speed = max(0.05, CLASSIC_FINAL_MAX_SPEED * (0.35 + 0.65 * speed_scale))
        max_lateral = max(0.03, CLASSIC_FINAL_LATERAL_SPEED * speed_scale)

        forward = float(np.clip(forward, -0.25 * max_speed, max_speed))
        lateral = float(np.clip(lateral, -max_lateral, max_lateral))
        shaped = goal_dir * forward + lateral_axis * lateral

        speed = float(np.linalg.norm(shaped))
        if speed > max_speed:
            shaped *= max_speed / speed
        return shaped

    def _smooth_velocity_command(self, velocity_xy, dt, current_xy):
        target = np.array(velocity_xy, dtype=float)
        previous = np.array(self._last_safe_velocity, dtype=float)
        if not np.all(np.isfinite(previous)):
            previous = np.zeros(2, dtype=float)

        goal_distance = self._effective_goal_distance(current_xy)
        accel_limit = CLASSIC_COMMAND_MAX_ACCEL_MPS2
        if goal_distance <= CLASSIC_FINAL_APPROACH_DISTANCE:
            accel_limit = CLASSIC_FINAL_COMMAND_MAX_ACCEL_MPS2

        delta = target - previous
        max_delta = max(1e-6, float(accel_limit) * max(float(dt), 1e-3))
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            target = previous + delta * (max_delta / delta_norm)

        alpha = float(np.clip(CLASSIC_COMMAND_SMOOTHING_ALPHA, 0.0, 1.0))
        smoothed = alpha * target + (1.0 - alpha) * previous
        self._last_safe_velocity = smoothed
        return smoothed

    def _pedestrian_agents(self, dt):
        agents = []
        next_positions = {}
        next_velocities = {}

        for person in self.people:
            name = self._person_name(person)
            state = getattr(person, "state", None)
            position = getattr(state, "position", None)
            if position is None:
                continue

            position_xy = np.array([float(position[0]), float(position[1])], dtype=float)
            measured_velocity = self._person_state_velocity(state)
            previous_position = self._person_positions.get(name)
            if measured_velocity is None and previous_position is not None and dt > 1e-4:
                measured_velocity = (position_xy - previous_position) / dt
            if measured_velocity is None:
                measured_velocity = np.zeros(2, dtype=float)

            speed = float(np.linalg.norm(measured_velocity))
            if speed > CLASSIC_MAX_PERSON_SPEED_ESTIMATE:
                measured_velocity *= CLASSIC_MAX_PERSON_SPEED_ESTIMATE / speed

            previous_velocity = self._person_velocities.get(name, np.zeros(2, dtype=float))
            velocity = (
                CLASSIC_PERSON_VEL_FILTER * measured_velocity
                + (1.0 - CLASSIC_PERSON_VEL_FILTER) * previous_velocity
            )

            next_positions[name] = position_xy
            next_velocities[name] = velocity
            agents.append({"position": position_xy, "velocity": velocity})

        self._person_positions = next_positions
        self._person_velocities = next_velocities
        return agents

    @staticmethod
    def _person_state_velocity(state):
        velocity = getattr(state, "linear_velocity", None)
        if velocity is None:
            return None
        try:
            velocity = np.array([float(velocity[0]), float(velocity[1])], dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(velocity)):
            return None
        return velocity

    def _world_velocity_to_body(self, velocity_xy):
        attitude = self._drone_attitude()
        if attitude is None:
            return 0.0, 0.0
        rot = Rotation.from_quat(attitude)
        rot_mat = rot.as_matrix()
        x_body = rot_mat[:2, 0]
        y_body = rot_mat[:2, 1]
        return float(np.dot(velocity_xy, x_body)), float(np.dot(velocity_xy, y_body))

    def _height_velocity(self, prefer_sim=False):
        position = self._drone_position(prefer_sim=prefer_sim)
        if position is None:
            return 0.0
        error = CLASSIC_CRUISE_HEIGHT - float(position[2])
        return float(np.clip(CLASSIC_Z_KP * error, -CLASSIC_MAX_Z_SPEED, CLASSIC_MAX_Z_SPEED))

    def _yaw_rate_for_goal(self, fallback_velocity_xy):
        current_xy = self._current_xy()
        goal_dir = self._goal_direction(current_xy)
        if goal_dir is not None:
            desired_yaw = math.atan2(float(goal_dir[1]), float(goal_dir[0]))
        else:
            speed = float(np.linalg.norm(fallback_velocity_xy))
            if speed < 0.05:
                return 0.0
            desired_yaw = math.atan2(float(fallback_velocity_xy[1]), float(fallback_velocity_xy[0]))

        attitude = self._drone_attitude()
        if attitude is None:
            return 0.0
        current_yaw = Rotation.from_quat(np.array(attitude, dtype=float)).as_euler(
            "XYZ",
            degrees=False,
        )[2]
        yaw_error = self._wrap_pi(desired_yaw - current_yaw)
        yaw_rate = float(np.clip(CLASSIC_YAW_KP * yaw_error, -YAW_RATE, YAW_RATE))
        goal_distance = self._effective_goal_distance(current_xy)
        if goal_distance <= CLASSIC_FINAL_APPROACH_DISTANCE:
            distance_scale = min(1.0, max(0.0, goal_distance / max(CLASSIC_FINAL_APPROACH_DISTANCE, 1e-6)))
            yaw_scale = float(CLASSIC_FINAL_YAW_RATE_SCALE) + (1.0 - float(CLASSIC_FINAL_YAW_RATE_SCALE)) * distance_scale
            yaw_rate *= yaw_scale
        return yaw_rate

    def _goal_direction(self, current_xy):
        if current_xy is None:
            return None
        delta = self._effective_goal_delta(current_xy)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return None
        return delta / distance

    def _effective_goal_xy(self, current_xy):
        if self._benchmark_mode:
            return (float(self.target_point[0]), float(self.target_point[1]))
        if current_xy is None:
            return (float(self.target_point[0]), float(self.target_point[1]))

        x_min, x_max = (float(DATASET_GOAL_X_RANGE[0]), float(DATASET_GOAL_X_RANGE[1]))
        guard = max(0.0, float(CLASSIC_GOAL_REGION_X_GUARD))
        if x_max - x_min > 2.0 * guard:
            x_min += guard
            x_max -= guard

        current_x = float(current_xy[0])
        target_x = min(max(current_x, x_min), x_max)
        return (target_x, float(DATASET_GOAL_Y_MIN))

    def _effective_goal_delta(self, current_xy):
        goal_xy = np.array(self._effective_goal_xy(current_xy), dtype=float)
        return goal_xy - np.array(current_xy, dtype=float)

    def _effective_goal_distance(self, current_xy):
        return float(np.linalg.norm(self._effective_goal_delta(current_xy)))

    @staticmethod
    def _unique_headings(headings):
        unique = []
        for heading in headings:
            if all(abs(ClassicAlgorithmController._wrap_pi(heading - value)) > math.radians(2.0) for value in unique):
                unique.append(heading)
        return unique

    def _inflated_static_obstacles(self):
        obstacle_aabbs = self.obstacle_aabbs_getter() if self.obstacle_aabbs_getter else []
        return [
            inflate_aabb_2d(aabb, CLASSIC_STATIC_OBSTACLE_CLEARANCE)
            for aabb in obstacle_aabbs
        ]

    def _drone_position(self, prefer_sim=False):
        if not prefer_sim:
            state = self._drone_state()
            if state is not None:
                try:
                    return np.array(state.position, dtype=float)
                except Exception:
                    pass
        sim_position = self._sim_drone_position()
        if sim_position is not None:
            return sim_position
        if prefer_sim:
            state = self._drone_state()
            if state is not None:
                try:
                    return np.array(state.position, dtype=float)
                except Exception:
                    pass
        return None

    def _sim_drone_position(self):
        try:
            return np.array(self.drone.state.position, dtype=float)
        except Exception:
            return None

    def _drone_velocity(self, prefer_sim=False):
        if not prefer_sim:
            state = self._drone_state()
            if state is not None:
                velocity = getattr(state, "linear_velocity", None)
                if velocity is not None:
                    try:
                        return np.array(velocity, dtype=float)
                    except Exception:
                        pass
        sim_velocity = self._sim_drone_velocity()
        if sim_velocity is not None:
            return sim_velocity
        if prefer_sim:
            state = self._drone_state()
            if state is not None:
                velocity = getattr(state, "linear_velocity", None)
                if velocity is not None:
                    try:
                        return np.array(velocity, dtype=float)
                    except Exception:
                        pass
        return None

    def _sim_drone_velocity(self):
        try:
            return np.array(self.drone.state.linear_velocity, dtype=float)
        except Exception:
            return None

    def _drone_attitude(self):
        state = self._drone_state()
        if state is not None:
            attitude = getattr(state, "attitude", None)
            if attitude is not None:
                try:
                    return np.array(attitude, dtype=float)
                except Exception:
                    pass
        try:
            return np.array(self.drone.state.attitude, dtype=float)
        except Exception:
            return None

    def _drone_state(self):
        if self.state_provider is None:
            return None
        try:
            return self.state_provider()
        except Exception:
            return None

    def _current_xy(self):
        position = self._drone_position()
        if position is None:
            return None
        return (float(position[0]), float(position[1]))

    @staticmethod
    def _wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _now(self):
        if self.time_source is not None:
            try:
                value = self.time_source()
                if value is not None and math.isfinite(float(value)):
                    return float(value)
            except Exception:
                pass
        return time.perf_counter()

    def _set_motion(self, vx_body, vy_body, vz_world, yaw_rate):
        self.cmd.set_motion(vx_body, vy_body, vz_world, yaw_rate)
        if self.command_sink is not None:
            self.command_sink.set_motion(vx_body, vy_body, vz_world, yaw_rate)

    def _reset_motion(self):
        self.cmd.reset()
        if self.command_sink is not None:
            reset = getattr(self.command_sink, "reset_after_episode", None)
            if callable(reset):
                reset()

    def _trigger_takeoff(self):
        self.cmd.trigger_takeoff()
        if self.command_sink is not None:
            trigger = getattr(self.command_sink, "trigger_takeoff", None)
            if callable(trigger):
                trigger()

    def _trigger_land(self):
        self.cmd.trigger_land()
        if self.command_sink is not None:
            trigger = getattr(self.command_sink, "trigger_land", None)
            if callable(trigger):
                trigger()

    def _command_sink_offboard_started(self):
        if self.command_sink is None:
            return True
        getter = getattr(self.command_sink, "is_offboard_started", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _command_sink_ready_for_takeoff(self):
        if self.command_sink is None:
            return True
        getter = getattr(self.command_sink, "is_ready_for_takeoff", None)
        if not callable(getter):
            # Non-PX4/local command sinks have no pre-arm health state.
            return True
        try:
            return bool(getter())
        except Exception:
            return False

    @staticmethod
    def _person_name(person):
        stage_prefix = getattr(person, "_stage_prefix", None)
        if stage_prefix:
            return stage_prefix.rstrip("/").split("/")[-1]
        return getattr(person, "name", "person")

    def _on_keyboard_event(self, event):
        key_name = self._keyboard_key_name(getattr(event, "input", None))
        if key_name is None:
            return True
        if self._keyboard_event_is_release(event):
            self._pressed_keys.discard(key_name)
            return True
        if not self._keyboard_event_is_press(event) or key_name in self._pressed_keys:
            return True
        self._pressed_keys.add(key_name)
        if key_name == self._normalize_key_name(CLASSIC_START_KEY):
            self.request_start()
        return True

    @staticmethod
    def _keyboard_event_is_press(event):
        event_type = ClassicAlgorithmController._enum_name(getattr(event, "type", None))
        return event_type in ("KEY_PRESS", "KEY_REPEAT", "PRESS")

    @staticmethod
    def _keyboard_event_is_release(event):
        event_type = ClassicAlgorithmController._enum_name(getattr(event, "type", None))
        return event_type in ("KEY_RELEASE", "RELEASE")

    @staticmethod
    def _keyboard_key_name(key):
        if key is None:
            return None
        return ClassicAlgorithmController._normalize_key_name(
            getattr(key, "name", None) or str(key)
        )

    @staticmethod
    def _normalize_key_name(name):
        name = str(name).split(".")[-1].upper()
        return name[4:] if name.startswith("KEY_") else name

    @staticmethod
    def _enum_name(value):
        if value is None:
            return ""
        return (getattr(value, "name", None) or str(value)).split(".")[-1].upper()
