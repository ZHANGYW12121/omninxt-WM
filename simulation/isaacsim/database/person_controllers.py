#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from pegasus.simulator.logic.people.person_controller import PersonController

from geometry_utils import (
    point_is_walkable,
    sample_point_in_polygon,
    sample_point_in_polygon_avoiding_obstacles,
    sample_point_in_rectangle,
    segment_is_clear_2d,
)


class RandomRectanglePersonController(PersonController):
    def __init__(self, xmin, xmax, ymin, ymax, change_interval=4.0, speed=1.0):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.change_interval = change_interval
        self.speed = speed
        self.elapsed = 0.0
        self.target = None

    def _new_target(self):
        self.target = sample_point_in_rectangle(self.xmin, self.xmax, self.ymin, self.ymax)

    def update(self, dt: float):
        self.elapsed += dt
        if self.target is None or self.elapsed >= self.change_interval:
            self._new_target()
            self.elapsed = 0.0
        self._person.update_target_position(self.target, self.speed)


class RandomPolygonPersonController(PersonController):
    def __init__(self, polygon_points, change_interval=4.0, speed=1.0):
        super().__init__()
        self.polygon = polygon_points
        self.change_interval = change_interval
        self.speed = speed
        self.elapsed = 0.0
        self.target = None

    def _new_target(self):
        self.target = sample_point_in_polygon(self.polygon)

    def update(self, dt: float):
        self.elapsed += dt
        if self.target is None or self.elapsed >= self.change_interval:
            self._new_target()
            self.elapsed = 0.0
        self._person.update_target_position(self.target, self.speed)


class CrowdSeparationMixin:
    def configure_crowd_separation(
        self,
        people,
        min_distance=0.7,
        active_distance=1.0,
        step_distance=0.75,
        strength=1.35,
        prediction_time=2.2,
        path_conflict_distance=0.9,
        path_time_margin=0.8,
        path_side_step=1.0,
    ):
        self.crowd_people = list(people)
        self.person_min_distance = float(min_distance)
        self.person_active_distance = max(float(active_distance), self.person_min_distance)
        self.person_separation_step = float(step_distance)
        self.person_separation_strength = float(strength)
        self.person_prediction_time = float(prediction_time)
        self.person_path_conflict_distance = max(
            float(path_conflict_distance),
            self.person_min_distance,
        )
        self.person_path_time_margin = float(path_time_margin)
        self.person_path_side_step = float(path_side_step)
        self.avoidance_target = None
        self._last_commanded_target = None
        self._last_commanded_speed = getattr(self, "speed", 1.0)

    def _target_with_separation(self, target, speed):
        target, speed = self._target_with_predictive_avoidance(target, speed)
        target, speed = self._target_with_reactive_separation(target, speed)
        self._last_commanded_target = list(target)
        self._last_commanded_speed = float(speed)
        return target, speed

    def _target_with_predictive_avoidance(self, target, speed):
        people = getattr(self, "crowd_people", None)
        if not people or self._person is None:
            return target, speed

        active_avoidance = self._active_avoidance_target(target, speed)
        if active_avoidance is not None:
            return active_avoidance, speed

        conflict = self._find_predictive_path_conflict(target, speed)
        if conflict is None:
            return target, speed

        candidate = self._choose_predictive_avoidance_target(target, speed, conflict)
        if candidate is not None:
            self.avoidance_target = candidate
            return candidate, speed

        # When the dense crowd leaves no conflict-free side-step, keep the
        # original walking direction and slow down. Commanding the current
        # position here made the character alternate between walking and
        # stopping on consecutive updates, which appeared as body twitching.
        return target, max(0.25, min(speed, getattr(self, "speed", speed)) * 0.45)

    def _target_with_reactive_separation(self, target, speed):
        people = getattr(self, "crowd_people", None)
        if not people or self._person is None:
            return target, speed

        current = self._current_xy()
        repel_x = 0.0
        repel_y = 0.0
        nearest = None

        for other in people:
            if other is self._person:
                continue

            other_pos = getattr(getattr(other, "state", None), "position", None)
            if other_pos is None:
                continue

            dx = current[0] - float(other_pos[0])
            dy = current[1] - float(other_pos[1])
            distance = math.hypot(dx, dy)
            if nearest is None or distance < nearest:
                nearest = distance
            if self._same_crowd_group(other) and distance >= self.person_min_distance:
                continue
            if distance >= self.person_active_distance:
                continue

            if distance < 1e-4:
                unit_x, unit_y = self._stable_separation_direction(other)
                distance = 1e-4
            else:
                unit_x = dx / distance
                unit_y = dy / distance

            weight = (self.person_active_distance - distance) / self.person_active_distance
            if distance < self.person_min_distance:
                weight += (self.person_min_distance - distance) / self.person_min_distance
            repel_x += unit_x * weight
            repel_y += unit_y * weight

        repel_len = math.hypot(repel_x, repel_y)
        if repel_len < 1e-6:
            return target, speed

        unit_x = repel_x / repel_len
        unit_y = repel_y / repel_len
        urgent = nearest is not None and nearest < self.person_min_distance
        if self.avoidance_target is not None and not urgent:
            # Predictive avoidance already selected a stable passing side.
            # Stacking soft reactive offsets on top of that temporary target
            # makes dense crossing flows continually change heading.
            return target, speed
        step = self.person_separation_step * min(1.5, max(0.35, repel_len))
        if urgent:
            candidates = self._separation_candidates_from_current(current, unit_x, unit_y, step)
        else:
            candidates = self._separation_candidates_from_target(target, unit_x, unit_y, step)
            candidates.extend(self._separation_candidates_from_current(current, unit_x, unit_y, step * 0.7))

        for candidate in candidates:
            if self._is_separation_candidate_valid(candidate):
                adjusted_speed = speed if not urgent else max(0.35, min(speed, self.speed))
                return candidate, adjusted_speed

        return target, speed

    def _active_avoidance_target(self, original_target, speed):
        avoidance_target = getattr(self, "avoidance_target", None)
        if avoidance_target is None:
            return None

        current = self._current_xy()
        dx = current[0] - float(avoidance_target[0])
        dy = current[1] - float(avoidance_target[1])
        if math.hypot(dx, dy) <= 0.35:
            self.avoidance_target = None
            return None

        if not self._is_separation_candidate_valid(avoidance_target):
            self.avoidance_target = None
            return None

        # Keep the chosen side until the temporary waypoint is reached. In a
        # dense crowd a still-detected conflict should not flip the avoidance
        # target to the opposite side on every controller update.
        return avoidance_target

    def _find_predictive_path_conflict(self, target, speed):
        current_segment = self._predictive_segment_for_target(target, speed)
        if current_segment is None:
            return None

        best_conflict = None
        best_distance = None
        for other in getattr(self, "crowd_people", []):
            if other is self._person:
                continue
            if self._same_crowd_group(other):
                continue
            if not self._should_yield_to(other):
                continue

            other_segment = self._other_predictive_segment(other)
            if other_segment is None:
                continue

            distance, t_self, t_other = _segment_distance_with_params(
                current_segment["start"],
                current_segment["end"],
                other_segment["start"],
                other_segment["end"],
            )
            if distance > self.person_path_conflict_distance:
                continue

            self_time = t_self * current_segment["duration"]
            other_time = t_other * other_segment["duration"]
            if (
                abs(self_time - other_time) > self.person_path_time_margin
                and distance > self.person_min_distance
            ):
                continue

            conflict = {
                "other": other,
                "distance": distance,
                "self_point": _lerp2(current_segment["start"], current_segment["end"], t_self),
                "other_point": _lerp2(other_segment["start"], other_segment["end"], t_other),
            }
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_conflict = conflict

        return best_conflict

    def _predictive_segment_for_target(self, target, speed):
        current = self._current_xy()
        target_xy = (float(target[0]), float(target[1]))
        dx = target_xy[0] - current[0]
        dy = target_xy[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            return None

        safe_speed = max(float(speed), 0.1)
        lookahead = min(distance, safe_speed * self.person_prediction_time)
        end = (
            current[0] + dx / distance * lookahead,
            current[1] + dy / distance * lookahead,
        )
        return {
            "start": current,
            "end": end,
            "duration": max(lookahead / safe_speed, 1e-3),
        }

    def _other_predictive_segment(self, other):
        other_pos = getattr(getattr(other, "state", None), "position", None)
        if other_pos is None:
            return None

        controller = getattr(other, "_controller", None)
        target = None
        speed = None
        if controller is not None:
            target = getattr(controller, "_last_commanded_target", None)
            speed = getattr(controller, "_last_commanded_speed", None)
            if target is None:
                get_target = getattr(controller, "_get_active_target_for_prediction", None)
                if callable(get_target):
                    target = get_target()
            if speed is None:
                speed = getattr(controller, "speed", None)

        if target is None:
            target = getattr(other, "_target_position", None)
        if target is None:
            return None

        if speed is None:
            speed = getattr(other, "_target_speed", 1.0)
        return self._segment_from_position_to_target(other_pos, target, speed)

    def _get_active_target_for_prediction(self):
        path = getattr(self, "path", None)
        if path:
            return path[0]
        target = getattr(getattr(self, "_person", None), "_target_position", None)
        if target is not None:
            return target
        return None

    def _segment_from_position_to_target(self, position, target, speed):
        start = (float(position[0]), float(position[1]))
        target_xy = (float(target[0]), float(target[1]))
        dx = target_xy[0] - start[0]
        dy = target_xy[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            return None

        safe_speed = max(float(speed), 0.1)
        lookahead = min(distance, safe_speed * self.person_prediction_time)
        end = (
            start[0] + dx / distance * lookahead,
            start[1] + dy / distance * lookahead,
        )
        return {
            "start": start,
            "end": end,
            "duration": max(lookahead / safe_speed, 1e-3),
        }

    def _choose_predictive_avoidance_target(self, target, speed, conflict):
        current = self._current_xy()
        target_xy = (float(target[0]), float(target[1]))
        dx = target_xy[0] - current[0]
        dy = target_xy[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            return None

        forward_x = dx / distance
        forward_y = dy / distance
        lateral_x = -forward_y
        lateral_y = forward_x
        away_x = current[0] - conflict["other_point"][0]
        away_y = current[1] - conflict["other_point"][1]
        preferred_sign = 1.0 if lateral_x * away_x + lateral_y * away_y >= 0.0 else -1.0
        signs = (preferred_sign, -preferred_sign)

        forward_step = min(distance, max(0.8, float(speed) * self.person_prediction_time * 0.55))
        side_step = self.person_path_side_step
        z = float(target[2]) if len(target) >= 3 else float(self._person.state.position[2])
        candidates = []
        for sign in signs:
            candidates.append([
                current[0] + forward_x * forward_step + lateral_x * side_step * sign,
                current[1] + forward_y * forward_step + lateral_y * side_step * sign,
                z,
            ])
            candidates.append([
                current[0] + forward_x * forward_step * 0.6 + lateral_x * side_step * 1.35 * sign,
                current[1] + forward_y * forward_step * 0.6 + lateral_y * side_step * 1.35 * sign,
                z,
            ])
            candidates.append([
                target_xy[0] + lateral_x * side_step * sign,
                target_xy[1] + lateral_y * side_step * sign,
                z,
            ])

        for candidate in candidates:
            if not self._is_separation_candidate_valid(candidate):
                continue
            if self._find_predictive_path_conflict(candidate, speed) is None:
                return candidate

        for candidate in candidates:
            if self._is_separation_candidate_valid(candidate):
                return candidate

        return None

    def _same_crowd_group(self, other):
        my_group = getattr(self, "crowd_group_id", None)
        other_controller = getattr(other, "_controller", None)
        other_group = getattr(other_controller, "crowd_group_id", None)
        return my_group is not None and my_group == other_group

    def _should_yield_to(self, other):
        my_name = getattr(self._person, "_stage_prefix", "")
        other_name = getattr(other, "_stage_prefix", "")
        return str(my_name) > str(other_name)

    def _separation_candidates_from_target(self, target, unit_x, unit_y, step):
        z = float(target[2]) if len(target) >= 3 else float(self._person.state.position[2])
        return [
            [target[0] + unit_x * step * self.person_separation_strength, target[1] + unit_y * step * self.person_separation_strength, z],
            [target[0] + unit_y * step, target[1] - unit_x * step, z],
            [target[0] - unit_y * step, target[1] + unit_x * step, z],
        ]

    def _separation_candidates_from_current(self, current, unit_x, unit_y, step):
        z = float(self._person.state.position[2])
        return [
            [current[0] + unit_x * step, current[1] + unit_y * step, z],
            [current[0] + unit_y * step, current[1] - unit_x * step, z],
            [current[0] - unit_y * step, current[1] + unit_x * step, z],
            [current[0] + unit_x * step * 0.5, current[1] + unit_y * step * 0.5, z],
        ]

    def _is_separation_candidate_valid(self, candidate):
        polygon = getattr(self, "polygon", None)
        obstacle_aabbs = getattr(self, "obstacle_aabbs", [])
        if polygon is not None and not point_is_walkable(candidate[0], candidate[1], polygon, obstacle_aabbs):
            return False
        if obstacle_aabbs and not segment_is_clear_2d(self._current_xy(), (candidate[0], candidate[1]), obstacle_aabbs):
            return False
        return True

    def _stable_separation_direction(self, other):
        name = getattr(self._person, "_stage_prefix", "person")
        other_name = getattr(other, "_stage_prefix", "other")
        seed = sum(ord(ch) for ch in f"{name}|{other_name}")
        angle = (seed % 360) * math.pi / 180.0
        return math.cos(angle), math.sin(angle)


def _lerp2(p0, p1, t):
    return (
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
    )


def _segment_distance_with_params(a0, a1, b0, b1):
    intersect, t_intersect, u_intersect = _segment_intersection_params(a0, a1, b0, b1)
    if intersect:
        return 0.0, t_intersect, u_intersect

    candidates = []
    distance, u = _point_segment_distance_with_param(a0, b0, b1)
    candidates.append((distance, 0.0, u))
    distance, u = _point_segment_distance_with_param(a1, b0, b1)
    candidates.append((distance, 1.0, u))
    distance, t = _point_segment_distance_with_param(b0, a0, a1)
    candidates.append((distance, t, 0.0))
    distance, t = _point_segment_distance_with_param(b1, a0, a1)
    candidates.append((distance, t, 1.0))
    return min(candidates, key=lambda item: item[0])


def _segment_intersection_params(a0, a1, b0, b1):
    rx = a1[0] - a0[0]
    ry = a1[1] - a0[1]
    sx = b1[0] - b0[0]
    sy = b1[1] - b0[1]
    denom = _cross2(rx, ry, sx, sy)
    if abs(denom) < 1e-8:
        return False, 0.0, 0.0

    qpx = b0[0] - a0[0]
    qpy = b0[1] - a0[1]
    t = _cross2(qpx, qpy, sx, sy) / denom
    u = _cross2(qpx, qpy, rx, ry) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, t, u
    return False, t, u


def _point_segment_distance_with_param(point, seg0, seg1):
    vx = seg1[0] - seg0[0]
    vy = seg1[1] - seg0[1]
    length_sq = vx * vx + vy * vy
    if length_sq < 1e-10:
        dx = point[0] - seg0[0]
        dy = point[1] - seg0[1]
        return math.hypot(dx, dy), 0.0

    t = ((point[0] - seg0[0]) * vx + (point[1] - seg0[1]) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (seg0[0] + vx * t, seg0[1] + vy * t)
    dx = point[0] - closest[0]
    dy = point[1] - closest[1]
    return math.hypot(dx, dy), t


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


class ObstacleAwarePolygonPersonController(CrowdSeparationMixin, PersonController):
    def __init__(
        self,
        polygon_points,
        obstacle_aabbs,
        change_interval=6.0,
        speed=1.0,
        waypoint_reach_distance=0.4,
        sample_attempts=200,
    ):
        super().__init__()
        self.polygon = polygon_points
        self.obstacle_aabbs = list(obstacle_aabbs)
        self.change_interval = change_interval
        self.speed = speed
        self.waypoint_reach_distance = waypoint_reach_distance
        self.sample_attempts = sample_attempts
        self.elapsed = 0.0
        self.path = []

    def start(self):
        self.resume_after_pause()

    def resume_after_pause(self):
        self.path = []
        self.elapsed = self.change_interval
        self.avoidance_target = None

    def set_obstacles(self, obstacle_aabbs):
        self.obstacle_aabbs = list(obstacle_aabbs)
        self.path = []
        self.elapsed = self.change_interval
        self.avoidance_target = None

    def _current_xy(self):
        pos = self._person.state.position
        return (float(pos[0]), float(pos[1]))

    def _sample_target(self):
        return sample_point_in_polygon_avoiding_obstacles(
            self.polygon,
            self.obstacle_aabbs,
            max_attempts=self.sample_attempts,
        )

    def _segment_clear(self, p0, p1):
        return segment_is_clear_2d(
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
            self.obstacle_aabbs,
        )

    def _plan_path(self):
        start = self._current_xy()
        target = self._sample_target()
        if self._segment_clear(start, target):
            self.path = [target]
            return

        for _ in range(self.sample_attempts):
            mid = self._sample_target()
            if self._segment_clear(start, mid) and self._segment_clear(mid, target):
                self.path = [mid, target]
                return

        for _ in range(self.sample_attempts):
            mid1 = self._sample_target()
            mid2 = self._sample_target()
            if (
                self._segment_clear(start, mid1)
                and self._segment_clear(mid1, mid2)
                and self._segment_clear(mid2, target)
            ):
                self.path = [mid1, mid2, target]
                return

        self.path = [target]

    def _drop_reached_waypoints(self):
        if not self.path:
            return

        current = self._current_xy()
        while self.path:
            target = self.path[0]
            dx = current[0] - target[0]
            dy = current[1] - target[1]
            if (dx * dx + dy * dy) ** 0.5 > self.waypoint_reach_distance:
                break
            self.path.pop(0)

    def update(self, dt: float):
        self.elapsed += dt
        self._drop_reached_waypoints()

        if not self.path or self.elapsed >= self.change_interval:
            self._plan_path()
            self.elapsed = 0.0

        if self.path:
            target, speed = self._target_with_separation(self.path[0], self.speed)
            self._person.update_target_position(target, speed)


class WaypointPersonController(CrowdSeparationMixin, PersonController):
    def __init__(
        self,
        waypoints,
        obstacle_aabbs=None,
        polygon_points=None,
        speed=1.0,
        loop=False,
        waypoint_reach_distance=0.45,
        sample_attempts=200,
    ):
        super().__init__()
        self.waypoints = [self._to_xyz(waypoint) for waypoint in waypoints]
        self.obstacle_aabbs = list(obstacle_aabbs or [])
        self.polygon = polygon_points
        self.speed = speed
        self.loop = loop
        self.waypoint_reach_distance = waypoint_reach_distance
        self.sample_attempts = sample_attempts
        self.target_index = 0
        self.path = []
        self.has_active_target = False
        self.done = False

    def start(self):
        self.resume_after_pause()

    def resume_after_pause(self):
        if self.loop and self.done:
            self.target_index = 0
            self.done = False
        if self.done:
            return
        self.path = []
        self.has_active_target = False
        self.avoidance_target = None

    def _reset_plan(self):
        self.target_index = 0
        self.path = []
        self.has_active_target = False
        self.done = False
        self.avoidance_target = None

    def set_waypoints(self, waypoints):
        self.waypoints = [self._to_xyz(waypoint) for waypoint in waypoints]
        self._reset_plan()

    def set_obstacles(self, obstacle_aabbs):
        self.obstacle_aabbs = list(obstacle_aabbs)
        self._reset_plan()

    @staticmethod
    def _to_xyz(point):
        if len(point) >= 3:
            return [float(point[0]), float(point[1]), float(point[2])]
        return [float(point[0]), float(point[1]), 0.0]

    def _current_xyz(self):
        pos = self._person.state.position
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    def _current_xy(self):
        pos = self._person.state.position
        return (float(pos[0]), float(pos[1]))

    def _segment_clear(self, p0, p1):
        return segment_is_clear_2d(
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
            self.obstacle_aabbs,
        )

    def _sample_midpoint(self):
        if self.polygon is None:
            return None
        return sample_point_in_polygon_avoiding_obstacles(
            self.polygon,
            self.obstacle_aabbs,
            max_attempts=self.sample_attempts,
        )

    def _plan_path_to(self, target):
        start = self._current_xy()
        target_xy = (target[0], target[1])
        if self._segment_clear(start, target_xy):
            return [target]

        for _ in range(self.sample_attempts):
            mid = self._sample_midpoint()
            if mid is None:
                break
            mid_xy = (mid[0], mid[1])
            if self._segment_clear(start, mid_xy) and self._segment_clear(mid_xy, target_xy):
                return [self._to_xyz(mid), target]

        for _ in range(self.sample_attempts):
            mid1 = self._sample_midpoint()
            mid2 = self._sample_midpoint()
            if mid1 is None or mid2 is None:
                break
            mid1_xy = (mid1[0], mid1[1])
            mid2_xy = (mid2[0], mid2[1])
            if (
                self._segment_clear(start, mid1_xy)
                and self._segment_clear(mid1_xy, mid2_xy)
                and self._segment_clear(mid2_xy, target_xy)
            ):
                return [self._to_xyz(mid1), self._to_xyz(mid2), target]

        return [target]

    def _plan_next_waypoint(self):
        if not self.waypoints:
            self.done = True
            return

        if self.target_index >= len(self.waypoints):
            if not self.loop:
                self.done = True
                return
            self.target_index = 0

        target = self.waypoints[self.target_index]
        self.path = self._plan_path_to(target)
        self.has_active_target = True

    def _drop_reached_waypoints(self):
        if not self.path:
            return

        current = self._current_xy()
        while self.path:
            target = self.path[0]
            dx = current[0] - target[0]
            dy = current[1] - target[1]
            if (dx * dx + dy * dy) ** 0.5 > self.waypoint_reach_distance:
                break
            self.path.pop(0)

        if not self.path and self.has_active_target:
            self.target_index += 1
            self.has_active_target = False

    def update(self, dt: float):
        self._drop_reached_waypoints()

        if not self.path and not self.done:
            self._plan_next_waypoint()

        if self.path:
            target, speed = self._target_with_separation(self.path[0], self.speed)
            self._person.update_target_position(target, speed)
        else:
            current = self._current_xyz()
            self._last_commanded_target = current
            self._last_commanded_speed = 0.0
            self._person.update_target_position(current, 0.0)
