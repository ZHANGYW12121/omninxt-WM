#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import time

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
        same_group_min_distance=1.0,
        max_avoidance_turn_deg=60.0,
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
        self.max_avoidance_turn_radians = math.radians(
            max(20.0, min(85.0, float(max_avoidance_turn_deg)))
        )
        self.avoidance_target = None
        self._last_commanded_target = None
        self._last_commanded_speed = getattr(self, "speed", 1.0)
        # Keep the commanded heading continuous briefly after a temporary
        # person-avoidance waypoint disappears. Without this hysteresis a
        # walker can alternate between the avoidance heading and its restored
        # route/formation slot at the edge of the conflict radius.
        self._avoidance_recovery_ticks = 0
        self._avoidance_recovery_duration_ticks = 30
        self._avoidance_heading_step_radians = math.radians(10.0)
        self._avoidance_command_active = False
        # Group members deliberately walk shoulder-to-shoulder. They ignore
        # normal social avoidance and only separate at a small body-overlap
        # radius. Unrelated walkers still see every member independently.
        self.same_group_hard_distance = max(
            0.25, min(float(same_group_min_distance), self.person_min_distance)
        )

    def _target_with_separation(self, target, speed):
        original_target = list(target)
        self._emergency_separation_active = False
        target, speed = self._target_with_predictive_avoidance(target, speed)
        target, speed = self._target_with_reactive_separation(target, speed)
        avoidance_active = (
            self.avoidance_target is not None
            or math.hypot(
                float(target[0]) - float(original_target[0]),
                float(target[1]) - float(original_target[1]),
            ) > 0.05
        )
        self._avoidance_command_active = avoidance_active
        if avoidance_active:
            self._avoidance_recovery_ticks = self._avoidance_recovery_duration_ticks
        elif self._avoidance_recovery_ticks > 0:
            self._avoidance_recovery_ticks -= 1
        if not self._emergency_separation_active:
            target = self._limit_avoidance_heading(original_target, target)
            if avoidance_active or self._avoidance_recovery_ticks > 0:
                target = self._limit_temporal_avoidance_heading(target)
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

    def _target_with_reactive_separation(self, target, speed, emergency_only=False):
        people = getattr(self, "crowd_people", None)
        if not people or self._person is None:
            return target, speed

        current = self._current_xy()
        repel_x = 0.0
        repel_y = 0.0
        nearest = None
        nearest_unit = None
        urgent_overlap = False

        for other in people:
            if other is self._person:
                continue

            other_pos = getattr(getattr(other, "state", None), "position", None)
            if other_pos is None:
                continue

            dx = current[0] - float(other_pos[0])
            dy = current[1] - float(other_pos[1])
            distance = math.hypot(dx, dy)
            same_group = self._same_crowd_group(other)
            safety_distance = self._required_person_clearance(other)
            interaction_distance = (
                safety_distance
                if same_group or emergency_only
                else self.person_active_distance
            )
            if distance >= interaction_distance:
                continue
            # Outside the emergency overlap radius, exactly one member of a
            # pair yields. Below the full safety distance both walkers escape;
            # group passing is designed to shift lanes before reaching it.
            if distance >= safety_distance and not self._should_yield_to(other):
                continue
            urgent_overlap = urgent_overlap or distance < safety_distance

            if distance < 1e-4:
                unit_x, unit_y = self._stable_separation_direction(other)
                distance = 1e-4
            else:
                unit_x = dx / distance
                unit_y = dy / distance

            if nearest is None or distance < nearest:
                nearest = distance
                nearest_unit = (unit_x, unit_y)

            weight = (interaction_distance - distance) / max(interaction_distance, 1e-3)
            if distance < safety_distance:
                weight += (safety_distance - distance) / max(safety_distance, 1e-3)
            repel_x += unit_x * weight
            repel_y += unit_y * weight

        repel_len = math.hypot(repel_x, repel_y)
        if repel_len < 1e-6 and nearest_unit is not None:
            # Symmetric three/four-person clusters can cancel the summed
            # repulsion vector exactly. Always retain an escape direction.
            repel_x, repel_y = nearest_unit
            repel_len = 1.0
        if repel_len < 1e-6:
            return target, speed

        unit_x = repel_x / repel_len
        unit_y = repel_y / repel_len
        urgent = nearest is not None and urgent_overlap
        self._emergency_separation_active = urgent
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
                if urgent:
                    adjusted_speed = min(speed, 0.26)
                else:
                    clearance_fraction = min(
                        1.0,
                        max(0.0, (nearest - self.person_min_distance) /
                            max(self.person_active_distance - self.person_min_distance, 1e-3)),
                    )
                    adjusted_speed = min(
                        speed,
                        speed * (0.45 + 0.55 * clearance_fraction),
                    )
                return candidate, adjusted_speed

        # A blocked emergency side-step must not fall back to walking through
        # the other character at full speed.
        if urgent:
            current_z = float(self._person.state.position[2])
            return [current[0], current[1], current_z], 0.0
        return target, min(speed, speed * 0.45)

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

        padding_getter = getattr(self, "_predictive_formation_padding", None)
        formation_padding = (
            max(0.0, float(padding_getter()))
            if callable(padding_getter)
            else 0.0
        )
        conflict_distance = self.person_path_conflict_distance + formation_padding
        clearance_distance = self.person_min_distance + formation_padding

        best_conflict = None
        best_distance = None
        for other in getattr(self, "crowd_people", []):
            if other is self._person:
                continue
            # Parallel members of one social group do not negotiate passing
            # sides with each other. Their offset routes and small hard-radius
            # check handle formation safety without twitching.
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
            if distance > conflict_distance:
                continue

            self_time = t_self * current_segment["duration"]
            other_time = t_other * other_segment["duration"]
            if (
                abs(self_time - other_time) > self.person_path_time_margin
                and distance > clearance_distance
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
        # Use a fixed right-hand passing convention.  Selecting the side from
        # tiny frame-to-frame position differences caused left/right flips.
        signs = (-1.0, 1.0)

        forward_step = min(distance, max(0.8, float(speed) * self.person_prediction_time * 0.55))
        # A leader moves its whole formation, while a single walker must pass
        # around the outer edge of a pair/triple instead of aiming at the gap
        # between its members.
        padding_getter = getattr(self, "_predictive_formation_padding", None)
        own_padding = (
            max(0.0, float(padding_getter()))
            if callable(padding_getter)
            else 0.0
        )
        other_padding = self._other_group_span_half_width(conflict.get("other"))
        side_step = self.person_path_side_step + own_padding + other_padding
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

    def _required_person_clearance(self, other):
        if self._same_crowd_group(other):
            return self.same_group_hard_distance
        return self.person_min_distance

    def _should_yield_to(self, other):
        other_controller = getattr(other, "_controller", None)
        other_speed = float(
            getattr(other_controller, "_last_commanded_speed", 0.0) or 0.0
        )
        my_speed = float(getattr(self, "_last_commanded_speed", 0.0) or 0.0)
        # A moving walker always yields to a HOLD/idle walker. Name priority
        # alone allowed a lower-numbered person to walk into a stopped one.
        if other_speed <= 0.08 < my_speed:
            return True
        if my_speed <= 0.08 < other_speed:
            return False
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
        current = self._current_xy()
        for other in getattr(self, "crowd_people", []):
            if other is self._person:
                continue
            other_pos = getattr(getattr(other, "state", None), "position", None)
            if other_pos is None:
                continue
            required = self._required_person_clearance(other)
            candidate_distance = math.hypot(
                float(candidate[0]) - float(other_pos[0]),
                float(candidate[1]) - float(other_pos[1]),
            )
            current_distance = math.hypot(
                current[0] - float(other_pos[0]),
                current[1] - float(other_pos[1]),
            )
            segment_distance, _ = _point_segment_distance_with_param(
                (float(other_pos[0]), float(other_pos[1])),
                current,
                (float(candidate[0]), float(candidate[1])),
            )
            # An endpoint can be clear while the straight AnimationGraph path
            # to it crosses a person's body. Reject that entire segment.
            if current_distance >= required and segment_distance < required:
                return False
            # While already overlapping, allow only candidates that increase
            # clearance.  Otherwise never steer toward another person's body.
            # When already inside the social radius, allow tangential motion
            # that keeps roughly the same clearance. Requiring an immediate
            # +5 cm improvement rejected every sideways candidate at 0.99 m
            # and produced permanent face-to-face gridlock.
            if (
                candidate_distance < required
                and candidate_distance < current_distance - 0.03
            ):
                return False

        # Treat every unrelated pair/triple as one filled capsule. Checking
        # only the individual body circles leaves a traversable slit between
        # shoulder-to-shoulder members, so a single person can unnaturally cut
        # through a temporarily stretched formation.
        candidate_xy = (float(candidate[0]), float(candidate[1]))
        for span_start, span_end in self._other_group_capsules():
            current_distance, _ = _point_segment_distance_with_param(
                current, span_start, span_end
            )
            candidate_distance, _ = _point_segment_distance_with_param(
                candidate_xy, span_start, span_end
            )
            path_distance, _, _ = _segment_distance_with_params(
                current, candidate_xy, span_start, span_end
            )
            required = self.person_min_distance
            if current_distance >= required and path_distance < required:
                return False
            if (
                candidate_distance < required
                and candidate_distance < current_distance - 0.03
            ):
                return False
        return True

    def _other_group_capsules(self):
        my_group = getattr(self, "crowd_group_id", None)
        grouped = {}
        for other in getattr(self, "crowd_people", []):
            if other is self._person:
                continue
            controller = getattr(other, "_controller", None)
            group_id = getattr(controller, "crowd_group_id", None)
            if group_id is None or group_id == my_group:
                continue
            if int(getattr(controller, "group_size", 1) or 1) <= 1:
                continue
            position = getattr(getattr(other, "state", None), "position", None)
            if position is None:
                continue
            grouped.setdefault(group_id, []).append(
                (float(position[0]), float(position[1]))
            )

        for positions in grouped.values():
            if len(positions) < 2:
                continue
            span_start, span_end = max(
                (
                    (first, second)
                    for index, first in enumerate(positions)
                    for second in positions[index + 1:]
                ),
                key=lambda pair: (
                    (pair[0][0] - pair[1][0]) ** 2
                    + (pair[0][1] - pair[1][1]) ** 2
                ),
            )
            yield span_start, span_end

    def _other_group_span_half_width(self, other):
        if other is None:
            return 0.0
        controller = getattr(other, "_controller", None)
        group_id = getattr(controller, "crowd_group_id", None)
        if group_id is None or int(getattr(controller, "group_size", 1) or 1) <= 1:
            return 0.0
        positions = []
        for person in getattr(self, "crowd_people", []):
            person_controller = getattr(person, "_controller", None)
            if getattr(person_controller, "crowd_group_id", None) != group_id:
                continue
            position = getattr(getattr(person, "state", None), "position", None)
            if position is not None:
                positions.append((float(position[0]), float(position[1])))
        if len(positions) < 2:
            return 0.0
        diameter = max(
            math.hypot(first[0] - second[0], first[1] - second[1])
            for index, first in enumerate(positions)
            for second in positions[index + 1:]
        )
        return min(1.5, 0.5 * diameter)

    def _limit_avoidance_heading(self, original_target, adjusted_target):
        """Prevent a temporary avoidance target from making the body turn back."""
        current = self._current_xy()
        base_x = float(original_target[0]) - current[0]
        base_y = float(original_target[1]) - current[1]
        adjusted_x = float(adjusted_target[0]) - current[0]
        adjusted_y = float(adjusted_target[1]) - current[1]
        base_length = math.hypot(base_x, base_y)
        adjusted_length = math.hypot(adjusted_x, adjusted_y)
        if base_length < 1e-4 or adjusted_length < 1e-4:
            return adjusted_target

        base_angle = math.atan2(base_y, base_x)
        adjusted_angle = math.atan2(adjusted_y, adjusted_x)
        angle_delta = math.atan2(
            math.sin(adjusted_angle - base_angle),
            math.cos(adjusted_angle - base_angle),
        )
        limit = self.max_avoidance_turn_radians
        if abs(angle_delta) <= limit:
            return adjusted_target

        clamped_angle = base_angle + max(-limit, min(limit, angle_delta))
        distance = min(adjusted_length, max(0.65, base_length))
        z = (
            float(adjusted_target[2])
            if len(adjusted_target) >= 3
            else float(self._person.state.position[2])
        )
        candidate = [
            current[0] + math.cos(clamped_angle) * distance,
            current[1] + math.sin(clamped_angle) * distance,
            z,
        ]
        if self._is_separation_candidate_valid(candidate):
            return candidate
        return original_target

    def _limit_temporal_avoidance_heading(self, proposed_target):
        """Slew an avoidance/recovery heading instead of snapping each tick."""
        previous_target = getattr(self, "_last_commanded_target", None)
        if previous_target is None:
            return proposed_target
        current = self._current_xy()
        previous_x = float(previous_target[0]) - current[0]
        previous_y = float(previous_target[1]) - current[1]
        proposed_x = float(proposed_target[0]) - current[0]
        proposed_y = float(proposed_target[1]) - current[1]
        previous_length = math.hypot(previous_x, previous_y)
        proposed_length = math.hypot(proposed_x, proposed_y)
        if previous_length < 0.18 or proposed_length < 0.18:
            return proposed_target

        previous_angle = math.atan2(previous_y, previous_x)
        proposed_angle = math.atan2(proposed_y, proposed_x)
        delta = math.atan2(
            math.sin(proposed_angle - previous_angle),
            math.cos(proposed_angle - previous_angle),
        )
        limit = self._avoidance_heading_step_radians
        if abs(delta) <= limit:
            return proposed_target

        limited_angle = previous_angle + max(-limit, min(limit, delta))
        z = (
            float(proposed_target[2])
            if len(proposed_target) >= 3
            else float(self._person.state.position[2])
        )
        candidate = [
            current[0] + math.cos(limited_angle) * proposed_length,
            current[1] + math.sin(limited_angle) * proposed_length,
            z,
        ]
        if self._is_separation_candidate_valid(candidate):
            return candidate
        return proposed_target

    def _stable_separation_direction(self, other):
        name = getattr(self._person, "_stage_prefix", "person")
        other_name = getattr(other, "_stage_prefix", "other")
        first, second = sorted((str(name), str(other_name)))
        seed = sum(ord(ch) for ch in f"{first}|{second}")
        angle = (seed % 360) * math.pi / 180.0
        sign = 1.0 if str(name) == first else -1.0
        return sign * math.cos(angle), sign * math.sin(angle)


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

        # Never command a known blocked straight segment merely because the
        # randomized detour search failed.  Holding and retrying is safer than
        # allowing an animated pedestrian to walk through a rack or wall.
        return []

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


class NaturalFormationWaypointController(WaypointPersonController):
    """Smooth ping-pong route following with weak per-group formation control."""

    def __init__(
        self,
        waypoints,
        obstacle_aabbs=None,
        polygon_points=None,
        speed=1.0,
        start_waypoint_index=0,
        route_direction=1,
        speed_seed=1,
        speed_change_interval=5.0,
        formation_lateral=0.0,
        formation_longitudinal=0.0,
        group_size=1,
        member_index=0,
        traffic_axis="",
        traffic_gates=(),
        traffic_cycle_sec=18.0,
        traffic_green_start_sec=0.0,
        traffic_green_duration_sec=7.5,
        traffic_clearance_sec=1.5,
        planned_walk_sec=0.0,
        planned_pause_sec=0.0,
        planned_pause_phase_sec=0.0,
        waypoint_reach_distance=0.55,
        sample_attempts=30,
        fallback_sample_attempts=30,
        control_hz=20.0,
        replan_interval=1.0,
        log_interval=5.0,
    ):
        super().__init__(
            waypoints=waypoints,
            obstacle_aabbs=obstacle_aabbs,
            polygon_points=polygon_points,
            speed=speed,
            loop=True,
            waypoint_reach_distance=waypoint_reach_distance,
            sample_attempts=sample_attempts,
        )
        self.base_speed = float(speed)
        self.current_speed = max(0.25, float(speed))
        self.target_speed = float(speed)
        self.speed_change_interval = float(speed_change_interval)
        self.speed_change_elapsed = 0.0
        self.speed_rng = random.Random(int(speed_seed))
        self.acceleration_limit = 0.30
        self.turn_slow_distance = 1.25
        self.turn_min_speed = 0.30
        self.lookahead_distance = 1.35
        self.formation_lateral = float(formation_lateral)
        self.formation_longitudinal = float(formation_longitudinal)
        self.group_size = int(group_size)
        self.member_index = int(member_index)
        self.traffic_axis = str(traffic_axis or "")
        self.traffic_gates = tuple(
            tuple(float(value) for value in gate)
            for gate in (traffic_gates or ())
        )
        self.traffic_cycle_sec = max(2.0, float(traffic_cycle_sec))
        self.traffic_green_start_sec = float(traffic_green_start_sec)
        self.traffic_green_duration_sec = max(
            0.5, float(traffic_green_duration_sec)
        )
        self.traffic_clearance_sec = max(0.0, float(traffic_clearance_sec))
        self.traffic_schedule_elapsed = 0.0
        self.traffic_waiting = False
        self.traffic_waiting_gate = None
        self.traffic_waiting_opponent = None
        self.traffic_yield_lock = None
        self.traffic_cycle_override = False
        self.traffic_cycle_log_signature = None
        self.traffic_cycle_log_wall_time = 0.0
        self.planned_walk_sec = max(0.0, float(planned_walk_sec))
        self.planned_pause_sec = max(0.0, float(planned_pause_sec))
        self.planned_pause_phase_sec = max(
            0.0, float(planned_pause_phase_sec)
        )
        self.planned_pause_active = False
        self.formation_deadband = 0.14
        self.formation_max_correction = 0.28
        self.formation_gain = 0.65
        # Followers track a short, leader-relative lookahead instead of the
        # leader's full navigation target.  The latter can jump across the
        # leader at a waypoint/route reversal and make a close pair alternate
        # its facing direction from one control tick to the next.
        self.formation_follow_lookahead = 0.90
        self.same_group_hard_distance = 0.30
        self.start_waypoint_index = int(start_waypoint_index)
        self.route_direction = 1 if int(route_direction) >= 0 else -1
        self.target_index = self._first_target_index()
        self.route_plan_failures = 0
        self.fallback_active = False
        self.fallback_sample_attempts = max(1, int(fallback_sample_attempts))
        self.control_interval = 1.0 / max(float(control_hz), 1.0)
        self.control_elapsed = self.control_interval
        self.replan_interval = max(0.1, float(replan_interval))
        self.replan_cooldown_remaining = 0.0
        self.log_interval = max(0.5, float(log_interval))
        self.last_hold_log_wall_time = -float("inf")
        self.last_fallback_log_wall_time = -float("inf")
        self.group_passing_active = False
        self.group_passing_opponent_id = None
        self.group_passing_forward = None
        self.group_passing_right = None
        self.group_passing_lateral_coordinate = None
        self.group_passing_detection_distance = 7.0
        self.group_passing_clearance_margin = 0.40
        self.group_passing_forward_step = 1.8
        self.group_passing_recent_opponent_id = None
        self.group_passing_rearm_distance = 4.5
        self._formation_coordinated_avoidance = False
        self.deadlock_timeout = 3.0
        self.deadlock_escape_duration = 4.5
        self.deadlock_anchor = None
        self.deadlock_elapsed = 0.0
        self.deadlock_escape_target = None
        self.deadlock_escape_elapsed = 0.0
        self.deadlock_escape_cooldown = 0.0
        self.deadlock_escape_attempt = 0

    def configure_natural_motion(
        self,
        *,
        speed,
        start_waypoint_index,
        route_direction,
        speed_seed,
        speed_change_interval,
        formation_lateral,
        formation_longitudinal,
        group_size,
        member_index,
        traffic_axis="",
        traffic_gates=(),
        traffic_cycle_sec=18.0,
        traffic_green_start_sec=0.0,
        traffic_green_duration_sec=7.5,
        traffic_clearance_sec=1.5,
        planned_walk_sec=0.0,
        planned_pause_sec=0.0,
        planned_pause_phase_sec=0.0,
    ):
        self.base_speed = float(speed)
        self.speed = float(speed)
        self.current_speed = max(0.25, float(speed))
        self.target_speed = float(speed)
        self.speed_rng = random.Random(int(speed_seed))
        self.speed_change_interval = float(speed_change_interval)
        self.speed_change_elapsed = 0.0
        self.formation_lateral = float(formation_lateral)
        self.formation_longitudinal = float(formation_longitudinal)
        self.group_size = int(group_size)
        self.member_index = int(member_index)
        self.traffic_axis = str(traffic_axis or "")
        self.traffic_gates = tuple(
            tuple(float(value) for value in gate)
            for gate in (traffic_gates or ())
        )
        self.traffic_cycle_sec = max(2.0, float(traffic_cycle_sec))
        self.traffic_green_start_sec = float(traffic_green_start_sec)
        self.traffic_green_duration_sec = max(
            0.5, float(traffic_green_duration_sec)
        )
        self.traffic_clearance_sec = max(0.0, float(traffic_clearance_sec))
        self.traffic_schedule_elapsed = 0.0
        self.traffic_waiting = False
        self.traffic_waiting_gate = None
        self.traffic_waiting_opponent = None
        self.traffic_yield_lock = None
        self.traffic_cycle_override = False
        self.traffic_cycle_log_signature = None
        self.traffic_cycle_log_wall_time = 0.0
        self.planned_walk_sec = max(0.0, float(planned_walk_sec))
        self.planned_pause_sec = max(0.0, float(planned_pause_sec))
        self.planned_pause_phase_sec = max(
            0.0, float(planned_pause_phase_sec)
        )
        self.planned_pause_active = False
        self.start_waypoint_index = int(start_waypoint_index)
        self.route_direction = 1 if int(route_direction) >= 0 else -1
        self._reset_plan()
        self.target_index = self._first_target_index()
        self.route_plan_failures = 0
        self.fallback_active = False
        self.control_elapsed = self.control_interval
        self.replan_cooldown_remaining = 0.0
        self._clear_group_passing()
        self.group_passing_recent_opponent_id = None
        self._formation_coordinated_avoidance = False
        self._avoidance_command_active = False
        self._avoidance_recovery_ticks = 0
        self._reset_deadlock_watchdog()

    def _first_target_index(self):
        if not self.waypoints:
            return 0
        start = max(0, min(self.start_waypoint_index, len(self.waypoints) - 1))
        candidate = start + self.route_direction
        if candidate < 0 or candidate >= len(self.waypoints):
            self.route_direction *= -1
            candidate = start + self.route_direction
        return max(0, min(candidate, len(self.waypoints) - 1))

    def _segment_clear(self, p0, p1):
        if not super()._segment_clear(p0, p1):
            return False
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is None or leader_item[1] is not self:
            return True
        axis = self._formation_axis()
        if axis is None:
            return True
        forward_x, forward_y = axis
        lateral_x, lateral_y = -forward_y, forward_x
        for _person, controller in members:
            if controller is self:
                continue
            delta_x = forward_x * (
                controller.formation_longitudinal - self.formation_longitudinal
            ) + lateral_x * (controller.formation_lateral - self.formation_lateral)
            delta_y = forward_y * (
                controller.formation_longitudinal - self.formation_longitudinal
            ) + lateral_y * (controller.formation_lateral - self.formation_lateral)
            member_start = (float(p0[0]) + delta_x, float(p0[1]) + delta_y)
            member_end = (float(p1[0]) + delta_x, float(p1[1]) + delta_y)
            if self.polygon is not None and not point_is_walkable(
                member_end[0], member_end[1], self.polygon, self.obstacle_aabbs
            ):
                return False
            if not segment_is_clear_2d(member_start, member_end, self.obstacle_aabbs):
                return False
        return True

    def set_waypoints(self, waypoints):
        self.waypoints = [self._to_xyz(waypoint) for waypoint in waypoints]
        self._reset_plan()
        self.target_index = self._first_target_index()

    def resume_after_pause(self):
        self.path = []
        self.has_active_target = False
        self.done = False
        self.avoidance_target = None
        self.fallback_active = False
        self.current_speed = max(self.turn_min_speed, min(self.current_speed, self.base_speed))
        self.control_elapsed = self.control_interval
        self.replan_cooldown_remaining = 0.0
        self._clear_group_passing()
        self.group_passing_recent_opponent_id = None
        self._formation_coordinated_avoidance = False
        self._avoidance_command_active = False
        self._avoidance_recovery_ticks = 0
        self._reset_deadlock_watchdog()
        self.traffic_schedule_elapsed = 0.0
        self.traffic_waiting = False
        self.traffic_waiting_gate = None
        self.traffic_waiting_opponent = None
        self.traffic_yield_lock = None
        self.planned_pause_active = False

    def _advance_route(self, apply_turn_slowdown=True):
        if not self.waypoints:
            self.done = True
            return
        next_index = self.target_index + self.route_direction
        if next_index < 0 or next_index >= len(self.waypoints):
            self.route_direction *= -1
            next_index = self.target_index + self.route_direction
            if apply_turn_slowdown:
                self.current_speed = min(self.current_speed, self.turn_min_speed)
                self.target_speed = max(self.turn_min_speed, self.base_speed * 0.65)
                self.speed_change_elapsed = 0.0
        self.target_index = max(0, min(next_index, len(self.waypoints) - 1))

    def _drop_reached_waypoints(self):
        if not self.path:
            return
        current = self._current_xy()
        while self.path:
            target = self.path[0]
            if math.hypot(current[0] - target[0], current[1] - target[1]) > self.waypoint_reach_distance:
                break
            self.path.pop(0)
        if not self.path and self.has_active_target:
            self.has_active_target = False
            self._advance_route()

    def _plan_next_waypoint(self):
        if not self.waypoints:
            self.done = True
            return
        # A V2 route is created before USD obstacle AABBs are available.  Once
        # geometry is loaded, an otherwise valid control point can be separated
        # from the walker by a rack.  Never retry one unreachable point forever.
        attempted = 0
        while attempted < len(self.waypoints):
            target = self.waypoints[self.target_index]
            self.path = self._plan_path_to(target)
            if self.path:
                self.has_active_target = True
                self.fallback_active = False
                self.route_plan_failures = 0
                self.replan_cooldown_remaining = 0.0
                return
            self.route_plan_failures += 1
            self._advance_route(apply_turn_slowdown=False)
            attempted += 1

        fallback = self._reachable_fallback_target()
        if fallback is not None:
            self.path = [fallback]
            # A fallback is a temporary safe motion, not a completed route
            # waypoint.  Replan onto the canonical route after reaching it.
            self.has_active_target = False
            self.fallback_active = True
            if self.route_plan_failures == len(self.waypoints):
                self._log_plan_fallback(fallback)
            return

        self.path = []
        self.has_active_target = False
        self.fallback_active = False
        self.replan_cooldown_remaining = self.replan_interval
        self._log_plan_hold()

    def _reachable_fallback_target(self):
        if self.polygon is None:
            return None
        current = self._current_xy()
        route_target = self.waypoints[self.target_index]
        target_dx = float(route_target[0]) - current[0]
        target_dy = float(route_target[1]) - current[1]
        target_len = max(math.hypot(target_dx, target_dy), 1e-6)
        target_dir = (target_dx / target_len, target_dy / target_len)
        best = None
        best_score = None
        for _ in range(self.fallback_sample_attempts):
            sampled = self._sample_midpoint()
            if sampled is None:
                break
            candidate = self._to_xyz(sampled)
            dx, dy = candidate[0] - current[0], candidate[1] - current[1]
            distance = math.hypot(dx, dy)
            if distance < 1.0 or distance > 4.5:
                continue
            if not self._is_separation_candidate_valid(candidate):
                continue
            if not self._segment_clear(current, candidate):
                continue
            progress = (dx * target_dir[0] + dy * target_dir[1]) / distance
            # Prefer safe forward progress, but allow a side/back step when a
            # shelf blocks the canonical direction.
            score = progress - 0.08 * abs(distance - 2.5)
            if best_score is None or score > best_score:
                best, best_score = candidate, score
        return best

    def _log_plan_fallback(self, fallback):
        now = time.perf_counter()
        if now - self.last_fallback_log_wall_time < self.log_interval:
            return
        self.last_fallback_log_wall_time = now
        name = getattr(getattr(self, "_person", None), "_stage_prefix", "person")
        print(
            f"[CROWD][V2][FALLBACK] {name}: canonical route is blocked; "
            f"walking to reachable ({fallback[0]:.2f},{fallback[1]:.2f}) before replanning."
        )

    def _log_plan_hold(self):
        now = time.perf_counter()
        if now - self.last_hold_log_wall_time < self.log_interval:
            return
        self.last_hold_log_wall_time = now
        name = getattr(getattr(self, "_person", None), "_stage_prefix", "person")
        current = self._current_xy()
        print(
            f"[CROWD][V2][HOLD] {name}: no reachable route or fallback from "
            f"({current[0]:.2f},{current[1]:.2f}); retrying safely."
        )

    def _update_smooth_speed(self, dt, target):
        safe_dt = max(0.0, min(float(dt), 0.25))
        self.speed_change_elapsed += safe_dt
        if self.speed_change_elapsed >= self.speed_change_interval:
            self.speed_change_elapsed = 0.0
            self.speed_change_interval = self.speed_rng.uniform(3.0, 8.0)
            self.target_speed = max(0.55, self.base_speed + self.speed_rng.uniform(-0.15, 0.15))

        current = self._current_xy()
        distance = math.hypot(float(target[0]) - current[0], float(target[1]) - current[1])
        at_route_end = self.target_index in (0, len(self.waypoints) - 1)
        desired = self.target_speed
        if at_route_end and distance < self.turn_slow_distance:
            fraction = max(0.0, min(1.0, distance / self.turn_slow_distance))
            desired = min(desired, self.turn_min_speed + (self.target_speed - self.turn_min_speed) * fraction)

        maximum_change = self.acceleration_limit * safe_dt
        delta = max(-maximum_change, min(maximum_change, desired - self.current_speed))
        self.current_speed = max(self.turn_min_speed, self.current_speed + delta)
        return self.current_speed

    def _lookahead_target(self, target):
        current = self._current_xy()
        dx, dy = float(target[0]) - current[0], float(target[1]) - current[1]
        distance = math.hypot(dx, dy)
        if distance <= self.lookahead_distance or distance < 1e-6:
            return list(target)
        scale = self.lookahead_distance / distance
        z = float(target[2]) if len(target) >= 3 else float(self._person.state.position[2])
        return [current[0] + dx * scale, current[1] + dy * scale, z]

    def _formation_adjusted_target(self, target):
        if self.group_size <= 1 or self._person is None:
            return target
        members = []
        for other in getattr(self, "crowd_people", []):
            other_controller = getattr(other, "_controller", None)
            if getattr(other_controller, "crowd_group_id", None) != getattr(self, "crowd_group_id", None):
                continue
            position = getattr(getattr(other, "state", None), "position", None)
            if position is not None:
                members.append((other, other_controller, position))
        if len(members) <= 1:
            return target

        center_x = sum(float(item[2][0]) for item in members) / len(members)
        center_y = sum(float(item[2][1]) for item in members) / len(members)
        current = self._current_xy()
        if len(self.waypoints) < 2:
            return target
        axis_x = float(self.waypoints[-1][0]) - float(self.waypoints[0][0])
        axis_y = float(self.waypoints[-1][1]) - float(self.waypoints[0][1])
        axis_length = math.hypot(axis_x, axis_y)
        if axis_length < 1e-5:
            return target
        # Formation offsets are attached to the canonical route, not the
        # instantaneous travel direction.  Keeping this axis fixed prevents
        # left/right members from trying to swap places after a turnaround.
        forward_x, forward_y = axis_x / axis_length, axis_y / axis_length
        lateral_x, lateral_y = -forward_y, forward_x
        desired_x = center_x + forward_x * self.formation_longitudinal + lateral_x * self.formation_lateral
        desired_y = center_y + forward_y * self.formation_longitudinal + lateral_y * self.formation_lateral
        error_x, error_y = desired_x - current[0], desired_y - current[1]
        error = math.hypot(error_x, error_y)
        if error <= self.formation_deadband:
            return target
        magnitude = min(self.formation_max_correction, (error - self.formation_deadband) * self.formation_gain)
        corrected = list(target)
        corrected[0] += error_x / error * magnitude
        corrected[1] += error_y / error * magnitude
        if self._is_separation_candidate_valid(corrected):
            return corrected
        return target

    def _social_group_members(self):
        if self.group_size <= 1:
            return []
        group_id = getattr(self, "crowd_group_id", None)
        members = []
        for person in getattr(self, "crowd_people", []):
            controller = getattr(person, "_controller", None)
            if getattr(controller, "crowd_group_id", None) != group_id:
                continue
            if not isinstance(controller, NaturalFormationWaypointController):
                continue
            members.append((person, controller))
        return members

    def _all_group_members(self, group_id=None):
        if group_id is None:
            group_id = getattr(self, "crowd_group_id", None)
        members = []
        for person in getattr(self, "crowd_people", []):
            controller = getattr(person, "_controller", None)
            if getattr(controller, "crowd_group_id", None) != group_id:
                continue
            if isinstance(controller, NaturalFormationWaypointController):
                members.append((person, controller))
        return members

    def _clear_group_passing(self):
        self.group_passing_active = False
        self.group_passing_opponent_id = None
        self.group_passing_forward = None
        self.group_passing_right = None
        self.group_passing_lateral_coordinate = None

    @staticmethod
    def _group_center(members):
        positions = [
            getattr(getattr(person, "state", None), "position", None)
            for person, _controller in members
        ]
        positions = [position for position in positions if position is not None]
        if not positions:
            return None
        return (
            sum(float(position[0]) for position in positions) / len(positions),
            sum(float(position[1]) for position in positions) / len(positions),
        )

    @staticmethod
    def _group_half_width(members, leader):
        if not members:
            return 0.0
        return max(
            abs(controller.formation_lateral - leader.formation_lateral)
            for _person, controller in members
        )

    @staticmethod
    def _controller_heading(person, controller):
        position = getattr(getattr(person, "state", None), "position", None)
        target = getattr(controller, "_last_commanded_target", None)
        if position is not None and target is not None:
            dx = float(target[0]) - float(position[0])
            dy = float(target[1]) - float(position[1])
            length = math.hypot(dx, dy)
            if length > 0.20:
                return dx / length, dy / length
        path = getattr(controller, "path", None)
        if position is not None and path:
            dx = float(path[0][0]) - float(position[0])
            dy = float(path[0][1]) - float(position[1])
            length = math.hypot(dx, dy)
            if length > 0.20:
                return dx / length, dy / length
        axis = controller._formation_axis()
        if axis is None:
            return None
        direction = 1.0 if controller.route_direction >= 0 else -1.0
        return axis[0] * direction, axis[1] * direction

    def _find_head_on_group(self, forward):
        own_group = getattr(self, "crowd_group_id", None)
        own_members = self._all_group_members(own_group)
        own_center = self._group_center(own_members)
        if own_center is None:
            return None
        seen = set()
        best = None
        for person in getattr(self, "crowd_people", []):
            controller = getattr(person, "_controller", None)
            other_group = getattr(controller, "crowd_group_id", None)
            if other_group == own_group or other_group in seen:
                continue
            seen.add(other_group)
            other_members = self._all_group_members(other_group)
            other_leader_item = self._social_group_leader(other_members)
            if other_leader_item is None:
                continue
            other_person, other_leader = other_leader_item
            other_center = self._group_center(other_members)
            other_forward = self._controller_heading(other_person, other_leader)
            if other_center is None or other_forward is None:
                continue
            direction_dot = forward[0] * other_forward[0] + forward[1] * other_forward[1]
            if direction_dot > -0.55:
                continue
            rel_x = other_center[0] - own_center[0]
            rel_y = other_center[1] - own_center[1]
            if other_group == self.group_passing_recent_opponent_id:
                if math.hypot(rel_x, rel_y) < self.group_passing_rearm_distance:
                    continue
                self.group_passing_recent_opponent_id = None
            ahead = rel_x * forward[0] + rel_y * forward[1]
            other_ahead = -rel_x * other_forward[0] - rel_y * other_forward[1]
            if not (0.35 < ahead <= self.group_passing_detection_distance and other_ahead > 0.35):
                continue
            own_half = self._group_half_width(own_members, self)
            other_half = self._group_half_width(other_members, other_leader)
            lateral = abs(-forward[1] * rel_x + forward[0] * rel_y)
            conflict_width = own_half + other_half + self.person_min_distance + 0.60
            if lateral > conflict_width:
                continue
            if best is None or ahead < best[0]:
                best = (ahead, other_group, other_members, own_half, other_half)
        return best

    def _start_group_passing(self, forward, conflict):
        _ahead, other_group, _other_members, own_half, other_half = conflict
        right = (forward[1], -forward[0])
        shift = 0.75 * (
            own_half + other_half + self.person_min_distance
            + self.group_passing_clearance_margin
        )
        current = self._current_xy()
        candidate = [
            current[0] + forward[0] * self.group_passing_forward_step + right[0] * shift,
            current[1] + forward[1] * self.group_passing_forward_step + right[1] * shift,
            float(self._person.state.position[2]),
        ]
        if not self._segment_clear(current, candidate):
            return False
        self.group_passing_active = True
        self.group_passing_opponent_id = other_group
        self.group_passing_forward = forward
        self.group_passing_right = right
        self.group_passing_lateral_coordinate = current[0] * right[0] + current[1] * right[1] + shift
        print(
            f"[CROWD][V2][PASS] group={getattr(self, 'crowd_group_id', None)} "
            f"meeting group={other_group}; taking right lane by {shift:.2f}m."
        )
        return True

    def _group_passing_command(self, route_target, speed):
        members = self._all_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is None or leader_item[1] is not self:
            return None
        current = self._current_xy()
        if not self.group_passing_active:
            dx = float(route_target[0]) - current[0]
            dy = float(route_target[1]) - current[1]
            length = math.hypot(dx, dy)
            if length < 0.25:
                return None
            forward = (dx / length, dy / length)
            conflict = self._find_head_on_group(forward)
            if conflict is None or not self._start_group_passing(forward, conflict):
                return None

        opponent_members = self._all_group_members(self.group_passing_opponent_id)
        opponent_center = self._group_center(opponent_members)
        if opponent_center is None:
            self._clear_group_passing()
            return None
        forward = self.group_passing_forward
        right = self.group_passing_right
        rel_x = opponent_center[0] - current[0]
        rel_y = opponent_center[1] - current[1]
        if rel_x * forward[0] + rel_y * forward[1] < -1.25:
            print(
                f"[CROWD][V2][PASS_CLEAR] group={getattr(self, 'crowd_group_id', None)} "
                f"passed group={self.group_passing_opponent_id}."
            )
            self.group_passing_recent_opponent_id = self.group_passing_opponent_id
            self._clear_group_passing()
            return None
        lateral_error = self.group_passing_lateral_coordinate - (
            current[0] * right[0] + current[1] * right[1]
        )
        lateral_error = max(-1.2, min(1.2, lateral_error))
        # Establish lateral clearance first. Driving forward at full lookahead
        # while still changing lanes can put the outer formation member into
        # the oncoming group before the two lane centres have separated.
        forward_step = (
            0.45 if abs(lateral_error) > 0.45
            else self.group_passing_forward_step
        )
        target = [
            current[0] + forward[0] * forward_step + right[0] * lateral_error,
            current[1] + forward[1] * forward_step + right[1] * lateral_error,
            float(self._person.state.position[2]),
        ]
        if not self._segment_clear(current, target):
            return [current[0], current[1], target[2]], 0.0
        limited_speed = min(float(speed), self.base_speed)
        # Formation cohesion has priority over passing progress. If a follower
        # is blocked by the oncoming group, the leader must wait in its lane
        # instead of enforcing a non-zero minimum speed and abandoning it.
        if limited_speed <= 0.08:
            return target, 0.0
        return target, max(0.45, limited_speed)

    def _command_with_reactive_only(self, target, speed, emergency_only=False):
        self._emergency_separation_active = False
        target, speed = self._target_with_reactive_separation(
            target, speed, emergency_only=emergency_only
        )
        if self._formation_coordinated_avoidance:
            self._avoidance_recovery_ticks = self._avoidance_recovery_duration_ticks
            if not self._emergency_separation_active:
                target = self._limit_temporal_avoidance_heading(target)
        self._last_commanded_target = list(target)
        self._last_commanded_speed = float(speed)
        return target, speed

    @staticmethod
    def _social_group_leader(members):
        if not members:
            return None
        # The centre member leads a triple; the first member leads a pair.
        return min(
            members,
            key=lambda item: (
                abs(item[1].formation_lateral),
                item[1].member_index,
            ),
        )

    def _formation_axis(self):
        if len(self.waypoints) < 2:
            return None
        dx = float(self.waypoints[-1][0]) - float(self.waypoints[0][0])
        dy = float(self.waypoints[-1][1]) - float(self.waypoints[0][1])
        length = math.hypot(dx, dy)
        if length < 1e-5:
            return None
        return dx / length, dy / length

    def _predictive_formation_padding(self):
        """Extra predictive radius owned only by a formation leader."""
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is None or leader_item[1] is not self:
            return 0.0
        return min(1.8, self._group_half_width(members, self) + 0.12)

    def _group_follower_command(self):
        """Return a leader-relative parallel target for non-leader members."""
        self._formation_coordinated_avoidance = False
        if self.traffic_axis == "dense_x_flow":
            return self._dense_slot_follower_command()
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is None or leader_item[1] is self:
            return None
        leader_person, leader = leader_item
        leader_position = getattr(getattr(leader_person, "state", None), "position", None)
        if leader_position is None:
            return None
        axis = leader._formation_axis()
        if axis is None:
            return None
        forward_x, forward_y = axis
        lateral_x, lateral_y = -forward_y, forward_x

        leader_target = getattr(leader, "_last_commanded_target", None)
        if leader_target is None:
            leader_target = leader_position
        leader_speed = float(
            getattr(leader, "_last_commanded_speed", leader.current_speed) or 0.0
        )
        leader_avoidance_active = (
            getattr(leader, "avoidance_target", None) is not None
            or bool(getattr(leader, "_avoidance_command_active", False))
            or bool(getattr(leader, "_emergency_separation_active", False))
        )
        if leader_avoidance_active:
            # The follower target below already inherits the leader's lateral
            # displacement. Running predictive avoidance on the follower a
            # second time creates two temporary waypoints with different
            # release moments, causing post-encounter oscillation.
            self._formation_coordinated_avoidance = True
            self.avoidance_target = None

        # A passing group must copy the leader's lane-shift target exactly.
        # During ordinary walking, however, anchor the formation target close
        # to the leader.  This prevents a follower from chasing a 1.35 m target
        # that abruptly moves to the opposite side when the leader reverses.
        if leader.group_passing_active:
            anchor_x = float(leader_target[0])
            anchor_y = float(leader_target[1])
        else:
            target_dx = float(leader_target[0]) - float(leader_position[0])
            target_dy = float(leader_target[1]) - float(leader_position[1])
            target_distance = math.hypot(target_dx, target_dy)
            if target_distance > 0.10 and leader_speed > 0.08:
                anchor_step = min(self.formation_follow_lookahead, target_distance)
                anchor_x = (
                    float(leader_position[0])
                    + target_dx / target_distance * anchor_step
                )
                anchor_y = (
                    float(leader_position[1])
                    + target_dy / target_distance * anchor_step
                )
            else:
                anchor_x = float(leader_position[0])
                anchor_y = float(leader_position[1])
        delta_longitudinal = self.formation_longitudinal - leader.formation_longitudinal
        delta_lateral = self.formation_lateral - leader.formation_lateral
        z = float(self._person.state.position[2])

        # Keep one stable full-width slot.  Trying several compression scales
        # every tick made a pair expand/contract around the distance threshold,
        # which appeared as repeated body turns.  The leader already validates
        # the complete formation footprint against static geometry; group-outsider
        # avoidance is still applied later by _target_with_separation().
        target = [
            anchor_x + forward_x * delta_longitudinal
            + lateral_x * delta_lateral,
            anchor_y + forward_y * delta_longitudinal
            + lateral_y * delta_lateral,
            z,
        ]
        target_is_geometry_valid = (
            (self.polygon is None or point_is_walkable(
                target[0], target[1], self.polygon, self.obstacle_aabbs
            ))
            and self._segment_clear(self._current_xy(), target)
        )
        if not target_is_geometry_valid:
            return [
                float(self._person.state.position[0]),
                float(self._person.state.position[1]),
                z,
            ], 0.0, bool(
                leader.group_passing_active or leader_avoidance_active
            )

        current = self._current_xy()
        error = math.hypot(target[0] - current[0], target[1] - current[1])
        slot_x = (
            float(leader_position[0]) + forward_x * delta_longitudinal
            + lateral_x * delta_lateral
        )
        slot_y = (
            float(leader_position[1]) + forward_y * delta_longitudinal
            + lateral_y * delta_lateral
        )
        slot_error = math.hypot(slot_x - current[0], slot_y - current[1])
        if leader_speed <= 0.08 and error <= self.formation_deadband:
            speed = 0.0
        else:
            # Recover a displaced member decisively while the leader is
            # waiting. The previous +0.28 m/s cap allowed a sequence of two
            # encounters to stretch triples into a several-metre line.
            catchup = min(0.75, max(0.0, slot_error - 0.18) * 0.90)
            speed = max(
                self.turn_min_speed,
                min(leader.base_speed + 0.75, leader_speed + catchup),
            )
        return target, speed, bool(
            leader.group_passing_active or leader_avoidance_active
        )

    def _dense_slot_follower_command(self):
        """Track a shared route phase without copying leader pose drift."""
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is None or leader_item[1] is self:
            return None
        _leader_person, leader = leader_item
        if not self.waypoints or not leader.waypoints:
            return None

        # The leader contributes the route segment and progress projected onto
        # its canonical line.  Its lateral AnimationGraph error is discarded;
        # this member's target is reconstructed on its own absolute slot.
        self.target_index = max(
            0, min(int(leader.target_index), len(self.waypoints) - 1)
        )
        self.route_direction = 1 if int(leader.route_direction) >= 0 else -1
        self.path = []
        self.has_active_target = False
        self.fallback_active = False

        start_index = self.target_index - self.route_direction
        start_index = max(0, min(start_index, len(self.waypoints) - 1))
        leader_target_index = max(
            0, min(int(leader.target_index), len(leader.waypoints) - 1)
        )
        leader_start_index = leader_target_index - self.route_direction
        leader_start_index = max(
            0, min(leader_start_index, len(leader.waypoints) - 1)
        )
        leader_start = leader.waypoints[leader_start_index]
        leader_end = leader.waypoints[leader_target_index]
        segment_x = float(leader_end[0]) - float(leader_start[0])
        segment_y = float(leader_end[1]) - float(leader_start[1])
        segment_length_sq = segment_x * segment_x + segment_y * segment_y
        leader_position = leader._current_xy()
        if segment_length_sq <= 1e-8:
            progress = 1.0
            segment_length = 1e-4
        else:
            progress = (
                (leader_position[0] - float(leader_start[0])) * segment_x
                + (leader_position[1] - float(leader_start[1])) * segment_y
            ) / segment_length_sq
            progress = max(0.0, min(1.0, progress))
            segment_length = math.sqrt(segment_length_sq)

        own_start = self.waypoints[start_index]
        own_end = self.waypoints[self.target_index]
        slot_x = float(own_start[0]) + (
            float(own_end[0]) - float(own_start[0])
        ) * progress
        slot_y = float(own_start[1]) + (
            float(own_end[1]) - float(own_start[1])
        ) * progress

        # Every member evaluates the same seeded pause phase.  This branch is
        # reached before _planned_pause_command() in update(), so it must keep
        # followers synchronized with a pausing leader explicitly.
        pause_active = False
        if self.planned_walk_sec > 0.0 and self.planned_pause_sec > 0.0:
            cycle = self.planned_walk_sec + self.planned_pause_sec
            phase = (
                self.traffic_schedule_elapsed + self.planned_pause_phase_sec
            ) % cycle
            pause_active = phase >= self.planned_walk_sec
        self.planned_pause_active = pause_active
        leader_speed = float(
            getattr(leader, "_last_commanded_speed", leader.current_speed) or 0.0
        )
        current = self._current_xyz()
        if pause_active or leader_speed <= 0.08:
            slot_target = [slot_x, slot_y, current[2]]
            slot_error = math.hypot(slot_x - current[0], slot_y - current[1])
            correction_speed = min(0.38, max(0.0, slot_error - 0.10) * 0.8)
            if correction_speed < 0.08:
                slot_target = list(current)
                correction_speed = 0.0
            self._last_commanded_target = list(slot_target)
            self._last_commanded_speed = correction_speed
            return slot_target, correction_speed, False

        lookahead_progress = min(
            1.0, progress + self.formation_follow_lookahead / segment_length
        )
        target = [
            float(own_start[0])
            + (float(own_end[0]) - float(own_start[0])) * lookahead_progress,
            float(own_start[1])
            + (float(own_end[1]) - float(own_start[1])) * lookahead_progress,
            current[2],
        ]
        own_segment_x = float(own_end[0]) - float(own_start[0])
        own_segment_y = float(own_end[1]) - float(own_start[1])
        own_length = max(math.hypot(own_segment_x, own_segment_y), 1e-6)
        along_error = (
            (slot_x - current[0]) * own_segment_x
            + (slot_y - current[1]) * own_segment_y
        ) / own_length
        catchup = max(-0.35, min(0.55, along_error * 0.75))
        follower_speed = max(
            self.turn_min_speed,
            min(self.base_speed + 0.55, leader_speed + catchup),
        )
        self._last_commanded_target = list(target)
        self._last_commanded_speed = follower_speed
        return target, follower_speed, False

    def _leader_group_speed_limit(self, speed):
        """Slow the leader until displaced followers regain their slots."""
        if self.traffic_axis == "dense_x_flow":
            return speed
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is None or leader_item[1] is not self:
            return speed
        axis = self._formation_axis()
        leader_position = getattr(getattr(self._person, "state", None), "position", None)
        if axis is None or leader_position is None:
            return speed
        forward_x, forward_y = axis
        lateral_x, lateral_y = -forward_y, forward_x
        worst_error = 0.0
        for person, controller in members:
            if controller is self:
                continue
            position = getattr(getattr(person, "state", None), "position", None)
            if position is None:
                continue
            desired_x = float(leader_position[0]) + forward_x * (
                controller.formation_longitudinal - self.formation_longitudinal
            ) + lateral_x * (controller.formation_lateral - self.formation_lateral)
            desired_y = float(leader_position[1]) + forward_y * (
                controller.formation_longitudinal - self.formation_longitudinal
            ) + lateral_y * (controller.formation_lateral - self.formation_lateral)
            worst_error = max(
                worst_error,
                math.hypot(float(position[0]) - desired_x, float(position[1]) - desired_y),
            )
        # Treat the formation as a cohesive unit: wait as soon as a member is
        # clearly displaced, not only after a multi-metre split has formed.
        if worst_error >= 0.75:
            return 0.0
        if worst_error >= 0.50:
            return min(speed, 0.24)
        if worst_error >= 0.30:
            return min(speed, max(0.32, speed * 0.45))
        return speed

    def _reset_deadlock_watchdog(self):
        self.deadlock_anchor = None
        self.deadlock_elapsed = 0.0
        self.deadlock_escape_target = None
        self.deadlock_escape_elapsed = 0.0
        self.deadlock_escape_cooldown = 0.0

    def _traffic_schedule_command(self):
        """Yield at a preplanned crossing when another group arrives first."""
        self.traffic_waiting = False
        self.traffic_waiting_gate = None
        self.traffic_waiting_opponent = None
        self.traffic_cycle_override = False
        if not self.traffic_axis or not self.traffic_gates:
            return None

        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if (
            self.traffic_axis != "dense_x_flow"
            and leader_item is not None
            and leader_item[1] is not self
        ):
            return None

        axis = self._formation_axis()
        if axis is None:
            return None
        current = self._current_xy()
        own_center = self._group_center(members) if members else current
        if own_center is None:
            own_center = current
        coordinate = own_center[0] * axis[0] + own_center[1] * axis[1]
        direction = 1.0 if self.route_direction >= 0 else -1.0
        own_group = getattr(self, "crowd_group_id", None)
        if own_group is None:
            return None

        # Closely spaced crossings form one continuous conflict corridor.
        # Once a formation has entered any protected interval, it owns the
        # corridor until it has cleared every overlapping interval. Stopping
        # again at the next pairwise gate is what previously produced cyclic
        # waits such as G4->G5->G6->G11->G4.
        inside_any_gate = any(
            len(gate) >= 2
            and abs(float(gate[0]) - coordinate) <= float(gate[1])
            for gate in self.traffic_gates
        )
        if inside_any_gate:
            self.traffic_yield_lock = None
            return None

        nominal_speed = max(0.55, float(self.base_speed))
        braking_distance = (
            nominal_speed * nominal_speed
            / (2.0 * max(self.acceleration_limit, 0.1))
            + 0.65
        )
        best_yield = None

        for gate in self.traffic_gates:
            if len(gate) < 6:
                continue
            (
                gate_coordinate,
                radius,
                opponent_group,
                cross_x,
                cross_y,
                opponent_radius,
            ) = gate[:6]
            locked_here = (
                self.traffic_yield_lock is not None
                and abs(
                    float(self.traffic_yield_lock[0])
                    - float(gate_coordinate)
                ) < 1e-4
                and int(self.traffic_yield_lock[1])
                == int(opponent_group)
            )
            signed_distance = (gate_coordinate - coordinate) * direction
            entry_distance = signed_distance - radius
            if entry_distance < -0.05:
                if locked_here:
                    self.traffic_yield_lock = None
                continue
            if entry_distance > braking_distance:
                continue

            opponent_members = self._all_group_members(int(opponent_group))
            opponent_center = self._group_center(opponent_members)
            opponent_leader_item = self._social_group_leader(opponent_members)
            if opponent_center is None or opponent_leader_item is None:
                if locked_here:
                    self.traffic_yield_lock = None
                continue
            _opponent_person, opponent_controller = opponent_leader_item
            opponent_axis = opponent_controller._formation_axis()
            if opponent_axis is None:
                if locked_here:
                    self.traffic_yield_lock = None
                continue
            opponent_cross_coordinate = (
                cross_x * opponent_axis[0] + cross_y * opponent_axis[1]
            )
            opponent_coordinate = (
                opponent_center[0] * opponent_axis[0]
                + opponent_center[1] * opponent_axis[1]
            )
            opponent_direction = (
                1.0 if opponent_controller.route_direction >= 0 else -1.0
            )
            opponent_signed = (
                opponent_cross_coordinate - opponent_coordinate
            ) * opponent_direction
            if opponent_signed < -opponent_radius - 0.55:
                if locked_here:
                    self.traffic_yield_lock = None
                continue

            opponent_entry = max(0.0, opponent_signed - opponent_radius)
            opponent_speed = float(
                getattr(opponent_controller, "_last_commanded_speed", 0.0)
                or 0.0
            )
            opponent_inside = (
                abs(opponent_cross_coordinate - opponent_coordinate)
                <= opponent_radius
            )
            waiting_opponent = getattr(
                opponent_controller, "traffic_waiting_opponent", None
            )
            opponent_is_waiting_for_me = (
                bool(getattr(opponent_controller, "traffic_waiting", False))
                and waiting_opponent is not None
                and int(waiting_opponent) == int(own_group)
            )
            opponent_approach_limit = max(
                4.5,
                opponent_speed * opponent_speed
                / (2.0 * max(self.acceleration_limit, 0.1))
                + 1.0,
            )
            if not opponent_inside and opponent_entry > opponent_approach_limit:
                continue

            if locked_here:
                # Once yielding begins, keep the reservation until the named
                # group has crossed the far edge of this intersection.
                must_yield = True
            elif opponent_inside:
                # Occupancy overrides rank: never enter a corridor somebody is
                # already clearing.
                must_yield = True
            elif opponent_is_waiting_for_me:
                must_yield = False
            else:
                # Normal case remains first-arrival scheduling. Group rank is
                # used only by the explicit cycle breaker below.
                own_eta = max(0.0, entry_distance) / nominal_speed
                opponent_eta = (
                    opponent_entry / opponent_speed
                    if opponent_speed > 0.12
                    else float("inf")
                )
                if abs(own_eta - opponent_eta) <= 0.80:
                    must_yield = int(own_group) > int(opponent_group)
                else:
                    must_yield = own_eta > opponent_eta
            if not must_yield:
                continue
            candidate = (
                entry_distance,
                gate_coordinate,
                int(opponent_group),
                bool(opponent_inside),
            )
            candidate_key = (
                0 if candidate[3] else 1,
                candidate[0],
            )
            best_key = (
                None
                if best_yield is None
                else (0 if best_yield[3] else 1, best_yield[0])
            )
            if best_yield is None or candidate_key < best_key:
                best_yield = candidate

        if best_yield is None:
            return None

        (
            entry_distance,
            gate_coordinate,
            opponent_group,
            opponent_inside,
        ) = best_yield
        wait_cycle = (
            ()
            if opponent_inside
            else self._traffic_wait_cycle(opponent_group)
        )
        if wait_cycle and int(own_group) == min(wait_cycle):
            # ETA remains the normal policy. Only a real dependency cycle is
            # overridden, and exactly one group (the smallest id in that
            # cycle) is released, so the remaining wait graph becomes a chain.
            self.traffic_yield_lock = None
            self.traffic_cycle_override = True
            signature = tuple(sorted(wait_cycle))
            now = time.monotonic()
            if (
                signature != self.traffic_cycle_log_signature
                or now - self.traffic_cycle_log_wall_time >= 5.0
            ):
                self.traffic_cycle_log_signature = signature
                self.traffic_cycle_log_wall_time = now
                print(
                    f"[CROWD][V2][CYCLE] wait cycle={wait_cycle}; "
                    f"group={int(own_group)} gets temporary priority."
                )
            return None

        self.traffic_waiting = True
        self.traffic_waiting_gate = float(gate_coordinate)
        self.traffic_waiting_opponent = int(opponent_group)
        self.traffic_yield_lock = (
            float(gate_coordinate),
            int(opponent_group),
        )
        z = float(self._person.state.position[2])
        remaining = max(0.0, entry_distance - 0.25)
        if remaining <= 0.12:
            target = [current[0], current[1], z]
            speed = 0.0
        else:
            step = min(0.65, remaining)
            target = [current[0], current[1], z]
            target[0] += axis[0] * direction * step
            target[1] += axis[1] * direction * step
            speed = min(nominal_speed, max(0.18, remaining * 0.55))
        self.deadlock_anchor = current
        self.deadlock_elapsed = 0.0
        self._last_commanded_target = list(target)
        self._last_commanded_speed = float(speed)
        return target, speed

    def _traffic_wait_cycle(self, prospective_opponent):
        """Return the wait cycle closed by this prospective edge, if any."""
        own_group = getattr(self, "crowd_group_id", None)
        if own_group is None or prospective_opponent is None:
            return ()
        own_group = int(own_group)
        edges = {own_group: int(prospective_opponent)}
        leaders = {}
        for person in getattr(self, "crowd_people", []):
            controller = getattr(person, "_controller", None)
            if not isinstance(controller, NaturalFormationWaypointController):
                continue
            group_id = getattr(controller, "crowd_group_id", None)
            if group_id is None:
                continue
            group_id = int(group_id)
            if group_id == own_group:
                continue
            key = (
                abs(float(getattr(controller, "formation_lateral", 0.0))),
                int(getattr(controller, "member_index", 0)),
            )
            current = leaders.get(group_id)
            if current is None or key < current[0]:
                leaders[group_id] = (key, controller)

        for group_id, (_key, leader_controller) in leaders.items():
            opponent = getattr(
                leader_controller, "traffic_waiting_opponent", None
            )
            if (
                bool(getattr(leader_controller, "traffic_waiting", False))
                and opponent is not None
            ):
                edges[group_id] = int(opponent)

        order = []
        order_index = {}
        current = own_group
        while current in edges:
            if current in order_index:
                return tuple(order[order_index[current]:])
            order_index[current] = len(order)
            order.append(current)
            current = edges[current]
        return ()

    def _planned_pause_command(self):
        """Return a deterministic natural pause outside crossing zones."""
        self.planned_pause_active = False
        if self.traffic_cycle_override:
            return None
        if self.planned_walk_sec <= 0.0 or self.planned_pause_sec <= 0.0:
            return None
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if (
            self.traffic_axis != "dense_x_flow"
            and leader_item is not None
            and leader_item[1] is not self
        ):
            return None

        cycle = self.planned_walk_sec + self.planned_pause_sec
        phase = (
            self.traffic_schedule_elapsed + self.planned_pause_phase_sec
        ) % cycle
        if phase < self.planned_walk_sec:
            return None

        current = self._current_xy()
        own_center = self._group_center(members) if members else current
        if own_center is None:
            own_center = current
        axis = self._formation_axis()
        if axis is None:
            return None
        coordinate = own_center[0] * axis[0] + own_center[1] * axis[1]
        # Never start a sightseeing pause in an intersection or its stopping
        # approach. Finish clearing it, then use the remaining pause window.
        for gate in self.traffic_gates:
            if len(gate) >= 2 and abs(coordinate - gate[0]) <= gate[1] + 1.0:
                return None

        self.planned_pause_active = True
        self.deadlock_anchor = current
        self.deadlock_elapsed = 0.0
        target = [current[0], current[1], float(self._person.state.position[2])]
        self._last_commanded_target = list(target)
        self._last_commanded_speed = 0.0
        return target, 0.0

    def _deadlock_escape_command(self, dt):
        """Return a locked lateral escape after sustained real sim stagnation."""
        if self.traffic_axis == "dense_x_flow":
            # Formal dense routes do not intersect. Planned pauses and endpoint
            # turnarounds are valid stationary phases, not deadlocks; a lateral
            # escape would move the pedestrian into a neighbouring reserved
            # lane and create the collision it was intended to resolve.
            self._reset_deadlock_watchdog()
            return None
        current = self._current_xy()
        safe_dt = max(0.0, min(float(dt), 0.25))
        self.deadlock_escape_cooldown = max(
            0.0, self.deadlock_escape_cooldown - safe_dt
        )

        if self.deadlock_escape_target is not None:
            self.deadlock_escape_elapsed += safe_dt
            distance = math.hypot(
                float(self.deadlock_escape_target[0]) - current[0],
                float(self.deadlock_escape_target[1]) - current[1],
            )
            if distance <= 0.35 or self.deadlock_escape_elapsed >= self.deadlock_escape_duration:
                if self.avoidance_target is self.deadlock_escape_target:
                    self.avoidance_target = None
                self.deadlock_escape_target = None
                self.deadlock_escape_elapsed = 0.0
                self.deadlock_escape_cooldown = 1.5
                self.deadlock_anchor = current
                self.deadlock_elapsed = 0.0
                return None
            return self.deadlock_escape_target, 0.58

        if self.deadlock_anchor is None:
            self.deadlock_anchor = current
            return None
        if math.hypot(
            current[0] - self.deadlock_anchor[0],
            current[1] - self.deadlock_anchor[1],
        ) >= 0.22:
            self.deadlock_anchor = current
            self.deadlock_elapsed = 0.0
            return None
        if self.deadlock_escape_cooldown > 0.0:
            return None

        self.deadlock_elapsed += safe_dt
        if self.deadlock_elapsed < self.deadlock_timeout:
            return None

        active_target = getattr(self, "_last_commanded_target", None)
        if active_target is None:
            active_target = self.path[0] if self.path else None
        if active_target is not None:
            dx = float(active_target[0]) - current[0]
            dy = float(active_target[1]) - current[1]
        else:
            axis = self._formation_axis()
            if axis is None:
                return None
            direction = 1.0 if self.route_direction >= 0 else -1.0
            dx, dy = axis[0] * direction, axis[1] * direction
        length = math.hypot(dx, dy)
        if length < 0.15:
            axis = self._formation_axis()
            if axis is None:
                return None
            direction = 1.0 if self.route_direction >= 0 else -1.0
            forward = (axis[0] * direction, axis[1] * direction)
        else:
            forward = (dx / length, dy / length)
        right = (forward[1], -forward[0])
        preferred_sign = -1.0 if self.deadlock_escape_attempt % 2 else 1.0
        self.deadlock_escape_attempt += 1
        z = float(self._person.state.position[2])
        candidates = []
        # Start with side slips, then include diagonal/back-side exits for a
        # dense four-way knot where both pure side lanes are occupied.
        direction_weights = (
            (0.45, 1.0),
            (0.45, -1.0),
            (-0.35, 1.0),
            (-0.35, -1.0),
            (1.0, 0.70),
            (1.0, -0.70),
        )
        ordered_weights = (
            direction_weights
            if preferred_sign > 0.0
            else tuple(reversed(direction_weights))
        )
        for distance in (1.10, 1.50, 1.95, 2.35):
            for forward_weight, side_weight in ordered_weights:
                vx = forward[0] * forward_weight + right[0] * side_weight
                vy = forward[1] * forward_weight + right[1] * side_weight
                vector_length = max(math.hypot(vx, vy), 1e-6)
                candidates.append([
                    current[0] + vx / vector_length * distance,
                    current[1] + vy / vector_length * distance,
                    z,
                ])

        for candidate in candidates:
            if not self._is_separation_candidate_valid(candidate):
                continue
            if not self._segment_clear(current, candidate):
                continue
            self.deadlock_escape_target = candidate
            self.deadlock_escape_elapsed = 0.0
            self.deadlock_elapsed = 0.0
            self.avoidance_target = self.deadlock_escape_target
            name = getattr(self._person, "_stage_prefix", "person")
            print(
                f"[CROWD][V2][DEADLOCK_ESCAPE] {name} group="
                f"{getattr(self, 'crowd_group_id', None)} -> "
                f"({candidate[0]:.2f},{candidate[1]:.2f})."
            )
            return self.deadlock_escape_target, 0.58

        # Full social clearance can be impossible in a dense knot. Select the
        # geometry-clear candidate with the largest hard-body clearance so a
        # group can slide out without crossing a body or a static obstacle.
        physical_candidates = []
        for candidate in candidates:
            if not self._segment_clear(current, candidate):
                continue
            physical_clearance = self._deadlock_physical_clearance(candidate)
            if physical_clearance >= 0.58:
                physical_candidates.append((physical_clearance, candidate))
        if physical_candidates:
            physical_clearance, candidate = max(
                physical_candidates, key=lambda item: item[0]
            )
            self.deadlock_escape_target = candidate
            self.deadlock_escape_elapsed = 0.0
            self.deadlock_elapsed = 0.0
            self.avoidance_target = self.deadlock_escape_target
            name = getattr(self._person, "_stage_prefix", "person")
            print(
                f"[CROWD][V2][DEADLOCK_ESCAPE_HARD] {name} group="
                f"{getattr(self, 'crowd_group_id', None)}, clearance="
                f"{physical_clearance:.2f}m -> ({candidate[0]:.2f},{candidate[1]:.2f})."
            )
            return self.deadlock_escape_target, 0.58

        # Retry with the opposite side after one second rather than spinning
        # through candidates every control tick.
        self.deadlock_elapsed = max(0.0, self.deadlock_timeout - 1.0)
        return None

    def _deadlock_physical_clearance(self, leader_candidate):
        """Minimum body clearance for translating this leader/solo escape."""
        current = self._current_xy()
        delta_x = float(leader_candidate[0]) - current[0]
        delta_y = float(leader_candidate[1]) - current[1]
        own_group = getattr(self, "crowd_group_id", None)
        movers = []
        members = self._social_group_members()
        leader_item = self._social_group_leader(members)
        if leader_item is not None and leader_item[1] is self:
            for person, _controller in members:
                position = getattr(getattr(person, "state", None), "position", None)
                if position is not None:
                    start = (float(position[0]), float(position[1]))
                    movers.append((start, (start[0] + delta_x, start[1] + delta_y)))
        if not movers:
            movers = [(current, (current[0] + delta_x, current[1] + delta_y))]

        minimum = float("inf")
        for other in getattr(self, "crowd_people", []):
            controller = getattr(other, "_controller", None)
            if getattr(controller, "crowd_group_id", None) == own_group:
                continue
            position = getattr(getattr(other, "state", None), "position", None)
            if position is None:
                continue
            other_xy = (float(position[0]), float(position[1]))
            for start, end in movers:
                endpoint_distance = math.hypot(
                    end[0] - other_xy[0], end[1] - other_xy[1]
                )
                path_distance, _ = _point_segment_distance_with_param(
                    other_xy, start, end
                )
                minimum = min(minimum, endpoint_distance, path_distance)
        return minimum

    def update(self, dt: float):
        elapsed = max(0.0, float(dt))
        self.traffic_schedule_elapsed += elapsed
        self.control_elapsed += elapsed
        self.replan_cooldown_remaining = max(
            0.0, self.replan_cooldown_remaining - elapsed
        )
        if self.control_elapsed + 1e-9 < self.control_interval:
            return
        control_dt = self.control_elapsed
        self.control_elapsed = 0.0

        follower_command = self._group_follower_command()
        if follower_command is not None:
            target, speed, _group_passing = follower_command
            # A follower inherits the leader's formation-aware avoidance.
            # Independent predictive/soft avoidance is what opened a hole in
            # the group. Retain only last-resort body-overlap separation.
            target, speed = self._command_with_reactive_only(
                target, speed, emergency_only=True
            )
            self._person.update_target_position(target, speed)
            return

        traffic_command = self._traffic_schedule_command()
        if traffic_command is not None:
            target, speed = self._command_with_reactive_only(
                *traffic_command, emergency_only=True
            )
            self._person.update_target_position(target, speed)
            return

        pause_command = self._planned_pause_command()
        if pause_command is not None:
            target, speed = self._command_with_reactive_only(
                *pause_command, emergency_only=True
            )
            self._person.update_target_position(target, speed)
            return

        deadlock_command = self._deadlock_escape_command(control_dt)
        if deadlock_command is not None:
            target, speed = self._command_with_reactive_only(
                *deadlock_command, emergency_only=True
            )
            self._person.update_target_position(target, speed)
            return

        self._drop_reached_waypoints()
        if (
            not self.path
            and not self.done
            and self.replan_cooldown_remaining <= 0.0
        ):
            self._plan_next_waypoint()
        if not self.path:
            current = self._current_xyz()
            # HOLD pedestrians are still part of the crowd. Let them take a
            # slow emergency side-step when somebody enters their body radius;
            # otherwise moving walkers can accumulate around an immobile one.
            target, speed = self._target_with_separation(current, 0.32)
            if math.hypot(target[0] - current[0], target[1] - current[1]) < 0.05:
                speed = 0.0
                self._last_commanded_target = current
                self._last_commanded_speed = 0.0
            self._person.update_target_position(target, speed)
            return

        route_target = self.path[0]
        speed = self._update_smooth_speed(control_dt, route_target)
        speed = self._leader_group_speed_limit(speed)
        if self.fallback_active:
            speed = min(speed, max(0.55, self.base_speed * 0.75))
        target = self._lookahead_target(route_target)
        if self.traffic_axis:
            # The lane and every orthogonal crossing were planned before the
            # map was published. Do not let legacy predictive passing move a
            # formation into a neighbouring reserved lane. Retain only the
            # final hard-body safeguard for unexpected animation overshoot.
            target, speed = self._command_with_reactive_only(
                target, speed, emergency_only=True
            )
        else:
            passing_command = self._group_passing_command(target, speed)
            if passing_command is not None:
                target, speed = self._command_with_reactive_only(*passing_command)
            else:
                target, speed = self._target_with_separation(target, speed)
        self._person.update_target_position(target, speed)
