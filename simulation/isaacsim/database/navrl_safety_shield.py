"""Dependency-light planar port of NavRL's dynamic-obstacle ORCA shield."""

from __future__ import annotations

import math

import numpy as np


class NavRLSafetyShield:
    """Project a preferred XY velocity outside pedestrian velocity obstacles.

    NavRL's flight demo calls the C++ ``safe_action`` service after policy
    inference.  The Warehouse test is altitude-held, so the same dynamic ORCA
    construction is evaluated in the horizontal plane here.  This keeps the
    shield independent of ROS and of the PX4/local command backend.
    """

    def __init__(self, time_horizon, time_step, safety_distance, agent_radius,
                 obstacle_radius, max_speed):
        self.time_horizon = max(1e-3, float(time_horizon))
        self.time_step = max(1e-3, float(time_step))
        self.safety_distance = max(0.0, float(safety_distance))
        self.agent_radius = max(0.0, float(agent_radius))
        self.obstacle_radius = max(0.0, float(obstacle_radius))
        self.max_speed = max(1e-3, float(max_speed))

    def apply(self, agent_position, preferred_velocity, obstacles):
        preferred = np.asarray(preferred_velocity, dtype=float)[:2]
        constraints = []
        active_count = 0

        for obstacle in obstacles:
            obs_position = np.asarray(obstacle["position"], dtype=float)[:2]
            obs_velocity = np.asarray(obstacle["velocity"], dtype=float)[:2]
            plane = self._orca_half_plane(
                np.asarray(agent_position, dtype=float)[:2], preferred,
                obs_position, obs_velocity,
            )
            if plane is None:
                continue
            point, normal, in_velocity_obstacle = plane
            constraints.append((point, normal))
            active_count += int(in_velocity_obstacle)

        if active_count == 0:
            return self._limit_speed(preferred), 0

        safe = self._limit_speed(preferred)
        # Alternating projections solve the small convex half-plane problem
        # while preserving the velocity closest to the policy preference.
        for _ in range(max(4, 2 * len(constraints))):
            changed = False
            for point, normal in constraints:
                violation = float(np.dot(normal, point - safe))
                if violation > 1e-7:
                    safe = self._limit_speed(safe + violation * normal)
                    changed = True
            if not changed:
                break
        return safe, active_count

    def _orca_half_plane(self, agent_position, preferred_velocity,
                         obstacle_position, obstacle_velocity):
        relative_position = obstacle_position - agent_position
        relative_velocity = preferred_velocity - obstacle_velocity
        distance_sq = float(np.dot(relative_position, relative_position))
        combined_radius = (
            self.agent_radius + self.safety_distance + self.obstacle_radius
        )
        combined_radius_sq = combined_radius * combined_radius

        if distance_sq < 1e-12:
            normal = np.array([-1.0, 0.0])
            correction = combined_radius / self.time_step
            return preferred_velocity + correction * normal, normal, True

        in_vo = self._will_collide(
            relative_position, relative_velocity, combined_radius
        )

        if distance_sq <= combined_radius_sq:
            inverse_step = 1.0 / self.time_step
            w = relative_velocity - inverse_step * relative_position
            normal = self._unit(w, fallback=-relative_position)
            correction = (combined_radius * inverse_step - np.linalg.norm(w)) * normal
            return preferred_velocity + correction, normal, True

        inverse_horizon = 1.0 / self.time_horizon
        w = relative_velocity - inverse_horizon * relative_position
        w_length_sq = float(np.dot(w, w))
        dot_product = float(np.dot(w, relative_position))

        if dot_product < 0.0 and dot_product * dot_product > combined_radius_sq * w_length_sq:
            normal = self._unit(w, fallback=-relative_position)
            correction = (combined_radius * inverse_horizon - math.sqrt(w_length_sq)) * normal
        else:
            cross_value = (
                relative_position[0] * relative_velocity[1]
                - relative_position[1] * relative_velocity[0]
            )
            denominator = max(distance_sq - combined_radius_sq, 1e-9)
            a = distance_sq
            b = float(np.dot(relative_position, relative_velocity))
            c = float(np.dot(relative_velocity, relative_velocity)) - cross_value * cross_value / denominator
            discriminant = max(0.0, b * b - a * c)
            t = max(0.0, (b + math.sqrt(discriminant)) / a)
            cone_w = relative_velocity - t * relative_position
            normal = self._unit(cone_w, fallback=-relative_position)
            correction = (combined_radius * t - np.linalg.norm(cone_w)) * normal

        return preferred_velocity + correction, normal, in_vo

    def _will_collide(self, relative_position, relative_velocity, radius):
        speed_sq = float(np.dot(relative_velocity, relative_velocity))
        if speed_sq < 1e-12:
            return float(np.linalg.norm(relative_position)) <= radius
        closest_time = float(np.clip(
            np.dot(relative_position, relative_velocity) / speed_sq,
            0.0,
            self.time_horizon,
        ))
        separation = relative_position - closest_time * relative_velocity
        return float(np.dot(separation, separation)) <= radius * radius

    def _limit_speed(self, velocity):
        velocity = np.asarray(velocity, dtype=float)[:2]
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        return velocity.copy()

    @staticmethod
    def _unit(vector, fallback):
        vector = np.asarray(vector, dtype=float)[:2]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            vector = np.asarray(fallback, dtype=float)[:2]
            norm = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-9)
