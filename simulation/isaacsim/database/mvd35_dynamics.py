#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.thrusters.thrust_curve import ThrustCurve


class FirstOrderQuadraticThrustCurve(ThrustCurve):
    """Quadratic rotor thrust with first-order motor response.

    Pegasus consumes rotor angular velocity references in rad/s and applies
    forces as ``T = k_f * omega^2``. This class keeps that contract while adding
    separate up/down motor time constants for the MVD35 sim2real estimate.
    """

    def __init__(self, config=None):
        config = dict(config or {})
        self._num_rotors = int(config.get("num_rotors", 4))

        self._rotor_constant = _list_param(
            config.get("rotor_constant", [8.54858e-6] * self._num_rotors),
            self._num_rotors,
            "rotor_constant",
        )
        self._rolling_moment_coefficient = _list_param(
            config.get("rolling_moment_coefficient", [1.0e-6] * self._num_rotors),
            self._num_rotors,
            "rolling_moment_coefficient",
        )
        self._rot_dir = _list_param(
            config.get("rot_dir", [-1, -1, 1, 1]),
            self._num_rotors,
            "rot_dir",
        )
        self.min_rotor_velocity = _list_param(
            config.get("min_rotor_velocity", [0.0] * self._num_rotors),
            self._num_rotors,
            "min_rotor_velocity",
        )
        self.max_rotor_velocity = _list_param(
            config.get("max_rotor_velocity", [1100.0] * self._num_rotors),
            self._num_rotors,
            "max_rotor_velocity",
        )
        self._time_constant_up = _list_param(
            config.get("time_constant_up", [0.0] * self._num_rotors),
            self._num_rotors,
            "time_constant_up",
        )
        self._time_constant_down = _list_param(
            config.get("time_constant_down", [0.0] * self._num_rotors),
            self._num_rotors,
            "time_constant_down",
        )

        self._input_reference = [0.0 for _ in range(self._num_rotors)]
        self._velocity = [0.0 for _ in range(self._num_rotors)]
        self._force = [0.0 for _ in range(self._num_rotors)]
        self._rolling_moment = 0.0

    def set_input_reference(self, input_reference):
        self._input_reference = list(input_reference)

    def update(self, state: State, dt: float):
        rolling_moment = 0.0
        dt = max(float(dt), 0.0)

        for i in range(self._num_rotors):
            target = float(
                np.clip(
                    self._input_reference[i],
                    self.min_rotor_velocity[i],
                    self.max_rotor_velocity[i],
                )
            )
            current = float(self._velocity[i])
            tau = (
                self._time_constant_up[i]
                if target > current
                else self._time_constant_down[i]
            )
            if tau > 1.0e-9 and dt > 0.0:
                alpha = 1.0 - np.exp(-dt / tau)
                current += float(alpha) * (target - current)
            else:
                current = target

            current = float(
                np.clip(current, self.min_rotor_velocity[i], self.max_rotor_velocity[i])
            )
            self._velocity[i] = current
            self._force[i] = self._rotor_constant[i] * current * current
            rolling_moment += (
                self._rolling_moment_coefficient[i]
                * current
                * current
                * self._rot_dir[i]
            )

        self._rolling_moment = float(rolling_moment)
        return self._force, self._velocity, self._rolling_moment

    @property
    def force(self):
        return self._force

    @property
    def velocity(self):
        return self._velocity

    @property
    def rolling_moment(self):
        return self._rolling_moment

    @property
    def rot_dir(self):
        return self._rot_dir


def _list_param(value, expected_len, name):
    values = list(value)
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}")
    return values
