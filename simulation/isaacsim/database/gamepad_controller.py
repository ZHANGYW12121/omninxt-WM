#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carb
import omni.appwindow
import carb.input
from carb.input import GamepadInput

from app_config import VX, VY, VZ, YAW_RATE


def _debug_log(*args, **kwargs):
    pass


class GamepadController:
    def __init__(self, shared_cmd, toggle_recording_callback=None):
        self.cmd = shared_cmd
        self.toggle_recording_callback = toggle_recording_callback
        self.quit = False

        self.left_up = 0.0
        self.left_down = 0.0
        self.left_left = 0.0
        self.left_right = 0.0
        self.right_up = 0.0
        self.right_down = 0.0
        self.right_left = 0.0
        self.right_right = 0.0
        self.x_pressed = False
        self.deadzone = 0.12

        app_window = omni.appwindow.get_default_app_window()
        self.gamepad = app_window.get_gamepad(0)
        self.input_iface = carb.input.acquire_input_interface()

        if self.gamepad is None:
            _debug_log("[GAMEPAD] No gamepad found at index 0.")
            self.gp_sub = None
        else:
            self.gp_sub = self.input_iface.subscribe_to_gamepad_events(
                self.gamepad, self._on_gamepad_event
            )
            _debug_log("[GAMEPAD] Gamepad connected at index 0.")

        _debug_log(
            "\n"
            "================= Gamepad Control =================\n"
            "A      : takeoff / hover\n"
            "B      : land\n"
            "X      : start / stop dataset recording\n"
            "\n"
            "Left Stick  : translate x/y\n"
            "Right Stick : yaw / z\n"
            "===================================================\n"
        )

    def shutdown(self):
        if self.gp_sub is not None and self.gamepad is not None:
            self.input_iface.unsubscribe_to_gamepad_events(self.gamepad, self.gp_sub)
            self.gp_sub = None

    def _apply_deadzone(self, v: float):
        return 0.0 if abs(v) < self.deadzone else v

    def _on_gamepad_event(self, event):
        inp = event.input
        val = event.value

        if inp == GamepadInput.A:
            if val > 0.5:
                self.cmd.trigger_takeoff()
        elif inp == GamepadInput.B:
            if val > 0.5:
                self.cmd.trigger_land()
        elif inp == GamepadInput.X:
            pressed = val > 0.5
            if pressed and not self.x_pressed and self.toggle_recording_callback is not None:
                self.toggle_recording_callback()
            self.x_pressed = pressed
        elif inp == GamepadInput.LEFT_STICK_UP:
            self.left_up = self._apply_deadzone(val)
        elif hasattr(GamepadInput, "LEFT_STICK_DOWN") and inp == GamepadInput.LEFT_STICK_DOWN:
            self.left_down = self._apply_deadzone(val)
        elif inp == GamepadInput.LEFT_STICK_RIGHT:
            self.left_right = self._apply_deadzone(val)
        elif hasattr(GamepadInput, "LEFT_STICK_LEFT") and inp == GamepadInput.LEFT_STICK_LEFT:
            self.left_left = self._apply_deadzone(val)
        elif inp == GamepadInput.RIGHT_STICK_RIGHT:
            self.right_right = self._apply_deadzone(val)
        elif hasattr(GamepadInput, "RIGHT_STICK_LEFT") and inp == GamepadInput.RIGHT_STICK_LEFT:
            self.right_left = self._apply_deadzone(val)
        elif inp == GamepadInput.RIGHT_STICK_UP:
            self.right_up = self._apply_deadzone(val)
        elif hasattr(GamepadInput, "RIGHT_STICK_DOWN") and inp == GamepadInput.RIGHT_STICK_DOWN:
            self.right_down = self._apply_deadzone(val)

        self._update_motion()
        return True

    def _update_motion(self):
        left_y = self.left_up - self.left_down
        left_x = self.left_left - self.left_right
        right_y = self.right_up - self.right_down
        right_x = self.right_left - self.right_right

        vx_body = left_y * VX
        vy_body = left_x * VY
        vz_world = right_y * VZ
        yaw_rate = right_x * YAW_RATE

        self.cmd.set_motion(vx_body, vy_body, vz_world, yaw_rate)
