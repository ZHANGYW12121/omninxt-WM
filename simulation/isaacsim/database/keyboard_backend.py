#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State


class SharedCommand:
    def __init__(self):
        self.lock = threading.Lock()

        # 机体系速度指令
        self.vx_body = 0.0   # 前后
        self.vy_body = 0.0   # 左右
        self.vz_world = 0.0  # 上下（世界系，向上为正）
        self.yaw_rate = 0.0  # 偏航角速度，rad/s

        self.takeoff = False
        self.land = False
        self.kill = False

    def set_motion(self, vx_body, vy_body, vz_world, yaw_rate):
        with self.lock:
            self.vx_body = vx_body
            self.vy_body = vy_body
            self.vz_world = vz_world
            self.yaw_rate = yaw_rate

    def reset(self):
        with self.lock:
            self.vx_body = 0.0
            self.vy_body = 0.0
            self.vz_world = 0.0
            self.yaw_rate = 0.0
            self.takeoff = False
            self.land = False
            self.kill = False

    def get_motion(self):
        with self.lock:
            return self.vx_body, self.vy_body, self.vz_world, self.yaw_rate

    def trigger_takeoff(self):
        with self.lock:
            self.takeoff = True

    def consume_takeoff(self):
        with self.lock:
            f = self.takeoff
            self.takeoff = False
            return f

    def trigger_land(self):
        with self.lock:
            self.land = True

    def consume_land(self):
        with self.lock:
            f = self.land
            self.land = False
            return f

    def trigger_kill(self):
        with self.lock:
            self.kill = True

    def consume_kill(self):
        with self.lock:
            f = self.kill
            self.kill = False
            return f


class KeyboardVelocityController(Backend):
    def __init__(
        self,
        shared_cmd: SharedCommand,
        hover_height: float = 1.5,
        kp_pos=(0.55, 0.55, 6.0),
        kd_vel=(3.8, 3.8, 4.8),
        kr=(1.05, 1.05, 0.62),
        kw=(0.42, 0.42, 0.40),
        yaw_rate_deadband: float = 0.02,
        yaw_settle_rate: float = 0.05,
        xy_vel_slew_rate: float = 7.0,
        z_vel_slew_rate: float = 3.0,
        max_tilt_deg: float = 24.0,
        mass_kg: float = 1.50,
    ):
        self.shared_cmd = shared_cmd

        self.input_ref = [0.0, 0.0, 0.0, 0.0]

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.R = Rotation.identity()

        self.received_first_state = False

        self.m = float(mass_kg)
        self.g = 9.81

        self.Kp = np.diag(kp_pos)
        self.Kd = np.diag(kd_vel)
        self.Kr = np.diag(kr)
        self.Kw = np.diag(kw)

        self.hover_height = hover_height
        self.flight_mode = "idle"
        self.p_ref = None
        self.yaw_ref = 0.0
        self.yaw_rate_deadband = yaw_rate_deadband
        self.yaw_settle_rate = yaw_settle_rate
        self.xy_vel_slew_rate = xy_vel_slew_rate
        self.z_vel_slew_rate = z_vel_slew_rate
        self.max_tilt_rad = np.deg2rad(max_tilt_deg)
        self.v_cmd_body = np.zeros(3)
        self.applied_v_ref_world = np.zeros(3)
        self.applied_yaw_rate = 0.0
        self._yaw_command_active = False

    def start(self):
        self.flight_mode = "idle"
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self.p_ref = None
        self.v_cmd_body[:] = 0.0
        self.applied_v_ref_world[:] = 0.0
        self.applied_yaw_rate = 0.0
        self._yaw_command_active = False

    def stop(self):
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self.v_cmd_body[:] = 0.0
        self.applied_v_ref_world[:] = 0.0
        self.applied_yaw_rate = 0.0
        self._yaw_command_active = False

    def reset(self):
        self.flight_mode = "idle"
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.R = Rotation.identity()
        self.received_first_state = False
        self.p_ref = None
        self.yaw_ref = 0.0
        self.v_cmd_body[:] = 0.0
        self.applied_v_ref_world[:] = 0.0
        self.applied_yaw_rate = 0.0
        self._yaw_command_active = False

    def get_applied_motion(self):
        """Return the slew-limited velocity reference used by the backend."""
        return {
            "vx_body_mps": float(self.v_cmd_body[0]),
            "vy_body_mps": float(self.v_cmd_body[1]),
            "vz_world_mps": float(self.v_cmd_body[2]),
            "yaw_rate_rps": float(self.applied_yaw_rate),
            "velocity_reference_world_mps": self.applied_v_ref_world.astype(float).tolist(),
            "flight_mode": str(self.flight_mode),
        }

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state: State):
        self.p = state.position
        self.R = Rotation.from_quat(state.attitude)
        self.w = state.angular_velocity
        self.v = state.linear_velocity
        self.received_first_state = True

    def input_reference(self):
        return self.input_ref

    @staticmethod
    def vee(S):
        return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

    @staticmethod
    def wrap_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def slew_vector(current, target, max_delta):
        delta = target - current
        norm = np.linalg.norm(delta)
        if norm <= max_delta or norm < 1e-9:
            return target.copy()
        return current + delta * (max_delta / norm)

    @staticmethod
    def slew_scalar(current, target, max_delta):
        delta = target - current
        if abs(delta) <= max_delta:
            return target
        return current + np.sign(delta) * max_delta

    def update(self, dt: float):
        if not self.received_first_state:
            return

        if self.shared_cmd.consume_kill():
            self.flight_mode = "idle"
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            self.v_cmd_body[:] = 0.0
            self.applied_v_ref_world[:] = 0.0
            self.applied_yaw_rate = 0.0
            self._yaw_command_active = False
            return

        if self.p_ref is None:
            self.p_ref = self.p.copy()
            euler = self.R.as_euler("XYZ", degrees=False)
            self.yaw_ref = euler[2]

        if self.shared_cmd.consume_takeoff():
            self.flight_mode = "hover"
            self.p_ref = self.p.copy()
            self.p_ref[2] = self.hover_height
            self.yaw_ref = self.R.as_euler("XYZ", degrees=False)[2]
            self.v_cmd_body[:] = 0.0
            self._yaw_command_active = False

        if self.shared_cmd.consume_land():
            self.flight_mode = "land"

        vx_body, vy_body, vz_world, yaw_rate = self.shared_cmd.get_motion()

        if self.flight_mode == "idle":
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            self.v_cmd_body[:] = 0.0
            self.applied_v_ref_world[:] = 0.0
            self.applied_yaw_rate = 0.0
            self._yaw_command_active = False
            return

        if self.flight_mode == "land":
            vx_body = 0.0
            vy_body = 0.0
            vz_world = -0.5
            if self.p[2] < 0.15:
                self.flight_mode = "idle"
                self.input_ref = [0.0, 0.0, 0.0, 0.0]
                self.v_cmd_body[:] = 0.0
                self.applied_v_ref_world[:] = 0.0
                self.applied_yaw_rate = 0.0
                self._yaw_command_active = False
                return

        target_v_cmd_body = np.array([vx_body, vy_body, vz_world])
        self.v_cmd_body[:2] = self.slew_vector(
            self.v_cmd_body[:2],
            target_v_cmd_body[:2],
            self.xy_vel_slew_rate * dt,
        )
        self.v_cmd_body[2] = self.slew_scalar(
            self.v_cmd_body[2],
            target_v_cmd_body[2],
            self.z_vel_slew_rate * dt,
        )
        vx_body, vy_body, vz_world = self.v_cmd_body

        R_mat = self.R.as_matrix()
        x_b = R_mat[:, 0]
        y_b = R_mat[:, 1]

        # 机体系速度 -> 世界系速度参考
        v_ref_xy = vx_body * x_b + vy_body * y_b
        v_ref = np.array([v_ref_xy[0], v_ref_xy[1], vz_world])

        # ===== 关键修改：按住键时，让位置参考持续向前推进 =====
        # 这样无人机追踪的是“移动中的参考点”，而不是固定点
        self.p_ref[0] += v_ref[0] * dt
        self.p_ref[1] += v_ref[1] * dt
        self.p_ref[2] += vz_world * dt

        # 松开平移键时，锁定当前位置，进入原地悬停
        if (
            self.flight_mode == "hover"
            and np.linalg.norm(target_v_cmd_body[:2]) < 1e-4
            and np.linalg.norm(self.v_cmd_body[:2]) < 1e-3
        ):
            self.p_ref[0] = self.p[0]
            self.p_ref[1] = self.p[1]

        # 右摇杆按偏航速率控制；回中后先刹停，再锁住最终朝向。
        current_yaw = self.R.as_euler("XYZ", degrees=False)[2]
        yaw_command_active = abs(yaw_rate) > self.yaw_rate_deadband
        if yaw_command_active:
            self.yaw_ref = self.wrap_pi(self.yaw_ref + yaw_rate * dt)
        else:
            yaw_rate = 0.0
            if self._yaw_command_active or abs(self.w[2]) > self.yaw_settle_rate:
                self.yaw_ref = current_yaw
        self._yaw_command_active = yaw_command_active
        self.applied_v_ref_world[:] = v_ref
        self.applied_yaw_rate = float(yaw_rate)

        ep = self.p - self.p_ref
        ev = self.v - v_ref

        F_des = -(self.Kp @ ep) - (self.Kd @ ev) + np.array([0.0, 0.0, self.m * self.g])
        F_xy_norm = np.linalg.norm(F_des[:2])
        max_F_xy = max(abs(F_des[2]) * np.tan(self.max_tilt_rad), 1e-6)
        if F_xy_norm > max_F_xy:
            F_des[:2] *= max_F_xy / F_xy_norm

        Z_b = R_mat[:, 2]
        u_1 = F_des @ Z_b

        Z_b_des = F_des / np.linalg.norm(F_des)
        X_c_des = np.array([np.cos(self.yaw_ref), np.sin(self.yaw_ref), 0.0])

        Y_b_des = np.cross(Z_b_des, X_c_des)
        Y_b_des = Y_b_des / np.linalg.norm(Y_b_des)
        X_b_des = np.cross(Y_b_des, Z_b_des)

        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        e_R = 0.5 * self.vee((R_des.T @ R_mat) - (R_mat.T @ R_des))

        w_des = np.array([0.0, 0.0, yaw_rate])
        e_w = self.w - w_des

        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

        if self.vehicle is not None:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)
