#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import carb
except ImportError:
    carb = None

from app_config import (
    MAVSDK_BODY_DOWN_SIGN,
    MAVSDK_BODY_RIGHT_SIGN,
    MAVSDK_COMMAND_HZ,
    MAVSDK_CONNECT_TIMEOUT_SEC,
    MAVSDK_HEALTH_TIMEOUT_SEC,
    MAVSDK_OFFBOARD_RETRY_SEC,
    MAVSDK_START_OFFBOARD_ON_TAKEOFF,
    MAVSDK_SYSTEM_ADDRESS,
    MAVSDK_TELEMETRY_STALE_SEC,
    MAVSDK_YAWSPEED_SIGN,
)


def _log_warn(message):
    if carb is not None:
        carb.log_warn(message)
    else:
        print(message)


class MavsdkOffboardBridge:
    """Small companion-style bridge for PX4 offboard velocity control.

    The rest of the app keeps using the existing vx_body/vy_body/vz_world/yaw_rate
    action space. This bridge converts that action to MAVSDK's body-frame
    VelocityBodyYawspeed command and caches PX4 telemetry for planner/recorder
    consumers that should avoid reading perfect simulator state.
    """

    def __init__(
        self,
        system_address=MAVSDK_SYSTEM_ADDRESS,
        command_hz=MAVSDK_COMMAND_HZ,
        connect_timeout_sec=MAVSDK_CONNECT_TIMEOUT_SEC,
        health_timeout_sec=MAVSDK_HEALTH_TIMEOUT_SEC,
    ):
        self.system_address = str(system_address)
        self.command_hz = float(command_hz)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.health_timeout_sec = float(health_timeout_sec)

        self.quit = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._takeoff_requested = False
        self._land_requested = False
        self._reset_requested = False
        self._offboard_started = False
        self._connected = False
        self._position_ok = False
        self._armable = False
        self._armed = False
        self._in_air = False
        self._armed_known = False
        self._in_air_known = False
        self._health_ok = False
        self._last_error = None
        self._thread = None
        self._latest_state = None
        self._latest_state_wall_time = None
        self._latest_command_wall_time = None
        self._requested_motion = (0.0, 0.0, 0.0, 0.0)
        self._applied_command = self._empty_command()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.quit = False
        self._stop_event.clear()
        self._reset_runtime_state()
        self._thread = threading.Thread(
            target=self._run_thread,
            name="MavsdkOffboardBridge",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self):
        self.quit = True
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _reset_runtime_state(self):
        with self._lock:
            self._takeoff_requested = False
            self._land_requested = False
            self._reset_requested = False
            self._offboard_started = False
            self._connected = False
            self._position_ok = False
            self._armable = False
            self._armed = False
            self._in_air = False
            self._armed_known = False
            self._in_air_known = False
            self._health_ok = False
            self._last_error = None
            self._latest_state = None
            self._latest_state_wall_time = None
            self._latest_command_wall_time = None
            self._requested_motion = (0.0, 0.0, 0.0, 0.0)
            self._applied_command = self._empty_command()

    def trigger_takeoff(self):
        with self._lock:
            self._takeoff_requested = True

    def trigger_land(self):
        with self._lock:
            self._land_requested = True
            self._takeoff_requested = False

    def reset_after_episode(self):
        with self._lock:
            self._requested_motion = (0.0, 0.0, 0.0, 0.0)
            self._takeoff_requested = False
            self._reset_requested = True

    def set_motion(self, vx_body, vy_body, vz_world, yaw_rate):
        motion = (
            float(vx_body),
            float(vy_body),
            float(vz_world),
            float(yaw_rate),
        )
        with self._lock:
            self._requested_motion = motion
            self._latest_command_wall_time = time.perf_counter()

    def is_offboard_started(self):
        with self._lock:
            return bool(self._offboard_started)

    def get_state(self):
        with self._lock:
            state = self._latest_state
            stamp = self._latest_state_wall_time
        if state is None or stamp is None:
            return None
        if time.perf_counter() - stamp > float(MAVSDK_TELEMETRY_STALE_SEC):
            return None
        return SimpleNamespace(
            position=state.position.copy(),
            linear_velocity=state.linear_velocity.copy(),
            linear_acceleration=state.linear_acceleration.copy(),
            attitude=state.attitude.copy(),
            angular_velocity=np.zeros(3),
            linear_body_velocity=np.zeros(3),
        )

    def snapshot(self):
        with self._lock:
            command = dict(self._applied_command)
            state = self._latest_state
            latest_state_age = None
            if self._latest_state_wall_time is not None:
                latest_state_age = max(0.0, time.perf_counter() - self._latest_state_wall_time)
            return {
                "connected": bool(self._connected),
                "position_ok": bool(self._position_ok),
                "armable": bool(self._armable),
                "armed": bool(self._armed),
                "in_air": bool(self._in_air),
                "armed_known": bool(self._armed_known),
                "in_air_known": bool(self._in_air_known),
                "health_ok": bool(self._health_ok),
                "offboard_started": bool(self._offboard_started),
                "takeoff_requested": bool(self._takeoff_requested),
                "system_address": self.system_address,
                "command_hz": float(self.command_hz),
                "latest_error": self._last_error,
                "applied": command,
                "telemetry_age_sec": latest_state_age,
                "telemetry": None if state is None else state.to_dict(),
            }

    def _run_thread(self):
        try:
            asyncio.run(self._async_main())
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            _log_warn(f"[MAVSDK] Bridge thread exited with error: {exc}")

    async def _async_main(self):
        try:
            from mavsdk import System
            from mavsdk.action import ActionError
            from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
        except Exception as exc:
            with self._lock:
                self._last_error = f"Failed to import mavsdk: {exc}"
            _log_warn(f"[MAVSDK] Failed to import mavsdk: {exc}")
            return

        drone = System()
        _log_warn(f"[MAVSDK] Connecting to PX4 via {self.system_address}")
        await drone.connect(system_address=self.system_address)
        connected = await self._wait_connected(drone)
        with self._lock:
            self._connected = bool(connected)
        if not connected:
            return

        telemetry_tasks = [
            asyncio.create_task(self._watch_position_velocity(drone)),
            asyncio.create_task(self._watch_attitude(drone)),
            asyncio.create_task(self._watch_health(drone)),
            asyncio.create_task(self._watch_armed(drone)),
            asyncio.create_task(self._watch_in_air(drone)),
        ]

        try:
            await self._command_loop(drone, VelocityBodyYawspeed, ActionError, OffboardError)
        finally:
            for task in telemetry_tasks:
                task.cancel()
            await asyncio.gather(*telemetry_tasks, return_exceptions=True)
            with self._lock:
                self._offboard_started = False
                self._connected = False

    async def _wait_connected(self, drone):
        deadline = time.perf_counter() + self.connect_timeout_sec
        while not self._stop_event.is_set() and time.perf_counter() < deadline:
            try:
                async for state in drone.core.connection_state():
                    if state.is_connected:
                        _log_warn("[MAVSDK] Connected to PX4.")
                        return True
                    break
            except Exception as exc:
                with self._lock:
                    self._last_error = f"connection_state: {exc}"
            await asyncio.sleep(0.25)
        with self._lock:
            self._last_error = "PX4 connection timeout"
        _log_warn("[MAVSDK] Timed out while waiting for PX4 connection.")
        return False

    async def _watch_health(self, drone):
        async for health in drone.telemetry.health():
            position_ok = bool(
                getattr(health, "is_global_position_ok", False)
                or getattr(health, "is_local_position_ok", False)
            )
            armable = bool(getattr(health, "is_armable", False))
            with self._lock:
                self._position_ok = position_ok
                self._armable = armable
                self._refresh_health_ok_locked()
            if self._stop_event.is_set():
                return

    async def _watch_armed(self, drone):
        async for is_armed in drone.telemetry.armed():
            with self._lock:
                self._armed = bool(is_armed)
                self._armed_known = True
                self._refresh_health_ok_locked()
            if self._stop_event.is_set():
                return

    async def _watch_in_air(self, drone):
        async for is_in_air in drone.telemetry.in_air():
            with self._lock:
                self._in_air = bool(is_in_air)
                self._in_air_known = True
            if self._stop_event.is_set():
                return

    async def _watch_position_velocity(self, drone):
        previous_position = None
        previous_time = None
        async for sample in drone.telemetry.position_velocity_ned():
            now = time.perf_counter()
            position = self._ned_position_to_enu(sample.position)
            velocity = self._ned_velocity_to_enu(sample.velocity)
            acceleration = np.zeros(3, dtype=float)
            if previous_position is not None and previous_time is not None:
                dt = max(1e-6, now - previous_time)
                acceleration = (velocity - previous_velocity) / dt
            previous_position = position.copy()
            previous_velocity = velocity.copy()
            previous_time = now
            self._update_cached_state(position=position, velocity=velocity, acceleration=acceleration)
            if self._stop_event.is_set():
                return

    async def _watch_attitude(self, drone):
        async for attitude in drone.telemetry.attitude_euler():
            yaw_enu = math.radians(90.0 - float(attitude.yaw_deg))
            quat_xyzw = Rotation.from_euler("XYZ", [0.0, 0.0, yaw_enu]).as_quat()
            self._update_cached_state(attitude=quat_xyzw)
            if self._stop_event.is_set():
                return

    async def _command_loop(self, drone, velocity_cls, action_error_cls, offboard_error_cls):
        period = 1.0 / max(self.command_hz, 1e-6)
        last_offboard_attempt = 0.0
        retry_period = max(0.25, float(MAVSDK_OFFBOARD_RETRY_SEC))
        while not self._stop_event.is_set():
            takeoff_requested, land_requested, reset_requested, motion = self._consume_flags_and_motion()

            if land_requested:
                await self._try_land(drone, action_error_cls, offboard_error_cls)

            if reset_requested:
                await self._try_reset_vehicle(drone, action_error_cls, offboard_error_cls)

            if (
                MAVSDK_START_OFFBOARD_ON_TAKEOFF
                and takeoff_requested
                and not self._offboard_started
                and time.perf_counter() - last_offboard_attempt > retry_period
            ):
                last_offboard_attempt = time.perf_counter()
                await self._try_start_offboard(drone, velocity_cls, action_error_cls, offboard_error_cls)

            if self._offboard_started:
                command = self._motion_to_mavsdk_command(motion)
                try:
                    await drone.offboard.set_velocity_body(velocity_cls(**command))
                    with self._lock:
                        self._applied_command = dict(command)
                        self._last_error = None
                except offboard_error_cls as exc:
                    with self._lock:
                        self._last_error = f"set_velocity_body: {exc}"
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"set_velocity_body: {exc}"

            await asyncio.sleep(period)

    def _consume_flags_and_motion(self):
        with self._lock:
            takeoff_requested = self._takeoff_requested
            land_requested = self._land_requested
            reset_requested = self._reset_requested
            self._land_requested = False
            self._reset_requested = False
            motion = self._requested_motion
        return takeoff_requested, land_requested, reset_requested, motion

    async def _try_start_offboard(self, drone, velocity_cls, action_error_cls, offboard_error_cls):
        if not await self._wait_health_briefly():
            with self._lock:
                self._last_error = "PX4 is not ready for offboard yet"
            _log_warn("[MAVSDK] Waiting for PX4 position/armed state before offboard start.")
            return
        zero = self._motion_to_mavsdk_command((0.0, 0.0, 0.0, 0.0))
        try:
            await drone.offboard.set_velocity_body(velocity_cls(**zero))
            try:
                await drone.action.arm()
            except action_error_cls as exc:
                _log_warn(f"[MAVSDK] Arm command was rejected; trying offboard anyway: {exc}")
            await drone.offboard.start()
            with self._lock:
                self._offboard_started = True
                self._takeoff_requested = False
                self._applied_command = dict(zero)
                self._last_error = None
            _log_warn("[MAVSDK] PX4 armed and offboard velocity control started.")
        except (action_error_cls, offboard_error_cls) as exc:
            with self._lock:
                self._offboard_started = False
                self._last_error = f"start_offboard: {exc}"
            _log_warn(f"[MAVSDK] Failed to start offboard: {exc}")
        except Exception as exc:
            with self._lock:
                self._offboard_started = False
                self._last_error = f"start_offboard: {exc}"
            _log_warn(f"[MAVSDK] Failed to start offboard: {exc}")

    async def _try_land(self, drone, action_error_cls, offboard_error_cls):
        await self._try_stop_offboard(drone, offboard_error_cls)
        try:
            await drone.action.land()
            _log_warn("[MAVSDK] Land requested.")
        except action_error_cls as exc:
            with self._lock:
                self._last_error = f"land: {exc}"
        except Exception as exc:
            with self._lock:
                self._last_error = f"land: {exc}"

    async def _try_stop_offboard(self, drone, offboard_error_cls):
        if not self._offboard_started:
            return
        try:
            await drone.offboard.stop()
            _log_warn("[MAVSDK] Offboard stopped.")
        except offboard_error_cls:
            pass
        except Exception:
            pass
        with self._lock:
            self._offboard_started = False
            self._applied_command = self._empty_command()

    async def _try_reset_vehicle(self, drone, action_error_cls, offboard_error_cls):
        await self._try_stop_offboard(drone, offboard_error_cls)
        try:
            await drone.action.disarm()
            _log_warn("[MAVSDK] Disarmed after episode reset.")
        except action_error_cls as exc:
            with self._lock:
                self._last_error = f"reset_disarm: {exc}"
            _log_warn(f"[MAVSDK] Disarm after reset was rejected: {exc}")
        except Exception as exc:
            with self._lock:
                self._last_error = f"reset_disarm: {exc}"
            _log_warn(f"[MAVSDK] Disarm after reset failed: {exc}")

    async def _wait_health_briefly(self):
        deadline = time.perf_counter() + self.health_timeout_sec
        while time.perf_counter() < deadline and not self._stop_event.is_set():
            with self._lock:
                if self._health_ok:
                    return True
            await asyncio.sleep(0.2)
        return False

    def _refresh_health_ok_locked(self):
        self._health_ok = bool(self._position_ok and (self._armable or self._armed))

    def _update_cached_state(self, position=None, velocity=None, acceleration=None, attitude=None):
        with self._lock:
            current = self._latest_state or _CachedMavsdkState()
            if position is not None:
                current.position = np.asarray(position, dtype=float)
            if velocity is not None:
                current.linear_velocity = np.asarray(velocity, dtype=float)
            if acceleration is not None:
                current.linear_acceleration = np.asarray(acceleration, dtype=float)
            if attitude is not None:
                current.attitude = np.asarray(attitude, dtype=float)
            self._latest_state = current
            self._latest_state_wall_time = time.perf_counter()

    @staticmethod
    def _ned_position_to_enu(position_ned):
        return np.array(
            [
                float(position_ned.east_m),
                float(position_ned.north_m),
                -float(position_ned.down_m),
            ],
            dtype=float,
        )

    @staticmethod
    def _ned_velocity_to_enu(velocity_ned):
        return np.array(
            [
                float(velocity_ned.east_m_s),
                float(velocity_ned.north_m_s),
                -float(velocity_ned.down_m_s),
            ],
            dtype=float,
        )

    @staticmethod
    def _empty_command():
        return {
            "forward_m_s": 0.0,
            "right_m_s": 0.0,
            "down_m_s": 0.0,
            "yawspeed_deg_s": 0.0,
        }

    @staticmethod
    def _motion_to_mavsdk_command(motion):
        vx_body, vy_body, vz_world, yaw_rate = motion
        return {
            "forward_m_s": float(vx_body),
            "right_m_s": float(MAVSDK_BODY_RIGHT_SIGN) * float(vy_body),
            "down_m_s": float(MAVSDK_BODY_DOWN_SIGN) * float(vz_world),
            "yawspeed_deg_s": float(MAVSDK_YAWSPEED_SIGN) * math.degrees(float(yaw_rate)),
        }


class _CachedMavsdkState:
    def __init__(self):
        self.position = np.zeros(3, dtype=float)
        self.linear_velocity = np.zeros(3, dtype=float)
        self.linear_acceleration = np.zeros(3, dtype=float)
        self.attitude = Rotation.identity().as_quat()

    def to_dict(self):
        roll, pitch, yaw = Rotation.from_quat(self.attitude).as_euler("XYZ", degrees=False)
        return {
            "position": self.position.astype(float).tolist(),
            "velocity": self.linear_velocity.astype(float).tolist(),
            "acceleration": self.linear_acceleration.astype(float).tolist(),
            "quaternion_xyzw": self.attitude.astype(float).tolist(),
            "roll_pitch_yaw_rad": [float(roll), float(pitch), float(yaw)],
            "roll_pitch_yaw_deg": np.degrees([roll, pitch, yaw]).astype(float).tolist(),
        }
