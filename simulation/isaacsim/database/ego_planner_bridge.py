#!/usr/bin/env python3
"""Dependency-free UDP transport between Isaac Sim and the ROS1 sidecar."""

import json
import math
import socket
import struct
import time
import zlib

import numpy as np


_CLOUD_MAGIC = b"EGPC"
_CLOUD_HEADER = struct.Struct("!dIHH")
_MAX_CHUNK_BYTES = 58000


def _finite_vector(value, size):
    try:
        array = np.asarray(value, dtype=float).reshape(size)
    except Exception:
        return None
    return array if np.all(np.isfinite(array)) else None


class EgoPlannerUdpBridge:
    def __init__(self, host="127.0.0.1", ros_port=15100, isaac_port=15101):
        self.remote = (str(host), int(ros_port))
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", int(isaac_port)))
        self.rx.setblocking(False)
        self._cloud_sequence = 0
        self.latest_command = None
        self.latest_command_wall_time = None

    def close(self):
        self.tx.close()
        self.rx.close()

    def reset_after_episode(self):
        self.latest_command = None
        self.latest_command_wall_time = None
        while True:
            try:
                self.rx.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break

    def _send_json(self, message):
        self.tx.sendto(json.dumps(message, separators=(",", ":")).encode("utf-8"), self.remote)

    def send_odometry(self, stamp, position, velocity, quaternion_xyzw):
        self._send_json({
            "type": "odom",
            "stamp": float(stamp),
            "position": np.asarray(position, dtype=float).tolist(),
            "velocity": np.asarray(velocity, dtype=float).tolist(),
            "quaternion_xyzw": np.asarray(quaternion_xyzw, dtype=float).tolist(),
        })

    def send_clock(self, stamp):
        self._send_json({
            "type": "clock",
            "stamp": float(stamp),
        })

    def send_goal(self, stamp, position, yaw=0.0):
        self._send_json({
            "type": "goal",
            "stamp": float(stamp),
            "position": np.asarray(position, dtype=float).tolist(),
            "yaw": float(yaw),
        })

    def send_mission_complete(self, stamp, position, speed_xy):
        self._send_json({
            "type": "mission_complete",
            "stamp": float(stamp),
            "position": np.asarray(position, dtype=float).tolist(),
            "speed_xy": float(speed_xy),
        })

    def send_point_cloud(self, stamp, points_xyz):
        points = np.asarray(points_xyz, dtype="<f4").reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        compressed = zlib.compress(points.tobytes(order="C"), level=1)
        count = max(1, (len(compressed) + _MAX_CHUNK_BYTES - 1) // _MAX_CHUNK_BYTES)
        self._cloud_sequence = (self._cloud_sequence + 1) & 0xFFFFFFFF
        for index in range(count):
            part = compressed[index * _MAX_CHUNK_BYTES:(index + 1) * _MAX_CHUNK_BYTES]
            header = _CLOUD_MAGIC + _CLOUD_HEADER.pack(
                float(stamp), self._cloud_sequence, index, count
            )
            self.tx.sendto(header + part, self.remote)

    def poll_position_command(self):
        while True:
            try:
                payload, _ = self.rx.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                message = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            if message.get("type") != "position_command":
                continue
            position = _finite_vector(message.get("position"), 3)
            velocity = _finite_vector(message.get("velocity"), 3)
            acceleration = _finite_vector(message.get("acceleration", [0, 0, 0]), 3)
            yaw = float(message.get("yaw", 0.0))
            yaw_dot = float(message.get("yaw_dot", 0.0))
            if position is None or velocity is None or acceleration is None:
                continue
            if not math.isfinite(yaw) or not math.isfinite(yaw_dot):
                continue
            self.latest_command = {
                "stamp": float(message.get("stamp", 0.0)),
                "trajectory_id": int(message.get("trajectory_id", -1)),
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
                "yaw": yaw,
                "yaw_dot": yaw_dot,
            }
            self.latest_command_wall_time = time.monotonic()
        return self.latest_command

    def command_is_fresh(self, timeout_sec, simulation_now=None):
        if self.latest_command is None:
            return False
        # PositionCommand stamps now use Isaac /clock. Compare in simulation
        # time so a low real-time factor cannot make a valid held command look
        # stale between two simulation control ticks.
        if simulation_now is not None:
            try:
                command_stamp = float(self.latest_command.get("stamp", 0.0))
                age = float(simulation_now) - command_stamp
                if command_stamp > 0.0 and age >= -float(timeout_sec):
                    return age <= float(timeout_sec)
            except (TypeError, ValueError):
                pass
        return (
            self.latest_command_wall_time is not None
            and time.monotonic() - self.latest_command_wall_time <= float(timeout_sec)
        )
