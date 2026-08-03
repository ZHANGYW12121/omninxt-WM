#!/usr/bin/env python3
"""UDP transport between Isaac Sim and the official DPMPC ROS sidecar."""

import json
import math
import socket
import struct
import time
import zlib

import numpy as np


_CLOUD_MAGIC = b"DMPC"
# stamp, sequence, uncompressed point count, chunk index, chunk count
_CLOUD_HEADER = struct.Struct("!dIIHH")
_MAX_CHUNK_BYTES = 58000


def _finite_vector(value, size):
    try:
        array = np.asarray(value, dtype=float).reshape(size)
    except Exception:
        return None
    return array if np.all(np.isfinite(array)) else None


class DpmpcUdpBridge:
    """Dependency-free transport; ROS/ACADO never enter Isaac's Python process."""

    def __init__(self, host="127.0.0.1", planner_port=15200, isaac_port=15201):
        self.remote = (str(host), int(planner_port))
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", int(isaac_port)))
        self.rx.setblocking(False)
        self._cloud_sequence = 0
        self._observation_sequence = 0
        self.latest_command = None
        self.latest_command_wall_time = None

    def close(self):
        self.tx.close()
        self.rx.close()

    def reset_after_episode(self, reason="reset"):
        self._send_json({"type": "reset", "reason": str(reason)})
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
        self.tx.sendto(
            json.dumps(message, separators=(",", ":")).encode("utf-8"),
            self.remote,
        )

    def send_clock(self, stamp):
        self._send_json({"type": "clock", "stamp": float(stamp)})

    def send_goal(self, stamp, position, yaw=0.0):
        self._send_json({
            "type": "goal",
            "stamp": float(stamp),
            "position": np.asarray(position, dtype=float).reshape(3).tolist(),
            "yaw": float(yaw),
        })

    def send_static_cloud(self, stamp, points_xyz):
        points = np.asarray(points_xyz, dtype="<f4").reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        compressed = zlib.compress(points.tobytes(order="C"), level=1)
        chunk_count = max(
            1,
            (len(compressed) + _MAX_CHUNK_BYTES - 1) // _MAX_CHUNK_BYTES,
        )
        self._cloud_sequence = (self._cloud_sequence + 1) & 0xFFFFFFFF
        for chunk_index in range(chunk_count):
            part = compressed[
                chunk_index * _MAX_CHUNK_BYTES:
                (chunk_index + 1) * _MAX_CHUNK_BYTES
            ]
            header = _CLOUD_MAGIC + _CLOUD_HEADER.pack(
                float(stamp),
                self._cloud_sequence,
                int(len(points)),
                chunk_index,
                chunk_count,
            )
            self.tx.sendto(header + part, self.remote)
        return self._cloud_sequence

    def send_observation(
        self,
        stamp,
        position,
        velocity,
        quaternion_xyzw,
        obstacles,
    ):
        self._observation_sequence = (
            self._observation_sequence + 1
        ) & 0xFFFFFFFF
        self._send_json({
            "type": "observation",
            "stamp": float(stamp),
            "observation_id": self._observation_sequence,
            "position": np.asarray(position, dtype=float).reshape(3).tolist(),
            "velocity": np.asarray(velocity, dtype=float).reshape(3).tolist(),
            "quaternion_xyzw": (
                np.asarray(quaternion_xyzw, dtype=float).reshape(4).tolist()
            ),
            "obstacles": list(obstacles),
        })
        return self._observation_sequence

    def poll_command(self):
        while True:
            try:
                payload, _ = self.rx.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                message = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            if message.get("type") != "dpmpc_command":
                continue
            position = _finite_vector(message.get("position"), 3)
            velocity = _finite_vector(message.get("velocity", [0, 0, 0]), 3)
            acceleration = _finite_vector(
                message.get("acceleration", [0, 0, 0]), 3
            )
            try:
                yaw = float(message.get("yaw", 0.0))
                yaw_dot = float(message.get("yaw_dot", 0.0))
                stamp = float(message.get("stamp", 0.0))
                observation_id = int(message.get("observation_id", -1))
                solver_status = int(message.get("solver_status", -1))
                solve_time_ms = float(message.get("solve_time_ms", float("nan")))
            except (TypeError, ValueError):
                continue
            if position is None or velocity is None or acceleration is None:
                continue
            if not math.isfinite(yaw) or not math.isfinite(yaw_dot):
                continue
            self.latest_command = {
                "stamp": stamp,
                "observation_id": observation_id,
                "solver_status": solver_status,
                "solve_time_ms": solve_time_ms,
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
        if simulation_now is not None:
            try:
                age = float(simulation_now) - float(self.latest_command["stamp"])
                if age >= -float(timeout_sec):
                    return age <= float(timeout_sec)
            except (TypeError, ValueError):
                pass
        return (
            self.latest_command_wall_time is not None
            and time.monotonic() - self.latest_command_wall_time
            <= float(timeout_sec)
        )
