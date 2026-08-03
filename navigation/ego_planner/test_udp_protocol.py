#!/usr/bin/env python3
"""Loopback smoke test for the Isaac half of the EGO UDP protocol."""

import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "isaacsim" / "database"))
from ego_planner_bridge import EgoPlannerUdpBridge


def main():
    ros_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ros_sock.bind(("127.0.0.1", 0))
    ros_port = ros_sock.getsockname()[1]
    isaac_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    isaac_sock.bind(("127.0.0.1", 0))
    isaac_port = isaac_sock.getsockname()[1]
    isaac_sock.close()
    bridge = EgoPlannerUdpBridge("127.0.0.1", ros_port, isaac_port)
    try:
        bridge.send_odometry(1.0, [1, 2, 3], [0.1, 0.2, 0.3], [0, 0, 0, 1])
        message = json.loads(ros_sock.recv(65535).decode("utf-8"))
        assert message["type"] == "odom" and message["position"] == [1.0, 2.0, 3.0]
        bridge.send_point_cloud(2.0, np.arange(9000, dtype=np.float32).reshape(-1, 3))
        assert ros_sock.recv(65535).startswith(b"EGPC")
        command = {
            "type": "position_command", "stamp": 3.0,
            "position": [1, 2, 3], "velocity": [0, 0, 0],
            "acceleration": [0, 0, 0], "yaw": 0.1, "yaw_dot": 0.0,
        }
        ros_sock.sendto(json.dumps(command).encode(), ("127.0.0.1", isaac_port))
        time.sleep(0.01)
        assert bridge.poll_position_command()["position"].tolist() == [1.0, 2.0, 3.0]
        assert bridge.command_is_fresh(0.5)
    finally:
        bridge.close()
        ros_sock.close()
    print("EGO UDP protocol smoke test passed")


if __name__ == "__main__":
    main()
