#!/usr/bin/env python3
"""Feed synthetic Isaac data and require an EGO PositionCommand response."""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "isaacsim" / "database"))
from ego_planner_bridge import EgoPlannerUdpBridge


def main():
    bridge = EgoPlannerUdpBridge("127.0.0.1", 15100, 15101)
    position = np.array([0.0, -8.0, 1.0])
    # Sparse corridor boundary outside the direct start-goal segment.
    ys = np.linspace(-10.0, 10.0, 100, dtype=np.float32)
    zs = np.linspace(0.0, 2.5, 16, dtype=np.float32)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    cloud = np.concatenate([
        np.column_stack((np.full(yy.size, -5.0), yy.ravel(), zz.ravel())),
        np.column_stack((np.full(yy.size, 5.0), yy.ravel(), zz.ravel())),
    ]).astype(np.float32)
    deadline = time.monotonic() + 10.0
    tick = 0
    try:
        while time.monotonic() < deadline:
            stamp = time.monotonic()
            bridge.send_clock(stamp)
            bridge.send_odometry(stamp, position, [0, 0, 0], [0, 0, 0, 1])
            if tick % 2 == 0:
                bridge.send_point_cloud(stamp, cloud)
            if tick < 30:
                bridge.send_goal(stamp, [0.0, 8.0, 1.0])
            command = bridge.poll_position_command()
            if command is not None:
                print("EGO ROS loopback passed; velocity=", command["velocity"].tolist())
                return
            tick += 1
            time.sleep(0.05)
    finally:
        bridge.close()
    raise RuntimeError("EGO did not return PositionCommand within 10 seconds")


if __name__ == "__main__":
    main()
