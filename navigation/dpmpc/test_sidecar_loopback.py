#!/usr/bin/env python3
"""CPU-only DPMPC UDP/ACADO loopback smoke test."""

import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = REPO_ROOT / "simulation" / "isaacsim" / "database"
sys.path.insert(0, str(DATABASE_DIR))

from dpmpc_bridge import DpmpcUdpBridge  # noqa: E402


def main():
    bridge = DpmpcUdpBridge()
    try:
        stamp = 1.0
        bridge.send_goal(stamp, [5.0, 0.0, 1.0])
        # One far-away occupied voxel is sufficient for the protocol test.
        # Unknown space is intentionally ignored by the official planner.
        bridge.send_static_cloud(
            stamp, np.asarray([[100.0, 100.0, 100.0]], dtype=np.float32)
        )
        deadline = time.monotonic() + 30.0
        command = None
        while time.monotonic() < deadline:
            observation_id = bridge.send_observation(
                stamp,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [{
                    "id": 0,
                    "position": [2.5, 1.0, 1.1],
                    "velocity": [-0.1, 0.0, 0.0],
                    "size": [0.5, 0.5, 2.2],
                    "position_variance": [1e-4, 1e-4, 1e-4],
                    "velocity_variance": [1e-4, 1e-4, 1e-4],
                }],
            )
            wait_deadline = min(deadline, time.monotonic() + 0.25)
            while time.monotonic() < wait_deadline:
                command = bridge.poll_command()
                if (
                    command is not None
                    and command["observation_id"] == observation_id
                ):
                    break
                time.sleep(0.01)
            if command is not None:
                break
            stamp += 0.1
        if command is None:
            raise RuntimeError("no DPMPC command received within 30 seconds")
        if not np.all(np.isfinite(command["position"])):
            raise RuntimeError(f"non-finite command: {command}")
        print(
            "DPMPC loopback OK:",
            f"observation_id={command['observation_id']}",
            f"solver_status={command['solver_status']}",
            f"solve_time_ms={command['solve_time_ms']:.3f}",
            f"position={command['position'].tolist()}",
        )
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
