#!/usr/bin/env python3
"""Print the raw NavRL inference and safety result saved by diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()

    with np.load(args.snapshot, allow_pickle=False) as data:
        label = str(data["label"])
        local = data["local_action"]
        policy_world = data["policy_world_action"]
        final_xy = data["final_world_xy"]
        alpha = data["beta_alpha"]
        beta = data["beta_beta"]
        normalized = data["normalized_action"]
        safety_delta = final_xy - policy_world[:2]

        print(f"snapshot: {args.snapshot}")
        print(f"event: {label}")
        print(f"position: {data['position_world']}")
        print(f"goal: {data['goal_world']}")
        print(f"goal_distance: {float(data['goal_distance']):.3f} m")
        print(f"lidar_nearest: {float(data['lidar_nearest']):.3f} m")
        print(f"people_in: {int(data['people_in'])}")
        print(f"beta_alpha: {alpha}")
        print(f"beta_beta: {beta}")
        print(f"normalized_mean: {normalized}")
        print(f"network_local_action: {local}")
        print(f"network_world_action: {policy_world}")
        print(f"final_world_xy: {final_xy}")
        print(f"safety_delta_xy: {safety_delta}")
        print(f"network_xy_speed: {np.linalg.norm(local[:2]):.4f} m/s")
        print(f"final_xy_speed: {np.linalg.norm(final_xy):.4f} m/s")


if __name__ == "__main__":
    main()
