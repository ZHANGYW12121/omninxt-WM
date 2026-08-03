#!/usr/bin/env python3
"""Compare a saved Isaac input against the unmodified NavRL TorchRL policy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DIR = Path(
    os.environ.get(
        "NAVRL_OFFICIAL_DIR",
        str(REPO_ROOT / "navigation" / "navrl" / "quick-demos"),
    )
).expanduser().resolve()
WRAPPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OFFICIAL_DIR))
from agent import Agent  # noqa: E402
sys.path.insert(0, str(WRAPPER_DIR))
from navrl_policy import NavRLPolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()

    with np.load(args.snapshot, allow_pickle=False) as saved:
        state = torch.from_numpy(saved["state8"]).view(1, 8)
        lidar = torch.from_numpy(saved["lidar36x4"]).view(1, 1, 36, 4)
        dynamic = torch.from_numpy(saved["dynamic5x10"]).view(1, 1, 5, 10)
        direction = torch.from_numpy(saved["goal_direction_world"]).view(1, 1, 3)
        saved_normalized = saved["normalized_action"].astype(np.float64)
        saved_local = saved["local_action"].astype(np.float64)
        state_np = saved["state8"].copy()
        lidar_np = saved["lidar36x4"].copy()
        dynamic_np = saved["dynamic5x10"].copy()
        direction_np = saved["goal_direction_world"].copy()

    agent = Agent("cpu")
    observation = TensorDict({
        "agents": TensorDict({
            "observation": TensorDict({
                "state": state,
                "lidar": lidar,
                "direction": direction,
                "dynamic_obstacle": dynamic,
            })
        })
    }, device="cpu")
    with torch.inference_mode(), set_exploration_type(ExplorationType.MEAN):
        output = agent.policy(observation)

    alpha_tensor = output["alpha"]
    beta_tensor = output["beta"]
    normalized_tensor = output["agents", "action_normalized"]
    official_alpha = alpha_tensor.reshape(-1, 3)[0].detach().cpu().numpy()
    official_beta = beta_tensor.reshape(-1, 3)[0].detach().cpu().numpy()
    official_normalized = normalized_tensor.reshape(-1, 3)[0].detach().cpu().numpy()
    # Official ROS deployment uses vel_limit=1.0 even though quick-demo PPO
    # computes its world action with the training value 2.0.
    official_local_deploy = 2.0 * official_normalized - 1.0

    asset_root = Path(
        os.environ.get("OMNINXT_ASSET_ROOT", REPO_ROOT / ".local" / "assets")
    ).expanduser().resolve()
    checkpoint = Path(
        os.environ.get(
            "NAVRL_CHECKPOINT",
            asset_root / "models" / "navrl" / "navrl_checkpoint.pt",
        )
    ).expanduser().resolve()
    wrapper = NavRLPolicy(str(checkpoint), "cpu")
    wrapper.infer(state_np, lidar_np, dynamic_np, direction_np)
    wrapped_normalized = wrapper.last_normalized_action
    wrapped_local = wrapper.last_local_velocity

    print(f"snapshot: {args.snapshot}")
    print(f"official_shapes: alpha={tuple(alpha_tensor.shape)}, beta={tuple(beta_tensor.shape)}, action={tuple(normalized_tensor.shape)}")
    print(f"official_alpha: {official_alpha}")
    print(f"official_beta: {official_beta}")
    print(f"official_normalized_mean: {official_normalized}")
    print(f"wrapper_normalized_mean:  {wrapped_normalized}")
    print(f"normalized_abs_diff:     {np.abs(official_normalized - wrapped_normalized)}")
    print(f"official_local_deploy:   {official_local_deploy}")
    print(f"wrapper_local_action:    {wrapped_local}")
    print(f"local_abs_diff:          {np.abs(official_local_deploy - wrapped_local)}")
    print(f"max_normalized_abs_diff: {np.max(np.abs(official_normalized - wrapped_normalized)):.9g}")
    print(f"saved_old_normalized:    {saved_normalized}")
    print(f"saved_old_local_action:  {saved_local}")


if __name__ == "__main__":
    main()
