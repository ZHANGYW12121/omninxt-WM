#!/usr/bin/env python3
"""Numerically audit the lightweight NavRL wrapper against official TorchRL."""

from __future__ import annotations

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
ASSET_ROOT = Path(
    os.environ.get("OMNINXT_ASSET_ROOT", REPO_ROOT / ".local" / "assets")
).expanduser().resolve()
CHECKPOINT = Path(
    os.environ.get(
        "NAVRL_CHECKPOINT", ASSET_ROOT / "models" / "navrl" / "navrl_checkpoint.pt"
    )
).expanduser().resolve()
sys.path.insert(0, str(OFFICIAL_DIR))
from agent import Agent  # noqa: E402
from utils import vec_to_world  # noqa: E402
sys.path.insert(0, str(WRAPPER_DIR))
from navrl_policy import NavRLPolicy  # noqa: E402


def official_forward(agent, state, lidar, dynamic, direction):
    td = TensorDict({
        "agents": TensorDict({"observation": TensorDict({
            "state": torch.from_numpy(state).view(1, 8),
            "lidar": torch.from_numpy(lidar).view(1, 1, 36, 4),
            "direction": torch.from_numpy(direction).view(1, 1, 3),
            "dynamic_obstacle": torch.from_numpy(dynamic).view(1, 1, 5, 10),
        })})
    }, device="cpu")
    with torch.inference_mode(), set_exploration_type(ExplorationType.MEAN):
        out = agent.policy(td)
    alpha = out["alpha"].reshape(-1, 3)[0]
    beta = out["beta"].reshape(-1, 3)[0]
    normalized = out["agents", "action_normalized"].reshape(-1, 3)[0]
    local_deploy = 2.0 * normalized - 1.0
    world_deploy = vec_to_world(
        local_deploy.view(1, 1, 3), torch.from_numpy(direction).view(1, 1, 3)
    ).reshape(3)
    return tuple(x.detach().cpu().numpy() for x in (alpha, beta, normalized, world_deploy))


def main() -> None:
    torch.manual_seed(1234)
    rng = np.random.default_rng(1234)
    official = Agent("cpu")
    wrapper = NavRLPolicy(str(CHECKPOINT), "cpu")

    official_state = official.policy.state_dict()
    wrapper_state = wrapper.net.state_dict()
    weight_max = 0.0
    mapped_tensors = 0
    for old_prefix, new_prefix in wrapper._KEY_MAP.items():
        for suffix in ("weight", "bias"):
            old_key = f"{old_prefix}.{suffix}"
            new_key = f"{new_prefix}.{suffix}"
            if old_key not in official_state or new_key not in wrapper_state:
                raise RuntimeError(f"Missing mapping: {old_key} -> {new_key}")
            weight_max = max(
                weight_max,
                float(torch.max(torch.abs(official_state[old_key] - wrapper_state[new_key]))),
            )
            mapped_tensors += 1

    maxima = {"alpha": 0.0, "beta": 0.0, "normalized": 0.0, "world": 0.0}
    cases = 256
    for _ in range(cases):
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        state = rng.uniform(-1.0, 1.0, 8).astype(np.float32)
        state[3] = rng.uniform(0.0, 45.0)
        state[4] = rng.uniform(-3.0, 3.0)
        state[5:] = rng.uniform(-2.0, 2.0, 3)
        lidar = rng.uniform(0.0, 4.0, (36, 4)).astype(np.float32)
        dynamic = np.zeros((5, 10), dtype=np.float32)
        count = int(rng.integers(0, 6))
        if count:
            dynamic[:count, :3] = rng.uniform(-1.0, 1.0, (count, 3))
            norms = np.linalg.norm(dynamic[:count, :3], axis=1, keepdims=True)
            dynamic[:count, :3] /= np.maximum(norms, 1e-6)
            dynamic[:count, 3] = rng.uniform(0.0, 4.0, count)
            dynamic[:count, 4] = rng.uniform(-2.0, 2.0, count)
            dynamic[:count, 5:8] = rng.uniform(-2.0, 2.0, (count, 3))
            dynamic[:count, 8] = rng.integers(0, 4, count)
            dynamic[:count, 9] = rng.integers(0, 2, count)

        oa, ob, on, ow = official_forward(official, state, lidar, dynamic, direction)
        ww = wrapper.infer(state, lidar, dynamic, direction)
        maxima["alpha"] = max(maxima["alpha"], float(np.max(np.abs(oa - wrapper.last_alpha))))
        maxima["beta"] = max(maxima["beta"], float(np.max(np.abs(ob - wrapper.last_beta))))
        maxima["normalized"] = max(maxima["normalized"], float(np.max(np.abs(on - wrapper.last_normalized_action))))
        maxima["world"] = max(maxima["world"], float(np.max(np.abs(ow - ww))))

    print(f"mapped_parameter_tensors={mapped_tensors}")
    print(f"parameter_max_abs_diff={weight_max:.9g}")
    print(f"random_cases={cases}")
    for key, value in maxima.items():
        print(f"{key}_max_abs_diff={value:.9g}")


if __name__ == "__main__":
    main()
