#!/usr/bin/env python3
"""Offline world-model pretraining on recorded Isaac/OmniDepth sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.factorized_schema import FactorizedIsaacAdapter, compute_ego_state_statistics
from datasets.isaac_crowd import IsaacCrowdSequenceDataset
from factorized_agent import FactorizedDreamerAgent


def numeric_collate(items):
    keys = set.intersection(*(set(item) for item in items))
    result = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], np.ndarray) and values[0].dtype.kind in "biuf":
            result[key] = torch.from_numpy(np.stack(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stats-sequences", type=int, default=256)
    args = parser.parse_args()

    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise RuntimeError("Install the repository Hydra environment before training") from exc
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="configs", overrides=[
            "env=isaac_uav_human", "model=factorized_dreamer", f"device={args.device}",
            f"model.device={args.device}", f"model.rssm.device={args.device}",
        ])

    base = IsaacCrowdSequenceDataset(
        args.data_root, sequence_length=args.sequence_length, load_lidar=False,
        include_paths=False,
        missing_reward_policy="error",
    )
    dataset = FactorizedIsaacAdapter(
        base, args.pose_root, max_people=int(config.model.factorized.max_people),
        require_pose_cache=True, strict_altitude=True,
    )
    stats_count = min(len(dataset), max(1, args.stats_sequences))
    ego_mean, ego_std = compute_ego_state_statistics(
        [dataset[index]["ego_state"] for index in range(stats_count)]
    )
    from omegaconf import open_dict
    with open_dict(config.model.factorized):
        config.model.factorized.ego_state_mean = ego_mean.tolist()
        config.model.factorized.ego_state_std = ego_std.tolist()
    sample = dataset[0]
    obs_space = SimpleNamespace(spaces={
        "goal": SimpleNamespace(shape=sample["goal"].shape[1:]),
    })
    act_space = SimpleNamespace(shape=sample["action"].shape[1:])
    agent = FactorizedDreamerAgent(config.model, obs_space, act_space).to(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True, collate_fn=numeric_collate)
    iterator = iter(loader)
    output = Path(args.output).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
        metrics = agent._train_step(batch)
        agent._scheduler.step()
        if step % 100 == 0:
            summary = " ".join(f"{k}={v:.4g}" for k, v in metrics.items() if k.startswith("loss/"))
            print(f"step={step} {summary}", flush=True)
        if step % 5000 == 0 or step == args.steps:
            torch.save({"agent_state_dict": agent.state_dict(),
                        "factorized_training_state": agent.training_state_dict(),
                        "step": step}, output)


if __name__ == "__main__": main()
