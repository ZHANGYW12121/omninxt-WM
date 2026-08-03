#!/usr/bin/env python3
"""Small real-module forward/backward/imagination smoke benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factorized_dreamer import FactorizedDreamer
from factorized_rssm import FactorizedRSSM
from modules.factorized_encoders import FactorizedEncoderConfig, FactorizedObservationEncoder
from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads
from modules.latent_policy_attention import LatentPolicyAttentionConfig


class NormalHead(nn.Module):
    def __init__(self, inp, out=1): super().__init__(); self.net = nn.Linear(inp, out)
    def forward(self, x):
        mean = self.net(x); return Independent(Normal(mean, torch.ones_like(mean)), 1)


class ContinueHead(nn.Module):
    def __init__(self, inp): super().__init__(); self.net = nn.Linear(inp, 1)
    def forward(self, x): return Independent(Bernoulli(logits=self.net(x)), 1)


def rssm_config(device):
    return SimpleNamespace(stoch=8, deter=64, hidden=64, discrete=8, img_layers=1,
        obs_layers=1, dyn_layers=1, blocks=4, act="SiLU", norm=True,
        unimix_ratio=0.01, initial="learned", device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    b, t, n, j, d, action_dim = args.batch, args.length, 6, 17, 32, 4
    encoder = FactorizedObservationEncoder(FactorizedEncoderConfig(
        model_dim=d, human_hidden_dim=16, pose_history=4,
    ))
    dynamics = FactorizedRSSM(
        rssm_config(str(device)), {"ego": d, "human": d}, action_dim, goal_dim=8,
        latent_attention_config=LatentPolicyAttentionConfig(model_dim=d, num_heads=4),
    )
    actor = NormalHead(d, action_dim)
    heads = FactorizedPredictionHeads(
        dynamics.ego_rssm.feat_size, dynamics.human_rssm.feat_size,
        FactorizedPredictionConfig(hidden_dim=64),
    )
    model = FactorizedDreamer(
        encoder, dynamics, actor, NormalHead(d), NormalHead(d), ContinueHead(d), heads,
        kl_free=0.0,
    ).to(device)
    mask = torch.arange(n, device=device)[None, None] < torch.tensor([4], device=device)[:, None, None]
    mask = mask.expand(b, t, n).clone()
    joint_mask = mask[..., None].expand(b, t, n, j).clone()
    batch = {
        "ego_state": torch.randn(b, t, 14, device=device),
        "skeleton": torch.randn(b, t, n, j, 7, device=device),
        "human_mask": mask, "joint_mask": joint_mask,
        "human_is_first": torch.zeros(b, t, n, dtype=torch.bool, device=device),
        "action": torch.randn(b, t, action_dim, device=device),
        "reward": torch.randn(b, t, 1, device=device),
        "is_first": torch.zeros(b, t, dtype=torch.bool, device=device),
        "is_last": torch.zeros(b, t, 1, device=device),
        "is_terminal": torch.zeros(b, t, 1, device=device),
        "goal": torch.randn(b, t, 8, device=device),
        "goal_position": torch.randn(b, t, 3, device=device),
    }
    batch["is_first"][:, 0] = True; batch["human_is_first"][:, 0] = mask[:, 0]
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses, states, aux = model.world_model_loss(batch)
    pred_losses, _ = model.prediction_loss(states, batch)
    total = sum(losses.values()) + sum(pred_losses.values())
    total.backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    with torch.no_grad():
        model.imagine(
            model.last_state(states), 5, mask[:, -1],
            goal_position=batch["goal_position"][:, -1],
        )
    if device.type == "cuda": torch.cuda.synchronize(device)
    imagine_ms = (time.perf_counter() - start) * 1000
    result = {
        "device": str(device), "parameters": sum(x.numel() for x in model.parameters()),
        "forward_backward_ms": train_ms, "imagination_h5_ms": imagine_ms,
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "state_shapes": {branch: {key: list(value.shape) for key, value in values.items()}
                         for branch, values in states.items()},
        "finite": bool(torch.isfinite(total)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
