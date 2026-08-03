#!/usr/bin/env python3
"""Synthetic posterior/mask/reset smoke test for FactorizedRSSM."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factorized_rssm import FactorizedRSSM
from modules.latent_policy_attention import LatentPolicyAttentionConfig


def config() -> SimpleNamespace:
    return SimpleNamespace(stoch=8, deter=64, hidden=64, discrete=8, img_layers=2,
                           obs_layers=1, dyn_layers=1, blocks=4, act="SiLU", norm=True,
                           unimix_ratio=0.01, initial="learned", device="cpu")


def main() -> None:
    torch.manual_seed(3)
    b, t, n, e, a = 3, 5, 6, 32, 4
    model = FactorizedRSSM(
        config(), {"ego": e, "human": e}, a, goal_dim=8,
        latent_attention_config=LatentPolicyAttentionConfig(model_dim=96, num_heads=4),
    )
    counts = torch.tensor([[0, 2, 4, 6, 3], [6, 6, 5, 4, 2], [1, 0, 3, 5, 6]])
    human_mask = torch.arange(n)[None, None] < counts[..., None]
    human_is_first = torch.zeros(b, t, n, dtype=torch.bool)
    human_is_first[:, 0] = human_mask[:, 0]
    human_is_first[0, 3, 1] = True  # explicit slot reassignment
    is_first = torch.zeros(b, t, dtype=torch.bool); is_first[:, 0] = True
    embeds = {
        "ego": torch.randn(b, t, e),
        "human": torch.randn(b, t, n, e),
    }
    goal = torch.randn(b, t, 8)
    actions = torch.randn(b, t, a).clamp(-1, 1)
    states, aux = model.observe(
        embeds, actions, model.initial(b, n), is_first, human_mask, human_is_first,
        goal=goal,
    )
    assert states["ego"]["deter"].shape == (b, t, 64)
    assert states["human"]["deter"].shape == (b, t, n, 64)
    assert states["human"]["stoch"].shape == (b, t, n, 8, 8)
    assert aux["joint_feat"].shape == (b, t, 96)
    invalid_deter = states["human"]["deter"][~human_mask]
    invalid_stoch = states["human"]["stoch"][~human_mask]
    assert torch.count_nonzero(invalid_deter) == 0
    assert torch.count_nonzero(invalid_stoch) == 0
    assert torch.isfinite(aux["joint_feat"]).all()
    losses = model.kl_loss(aux["post_logits"], aux["prior_logits"], human_mask, free=1.0)
    assert set(losses) == {"dyn_ego", "rep_ego", "dyn_human", "rep_human"}
    assert all(torch.isfinite(value) for value in losses.values())

    actor = torch.nn.Sequential(torch.nn.Linear(96, a), torch.nn.Tanh())
    start = {branch: {key: value[:, -1] for key, value in states[branch].items()}
             for branch in ("ego", "human")}
    imag_states, imag = model.imagine(
        start, actor, horizon=7, human_mask=human_mask[:, -1], goal=goal[:, -1],
    )
    assert imag["joint_feat"].shape == (b, 7, 96)
    assert imag["action"].shape == (b, 7, a)
    assert imag_states["human"]["deter"].shape == (b, 7, n, 64)
    invalid = (~human_mask[:, -1])[:, None, :, None].expand_as(imag_states["human"]["deter"])
    assert torch.count_nonzero(imag_states["human"]["deter"][invalid]) == 0
    assert torch.isfinite(imag["joint_feat"]).all()
    print("factorized posterior+imagination smoke test passed", tuple(aux["joint_feat"].shape),
          tuple(imag["joint_feat"].shape), sorted(losses))


if __name__ == "__main__":
    main()
