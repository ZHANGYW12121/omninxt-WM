#!/usr/bin/env python3
"""Shape/mask smoke test for slot-preserving interaction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.interaction import InteractionConfig, MaskedHumanAttentionPool, SlotPreservingInteraction


def main() -> None:
    torch.manual_seed(0)
    b, n, d = 6, 18, 64
    counts = torch.tensor([0, 8, 10, 13, 15, 18])
    mask = torch.arange(n)[None] < counts[:, None]
    ego, env = torch.randn(b, 1, d), torch.randn(b, 1, d)
    humans = torch.randn(b, n, d)
    positions = torch.randn(b, n, 3)
    model = SlotPreservingInteraction(InteractionConfig(model_dim=d, num_heads=4))
    pool = MaskedHumanAttentionPool(d, num_heads=4)

    out = model(ego, env, humans, mask, positions)
    pooled = pool(out["human_context"], mask)
    assert out["ego_context"].shape == (b, 1, d)
    assert out["env_context"].shape == (b, 1, d)
    assert out["human_context"].shape == (b, n, d)
    assert pooled.shape == (b, d)
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(out["human_context"][~mask]) == 0

    # Invalid slots must not affect valid contexts or pooled policy input.
    changed = humans.clone()
    changed[~mask] = 1.0e6
    out_changed = model(ego, env, changed, mask, positions)
    pooled_changed = pool(out_changed["human_context"], mask)
    torch.testing.assert_close(out["ego_context"], out_changed["ego_context"])
    torch.testing.assert_close(out["env_context"], out_changed["env_context"])
    torch.testing.assert_close(pooled, pooled_changed)
    print("interaction smoke test passed", {"counts": counts.tolist(), "pooled": tuple(pooled.shape)})


if __name__ == "__main__":
    main()
