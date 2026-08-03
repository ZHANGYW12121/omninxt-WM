from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factorized_rssm import FactorizedRSSM
from modules.latent_policy_attention import LatentPolicyAttentionConfig
from modules.sparse_ego_human_attention import SparseEgoHumanAttentionConfig


def rssm_config():
    return SimpleNamespace(stoch=4, deter=32, hidden=32, discrete=4, img_layers=1,
                           obs_layers=1, dyn_layers=1, blocks=4, act="SiLU", norm=True,
                           unimix_ratio=0.01, initial="learned", device="cpu")


def make_model(embed_dim=16, action_dim=3):
    return FactorizedRSSM(
        rssm_config(), {"ego": embed_dim, "human": embed_dim}, action_dim, goal_dim=8,
        latent_attention_config=LatentPolicyAttentionConfig(model_dim=48, num_heads=4),
        coupling_attention_config=SparseEgoHumanAttentionConfig(model_dim=32, num_heads=4),
    )


class FactorizedRSSMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_two_branches_are_independent_and_padding_isolated(self):
        model = make_model()
        ego_params = {id(p) for p in model.ego_rssm.parameters()}
        human_params = {id(p) for p in model.human_rssm.parameters()}
        self.assertFalse(ego_params & human_params)
        b, n = 3, 5
        mask = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        state = model.initial(b, n)
        goal = torch.randn(b, 8)
        joint_a, _ = model.get_joint_feat(state, mask, goal)
        corrupt = {branch: {key: value.clone() for key, value in values.items()}
                   for branch, values in state.items()}
        corrupt["human"]["deter"][~mask] = 1.0e6
        corrupt["human"]["stoch"][~mask] = 1.0e6
        joint_b, _ = model.get_joint_feat(corrupt, mask, goal)
        torch.testing.assert_close(joint_a, joint_b)
        self.assertNotIn("env", state)

    def test_human_slot_reset_blocks_old_identity(self):
        model = make_model()
        b, n, e, a = 2, 3, 16, 3
        base = model.initial(b, n)
        stale = {branch: {key: value.clone() for key, value in values.items()}
                 for branch, values in base.items()}
        stale["human"]["deter"][:, 1] = 500.0
        stale["human"]["stoch"][:, 1] = 500.0
        embeds = {"ego": torch.randn(b, e), "human": torch.randn(b, n, e)}
        action, goal = torch.randn(b, a), torch.randn(b, 8)
        mask = torch.ones(b, n, dtype=torch.bool)
        slot_reset = torch.zeros(b, n, dtype=torch.bool); slot_reset[:, 1] = True
        episode_reset = torch.zeros(b, dtype=torch.bool)
        torch.manual_seed(99)
        clean, _ = model.obs_step(base, action, embeds, episode_reset, mask, slot_reset, goal=goal)
        torch.manual_seed(99)
        changed, _ = model.obs_step(stale, action, embeds, episode_reset, mask, slot_reset, goal=goal)
        torch.testing.assert_close(clean["ego"]["deter"], changed["ego"]["deter"])
        torch.testing.assert_close(clean["human"]["deter"][:, 1], changed["human"]["deter"][:, 1])
        torch.testing.assert_close(clean["human"]["stoch"][:, 1], changed["human"]["stoch"][:, 1])

    def test_observe_and_imagination_use_coupled_prior(self):
        model = make_model()
        b, t, n, e, a = 2, 3, 4, 16, 3
        mask = torch.tensor([[[1, 1, 0, 0]] * t, [[1, 1, 1, 1]] * t], dtype=torch.bool)
        first = torch.zeros(b, t, dtype=torch.bool); first[:, 0] = True
        human_first = torch.zeros(b, t, n, dtype=torch.bool); human_first[:, 0] = mask[:, 0]
        embeds = {
            "ego": torch.randn(b, t, e, requires_grad=True),
            "human": torch.randn(b, t, n, e, requires_grad=True),
        }
        goal = torch.randn(b, t, 8)
        states, aux = model.observe(
            embeds, torch.randn(b, t, a), model.initial(b, n), first, mask,
            human_first, goal=goal,
        )
        losses = model.kl_loss(aux["post_logits"], aux["prior_logits"], mask, free=0.0)
        total = aux["joint_feat"].square().mean() + sum(losses.values())
        total.backward()
        for name in ("ego_rssm", "human_rssm", "latent_policy_attention",
                     "latent_coupling_attention"):
            self.assertTrue(any(p.grad is not None for p in getattr(model, name).parameters()), name)
        actor = torch.nn.Sequential(torch.nn.Linear(48, a), torch.nn.Tanh())
        imagined_states, imagined = model.imagine(
            model.initial(b, n), actor, 4, mask[:, 0], goal=goal[:, 0],
        )
        self.assertEqual(imagined["joint_feat"].shape, (b, 4, 48))
        self.assertEqual(imagined_states["human"]["deter"].shape, (b, 4, n, 32))


if __name__ == "__main__":
    unittest.main()
