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

    def test_learned_initial_state_uses_categorical_prior_support(self):
        model = make_model()
        state = model.initial(2, 3)
        self.assertIsInstance(
            model.ego_rssm._initial_deter, torch.nn.Parameter)
        self.assertIsInstance(
            model.human_rssm._initial_deter, torch.nn.Parameter)
        torch.testing.assert_close(
            state["ego"]["stoch"].sum(-1),
            torch.ones(2, rssm_config().stoch),
        )
        torch.testing.assert_close(
            state["human"]["stoch"].sum(-1),
            torch.ones(2, 3, rssm_config().stoch),
        )
        self.assertFalse(bool(state["ego"]["stoch"].eq(0.0).all()))
        self.assertFalse(bool(state["human"]["stoch"].eq(0.0).all()))

    def test_invalid_initial_state_mode_is_rejected(self):
        config = rssm_config()
        config.initial = "ignored-but-not-implemented"
        with self.assertRaisesRegex(ValueError, "initial state"):
            FactorizedRSSM(
                config, {"ego": 16, "human": 16}, 3, goal_dim=8,
                latent_attention_config=LatentPolicyAttentionConfig(
                    model_dim=48, num_heads=4),
                coupling_attention_config=SparseEgoHumanAttentionConfig(
                    model_dim=32, num_heads=4),
            )

    def test_deterministic_posterior_mode_is_seed_invariant(self):
        model = make_model()
        batch, people, embed_dim, action_dim = 2, 3, 16, 3
        state = model.initial(batch, people, sample_state=False)
        embeddings = {
            "ego": torch.randn(batch, embed_dim),
            "human": torch.randn(batch, people, embed_dim),
        }
        action = torch.randn(batch, action_dim)
        mask = torch.ones(batch, people, dtype=torch.bool)
        reset = torch.ones(batch, dtype=torch.bool)
        human_reset = torch.ones(batch, people, dtype=torch.bool)
        goal = torch.randn(batch, 8)

        torch.manual_seed(101)
        rng_before = torch.random.get_rng_state().clone()
        first, _ = model.obs_step(
            state, action, embeddings, reset, mask, human_reset, goal=goal,
            sample_state=False)
        rng_after = torch.random.get_rng_state().clone()
        torch.manual_seed(202)
        second, _ = model.obs_step(
            state, action, embeddings, reset, mask, human_reset, goal=goal,
            sample_state=False)

        torch.testing.assert_close(rng_before, rng_after, rtol=0, atol=0)
        for branch in ("ego", "human"):
            for key in ("stoch", "deter"):
                torch.testing.assert_close(
                    first[branch][key], second[branch][key], rtol=0, atol=0)
            torch.testing.assert_close(
                first[branch]["stoch"].sum(-1),
                torch.ones_like(first[branch]["stoch"].sum(-1)),
                rtol=0, atol=0,
            )

        torch.manual_seed(303)
        stochastic_before = torch.random.get_rng_state().clone()
        model.obs_step(
            state, action, embeddings, reset, mask, human_reset, goal=goal,
            sample_state=True)
        self.assertFalse(torch.equal(
            stochastic_before, torch.random.get_rng_state()))

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

    def test_identity_attention_does_not_duplicate_branch_recurrence(self):
        class IdentityAttention(torch.nn.Module):
            def forward(self, ego, human, human_mask):
                del human_mask
                return {"ego": ego[:, None], "human": human}

        model = make_model()
        model.latent_coupling_attention = IdentityAttention()
        state = model.initial(2, 3, sample_state=False)
        context = model.get_transition_context(
            state, torch.tensor([[True, True, False], [True, False, False]]))
        torch.testing.assert_close(
            context["ego"], torch.zeros_like(context["ego"]), rtol=0, atol=0)
        torch.testing.assert_close(
            context["human"], torch.zeros_like(context["human"]),
            rtol=0, atol=0)

    def test_human_slot_reset_blocks_old_identity_from_new_human_branch(self):
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
        # Slot one is a destination-frame appearance.  It was absent at the
        # source, so neither its stale contents nor its future visibility may
        # enter the Ego transition.
        previous_mask = mask.clone(); previous_mask[:, 1] = False
        torch.manual_seed(99)
        clean, _ = model.obs_step(
            base, action, embeds, episode_reset, mask, slot_reset, goal=goal,
            previous_human_mask=previous_mask)
        torch.manual_seed(99)
        changed, _ = model.obs_step(
            stale, action, embeds, episode_reset, mask, slot_reset, goal=goal,
            previous_human_mask=previous_mask)
        torch.testing.assert_close(clean["ego"]["deter"], changed["ego"]["deter"])
        torch.testing.assert_close(clean["human"]["deter"][:, 1], changed["human"]["deter"][:, 1])
        torch.testing.assert_close(clean["human"]["stoch"][:, 1], changed["human"]["stoch"][:, 1])

    def test_ego_transition_uses_source_not_destination_human_mask(self):
        model = make_model()
        b, n, a = 1, 3, 3
        state = model.initial(b, n)
        state["human"]["deter"][:, 1] = 3.0
        state["human"]["stoch"][:, 1, 0, 0] = 1.0
        action = torch.randn(b, a)
        source_mask = torch.tensor([[False, True, False]])
        destination_absent = torch.zeros(b, n, dtype=torch.bool)
        destination_recycled = torch.tensor([[True, False, True]])
        reset_absent = ~destination_absent
        reset_recycled = torch.ones_like(destination_recycled)

        source_input_a, _, _ = model._transition_inputs(
            state, action, destination_absent,
            human_reset=reset_absent,
            context_human_mask=source_mask)
        source_input_b, _, _ = model._transition_inputs(
            state, action, destination_recycled,
            human_reset=reset_recycled,
            context_human_mask=source_mask)
        torch.testing.assert_close(source_input_a, source_input_b)

        no_source_input, _, _ = model._transition_inputs(
            state, action, destination_recycled,
            human_reset=reset_recycled,
            context_human_mask=torch.zeros_like(source_mask))
        self.assertGreater(
            float((source_input_a - no_source_input).abs().max()), 1.0e-6)

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
