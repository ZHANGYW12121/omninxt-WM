from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factorized_dreamer import FactorizedDreamer
from factorized_rssm import FactorizedRSSM
from factorized_trainer import FactorizedTrainStep
from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads
from modules.latent_policy_attention import LatentPolicyAttentionConfig


class Encoder(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.ego = nn.Linear(14, embed_dim)
        self.human = nn.Linear(7, embed_dim)

    def forward(self, batch):
        return {
            "ego_embed": self.ego(batch["ego_state"]),
            "human_tokens": self.human(batch["skeleton_stub"]),
            "human_token_mask": batch["human_mask"],
        }


class NormalHead(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        mean = self.linear(x)
        return Independent(Normal(mean, torch.ones_like(mean)), 1)


class ContinueHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return Independent(Bernoulli(logits=self.linear(x)), 1)


class Actor(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, action_dim)

    def forward(self, x):
        mean = self.linear(x)
        return Independent(Normal(mean, torch.ones_like(mean) * 0.5), 1)


def rssm_cfg():
    return SimpleNamespace(stoch=4, deter=32, hidden=32, discrete=4, img_layers=1,
        obs_layers=1, dyn_layers=1, blocks=4, act="SiLU", norm=True,
        unimix_ratio=0.01, initial="learned", device="cpu")


def make_model(embed=16, action=3, joint=32):
    dynamics = FactorizedRSSM(
        rssm_cfg(), {"ego": embed, "human": embed}, action, goal_dim=8,
        latent_attention_config=LatentPolicyAttentionConfig(model_dim=joint, num_heads=4),
    )
    predictions = FactorizedPredictionHeads(
        dynamics.ego_rssm.feat_size, dynamics.human_rssm.feat_size,
        FactorizedPredictionConfig(hidden_dim=32),
    )
    return FactorizedDreamer(
        Encoder(embed), dynamics, Actor(joint, action), NormalHead(joint),
        NormalHead(joint), ContinueHead(joint), predictions, kl_free=0.0,
    )


def make_batch(b=2, t=4, n=3, action=3):
    mask = torch.ones(b, t, n, dtype=torch.bool)
    ego = torch.randn(b, t, 14)
    yaw = torch.randn(b, t, 2)
    ego[..., 12:14] = yaw / yaw.norm(dim=-1, keepdim=True)
    batch = {
        "ego_state": ego, "skeleton_stub": torch.randn(b, t, n, 7),
        "skeleton": torch.randn(b, t, n, 12, 7), "human_mask": mask,
        "joint_mask": mask[..., None].expand(b, t, n, 12).clone(),
        "human_is_first": torch.zeros(b, t, n, dtype=torch.bool),
        "goal": torch.randn(b, t, 8), "goal_position": torch.randn(b, t, 3),
        "action": torch.randn(b, t, action), "is_first": torch.zeros(b, t, dtype=torch.bool),
        "reward": torch.randn(b, t, 1), "is_terminal": torch.zeros(b, t, 1),
        "is_last": torch.zeros(b, t, 1), "truncated_people": torch.zeros(b, t),
    }
    batch["is_first"][:, 0] = True
    batch["human_is_first"][:, 0] = True
    return batch


class FactorizedDreamerTest(unittest.TestCase):
    LOSS_SCALES = {
        "dyn_ego": 1, "rep_ego": .1, "dyn_human": 1, "rep_human": .1,
        "ego_recon": 1, "ego_pred": 1, "yaw_unit": .1,
        "human_root": 1, "human_mpjpe": 1, "human_presence": .2,
        "rew": 1, "con": 1, "policy": 1, "value": 1, "repval": .3,
    }

    def test_world_model_goal_imagination_and_online_action(self):
        torch.manual_seed(5)
        model = make_model()
        batch = make_batch()
        losses, states, aux = model.world_model_loss(batch)
        self.assertEqual(set(losses), {"dyn_ego", "rep_ego", "dyn_human", "rep_human", "rew", "con"})
        self.assertNotIn("env", states)
        ac_losses, _ = model.actor_critic_loss(states, aux, batch, imag_horizon=3)
        self.assertEqual(set(ac_losses), {"policy", "value", "repval"})
        imagined_states, imagined = model.imagine(
            model.last_state(states), 3, batch["human_mask"][:, -1],
            goal_position=batch["goal_position"][:, -1],
        )
        self.assertEqual(imagined["goal"].shape, (2, 3, 8))
        self.assertNotIn("env", imagined_states)
        online = {key: value[:, :1] for key, value in batch.items()}
        action, agent_state = model.act_step(
            online, model.initial_agent_state(2, 3, 3),
        )
        self.assertEqual(action.shape, (2, 3))
        restored = model.state_from_replay_fields(model.replay_state_fields(agent_state["latent"]))
        self.assertEqual(set(restored), {"ego", "human"})

    def test_complete_optimizer_step(self):
        torch.manual_seed(9)
        model = make_model(action=2)
        batch = make_batch(action=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = FactorizedTrainStep(
            model, optimizer, imag_horizon=3, loss_scales=self.LOSS_SCALES,
        )(batch)
        self.assertIn("loss/total", metrics)
        for component in (
            "dyn_ego", "dyn_human", "ego_recon", "human_mpjpe",
            "rew", "con", "policy", "value", "repval",
        ):
            self.assertIn(f"loss/{component}", metrics)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
