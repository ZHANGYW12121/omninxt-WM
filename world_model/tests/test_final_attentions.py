from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.latent_policy_attention import ActionTokenLatentAttention, LatentPolicyAttentionConfig
from modules.sparse_ego_human_attention import SparseEgoHumanAttention, SparseEgoHumanAttentionConfig


class FinalAttentionTest(unittest.TestCase):
    def test_sparse_observation_visibility_and_padding(self):
        torch.manual_seed(3)
        model = SparseEgoHumanAttention(
            SparseEgoHumanAttentionConfig(model_dim=16, num_heads=4, dropout=0.0)
        ).eval()
        ego, humans = torch.randn(2, 1, 16), torch.randn(2, 4, 16)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        out = model(ego, humans, mask)
        weights = out["attention_weights"]
        # Human 1 cannot attend Human 2, while Ego can attend every valid Human.
        self.assertEqual(torch.count_nonzero(weights[:, :, 1, 2:]), 0)
        self.assertGreater(float(weights[1, :, 0, 1:].sum().detach()), 0.0)
        corrupt = humans.clone(); corrupt[~mask] = 1.0e6
        changed = model(ego, corrupt, mask)
        torch.testing.assert_close(out["ego"], changed["ego"])

    def test_goal_action_readout_and_padding_isolation(self):
        torch.manual_seed(5)
        model = ActionTokenLatentAttention(
            20, 20, 8, LatentPolicyAttentionConfig(model_dim=16, num_heads=4, dropout=0.0)
        ).eval()
        goal, ego, human = torch.randn(2, 8), torch.randn(2, 20), torch.randn(2, 5, 20)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        out = model(goal, ego, human, mask)
        corrupt = human.clone(); corrupt[~mask] = 1.0e6
        changed = model(goal, ego, corrupt, mask)
        torch.testing.assert_close(out["joint_feat"], out["latent_tokens"][:, 0])
        torch.testing.assert_close(out["joint_feat"], changed["joint_feat"])
        self.assertEqual(out["goal_token"].shape, (2, 16))
        self.assertEqual(out["latent_tokens"].shape[1], 3 + human.shape[1])

    def test_metric_goal_path_preserves_remaining_distance(self):
        torch.manual_seed(9)
        model = ActionTokenLatentAttention(
            20, 20, 8,
            LatentPolicyAttentionConfig(
                model_dim=16, num_heads=4, dropout=0.0,
                goal_metric_scaling=True),
        ).eval()
        self.assertIsInstance(model.goal_projector, nn.Linear)
        goal = torch.tensor([
            [5.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        ])
        output = model(
            goal,
            torch.zeros(2, 20),
            torch.zeros(2, 1, 20),
            torch.zeros(2, 1, dtype=torch.bool),
        )
        torch.testing.assert_close(
            output["private_goal_token"][:, :8],
            goal / model.goal_metric_scale,
        )
        self.assertGreater(float(
            torch.linalg.vector_norm(
                output["private_goal_token"][0]
                - output["private_goal_token"][1])), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
