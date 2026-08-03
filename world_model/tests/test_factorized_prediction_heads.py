from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads


class HeadTest(unittest.TestCase):
    def test_two_branch_shapes_masks_and_gradients(self):
        torch.manual_seed(4)
        b, t, n, j, f = 2, 5, 6, 17, 40
        heads = FactorizedPredictionHeads(f, f, FactorizedPredictionConfig(hidden_dim=32))
        feats = {
            "ego": torch.randn(b, t, f, requires_grad=True),
            "human": torch.randn(b, t, n, f, requires_grad=True),
        }
        mask = torch.tensor([[[1, 1, 0, 0, 0, 0]] * t,
                             [[1, 1, 1, 1, 1, 1]] * t], dtype=torch.bool)
        joint = mask[..., None].expand(b, t, n, j).clone()
        batch = {
            "ego_state": torch.randn(b, t, 14),
            "skeleton": torch.randn(b, t, n, j, 7),
            "human_mask": mask, "joint_mask": joint,
        }
        pred, loss = heads.forward_loss(feats, batch)
        self.assertEqual(pred["ego"]["current_state"].shape, (b, t, 14))
        self.assertEqual(pred["human"]["joints"].shape, (b, t, n, j, 3))
        self.assertNotIn("env", pred)
        self.assertEqual(set(loss), {"ego_recon", "ego_pred", "yaw_unit",
                                    "human_root", "human_mpjpe", "human_presence"})
        self.assertTrue(all(torch.isfinite(value) for value in loss.values()))
        sum(loss.values()).backward()
        self.assertTrue(all(value.grad is not None for value in feats.values()))


if __name__ == "__main__":
    unittest.main()
