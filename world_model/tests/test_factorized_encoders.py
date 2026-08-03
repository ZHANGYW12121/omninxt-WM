from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.factorized_encoders import FactorizedEncoderConfig, FactorizedObservationEncoder


class FactorizedEncoderTest(unittest.TestCase):
    def test_two_branch_shapes_masks_and_gradients(self):
        torch.manual_seed(7)
        b, t, n, j, d = 3, 5, 8, 17, 32
        model = FactorizedObservationEncoder(FactorizedEncoderConfig(
            model_dim=d, ego_hidden_dim=32, human_hidden_dim=32, pose_history=4,
        )).eval()
        counts = torch.tensor([0, 4, 8])
        mask = (torch.arange(n)[None, None] < counts[:, None, None]).expand(b, t, n).clone()
        joint_mask = mask[..., None].expand(b, t, n, j).clone()
        skeleton = torch.randn(b, t, n, j, 7)
        skeleton[..., 6] = torch.rand(b, t, n, j)
        batch = {
            "ego_state": torch.randn(b, t, 14), "skeleton": skeleton,
            "human_mask": mask, "joint_mask": joint_mask,
            "human_is_first": torch.zeros_like(mask),
        }
        batch["human_is_first"][:, 0] = mask[:, 0]
        out = model(batch)
        self.assertEqual(out["ego_embed"].shape, (b, t, d))
        self.assertEqual(out["human_tokens"].shape, (b, t, n, d))
        self.assertEqual(out["human_root"].shape, (b, t, n, 10))
        self.assertEqual(out["human_joints"].shape, (b, t, n, j, 7))
        self.assertNotIn("env_embed", out)
        self.assertEqual(torch.count_nonzero(out["human_tokens"][~mask]), 0)
        (out["ego_embed"].square().mean() + out["human_tokens"].square().mean()).backward()
        self.assertTrue(any(p.grad is not None for p in model.observation_attention.parameters()))

    def test_pose_history_is_causal(self):
        torch.manual_seed(8)
        b, t, n, j = 1, 6, 2, 17
        model = FactorizedObservationEncoder(FactorizedEncoderConfig(
            model_dim=16, human_hidden_dim=16, pose_history=4,
        )).eval()
        mask = torch.ones(b, t, n, dtype=torch.bool)
        skeleton = torch.randn(b, t, n, j, 7)
        skeleton[..., 6] = 1.0
        batch = {
            "ego_state": torch.randn(b, t, 14), "skeleton": skeleton,
            "human_mask": mask, "joint_mask": mask[..., None].expand(b, t, n, j),
            "human_is_first": torch.zeros_like(mask),
        }
        batch["human_is_first"][:, 0] = True
        before = model(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["skeleton"][:, 5] += 1.0e4
        after = model(changed)
        torch.testing.assert_close(before["human_tokens"][:, :5], after["human_tokens"][:, :5])


if __name__ == "__main__":
    unittest.main()
