from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.factorized_encoders import FactorizedEncoderConfig, FactorizedObservationEncoder
from modules.goal_conditioning import goal_features_torch


class FinalArchitectureContractTest(unittest.TestCase):
    def test_goal_features_are_body_relative(self):
        ego = torch.zeros(2, 14)
        ego[:, 13] = 1.0
        ego[1, 12], ego[1, 13] = 1.0, 0.0  # +90 degree yaw
        goal = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        feature = goal_features_torch(ego, goal)
        torch.testing.assert_close(feature[0, :3], torch.tensor([2.0, 0.0, 0.0]))
        torch.testing.assert_close(feature[1, :3], torch.tensor([0.0, -2.0, 0.0]))

    def test_stgcn_history_is_causal_and_reset_safe(self):
        torch.manual_seed(19)
        b, t, n, j = 1, 6, 2, 17
        model = FactorizedObservationEncoder(FactorizedEncoderConfig(
            model_dim=16, human_hidden_dim=16, pose_history=4,
        )).eval()
        mask = torch.ones(b, t, n, dtype=torch.bool)
        first = torch.zeros_like(mask); first[:, 0] = True; first[:, 4, 0] = True
        skeleton = torch.randn(b, t, n, j, 7); skeleton[..., 6] = 1.0
        batch = {
            "ego_state": torch.randn(b, t, 14), "skeleton": skeleton,
            "human_mask": mask, "joint_mask": mask[..., None].expand(b, t, n, j),
            "human_is_first": first,
        }
        base = model(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["skeleton"][:, :4, 0] += 1.0e5
        reset = model(changed)
        torch.testing.assert_close(base["human_tokens"][:, 4, 0], reset["human_tokens"][:, 4, 0])


if __name__ == "__main__":
    unittest.main()
