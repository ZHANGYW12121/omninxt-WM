from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads
from modules.skeleton_topology import (
    COCO12_MAX_CENTER_RADIUS_M,
    COCO12_MAX_EDGE_LENGTH_M,
    project_coco12_geometry_torch,
)


class HeadTest(unittest.TestCase):
    def test_one_step_pose_loss_excludes_newly_observed_source_missing_joint(self):
        heads = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(hidden_dim=8))
        human = heads.human
        shape = (1, 2, 1)
        skeleton = torch.zeros(*shape, 12, 7)
        root = torch.zeros(*shape, 6)
        human_mask = torch.ones(*shape, dtype=torch.bool)
        joint_mask = torch.zeros(*shape, 12, dtype=torch.bool)
        joint_mask[:, 1, :, 2] = True
        measured_valid = torch.zeros_like(joint_mask)
        measured_valid[:, 1, :, 2] = True
        prediction = {
            "root": torch.zeros(*shape, 3),
            "root_velocity": torch.zeros(*shape, 3),
            "joints": torch.zeros(*shape, 12, 3),
            "survival_logit": torch.zeros(*shape),
            "birth_logit": torch.zeros(*shape),
        }
        prediction["joints"][:, 0, :, 2] = 100.0
        losses = human.loss(
            prediction, skeleton, root, human_mask, joint_mask, {
                "human_motion_valid": torch.tensor([[[True], [False]]]),
                "human_survival_valid": torch.tensor([[[True], [False]]]),
                "human_survival_target": torch.tensor([[[True], [False]]]),
                "measured_root_target_valid": torch.zeros(
                    *shape, dtype=torch.bool),
                "measured_velocity_target_valid": torch.zeros(
                    *shape, dtype=torch.bool),
                "measured_joint_target": skeleton[..., :3],
                "measured_joint_target_valid": measured_valid,
            },
        )
        self.assertEqual(float(losses["human_mpjpe"]), 0.0)

    def test_imagined_anthropometric_projection_is_bounded_and_differentiable(self):
        joints = torch.zeros(2, 12, 3, requires_grad=True)
        corrupted = joints + torch.nn.functional.one_hot(
            torch.tensor(4), 12).reshape(1, 12, 1) * torch.tensor(
                (8.0, 0.0, 0.0)).reshape(1, 1, 3)
        mask = torch.ones(2, 12, dtype=torch.bool)
        projected = project_coco12_geometry_torch(
            corrupted, mask, torch.zeros(2, 3))
        self.assertLessEqual(
            float(torch.linalg.vector_norm(
                projected.detach(), dim=-1).max()),
            COCO12_MAX_CENTER_RADIUS_M + 1.0e-5,
        )
        for edge, maximum in COCO12_MAX_EDGE_LENGTH_M.items():
            length = torch.linalg.vector_norm(
                projected[..., edge[0], :] - projected[..., edge[1], :],
                dim=-1,
            )
            self.assertLessEqual(float(length.detach().max()), maximum + 1.0e-5)
        projected.sum().backward()
        self.assertTrue(bool(torch.isfinite(joints.grad).all()))

    def test_unconditioned_birth_calibration_cannot_update_motion_trunk(self):
        heads = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(hidden_dim=8))
        human = heads.human
        feature = torch.zeros(2, 3, 4, requires_grad=True)
        prediction = human(feature, torch.zeros(2, 3, 6))
        prediction["birth_logit"].sum().backward()
        self.assertIsNotNone(human.birth.weight.grad)
        self.assertIsNotNone(human.birth.bias.grad)
        self.assertGreater(float(human.birth.bias.grad.abs().sum()), 0.0)
        self.assertTrue(all(
            parameter.grad is None for parameter in human.trunk.parameters()))
        self.assertIsNone(feature.grad)

    def test_joint_kinematic_mode_starts_at_cv_and_learns_bounded_velocity(self):
        torch.manual_seed(12)
        heads = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(
                hidden_dim=8,
                kinematic_velocity_only=True,
                kinematic_joint_velocity_only=True,
                max_joint_velocity_residual_mps=0.35,
            ))
        feat = torch.randn(2, 3, 4)
        root = torch.zeros(2, 3, 6)
        joints = torch.randn(2, 3, 12, 3) * 0.05
        velocity = torch.randn_like(joints) * 0.2
        prediction = heads.human(
            feat, root,
            current_joints=joints,
            current_joint_velocity=velocity,
            dt_s=0.1,
        )
        torch.testing.assert_close(
            prediction["joints"], joints + 0.1 * velocity)
        torch.testing.assert_close(
            prediction["joint_velocity_residual_current_body"],
            torch.zeros_like(joints),
        )

        prediction["joints"].sum().backward()
        self.assertGreater(
            float(heads.human.joints.weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            heads.human.joints.bias.fill_(100.0)
        bounded = heads.human(
            feat, root,
            current_joints=joints,
            current_joint_velocity=velocity,
        )["joint_velocity_residual_current_body"]
        self.assertTrue(bool((bounded.abs() <= 0.350001).all()))

    def test_velocity_only_mode_is_kinematically_consistent_and_load_compatible(self):
        config = FactorizedPredictionConfig(
            hidden_dim=8, kinematic_velocity_only=True)
        legacy = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(hidden_dim=8))
        heads = FactorizedPredictionHeads(4, 4, config)
        heads.load_state_dict(legacy.state_dict(), strict=True)
        human = heads.human
        with torch.no_grad():
            for parameter in human.parameters():
                parameter.zero_()
            human.root_delta.bias.fill_(100.0)
            human.velocity_delta.bias.copy_(torch.tensor((1.0, 2.0, 3.0)))

        root = torch.zeros(1, 1, 1, 6)
        pred = human(torch.zeros(1, 1, 1, 4), root, dt_s=0.1)
        expected_velocity = 0.35 * torch.tanh(
            torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(
            pred["root"], (0.1 * expected_velocity).reshape(1, 1, 1, 3))
        torch.testing.assert_close(
            pred["root_velocity"], expected_velocity.reshape(1, 1, 1, 3))
        torch.testing.assert_close(
            pred["root_residual_next_body"], torch.zeros(1, 1, 1, 3))
        self.assertFalse(human.root_delta.weight.requires_grad)
        self.assertFalse(human.root_delta.bias.requires_grad)

    def test_fresh_velocity_correction_is_zero_and_bounded(self):
        heads = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(
                hidden_dim=8,
                kinematic_velocity_only=True,
                max_velocity_residual_mps=0.35,
            ))
        human = heads.human
        pred = human(
            torch.randn(2, 3, 4),
            torch.zeros(2, 3, 6),
            dt_s=0.1,
        )
        torch.testing.assert_close(
            pred["velocity_residual_current_body"],
            torch.zeros(2, 3, 3),
        )
        with torch.no_grad():
            human.velocity_delta.bias.fill_(100.0)
        saturated = human(
            torch.randn(2, 3, 4),
            torch.zeros(2, 3, 6),
            dt_s=0.1,
        )["velocity_residual_current_body"]
        self.assertTrue(bool((saturated.abs() <= 0.350001).all()))

    def test_velocity_correction_scales_with_forced_terminal_exposure(self):
        heads = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(
                hidden_dim=8,
                kinematic_velocity_only=True,
                kinematic_joint_velocity_only=True,
                max_velocity_residual_mps=0.35,
                max_joint_velocity_residual_mps=0.35,
            ))
        with torch.no_grad():
            heads.human.velocity_delta.bias.fill_(100.0)
            heads.human.joints.bias.fill_(100.0)
        feat = torch.zeros(1, 1, 4)
        root = torch.zeros(1, 1, 6)
        joints = torch.zeros(1, 1, 12, 3)
        velocity = torch.zeros_like(joints)
        nominal = heads.human(
            feat, root, current_joints=joints,
            current_joint_velocity=velocity, dt_s=0.1)
        partial = heads.human(
            feat, root, current_joints=joints,
            current_joint_velocity=velocity, dt_s=0.02)
        torch.testing.assert_close(
            partial["velocity_residual_current_body"],
            0.2 * nominal["velocity_residual_current_body"])
        torch.testing.assert_close(
            partial["joint_velocity_residual_current_body"],
            0.2 * nominal["joint_velocity_residual_current_body"])

    def test_recursive_human_rollout_cannot_accumulate_unbounded_speed(self):
        heads = FactorizedPredictionHeads(
            4, 4, FactorizedPredictionConfig(
                hidden_dim=8,
                kinematic_velocity_only=True,
                max_velocity_residual_mps=0.70,
                max_root_speed_mps=2.50,
                kinematic_joint_velocity_only=True,
                max_joint_velocity_residual_mps=1.50,
                max_joint_speed_mps=6.00,
            ))
        with torch.no_grad():
            heads.human.velocity_delta.bias.fill_(100.0)
            heads.human.joints.bias.fill_(100.0)
        feat = torch.zeros(1, 1, 4)
        root = torch.zeros(1, 1, 6)
        joints = torch.zeros(1, 1, 12, 3)
        joint_velocity = torch.zeros_like(joints)
        for _ in range(30):
            prediction = heads.human(
                feat, root, current_joints=joints,
                current_joint_velocity=joint_velocity, dt_s=0.1)
            root = torch.cat((
                prediction["root"], prediction["root_velocity"]), dim=-1)
            joints = prediction["joints"]
            joint_velocity = prediction["joint_velocity"]
        self.assertLessEqual(
            float(torch.linalg.vector_norm(
                prediction["root_velocity"].detach(), dim=-1).max()),
            2.500001,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(
                prediction["joint_velocity"].detach(), dim=-1).max()),
            6.000001,
        )
        self.assertTrue(bool(torch.isfinite(joints).all()))

    def test_two_branch_shapes_masks_and_gradients(self):
        torch.manual_seed(4)
        b, t, n, j, f = 2, 5, 6, 12, 40
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
        self.assertEqual(set(loss), {
            "ego_recon", "ego_pred", "yaw_unit", "human_root",
            "human_velocity", "human_mpjpe", "human_presence",
        })
        self.assertTrue(all(torch.isfinite(value) for value in loss.values()))
        sum(loss.values()).backward()
        self.assertTrue(all(value.grad is not None for value in feats.values()))


if __name__ == "__main__":
    unittest.main()
