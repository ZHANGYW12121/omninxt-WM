import unittest

import torch

from modules.transition_event_head import (
    HumanSafetyCritic,
    NextHumanClearanceHead,
    TransitionEventHead,
    survival_human_collision_return,
)


class TransitionEventHeadTest(unittest.TestCase):
    def test_event_probabilities_are_closed(self):
        head = TransitionEventHead(7, 4, 16)
        output = head(torch.randn(3, 5, 7), torch.randn(3, 5, 4))
        self.assertEqual(output["probability"].shape, (3, 5, 5))
        torch.testing.assert_close(
            output["probability"].sum(-1), torch.ones(3, 5))
        non_goal = head.forward_non_goal(
            torch.randn(3, 5, 7), torch.randn(3, 5, 4))
        self.assertEqual(non_goal["probability"].shape, (3, 5, 5))
        torch.testing.assert_close(
            non_goal["probability"].sum(-1), torch.ones(3, 5))

    def test_literal_any_person_noisy_or_is_closed(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        for parameter in head.per_human_network.parameters():
            parameter.data.zero_()
        feature = torch.randn(1, 7)
        action = torch.randn(1, 4)
        output = head.forward_non_goal(
            feature, action,
            ego_feature=torch.randn(1, 5),
            human_feature=torch.randn(1, 3, 6),
            human_root=torch.randn(1, 3, 10),
            human_quality=torch.zeros(1, 3, 7),
            human_mask=torch.tensor([[True, True, False]]),
        )
        # Contact with either of two independently hazardous people is the
        # literal union probability 1-(1-.5)^2=.75.
        torch.testing.assert_close(
            output["observed_human_collision_probability"],
            torch.tensor([[0.75]]),
        )
        torch.testing.assert_close(
            output["probability"].sum(-1), torch.ones(1))
        torch.testing.assert_close(
            output["probability"].log(),
            output["log_probability"], atol=1e-5, rtol=1e-5)

    def test_empty_human_slots_have_zero_observed_but_finite_residual_hazard(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        output = head.forward_non_goal(
            torch.randn(2, 7), torch.randn(2, 4),
            ego_feature=torch.randn(2, 5),
            human_feature=torch.randn(2, 3, 6),
            human_root=torch.randn(2, 3, 10),
            human_quality=torch.zeros(2, 3, 7),
            human_mask=torch.zeros(2, 3, dtype=torch.bool),
        )
        torch.testing.assert_close(
            output["observed_human_collision_probability"],
            torch.zeros(2, 1),
        )
        self.assertTrue(torch.isfinite(
            output["human_collision_probability"]).all())
        torch.testing.assert_close(
            output["human_collision_probability"],
            output["unobserved_human_collision_probability"],
        )
        torch.testing.assert_close(
            output["probability"].sum(-1), torch.ones(2))

    def test_geometry_pool_preserves_nearest_signed_physical_state(self):
        head = TransitionEventHead(
            16, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        root = torch.zeros(1, 2, 10)
        root[0, 0, :6] = torch.tensor((1.2, 0.9, 0.3, 0.6, -0.3, 0.0))
        root[0, 1, :3] = torch.tensor((4.0, -2.0, 0.3))
        common = dict(
            human_feature=torch.randn(1, 2, 6),
            human_quality=torch.zeros(1, 2, 7),
            human_mask=torch.ones(1, 2, dtype=torch.bool),
            human_joint_clearance=torch.tensor([[0.25, 2.0]]),
        )
        original = head.human_geometry_pool(
            human_root=root, **common)["human_geometry_pool"]
        mirrored_root = root.clone()
        mirrored_root[..., 1] *= -1.0
        mirrored_root[..., 4] *= -1.0
        mirrored = head.human_geometry_pool(
            human_root=mirrored_root, **common)["human_geometry_pool"]
        # The reserved prefix is the exact normalized state of the nearest
        # person: x/6, y/6, z/3, vx/3, vy/3, vz/3, clearance, presence.
        torch.testing.assert_close(
            original[0, :8],
            torch.tensor((0.2, 0.15, 0.1, 0.2, -0.1, 0.0, 0.25, 1.0)),
        )
        torch.testing.assert_close(mirrored[0, 1], -original[0, 1])
        torch.testing.assert_close(mirrored[0, 4], -original[0, 4])
        torch.testing.assert_close(mirrored[0, 0], original[0, 0])

    def test_topk_geometry_preserves_multiple_people_in_distance_order(self):
        head = TransitionEventHead(
            32, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True,
            geometry_topk_physical_slots=3)
        root = torch.zeros(1, 4, 10)
        # Input order is deliberately far, near, masked, middle.
        root[0, :, 0] = torch.tensor((3.0, 1.0, 0.2, 2.0))
        root[0, :, 1] = torch.tensor((-1.2, 0.6, 9.0, -0.9))
        root[0, :, 3] = torch.tensor((0.3, 0.1, 9.0, 0.2))
        root[0, :, 4] = torch.tensor((-0.3, 0.2, 9.0, -0.2))
        clearance = torch.tensor([[2.5, 0.4, 0.0, 1.2]])
        pooled = head.human_geometry_pool(
            human_feature=torch.randn(1, 4, 6),
            human_root=root,
            human_quality=torch.zeros(1, 4, 7),
            human_mask=torch.tensor([[True, True, False, True]]),
            human_joint_clearance=clearance,
        )["human_geometry_pool"]
        expected = torch.tensor([
            # near slot 1
            1.0 / 6.0, 0.6 / 6.0, 0.0, 0.1 / 3.0, 0.2 / 3.0, 0.4,
            # middle slot 3
            2.0 / 6.0, -0.9 / 6.0, 0.0, 0.2 / 3.0, -0.2 / 3.0, 1.2,
            # far slot 0
            3.0 / 6.0, -1.2 / 6.0, 0.0, 0.3 / 3.0, -0.3 / 3.0, 2.5,
        ])
        torch.testing.assert_close(pooled[0, :18], expected)
        self.assertEqual(head.geometry_physical_dim, 18)

    def test_soft_presence_scales_single_slot_hazard(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        for parameter in head.per_human_network.parameters():
            parameter.data.zero_()
        common = dict(
            feature=torch.zeros(1, 7), action=torch.zeros(1, 4),
            ego_feature=torch.zeros(1, 5),
            human_feature=torch.zeros(1, 1, 6),
            human_root=torch.zeros(1, 1, 10),
            human_quality=torch.zeros(1, 1, 7),
            human_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        full = head.forward_non_goal(
            **common, human_presence=torch.ones(1, 1))
        fading = head.forward_non_goal(
            **common, human_presence=torch.full((1, 1), 0.10))
        torch.testing.assert_close(
            full["observed_human_collision_probability"],
            torch.tensor([[0.50]]),
        )
        torch.testing.assert_close(
            fading["observed_human_collision_probability"],
            torch.tensor([[0.05]]),
        )

    def test_v73_physical_geometry_exposes_lifecycle_probability(self):
        head = TransitionEventHead(
            16, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True,
            explicit_joint_kinematics=True,
            explicit_human_presence_physical=True,
            geometry_topk_physical_slots=1,
        )
        root = torch.zeros(1, 1, 10)
        root[0, 0, (0, 1, 3, 4)] = torch.tensor((1.2, -0.6, 0.3, -0.9))
        joints = torch.full((1, 1, 12, 3), 2.0)
        pooled = head.human_geometry_pool(
            human_feature=torch.zeros(1, 1, 6),
            human_root=root,
            human_quality=torch.zeros(1, 1, 7),
            human_mask=torch.ones(1, 1, dtype=torch.bool),
            human_joint_clearance=torch.tensor([[0.30]]),
            human_joints_body=joints,
            human_joint_velocity_body=torch.zeros_like(joints),
            human_joint_mask=torch.ones(1, 1, 12, dtype=torch.bool),
            human_presence=torch.tensor([[0.25]]),
        )["human_geometry_pool"]
        torch.testing.assert_close(
            pooled[0, :6],
            torch.tensor((0.2, -0.1, 0.1, -0.3, 0.25, 0.30)),
        )

    def test_full_actor_state_preserves_every_valid_joint_field(self):
        head = TransitionEventHead(
            16, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True,
            explicit_joint_kinematics=True,
            explicit_human_presence_physical=True,
            geometry_topk_physical_slots=1,
            actor_full_state_slots=2,
            actor_full_fields_per_slot=103,
        )
        root = (torch.randn(1, 2, 10) * 0.2).requires_grad_(True)
        quality = (torch.rand(1, 2, 7) * 0.2).requires_grad_(True)
        joints = (torch.randn(1, 2, 12, 3) * 0.2 + 1.0).requires_grad_(True)
        velocity = (torch.randn(1, 2, 12, 3) * 0.1).requires_grad_(True)
        state = head.human_geometry_pool(
            human_feature=torch.randn(1, 2, 6),
            human_root=root,
            human_quality=quality,
            human_mask=torch.ones(1, 2, dtype=torch.bool),
            human_joints_body=joints,
            human_joint_velocity_body=velocity,
            human_joint_mask=torch.ones(1, 2, 12, dtype=torch.bool),
            human_presence=torch.tensor([[0.8, 0.9]]),
        )["actor_human_state"]
        self.assertEqual(state.shape, (1, 2 * 103 + 16))
        gradients = torch.autograd.grad(
            state.sum(), (root, quality, joints, velocity))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(gradient.abs().gt(0.0).all())

    def test_full_actor_state_keeps_learned_context_per_person(self):
        torch.manual_seed(37)
        head = TransitionEventHead(
            16, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True,
            explicit_joint_kinematics=True,
            explicit_human_presence_physical=True,
            geometry_topk_physical_slots=1,
            actor_full_state_slots=2,
            actor_full_fields_per_slot=103,
            actor_full_learned_per_slot=True,
        )
        root = torch.zeros(1, 2, 10)
        root[..., 0] = torch.tensor((1.0, 2.0))
        joints = torch.zeros(1, 2, 12, 3)
        joints[..., 0] = root[..., None, 0]
        kwargs = {
            "human_root": root,
            "human_quality": torch.zeros(1, 2, 7),
            "human_mask": torch.ones(1, 2, dtype=torch.bool),
            "human_joints_body": joints,
            "human_joint_velocity_body": torch.zeros_like(joints),
            "human_joint_mask": torch.ones(1, 2, 12, dtype=torch.bool),
            "human_presence": torch.ones(1, 2),
        }
        human_feature = torch.randn(1, 2, 6)
        baseline = head.human_geometry_pool(
            human_feature, **kwargs)["actor_human_state"]
        changed_feature = human_feature.clone()
        changed_feature[:, 0] += 1.0
        changed = head.human_geometry_pool(
            changed_feature, **kwargs)["actor_human_state"]

        physical_dim = 2 * 103
        learned_per_slot = 16 - 12
        self.assertEqual(
            baseline.shape, (1, physical_dim + 2 * learned_per_slot))
        torch.testing.assert_close(
            baseline[..., :physical_dim], changed[..., :physical_dim])
        self.assertFalse(torch.equal(
            baseline[..., physical_dim:physical_dim + learned_per_slot],
            changed[..., physical_dim:physical_dim + learned_per_slot],
        ))
        torch.testing.assert_close(
            baseline[..., physical_dim + learned_per_slot:],
            changed[..., physical_dim + learned_per_slot:],
        )

    def test_event_hazard_directly_reads_every_articulated_joint_field(self):
        torch.manual_seed(41)
        head = TransitionEventHead(
            16, 4, 24, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True,
            explicit_joint_kinematics=True,
            explicit_human_presence_physical=True,
            geometry_topk_physical_slots=1,
            actor_full_state_slots=1,
            actor_full_fields_per_slot=103,
            event_full_articulated_state=True,
        )
        joints = (torch.randn(1, 1, 12, 3) * 0.2 + 1.0).requires_grad_(True)
        velocity = (torch.randn(1, 1, 12, 3) * 0.1).requires_grad_(True)
        hazard = head.forward_human_hazard(
            torch.randn(1, 16), torch.randn(1, 4),
            ego_feature=torch.randn(1, 5),
            human_feature=torch.randn(1, 1, 6),
            human_root=torch.randn(1, 1, 10) * 0.2,
            human_quality=torch.rand(1, 1, 7) * 0.2,
            human_mask=torch.ones(1, 1, dtype=torch.bool),
            human_joints_body=joints,
            human_joint_velocity_body=velocity,
            human_joint_mask=torch.ones(1, 1, 12, dtype=torch.bool),
        )["slot_hazard"].sum()
        gradients = torch.autograd.grad(hazard, (joints, velocity))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(gradient.abs().gt(0.0).all())

    def test_frozen_human_calibrator_preserves_raw_score_and_changes_probability(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        for parameter in head.per_human_network.parameters():
            parameter.data.zero_()
        common = dict(
            feature=torch.zeros(1, 7), action=torch.zeros(1, 4),
            ego_feature=torch.zeros(1, 5),
            human_feature=torch.zeros(1, 1, 6),
            human_root=torch.zeros(1, 1, 10),
            human_quality=torch.zeros(1, 1, 7),
            human_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        identity = head.forward_human_hazard(**common)
        torch.testing.assert_close(
            identity["human_collision_probability"],
            identity["raw_human_collision_probability"],
        )
        head.set_human_probability_calibration(
            log_temperature=0.0, bias=-1.0, count_bias=0.0)
        calibrated = head.forward_human_hazard(**common)
        torch.testing.assert_close(
            calibrated["raw_human_collision_probability"],
            identity["raw_human_collision_probability"],
        )
        self.assertLess(
            calibrated["human_collision_probability"].item(),
            identity["human_collision_probability"].item(),
        )
        self.assertTrue(bool(head.human_calibration_fitted))

    def test_identity_calibration_contract_rejects_hidden_buffer_mutation(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        head.require_identity_human_probability_calibration()
        head.human_calibration_bias.fill_(0.25)
        with self.assertRaisesRegex(RuntimeError, "identity Human Event"):
            head.require_identity_human_probability_calibration()

    def test_identity_calibration_contract_rejects_fitted_map(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        head.set_human_probability_calibration(
            log_temperature=0.0, bias=0.0, count_bias=0.0)
        with self.assertRaisesRegex(RuntimeError, "identity Human Event"):
            head.require_identity_human_probability_calibration()

    def test_calibration_prior_shrinkage_matches_frozen_probability_mix(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        for parameter in head.per_human_network.parameters():
            parameter.data.zero_()
        common = dict(
            feature=torch.zeros(2, 7), action=torch.zeros(2, 4),
            ego_feature=torch.zeros(2, 5),
            human_feature=torch.zeros(2, 1, 6),
            human_root=torch.zeros(2, 1, 10),
            human_quality=torch.zeros(2, 1, 7),
            human_mask=torch.tensor([[True], [False]]),
        )
        # Isolate the visible-person calibration calculation.  The separately
        # supervised unobserved-person branch is intentionally allowed to have
        # nonzero risk even when no visible slot exists.
        with torch.no_grad():
            head.unobserved_human_network[-1].weight.zero_()
            head.unobserved_human_network[-1].bias.fill_(-50.0)
        head.set_human_probability_calibration(
            log_temperature=0.0, bias=0.0, count_bias=0.0,
            shrinkage_weight=0.5, prior_probability=0.1)
        output = head.forward_human_hazard(**common)
        torch.testing.assert_close(
            output["human_collision_probability"],
            torch.tensor([[0.30], [0.05]]), atol=1.0e-6, rtol=0.0)

    def test_empty_human_pool_has_finite_zero_gradient(self):
        head = TransitionEventHead(
            16, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        feature = torch.randn(2, 3, 6, requires_grad=True)
        root = torch.randn(2, 3, 10, requires_grad=True)
        pooled = head.human_geometry_pool(
            human_feature=feature,
            human_root=root,
            human_quality=torch.zeros(2, 3, 7),
            human_mask=torch.zeros(2, 3, dtype=torch.bool),
        )["human_geometry_pool"]
        gradients = torch.autograd.grad(pooled.sum(), (feature, root))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_v20_binary_hazard_does_not_depend_on_global_nonhuman_logits(self):
        head = TransitionEventHead(
            7, 4, 16, ego_feat_dim=5, human_feat_dim=6,
            human_root_dim=10, human_quality_dim=7,
            explicit_human_geometry=True)
        common = dict(
            feature=torch.zeros(2, 7), action=torch.zeros(2, 4),
            ego_feature=torch.zeros(2, 5),
            human_feature=torch.zeros(2, 1, 6),
            human_root=torch.zeros(2, 1, 10),
            human_quality=torch.zeros(2, 1, 7),
            human_mask=torch.ones(2, 1, dtype=torch.bool),
        )
        before = head.forward_human_hazard(**common)[
            "human_collision_probability"]
        with torch.no_grad():
            for parameter in head.non_goal_network.parameters():
                parameter.fill_(1000.0)
        after = head.forward_human_hazard(**common)[
            "human_collision_probability"]
        torch.testing.assert_close(before, after)
        survival = head.forward_human_hazard(**common)[
            "human_survival_probability"]
        torch.testing.assert_close(before + survival, torch.ones_like(before))

    def test_clearance_and_safety_value_are_bounded(self):
        clearance = NextHumanClearanceHead(7, 4, 16)
        safety = HumanSafetyCritic(7, 16)
        feature = torch.randn(6, 7)
        action = torch.randn(6, 4)
        self.assertTrue(bool((clearance(feature, action) >= 0.0).all()))
        probability = safety(feature)
        self.assertTrue(bool(((probability >= 0.0) & (probability <= 1.0)).all()))

    def test_survival_return_does_not_double_count_overlapping_hazards(self):
        hazard = torch.tensor([[[0.2], [0.3], [0.4]]])
        continuation = torch.tensor([[[0.8], [0.7], [0.6]]])
        bootstrap = torch.tensor([[0.5]])
        result = survival_human_collision_return(
            hazard, continuation, bootstrap)
        # q2=.4+.6*.5=.7; q1=.3+.7*.7=.79; q0=.2+.8*.79=.832
        torch.testing.assert_close(
            result,
            torch.tensor([[[0.832], [0.79], [0.70]]]),
        )
        self.assertTrue(bool(((result >= 0.0) & (result <= 1.0)).all()))


if __name__ == "__main__":
    unittest.main()
