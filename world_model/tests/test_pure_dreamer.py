from __future__ import annotations

import copy
import dataclasses
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.transition_event_head import (  # noqa: E402
    NextHumanClearanceHead,
    TransitionEventHead,
)
from modules.action_adapter import HorizontalActionAdapter  # noqa: E402
from modules.action_smoother import (  # noqa: E402
    ActionSmootherConfig,
    SOFT_ABSOLUTE_TARGET,
    StatefulActionSmoother,
)
from modules.reward_components import symexp as reward_component_symexp  # noqa: E402
from modules.factorized_actor_v62 import (  # noqa: E402
    FactorizedActorV62,
    FactorizedActorV62Config,
    PermutationInvariantDecisionHead,
)
from modules.factorized_encoders import (  # noqa: E402
    FactorizedEncoderConfig,
    FactorizedObservationEncoder,
)
from modules.latent_policy_attention import (  # noqa: E402
    ActionTokenLatentAttention,
    LatentPolicyAttentionConfig,
)
from modules.factorized_prediction_heads import (  # noqa: E402
    FactorizedPredictionConfig,
    FactorizedPredictionHeads,
)
from modules.task_geometry import task_physical_state_dim  # noqa: E402
from pure_dreamer import PureDreamerConfig, PureDreamerTrainer  # noqa: E402
from pure_dreamer.trainer import (  # noqa: E402
    _all_floating_tensors_finite,
    _clip_grad_norm_with_float64_fallback_,
)
from test_factorized_dreamer import make_batch, make_model  # noqa: E402


def pure_model(*, action: int = 4, overshoot=()):
    """Build a small but contract-complete version of the production model.

    Algorithm tests used to instantiate a legacy joint-token Actor and then
    rely on production validation being permissive.  That made the suite
    incapable of exercising the exact Section-10 state path.  Keep dimensions
    small here, but preserve the production applied4 -> policy3 adapter,
    unified Actor/Critic state, articulated Human state, task geometry/memory,
    structured Human rollout, and learned categorical reset semantics.
    """
    if int(action) != 4:
        raise ValueError("current pure Dreamer tests require applied action4")
    feature = 32
    human_slots = 3
    human_fields = 103
    task_obstacle_slots = 256
    model = make_model(
        embed=feature,
        action=4,
        joint=feature,
        overshoot=(1, 5, 10, 15),
    )
    model.encoder = FactorizedObservationEncoder(FactorizedEncoderConfig(
        model_dim=feature,
        ego_metric_scaling=True,
        human_root_dim=10,
        human_root_metric_scaling=True,
        human_quality_metric_scaling=True,
        human_feat_dim=7,
        use_stgcn=True,
        pose_history=8,
        observation_heads=4,
        observation_ff_mult=2,
    ))
    model.rssm.latent_policy_attention = ActionTokenLatentAttention(
        model.rssm.ego_rssm.feat_size,
        model.rssm.human_rssm.feat_size,
        8,
        LatentPolicyAttentionConfig(
            model_dim=feature,
            num_heads=4,
            num_layers=1,
            ff_mult=2,
            goal_metric_scaling=True,
        ),
    )
    model.prediction_heads = FactorizedPredictionHeads(
        model.rssm.ego_rssm.feat_size,
        model.rssm.human_rssm.feat_size,
        FactorizedPredictionConfig(
            hidden_dim=32,
            kinematic_velocity_only=True,
            max_velocity_residual_mps=0.70,
            max_root_speed_mps=2.50,
            kinematic_joint_velocity_only=True,
            max_joint_velocity_residual_mps=1.50,
            max_joint_speed_mps=6.00,
        ),
    )
    feature = model.rssm.joint_feat_size
    ego_feature = model.rssm.ego_rssm.feat_size
    human_feature = model.rssm.human_rssm.feat_size
    model.transition_event = TransitionEventHead(
        feature, 4,
        hidden_dim=32,
        ego_feat_dim=ego_feature,
        human_feat_dim=human_feature,
        human_root_dim=10,
        human_quality_dim=7,
        explicit_human_geometry=True,
        geometry_topk_physical_slots=2,
        explicit_joint_kinematics=True,
        explicit_human_presence_physical=True,
        analytic_task_memory_events=True,
        actor_full_state_slots=human_slots,
        actor_full_fields_per_slot=human_fields,
        actor_full_learned_per_slot=True,
        event_full_articulated_state=True,
    )
    model.next_human_clearance = NextHumanClearanceHead(
        feature, 4,
        hidden_dim=32,
        ego_feat_dim=ego_feature,
        human_feat_dim=human_feature,
        human_root_dim=10,
        human_quality_dim=7,
        explicit_human_geometry=True,
    )
    model.action_adapter = HorizontalActionAdapter()
    model.action_smoother = StatefulActionSmoother(ActionSmootherConfig(
        time_constant_s=0.10,
        slew_rate_per_s=(3.0, 3.0, 0.45),
        parameterization=SOFT_ABSOLUTE_TARGET))
    model.uses_internal_action_adapter = True
    model.task_geometry_enabled = True
    model.direct_task_geometry_enabled = True
    model.task_memory_enabled = True
    model.direct_task_memory_enabled = True
    model.actor_authoritative_ego_token_enabled = True
    model.actor_direct_ego_task_state_enabled = True
    model.actor_explicit_human_geometry_enabled = True
    model.actor_task_physical_state_enabled = True
    model.actor_task_physical_obstacle_slots = task_obstacle_slots
    model.critic_full_state_enabled = True
    model.deterministic_evaluation_state_enabled = True
    model.conservative_human_clearance_enabled = False
    model.human_only_overshooting = True
    model.require_gt_displacement_supervision = True
    model.collision_tail_regret_supervision = True
    human_geometry_dim = model.transition_event.actor_full_state_dim
    task_geometry_dim = task_physical_state_dim(task_obstacle_slots)
    model.actor = FactorizedActorV62(FactorizedActorV62Config(
        token_dim=feature,
        hidden_dim=32,
        policy_action_dim=3,
        goal_output_scale=0.01,
        min_std=(0.03, 0.05, 0.02),
        initial_std=(0.12, 0.18, 0.10),
        max_std=(0.35, 0.50, 0.35),
        std_parameterization="learned_bounded",
        pre_tanh_mean_bound=3.0,
        pre_tanh_mean_parameterization="algebraic_sqrt",
        human_physical_slots=human_slots,
        human_physical_fields_per_slot=human_fields,
        human_physical_presence_state=True,
        human_geometry_dim=human_geometry_dim,
        task_geometry_dim=task_geometry_dim,
        unified_policy=True,
        permutation_invariant_entities=True,
        human_learned_fields_per_slot=(
            model.transition_event.actor_full_learned_fields_per_slot),
        task_physical_obstacle_slots=task_obstacle_slots,
    ))
    critic_encoder = copy.deepcopy(model.actor.decision_encoder)
    model.value = PermutationInvariantDecisionHead(
        critic_encoder, type(model.value)(critic_encoder.output_dim))
    model.slow_value = copy.deepcopy(model.value)
    for parameter in model.slow_value.parameters():
        parameter.requires_grad_(False)
    return model


def pure_batch(*, action: int = 4, t: int = 16):
    if int(action) != 4:
        raise ValueError("current pure Dreamer tests require applied action4")
    batch = make_batch(action=4, t=t)
    batch["action"] = (0.10 * batch["action"]).clamp(-0.5, 0.5)
    batch["ego_state"][..., 3:5].clamp_(-1.0, 1.0)
    batch["ego_state"][..., 2] = 1.0
    batch["ego_state"][..., 9] = 1.0
    root_position = batch["human_root"][..., :3]
    root_velocity = batch["skeleton"][..., 3:6].mean(-2)
    batch["human_root"] = torch.cat((
        root_position,
        root_velocity,
        torch.zeros(*root_position.shape[:-1], 4),
    ), -1)
    batch["human_observation_quality"] = torch.zeros(
        *batch["human_root"].shape[:-1], 7)
    batch["reward_components"] = torch.zeros(*batch["reward"].shape[:-1], 6)
    batch["transition_event_target"] = torch.zeros_like(
        batch["reward"], dtype=torch.long)
    batch["transition_event_target"][0, 1, 0] = 1
    batch["transition_event_valid"] = torch.ones_like(
        batch["reward"], dtype=torch.bool)
    batch["next_human_clearance_m"] = torch.ones_like(batch["reward"])
    batch["next_human_clearance_valid"] = torch.ones_like(
        batch["reward"], dtype=torch.bool)
    b, sequence, people = batch["human_mask"].shape
    leading = (b, sequence)
    batch["sequence_valid"] = torch.ones(
        *leading, 1, dtype=torch.bool)
    batch["physical_is_first"] = batch["is_first"].bool().unsqueeze(-1)
    batch["dt_s"] = torch.full((*leading, 1), 0.10)
    batch["termination_code"] = torch.zeros(
        *leading, 1, dtype=torch.long)
    batch["human_motion_valid"] = batch["human_mask"].clone()
    batch["human_survival_valid"] = batch["human_mask"].clone()
    batch["human_survival_target"] = batch["human_mask"].clone()
    batch["human_birth_target"] = batch["human_is_first"].clone()
    batch["human_birth_valid"] = ~batch["human_mask"].clone()
    batch["measured_root_target"] = batch["human_root"][..., :6].clone()
    batch["measured_root_target_valid"] = batch["human_mask"].clone()
    batch["measured_velocity_target_valid"] = batch["human_mask"].clone()
    batch["measured_joint_target"] = batch["skeleton"][..., :3].clone()
    batch["measured_joint_target_valid"] = batch["joint_mask"].clone()

    identities = torch.arange(people).reshape(1, 1, people).expand(
        b, sequence, people).clone()
    batch["human_gt_id"] = identities
    batch["human_gt_identity_id"] = identities.clone()
    batch["human_gt_match_valid"] = batch["human_mask"].clone()
    batch["human_gt_identity_valid"] = batch["human_mask"].clone()
    batch["human_gt_match_error_m"] = torch.zeros(
        b, sequence, people)
    batch["human_gt_pelvis_body"] = batch["human_root"][..., :3].clone()
    batch["human_gt_pelvis_episode"] = (
        batch["human_root"][..., :3]
        + batch["ego_state"][..., None, :3]
    )
    batch["priv_human_id"] = identities.clone()
    batch["priv_human_mask"] = batch["human_mask"].clone()
    batch["priv_collision_human_id"] = torch.full(
        (*leading, 1), -1, dtype=torch.long)
    batch["future_collision_human_id"] = torch.full(
        (*leading, 1), -1, dtype=torch.long)
    batch["priv_collision_joints_episode"] = (
        batch["skeleton"][..., :10, :3]
        + batch["ego_state"][..., None, None, :3]
    )
    batch["priv_collision_joint_valid"] = batch[
        "joint_mask"][..., :10].clone()
    batch["privileged_geometry_available"] = torch.ones(
        *leading, 1, dtype=torch.bool)
    batch["counterfactual_collision_surface_radii_m"] = torch.tensor(
        (0.14, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.12),
    ).reshape(1, 1, 10).expand(b, sequence, 10).clone()
    batch["counterfactual_collision_contact_offset_m"] = torch.full(
        (*leading, 1), 0.02)

    batch["counterfactual_origin_xyz"] = torch.zeros(*leading, 3)
    batch["counterfactual_origin_yaw"] = torch.zeros(*leading, 1)
    batch["counterfactual_flight_bounds_world"] = torch.tensor(
        (-20.0, 20.0, -20.0, 20.0),
    ).reshape(1, 1, 4).expand(*leading, 4).clone()
    batch["counterfactual_bounds_valid"] = torch.ones(
        *leading, 1, dtype=torch.bool)
    batch["counterfactual_maximum_altitude_m"] = torch.full(
        (*leading, 1), 5.0)
    batch["counterfactual_maximum_altitude_valid"] = torch.ones(
        *leading, 1, dtype=torch.bool)
    batch["counterfactual_static_obstacle_aabbs_world"] = torch.zeros(
        *leading, 1, 4)
    batch["counterfactual_static_obstacle_valid"] = torch.zeros(
        *leading, 1, dtype=torch.bool)
    batch["counterfactual_static_terminal_valid"] = torch.ones(
        *leading, 1, dtype=torch.bool)
    batch["counterfactual_goal_radius_m"] = torch.ones(*leading, 1)
    batch["counterfactual_crash_altitude_m"] = torch.full(
        (*leading, 1), 0.15)
    batch["counterfactual_crash_min_elapsed_s"] = torch.zeros(*leading, 1)
    batch["counterfactual_watchdog_min_elapsed_s"] = torch.full(
        (*leading, 1), 5.0)
    batch["counterfactual_watchdog_no_progress_timeout_s"] = torch.full(
        (*leading, 1), 30.0)
    batch["counterfactual_watchdog_progress_epsilon_m"] = torch.full(
        (*leading, 1), 0.25)
    batch["counterfactual_watchdog_stuck_max_horizontal_speed_mps"] = (
        torch.full((*leading, 1), 0.15))
    batch["task_geometry"] = torch.zeros(*leading, 8)
    batch["task_memory"] = torch.zeros(*leading, 9)
    batch["task_memory"][..., 0] = (
        torch.arange(sequence).reshape(1, sequence).expand(b, -1) * 0.10)
    batch["task_memory"][..., 1] = torch.linalg.vector_norm(
        batch["goal_position"] - batch["ego_state"][..., :3], dim=-1)
    batch["task_memory"][..., 6:9] = (
        HorizontalActionAdapter.policy_from_applied(batch["action"]))
    return batch


class PureDreamerTest(unittest.TestCase):
    def test_grouped_finite_check_rejects_nan_and_inf(self):
        self.assertTrue(_all_floating_tensors_finite((
            torch.tensor([1.0, -2.0]),
            torch.tensor(3, dtype=torch.int64),
        )))
        self.assertFalse(_all_floating_tensors_finite((
            torch.tensor([1.0, float("nan")]),
            torch.tensor([2.0]),
        )))
        self.assertFalse(_all_floating_tensors_finite((
            torch.tensor([float("inf")]),
        )))

    def test_gradient_clip_uses_float64_only_for_finite_norm_overflow(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([1.0e20, -1.0e20])
        total_norm, used_fallback = _clip_grad_norm_with_float64_fallback_(
            (parameter,), 1.0)
        self.assertTrue(used_fallback)
        self.assertTrue(bool(torch.isfinite(total_norm)))
        self.assertGreater(float(total_norm), 1.0e20)
        torch.testing.assert_close(
            torch.linalg.vector_norm(parameter.grad), torch.tensor(1.0),
            rtol=1.0e-5, atol=1.0e-6)

    def test_gradient_clip_rejects_actual_nonfinite_elements(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        parameter.grad = torch.tensor([float("inf")])
        with self.assertRaises(RuntimeError):
            _clip_grad_norm_with_float64_fallback_((parameter,), 1.0)

    def test_optimizer_specific_gradient_caps_fit_common_ceiling(self):
        config = PureDreamerConfig()
        self.assertEqual(config.world_grad_clip, 50.0)
        self.assertEqual(config.actor_grad_clip, 5.0)
        self.assertEqual(config.critic_grad_clip, 100.0)
        with self.assertRaisesRegex(
            ValueError, "cannot exceed grad_clip",
        ):
            PureDreamerConfig(world_grad_clip=101.0)

    def test_imagined_human_terminal_uses_learned_swept_joint_contact(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        joints = torch.zeros(1, 2, 1, 12, 3)
        joints[:, 0, ..., 0] = 1.0
        joints[:, 1, ..., 0] = 0.10
        ego = torch.zeros(1, 2, 14)
        ego[..., 13] = 1.0
        imagined = {
            "ego_state": ego,
            "human_joints_body": joints,
            "human_joint_velocity_body": torch.zeros_like(joints),
            "human_joint_mask": torch.ones(1, 1, 12, dtype=torch.bool),
            "human_mask_rollout": torch.ones(1, 2, 1, dtype=torch.bool),
        }
        terminal, fraction, gap, valid = (
            trainer._imagined_swept_human_contact(imagined))
        torch.testing.assert_close(terminal, torch.ones_like(terminal))
        torch.testing.assert_close(valid, torch.ones_like(valid))
        self.assertGreater(float(fraction), 0.0)
        self.assertLess(float(fraction), 1.0)
        self.assertLessEqual(float(gap), 0.0)

    def test_imagined_human_sweep_stops_at_earlier_task_terminal(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        joints = torch.zeros(1, 2, 1, 12, 3)
        joints[:, 0, ..., 0] = 1.0
        joints[:, 1, ..., 0] = 0.10
        joints.requires_grad_()
        ego = torch.zeros(1, 2, 14)
        ego[..., 13] = 1.0
        imagined = {
            "ego_state": ego,
            "human_joints_body": joints,
            "human_joint_velocity_body": torch.zeros_like(joints),
            "human_joint_mask": torch.ones(1, 1, 12, dtype=torch.bool),
            "human_mask_rollout": torch.ones(1, 2, 1, dtype=torch.bool),
            "analytic_terminal": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]),
            "analytic_terminal_first_fraction": torch.tensor([[[0.20]]]),
        }

        terminal, fraction, gap, valid = (
            trainer._imagined_swept_human_contact(imagined))

        torch.testing.assert_close(terminal, torch.zeros_like(terminal))
        torch.testing.assert_close(valid, torch.ones_like(valid))
        torch.testing.assert_close(fraction, torch.ones_like(fraction))
        self.assertGreater(float(gap.detach()), 0.0)
        (fraction.mean() + gap.mean()).backward()
        self.assertTrue(bool(torch.isfinite(joints.grad).all()))


    def test_task_memory_and_previous_applied_action_are_one_state(self):
        trainer = object.__new__(PureDreamerTrainer)
        trainer.model = SimpleNamespace(
            task_memory_enabled=True,
            policy_from_applied_action=(
                HorizontalActionAdapter.policy_from_applied),
        )
        action = torch.tensor(((0.2, -0.3, 0.1, 0.4),))
        memory = torch.zeros(1, 9)
        memory[:, 6:9] = torch.tensor(((0.2, -0.3, 0.4),))
        values = {"previous_action": action, "task_memory": memory}
        trainer._require_task_memory_action_contract(
            values, context="test")

        values["task_memory"] = memory.clone()
        values["task_memory"][:, 7] = 0.1
        with self.assertRaisesRegex(
            ValueError, "encode different previous smoother states"
        ):
            trainer._require_task_memory_action_contract(
                values, context="test")

    def test_counterfactual_clearance_ranking_uses_every_ordered_pair(self):
        truth = torch.tensor([[0.00, 0.20, 0.10]])
        valid = torch.ones_like(truth, dtype=torch.bool)
        correctly_ranked = truth.clone().requires_grad_(True)
        correct = PureDreamerTrainer._all_pair_counterfactual_clearance_ranking(
            correctly_ranked,
            truth,
            valid,
            minimum_truth_margin_m=0.05,
            required_prediction_margin_m=0.05,
            normalization_m=0.10,
        )
        self.assertEqual(float(correct[1]), 3.0)
        self.assertEqual(float(correct[2]), 1.0)
        torch.testing.assert_close(correct[0], torch.zeros_like(correct[0]))

        reversed_order = torch.tensor(
            [[0.20, 0.00, 0.10]], requires_grad=True)
        reversed_result = (
            PureDreamerTrainer._all_pair_counterfactual_clearance_ranking(
                reversed_order,
                truth,
                valid,
                minimum_truth_margin_m=0.05,
                required_prediction_margin_m=0.05,
                normalization_m=0.10,
            )
        )
        self.assertEqual(float(reversed_result[1]), 3.0)
        self.assertEqual(float(reversed_result[2]), 0.0)
        self.assertGreater(float(reversed_result[0].detach()), 0.0)
        reversed_result[0].backward()
        self.assertTrue(bool(torch.isfinite(reversed_order.grad).all()))
        self.assertGreater(float(reversed_order.grad.abs().sum()), 0.0)

    def test_counterfactual_clearance_ranking_masks_ties_and_invalids(self):
        truth = torch.tensor([[0.00, 0.03, 0.20, 0.40]])
        prediction = torch.tensor(
            [[0.00, 0.05, 0.10, torch.inf]], requires_grad=True)
        valid = torch.tensor([[True, True, True, False]])
        result = PureDreamerTrainer._all_pair_counterfactual_clearance_ranking(
            prediction,
            truth,
            valid,
            minimum_truth_margin_m=0.05,
            required_prediction_margin_m=0.05,
            normalization_m=0.10,
        )
        # The near-tie (0,1) and every pair containing candidate 3 are absent.
        self.assertEqual(float(result[1]), 2.0)
        self.assertEqual(float(result[2]), 1.0)
        result[0].backward()
        self.assertEqual(float(prediction.grad[0, 3]), 0.0)

    def test_counterfactual_clearance_ranking_is_candidate_permutation_invariant(self):
        truth = torch.tensor([
            [0.00, 0.20, 0.10, -0.10],
            [0.40, 0.10, 0.30, 0.20],
        ])
        prediction = torch.tensor([
            [0.05, 0.08, 0.15, -0.05],
            [0.20, 0.40, 0.10, 0.30],
        ])
        valid = torch.tensor([
            [True, True, True, True],
            [True, False, True, True],
        ])

        def rank(predict, target, mask):
            return PureDreamerTrainer._all_pair_counterfactual_clearance_ranking(
                predict,
                target,
                mask,
                minimum_truth_margin_m=0.05,
                required_prediction_margin_m=0.05,
                normalization_m=0.10,
            )

        original = rank(prediction, truth, valid)
        order = torch.tensor([2, 0, 3, 1])
        permuted = rank(
            prediction[:, order], truth[:, order], valid[:, order])
        for left, right in zip(original, permuted):
            torch.testing.assert_close(left, right)

    def test_counterfactual_clearance_cvar_exposes_severe_state_local_tail(self):
        truth = torch.tensor([
            [0.00, 0.20, 0.10],
            [0.00, 0.20, 0.10],
        ])
        prediction = torch.tensor([
            [0.00, 0.20, 0.10],
            [0.20, 0.00, 0.10],
        ], requires_grad=True)
        valid = torch.ones_like(truth, dtype=torch.bool)
        loss, state_count, shortfall = (
            PureDreamerTrainer._statewise_counterfactual_clearance_cvar(
                prediction,
                truth,
                valid,
                minimum_truth_margin_m=0.05,
                required_prediction_margin_m=0.05,
                normalization_m=0.10,
                tail_fraction=0.25,
            )
        )
        self.assertEqual(float(state_count), 2.0)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(shortfall.detach()), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(prediction.grad[1].abs().sum()), 0.0)

        order = torch.tensor([2, 0, 1])
        permuted = PureDreamerTrainer._statewise_counterfactual_clearance_cvar(
            prediction.detach()[:, order],
            truth[:, order],
            valid[:, order],
            minimum_truth_margin_m=0.05,
            required_prediction_margin_m=0.05,
            normalization_m=0.10,
            tail_fraction=0.25,
        )
        torch.testing.assert_close(loss.detach(), permuted[0])
        torch.testing.assert_close(state_count, permuted[1])
        torch.testing.assert_close(shortfall, permuted[2])

    def test_counterfactual_clearance_risk_mixture_preserves_total_mass(self):
        population = torch.tensor(2.0, requires_grad=True)
        tail = torch.tensor(6.0, requires_grad=True)
        mixture = PureDreamerTrainer._convex_counterfactual_clearance_ranking(
            population, tail, tail_weight=0.25)

        torch.testing.assert_close(mixture, torch.tensor(3.0))
        mixture.backward()
        torch.testing.assert_close(population.grad, torch.tensor(0.75))
        torch.testing.assert_close(tail.grad, torch.tensor(0.25))
        torch.testing.assert_close(
            PureDreamerTrainer._convex_counterfactual_clearance_ranking(
                population.detach(), tail.detach(), tail_weight=0.0),
            population.detach(),
        )
        torch.testing.assert_close(
            PureDreamerTrainer._convex_counterfactual_clearance_ranking(
                population.detach(), tail.detach(), tail_weight=1.0),
            tail.detach(),
        )
        with self.assertRaisesRegex(ValueError, "tail weight"):
            PureDreamerTrainer._convex_counterfactual_clearance_ranking(
                population.detach(), tail.detach(), tail_weight=1.01)

    def test_candidate_human_exogeneity_is_symmetric_and_differentiable(self):
        positions = torch.zeros(2, 4, 3, 2, 3)
        positions[:, 1, :, :, 0] = 0.20
        positions[:, 2, :, :, 1] = -0.10
        positions.requires_grad_(True)
        valid = torch.ones(2, 4, 3, 2, dtype=torch.bool)

        original = (
            PureDreamerTrainer._candidate_episode_trajectory_exogeneity(
                positions, valid, normalization_m=0.10))
        self.assertGreater(float(original[0].detach()), 0.0)
        self.assertEqual(float(original[1]), 48.0)
        self.assertGreater(float(original[2]), 0.0)
        self.assertGreater(float(original[3]), float(original[2]))

        order = torch.tensor([2, 0, 3, 1])
        permuted = (
            PureDreamerTrainer._candidate_episode_trajectory_exogeneity(
                positions[:, order].detach(), valid[:, order],
                normalization_m=0.10))
        for left, right in zip(original, permuted):
            torch.testing.assert_close(left.detach(), right.detach())

        common_translation = torch.randn(2, 1, 3, 1, 3)
        translated = (
            PureDreamerTrainer._candidate_episode_trajectory_exogeneity(
                positions.detach() + common_translation,
                valid,
                normalization_m=0.10))
        for left, right in zip(original, translated):
            torch.testing.assert_close(left.detach(), right.detach())

        original[0].backward()
        self.assertTrue(bool(torch.isfinite(positions.grad).all()))
        self.assertGreater(float(positions.grad.abs().sum()), 0.0)

    def test_candidate_human_exogeneity_requires_common_validity(self):
        positions = torch.zeros(1, 3, 2, 1, 3, requires_grad=True)
        with torch.no_grad():
            positions[0, 2, 1, 0, 0] = 100.0
        valid = torch.ones(1, 3, 2, 1, dtype=torch.bool)
        valid[0, 0, 1, 0] = False

        loss, count, mean_m, maximum_m = (
            PureDreamerTrainer._candidate_episode_trajectory_exogeneity(
                positions, valid, normalization_m=0.10))

        torch.testing.assert_close(loss, torch.zeros_like(loss))
        self.assertEqual(float(count), 3.0)
        torch.testing.assert_close(mean_m, torch.zeros_like(mean_m))
        torch.testing.assert_close(maximum_m, torch.zeros_like(maximum_m))
        loss.backward()
        self.assertEqual(float(positions.grad.abs().sum()), 0.0)

    def test_counterfactual_human_exogeneity_scale_is_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "exogeneity scale"):
            PureDreamerConfig(counterfactual_human_exogeneity_scale=-0.1)

    @staticmethod
    def _counterfactual_start_test_trainer():
        trainer = object.__new__(PureDreamerTrainer)
        trainer.config = SimpleNamespace(
            imagination_horizon=2,
            human_safe_clearance_m=0.70,
            counterfactual_safety_max_starts=8,
        )
        trainer.model = SimpleNamespace(
            action_smoother=StatefulActionSmoother(ActionSmootherConfig()),
            policy_from_applied_action=(
                HorizontalActionAdapter.policy_from_applied),
        )
        trainer._matched_gt_future_clearance = lambda batch, **_: (
            torch.full((batch["action"].shape[0], 2), 0.25),
            torch.ones(batch["action"].shape[0], 2, dtype=torch.bool),
            torch.full((batch["action"].shape[0], 2, 2), 0.25),
        )
        return trainer

    def test_counterfactual_start_requires_reachable_destination_action(self):
        trainer = self._counterfactual_start_test_trainer()
        action = torch.zeros(1, 4, 4)
        action[0, :, 0] = torch.tensor((0.0, 0.20, 0.21, 0.22))
        batch = {
            "action": action,
            "action_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "physical_is_first": torch.zeros(1, 4, 1, dtype=torch.bool),
            "privileged_geometry_available": torch.ones(
                1, 4, 1, dtype=torch.bool),
            "human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "joint_mask": torch.ones(1, 4, 1, 12, dtype=torch.bool),
            "is_last": torch.zeros(1, 4, 1, dtype=torch.bool),
            "sequence_valid": torch.ones(1, 4, 1, dtype=torch.bool),
        }
        batch.update({
            "priv_human_id": torch.zeros(1, 4, 1, dtype=torch.long),
            "priv_human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "priv_collision_joint_valid": torch.ones(
                1, 4, 1, 10, dtype=torch.bool),
            "priv_collision_joints_episode": torch.zeros(1, 4, 1, 10, 3),
            "counterfactual_collision_surface_radii_m": torch.full(
                (1, 4, 10), 0.25),
            "counterfactual_collision_contact_offset_m": torch.full(
                (1, 4, 1), 0.02),
            "human_gt_id": torch.zeros(1, 4, 1, dtype=torch.long),
            "human_gt_match_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "human_gt_match_error_m": torch.zeros(1, 4, 1),
        })
        flat, _, time, depth = (
            trainer._counterfactual_danger_start_indices(batch))
        # t=0 -> t=1 jumps by 0.20, beyond the current 0.12 slew
        # limit.  t=1 -> t=2 is exactly reconstructible.
        self.assertEqual(flat.tolist(), [1])
        self.assertEqual(time.tolist(), [1])
        self.assertEqual(depth.tolist(), [1])

    def test_counterfactual_start_validates_destination_not_only_source(self):
        trainer = self._counterfactual_start_test_trainer()
        batch = {
            "action": torch.zeros(1, 4, 4),
            "action_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "physical_is_first": torch.zeros(1, 4, 1, dtype=torch.bool),
            "privileged_geometry_available": torch.ones(
                1, 4, 1, dtype=torch.bool),
            "human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "joint_mask": torch.ones(1, 4, 1, 12, dtype=torch.bool),
            "is_last": torch.zeros(1, 4, 1, dtype=torch.bool),
            "sequence_valid": torch.ones(1, 4, 1, dtype=torch.bool),
        }
        batch.update({
            "priv_human_id": torch.zeros(1, 4, 1, dtype=torch.long),
            "priv_human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "priv_collision_joint_valid": torch.ones(
                1, 4, 1, 10, dtype=torch.bool),
            "priv_collision_joints_episode": torch.zeros(1, 4, 1, 10, 3),
            "counterfactual_collision_surface_radii_m": torch.full(
                (1, 4, 10), 0.25),
            "counterfactual_collision_contact_offset_m": torch.full(
                (1, 4, 1), 0.02),
            "human_gt_id": torch.zeros(1, 4, 1, dtype=torch.long),
            "human_gt_match_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "human_gt_match_error_m": torch.zeros(1, 4, 1),
        })
        batch["action_valid"][0, 1] = False
        flat, _, time, depth = (
            trainer._counterfactual_danger_start_indices(batch))
        # Missing row 1 invalidates t=0 as a destination and t=1 as the
        # previous smoother state.  It must not be mistaken for a valid
        # source-only action contract.
        self.assertEqual(flat.numel(), 0)
        self.assertEqual(time.numel(), 0)
        self.assertEqual(depth.numel(), 0)

    def test_counterfactual_starts_cover_early_warning_and_severity(self):
        trainer = self._counterfactual_start_test_trainer()
        trainer.config.counterfactual_safety_max_starts = 2
        trainer._matched_gt_future_clearance = lambda batch, **_: (
            torch.tensor([[0.20, 0.10]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[[1.00, 0.20], [0.10, 0.10]]]),
        )
        batch = {
            "action": torch.zeros(1, 4, 4),
            "action_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "physical_is_first": torch.zeros(1, 4, 1, dtype=torch.bool),
            "privileged_geometry_available": torch.ones(
                1, 4, 1, dtype=torch.bool),
            "human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "joint_mask": torch.ones(1, 4, 1, 12, dtype=torch.bool),
            "is_last": torch.zeros(1, 4, 1, dtype=torch.bool),
            "sequence_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "priv_human_id": torch.zeros(1, 4, 1, dtype=torch.long),
            "priv_human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "priv_collision_joint_valid": torch.ones(
                1, 4, 1, 10, dtype=torch.bool),
            "priv_collision_joints_episode": torch.zeros(1, 4, 1, 10, 3),
            "counterfactual_collision_surface_radii_m": torch.full(
                (1, 4, 10), 0.25),
            "counterfactual_collision_contact_offset_m": torch.full(
                (1, 4, 1), 0.02),
            "human_gt_id": torch.zeros(1, 4, 1, dtype=torch.long),
            "human_gt_match_valid": torch.ones(1, 4, 1, dtype=torch.bool),
            "human_gt_match_error_m": torch.zeros(1, 4, 1),
        }

        flat, _, time, depth = (
            trainer._counterfactual_danger_start_indices(batch))

        self.assertEqual(flat.tolist(), [0, 1])
        self.assertEqual(time.tolist(), [0, 1])
        self.assertEqual(depth.tolist(), [2, 1])

    def test_counterfactual_starts_cover_distinct_sequences_before_severity(self):
        trainer = self._counterfactual_start_test_trainer()
        trainer.config.counterfactual_safety_max_starts = 3
        trainer._matched_gt_future_clearance = lambda batch, **_: (
            torch.tensor([
                [0.20, 0.10],
                [0.30, 0.05],
                [0.25, 0.15],
            ]),
            torch.ones(3, 2, dtype=torch.bool),
            torch.tensor([
                [[1.00, 0.20], [0.10, 0.10]],
                [[1.10, 0.30], [0.05, 0.05]],
                [[0.90, 0.25], [0.15, 0.15]],
            ]),
        )
        batch = {
            "action": torch.zeros(3, 4, 4),
            "action_valid": torch.ones(3, 4, 1, dtype=torch.bool),
            "physical_is_first": torch.zeros(3, 4, 1, dtype=torch.bool),
            "privileged_geometry_available": torch.ones(
                3, 4, 1, dtype=torch.bool),
            "human_mask": torch.ones(3, 4, 1, dtype=torch.bool),
            "joint_mask": torch.ones(3, 4, 1, 12, dtype=torch.bool),
            "is_last": torch.zeros(3, 4, 1, dtype=torch.bool),
            "sequence_valid": torch.ones(3, 4, 1, dtype=torch.bool),
            "priv_human_id": torch.zeros(3, 4, 1, dtype=torch.long),
            "priv_human_mask": torch.ones(3, 4, 1, dtype=torch.bool),
            "priv_collision_joint_valid": torch.ones(
                3, 4, 1, 10, dtype=torch.bool),
            "priv_collision_joints_episode": torch.zeros(
                3, 4, 1, 10, 3),
            "counterfactual_collision_surface_radii_m": torch.full(
                (3, 4, 10), 0.25),
            "counterfactual_collision_contact_offset_m": torch.full(
                (3, 4, 1), 0.02),
            "human_gt_id": torch.zeros(3, 4, 1, dtype=torch.long),
            "human_gt_match_valid": torch.ones(3, 4, 1, dtype=torch.bool),
            "human_gt_match_error_m": torch.zeros(3, 4, 1),
        }

        torch.manual_seed(7)
        flat, batch_index, time, depth = (
            trainer._counterfactual_danger_start_indices(batch))

        self.assertEqual(sorted(batch_index.tolist()), [0, 1, 2])
        self.assertTrue(bool(time.eq(0).all()))
        self.assertTrue(bool(depth.eq(2).all()))
        self.assertEqual(len(set(flat.tolist())), 3)

    def test_goal_radius_is_a_positive_fixed_task_constant(self):
        with self.assertRaisesRegex(ValueError, "goal radius"):
            PureDreamerConfig(goal_radius_m=0.0)
        with self.assertRaisesRegex(ValueError, "goal radius"):
            PureDreamerConfig(goal_radius_m=float("nan"))

    def test_counterfactual_geometry_validity_broadcasts_over_people(self):
        joints = torch.ones(4, 20, 10, dtype=torch.bool)
        source = torch.tensor([True, False, True, True])
        destination = torch.tensor([True, True, False, True])

        valid = PureDreamerTrainer._counterfactual_swept_slot_valid(
            joints, source, destination)

        self.assertEqual(valid.shape, (4, 20))
        self.assertTrue(bool(valid[0].all()))
        self.assertFalse(bool(valid[1].any()))
        self.assertFalse(bool(valid[2].any()))
        self.assertTrue(bool(valid[3].all()))

    def test_geometry_overlap_never_fabricates_a_nonhuman_event_label(self):
        model = pure_model()
        joints = torch.zeros(1, 2, 1, 10, 3)
        joints[..., 0] = 0.10
        batch = {
            "human_gt_id": torch.full((1, 2, 1), 7, dtype=torch.long),
            "human_gt_match_valid": torch.ones(
                1, 2, 1, dtype=torch.bool),
            "human_mask": torch.ones(1, 2, 1, dtype=torch.bool),
            "priv_human_id": torch.full(
                (1, 2, 1), 7, dtype=torch.long),
            "priv_human_mask": torch.ones(1, 2, 1, dtype=torch.bool),
            "priv_collision_joints_episode": joints,
            "priv_collision_joint_valid": torch.ones(
                1, 2, 1, 10, dtype=torch.bool),
            "privileged_geometry_available": torch.ones(
                1, 2, 1, dtype=torch.bool),
            "counterfactual_collision_surface_radii_m": torch.full(
                (1, 2, 10), 0.25),
            "counterfactual_collision_contact_offset_m": torch.full(
                (1, 2, 1), 0.02),
            "priv_collision_human_id": torch.full(
                (1, 2, 1), -1, dtype=torch.long),
            "termination_code": torch.zeros(1, 2, 1, dtype=torch.long),
            "ego_state": torch.zeros(1, 2, 14),
        }
        target, valid, swept_gap = (
            model.privileged_per_human_swept_collision_targets(batch))
        self.assertTrue(bool(valid.item()))
        self.assertFalse(bool(target.item()))
        self.assertLess(float(swept_gap.item()), 0.0)

    def test_empirical_event_prior_removes_random_terminal_logits(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        prior = trainer.initialize_empirical_event_priors({
            "continue": 8226,
            "human_collision": 25,
            "static_collision": 0,
            "reached_goal": 0,
            "other_task_terminal": 12,
            "actor_observable_human_collision": 20,
            "unobserved_human_collision": 5,
            "visible_human_transition_rows": 7000,
            "visible_human_exposure": 12000,
            "visible_human_exposure_s": 1200.0,
        })
        feature = torch.randn(3, model.rssm.joint_feat_size)
        action = torch.randn(3, 4)
        probability = model.transition_event(feature, action)["probability"]
        expected = torch.tensor([
            prior["categorical_probability"][key]
            for key in (
                "continue", "human_collision", "static_collision",
                "reached_goal", "other_task_terminal")
        ], dtype=probability.dtype).expand_as(probability)
        torch.testing.assert_close(probability, expected)
        continuation = model.cont(feature).mean
        torch.testing.assert_close(
            continuation,
            torch.full_like(
                continuation, prior["continuation_probability"]),
        )
        self.assertLess(
            prior["explicit_human_probability"], 0.01)
        self.assertGreater(
            prior["continuation_probability"], 0.99)

    def test_lambda_return(self):
        reward = torch.ones(1, 2, 1)
        continuation = torch.ones_like(reward)
        value = torch.zeros(1, 3, 1)
        result = PureDreamerTrainer.lambda_return(
            reward, continuation, value, discount=1.0, lamb=1.0)
        torch.testing.assert_close(result, torch.tensor([[[2.0], [1.0]]]))

    def test_first_update_jointly_changes_world_actor_and_critic(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        world_before = trainer.world_parameters[0].detach().clone()
        actor_before = next(model.actor.parameters()).detach().clone()
        critic_before = next(model.value.parameters()).detach().clone()
        return_ema_before = model.return_ema.ema_vals.detach().clone()
        metrics = trainer(pure_batch())
        self.assertEqual(metrics["opt/update_count"], 1.0)
        self.assertGreater(metrics["grad/world"], 0.0)
        self.assertGreater(metrics["grad/actor"], 0.0)
        self.assertGreater(metrics["grad/critic"], 0.0)
        self.assertFalse(torch.equal(world_before, trainer.world_parameters[0]))
        self.assertFalse(torch.equal(actor_before, next(model.actor.parameters())))
        self.assertFalse(torch.equal(critic_before, next(model.value.parameters())))
        self.assertFalse(torch.equal(return_ema_before, model.return_ema.ema_vals))

    def test_actor_uses_return_normalized_dynamics_objective(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        metrics = trainer(pure_batch())
        for name in (
            "imag/log_probability",
            "imag/return_005",
            "imag/return_095",
            "imag/return_scale",
            "imag/return_ema_bootstrapped",
            "imag/normalized_return",
            "imag/dynamics_objective",
            "imag/advantage",
            "imag/advantage_std",
        ):
            self.assertIn(name, metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])), name)
        self.assertGreaterEqual(metrics["imag/return_scale"], 1.0)
        self.assertEqual(metrics["imag/return_ema_bootstrapped"], 1.0)

    def test_actor_has_gradient_through_imagined_dynamics(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            max_imagination_starts=4,
            actor_entropy=0.0,
        ))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)

        def future_latent_objective(imagined):
            # The first returned state is context.  These targets begin at the
            # state reached after a sampled Actor action, so a non-zero Actor
            # gradient requires action -> RSSM dynamics differentiation.
            returns = imagined["joint_feat"][:, 1:, :1]
            weights = torch.ones_like(returns)
            zeros = torch.zeros_like(returns)
            return returns, weights, zeros, zeros, {}

        trainer._imagined_objective = future_latent_objective
        _, _, actor_norm, _, _, _ = trainer._actor_critic_update(
            states, auxiliary, batch)
        self.assertGreater(float(actor_norm), 0.0)

    def test_human_birth_probability_is_not_an_actor_cost(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)
            shape = (*imagined["joint_feat"].shape[:-1], 1)
            without_birth = dict(imagined)
            without_birth["human_birth_probability"] = torch.zeros(shape)
            with_birth = dict(imagined)
            with_birth["human_birth_probability"] = torch.ones(shape)
            return_without = trainer._imagined_objective(without_birth)[0]
            return_with = trainer._imagined_objective(with_birth)[0]
        torch.testing.assert_close(return_with, return_without)

    def test_imagination_retains_exact_pre_tanh_policy_sample(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)
        self.assertIn("policy_pre_tanh", imagined)
        torch.testing.assert_close(
            torch.tanh(imagined["policy_pre_tanh"]),
            imagined["policy_action"])

    def test_visibility_survival_cannot_erase_observed_human_safety_geometry(self):
        """A track leaving view is still an obstacle over the short rollout."""
        model = pure_model(action=4)
        # Force the learned lifecycle head to predict immediate disappearance.
        # Safety geometry must nevertheless retain every source-observed slot.
        with torch.no_grad():
            model.prediction_heads.human.survival.weight.zero_()
            model.prediction_heads.human.survival.bias.fill_(-100.0)
        batch = pure_batch(action=4)
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=3, max_imagination_starts=4))
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)

        expected_mask = values["human_mask"][:, None].expand_as(
            imagined["human_mask_rollout"])
        self.assertTrue(torch.equal(
            imagined["human_mask_rollout"], expected_mask))
        torch.testing.assert_close(
            imagined["human_presence"],
            expected_mask.to(imagined["human_presence"]),
        )
        self.assertTrue(torch.isfinite(
            imagined["constant_velocity_human_joint_clearance_next"]
        ).all())

    def test_no_planner_or_teacher_parameter_group(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2))
        owned = (
            trainer.world_parameters
            + trainer.actor_parameters
            + trainer.critic_parameters)
        self.assertEqual(len(owned), len({id(item) for item in owned}))
        for name, parameter in model.named_parameters():
            if name.startswith((
                "action_risk.", "safety_value.", "slow_safety_value.",
                "slow_value.",
            )):
                self.assertFalse(parameter.requires_grad, name)

    def test_natural_human_event_likelihood_is_trained_inside_world_update(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch()
        # transition_event_target is source-aligned; the physical contact and
        # contact-person identity live on its destination row.
        batch["transition_event_target"][0, 1, 0] = 1
        batch["termination_code"][0, 2, 0] = 2
        batch["priv_collision_human_id"][0, 2, 0] = 0
        metrics = trainer(batch)
        # Class/focal reweighting is deliberately disabled because Actor return
        # consumes this probability absolutely.  The natural cause NLL is the
        # optimized and calibrated Event objective.
        self.assertEqual(metrics["world/rare_event_aux"], 0.0)
        self.assertGreater(metrics["world/transition_human_geometry"], 0.0)
        self.assertGreater(
            metrics["event_prediction/geometry_human_support"], 0.0)
        for name in (
            "event_prediction/geometry_human_recall",
            "event_prediction/geometry_human_precision",
            "event_prediction/geometry_human_true_positive",
            "event_prediction/geometry_human_predicted_positive",
        ):
            self.assertIn(name, metrics)

    def test_rare_reward_uses_destination_row_and_macro_terminal_classes(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch(t=5)
        batch["sequence_valid"] = torch.ones(2, 5, 1, dtype=torch.bool)
        batch["transition_event_target"].zero_()
        batch["transition_event_valid"].fill_(True)
        batch["is_last"].zero_()
        batch["reward"].zero_()
        batch["reward_components"].zero_()
        # Source row 1's action produces the Human terminal reward stored on
        # destination row 2. Source row 2 similarly produces a goal at row 3.
        batch["transition_event_target"][0, 1, 0] = 1
        batch["reward"][0, 2, 0] = -120.0
        batch["reward_components"][0, 2, 0] = -120.0
        batch["transition_event_target"][1, 2, 0] = 3
        batch["reward"][1, 3, 0] = 100.0
        batch["reward_components"][1, 3, 0] = 100.0
        _, auxiliary, _ = model.posterior(batch)

        loss, metrics = trainer._rare_reward_loss(
            auxiliary["joint_feat"], batch)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(metrics["class_count"]), 2.0)
        self.assertEqual(float(metrics["transition_count"]), 2.0)
        self.assertGreater(float(metrics["ordinary_transition_count"]), 0.0)
        self.assertAlmostEqual(float(metrics["target_mean"]), -10.0)
        self.assertGreater(float(metrics["scalar_nll"]), 0.0)
        self.assertGreater(float(metrics["component_event"]), 0.0)
        loss.backward()
        self.assertGreater(
            float(model.reward_components.net[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_component_event_prediction_respects_task_support(self):
        model = pure_model()
        feature = torch.zeros(2, 3, model.rssm.joint_feat_size)
        with torch.no_grad():
            model.reward_components.net[-1].weight[0].zero_()
            model.reward_components.net[-1].bias[0] = -20.0
        negative = reward_component_symexp(
            model.reward_components(feature)[..., 0])
        self.assertTrue(bool((negative >= -120.001).all()))
        self.assertTrue(bool((negative < -119.0).all()))

        with torch.no_grad():
            model.reward_components.net[-1].bias[0] = 20.0
        positive = reward_component_symexp(
            model.reward_components(feature)[..., 0])
        self.assertTrue(bool((positive <= 100.001).all()))
        self.assertTrue(bool((positive > 99.0).all()))

    def test_bounded_component_event_keeps_dynamics_gradient(self):
        model = pure_model()
        feature = torch.randn(
            2, 3, model.rssm.joint_feat_size, requires_grad=True)
        event = model.policy_reward_components(feature)[..., 0].sum()
        gradient, = torch.autograd.grad(event, (feature,))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_rare_reward_aux_balances_ordinary_against_terminal_bias(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch(t=5)
        batch["sequence_valid"] = torch.ones(2, 5, 1, dtype=torch.bool)
        batch["transition_event_target"].zero_()
        batch["transition_event_valid"].fill_(True)
        batch["is_last"].zero_()
        batch["reward"].zero_()
        batch["reward_components"].zero_()
        batch["transition_event_target"][0, 1, 0] = 1
        batch["reward"][0, 2, 0] = -120.0
        batch["reward_components"][0, 2, 0] = -120.0
        with torch.no_grad():
            for parameter in model.reward_components.parameters():
                parameter.zero_()
            # -2 lies in the soft clip's near-identity interior. Both
            # Smooth-L1 terms are on opposite unit-slope branches and must
            # cancel under the 50/50 auxiliary instead of pushing every state
            # toward a terminal reward.
            model.reward_components.net[-1].bias[0] = -2.0
            _, auxiliary, _ = model.posterior(batch)

        loss, metrics = trainer._rare_reward_loss(
            auxiliary["joint_feat"], batch)
        loss.backward()

        self.assertEqual(float(metrics["class_count"]), 1.0)
        self.assertGreater(float(metrics["ordinary_transition_count"]), 0.0)
        self.assertAlmostEqual(
            float(model.reward_components.net[-1].bias.grad[0]), 0.0,
            places=6,
        )

    def test_imagination_excludes_learned_reward_components_from_actor_return(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
        imagined = trainer._imagine(initial, values)
        original = model.policy_reward_components
        calls = []

        def fixed_policy_reward_components(feature):
            calls.append(feature)
            result = feature.new_zeros(*feature.shape[:-1], 6)
            result[..., 0] = 99.0
            result[..., 1] = 88.0
            result[..., 2] = 77.0
            result[..., 3] = 2.5
            return result

        model.policy_reward_components = fixed_policy_reward_components
        try:
            _, _, _, _, metrics = trainer._imagined_objective(imagined)
        finally:
            model.policy_reward_components = original
        self.assertEqual(len(calls), 1)
        self.assertEqual(float(metrics["learned_dense_reward"].detach()), 0.0)
        self.assertAlmostEqual(
            float(metrics["reward_component_smoothness_mean"].detach()),
            2.5,
            places=5,
        )
        self.assertIn("scalar_reward_diagnostic", metrics)

    def test_impossible_learned_component_cannot_change_actor_return(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            max_imagination_starts=4,
            imagined_reward_abs_max_limit=150.0,
        ))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
        imagined = trainer._imagine(initial, values)
        baseline = trainer._imagined_objective(imagined)[0]
        original = model.policy_reward_components
        def impossible_components(feature):
            result = feature.new_zeros(*feature.shape[:-1], 6)
            result[..., 3] = 500.0
            return result
        model.policy_reward_components = impossible_components
        try:
            modified, _, _, _, metrics = trainer._imagined_objective(imagined)
        finally:
            model.policy_reward_components = original
        torch.testing.assert_close(modified, baseline)
        self.assertEqual(float(metrics["learned_dense_reward"]), 0.0)
        self.assertEqual(
            float(metrics["reward_component_smoothness_abs_max"]), 500.0)

    def test_component_policy_reward_keeps_actor_dynamics_gradient(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            max_imagination_starts=4,
            actor_entropy=0.0,
        ))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
        imagined = trainer._imagine(initial, values)
        task_reward = trainer._imagined_objective(imagined)[0].sum()
        gradients = torch.autograd.grad(
            task_reward,
            tuple(model.actor.parameters()),
            allow_unused=True,
        )
        gradient_norm = sum(
            float(gradient.abs().sum())
            for gradient in gradients if gradient is not None)
        self.assertGreater(gradient_norm, 0.0)

    def test_actor_saturation_metrics_are_visible(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        metrics = trainer(pure_batch())
        expected = (
            "imag/actor_mode_abs_mean",
            "imag/actor_mode_saturation_ratio",
            "imag/actor_raw_pre_tanh_abs_max",
            "imag/actor_raw_headroom_penalty",
            "imag/actor_raw_headroom_loss",
            "imag/actor_raw_headroom_exceed_fraction",
            "imag/actor_mean_bound_jacobian_mean",
            "imag/actor_mean_bound_jacobian_min",
            "imag/actor_mean_bound_zero_jacobian_fraction",
            "imag/actor_raw_to_mode_jacobian_mean",
            "imag/actor_raw_to_mode_jacobian_min",
            "imag/actor_raw_std_residual_abs_max",
            "imag/actor_std_mean",
            "imag/actor_std_forward_mean",
            "imag/actor_std_lateral_mean",
            "imag/actor_std_yaw_mean",
            "imag/action_smoother_local_scale_mean",
            "imag/action_smoother_nonzero_jacobian_fraction",
            "imag/route_heading_potential_reward",
            "imag/route_heading_error_rad",
            "imag/route_deviation_reward",
        )
        for name in expected:
            self.assertIn(name, metrics)
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])), name)

    def test_actor_raw_headroom_penalty_acts_only_beyond_soft_limit(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            max_imagination_starts=4,
            actor_raw_pre_tanh_soft_limit=6.0,
            actor_raw_pre_tanh_headroom_scale=3.0e-3,
        ))
        with torch.no_grad():
            model.actor.mean_net[-1].weight.zero_()
            model.actor.mean_net[-1].bias.fill_(8.0)
        metrics = trainer(pure_batch())
        self.assertAlmostEqual(
            metrics["imag/actor_raw_headroom_penalty"], 4.0, places=4)
        self.assertAlmostEqual(
            metrics["imag/actor_raw_headroom_loss"], 0.012, places=5)
        self.assertAlmostEqual(
            metrics["imag/actor_raw_headroom_exceed_fraction"], 1.0,
            places=6)
        self.assertAlmostEqual(
            metrics["loss/actor"],
            metrics["imag/actor_dynamics_entropy_loss"]
            + metrics["imag/actor_raw_headroom_loss"],
            places=5,
        )

    def test_actor_pre_tanh_divergence_fails_before_optimizer_step(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4,
            actor_pre_tanh_abs_max_limit=12.0))
        # Exercise the fail-fast path independently of the production soft
        # bound, whose contract is already validated when the trainer is built.
        model.actor.config = dataclasses.replace(
            model.actor.config, pre_tanh_mean_bound=0.0)
        with torch.no_grad():
            model.actor.mean_net[-1].weight.zero_()
            model.actor.mean_net[-1].bias.fill_(20.0)
        # Production installs the replay prior before the first transactional
        # optimizer call, so exclude that explicit initialization step from
        # this rejected-update snapshot.
        trainer.initialize_empirical_event_priors({
            "continue": 10,
            "human_collision": 1,
            "static_collision": 0,
            "reached_goal": 0,
            "other_task_terminal": 0,
            "actor_observable_human_collision": 1,
            "unobserved_human_collision": 0,
            "visible_human_transition_rows": 10,
            "visible_human_exposure": 20,
            "visible_human_exposure_s": 2.0,
        })
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return_ema_before = {
            name: value.detach().clone()
            for name, value in model.return_ema.state_dict().items()
        }
        with self.assertRaisesRegex(
            FloatingPointError, "pre-tanh mean exceeded fail-fast limit"
        ):
            trainer(pure_batch())
        self.assertEqual(trainer.update_count, 0)
        self.assertFalse(trainer.checkpoint_safe)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, parameters_before[name])
        for name, value in model.return_ema.state_dict().items():
            torch.testing.assert_close(value, return_ema_before[name])
        self.assertEqual(len(trainer.world_optimizer.state), 0)
        self.assertEqual(len(trainer.actor_optimizer.state), 0)
        self.assertEqual(len(trainer.critic_optimizer.state), 0)

    def test_actor_soft_bounds_do_not_hide_nonfinite_raw_heads(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        feature = torch.zeros(2, model.actor.input_dim)
        with torch.no_grad():
            model.actor.mean_net[-1].bias.fill_(float("inf"))
        with self.assertRaisesRegex(
            FloatingPointError, "Actor raw mean contains NaN/Inf"
        ):
            trainer._assert_actor_output_guard(
                feature, context="nonfinite raw-mean test")

        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        feature = torch.zeros(2, model.actor.input_dim)
        with torch.no_grad():
            model.actor.std_net[-1].bias.fill_(float("inf"))
        with self.assertRaisesRegex(
            FloatingPointError,
            "Actor raw standard deviation contains NaN/Inf",
        ):
            trainer._assert_actor_output_guard(
                feature, context="nonfinite raw-std test")

    def test_actor_saturation_after_optimizer_step_is_never_publishable(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4,
            actor_pre_tanh_abs_max_limit=4.0))
        model.actor.config = dataclasses.replace(
            model.actor.config, pre_tanh_mean_bound=0.0)
        trainer.initialize_empirical_event_priors({
            "continue": 10,
            "human_collision": 1,
            "static_collision": 0,
            "reached_goal": 0,
            "other_task_terminal": 0,
            "actor_observable_human_collision": 1,
            "unobserved_human_collision": 0,
            "visible_human_transition_rows": 10,
            "visible_human_exposure": 20,
            "visible_human_exposure_s": 2.0,
        })
        original_step = trainer.actor_optimizer.step

        def saturating_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                model.actor.mean_net[-1].bias.fill_(20.0)
            return result

        trainer.actor_optimizer.step = saturating_step
        with self.assertRaisesRegex(
            FloatingPointError, "after optimizer step"
        ):
            trainer(pure_batch())
        self.assertEqual(trainer.update_count, 0)
        self.assertFalse(trainer.checkpoint_safe)
        self.assertTrue(trainer._optimizer_step_started)
        self.assertEqual(len(trainer.critic_optimizer.state), 0)

    def test_imagination_start_uses_physical_reset_and_window_burn_in(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            imagination_window_burn_in=2,
            max_imagination_starts=16,
        ))
        batch = pure_batch(t=4)
        batch["sequence_valid"] = torch.ones(2, 4, 1, dtype=torch.bool)
        batch["physical_is_first"] = torch.zeros(
            2, 4, 1, dtype=torch.bool)
        batch["physical_is_first"][0, 0] = True
        # The reset row legitimately has no preceding applied action.
        batch["action_valid"][0, 0] = False
        indices = trainer._flat_valid_starts(batch)
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 6, 7])

    def test_actor_starts_uniformly_sample_valid_posterior_rows(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            imagination_window_burn_in=0,
            max_imagination_starts=4,
        ))
        batch = pure_batch(t=6)
        batch["goal_position"][..., 0] = 100.0
        valid = trainer._flat_valid_starts(batch)
        self.assertGreater(int(valid.numel()), 4)
        rng = torch.random.get_rng_state()
        expected = valid.index_select(
            0, torch.randperm(valid.numel())[:4])
        torch.random.set_rng_state(rng)
        selected = trainer._sample_imagination_start_indices(batch)
        torch.testing.assert_close(selected, expected)

        # The production state constructor must consume exactly those rows.
        batch["ego_state"][..., 0] = torch.arange(
            batch["ego_state"].shape[0] * batch["ego_state"].shape[1],
            dtype=batch["ego_state"].dtype,
        ).reshape(batch["ego_state"].shape[:2])
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            torch.random.set_rng_state(rng)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
        self.assertEqual(int(initial["ego"]["deter"].shape[0]), 4)
        self.assertEqual(float(values["actor_imagination_start_count"]), 4.0)
        torch.testing.assert_close(
            values["ego_state"][:, 0], expected.to(values["ego_state"]))

    def test_actor_start_stratification_is_uniform_expectation_unbiased(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=15,
            imagination_window_burn_in=0,
            max_imagination_starts=4,
        ))
        batch = pure_batch(t=8)
        batch["goal_position"][..., 0] = 100.0
        valid = trainer._flat_valid_starts(batch)
        self.assertGreaterEqual(int(valid.numel()), 9)
        collision = batch["future_collision"].reshape(-1)
        ttc_valid = batch["time_to_collision_valid"].reshape(-1)
        ttc = batch["time_to_collision_s"].reshape(-1)
        actionable = valid[:4]
        imminent = valid[4:7]
        collision[actionable] = True
        collision[imminent] = True
        ttc_valid[actionable] = True
        ttc_valid[imminent] = True
        ttc[actionable] = min(
            1.5,
            0.5 * (trainer.actor_actionable_response_time_s + 1.5),
        )
        ttc[imminent] = 0.5 * trainer.actor_actionable_response_time_s

        torch.manual_seed(59)
        selected, importance, metrics = trainer._sample_imagination_starts(
            batch)
        selected_set = set(selected.tolist())
        actionable_set = set(actionable.tolist())
        imminent_set = set(imminent.tolist())
        selected_actionable = len(selected_set & actionable_set)
        selected_imminent = len(selected_set & imminent_set)
        selected_ordinary = 4 - selected_actionable - selected_imminent
        self.assertEqual(
            (selected_actionable, selected_imminent, selected_ordinary),
            (2, 1, 1),
        )
        self.assertEqual(
            float(metrics["actor_imagination_actionable_selected_count"]),
            2.0,
        )
        torch.testing.assert_close(importance.mean(), torch.ones(()))

        # With a stratum-constant statistic, the corrected sampled mean is
        # exactly the full uniform replay-state mean for every random draw.
        statistic = torch.full(
            (batch["is_last"].shape[0] * batch["is_last"].shape[1],),
            7.0,
        )
        statistic[actionable] = 1.0
        statistic[imminent] = 3.0
        expected = statistic.index_select(0, valid).mean()
        estimated = (
            statistic.index_select(0, selected) * importance
        ).mean()
        torch.testing.assert_close(estimated, expected)

    def test_actor_start_stratification_covers_near_goal_without_reweighting(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=15,
            imagination_window_burn_in=0,
            max_imagination_starts=4,
        ))
        batch = pure_batch(t=8)
        batch["goal_position"][..., 0] = 100.0
        valid = trainer._flat_valid_starts(batch)
        near_goal = valid[:3]
        flat_goal = batch["goal_position"].reshape(-1, 3)
        flat_ego = batch["ego_state"].reshape(-1, 14)
        flat_goal[near_goal, :3] = flat_ego[near_goal, :3]

        torch.manual_seed(61)
        selected, importance, metrics = trainer._sample_imagination_starts(
            batch)
        self.assertGreaterEqual(
            len(set(selected.tolist()) & set(near_goal.tolist())), 1)
        self.assertEqual(
            float(metrics["actor_imagination_near_goal_population_count"]),
            3.0,
        )
        torch.testing.assert_close(importance.mean(), torch.ones(()))

    def test_actor_start_stratification_covers_preemptive_and_boundary_states(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=15,
            imagination_window_burn_in=0,
            max_imagination_starts=6,
        ))
        batch = pure_batch(t=10)
        batch["goal_position"][..., 0] = 100.0
        batch["human_mask"].zero_()
        valid = trainer._flat_valid_starts(batch)
        self.assertGreaterEqual(int(valid.numel()), 12)
        collision = batch["future_collision"].reshape(-1)
        ttc_valid = batch["time_to_collision_valid"].reshape(-1)
        ttc = batch["time_to_collision_s"].reshape(-1)
        actionable = valid[:3]
        imminent = valid[3:5]
        preemptive = valid[5:8]
        near_boundary = valid[8:11]
        for indices in (actionable, imminent, preemptive):
            collision[indices] = True
            ttc_valid[indices] = True
        ttc[actionable] = 1.2
        ttc[imminent] = 0.2
        ttc[preemptive] = 2.0
        flat_ego = batch["ego_state"].reshape(-1, 14)
        flat_ego[near_boundary, 0] = 19.0

        torch.manual_seed(67)
        selected, importance, metrics = trainer._sample_imagination_starts(
            batch)
        selected_set = set(selected.tolist())
        self.assertEqual(
            len(selected_set & set(actionable.tolist())), 2)
        self.assertEqual(
            len(selected_set & set(imminent.tolist())), 1)
        self.assertEqual(
            len(selected_set & set(preemptive.tolist())), 1)
        self.assertEqual(
            len(selected_set & set(near_boundary.tolist())), 1)
        self.assertEqual(
            float(metrics[
                "actor_imagination_preemptive_population_count"]),
            3.0,
        )
        self.assertEqual(
            float(metrics[
                "actor_imagination_near_boundary_population_count"]),
            3.0,
        )
        torch.testing.assert_close(importance.mean(), torch.ones(()))

        statistic = torch.full(
            (batch["is_last"].shape[0] * batch["is_last"].shape[1],),
            11.0,
        )
        statistic[actionable] = 1.0
        statistic[imminent] = 3.0
        statistic[preemptive] = 5.0
        statistic[near_boundary] = 7.0
        expected = statistic.index_select(0, valid).mean()
        estimated = (
            statistic.index_select(0, selected) * importance
        ).mean()
        torch.testing.assert_close(estimated, expected)

    def test_actor_start_stratification_covers_compound_task_states_unbiased(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=15,
            imagination_window_burn_in=0,
            max_imagination_starts=5,
        ))
        batch = pure_batch(t=10)
        batch["future_collision"].zero_()
        batch["time_to_collision_valid"].zero_()
        batch["human_mask"].zero_()
        batch["human_root"].zero_()
        batch["goal_position"][..., :3] = torch.tensor((100.0, 0.0, 1.0))
        valid = trainer._flat_valid_starts(batch)
        self.assertGreaterEqual(int(valid.numel()), 13)
        near_goal_ymax = valid[:3]
        side_front_human = valid[3:6]
        near_goal = valid[6:9]
        near_boundary = valid[9:12]
        flat_ego = batch["ego_state"].reshape(-1, 14)
        flat_goal = batch["goal_position"].reshape(-1, 3)
        flat_human_root = batch["human_root"].reshape(
            -1, batch["human_root"].shape[-2],
            batch["human_root"].shape[-1])
        flat_human_mask = batch["human_mask"].reshape(
            -1, batch["human_mask"].shape[-1])

        # Exercise the recoverable H15 band outside the generic 1.5 m
        # boundary zone. Two metres of route cross-track error makes these
        # genuinely misaligned near-goal states rather than ordinary goal
        # approaches; later rows remain ordinary replay instead of receiving
        # a second compound-state quota.
        flat_ego[near_goal_ymax, 0] = 2.0
        flat_ego[near_goal_ymax, 1] = torch.tensor((16.0, 17.0, 18.0))
        flat_goal[near_goal_ymax] = torch.tensor((0.0, 18.0, 1.0))

        flat_ego[side_front_human, 0] = 17.0
        flat_ego[side_front_human, 1] = 0.0
        flat_human_mask[side_front_human, 0] = True
        flat_human_root[side_front_human, 0, :3] = torch.tensor(
            (2.0, 0.0, 0.0))

        flat_ego[near_goal, 0] = 0.0
        flat_ego[near_goal, 1] = 0.0
        flat_goal[near_goal] = torch.tensor((2.0, 0.0, 1.0))
        flat_ego[near_boundary, 0] = -19.0
        flat_ego[near_boundary, 1] = 0.0

        strata = trainer._imagination_start_strata(batch, valid)
        expected_parts = (
            near_goal_ymax, side_front_human, near_goal, near_boundary)
        for stratum, expected in zip(strata[3:7], expected_parts, strict=True):
            self.assertEqual(set(stratum.tolist()), set(expected.tolist()))
        flattened = torch.cat(strata)
        self.assertEqual(int(flattened.numel()), int(valid.numel()))
        self.assertEqual(len(set(flattened.tolist())), int(valid.numel()))

        torch.manual_seed(71)
        selected, importance, metrics = trainer._sample_imagination_starts(
            batch)
        selected_set = set(selected.tolist())
        for expected in expected_parts:
            self.assertEqual(len(selected_set & set(expected.tolist())), 1)
        self.assertEqual(
            float(metrics[
                "actor_imagination_near_goal_ymax_misaligned_"
                "population_count"]),
            3.0,
        )
        self.assertEqual(
            float(metrics[
                "actor_imagination_side_boundary_front_human_"
                "population_count"]),
            3.0,
        )
        torch.testing.assert_close(importance.mean(), torch.ones(()))

        statistic = torch.full(
            (batch["is_last"].shape[0] * batch["is_last"].shape[1],),
            11.0,
        )
        for value, indices in enumerate(expected_parts, start=1):
            statistic[indices] = float(value)
        expected = statistic.index_select(0, valid).mean()
        estimated = (
            statistic.index_select(0, selected) * importance
        ).mean()
        torch.testing.assert_close(estimated, expected)

    def test_actor_start_sampling_contract_migrates_v50_checkpoint_state(self):
        trainer = PureDreamerTrainer(pure_model(), PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        state = trainer.state_dict()
        predecessor = dict(state)
        predecessor["imagination_start_sampling_contract"] = next(iter(
            trainer.LEGACY_IMAGINATION_START_SAMPLING_CONTRACTS))
        trainer.load_state_dict(predecessor)
        state.pop("imagination_start_sampling_contract")
        trainer.load_state_dict(state)
        incompatible = dict(state)
        incompatible["imagination_start_sampling_contract"] = "unknown"
        with self.assertRaisesRegex(
            RuntimeError, "imagination-start sampling mismatch",
        ):
            trainer.load_state_dict(incompatible)

    def test_actor_start_sampling_keeps_all_valid_rows_below_budget(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=15,
            imagination_window_burn_in=0,
            max_imagination_starts=64,
        ))
        batch = pure_batch(t=4)
        expected = trainer._flat_valid_starts(batch)
        selected, importance, _ = trainer._sample_imagination_starts(batch)
        torch.testing.assert_close(selected, expected)
        torch.testing.assert_close(importance, torch.ones_like(importance))
        self.assertGreaterEqual(
            trainer.actor_human_action_response_time_s,
            trainer.config.imagination_dt_s,
        )
        self.assertLessEqual(
            trainer.actor_actionable_response_time_s,
            trainer.actor_human_action_response_time_s,
        )
        self.assertGreaterEqual(
            trainer.actor_human_action_horizon_response_fraction,
            trainer.ACTOR_HORIZONTAL_STEP_RESPONSE_TARGET,
        )

    def test_human_prediction_bias_growth_must_be_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "bias/scale"):
            PureDreamerConfig(
                human_collision_prediction_bias_growth_m_per_step=-0.001)

    def test_counterfactual_forward_probes_are_ordered_positive_speeds(self):
        with self.assertRaisesRegex(ValueError, "0 < mild < fast"):
            PureDreamerConfig(
                counterfactual_mild_forward_policy=-0.20)
        with self.assertRaisesRegex(ValueError, "0 < mild < fast"):
            PureDreamerConfig(
                counterfactual_mild_forward_policy=0.70,
                counterfactual_fast_forward_policy=0.65,
            )

    def test_explicit_human_geometry_is_owned_by_world_update(self):
        model = pure_model()
        batch = pure_batch()
        batch["human_observation_quality"] = torch.zeros(
            *batch["human_root"].shape[:-1], 7)
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        # The empirical prior deliberately zeroes the final hazard weights;
        # the first update must therefore move that output layer before
        # gradients can reach earlier geometry MLP layers on later updates.
        parameter = model.transition_event.per_human_network[-1].weight
        before = parameter.detach().clone()
        metrics = trainer(batch)
        self.assertGreater(metrics["world/transition_human_geometry"], 0.0)
        self.assertFalse(torch.equal(parameter, before))
        self.assertIn("imag/human_birth_probability", metrics)
        self.assertGreaterEqual(metrics["imag/human_birth_probability"], 0.0)
        self.assertLessEqual(metrics["imag/human_birth_probability"], 1.0)

    def test_multistep_ego_and_privileged_clearance_losses_are_selected(self):
        torch.manual_seed(53)
        model = pure_model(overshoot=(1,))
        model.human_only_overshooting = True
        batch = pure_batch(t=16)
        batch["sequence_valid"] = torch.ones(2, 16, 1, dtype=torch.bool)
        batch["privileged_geometry_available"] = torch.ones(
            2, 16, 1, dtype=torch.bool)
        relative = torch.zeros(2, 16, 3, 10, 3)
        relative[..., 0] = 0.55
        batch["priv_collision_joints_episode"] = (
            relative + batch["ego_state"][..., None, None, :3])
        batch["priv_collision_joint_valid"] = torch.ones(
            2, 16, 3, 10, dtype=torch.bool)
        batch["future_collision"][:, 1] = True
        batch["time_to_collision_valid"][:, 1] = True
        batch["time_to_collision_s"][:, 1] = 0.1
        batch["priv_collision_human_id"][:, 2, 0] = 0
        batch["termination_code"][:, 2, 0] = 2
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))

        loss, metrics, _, _ = trainer._world_loss(batch)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("world/ego_overshoot_1", metrics)
        self.assertIn("world/human_joint_clearance_1", metrics)
        self.assertIn("world/human_cv_regret_1", metrics)
        self.assertIn("world/human_joint_collision_tail_1", metrics)
        self.assertIn("world_rollout/h1/ego_position_fde_m", metrics)
        self.assertIn("world_rollout/h1/joint_clearance_false_safe_m", metrics)
        self.assertIn("world_rollout/h1/human_cv_regret_m2", metrics)
        self.assertIn("world_rollout/h1/human_gt_cv_regret_m2", metrics)
        self.assertIn("world_rollout/h1/collision_tail_loss_m2", metrics)
        self.assertIn(
            "counterfactual_safety/human_episode_exogeneity", metrics)
        self.assertIn(
            "counterfactual_safety/"
            "human_joint_episode_exogeneity_mean_deviation_m",
            metrics,
        )
        self.assertGreater(
            float(metrics["world_rollout/h1/joint_clearance_valid_count"]),
            0.0,
        )
        self.assertEqual(
            float(metrics[
                "world_rollout/h1/collision_start_selected_fraction"]),
            1.0,
        )

    def test_actor_action_reaches_future_human_geometry_through_ego(self):
        """Regression for the removed next-Ego detach in imagination."""
        torch.manual_seed(59)
        model = pure_model(action=4)
        batch = pure_batch(action=4)
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2,
            max_imagination_starts=4,
            actor_entropy=0.0,
        ))
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
        initial, values = trainer._imagination_inputs(
            states, auxiliary, batch)

        # Remove the RSSM action path for this focused test.  The remaining
        # route is Actor action -> analytic Ego dynamics -> relative Human
        # prediction; it would be exactly zero with the historical detach.
        original_img_step = model.rssm.img_step

        def action_independent_img_step(state, action, human_mask, *args, **kwargs):
            return original_img_step(
                state, torch.zeros_like(action), human_mask, *args, **kwargs)

        model.rssm.img_step = action_independent_img_step
        try:
            imagined = trainer._imagine(initial, values)
            future_root = imagined["human_root"][:, 1, ..., :3].sum()
            gradients = torch.autograd.grad(
                future_root,
                tuple(model.actor.parameters()),
                allow_unused=True,
            )
        finally:
            model.rssm.img_step = original_img_step
        gradient_norm = sum(
            float(gradient.abs().sum())
            for gradient in gradients if gradient is not None)
        self.assertGreater(gradient_norm, 0.0)

    def test_pure_imagination_uses_analytic_ego_as_task_state(self):
        torch.manual_seed(61)
        model = pure_model(action=4)
        batch = pure_batch(action=4)
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=3,
            max_imagination_starts=4,
            authoritative_analytic_ego=True,
        ))
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)

        torch.testing.assert_close(
            imagined["ego_state"], imagined["analytic_ego_state"])
        self.assertEqual(float(imagined["authoritative_analytic_ego"]), 1.0)
        # The learned residual remains alive as a diagnostic/training target;
        # it is merely prevented from defining progress and safety geometry.
        self.assertGreater(
            float(torch.linalg.vector_norm(
                imagined["learned_ego_state"][..., :3]
                - imagined["ego_state"][..., :3], dim=-1).amax()),
            0.0,
        )

    def test_training_and_online_initial_joint_clearance_semantics_match(self):
        torch.manual_seed(63)
        model = pure_model(action=4)
        batch = pure_batch(action=4)
        # Root centre is far away, but an articulated joint protrudes into the
        # safety tube. The old trainer fallback used root distance and missed it.
        batch["skeleton"][..., :3] = 3.0
        # COCO12 index 5 is the deployable right-hand sphere (8 cm radius).
        batch["skeleton"][..., 5, :3] = torch.tensor((0.40, 0.0, 0.0))
        batch["joint_mask"].fill_(True)
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=32))
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)
        torch.testing.assert_close(
            values["human_joint_clearance"],
            torch.full_like(values["human_joint_clearance"], 0.15),
            atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(
            imagined["human_joint_clearance"][:, 0],
            values["human_joint_clearance"])

    def test_actor_lambda_return_uses_same_closed_event_continuation(self):
        model = pure_model()
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        batch = pure_batch()
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)
            _, _, _, _, metrics = trainer._imagined_objective(imagined)
        torch.testing.assert_close(
            metrics["continuation"], metrics["event_closed_continuation"])

    def test_counterfactual_truth_ignores_unmatched_closer_person(self):
        model = pure_model()
        trainer = object.__new__(PureDreamerTrainer)
        trainer.model = model
        batch = {
            "ego_state": torch.zeros(1, 4, 14),
            "human_gt_id": torch.tensor([[[10, -1], [10, -1],
                                           [10, -1], [10, -1]]]),
            "human_gt_match_valid": torch.tensor([[[True, False],
                                                     [True, False],
                                                     [True, False],
                                                     [True, False]]]),
            "human_mask": torch.tensor([[[True, False], [True, False],
                                           [True, False], [True, False]]]),
            "joint_mask": torch.ones(
                1, 4, 2, 12, dtype=torch.bool),
            "priv_human_id": torch.tensor([[[10, 99, -1], [99, 10, -1],
                                              [10, 99, -1], [99, 10, -1]]]),
            "priv_human_mask": torch.tensor([[[True, True, False]] * 4]),
            "priv_collision_joints_episode": torch.zeros(1, 4, 3, 10, 3),
            "priv_collision_joint_valid": torch.ones(
                1, 4, 3, 10, dtype=torch.bool),
            "counterfactual_collision_surface_radii_m": torch.full(
                (1, 4, 10), 0.25),
        }
        # Person 99 is almost touching, while source-matched person 10 remains
        # at 2 m and changes privileged slot ordering on every row.
        for time in range(4):
            ids = batch["priv_human_id"][0, time]
            for slot in range(2):
                x = 2.0 if int(ids[slot]) == 10 else 0.10
                batch["priv_collision_joints_episode"][0, time, slot, :, 0] = x
            batch["priv_collision_joint_valid"][0, time, 2] = False
        minimum, valid, profile = trainer._matched_gt_future_clearance(
            batch, source_count=2, horizon=2)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(
            minimum, torch.full_like(minimum, 1.75))
        torch.testing.assert_close(
            profile, torch.full_like(profile, 1.75))

    def test_counterfactual_truth_ignores_source_unavailable_spheres(self):
        model = pure_model()
        trainer = object.__new__(PureDreamerTrainer)
        trainer.model = model
        batch = {
            "ego_state": torch.zeros(1, 4, 14),
            "human_gt_id": torch.tensor([[[10], [10], [10], [10]]]),
            "human_gt_match_valid": torch.ones(
                1, 4, 1, dtype=torch.bool),
            "human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "joint_mask": torch.zeros(1, 4, 1, 12, dtype=torch.bool),
            "priv_human_id": torch.tensor([[[10], [10], [10], [10]]]),
            "priv_human_mask": torch.ones(1, 4, 1, dtype=torch.bool),
            "priv_collision_joints_episode": torch.full(
                (1, 4, 1, 10, 3), 0.0),
            "priv_collision_joint_valid": torch.ones(
                1, 4, 1, 10, dtype=torch.bool),
            "counterfactual_collision_surface_radii_m": torch.full(
                (1, 4, 10), 0.25),
        }
        # Only COCO right hand (deployable sphere index 1) existed in the source
        # observation.  All other simulator spheres are much closer but cannot
        # be activated or predicted by the deployed rollout.
        batch["joint_mask"][..., 5] = True
        batch["priv_collision_joints_episode"][..., 0] = 0.10
        batch["priv_collision_joints_episode"][..., 1, 0] = 2.0
        minimum, valid, profile = trainer._matched_gt_future_clearance(
            batch, source_count=2, horizon=2)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(
            minimum, torch.full_like(minimum, 1.75))
        torch.testing.assert_close(
            profile, torch.full_like(profile, 1.75))

    def test_value_bootstrap_reads_explicit_decision_feature(self):
        torch.manual_seed(67)
        model = pure_model(action=4)
        batch = pure_batch(action=4)
        trainer = PureDreamerTrainer(model, PureDreamerConfig(
            imagination_horizon=2, max_imagination_starts=4))
        with torch.no_grad():
            states, auxiliary, _ = model.posterior(batch)
            initial, values = trainer._imagination_inputs(
                states, auxiliary, batch)
            imagined = trainer._imagine(initial, values)
        self.assertFalse(torch.equal(
            imagined["decision_feat"], imagined["joint_feat"]))

        captured_value = []
        captured_slow = []
        value_hook = model.value.register_forward_pre_hook(
            lambda _module, args: captured_value.append(args[0].detach().clone()))
        slow_hook = model.slow_value.register_forward_pre_hook(
            lambda _module, args: captured_slow.append(args[0].detach().clone()))
        try:
            trainer._imagined_objective(imagined)
        finally:
            value_hook.remove()
            slow_hook.remove()
        self.assertEqual(len(captured_value), 1)
        self.assertEqual(len(captured_slow), 1)
        torch.testing.assert_close(captured_value[0], imagined["decision_feat"])
        torch.testing.assert_close(captured_slow[0], imagined["decision_feat"])


if __name__ == "__main__":
    unittest.main()
