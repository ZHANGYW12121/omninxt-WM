from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factorized_dreamer import FactorizedDreamer
from factorized_rssm import FactorizedRSSM
from factorized_trainer import FactorizedTrainStep, resolve_amp_dtype
from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads
from modules.action_risk_head import ActionRiskHead
from modules.action_adapter import HorizontalActionAdapter
from modules.factorized_actor_v62 import (
    FactorizedActorV62,
    FactorizedActorV62Config,
)
from modules.latent_policy_attention import LatentPolicyAttentionConfig
from modules.task_memory import TASK_MEMORY_SCALE
from modules.transition_event_head import TransitionEventHead


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


def make_model(embed=16, action=3, joint=32, overshoot=()):
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
        NormalHead(joint), ContinueHead(joint),
        ActionRiskHead(joint, action, hidden_dim=32), predictions, kl_free=0.0,
        overshoot_horizons=tuple(overshoot),
        overshoot_starts_per_sequence=2,
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
        "observed_human_clearance_m": torch.ones(b, t),
        "observed_human_away_xy": torch.randn(b, t, 2),
        "action": torch.randn(b, t, action), "is_first": torch.zeros(b, t, dtype=torch.bool),
        "reward": torch.randn(b, t, 1), "is_terminal": torch.zeros(b, t, 1),
        "is_last": torch.zeros(b, t, 1), "truncated_people": torch.zeros(b, t),
        "action_valid": torch.ones(b, t, 1, dtype=torch.bool),
        "episode_success": torch.ones(b, t, 1, dtype=torch.bool),
        "future_collision": torch.zeros(b, t, 1, dtype=torch.bool),
        "future_min_human_clearance_m": torch.ones(b, t, 1),
        "future_min_human_clearance_valid": torch.ones(
            b, t, 1, dtype=torch.bool),
        "time_to_collision_s": torch.ones(b, t, 1),
        "time_to_collision_valid": torch.zeros(b, t, 1, dtype=torch.bool),
        "priv_min_human_clearance_m": torch.ones(b, t),
        "priv_min_human_clearance_valid": torch.ones(
            b, t, dtype=torch.bool),
        "priv_min_human_ttc_s": torch.full((b, t), 10.0),
        "priv_min_human_ttc_valid": torch.zeros(
            b, t, dtype=torch.bool),
        "human_root": torch.randn(b, t, n, 3),
        "human_ids": torch.arange(n).reshape(1, 1, n).expand(
            b, t, n).clone(),
    }
    batch["is_first"][:, 0] = True
    batch["human_is_first"][:, 0] = True
    return batch


class FactorizedDreamerTest(unittest.TestCase):
    LOSS_SCALES = {
        "dyn_ego": 1, "rep_ego": .1, "dyn_human": 1, "rep_human": .1,
        "ego_recon": 1, "ego_pred": 1, "yaw_unit": .1,
        "human_root": 1, "human_mpjpe": 1, "human_presence": .2,
        "rew": 1, "con": 1, "policy": 1, "behavior_clone": 1,
        "safety_policy": 1,
        "risk_collision": 1, "risk_clearance": 1,
        "risk_false_safe": 1,
        "risk_false_danger": .5,
        "collision_risk": 1, "clearance_barrier": 1,
        "unsafe_speed": 1, "replay_unsafe_speed": 1,
        "vertical_action": 1,
        "lateral_candidate": 1,
        "goal_directed": 1,
        "action_smoothness": 1,
        "value": 1, "repval": .3,
        "human_overshoot_5": .5,
        "human_overshoot_10": .75,
        "human_joint_overshoot_5": .5,
        "human_joint_overshoot_10": .75,
        "human_joint_clearance_5": .5,
        "human_joint_clearance_10": .75,
        "risk_collision_overshoot_5": .5,
        "risk_collision_overshoot_10": .75,
        "risk_clearance_overshoot_5": .5,
        "risk_clearance_overshoot_10": .75,
        "risk_false_safe_overshoot_5": .5,
        "risk_false_safe_overshoot_10": .75,
        "risk_false_danger_overshoot_5": .25,
        "risk_false_danger_overshoot_10": .375,
    }

    def test_direct_actor_ego_task_prefix_keeps_independent_exact_blocks(self):
        model = make_model(joint=32)
        metric_encoder = nn.Module()
        metric_encoder.register_buffer("metric_scale", torch.tensor((
            32.0, 12.0, 1.0, 3.0, 3.0, 2.0, 8.0, 8.0, 8.0,
            2.0, 1.0, 1.0, 1.0, 1.0,
        )))
        model.encoder.ego_encoder = metric_encoder
        model.actor_direct_ego_task_state_enabled = True
        ego = torch.arange(1.0, 15.0).reshape(1, 14)
        geometry = torch.linspace(0.0, 1.0, 8).reshape(1, 8)
        memory_scale = torch.tensor(TASK_MEMORY_SCALE).reshape(1, 9)
        memory = 0.25 * memory_scale
        latent = torch.randn(1, 32)

        token = model.actor_ego_task_token(
            ego, latent, geometry, memory)

        expected = torch.cat((
            (ego / metric_encoder.metric_scale).clamp(-5.0, 5.0),
            geometry,
            torch.full((1, 9), 0.25),
        ), -1)
        torch.testing.assert_close(token[..., :31], expected)
        torch.testing.assert_close(token[..., 31:], latent[..., 31:])

    def test_r2_candidate_support_separates_event_and_action_validity(self):
        descriptor = torch.tensor([[2.0, 0.0]])
        candidates = torch.tensor([[[0.0, 0.0, 0.0],
                                    [0.5, 0.0, 0.0]]])
        support = {
            "descriptors": torch.zeros(2, 2),
            "actions": torch.zeros(2, 3),
            "mean": torch.zeros(2),
            "std": torch.ones(2),
            "state_radius": 0.5,
            "action_radius": 0.1,
            "neighbors": 1,
        }

        event, action, action_distance, state_distance = (
            FactorizedDreamer._r2_planner_candidate_support(
                descriptor, candidates, support))

        self.assertEqual(event.tolist(), [[False, False]])
        self.assertEqual(action.tolist(), [[True, False]])
        torch.testing.assert_close(
            action_distance, torch.tensor([[0.0, 0.5]]))
        torch.testing.assert_close(state_distance, torch.tensor([2.0]))

    def test_r2_goal_aligned_candidate_preserves_horizontal_speed(self):
        policy = torch.tensor([[0.3, -0.4, 0.2]])
        ego = torch.zeros(1, 14)
        ego[:, 13] = 1.0
        goal = torch.tensor([[0.0, 5.0, 0.0]])

        aligned = FactorizedDreamer._r2_planner_goal_aligned_action(
            policy, ego, goal)

        torch.testing.assert_close(
            aligned[:, :2].norm(dim=-1),
            policy[:, :2].norm(dim=-1),
        )
        self.assertGreater(float(aligned[0, 1]), 0.0)
        torch.testing.assert_close(aligned[:, 2], policy[:, 2])

    def test_gt_displacement_supervision_cancels_absolute_pelvis_offset(self):
        predicted_source = torch.tensor([[[1.0, 2.0, 0.0]]])
        predicted_target = torch.tensor([[[1.6, 1.8, 0.1]]], requires_grad=True)
        detector_to_gt_offset = torch.tensor([[[4.0, -3.0, 0.5]]])
        target_source = predicted_source + detector_to_gt_offset
        target_target = predicted_target.detach() + detector_to_gt_offset
        valid = torch.ones(1, 1, dtype=torch.bool)

        loss, error = FactorizedDreamer._gt_displacement_supervision(
            predicted_target, predicted_source,
            target_target, target_source, valid)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-6, rtol=0)

        wrong_target = target_target + torch.tensor([[[0.4, 0.0, 0.0]]])
        wrong_loss, _ = FactorizedDreamer._gt_displacement_supervision(
            predicted_target, predicted_source,
            wrong_target, target_source, valid)
        self.assertGreater(float(wrong_loss.detach()), 0.0)
        wrong_loss.backward()
        self.assertGreater(float(predicted_target.grad.abs().sum()), 0.0)

    def test_gt_cv_regret_is_offset_invariant_and_one_sided(self):
        predicted_source = torch.tensor([[[1.0, 2.0, 0.0]]])
        simulator_offset = torch.tensor([[[4.0, -3.0, 0.5]]])
        target_source = predicted_source + simulator_offset
        target_target = target_source + torch.tensor([[[1.0, 0.0, 0.0]]])
        constant_velocity = predicted_source + torch.tensor(
            [[[1.2, 0.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)

        better_prediction = (
            predicted_source + torch.tensor([[[1.1, 0.0, 0.0]]]))
        better_loss, better_error, baseline_error = (
            FactorizedDreamer._gt_cv_regret_supervision(
                better_prediction,
                predicted_source,
                constant_velocity,
                target_target,
                target_source,
                valid,
            ))
        torch.testing.assert_close(
            better_loss, torch.zeros_like(better_loss), atol=1e-6, rtol=0)
        self.assertLess(float(better_error), float(baseline_error))

        worse_prediction = (
            predicted_source + torch.tensor([[[1.7, 0.0, 0.0]]])
        ).detach().requires_grad_(True)
        shifted_offset = torch.tensor([[[8.0, 6.0, -2.0]]])
        worse_loss, worse_error, shifted_baseline_error = (
            FactorizedDreamer._gt_cv_regret_supervision(
                worse_prediction,
                predicted_source,
                constant_velocity,
                target_target + shifted_offset,
                target_source + shifted_offset,
                valid,
            ))
        self.assertGreater(
            float(worse_error.detach()),
            float(shifted_baseline_error.detach()),
        )
        self.assertGreater(float(worse_loss.detach()), 0.0)
        worse_loss.backward()
        self.assertTrue(bool(torch.isfinite(worse_prediction.grad).all()))
        self.assertLessEqual(float(worse_prediction.grad.norm()), 0.1 + 1e-6)

        # Below beta=0.05, 2*beta*SmoothL1 is exactly squared regret.
        locally_worse = (
            predicted_source + torch.tensor([[[1.23, 0.0, 0.0]]])
        ).detach().requires_grad_(True)
        local_loss, _, _ = FactorizedDreamer._gt_cv_regret_supervision(
            locally_worse,
            predicted_source,
            constant_velocity,
            target_target,
            target_source,
            valid,
        )
        torch.testing.assert_close(
            local_loss, torch.tensor(0.03 ** 2), atol=1e-7, rtol=1e-6)
        local_loss.backward()
        torch.testing.assert_close(
            locally_worse.grad.norm(), torch.tensor(2.0 * 0.03),
            atol=1e-6, rtol=1e-6)

    def test_explicit_human_geometry_online_action_uses_nested_tokens(self):
        """Exercise the deployed Actor feature path, not only imagination.

        ``FactorizedRSSM.obs_step`` exposes goal/ego tokens under
        ``aux['latent_attention']``.  The explicit-geometry online branch used
        to look for those keys at the top level, so training and imagination
        passed while every real policy request failed with ``goal_token``.
        """
        torch.manual_seed(23)
        model = make_model(action=4)
        token_dim = model.rssm.joint_feat_size
        model.actor = FactorizedActorV62(FactorizedActorV62Config(
            token_dim=token_dim,
            hidden_dim=32,
            policy_action_dim=3,
            min_std=(0.01, 0.01, 0.01),
            max_std=(0.10, 0.10, 0.10),
            initial_std=(0.02, 0.02, 0.02),
        ))
        model.action_adapter = HorizontalActionAdapter()
        model.uses_internal_action_adapter = True
        model.transition_event = TransitionEventHead(
            token_dim,
            4,
            hidden_dim=32,
            ego_feat_dim=model.rssm.ego_rssm.feat_size,
            human_feat_dim=model.rssm.human_rssm.feat_size,
            human_root_dim=3,
            human_quality_dim=7,
            explicit_human_geometry=True,
        )
        batch = make_batch(action=4)
        batch["human_observation_quality"] = torch.zeros(
            *batch["human_root"].shape[:-1], 7)
        online = {key: value[:, :1] for key, value in batch.items()}

        action, state = model.act_step(
            online,
            model.initial_agent_state(2, 3, 4),
            evaluation=True,
        )

        self.assertEqual(action.shape, (2, 4))
        self.assertEqual(state["policy_action"].shape, (2, 3))
        self.assertTrue(torch.isfinite(action).all())

        def reject_legacy_shadow_risk(*_args, **_kwargs):
            raise AssertionError(
                "explicit-geometry shadow must not call legacy action_risk")

        model.action_risk.forward = reject_legacy_shadow_risk
        shadow = model.shadow_imagination(
            state["latent"],
            online["human_mask"][:, -1],
            online["goal_position"][:, -1],
            online["ego_state"][:, -1],
            action,
            horizon_steps=3,
            human_root=online["human_root"][:, -1],
            human_quality=online[
                "human_observation_quality"][:, -1],
            human_joint_mask=online["joint_mask"][:, -1],
            human_joints_body=online["skeleton"][:, -1, ..., :3],
        )
        self.assertEqual(shadow["critic_value"].shape, (2, 3))
        self.assertEqual(
            shadow["geometry_collision_probability"].shape, (2, 3))
        self.assertEqual(
            shadow["learned_collision_probability"].shape, (2, 3))
        self.assertTrue(torch.isfinite(shadow["critic_value"]).all())
        self.assertTrue(torch.isfinite(
            shadow["predicted_min_human_clearance_m"]).all())

    def test_world_model_goal_imagination_and_online_action(self):
        torch.manual_seed(5)
        model = make_model()
        batch = make_batch()
        losses, states, aux = model.world_model_loss(batch)
        self.assertEqual(set(losses), {
            "dyn_ego", "rep_ego", "dyn_human", "rep_human", "rew", "con",
            "risk_collision", "risk_clearance", "risk_false_safe",
            "risk_false_danger",
        })
        self.assertNotIn("env", states)
        ac_losses, _ = model.actor_critic_loss(states, aux, batch, imag_horizon=3)
        self.assertEqual(
            set(ac_losses), {
                "policy", "safety_policy", "behavior_clone", "collision_risk",
                "clearance_barrier", "unsafe_speed", "action_smoothness",
                "replay_unsafe_speed", "vertical_action",
                "lateral_candidate", "goal_directed",
                "value", "repval",
            })
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
        # Online collection samples from the already-trained Actor but keeps
        # every dimension inside an explicit delta tube around its mode.
        limits = torch.tensor([0.05, 0.12, 0.015])
        torch.manual_seed(19)
        mode_action, _ = model.act_step(
            online, model.initial_agent_state(2, 3, 3), evaluation=True)
        torch.manual_seed(19)
        explored_action, explored_state = model.act_step(
            online,
            model.initial_agent_state(2, 3, 3),
            evaluation=False,
            exploration_scale=1.0,
            exploration_delta_limits=limits,
        )
        self.assertTrue(torch.all(
            (explored_action - mode_action).abs() <= limits + 5.0e-6))
        self.assertEqual(explored_state["exploration_scale"], 1.0)
        restored = model.state_from_replay_fields(model.replay_state_fields(agent_state["latent"]))
        self.assertEqual(set(restored), {"ego", "human"})
        with self.assertRaisesRegex(
            RuntimeError, "legacy action_risk fallback is disabled",
        ):
            model.shadow_imagination(
                agent_state["latent"],
                online["human_mask"][:, -1],
                online["goal_position"][:, -1],
                online["ego_state"][:, -1],
                action,
                horizon_steps=4,
            )

    def test_evaluation_action_and_posterior_are_seed_invariant(self):
        model = make_model()
        model.deterministic_evaluation_state_enabled = True
        online = {
            key: value[:, :1] for key, value in make_batch().items()
        }

        torch.manual_seed(101)
        action_a, state_a = model.act_step(
            online, model.initial_agent_state(2, 3, 3), evaluation=True)
        torch.manual_seed(202)
        action_b, state_b = model.act_step(
            online, model.initial_agent_state(2, 3, 3), evaluation=True)

        torch.testing.assert_close(action_a, action_b, rtol=0, atol=0)
        for branch in ("ego", "human"):
            for key in ("stoch", "deter"):
                torch.testing.assert_close(
                    state_a["latent"][branch][key],
                    state_b["latent"][branch][key], rtol=0, atol=0)
        planned_action, planned_state = model.act_step(
            online,
            model.initial_agent_state(2, 3, 3),
            evaluation=True,
            use_planner=True,
        )
        self.assertEqual(planned_action.shape, (2, 3))
        planner = planned_state["planner"]
        self.assertEqual(planner["scores"].shape, (2, 4))
        self.assertEqual(planner["triggered"].shape, (2,))
        self.assertEqual(planner["emergency"].shape, (2,))
        self.assertEqual(planner["intervened"].shape, (2,))
        self.assertEqual(planner["imagined_best_index"].shape, (2,))
        self.assertTrue(torch.isfinite(planner["scores"]).all())
        self.assertTrue(torch.isfinite(planned_action).all())

        no_human = {key: value.clone() for key, value in online.items()}
        no_human["human_mask"].zero_()
        no_human["joint_mask"].zero_()
        no_human["human_is_first"].zero_()
        no_human["skeleton_stub"].zero_()
        no_human["human_root"].zero_()
        no_human_action, no_human_state = model.act_step(
            no_human,
            model.initial_agent_state(2, 3, 3),
            evaluation=True,
            use_planner=True,
        )
        self.assertFalse(no_human_state["planner"]["triggered"].any())
        self.assertFalse(no_human_state["planner"]["intervened"].any())
        torch.testing.assert_close(
            no_human_action,
            no_human_state["planner"]["actor_action"],
        )

    def test_formal_posterior_and_prior_modes_do_not_consume_rng(self):
        model = make_model()
        batch = make_batch()

        torch.manual_seed(101)
        posterior_rng = torch.random.get_rng_state().clone()
        states_a, auxiliary_a, _ = model.posterior(
            batch, sample_state=False)
        torch.testing.assert_close(
            posterior_rng, torch.random.get_rng_state(), rtol=0, atol=0)
        torch.manual_seed(202)
        states_b, auxiliary_b, _ = model.posterior(
            batch, sample_state=False)
        for branch in ("ego", "human"):
            for key in ("stoch", "deter"):
                torch.testing.assert_close(
                    states_a[branch][key], states_b[branch][key],
                    rtol=0, atol=0)
        torch.testing.assert_close(
            auxiliary_a["actor_feat"], auxiliary_b["actor_feat"],
            rtol=0, atol=0)

        initial = model.last_state(states_a)
        torch.manual_seed(303)
        prior_rng = torch.random.get_rng_state().clone()
        _, imagined_a = model.imagine(
            initial,
            4,
            batch["human_mask"][:, -1],
            goal_position=batch["goal_position"][:, -1],
            ego_state=batch["ego_state"][:, -1],
            sample=False,
        )
        torch.testing.assert_close(
            prior_rng, torch.random.get_rng_state(), rtol=0, atol=0)
        torch.manual_seed(404)
        _, imagined_b = model.imagine(
            initial,
            4,
            batch["human_mask"][:, -1],
            goal_position=batch["goal_position"][:, -1],
            ego_state=batch["ego_state"][:, -1],
            sample=False,
        )
        for key in ("action", "ego_state", "joint_feat"):
            torch.testing.assert_close(
                imagined_a[key], imagined_b[key], rtol=0, atol=0)

    def test_emergency_escape_overrides_nonpositive_actor_forward(self):
        model = make_model()
        state = model.rssm.initial(1, 3)
        human_mask = torch.ones(1, 3, dtype=torch.bool)
        goal_feature = torch.zeros(1, 8)
        joint_feature, _ = model.rssm.get_joint_feat(
            state, human_mask, goal_feature)

        def always_risky(_joint, action, detach_parameters=False):
            del detach_parameters
            shape = action.shape[:-1] + (1,)
            return {
                "collision_probability": torch.ones(
                    shape, dtype=action.dtype, device=action.device),
                "min_human_clearance_m": torch.zeros(
                    shape, dtype=action.dtype, device=action.device),
            }

        model.action_risk.forward = always_risky
        actor_action = torch.tensor(((-0.20, 0.10, 0.0),))
        ego_state = torch.zeros(1, 14)
        ego_state[:, 13] = 1.0
        action, planner = model.plan_online_action(
            state,
            joint_feature,
            human_mask,
            actor_action,
            actor_action,
            ego_state,
            torch.tensor(((10.0, 0.0, 0.0),)),
            torch.tensor((0.5,)),
            torch.tensor(((1.0, 0.0),)),
        )
        self.assertTrue(planner["emergency"].item())
        self.assertTrue(planner["intervened"].item())
        torch.testing.assert_close(
            action[0, :2], torch.tensor((0.43333334, 0.0)))

    def test_caution_geometry_slows_without_unnecessary_escape(self):
        model = make_model()
        state = model.rssm.initial(1, 3)
        human_mask = torch.ones(1, 3, dtype=torch.bool)
        joint_feature, _ = model.rssm.get_joint_feat(
            state, human_mask, torch.zeros(1, 8))

        def always_safe(_joint, action, detach_parameters=False):
            del detach_parameters
            shape = action.shape[:-1] + (1,)
            return {
                "collision_probability": torch.zeros(shape),
                "min_human_clearance_m": torch.full(shape, 10.0),
            }

        model.action_risk.forward = always_safe
        ego_state = torch.zeros(1, 14)
        ego_state[:, 13] = 1.0
        action, planner = model.plan_online_action(
            state,
            joint_feature,
            human_mask,
            torch.tensor(((0.40, 0.0, 0.0),)),
            torch.zeros(1, 3),
            ego_state,
            torch.tensor(((10.0, 0.0, 0.0),)),
            torch.tensor((1.5,)),
            torch.tensor(((0.0, 1.0),)),
        )
        self.assertFalse(planner["emergency"].item())
        self.assertTrue(planner["intervened"].item())
        torch.testing.assert_close(
            action[0, :2], torch.tensor((0.225, 0.0)))
        self.assertAlmostEqual(
            float(planner["forward_cap_action"]), 0.225, places=6)

    def test_close_observed_geometry_forces_escape_despite_low_risk(self):
        model = make_model()
        state = model.rssm.initial(1, 3)
        human_mask = torch.ones(1, 3, dtype=torch.bool)
        joint_feature, _ = model.rssm.get_joint_feat(
            state, human_mask, torch.zeros(1, 8))

        def always_safe(_joint, action, detach_parameters=False):
            del detach_parameters
            shape = action.shape[:-1] + (1,)
            return {
                "collision_probability": torch.zeros(shape),
                "min_human_clearance_m": torch.full(shape, 10.0),
            }

        model.action_risk.forward = always_safe
        ego_state = torch.zeros(1, 14)
        ego_state[:, 13] = 1.0
        action, planner = model.plan_online_action(
            state,
            joint_feature,
            human_mask,
            torch.tensor(((0.40, 0.0, 0.0),)),
            torch.zeros(1, 3),
            ego_state,
            torch.tensor(((10.0, 0.0, 0.0),)),
            torch.tensor((0.65,)),
            torch.tensor(((0.0, 1.0),)),
        )
        self.assertTrue(planner["emergency"].item())
        self.assertTrue(planner["intervened"].item())
        torch.testing.assert_close(
            action[0, :2], torch.tensor((0.0, 0.36333334)))

    def test_hard_stop_removes_motion_toward_nearest_human(self):
        model = make_model()
        state = model.rssm.initial(1, 3)
        human_mask = torch.ones(1, 3, dtype=torch.bool)
        joint_feature, _ = model.rssm.get_joint_feat(
            state, human_mask, torch.zeros(1, 8))

        def always_risky(_joint, action, detach_parameters=False):
            del detach_parameters
            shape = action.shape[:-1] + (1,)
            return {
                "collision_probability": torch.ones(shape),
                "min_human_clearance_m": torch.zeros(shape),
            }

        model.action_risk.forward = always_risky
        ego_state = torch.zeros(1, 14)
        ego_state[:, 13] = 1.0
        action, planner = model.plan_online_action(
            state,
            joint_feature,
            human_mask,
            torch.tensor(((0.40, 0.0, 0.0),)),
            torch.zeros(1, 3),
            ego_state,
            torch.tensor(((10.0, 0.0, 0.0),)),
            torch.tensor((0.2,)),
            torch.tensor(((0.8, 0.6),)),
            current_observed_clearance_m=torch.tensor((0.2,)),
            observed_nearest_xy=torch.tensor(((1.0, 0.0),)),
        )
        self.assertTrue(planner["hard_stop"].item())
        self.assertLessEqual(float(action[0, 0]), 1.0e-6)
        self.assertGreater(float(action[0, 1]), 0.5)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(action[0, :2])), 0.55, places=5)

    def test_behavior_clone_uses_next_row_action(self):
        torch.manual_seed(7)
        model = make_model(action=3)
        batch = make_batch(action=3)
        _, _, aux = model.world_model_loss(batch)
        original, _ = model.behavior_clone_objective(aux, batch)

        changed_previous = dict(batch)
        changed_previous["action"] = batch["action"].clone()
        changed_previous["action"][:, 0] = 100.0
        same, _ = model.behavior_clone_objective(aux, changed_previous)
        self.assertTrue(torch.allclose(original, same))

        changed_target = dict(batch)
        changed_target["action"] = batch["action"].clone()
        changed_target["action"][:, 1] = -1.0
        different, _ = model.behavior_clone_objective(aux, changed_target)
        self.assertFalse(torch.allclose(original, different))

        collision_batch = dict(changed_target)
        collision_batch["episode_success"] = torch.zeros_like(
            batch["episode_success"])
        excluded, metrics = model.behavior_clone_objective(aux, collision_batch)
        self.assertEqual(float(excluded.detach()), 0.0)
        self.assertEqual(float(metrics["bc_transition_ratio"]), 0.0)

        near_miss_batch = dict(batch)
        near_miss_batch["future_min_human_clearance_m"] = torch.full_like(
            batch["future_min_human_clearance_m"], 0.10)
        near_miss, metrics = model.behavior_clone_objective(
            aux, near_miss_batch)
        self.assertEqual(float(near_miss.detach()), 0.0)
        self.assertEqual(float(metrics["bc_transition_ratio"]), 0.0)
        self.assertEqual(float(metrics["near_miss_excluded_ratio"]), 1.0)

    def test_actor_safety_uses_frozen_risk_estimator(self):
        torch.manual_seed(8)
        model = make_model(action=3)
        feature = torch.randn(2, 5, model.rssm.joint_feat_size)
        losses, _ = model.actor_safety_objective(feature)
        sum(losses.values()).backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.actor.parameters()))
        self.assertTrue(all(
            parameter.grad is None for parameter in model.action_risk.parameters()))

    def test_safety_return_assigns_next_state_cost_to_current_action(self):
        cost = torch.tensor([[[10.0], [1.0], [2.0], [3.0]]])
        continuation = torch.ones_like(cost)
        result = FactorizedDreamer.discounted_safety_return(
            cost, continuation, discount=1.0, trace_lambda=1.0)
        torch.testing.assert_close(
            result, torch.tensor([[[6.0], [5.0], [3.0]]]))

    def test_replay_danger_speed_has_direct_actor_gradient(self):
        torch.manual_seed(10)
        model = make_model(action=3)
        batch = make_batch(action=3)
        batch["future_min_human_clearance_m"].fill_(0.2)
        with torch.no_grad():
            model.actor.linear.weight.zero_()
            model.actor.linear.bias.fill_(0.8)
        _, _, aux = model.world_model_loss(batch)
        loss, metrics = model.replay_unsafe_speed_objective(aux, batch)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(metrics["forward_excess_mean"].detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(model.actor.linear.bias.grad)
        self.assertGreater(float(model.actor.linear.bias.grad[0]), 0.0)

    def test_policy_reward_target_separates_human_safety_penalty(self):
        model = make_model(action=3)
        batch = make_batch(action=3)
        batch["reward_components"] = torch.zeros(
            *batch["reward"].shape[:-1], 6)
        # event, progress, human_clearance, smoothness, height, time
        batch["reward_components"][..., 1] = 2.0
        batch["reward_components"][..., 2] = -9.0
        batch["reward_components"][..., 5] = -0.1
        target = model.policy_reward_target(batch)
        torch.testing.assert_close(
            target, torch.full_like(target, 1.9))

    def test_lateral_candidates_cover_danger_and_form_distribution(self):
        torch.manual_seed(11)
        model = make_model(action=3)
        model.lateral_candidate_min_score_margin = 0.0
        model.lateral_candidate_min_hold_improvement = 0.0
        batch = make_batch(action=3)
        batch["future_collision"].fill_(True)
        batch["time_to_collision_valid"].fill_(True)
        batch["time_to_collision_s"].fill_(0.5)
        batch["priv_min_human_clearance_m"].fill_(0.4)
        losses, states, aux = model.world_model_loss(batch)
        self.assertTrue(torch.isfinite(sum(losses.values())))
        loss, metrics = model.lateral_candidate_objective(
            states, aux, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(metrics["active_transition_ratio"]), 0.0)
        selection_sum = sum(float(metrics[name]) for name in (
            "selected_hold_ratio", "selected_slow_ratio",
            "selected_left_ratio", "selected_right_ratio"))
        self.assertAlmostEqual(selection_sum, 1.0, places=5)
        # Hold is now deliberately excluded from pseudo-label supervision;
        # only candidates that improve on it update the Actor.
        self.assertGreater(float(metrics["supervised_danger_ratio"]), 0.0)
        self.assertLessEqual(float(metrics["supervised_danger_ratio"]), 1.0)
        loss.backward()
        actor_gradient = torch.stack([
            parameter.grad.detach().float().norm()
            for parameter in model.actor.parameters()
            if parameter.grad is not None
        ]).norm()
        self.assertGreater(float(actor_gradient), 0.0)

    def test_goal_directed_objective_removes_open_space_side_bias(self):
        torch.manual_seed(12)
        model = make_model(action=3)
        batch = make_batch(action=3)
        batch["priv_min_human_clearance_m"].fill_(3.0)
        batch["goal"].zero_()
        batch["goal"][..., 0] = 10.0
        batch["goal"][..., 3] = 10.0
        batch["goal"][..., 4] = 1.0
        with torch.no_grad():
            model.actor.linear.weight.zero_()
            model.actor.linear.bias[:] = torch.tensor((0.2, 0.5, 0.0))
        _, _, aux = model.world_model_loss(batch)
        loss, metrics = model.goal_directed_objective(aux, batch)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(metrics["active_transition_ratio"]), 0.0)
        self.assertGreater(
            float(metrics["aligned_unnecessary_lateral_ratio"]), 0.0)
        loss.backward()
        self.assertIsNotNone(model.actor.linear.bias.grad)
        self.assertGreater(float(model.actor.linear.bias.grad[1]), 0.0)

    def test_kinematic_counterfactual_clearance_distinguishes_sides(self):
        model = make_model(action=3)
        skeleton = torch.zeros(1, 1, 1, 7)
        skeleton[..., 0] = 1.0
        skeleton[..., 1] = 0.35  # pedestrian is on the left
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        candidates = torch.zeros(1, 2, 3)
        candidates[:, 0, :2] = torch.tensor((0.30, 0.35))
        candidates[:, 1, :2] = torch.tensor((0.30, -0.35))
        clearance, valid = model._kinematic_candidate_clearance(
            skeleton, mask, candidates)
        self.assertTrue(bool(valid.item()))
        # Moving right increases distance from a pedestrian on the left.
        self.assertGreater(float(clearance[0, 1]), float(clearance[0, 0]))

    def test_behavior_mode_support_uses_next_action_and_updates_actor_only(self):
        torch.manual_seed(13)
        model = make_model(action=3)
        batch = make_batch(b=1, t=3, action=3)
        feature = torch.zeros(1, 3, 32)
        with torch.no_grad():
            model.actor.linear.weight.zero_()
            model.actor.linear.bias.zero_()
            batch["action"].zero_()
            # Row zero produced observation zero and must not supervise the
            # Actor at source state zero.
            batch["action"][:, 0, 0] = 1.0
        zero_loss, _ = model.behavior_mode_support_objective(
            {"actor_feat": feature}, batch)
        self.assertAlmostEqual(float(zero_loss.detach()), 0.0, places=7)

        batch["action"][:, 1, 0] = 0.50
        loss, metrics = model.behavior_mode_support_objective(
            {"actor_feat": feature}, batch)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(metrics["rejected_sample_ratio"]), 0.0)
        loss.backward()
        self.assertIsNotNone(model.actor.linear.bias.grad)
        self.assertLess(float(model.actor.linear.bias.grad[0]), 0.0)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder.parameters()))

    def test_complete_optimizer_step(self):
        torch.manual_seed(9)
        model = make_model(action=2)
        batch = make_batch(action=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        actor_before = model.actor.linear.weight.detach().clone()
        value_before = model.value.linear.weight.detach().clone()
        trainer = FactorizedTrainStep(
            model, optimizer, imag_horizon=3, loss_scales=self.LOSS_SCALES,
        )
        metrics = trainer(batch)
        self.assertIn("loss/total", metrics)
        for component in (
            "dyn_ego", "dyn_human", "ego_recon", "human_mpjpe",
            "rew", "con", "policy", "safety_policy", "behavior_clone", "value", "repval",
            "risk_collision", "risk_clearance", "risk_false_safe",
            "risk_false_danger",
            "collision_risk",
            "clearance_barrier", "unsafe_speed", "action_smoothness",
            "replay_unsafe_speed", "vertical_action",
            "lateral_candidate", "goal_directed",
        ):
            self.assertIn(f"loss/{component}", metrics)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertEqual(metrics["opt/step_skipped"], 0.0)
        self.assertGreater(metrics["grad/actor_norm"], 0.0)
        self.assertGreater(metrics["grad/value_norm"], 0.0)
        self.assertFalse(torch.equal(actor_before, model.actor.linear.weight))
        self.assertFalse(torch.equal(value_before, model.value.linear.weight))
        for branch_state in trainer.last_states.values():
            for value in branch_state.values():
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)

    def test_risk_auxiliary_uses_transition_class_balance(self):
        torch.manual_seed(12)
        model = make_model(action=2)
        batch = make_batch(action=2)
        batch["future_collision"].zero_()
        batch["future_collision"][0, -2] = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = FactorizedTrainStep(
            model, optimizer, imag_horizon=3,
            loss_scales=self.LOSS_SCALES,
            separate_risk_auxiliary=True,
        )
        risk_before = next(model.action_risk.parameters()).detach().clone()
        trainer(batch)
        self.assertTrue(torch.equal(
            risk_before, next(model.action_risk.parameters())))
        actor_before = model.actor.linear.weight.detach().clone()
        metrics = trainer.risk_auxiliary_step(batch)
        self.assertGreater(
            metrics["diagnostic/collision_positive_weight"],
            model.risk_collision_positive_weight,
        )
        self.assertTrue(torch.equal(actor_before, model.actor.linear.weight))
        self.assertFalse(torch.equal(
            risk_before, next(model.action_risk.parameters())))

    def test_ego_displacement_anchor_cancels_constant_decoder_bias(self):
        source_truth = torch.zeros(2, 14)
        source_truth[:, 12:14] = torch.tensor((0.0, 1.0))
        target_truth = source_truth.clone()
        target_truth[:, :3] = torch.tensor((1.0, -0.5, 0.1))
        target_yaw = torch.tensor(0.2)
        target_truth[:, 12] = torch.sin(target_yaw)
        target_truth[:, 13] = torch.cos(target_yaw)
        source_decoded = source_truth.clone()
        target_decoded = target_truth.clone()
        bias = torch.tensor((4.0, -3.0, 0.6))
        source_decoded[:, :3] += bias
        target_decoded[:, :3] += bias
        anchored = FactorizedDreamer.anchor_ego_displacement(
            source_truth, source_decoded, target_decoded)
        torch.testing.assert_close(anchored[:, :3], target_truth[:, :3])
        torch.testing.assert_close(anchored[:, 12:14], target_truth[:, 12:14])

    def test_multistep_overshooting_backpropagates_human_and_risk(self):
        torch.manual_seed(13)
        model = make_model(action=3, overshoot=(5, 10))
        batch = make_batch(t=12, action=3)
        batch["future_collision"][:, 7:].fill_(True)
        batch["future_min_human_clearance_m"][:, 7:].fill_(0.2)
        losses, _, aux = model.world_model_loss(batch)
        expected = {
            "human_overshoot_5", "human_overshoot_10",
            "risk_collision_overshoot_5", "risk_collision_overshoot_10",
            "risk_clearance_overshoot_5", "risk_clearance_overshoot_10",
            "risk_false_safe_overshoot_5",
            "risk_false_safe_overshoot_10",
            "risk_false_danger_overshoot_5",
            "risk_false_danger_overshoot_10",
        }
        self.assertTrue(expected.issubset(losses))
        self.assertIn("h10/human_fde_m", aux["overshoot_metrics"])
        sum(losses[name] for name in expected).backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.rssm.human_rssm.parameters()))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.action_risk.parameters()))

    def test_human_subset_tail_metrics_use_their_own_masks(self):
        """Section-10 subgroup gates must not all report the final mask."""
        model = make_model(action=3, overshoot=(1,))
        model.human_only_overshooting = True
        batch = make_batch(b=4, t=2, n=1, action=3)
        batch["action"].zero_()
        batch["human_root"].zero_()
        batch["skeleton"].zero_()
        batch["priv_min_human_clearance_m"].fill_(2.0)
        batch["priv_min_human_clearance_m"][1, 1] = 0.1
        batch["human_persistence_mask"] = torch.zeros(
            4, 2, 1, dtype=torch.bool)
        batch["human_persistence_mask"][2, 0, 0] = True
        batch["action"][3, 1, 0] = 0.99
        learned_error = torch.arange(1.0, 5.0).reshape(4, 1, 1, 1)
        baseline_error = 10.0 * learned_error

        def factual_rollout(
            _self, _states, _batch, _indices, _horizon,
            *, use_recorded_ego=False, rollout_valid=None,
        ):
            del _self, _states, _horizon, use_recorded_ego, rollout_valid
            root = learned_error.expand(-1, 1, 1, 3).clone()
            baseline_root = baseline_error.expand(-1, 1, 1, 3).clone()
            joints = root[..., None, :].expand(-1, -1, -1, 12, -1)
            baseline_joints = baseline_root[..., None, :].expand(
                -1, -1, -1, 12, -1)
            return {
                "root": root,
                "constant_velocity_root": baseline_root,
                "zero_velocity_root": torch.zeros_like(root),
                "joints": joints,
                "constant_velocity_joints": baseline_joints,
                "source_ids": _batch["human_ids"][:, 0],
                "source_mask": _batch["human_mask"][:, 0],
            }

        model.human_factual_rollout = MethodType(factual_rollout, model)
        _, metrics = model.multistep_overshooting_loss({}, batch)

        # Each row belongs to exactly one subgroup.  The 3-D vector norm adds
        # sqrt(3), but the four source amplitudes remain distinguishable.
        scale = 3.0 ** 0.5
        expected = {
            "ordinary": 1.0 * scale,
            "near_collision": 2.0 * scale,
            "occlusion": 3.0 * scale,
            "action_saturation": 4.0 * scale,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(
                float(metrics[f"h1/subset_{name}_human_fde_p95_m"]),
                value,
                places=5,
            )

    def test_ego_multistep_overshooting_uses_deployed_path_and_backpropagates(self):
        torch.manual_seed(41)
        model = make_model(action=3, overshoot=(1, 3))
        batch = make_batch(t=5, action=3)
        states, _, _ = model.posterior(batch)
        losses, metrics = model.ego_multistep_overshooting_loss(states, batch)
        self.assertEqual(set(losses), {"ego_overshoot_1", "ego_overshoot_3"})
        self.assertIn("h3/ego_position_fde_m", metrics)
        self.assertGreater(float(metrics["h3/ego_valid_start_count"]), 0.0)
        sum(losses.values()).backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.rssm.ego_rssm.parameters()))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.prediction_heads.ego.parameters()))

    def test_left_padded_short_episode_only_selects_real_overshoot_windows(self):
        """Padding must neither crash h15 nor replace the few usable starts."""
        torch.manual_seed(42)
        model = make_model(action=3, overshoot=(1, 3))
        model.human_only_overshooting = True
        batch = make_batch(b=1, t=8, n=2, action=3)
        batch["sequence_valid"] = torch.zeros(1, 8, 1, dtype=torch.bool)
        batch["sequence_valid"][:, 5:] = True
        batch["action_valid"].zero_()
        batch["action_valid"][:, 6:] = True
        batch["dt_s"] = torch.zeros(1, 8, 1)
        batch["dt_s"][:, 6:] = 0.1
        batch["is_first"].zero_()
        batch["is_first"][:, 5] = True
        batch["human_is_first"].zero_()
        batch["human_is_first"][:, 5] = True
        batch["is_last"].zero_()
        batch["is_last"][:, 7] = True

        h1_indices, h1_valid = model._overshoot_start_indices(batch, 1)
        self.assertEqual(h1_indices.tolist(), [[5, 6]])
        self.assertEqual(h1_valid.tolist(), [True, True])
        h3_indices, h3_valid = model._overshoot_start_indices(batch, 3)
        self.assertEqual(h3_indices.tolist(), [[0, 0]])
        self.assertEqual(h3_valid.tolist(), [False, False])

        states, _, _ = model.posterior(batch)
        ego_losses, ego_metrics = model.ego_multistep_overshooting_loss(
            states, batch)
        human_losses, human_metrics = model.multistep_overshooting_loss(
            states, batch)
        self.assertEqual(float(ego_metrics["h3/ego_valid_start_count"]), 0.0)
        self.assertEqual(float(human_metrics["h3/valid_start_count"]), 0.0)
        for loss in (*ego_losses.values(), *human_losses.values()):
            self.assertTrue(bool(torch.isfinite(loss)))

    def test_ego_factual_rollout_consumes_target_row_action(self):
        torch.manual_seed(43)
        model = make_model(action=3, overshoot=(1,))
        batch = make_batch(b=1, t=3, action=3)
        states, _, _ = model.posterior(batch)
        indices = torch.zeros(1, 1, dtype=torch.long)
        torch.manual_seed(44)
        reference = model.ego_factual_rollout(
            states, batch, indices, 1)["ego_state"]
        source_changed = {**batch, "action": batch["action"].clone()}
        source_changed["action"][:, 0] = 100.0
        torch.manual_seed(44)
        source_result = model.ego_factual_rollout(
            states, source_changed, indices, 1)["ego_state"]
        torch.testing.assert_close(source_result, reference)
        target_changed = {**batch, "action": batch["action"].clone()}
        target_changed["action"][:, 1] = -target_changed["action"][:, 1]
        torch.manual_seed(44)
        target_result = model.ego_factual_rollout(
            states, target_changed, indices, 1)["ego_state"]
        self.assertFalse(torch.allclose(target_result, reference))

    def test_h1_human_motion_decodes_destination_prior(self):
        """h1 must supervise the action-selected Human prior transition."""
        torch.manual_seed(47)
        model = make_model(action=3, overshoot=(1,))
        batch = make_batch(b=1, t=3, n=2, action=3)
        states, _, _ = model.posterior(batch)
        detached_states = {
            branch: {
                key: value.detach() for key, value in fields.items()
            }
            for branch, fields in states.items()
        }
        batch["action"] = batch["action"].detach().requires_grad_(True)
        indices = torch.zeros(1, 1, dtype=torch.long)

        rollout = model.human_factual_rollout(
            detached_states, batch, indices, 1, use_recorded_ego=True)
        objective = rollout["root"].sum() + rollout["joints"].sum()
        objective.backward()

        self.assertGreater(
            float(batch["action"].grad[:, 1].abs().sum()), 0.0)
        self.assertTrue(any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.rssm.human_rssm.parameters()
        ))

    def test_required_gt_displacement_labels_join_human_overshooting(self):
        torch.manual_seed(29)
        model = make_model(action=3, overshoot=(5,))
        model.human_only_overshooting = True
        model.require_gt_displacement_supervision = True
        batch = make_batch(t=7, action=3)
        with self.assertRaisesRegex(KeyError, "GT displacement supervision"):
            model.world_model_loss(batch)

        batch["human_gt_id"] = batch["human_ids"].clone()
        batch["human_gt_match_valid"] = batch["human_mask"].clone()
        batch["human_gt_pelvis_body"] = batch["human_root"][..., :3].clone()
        batch["human_gt_pelvis_episode"] = (
            batch["human_root"][..., :3].clone())
        losses, _, aux = model.world_model_loss(batch)
        gt_loss = losses["human_gt_displacement_5"]
        self.assertTrue(bool(torch.isfinite(gt_loss)))
        self.assertGreater(
            float(aux["overshoot_metrics"]["h5/human_gt_matched_count"]),
            0.0)
        gt_loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.rssm.human_rssm.parameters()))

    def test_privileged_joint_clearance_supervises_articulated_rollout(self):
        torch.manual_seed(31)
        model = make_model(action=3, overshoot=(5,))
        model.human_only_overshooting = True
        batch = make_batch(t=7, action=3)
        batch["sequence_valid"] = torch.ones(2, 7, 1, dtype=torch.bool)
        batch["privileged_geometry_available"] = torch.ones(
            2, 7, 1, dtype=torch.bool)
        # Simulator collision geometry is episode-local.  Put every valid GT
        # joint at a known body-relative offset along the recorded Ego path.
        relative = torch.zeros(2, 7, 3, 10, 3)
        relative[..., 0] = 0.55
        relative[..., 1] = torch.linspace(-0.10, 0.10, 10)
        batch["priv_collision_joints_episode"] = (
            relative + batch["ego_state"][..., None, None, :3])
        batch["priv_collision_joint_valid"] = torch.ones(
            2, 7, 3, 10, dtype=torch.bool)
        batch["priv_human_id"] = batch["human_ids"].clone()
        batch["priv_human_mask"] = batch["human_mask"].clone()
        batch["human_gt_id"] = batch["human_ids"].clone()
        batch["human_gt_match_valid"] = batch["human_mask"].clone()

        losses, _, aux = model.world_model_loss(batch)
        loss = losses["human_joint_clearance_5"]
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(
            float(aux["overshoot_metrics"][
                "h5/joint_clearance_valid_count"]), 0.0)
        # Two starts per sequence x two sequences x five transitions x three
        # source identities.  A scene-level minimum used to report only 20
        # rows and allowed one Human's prediction to explain another Human's
        # PhysX clearance target.
        self.assertEqual(
            float(aux["overshoot_metrics"][
                "h5/joint_clearance_valid_count"]), 60.0)
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.prediction_heads.human.parameters()))

    def test_safe_clearance_false_danger_calibration(self):
        model = make_model()
        target = torch.tensor([0.3, 0.6, 1.2, 1.8])
        valid = torch.ones_like(target, dtype=torch.bool)
        conservative = torch.tensor([0.3, 0.6, 0.4, 0.5])
        calibrated = torch.tensor([0.3, 0.6, 1.2, 1.8])

        conservative_terms = model._balanced_clearance_supervision(
            conservative, target, valid)
        calibrated_terms = model._balanced_clearance_supervision(
            calibrated, target, valid)

        self.assertGreater(float(conservative_terms["false_danger_loss"]), 0.0)
        self.assertEqual(float(calibrated_terms["false_danger_loss"]), 0.0)
        self.assertEqual(float(conservative_terms["false_safe_loss"]), 0.0)
        self.assertGreater(
            float(conservative_terms["regression_loss"]),
            float(calibrated_terms["regression_loss"]),
        )

        unsafe_overestimate = torch.tensor([1.2, 1.1, 1.2, 1.8])
        unsafe_terms = model._balanced_clearance_supervision(
            unsafe_overestimate, target, valid)
        self.assertGreater(float(unsafe_terms["false_safe_loss"]), 0.0)

    def test_world_model_only_step_keeps_actor_and_value_unchanged(self):
        torch.manual_seed(14)
        model = make_model(action=2, overshoot=(5, 10))
        batch = make_batch(t=12, action=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        actor_before = model.actor.linear.weight.detach().clone()
        value_before = model.value.linear.weight.detach().clone()
        trainer = FactorizedTrainStep(
            model, optimizer, imag_horizon=3, loss_scales=self.LOSS_SCALES,
            world_model_only=True,
        )
        metrics = trainer(batch)
        torch.testing.assert_close(actor_before, model.actor.linear.weight)
        torch.testing.assert_close(value_before, model.value.linear.weight)
        self.assertGreater(metrics["grad/human_rssm_norm"], 0.0)
        self.assertGreater(metrics["grad/action_risk_norm"], 0.0)

    def test_validation_step_does_not_update_parameters(self):
        torch.manual_seed(12)
        model = make_model(action=2)
        batch = make_batch(action=2)
        trainer = FactorizedTrainStep(
            model, torch.optim.Adam(model.parameters(), lr=1e-3),
            imag_horizon=3, loss_scales=self.LOSS_SCALES,
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = trainer.evaluate(batch)
        self.assertIn("loss/world_total", metrics)
        self.assertIn("loss/behavior_clone", metrics)
        self.assertIn("loss/goal_directed", metrics)
        self.assertIn("goal_directed/local_danger_ratio", metrics)
        self.assertIn("policy/action_mae", metrics)
        self.assertIn("selection/score", metrics)
        self.assertIn("selection/safety_calibration_score", metrics)
        self.assertIn("selection/online_world_score", metrics)
        self.assertIn("selection/online_actor_score", metrics)
        self.assertIn("imag/reward_mean", metrics)
        self.assertIn("imag/safety_cost_mean", metrics)
        self.assertIn("prediction/human_root_ade", metrics)
        self.assertTrue(model.training)
        for old, new in zip(before, model.parameters()):
            torch.testing.assert_close(old, new)

    def test_amp_dtype_contract(self):
        self.assertEqual(resolve_amp_dtype("bf16"), ("bfloat16", torch.bfloat16))
        self.assertEqual(resolve_amp_dtype("fp16"), ("float16", torch.float16))
        self.assertEqual(resolve_amp_dtype("float32"), ("float32", torch.float32))
        with self.assertRaisesRegex(ValueError, "Unsupported amp_dtype"):
            resolve_amp_dtype("tf32")

        model = make_model(action=2)
        trainer = FactorizedTrainStep(
            model, torch.optim.Adam(model.parameters(), lr=1e-3),
            imag_horizon=3, loss_scales=self.LOSS_SCALES,
            amp_device="cpu", amp_dtype="bfloat16", amp_init_scale=1.0,
        )
        self.assertEqual(trainer.amp_dtype_name, "bfloat16")
        self.assertFalse(trainer.autocast_enabled)
        self.assertFalse(trainer.scaler.is_enabled())

        invalid_model = make_model(action=2)
        with self.assertRaisesRegex(ValueError, "amp_init_scale must be positive"):
            FactorizedTrainStep(
                invalid_model,
                torch.optim.Adam(invalid_model.parameters(), lr=1e-3),
                imag_horizon=3, loss_scales=self.LOSS_SCALES,
                amp_init_scale=0.0,
            )


if __name__ == "__main__":
    unittest.main()
