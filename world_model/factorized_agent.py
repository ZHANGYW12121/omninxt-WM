"""OnlineTrainer-compatible construction and lifecycle for FactorizedDreamer."""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

import networks
from factorized_dreamer import FactorizedDreamer
from factorized_rssm import FactorizedRSSM
from factorized_trainer import FactorizedTrainStep
from modules.factorized_encoders import FactorizedEncoderConfig, FactorizedObservationEncoder
from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads
from modules.action_risk_head import ActionRiskHead
from modules.action_adapter import ActionAdapterConfig, HorizontalActionAdapter
from modules.action_smoother import (
    ActionSmootherConfig,
    StatefulActionSmoother,
)
from modules.factorized_actor_v62 import (
    FactorizedActorV62, FactorizedActorV62Config,
    PermutationInvariantDecisionHead,
)
from modules.latent_policy_attention import LatentPolicyAttentionConfig
from modules.sparse_ego_human_attention import SparseEgoHumanAttentionConfig
from modules.transition_event_head import (
    HumanSafetyCritic,
    NextHumanClearanceHead,
    TransitionEventHead,
)
from modules.task_geometry import task_physical_state_dim
from optim import LaProp


def _shape(space):
    return tuple(int(x) for x in space.shape)


def load_compatible_factorized_state(
    agent: "FactorizedDreamerAgent",
    source_state: dict[str, torch.Tensor],
) -> dict[str, tuple[str, ...]]:
    """Load exact-shape weights across factorized architecture revisions.

    v6.3 retains the v6.2 goal/avoid/gate mean branches but explicitly resets
    Actor std, transition heads, and both Safety Critics. Older migrations may
    still initialize SafetyValue from Value when v6.3 is disabled.
    """
    target = agent.state_dict()
    compatible = {
        name: value for name, value in source_state.items()
        if name in target and target[name].shape == value.shape
        and not (
            getattr(agent, "v63_enabled", False)
            and (
                name.startswith("model.actor.std_net.")
                or name.startswith("model.safety_value.")
                or name.startswith("model.slow_safety_value.")
                or name.startswith("model.transition_event.")
                or name.startswith("model.next_human_clearance.")
                or name.startswith("model.safety_return_ema.")
                or name in (
                    "model.safety_lambda",
                    "model.safety_cost_ewma",
                    "model.safety_lambda_update_count",
                )
            )
        )
    }
    migrated = []
    for target_prefix, source_prefix in (
        ("model.safety_value.", "model.value."),
        ("model.slow_safety_value.", "model.slow_value."),
    ):
        if getattr(agent, "v63_enabled", False):
            break
        for target_name, target_value in target.items():
            if not target_name.startswith(target_prefix) or target_name in compatible:
                continue
            source_name = source_prefix + target_name[len(target_prefix):]
            source_value = source_state.get(source_name)
            if source_value is not None and source_value.shape == target_value.shape:
                compatible[target_name] = source_value
                migrated.append(f"{source_name}->{target_name}")
    agent.load_state_dict(compatible, strict=False)
    skipped_source = tuple(sorted(
        name for name in source_state if name not in compatible))
    initialized_target = tuple(sorted(
        name for name in target if name not in compatible))
    return {
        "loaded": tuple(sorted(compatible)),
        "migrated": tuple(sorted(migrated)),
        "skipped_source": skipped_source,
        "initialized_target": initialized_target,
    }


class FactorizedDreamerAgent(nn.Module):
    """Adapter exposing the same act/update/state surface used by OnlineTrainer."""

    is_factorized = True

    def __init__(self, config, obs_space, act_space) -> None:
        super().__init__()
        self.device = torch.device(config.device)
        try:
            from omegaconf import OmegaConf
            resolved_config = OmegaConf.to_container(
                config, resolve=True, enum_to_str=True)
        except Exception:
            resolved_config = str(config)
        serialized_config = json.dumps(
            resolved_config, sort_keys=True, separators=(",", ":"), default=str)
        self.resolved_config_hash = hashlib.sha256(
            serialized_config.encode("utf-8")).hexdigest()
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        fcfg = config.factorized
        v62_cfg = fcfg.get("v62", None)
        self.v62_enabled = bool(
            v62_cfg is not None and v62_cfg.get("enabled", False))
        v63_cfg = fcfg.get("v63", None)
        self.v63_enabled = bool(
            v63_cfg is not None and v63_cfg.get("enabled", False))
        if self.v63_enabled and not self.v62_enabled:
            raise ValueError("v6.3 extends the v6.2 factorized Actor contract")
        if self.v62_enabled and self.act_dim != 4:
            raise ValueError("v6.2 requires the applied 4D flight action space")
        self.policy_act_dim = 3 if self.v62_enabled else self.act_dim
        private_dim = int(fcfg.private_dim)
        if "goal" not in obs_space.spaces:
            raise ValueError("two-branch factorized observations require goal [G]")
        goal_dim = _shape(obs_space.spaces["goal"])[-1]
        encoder = FactorizedObservationEncoder(FactorizedEncoderConfig(
            model_dim=private_dim,
            ego_state_mean=tuple(fcfg.ego_state_mean) if fcfg.get("ego_state_mean") is not None else None,
            ego_state_std=tuple(fcfg.ego_state_std) if fcfg.get("ego_state_std") is not None else None,
            ego_metric_scaling=bool(
                fcfg.get("ego_metric_scaling", False)),
            human_root_dim=int(fcfg.human_root_dim),
            human_root_metric_scaling=bool(
                fcfg.get("human_root_metric_scaling", False)),
            human_quality_metric_scaling=bool(
                fcfg.get("human_quality_metric_scaling", False)),
            human_feat_dim=int(fcfg.get("human_feature_dim", 7)),
            use_stgcn=True, pose_history=int(fcfg.pose_history),
            observation_heads=int(fcfg.observation_attention.num_heads),
            observation_ff_mult=int(fcfg.observation_attention.ff_mult),
        ))
        latent_cfg = LatentPolicyAttentionConfig(
            model_dim=int(fcfg.latent_policy_attention.model_dim),
            num_heads=int(fcfg.latent_policy_attention.num_heads),
            num_layers=int(fcfg.latent_policy_attention.num_layers),
            ff_mult=int(fcfg.latent_policy_attention.ff_mult),
            goal_metric_scaling=bool(
                fcfg.get("goal_metric_scaling", False)),
        )
        dynamics = FactorizedRSSM(
            config.rssm, {"ego": private_dim, "human": private_dim}, self.act_dim,
            goal_dim=goal_dim, latent_attention_config=latent_cfg,
            coupling_attention_config=SparseEgoHumanAttentionConfig(
                model_dim=int(fcfg.prior_coupling.model_dim),
                num_heads=int(fcfg.prior_coupling.num_heads),
                ff_mult=int(fcfg.prior_coupling.ff_mult),
            ),
        )
        joint_dim = dynamics.joint_feat_size
        config.actor.shape = (
            (self.policy_act_dim,) if self.v62_enabled
            else ((act_space.n,) if hasattr(act_space, "n") else _shape(act_space))
        )
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
        else:
            config.actor.dist = config.actor.dist.cont
        if self.v62_enabled:
            actor_human_slots = int(
                v62_cfg.get("human_physical_slots", 0))
            actor_human_fields = int(
                v62_cfg.get("human_physical_fields_per_slot", 6))
            if bool(v62_cfg.get("unified_policy", False)) and (
                actor_human_slots != int(fcfg.max_people)
            ):
                raise ValueError(
                    "unified full-state Actor must preserve every configured "
                    "Human perception slot")
            per_slot_learned_state = bool(
                self.v63_enabled
                and v63_cfg.get("actor_full_learned_per_slot", False))
            learned_fields_per_slot = (
                joint_dim - min(12, joint_dim)
                if per_slot_learned_state else 0)
            actor_human_geometry_dim = (
                actor_human_slots * (
                    actor_human_fields + learned_fields_per_slot)
                if bool(v62_cfg.get("unified_policy", False))
                and per_slot_learned_state
                else joint_dim + actor_human_slots * actor_human_fields
                if bool(v62_cfg.get("unified_policy", False))
                else joint_dim
            )
            actor_task_obstacle_slots = int(
                v62_cfg.get("task_physical_obstacle_slots", 0))
            actor_task_geometry_dim = (
                task_physical_state_dim(actor_task_obstacle_slots)
                if bool(v62_cfg.get("unified_policy", False))
                and actor_task_obstacle_slots > 0 else 0
            )
            actor = FactorizedActorV62(FactorizedActorV62Config(
                token_dim=joint_dim,
                hidden_dim=int(v62_cfg.get("actor_hidden_dim", 256)),
                policy_action_dim=self.policy_act_dim,
                residual_bound=float(v62_cfg.get("residual_bound", 0.75)),
                goal_output_scale=float(config.actor.outscale),
                min_std=(
                    tuple(float(value) for value in v63_cfg.std_min)
                    if self.v63_enabled else float(config.actor.dist.min_std)
                ),
                max_std=(
                    tuple(float(value) for value in v63_cfg.std_max)
                    if self.v63_enabled else float(config.actor.dist.max_std)
                ),
                initial_std=(
                    tuple(float(value) for value in v63_cfg.std_initial)
                    if self.v63_enabled else None
                ),
                std_parameterization=str(
                    v63_cfg.get("std_parameterization", "annealed_cap")),
                pre_tanh_mean_bound=float(
                    v62_cfg.get("pre_tanh_mean_bound", 0.0)),
                pre_tanh_mean_parameterization=str(v62_cfg.get(
                    "pre_tanh_mean_parameterization", "tanh")),
                initial_gate_bias=float(
                    v62_cfg.get("initial_gate_bias", 0.0)),
                human_conditioned_residual=bool(
                    v62_cfg.get("human_conditioned_residual", False)),
                human_reflection_equivariant=bool(
                    v62_cfg.get("human_reflection_equivariant", False)),
                human_physical_slots=actor_human_slots,
                human_physical_fields_per_slot=actor_human_fields,
                human_physical_presence_state=bool(
                    v62_cfg.get("human_physical_presence_state", False)),
                human_geometry_dim=actor_human_geometry_dim,
                task_geometry_dim=actor_task_geometry_dim,
                unified_policy=bool(v62_cfg.get("unified_policy", False)),
                permutation_invariant_entities=bool(v62_cfg.get(
                    "permutation_invariant_entities", False)),
                human_learned_fields_per_slot=learned_fields_per_slot,
                task_physical_obstacle_slots=actor_task_obstacle_slots,
                decision_entity_encoder_contract=str(v62_cfg.get(
                    "decision_entity_encoder_contract",
                    "clearance_cpa_exact_interaction_context_query_v3",
                )),
            ))
            adapter_cfg = v62_cfg.action_adapter
            action_adapter = HorizontalActionAdapter(ActionAdapterConfig(
                target_altitude_agl_m=float(
                    adapter_cfg.target_altitude_agl_m),
                altitude_kp=float(adapter_cfg.altitude_kp),
                vertical_velocity_kd=float(
                    adapter_cfg.vertical_velocity_kd),
                maximum_vertical_action=float(
                    adapter_cfg.maximum_vertical_action),
            ))
            smoother_cfg = v63_cfg.get("action_smoothing", {})
            action_smoother = (
                StatefulActionSmoother(ActionSmootherConfig(
                    time_step_s=float(smoother_cfg.get("time_step_s", 0.1)),
                    time_constant_s=float(
                        smoother_cfg.get("time_constant_s", 0.35)),
                    slew_rate_per_s=tuple(float(value) for value in
                        smoother_cfg.get(
                            "slew_rate_per_s", (1.2, 1.0, 0.45))),
                    parameterization=str(smoother_cfg.get(
                        "parameterization", "legacy_absolute_target")),
                ))
                if self.v63_enabled
                and bool(smoother_cfg.get("enabled", False))
                else None
            )
        else:
            actor = networks.MLPHead(config.actor, joint_dim)
            action_adapter = None
            action_smoother = None
        critic_input_dim = (
            int(actor.input_dim)
            if self.v62_enabled
            and bool(v62_cfg.get("unified_policy", False))
            and bool(v62_cfg.get("critic_full_state", False))
            else joint_dim
        )
        if (
            self.v62_enabled
            and bool(v62_cfg.get("unified_policy", False))
            and bool(v62_cfg.get("critic_full_state", False))
            and bool(v62_cfg.get("permutation_invariant_entities", False))
        ):
            critic_encoder = copy.deepcopy(actor.decision_encoder)
            value = PermutationInvariantDecisionHead(
                critic_encoder,
                networks.MLPHead(config.critic, critic_encoder.output_dim),
            )
        else:
            value = networks.MLPHead(config.critic, critic_input_dim)
        safety_hidden_dim = int(
            v63_cfg.get("safety_critic_hidden_dim", 256)
            if self.v63_enabled else 256)
        safety_value = (
            HumanSafetyCritic(joint_dim, safety_hidden_dim)
            if self.v63_enabled else networks.MLPHead(config.critic, joint_dim)
        )
        reward = networks.MLPHead(config.reward, joint_dim)
        cont = networks.MLPHead(config.cont, joint_dim)

        human_dynamics_cfg = fcfg.get("human_dynamics", {})
        prediction_cfg = FactorizedPredictionConfig(
            hidden_dim=int(config.get("units", 256)),
            num_joints=int(fcfg.num_joints),
            kinematic_velocity_only=bool(
                human_dynamics_cfg.get("kinematic_velocity_only", False)),
            max_velocity_residual_mps=float(
                human_dynamics_cfg.get("max_velocity_residual_mps", 0.35)),
            max_root_speed_mps=(
                None if human_dynamics_cfg.get(
                    "max_root_speed_mps", None) is None
                else float(human_dynamics_cfg.max_root_speed_mps)),
            kinematic_joint_velocity_only=bool(
                human_dynamics_cfg.get(
                    "kinematic_joint_velocity_only", False)),
            max_joint_velocity_residual_mps=float(
                human_dynamics_cfg.get(
                    "max_joint_velocity_residual_mps", 0.35)),
            max_joint_speed_mps=(
                None if human_dynamics_cfg.get(
                    "max_joint_speed_mps", None) is None
                else float(human_dynamics_cfg.max_joint_speed_mps)),
        )
        prediction = FactorizedPredictionHeads(
            dynamics.ego_rssm.feat_size, dynamics.human_rssm.feat_size, prediction_cfg,
        )
        safety_cfg = fcfg.offline_actor.safety
        overshoot_cfg = fcfg.get("overshooting", None)
        action_risk = ActionRiskHead(
            joint_dim, self.act_dim,
            hidden_dim=int(safety_cfg.risk_hidden_dim),
        )
        transition_event = (
            TransitionEventHead(
                joint_dim,
                self.act_dim,
                hidden_dim=int(v63_cfg.get("event_hidden_dim", 256)),
                ego_feat_dim=dynamics.ego_rssm.feat_size,
                human_feat_dim=dynamics.human_rssm.feat_size,
                human_root_dim=int(fcfg.human_root_dim),
                human_quality_dim=7,
                explicit_human_geometry=bool(
                    v63_cfg.get("explicit_human_geometry", False)),
                geometry_topk_physical_slots=int(
                    v63_cfg.get("geometry_topk_physical_slots", 0)),
                explicit_joint_kinematics=bool(
                    v63_cfg.get("explicit_joint_kinematics", False)),
                explicit_human_presence_physical=bool(
                    v63_cfg.get(
                        "explicit_human_presence_physical", False)),
                analytic_task_memory_events=bool(
                    v63_cfg.get("analytic_task_memory_events", False)),
                actor_full_state_slots=(
                    actor_human_slots
                    if bool(v62_cfg.get("unified_policy", False)) else 0),
                actor_full_fields_per_slot=(
                    actor_human_fields
                    if bool(v62_cfg.get("unified_policy", False)) else None),
                actor_full_learned_per_slot=bool(
                    v63_cfg.get("actor_full_learned_per_slot", False)),
                event_full_articulated_state=bool(
                    v63_cfg.get("event_full_articulated_state", False)),
            ) if self.v63_enabled else None
        )
        next_human_clearance = (
            NextHumanClearanceHead(
                joint_dim,
                self.act_dim,
                hidden_dim=int(v63_cfg.get("clearance_hidden_dim", 256)),
                ego_feat_dim=dynamics.ego_rssm.feat_size,
                human_feat_dim=dynamics.human_rssm.feat_size,
                human_root_dim=int(fcfg.human_root_dim),
                human_quality_dim=7,
                explicit_human_geometry=bool(
                    v63_cfg.get("explicit_human_geometry", False)),
            ) if self.v63_enabled else None
        )
        self.model = FactorizedDreamer(
            encoder, dynamics, actor, value, reward, cont, action_risk, prediction,
            action_adapter=action_adapter, safety_value=safety_value,
            action_smoother=action_smoother,
            transition_event=transition_event,
            next_human_clearance=next_human_clearance,
            kl_free=float(config.kl_free), horizon=int(config.horizon), lamb=float(config.lamb),
            act_entropy=float(config.act_entropy), slow_target_fraction=float(config.slow_target_fraction),
            risk_collision_positive_weight=float(
                safety_cfg.collision_positive_weight),
            risk_clearance_normalization_m=float(
                safety_cfg.clearance_normalization_m),
            safe_clearance_m=float(safety_cfg.safe_clearance_m),
            planning_safe_clearance_m=float(
                safety_cfg.planning_safe_clearance_m),
            clearance_temperature_m=float(safety_cfg.clearance_temperature_m),
            safe_forward_action=float(safety_cfg.safe_forward_action),
            behavior_clone_beta=float(fcfg.offline_actor.behavior_clone_beta),
            behavior_clone_min_clearance_m=float(
                fcfg.offline_actor.behavior_clone_min_clearance_m),
            clearance_danger_weight=float(
                safety_cfg.clearance_danger_weight),
            clearance_near_weight=float(
                safety_cfg.clearance_near_weight),
            clearance_far_weight=float(
                safety_cfg.clearance_far_weight),
            clearance_overestimate_weight=float(
                safety_cfg.clearance_overestimate_weight),
            clearance_safe_margin_m=float(
                safety_cfg.clearance_safe_margin_m),
            safety_collision_cost=float(
                safety_cfg.imagination_collision_cost),
            safety_clearance_cost=float(
                safety_cfg.imagination_clearance_cost),
            safety_speed_cost=float(
                safety_cfg.imagination_speed_cost),
            safety_return_lambda=float(safety_cfg.return_lambda),
            safety_rollout_samples=int(safety_cfg.rollout_samples),
            lateral_candidate_action=float(
                safety_cfg.lateral_candidate_action),
            lateral_candidate_slow_forward_action=float(
                safety_cfg.lateral_candidate_slow_forward_action),
            lateral_candidate_rollout_steps=int(
                safety_cfg.lateral_candidate_rollout_steps),
            lateral_candidate_commit_steps=int(
                safety_cfg.get("lateral_candidate_commit_steps", 2)),
            lateral_candidate_collision_cost=float(
                safety_cfg.lateral_candidate_collision_cost),
            lateral_candidate_clearance_cost=float(
                safety_cfg.lateral_candidate_clearance_cost),
            lateral_candidate_progress_cost=float(
                safety_cfg.lateral_candidate_progress_cost),
            lateral_candidate_action_change_cost=float(
                safety_cfg.lateral_candidate_action_change_cost),
            lateral_candidate_min_score_margin=float(
                safety_cfg.lateral_candidate_min_score_margin),
            lateral_candidate_min_hold_improvement=float(
                safety_cfg.get(
                    "lateral_candidate_min_hold_improvement", 0.02)),
            lateral_candidate_local_clearance_m=float(
                safety_cfg.get("lateral_candidate_local_clearance_m", 1.20)),
            lateral_candidate_imminent_ttc_s=float(
                safety_cfg.get("lateral_candidate_imminent_ttc_s", 1.50)),
            lateral_candidate_geometry_steps=int(
                safety_cfg.get("lateral_candidate_geometry_steps", 5)),
            lateral_candidate_step_duration_s=float(
                safety_cfg.get("lateral_candidate_step_duration_s", 0.10)),
            lateral_candidate_surface_radius_m=float(
                safety_cfg.get("lateral_candidate_surface_radius_m", 0.25)),
            lateral_candidate_geometry_cost=float(
                safety_cfg.get("lateral_candidate_geometry_cost", 1.0)),
            risk_counterfactual_scale=float(
                safety_cfg.get("risk_counterfactual_scale", 0.5)),
            goal_directed_forward_action=float(
                safety_cfg.get("goal_directed_forward_action", 0.42)),
            goal_directed_lateral_action=float(
                safety_cfg.get("goal_directed_lateral_action", 0.42)),
            goal_directed_slow_radius_m=float(
                safety_cfg.get("goal_directed_slow_radius_m", 3.0)),
            goal_directed_deadband=float(
                safety_cfg.get("goal_directed_deadband", 0.03)),
            danger_lateral_smoothness_scale=float(
                safety_cfg.danger_lateral_smoothness_scale),
            planner_collision_trigger=float(
                safety_cfg.online_planner.collision_trigger),
            planner_clearance_trigger_m=float(
                safety_cfg.online_planner.clearance_trigger_m),
            planner_emergency_collision_trigger=float(
                safety_cfg.online_planner.get(
                    "emergency_collision_trigger", 0.25)),
            planner_emergency_clearance_m=float(
                safety_cfg.online_planner.get(
                    "emergency_clearance_m", 0.60)),
            planner_emergency_forward_action=float(
                safety_cfg.online_planner.get(
                    "emergency_forward_action", 0.0)),
            planner_observed_caution_clearance_m=float(
                safety_cfg.online_planner.get(
                    "observed_caution_clearance_m", 1.8)),
            planner_observed_emergency_clearance_m=float(
                safety_cfg.online_planner.get(
                    "observed_emergency_clearance_m", 1.0)),
            planner_observed_forced_escape_clearance_m=float(
                safety_cfg.online_planner.get(
                    "observed_forced_escape_clearance_m", 0.7)),
            planner_observed_hard_stop_clearance_m=float(
                safety_cfg.online_planner.get(
                    "observed_hard_stop_clearance_m", 0.5)),
            planner_observed_critical_clearance_m=float(
                safety_cfg.online_planner.get(
                    "observed_critical_clearance_m", 0.25)),
            planner_critical_repulsion_blend=float(
                safety_cfg.online_planner.get(
                    "critical_repulsion_blend", 0.35)),
            planner_emergency_escape_min_action=float(
                safety_cfg.online_planner.get(
                    "emergency_escape_min_action", 0.20)),
            planner_emergency_escape_action=float(
                safety_cfg.online_planner.get(
                    "emergency_escape_action", 0.55)),
            planner_min_score_improvement=float(
                safety_cfg.online_planner.min_score_improvement),
            planner_speed_governor=bool(
                safety_cfg.online_planner.get("speed_governor", True)),
            overshoot_horizons=(
                tuple(int(value) for value in overshoot_cfg.horizons)
                if overshoot_cfg is not None
                and bool(overshoot_cfg.get("enabled", False))
                else ()
            ),
            overshoot_starts_per_sequence=(
                int(overshoot_cfg.starts_per_sequence)
                if overshoot_cfg is not None else 2
            ),
            human_only_overshooting=(
                bool(overshoot_cfg.get("human_only", False))
                if overshoot_cfg is not None else False
            ),
            collision_tail_regret_supervision=(
                bool(overshoot_cfg.get("collision_tail_regret", False))
                if overshoot_cfg is not None else False
            ),
            require_gt_displacement_supervision=(
                bool(overshoot_cfg.get(
                    "require_gt_displacement_labels", False))
                if overshoot_cfg is not None else False
            ),
            policy_reward_component_weights=dict(
                fcfg.offline_actor.policy_reward_component_weights),
            imagination_support_delta_limits=(
                tuple(float(value) for value in v63_cfg.support_delta_limits)
                if self.v63_enabled else (0.05, 0.12, 0.03)),
            imagination_support_violation_limit=(
                float(v63_cfg.get("support_violation_limit", 0.10))
                if self.v63_enabled else 0.10),
            behavior_mode_support_delta_limits=(
                tuple(float(value) for value in v63_cfg.get(
                    "behavior_mode_support_delta_limits", (0.15, 0.25, 0.10)))
                if self.v63_enabled else (0.15, 0.25, 0.10)),
            task_goal_event_reward=(
                float(v63_cfg.task_event_rewards.reached_goal)
                if self.v63_enabled else 100.0),
            task_human_collision_reward=(
                float(v63_cfg.task_event_rewards.get(
                    "human_collision", -120.0))
                if self.v63_enabled else -120.0),
            task_static_collision_reward=(
                float(v63_cfg.task_event_rewards.static_collision)
                if self.v63_enabled else -120.0),
            task_other_terminal_reward=(
                float(v63_cfg.task_event_rewards.other_task_terminal)
                if self.v63_enabled else -100.0),
            task_stuck_terminal_reward=(
                float(v63_cfg.task_event_rewards.get(
                    "stuck_timeout", -20.0))
                if self.v63_enabled else -20.0),
            safety_probability_discount=(
                float(v63_cfg.get("safety_gamma", 1.0))
                if self.v63_enabled else 1.0),
            analytic_ego_velocity_response=(
                tuple(float(value) for value in v63_cfg.get(
                    "analytic_ego_velocity_response", (1.0, 1.0, 1.0)))
                if self.v63_enabled else (1.0, 1.0, 1.0)),
            analytic_ego_attitude_coefficients=(
                tuple(
                    tuple(float(value) for value in row)
                    for row in v63_cfg.get(
                        "analytic_ego_attitude_coefficients", ())
                )
                if self.v63_enabled and v63_cfg.get(
                    "analytic_ego_attitude_coefficients") is not None
                else None),
        )
        # This is deliberately assigned after construction because it changes
        # only the Actor readout contract, not the coupled world dynamics.
        self.model.actor_authoritative_ego_token_enabled = bool(
            v62_cfg.get("authoritative_ego_token", False)
            if self.v62_enabled else False)
        self.model.actor_direct_ego_task_state_enabled = bool(
            v62_cfg.get("direct_ego_task_state", False)
            if self.v62_enabled else False)
        self.model.actor_task_physical_obstacle_slots = int(
            actor_task_obstacle_slots if self.v62_enabled else 0)
        self.model.actor_task_physical_state_enabled = bool(
            self.v62_enabled and actor_task_geometry_dim > 0)
        self.model.critic_full_state_enabled = bool(
            v62_cfg.get("critic_full_state", False)
            if self.v62_enabled else False)
        self.model.deterministic_evaluation_state_enabled = bool(
            v63_cfg.get("deterministic_evaluation_state", False)
            if self.v63_enabled else False)
        self.model.direct_task_geometry_enabled = bool(
            v63_cfg.get("direct_task_geometry", False)
            if self.v63_enabled else False)
        self.model.task_memory_enabled = bool(
            v63_cfg.get("task_memory", False)
            if self.v63_enabled else False)
        self.model.direct_task_memory_enabled = bool(
            v63_cfg.get("direct_task_memory", False)
            if self.v63_enabled else False)
        # Historical runs used min(learned, CV) as the Actor return.  v18 keeps
        # CV as an explicit audit baseline but trains against one coherent
        # learned kinematic world so gradients and reported predictions share
        # the same semantics.
        self.model.conservative_human_clearance_enabled = bool(
            human_dynamics_cfg.get("conservative_cv_clearance", True))
        if self.v63_enabled:
            self.model.register_buffer(
                "actor_local_step", torch.zeros((), dtype=torch.long))
        self._named_params = OrderedDict(self.model.named_parameters())
        if self.v62_enabled:
            excluded = (
                "actor.", "value.", "slow_value.", "safety_value.",
                "slow_safety_value.", "action_risk.",
            )
            self._world_named_params = OrderedDict(
                (name, parameter)
                for name, parameter in self._named_params.items()
                if not name.startswith(excluded)
            )
        else:
            self._world_named_params = self._named_params
        if self.v63_enabled:
            event_prefixes = ("transition_event.", "next_human_clearance.")
            event_parameters = [
                parameter for name, parameter in self._world_named_params.items()
                if name.startswith(event_prefixes)
            ]
            base_world_parameters = [
                parameter for name, parameter in self._world_named_params.items()
                if not name.startswith(event_prefixes)
            ]
            if not event_parameters:
                raise ValueError("v6.3 event parameter group is empty")
            self._optimizer = LaProp(
                [
                    {"params": base_world_parameters},
                    {
                        "params": event_parameters,
                        "lr": float(v63_cfg.get("event_lr", 3.0e-4)),
                    },
                ],
                lr=config.lr,
                betas=(config.beta1, config.beta2), eps=config.eps,
            )
        else:
            self._optimizer = LaProp(
                self._world_named_params.values(), lr=config.lr,
                betas=(config.beta1, config.beta2), eps=config.eps,
            )
        self._actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(), lr=float(
                v62_cfg.get("actor_lr", config.lr)
                if self.v62_enabled else config.lr),
            eps=1.0e-8, weight_decay=1.0e-4,
        )
        self._task_critic_optimizer = torch.optim.AdamW(
            self.model.value.parameters(), lr=float(
                v62_cfg.get("critic_lr", config.lr)
                if self.v62_enabled else config.lr),
            eps=1.0e-8, weight_decay=1.0e-4,
        )
        self._safety_critic_optimizer = torch.optim.AdamW(
            self.model.safety_value.parameters(), lr=float(
                v62_cfg.get("critic_lr", config.lr)
                if self.v62_enabled else config.lr),
            eps=1.0e-8, weight_decay=1.0e-4,
        )
        base_world_schedule = (
            lambda step: min(1.0, (step + 1) / config.warmup)
            if config.warmup else 1.0)
        self._scheduler = LambdaLR(
            self._optimizer,
            lr_lambda=(
                [base_world_schedule, lambda _step: 1.0]
                if self.v63_enabled else base_world_schedule),
        )
        # Risk calibration has a different sampling distribution and must not
        # share LaProp moments with RSSM/Actor/Critic optimization.
        self._risk_optimizer = torch.optim.AdamW(
            self.model.action_risk.parameters(),
            lr=float(config.get("risk_lr", 1.0e-5)),
            eps=1.0e-8,
            weight_decay=float(config.get("risk_weight_decay", 1.0e-4)),
        )
        self._train_step = FactorizedTrainStep(
            self.model, self._optimizer, imag_horizon=int(config.imag_horizon),
            loss_scales=dict(config.loss_scales), agc=float(config.agc), pmin=float(config.pmin),
            amp_device=str(config.device),
            amp_dtype=str(config.get("amp_dtype", "bfloat16")),
            amp_init_scale=float(config.get("amp_init_scale", 1.0)),
            slow_target_update=int(config.slow_target_update),
            behavior_warmup_steps=int(
                fcfg.offline_actor.behavior_warmup_steps),
            policy_ramp_steps=int(fcfg.offline_actor.policy_ramp_steps),
            behavior_clone_decay_steps=int(
                fcfg.offline_actor.get("behavior_clone_decay_steps", 0)),
            behavior_clone_final_scale=float(
                fcfg.offline_actor.get("behavior_clone_final_scale", 1.0)),
            safety_warmup_steps=int(
                fcfg.offline_actor.safety.warmup_steps),
            safety_ramp_steps=int(
                fcfg.offline_actor.safety.ramp_steps),
            selection_weights=dict(fcfg.offline_actor.selection_weights),
            risk_optimizer=self._risk_optimizer,
            risk_positive_weight_max=float(
                fcfg.offline_actor.safety.get(
                    "online_positive_weight_max", 8.0)),
            risk_calibration_scale=float(
                fcfg.offline_actor.safety.get(
                    "online_calibration_scale", 1.0)),
            risk_balanced_scale=float(
                fcfg.offline_actor.safety.get(
                    "online_balanced_scale", 0.5)),
            actor_std_schedule_steps=(
                int(v63_cfg.get("std_schedule_steps", 0))
                if self.v63_enabled else 0),
            module_optimizers=(
                {
                    "world": self._optimizer,
                    "actor": self._actor_optimizer,
                    "task_critic": self._task_critic_optimizer,
                    "safety_critic": self._safety_critic_optimizer,
                }
                if self.v62_enabled else None
            ),
        )
        self.max_people = int(fcfg.max_people)

    def _state_dict(self, state):
        return self.model.state_from_replay_fields({key: state[key] for key in (
            "ego_stoch", "ego_deter", "human_stoch", "human_deter")})

    @torch.no_grad()
    def get_initial_state(self, batch_size):
        latent = self.model.rssm.initial(batch_size, self.max_people)
        fields = self.model.replay_state_fields(latent)
        fields["prev_action"] = torch.zeros(batch_size, self.act_dim, device=self.device)
        return TensorDict(fields, batch_size=(batch_size,))

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        latent = self._state_dict(state)
        action, next_state = self.model.act_step(
            obs, {"latent": latent, "prev_action": state["prev_action"]}, evaluation=eval,
        )
        fields = self.model.replay_state_fields(next_state["latent"])
        fields["prev_action"] = action
        return action, TensorDict(fields, batch_size=state.batch_size)

    def update(self, replay_buffer):
        data, index, initial_fields = replay_buffer.sample()
        initial = self.model.state_from_replay_fields(initial_fields)
        metrics = self._train_step(data, initial)
        if (
            not self._train_step.last_step_skipped
            and (
                self._train_step.module_optimizers is None
                or self._train_step.world_optimizer_stepped
            )
        ):
            self._scheduler.step()
        metrics["opt/lr"] = float(self._scheduler.get_last_lr()[0])
        if hasattr(replay_buffer, "update_factorized"):
            replay_buffer.update_factorized(
                index, self.model.replay_state_fields(self._train_step.last_states)
            )
        return metrics

    def training_state_dict(self):
        return {
            "optimizer": self._optimizer.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "task_critic_optimizer": self._task_critic_optimizer.state_dict(),
            "safety_critic_optimizer": self._safety_critic_optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": self._train_step.scaler.state_dict(),
            "update_count": self._train_step.update_count,
            "actor_local_update_count": self._train_step.actor_local_update_count,
            "skipped_update_count": self._train_step.skipped_update_count,
            "risk_aux_update_count": self._train_step.risk_aux_update_count,
            "event_aux_update_count": self._train_step.event_aux_update_count,
            "risk_optimizer": self._risk_optimizer.state_dict(),
            "behavior_clone_decay_enabled": bool(
                self._train_step.behavior_clone_decay_enabled),
            "actor_updates_enabled": bool(
                self._train_step.actor_updates_enabled),
            "closed_loop_baseline_observed": bool(
                self._train_step.closed_loop_baseline_observed),
            "resolved_config_hash": self.resolved_config_hash,
            "safety_lambda": float(self.model.safety_lambda.detach()),
            "task_return_ema": self.model.return_ema.state_dict(),
            # Kept only for loading historical v6.2 checkpoints. v6.3 safety
            # advantages are never normalized by this adaptive statistic.
            "safety_return_ema": (
                None if self.v63_enabled
                else self.model.safety_return_ema.state_dict()),
        }

    def load_training_state_dict(self, payload):
        saved_config_hash = payload.get("resolved_config_hash")
        if (
            saved_config_hash is not None
            and str(saved_config_hash) != self.resolved_config_hash
        ):
            raise RuntimeError(
                "Refusing optimizer resume with a different resolved config: "
                f"{saved_config_hash} != {self.resolved_config_hash}")
        if "optimizer" in payload: self._optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload: self._scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload: self._train_step.scaler.load_state_dict(payload["scaler"])
        if "risk_optimizer" in payload:
            self._risk_optimizer.load_state_dict(payload["risk_optimizer"])
        if "actor_optimizer" in payload:
            self._actor_optimizer.load_state_dict(
                payload["actor_optimizer"])
        if "task_critic_optimizer" in payload:
            self._task_critic_optimizer.load_state_dict(
                payload["task_critic_optimizer"])
        if "safety_critic_optimizer" in payload:
            self._safety_critic_optimizer.load_state_dict(
                payload["safety_critic_optimizer"])
        if "task_return_ema" in payload:
            self.model.return_ema.load_state_dict(
                payload["task_return_ema"])
        if "safety_return_ema" in payload and payload["safety_return_ema"] is not None:
            self.model.safety_return_ema.load_state_dict(
                payload["safety_return_ema"])
        if "safety_lambda" in payload:
            self.model.safety_lambda.fill_(
                float(payload["safety_lambda"]))
        self._train_step.update_count = int(payload.get("update_count", 0))
        self._train_step.actor_local_update_count = int(
            payload.get(
                "actor_local_update_count",
                int(getattr(self.model, "actor_local_step", 0)),
            )
        )
        if self.v63_enabled:
            self.model.actor_local_step.fill_(
                self._train_step.actor_local_update_count)
            # std_schedule_progress is a model buffer saved in the exact v6.3
            # state dict. It advances only after an in-support Actor update;
            # rebuilding it from the local clock would silently bypass that
            # gate after resume.
        self._train_step.skipped_update_count = int(
            payload.get("skipped_update_count", 0)
        )
        self._train_step.risk_aux_update_count = int(
            payload.get("risk_aux_update_count", 0)
        )
        self._train_step.event_aux_update_count = int(
            payload.get("event_aux_update_count", 0)
        )
        self._train_step.behavior_clone_decay_enabled = bool(
            payload.get("behavior_clone_decay_enabled", True))
        self._train_step.actor_updates_enabled = bool(
            payload.get("actor_updates_enabled", True))
        self._train_step.closed_loop_baseline_observed = bool(
            payload.get("closed_loop_baseline_observed", False))

    @torch.no_grad()
    def video_pred(self, data, initial):
        raise NotImplementedError("Factorized model logs BEV/skeleton open-loop metrics instead of RGB video")
