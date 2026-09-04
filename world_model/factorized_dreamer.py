"""DreamerV3 world-model core backed by :class:`FactorizedRSSM`.

This module is intentionally separate from the original ``dreamer.Dreamer``.
It establishes the factorized posterior, shared task heads, and world-model
loss API first; online replay/optimizer orchestration is added without changing
the vanilla agent.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
import networks

from factorized_rssm import FactorizedRSSM, FactorizedState
from modules.goal_conditioning import goal_features_torch
from modules.task_reward import analytic_progress_reward
from modules.task_geometry import (
    TASK_GEOMETRY_DIM,
    TASK_GEOMETRY_MAX_RANGE_M,
    episode_to_world_geometry_torch,
    episode_to_world_task_physical_state_torch,
    episode_to_world_task_clearance_torch,
    task_geometry_feature,
)
from modules.task_memory import (
    TASK_MEMORY_DIM,
    TASK_MEMORY_SCALE,
    task_memory_feature,
    task_memory_step_torch,
)
from modules.reward_components import (
    REWARD_COMPONENT_KEYS,
    RewardComponentHead,
    symexp,
)
from modules.action_risk_head import ActionRiskHead
from modules.action_adapter import HorizontalActionAdapter
from modules.action_smoother import StatefulActionSmoother
from modules.se2_relative_dynamics import (
    DEPLOYABLE_DRONE_COLLISION_RADIUS_M,
    analytic_ego_step,
    analytic_joint_signed_surface_gap,
    analytic_joint_surface_gap,
    analytic_relative_human_step,
    body_points_to_episode,
    coco12_collision_spheres,
    persistent_rollout_joint_mask,
    swept_relative_point_signed_gap,
)
from modules.transition_event_head import (
    CONTINUE_INDEX,
    HUMAN_COLLISION_INDEX,
    TRANSITION_EVENT_KEYS,
    HumanSafetyCritic,
    NextHumanClearanceHead,
    TransitionEventHead,
    survival_human_collision_return,
)
from modules.task_reward import rescale_interval_probability

_EVENT_STATIC_COLLISION_INDEX = 2
_EVENT_REACHED_GOAL_INDEX = 3
_EVENT_OTHER_TERMINAL_INDEX = 4


R2_STRUCTURED_CANDIDATE_NAMES = (
    "actor",
    "goal_aligned",
    "brake_hold",
    "reverse",
    "strong_left",
    "strong_right",
    "brake_left",
    "brake_right",
    "accelerate_left",
    "accelerate_right",
)


class FactorizedDreamer(nn.Module):
    """Factorized world model plus one joint Actor/Critic/Reward/Continue set."""

    def __init__(
        self,
        encoder: nn.Module,
        dynamics: FactorizedRSSM,
        actor: nn.Module,
        value: nn.Module,
        reward: nn.Module,
        cont: nn.Module,
        action_risk: ActionRiskHead,
        prediction_heads: nn.Module | None = None,
        *,
        transition_event: TransitionEventHead | None = None,
        next_human_clearance: NextHumanClearanceHead | None = None,
        action_adapter: HorizontalActionAdapter | None = None,
        action_smoother: StatefulActionSmoother | None = None,
        safety_value: nn.Module | None = None,
        kl_free: float = 1.0,
        horizon: int = 333,
        lamb: float = 0.95,
        act_entropy: float = 3.0e-4,
        slow_target_fraction: float = 0.02,
        risk_collision_positive_weight: float = 2.0,
        risk_clearance_normalization_m: float = 2.0,
        safe_clearance_m: float = 0.7,
        planning_safe_clearance_m: float | None = None,
        clearance_temperature_m: float = 0.1,
        safe_forward_action: float = 0.30,
        behavior_clone_beta: float = 0.1,
        behavior_clone_min_clearance_m: float = 0.50,
        clearance_danger_weight: float = 1.0,
        clearance_near_weight: float = 2.0,
        clearance_far_weight: float = 0.5,
        clearance_overestimate_weight: float = 3.0,
        clearance_safe_margin_m: float = 1.0,
        safety_collision_cost: float = 1.0,
        safety_clearance_cost: float = 0.5,
        safety_speed_cost: float = 0.25,
        safety_return_lambda: float = 0.8,
        safety_rollout_samples: int = 2,
        lateral_candidate_action: float = 0.35,
        lateral_candidate_slow_forward_action: float = 0.10,
        lateral_candidate_rollout_steps: int = 5,
        lateral_candidate_commit_steps: int = 2,
        lateral_candidate_collision_cost: float = 1.0,
        lateral_candidate_clearance_cost: float = 0.5,
        lateral_candidate_progress_cost: float = 0.25,
        lateral_candidate_action_change_cost: float = 0.05,
        lateral_candidate_min_score_margin: float = 0.01,
        lateral_candidate_min_hold_improvement: float = 0.02,
        lateral_candidate_local_clearance_m: float = 1.20,
        lateral_candidate_imminent_ttc_s: float = 1.50,
        lateral_candidate_geometry_steps: int = 5,
        lateral_candidate_step_duration_s: float = 0.10,
        lateral_candidate_surface_radius_m: float = 0.25,
        lateral_candidate_geometry_cost: float = 1.0,
        risk_counterfactual_scale: float = 0.5,
        goal_directed_forward_action: float = 0.42,
        goal_directed_lateral_action: float = 0.42,
        goal_directed_slow_radius_m: float = 3.0,
        goal_directed_deadband: float = 0.03,
        danger_lateral_smoothness_scale: float = 0.20,
        planner_collision_trigger: float = 0.15,
        planner_clearance_trigger_m: float = 0.80,
        planner_emergency_collision_trigger: float = 0.25,
        planner_emergency_clearance_m: float = 0.60,
        planner_emergency_forward_action: float = 0.0,
        planner_observed_caution_clearance_m: float = 1.80,
        planner_observed_emergency_clearance_m: float = 1.00,
        planner_observed_forced_escape_clearance_m: float = 0.70,
        planner_observed_hard_stop_clearance_m: float = 0.50,
        planner_observed_critical_clearance_m: float = 0.25,
        planner_critical_repulsion_blend: float = 0.35,
        planner_emergency_escape_min_action: float = 0.20,
        planner_emergency_escape_action: float = 0.55,
        planner_min_score_improvement: float = 0.03,
        planner_speed_governor: bool = True,
        overshoot_horizons: tuple[int, ...] = (),
        overshoot_starts_per_sequence: int = 2,
        human_only_overshooting: bool = False,
        collision_tail_regret_supervision: bool = False,
        require_gt_displacement_supervision: bool = False,
        policy_reward_component_weights: dict[str, float] | None = None,
        imagination_support_delta_limits: tuple[float, ...] = (0.05, 0.12, 0.03),
        imagination_support_violation_limit: float = 0.10,
        behavior_mode_support_delta_limits: tuple[float, ...] = (0.15, 0.25, 0.10),
        task_goal_event_reward: float = 100.0,
        task_human_collision_reward: float = -120.0,
        task_static_collision_reward: float = -120.0,
        task_other_terminal_reward: float = -100.0,
        task_stuck_terminal_reward: float = -20.0,
        safety_probability_discount: float = 1.0,
        analytic_ego_velocity_response: tuple[float, float, float] = (
            1.0, 1.0, 1.0),
        analytic_ego_attitude_coefficients: tuple[
            tuple[float, float], ...] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # v6.9 enables a zero-initialized learned static-task adapter. Legacy
        # runtimes leave this false when loading v6.8.
        self.task_geometry_enabled = False
        self.direct_task_geometry_enabled = False
        self.task_memory_enabled = False
        self.direct_task_memory_enabled = False
        # Legacy/R2 policies used the Event geometry network as a fourth Actor
        # token.  Pure v5 disables that coupling and consumes the RSSM's
        # trained Human pool instead; the explicit hazard remains trainable as
        # a world-model diagnostic/auxiliary.
        self.actor_explicit_human_geometry_enabled = True
        # When enabled, the Actor goal branch consumes Ego14 through the
        # observation Ego encoder.  The coupled RSSM Ego state remains intact
        # for world prediction, while Humans can affect action only through
        # the explicit unified full-state Actor input.
        self.actor_authoritative_ego_token_enabled = False
        # Fresh complete-state policies reserve independent exact coordinates
        # for Ego14, TaskGeometry8 and TaskMemory9.  Summing learned/tiled task
        # adapters is useful context for world heads but is not injective and
        # therefore cannot by itself constitute a complete Markov Actor state.
        self.actor_direct_ego_task_state_enabled = False
        # A separately concatenated exact state exposes all world-fixed task
        # constraints used by the analytic return.  Its width/capacity is
        # versioned by the Actor config and deployment checkpoint.
        self.actor_task_physical_state_enabled = False
        self.actor_task_physical_obstacle_slots = 0
        # Fresh pure Dreamer constructs Value on the exact same complete
        # deployable decision state as Actor.  Legacy models retain the old
        # joint-token Critic by leaving this compatibility switch false.
        self.critic_full_state_enabled = False
        # A deterministic Actor mode is not sufficient for deterministic
        # deployment when its posterior state is still sampled.  Keep this an
        # explicit versioned flag so legacy checkpoints retain their exact
        # online-state semantics while current evaluation uses posterior modes.
        self.deterministic_evaluation_state_enabled = False
        # Compatibility default for pre-v18 configurations.  Pure Dreamer v18
        # disables this and retains constant velocity as a diagnostic baseline
        # instead of silently changing the learned model's return with a min.
        self.conservative_human_clearance_enabled = True
        self.rssm = dynamics
        self.actor = actor
        self.value = value
        self.safety_value = (
            copy.deepcopy(value) if safety_value is None else safety_value)
        self.action_adapter = action_adapter
        self.action_smoother = action_smoother
        self.uses_internal_action_adapter = action_adapter is not None
        if (
            self.action_smoother is not None
            and not self.uses_internal_action_adapter
        ):
            raise ValueError(
                "stateful action smoothing requires the horizontal adapter")
        self.reward = reward
        # The shared task contract assigns Human/static collision the same
        # -120 Event reward.  Parameterize this learned reward output on the
        # configured terminal support so recursive imagination cannot turn a
        # small symlog extrapolation into a physically impossible reward.
        self.reward_components = RewardComponentHead(
            dynamics.joint_feat_size,
            event_value_bounds=(
                min(0.0, task_static_collision_reward,
                    task_other_terminal_reward),
                max(0.0, task_goal_event_reward),
            ),
        )
        reward_weights = {
            "event": 1.0,
            "progress": 1.0,
            "human_clearance": 0.0,
            "smoothness": 1.0,
            "height": 1.0,
            "time": 1.0,
            **({} if policy_reward_component_weights is None
               else policy_reward_component_weights),
        }
        unknown_reward_components = set(reward_weights).difference(
            REWARD_COMPONENT_KEYS)
        if unknown_reward_components:
            raise ValueError(
                "Unknown policy reward components: "
                f"{sorted(unknown_reward_components)}")
        self.policy_reward_component_weights = tuple(
            float(reward_weights[key]) for key in REWARD_COMPONENT_KEYS)
        self.imagination_support_delta_limits = tuple(
            float(value) for value in imagination_support_delta_limits)
        self.imagination_support_violation_limit = float(
            imagination_support_violation_limit)
        self.behavior_mode_support_delta_limits = tuple(
            float(value) for value in behavior_mode_support_delta_limits)
        self.task_goal_event_reward = float(task_goal_event_reward)
        self.task_human_collision_reward = float(
            task_human_collision_reward)
        self.task_static_collision_reward = float(
            task_static_collision_reward)
        self.task_other_terminal_reward = float(task_other_terminal_reward)
        self.task_stuck_terminal_reward = float(task_stuck_terminal_reward)
        self.analytic_ego_velocity_response = tuple(
            float(value) for value in analytic_ego_velocity_response)
        if (
            len(self.analytic_ego_velocity_response) != 3
            or any(
                not 0.0 < value <= 1.0
                for value in self.analytic_ego_velocity_response)
        ):
            raise ValueError(
                "analytic Ego velocity response must contain three values in "
                "(0,1]")
        self.analytic_ego_attitude_coefficients = (
            None
            if analytic_ego_attitude_coefficients is None
            else tuple(
                tuple(float(value) for value in row)
                for row in analytic_ego_attitude_coefficients
            )
        )
        if self.analytic_ego_attitude_coefficients is not None:
            coefficients = torch.tensor(
                self.analytic_ego_attitude_coefficients, dtype=torch.float64)
            if coefficients.shape != (7, 2) or not torch.isfinite(
                coefficients
            ).all():
                raise ValueError(
                    "analytic Ego attitude coefficients must be finite [7,2]")
            attitude_transition = coefficients[1:3].T
            spectral_radius = torch.linalg.eigvals(
                attitude_transition).abs().max()
            if float(spectral_radius) >= 1.0:
                raise ValueError(
                    "analytic Ego attitude transition must be asymptotically stable")
        self.safety_probability_discount = float(safety_probability_discount)
        if self.safety_probability_discount != 1.0:
            raise ValueError(
                "v6.3 episode human-collision probability requires safety gamma=1")
        if any(value <= 0.0 for value in self.imagination_support_delta_limits):
            raise ValueError("imagination support delta limits must be positive")
        if not 0.0 <= self.imagination_support_violation_limit < 1.0:
            raise ValueError("support violation limit must lie in [0,1)")
        if any(value <= 0.0 for value in self.behavior_mode_support_delta_limits):
            raise ValueError("behavior mode support limits must be positive")
        self.cont = cont
        self.action_risk = action_risk
        self.transition_event = transition_event
        self.next_human_clearance = next_human_clearance
        self.v63_enabled = (
            transition_event is not None and next_human_clearance is not None
            and isinstance(self.safety_value, HumanSafetyCritic)
        )
        self.prediction_heads = prediction_heads
        self.kl_free = float(kl_free)
        self.horizon = int(horizon)
        self.lamb = float(lamb)
        self.act_entropy = float(act_entropy)
        self.slow_target_fraction = float(slow_target_fraction)
        self.risk_collision_positive_weight = float(
            risk_collision_positive_weight)

        self.risk_clearance_normalization_m = float(
            risk_clearance_normalization_m)
        self.safe_clearance_m = float(safe_clearance_m)
        self.planning_safe_clearance_m = float(
            safe_clearance_m
            if planning_safe_clearance_m is None
            else planning_safe_clearance_m)
        self.clearance_temperature_m = float(clearance_temperature_m)
        self.safe_forward_action = float(safe_forward_action)
        self.behavior_clone_beta = float(behavior_clone_beta)
        self.behavior_clone_min_clearance_m = float(
            behavior_clone_min_clearance_m)
        self.clearance_danger_weight = float(clearance_danger_weight)
        self.clearance_near_weight = float(clearance_near_weight)
        self.clearance_far_weight = float(clearance_far_weight)
        self.clearance_overestimate_weight = float(
            clearance_overestimate_weight)
        self.clearance_safe_margin_m = float(clearance_safe_margin_m)
        self.safety_collision_cost = float(safety_collision_cost)
        self.safety_clearance_cost = float(safety_clearance_cost)
        self.safety_speed_cost = float(safety_speed_cost)
        self.safety_return_lambda = float(safety_return_lambda)
        self.safety_rollout_samples = int(safety_rollout_samples)
        self.lateral_candidate_action = float(lateral_candidate_action)
        self.lateral_candidate_slow_forward_action = float(
            lateral_candidate_slow_forward_action)
        self.lateral_candidate_rollout_steps = int(
            lateral_candidate_rollout_steps)
        self.lateral_candidate_commit_steps = int(
            lateral_candidate_commit_steps)
        self.lateral_candidate_collision_cost = float(
            lateral_candidate_collision_cost)
        self.lateral_candidate_clearance_cost = float(
            lateral_candidate_clearance_cost)
        self.lateral_candidate_progress_cost = float(
            lateral_candidate_progress_cost)
        self.lateral_candidate_action_change_cost = float(
            lateral_candidate_action_change_cost)
        self.lateral_candidate_min_score_margin = float(
            lateral_candidate_min_score_margin)
        self.lateral_candidate_min_hold_improvement = float(
            lateral_candidate_min_hold_improvement)
        self.lateral_candidate_local_clearance_m = float(
            lateral_candidate_local_clearance_m)
        self.lateral_candidate_imminent_ttc_s = float(
            lateral_candidate_imminent_ttc_s)
        self.lateral_candidate_geometry_steps = int(
            lateral_candidate_geometry_steps)
        self.lateral_candidate_step_duration_s = float(
            lateral_candidate_step_duration_s)
        self.lateral_candidate_surface_radius_m = float(
            lateral_candidate_surface_radius_m)
        self.lateral_candidate_geometry_cost = float(
            lateral_candidate_geometry_cost)
        self.risk_counterfactual_scale = float(risk_counterfactual_scale)
        self.goal_directed_forward_action = float(
            goal_directed_forward_action)
        self.goal_directed_lateral_action = float(
            goal_directed_lateral_action)
        self.goal_directed_slow_radius_m = float(
            goal_directed_slow_radius_m)
        self.goal_directed_deadband = float(goal_directed_deadband)
        self.danger_lateral_smoothness_scale = float(
            danger_lateral_smoothness_scale)
        self.planner_collision_trigger = float(planner_collision_trigger)
        self.planner_clearance_trigger_m = float(
            planner_clearance_trigger_m)
        self.planner_emergency_collision_trigger = float(
            planner_emergency_collision_trigger)
        self.planner_emergency_clearance_m = float(
            planner_emergency_clearance_m)
        self.planner_emergency_forward_action = float(
            planner_emergency_forward_action)
        self.planner_observed_caution_clearance_m = float(
            planner_observed_caution_clearance_m)
        self.planner_observed_emergency_clearance_m = float(
            planner_observed_emergency_clearance_m)
        self.planner_observed_forced_escape_clearance_m = float(
            planner_observed_forced_escape_clearance_m)
        self.planner_observed_hard_stop_clearance_m = float(
            planner_observed_hard_stop_clearance_m)
        self.planner_observed_critical_clearance_m = float(
            planner_observed_critical_clearance_m)
        self.planner_critical_repulsion_blend = float(
            planner_critical_repulsion_blend)
        self.planner_emergency_escape_min_action = float(
            planner_emergency_escape_min_action)
        self.planner_emergency_escape_action = float(
            planner_emergency_escape_action)
        self.planner_min_score_improvement = float(
            planner_min_score_improvement)
        self.planner_speed_governor = bool(planner_speed_governor)
        self.overshoot_horizons = tuple(sorted({
            int(value) for value in overshoot_horizons
        }))
        self.overshoot_starts_per_sequence = int(
            overshoot_starts_per_sequence)
        self.human_only_overshooting = bool(human_only_overshooting)
        self.collision_tail_regret_supervision = bool(
            collision_tail_regret_supervision)
        self.require_gt_displacement_supervision = bool(
            require_gt_displacement_supervision)
        if (
            self.risk_collision_positive_weight <= 0.0
            or self.risk_clearance_normalization_m <= 0.0
        ):
            raise ValueError("risk loss scaling parameters must be positive")
        if (
            self.safe_clearance_m <= 0.0
            or self.planning_safe_clearance_m <= 0.0
            or self.clearance_temperature_m <= 0.0
        ):
            raise ValueError("clearance safety parameters must be positive")
        if not 0.0 <= self.safe_forward_action <= 1.0:
            raise ValueError("safe_forward_action must be within [0, 1]")
        if self.behavior_clone_beta <= 0.0:
            raise ValueError("behavior_clone_beta must be positive")
        if self.behavior_clone_min_clearance_m < 0.0:
            raise ValueError(
                "behavior_clone_min_clearance_m must be non-negative")
        if min(
            self.clearance_danger_weight,
            self.clearance_near_weight,
            self.clearance_far_weight,
            self.clearance_overestimate_weight,
        ) <= 0.0:
            raise ValueError("clearance sample weights must be positive")
        if self.clearance_safe_margin_m <= self.safe_clearance_m:
            raise ValueError(
                "clearance_safe_margin_m must exceed safe_clearance_m")
        if min(
            self.safety_collision_cost,
            self.safety_clearance_cost,
            self.safety_speed_cost,
        ) < 0.0:
            raise ValueError("imagined safety cost weights must be non-negative")
        if not 0.0 <= self.safety_return_lambda <= 1.0:
            raise ValueError("safety_return_lambda must be within [0, 1]")
        if self.safety_rollout_samples < 2:
            raise ValueError(
                "safety_rollout_samples must be at least 2 for a same-state baseline")
        if not 0.0 < self.lateral_candidate_action <= 1.0:
            raise ValueError("lateral_candidate_action must be within (0, 1]")
        if not 0.0 <= self.lateral_candidate_slow_forward_action <= 1.0:
            raise ValueError(
                "lateral_candidate_slow_forward_action must be within [0, 1]")
        if self.lateral_candidate_rollout_steps <= 0:
            raise ValueError("lateral_candidate_rollout_steps must be positive")
        if not 0 < self.lateral_candidate_commit_steps <= self.lateral_candidate_rollout_steps:
            raise ValueError(
                "lateral_candidate_commit_steps must be in "
                "[1, lateral_candidate_rollout_steps]")
        if min(
            self.lateral_candidate_collision_cost,
            self.lateral_candidate_clearance_cost,
            self.lateral_candidate_progress_cost,
            self.lateral_candidate_action_change_cost,
            self.lateral_candidate_min_score_margin,
            self.lateral_candidate_min_hold_improvement,
        ) < 0.0:
            raise ValueError("lateral candidate cost parameters must be non-negative")
        if self.lateral_candidate_local_clearance_m <= 0.0:
            raise ValueError("lateral_candidate_local_clearance_m must be positive")
        if self.lateral_candidate_imminent_ttc_s <= 0.0:
            raise ValueError("lateral_candidate_imminent_ttc_s must be positive")
        if self.lateral_candidate_geometry_steps <= 0:
            raise ValueError("lateral_candidate_geometry_steps must be positive")
        if self.lateral_candidate_step_duration_s <= 0.0:
            raise ValueError("lateral_candidate_step_duration_s must be positive")
        if self.lateral_candidate_surface_radius_m <= 0.0:
            raise ValueError("lateral_candidate_surface_radius_m must be positive")
        if self.goal_directed_slow_radius_m <= 0.0:
            raise ValueError("goal_directed_slow_radius_m must be positive")
        if not 0.0 <= self.danger_lateral_smoothness_scale <= 1.0:
            raise ValueError(
                "danger_lateral_smoothness_scale must be within [0, 1]")
        if not 0.0 <= self.planner_collision_trigger <= 1.0:
            raise ValueError("planner collision trigger must be within [0, 1]")
        if self.planner_clearance_trigger_m <= 0.0:
            raise ValueError("planner clearance trigger must be positive")
        if not (
            self.planner_collision_trigger
            <= self.planner_emergency_collision_trigger
            <= 1.0
        ):
            raise ValueError(
                "planner emergency collision trigger must be no lower than "
                "the caution trigger and within [0, 1]")
        if not (
            0.0 < self.planner_emergency_clearance_m
            <= self.planner_clearance_trigger_m
        ):
            raise ValueError(
                "planner emergency clearance must be positive and no higher "
                "than the caution clearance")
        if not -1.0 <= self.planner_emergency_forward_action <= 1.0:
            raise ValueError(
                "planner emergency forward action must be within [-1, 1]")
        if not (
            0.0 < self.planner_observed_emergency_clearance_m
            <= self.planner_observed_caution_clearance_m
        ):
            raise ValueError(
                "observed emergency clearance must be positive and no higher "
                "than observed caution clearance")
        if not (
            0.0 <= self.planner_emergency_escape_min_action
            <= self.planner_emergency_escape_action
            <= 1.0
        ):
            raise ValueError(
                "planner emergency escape actions must be ordered within "
                "[0, 1]")
        if not (
            0.0 < self.planner_observed_forced_escape_clearance_m
            <= self.planner_observed_emergency_clearance_m
        ):
            raise ValueError(
                "forced observed escape clearance must be positive and no "
                "higher than observed emergency clearance")
        if not (
            0.0 < self.planner_observed_critical_clearance_m
            <= self.planner_observed_hard_stop_clearance_m
            <= self.planner_observed_forced_escape_clearance_m
        ):
            raise ValueError(
                "observed critical/hard-stop clearances must be positive, "
                "ordered, and no higher than forced escape clearance")
        if not 0.0 <= self.planner_critical_repulsion_blend <= 1.0:
            raise ValueError(
                "planner critical repulsion blend must be within [0, 1]")
        if self.planner_min_score_improvement < 0.0:
            raise ValueError("planner score improvement must be non-negative")
        if any(value <= 0 for value in self.overshoot_horizons):
            raise ValueError("overshoot horizons must be positive")
        if self.overshoot_starts_per_sequence <= 0:
            raise ValueError("overshoot_starts_per_sequence must be positive")
        self.slow_value = copy.deepcopy(value)
        self.slow_safety_value = copy.deepcopy(self.safety_value)
        self.return_ema = networks.ReturnEMA(device=dynamics.ego_rssm._device)
        self.safety_return_ema = networks.ReturnEMA(
            device=dynamics.ego_rssm._device)
        # Appended after all legacy modules so v6.8 parameter ordering remains
        # a strict prefix during the explicit v4->v5 optimizer migration.
        self.task_geometry_adapter = nn.Linear(
            TASK_GEOMETRY_DIM, int(self.rssm.joint_feat_size), bias=False)
        nn.init.zeros_(self.task_geometry_adapter.weight)
        self.task_memory_adapter = nn.Linear(
            TASK_MEMORY_DIM, int(self.rssm.joint_feat_size), bias=False)
        nn.init.zeros_(self.task_memory_adapter.weight)
        self.register_buffer("safety_lambda", torch.tensor(1.0))
        self.register_buffer("safety_cost_ewma", torch.tensor(0.20))
        self.register_buffer(
            "safety_lambda_update_count", torch.zeros((), dtype=torch.long))
        for parameter in self.slow_value.parameters():
            parameter.requires_grad_(False)
        for parameter in self.slow_safety_value.parameters():
            parameter.requires_grad_(False)

    def decision_feature(
        self,
        joint_feature: torch.Tensor,
        human_feature: torch.Tensor,
        human_root: torch.Tensor,
        human_quality: torch.Tensor,
        human_mask: torch.Tensor,
        *,
        human_joint_clearance: torch.Tensor | None = None,
        human_joints_body: torch.Tensor | None = None,
        human_joint_velocity_body: torch.Tensor | None = None,
        human_joint_mask: torch.Tensor | None = None,
        human_presence: torch.Tensor | None = None,
        detach_geometry_parameters: bool = False,
    ) -> torch.Tensor:
        """Fuse explicit Human geometry into the fixed-size decision state."""
        if (
            self.transition_event is None
            or not self.transition_event.explicit_human_geometry
        ):
            return joint_feature
        geometry = self.transition_event.human_geometry_pool(
            human_feature, human_root, human_quality, human_mask,
            human_joint_clearance=human_joint_clearance,
            human_joints_body=human_joints_body,
            human_joint_velocity_body=human_joint_velocity_body,
            human_joint_mask=human_joint_mask,
            human_presence=human_presence,
            detach_parameters=detach_geometry_parameters,
        )["human_geometry_pool"]
        if geometry.shape != joint_feature.shape:
            raise ValueError("Human geometry and joint decision features differ")
        return self._merge_decision_geometry(joint_feature, geometry)

    def actor_ego_token(
        self,
        ego_state: torch.Tensor,
        latent_ego_token: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact Ego token used by both training and deployment.

        The RSSM Ego posterior is intentionally Human-coupled because this is
        useful for prediction.  It is therefore not a Human-independent input
        for the Actor's goal branch.  Ego14 is directly observed online and is
        advanced by the differentiable analytic dynamics in imagination, so
        reusing the already-trained Ego encoder removes that hidden Human path
        without introducing an action target or breaking dynamics-gradient.
        """
        if not self.actor_authoritative_ego_token_enabled:
            return latent_ego_token
        token = self.encoder.ego_encoder(ego_state.float())
        if token.shape != latent_ego_token.shape:
            raise ValueError(
                "authoritative Actor Ego token and latent Ego token differ: "
                f"{tuple(token.shape)} != {tuple(latent_ego_token.shape)}")
        return token.to(latent_ego_token)

    def actor_ego_task_token(
        self,
        ego_state: torch.Tensor,
        latent_ego_token: torch.Tensor,
        task_geometry: torch.Tensor | None,
        task_memory: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the Actor Ego token with an injective physical-state prefix.

        The prefix order is ``Ego14, TaskGeometry8, TaskMemory9``.  Each block
        has its own fixed scale and is copied verbatim after normalization;
        no two blocks are added together.  The remaining coordinates retain
        the learned authoritative Ego embedding.
        """
        token = self.actor_ego_token(ego_state, latent_ego_token)
        if not self.actor_direct_ego_task_state_enabled:
            return token
        if task_geometry is None or task_memory is None:
            raise ValueError(
                "direct complete Actor state requires task geometry and memory")
        if ego_state.shape[-1] != 14:
            raise ValueError("direct complete Actor state requires Ego14")
        if task_geometry.shape != ego_state.shape[:-1] + (TASK_GEOMETRY_DIM,):
            raise ValueError("task geometry does not match Actor Ego rows")
        if task_memory.shape != ego_state.shape[:-1] + (TASK_MEMORY_DIM,):
            raise ValueError("task memory does not match Actor Ego rows")
        metric_scale = getattr(self.encoder.ego_encoder, "metric_scale", None)
        if metric_scale is None:
            raise RuntimeError(
                "direct complete Actor state requires metric Ego scaling")
        exact = torch.cat((
            (ego_state.float() / metric_scale.to(ego_state)).clamp(-5.0, 5.0),
            task_geometry.float().clamp(0.0, 1.0),
            (task_memory.float() / task_memory.new_tensor(
                TASK_MEMORY_SCALE)).clamp(-5.0, 5.0),
        ), -1).to(token)
        if exact.shape[-1] > token.shape[-1]:
            raise RuntimeError(
                "direct Ego/task prefix exceeds the Actor token width")
        return torch.cat((exact, token[..., exact.shape[-1]:]), -1)

    @staticmethod
    def deployable_human_collision_geometry(
        joints_body: torch.Tensor,
        joint_velocity_body: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the collision topology available to the deployed policy."""
        return coco12_collision_spheres(
            joints_body, joint_mask, joint_velocity_body)

    @classmethod
    def deployable_human_clearance(
        cls,
        joints_body: torch.Tensor,
        joint_velocity_body: torch.Tensor,
        joint_mask: torch.Tensor,
        *,
        per_human: bool = True,
    ) -> torch.Tensor:
        """Signed UAV-to-Isaac-sphere gap from deployable COCO12 state."""
        spheres, _, sphere_mask, radii = (
            cls.deployable_human_collision_geometry(
                joints_body, joint_velocity_body, joint_mask))
        return analytic_joint_signed_surface_gap(
            spheres,
            sphere_mask,
            drone_radius_m=DEPLOYABLE_DRONE_COLLISION_RADIUS_M,
            joint_radii_m=radii,
            fallback_gap_m=6.0,
            per_human=per_human,
        )

    def _merge_decision_geometry(
        self,
        joint_feature: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve the explicit physical prefix in the fixed-width state.

        Adding two unconstrained tokens let the arbitrary first coordinates of
        ``joint_feature`` overwrite the signed metric geometry that the value
        function needs.  The learned suffix remains a residual fusion, while
        the reserved prefix is copied exactly from the Human geometry token.
        """
        if geometry.shape != joint_feature.shape:
            raise ValueError("Human geometry and joint decision features differ")
        fused = joint_feature + geometry
        physical_dim = int(getattr(
            self.transition_event, "geometry_physical_dim", 0))
        if physical_dim <= 0:
            return fused
        return torch.cat((
            geometry[..., :physical_dim],
            fused[..., physical_dim:],
        ), -1)

    def _augment_task_geometry(
        self,
        joint_feature: torch.Tensor,
        actor_feature: torch.Tensor,
        task_geometry: torch.Tensor | None,
        task_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Inject deployable task geometry and Markov memory into decisions."""
        features: list[torch.Tensor] = []
        if self.task_geometry_enabled and task_geometry is not None:
            geometry = self.task_geometry_adapter(
                task_geometry.float()).to(joint_feature)
            if self.direct_task_geometry_enabled:
                geometry = geometry + task_geometry_feature(
                    task_geometry.float(), joint_feature.shape[-1]
                ).to(joint_feature)
            features.append(geometry)
        if self.task_memory_enabled and task_memory is not None:
            memory = self.task_memory_adapter(
                task_memory.float()).to(joint_feature)
            if self.direct_task_memory_enabled:
                memory = memory + task_memory_feature(
                    task_memory.float(), joint_feature.shape[-1]
                ).to(joint_feature)
            features.append(memory)
        if not features:
            return joint_feature, actor_feature, None
        geometry = torch.stack(features, 0).sum(0)
        joint_feature = joint_feature + geometry
        token_dim = joint_feature.shape[-1]
        if actor_feature.shape[-1] == 4 * token_dim:
            goal, ego, _joint, human = actor_feature.split(token_dim, dim=-1)
            actor_feature = torch.cat((
                goal, ego + geometry, joint_feature, human,
            ), -1)
        else:
            if actor_feature.shape[-1] != geometry.shape[-1]:
                raise ValueError(
                    "non-tokenized Actor task geometry width mismatch")
            actor_feature = actor_feature + geometry.to(actor_feature)
        return joint_feature, actor_feature, geometry

    @staticmethod
    def _embeddings(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # New encoders should expose per-person tokens. A pooled human_embed is
        # deliberately rejected because it cannot drive per-slot Human RSSM.
        human = encoded.get("human_tokens")
        if human is None:
            human = encoded.get("human_embed_slots")
        if human is None:
            raise KeyError("factorized encoder must return human_tokens [B,T,N,E]")
        return {
            "ego": encoded["ego_embed"],
            "human": human,
        }

    @staticmethod
    def _mean_nll(
        head: nn.Module, feat: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = head(feat)
        if not hasattr(dist, "log_prob"):
            raise TypeError("Reward/Continue heads must return distribution-like objects")
        loss = -dist.log_prob(target.float())
        if mask is None:
            return loss.mean()
        if mask.shape[:2] != loss.shape[:2]:
            raise ValueError("sequence mask and prediction leading axes differ")
        weight = mask.bool().reshape(*mask.shape[:2], -1).all(-1).to(loss)
        while weight.ndim < loss.ndim:
            weight = weight.unsqueeze(-1)
        return (loss * weight).sum() / weight.expand_as(loss).sum().clamp_min(1)

    def posterior(
        self,
        batch: dict[str, torch.Tensor],
        initial: FactorizedState | None = None,
        *,
        sample_state: bool = True,
    ) -> tuple[FactorizedState, dict[str, Any], dict[str, torch.Tensor]]:
        encoded = self.encoder(batch)
        embeds = self._embeddings(encoded)
        human_mask = batch.get("human_mask", encoded.get("human_token_mask"))
        if human_mask is None:
            raise KeyError("batch/encoder must provide human_mask or human_token_mask")
        human_is_first = batch.get("human_is_first")
        if human_is_first is None:
            raise KeyError("batch must provide human_is_first for slot-safe posterior updates")
        batch_size, _, max_people = human_mask.shape
        if "goal" not in batch:
            raise KeyError("batch must provide body-relative goal features")
        if initial is None:
            initial = self.rssm.initial(
                batch_size, max_people, sample_state=sample_state)
        state, aux = self.rssm.observe(
            embeds,
            batch["action"],
            initial,
            batch["is_first"],
            human_mask,
            human_is_first,
            goal=batch["goal"],
            sample_state=sample_state,
        )
        joint, actor_feature, task_feature = self._augment_task_geometry(
            aux["joint_feat"], aux["actor_feat"], batch.get("task_geometry"),
            batch.get("task_memory"))
        aux["joint_feat"] = joint
        aux["actor_feat"] = actor_feature
        if task_feature is not None:
            aux["task_geometry_feature"] = task_feature
        return state, aux, encoded

    def world_model_loss(
        self,
        batch: dict[str, torch.Tensor],
        initial: FactorizedState | None = None,
    ) -> tuple[dict[str, torch.Tensor], FactorizedState, dict[str, Any]]:
        profile_runtime = os.environ.get(
            "PURE_DREAMER_PROFILE_UPDATE_TIMES", "0") == "1"
        profile_previous = time.perf_counter()
        profile_metrics: dict[str, torch.Tensor] = {}

        def profile_mark(name: str) -> None:
            nonlocal profile_previous
            if not profile_runtime:
                return
            device = next(iter(self.parameters())).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            profile_metrics[f"profile_{name}_ms"] = batch[
                "action"].new_tensor((now - profile_previous) * 1000.0)
            profile_previous = now

        states, aux, encoded = self.posterior(batch, initial)
        profile_mark("posterior")
        losses = self.rssm.kl_loss(
            aux["post_logits"], aux["prior_logits"], aux["human_mask"],
            self.kl_free, sequence_valid=batch.get("sequence_valid"),
        )
        joint = aux["joint_feat"]
        losses["rew"] = self._mean_nll(
            self.reward, joint, batch["reward"], batch.get("sequence_valid"))
        if "reward_components" in batch:
            losses["rew_components"] = self.reward_components.loss(
                joint, batch["reward_components"], batch.get("sequence_valid")
            )
        continuation = 1.0 - batch["is_terminal"].float()
        losses["con"] = self._mean_nll(
            self.cont, joint, continuation, batch.get("sequence_valid"))
        profile_mark("basic_heads")
        transition_losses, transition_metrics = (
            self.transition_event_prediction_loss(joint, batch, states=states))
        profile_mark("transition_event")
        losses.update(transition_losses)
        risk_losses, risk_metrics = self.action_risk_prediction_loss(
            joint, batch)
        profile_mark("action_risk")
        losses.update(risk_losses)
        human_overshoot_losses, human_overshoot_metrics = (
            self.multistep_overshooting_loss(states, batch))
        profile_mark("human_overshoot")
        losses.update(human_overshoot_losses)
        return losses, states, {
            **aux,
            "encoded": encoded,
            "risk_metrics": risk_metrics,
            "transition_metrics": transition_metrics,
            "overshoot_metrics": {
                **human_overshoot_metrics,
                **profile_metrics,
            },
        }

    def transition_event_prediction_loss(
        self,
        joint_feat: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        states: FactorizedState | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Train the source-aligned v6.3 one-step safety contract."""
        if self.transition_event is None or self.next_human_clearance is None:
            zero = joint_feat.sum() * 0.0
            return {}, {"label_available": zero.detach()}
        feature = joint_feat[:, :-1]
        action = batch["action"][:, 1:].float().clamp(-1.0, 1.0)
        event = self.transition_event(feature, action)
        explicit_context_available = (
            bool(self.transition_event.explicit_human_geometry)
            and states is not None
            and all(key in batch for key in (
                "human_root", "human_observation_quality",
                "human_mask", "joint_mask", "skeleton")))
        geometry_event = None
        if explicit_context_available:
            branch = self.rssm.get_branch_feats(states)
            human_root = batch["human_root"][:, :-1].float()
            human_quality = batch[
                "human_observation_quality"][:, :-1].float()
            human_mask = batch["human_mask"][:, :-1].bool()
            human_joints_body = batch[
                "skeleton"][:, :-1, ..., :3].float()
            human_joint_velocity_body = batch[
                "skeleton"][:, :-1, ..., 3:6].float()
            human_joint_mask = batch["joint_mask"][:, :-1].bool()
            human_joint_clearance = self.deployable_human_clearance(
                human_joints_body,
                human_joint_velocity_body,
                human_joint_mask,
                per_human=True,
            )
            geometry_event = self.transition_event.forward_human_hazard(
                feature,
                action,
                ego_feature=branch["ego"][:, :-1],
                human_feature=branch["human"][:, :-1],
                human_root=human_root,
                human_quality=human_quality,
                human_mask=human_mask,
                human_joint_clearance=human_joint_clearance,
                human_joints_body=human_joints_body,
                human_joint_velocity_body=human_joint_velocity_body,
                human_joint_mask=human_joint_mask,
            )
            clearance_prediction = self.next_human_clearance(
                feature,
                action,
                ego_feature=branch["ego"][:, :-1],
                human_feature=branch["human"][:, :-1],
                human_root=human_root,
                human_quality=human_quality,
                human_mask=human_mask,
            )
        else:
            clearance_prediction = self.next_human_clearance(feature, action)
        action_valid = batch.get("action_valid")
        if action_valid is None:
            action_valid = torch.ones_like(action[..., :1], dtype=torch.bool)
        else:
            action_valid = action_valid[:, 1:].bool().reshape(
                *action.shape[:-1], -1).all(-1, keepdim=True)
        source_live = ~batch["is_last"][:, :-1].bool().reshape(
            *action.shape[:-1], -1).any(-1, keepdim=True)
        transition_exposure_fraction = action.new_ones(
            (*action.shape[:-1], 1))
        if "dt_s" in batch:
            transition_dt = batch["dt_s"][:, 1:].float()
            if transition_dt.shape != transition_exposure_fraction.shape:
                raise ValueError(
                    "destination dt_s must match source-aligned transitions")
            if not torch.isfinite(transition_dt).all() \
                    or bool((transition_dt < 0.0).any()):
                raise ValueError("transition dt_s must be finite/non-negative")
            transition_exposure_fraction = (
                transition_dt / float(
                    self.lateral_candidate_step_duration_s))

        target = batch.get("transition_event_target")
        target_valid = batch.get("transition_event_valid")
        next_clearance = batch.get("next_human_clearance_m")
        next_clearance_valid = batch.get("next_human_clearance_valid")
        if (
            target is None or target_valid is None
            or next_clearance is None or next_clearance_valid is None
        ):
            zero = (
                event["logits"].sum() + clearance_prediction.sum()) * 0.0
            return {
                "transition_event": zero,
                "transition_non_human_event": zero,
                "next_clearance": zero,
                "next_clearance_false_safe": zero,
                "next_clearance_false_danger": zero,
            }, {"label_available": zero.detach()}

        valid = target_valid[:, :-1].bool() & action_valid & source_live
        if "sequence_valid" in batch:
            valid &= (
                batch["sequence_valid"][:, :-1].bool()
                & batch["sequence_valid"][:, 1:].bool())
        if "dt_s" in batch and bool((
            batch["dt_s"][:, 1:].float().le(0.0) & valid
        ).any()):
            raise ValueError("valid Event transitions require positive dt_s")
        target = target[:, :-1].long().squeeze(-1)
        cross_entropy = F.cross_entropy(
            event["logits"].reshape(-1, len(TRANSITION_EVENT_KEYS)),
            target.reshape(-1),
            reduction="none",
        ).reshape_as(target)[..., None]
        event_loss = self._masked_mean(cross_entropy, valid)
        if geometry_event is None:
            geometry_human_loss = event_loss.detach() * 0.0
            non_human_event_loss = event_loss.detach() * 0.0
            non_human_probability = event["probability"][..., (0, 2)]
            non_human_target = target.new_zeros(target.shape)
            non_human_valid = torch.zeros_like(valid)
        else:
            human_target = target.eq(HUMAN_COLLISION_INDEX).float()[..., None]
            # A per-person noisy-OR cannot learn correct attribution from an
            # any-person label alone: on a crowded positive row every slot
            # otherwise receives the same aggregate gradient. Compact-v3
            # carries the recorder's exact PhysX contact-person identity, so
            # the component likelihood below uses that terminal label. Swept
            # sphere gaps remain metric supervision/audit only: treating their
            # conservative contactOffset proxy as a factual Event label creates
            # positives on transitions where the recorder did not terminate.
            privileged_slot_fields = {
                "human_gt_id", "human_gt_match_valid", "human_mask",
                "priv_human_id", "priv_human_mask",
                "priv_collision_joints_episode",
                "priv_collision_joint_valid",
                "privileged_geometry_available",
                "counterfactual_collision_surface_radii_m",
                "counterfactual_collision_contact_offset_m",
                "priv_collision_human_id",
            }
            slot_geometry_human_loss = event_loss.detach() * 0.0
            unobserved_geometry_human_loss = event_loss.detach() * 0.0
            slot_target = torch.zeros_like(human_mask, dtype=torch.bool)
            slot_valid = torch.zeros_like(human_mask, dtype=torch.bool)
            slot_gap = torch.zeros_like(human_mask, dtype=feature.dtype)
            if privileged_slot_fields.issubset(batch):
                slot_target, slot_valid, slot_gap = (
                    self.privileged_per_human_swept_collision_targets(batch))
                slot_valid &= valid.squeeze(-1)[..., None]
                slot_probability = rescale_interval_probability(
                    geometry_event["slot_hazard"],
                    transition_exposure_fraction,
                ).clamp(1.0e-6, 1.0 - 1.0e-6)
                slot_raw = self._probability_binary_cross_entropy(
                    slot_probability, slot_target.to(slot_probability))
                slot_geometry_human_loss = self._masked_mean(
                    slot_raw, slot_valid)
            geometry_human_probability = rescale_interval_probability(
                geometry_event["human_collision_probability"],
                transition_exposure_fraction,
            ).clamp(1.0e-6, 1.0 - 1.0e-6)
            geometry_human_raw = self._probability_binary_cross_entropy(
                geometry_human_probability, human_target)
            aggregate_human_valid = valid
            if privileged_slot_fields.issubset(batch):
                attributable_positive = (
                    slot_target & slot_valid).any(-1, keepdim=True)
                unobserved_target = (
                    human_target.bool() & ~attributable_positive)
                unobserved_probability = rescale_interval_probability(
                    geometry_event[
                        "unobserved_human_collision_probability"],
                    transition_exposure_fraction,
                ).clamp(1.0e-6, 1.0 - 1.0e-6)
                unobserved_raw = self._probability_binary_cross_entropy(
                    unobserved_probability,
                    unobserved_target.to(unobserved_probability),
                )
                unobserved_geometry_human_loss = self._masked_mean(
                    unobserved_raw, valid)
            else:
                attributable_positive = torch.zeros_like(
                    human_target, dtype=torch.bool)
                unobserved_target = torch.zeros_like(
                    human_target, dtype=torch.bool)
            aggregate_geometry_human_loss = self._masked_mean(
                geometry_human_raw, aggregate_human_valid)
            if privileged_slot_fields.issubset(batch):
                # This is the natural joint negative log-likelihood of the
                # independent cause hazards used by the noisy-OR forward model:
                # sum visible-slot Bernoulli NLLs plus the hidden-cause NLL,
                # averaged over physical transitions. The aggregate any-Human
                # BCE is redundant with those causes and is diagnostic only;
                # optimizing it again would double-count every event and give
                # absolute calibration an arbitrary 1/3 weighting.
                slot_nll_per_transition = (
                    slot_raw * slot_valid.to(slot_raw)
                ).sum(-1, keepdim=True)
                slot_geometry_human_loss = self._masked_mean(
                    slot_nll_per_transition, valid)
                geometry_human_loss = (
                    slot_geometry_human_loss
                    + unobserved_geometry_human_loss)
            else:
                geometry_human_loss = aggregate_geometry_human_loss
            non_goal_event = self.transition_event.forward_non_goal(
                feature,
                action,
                ego_feature=branch["ego"][:, :-1],
                human_feature=branch["human"][:, :-1],
                human_root=human_root,
                human_quality=human_quality,
                human_mask=human_mask,
                human_joint_clearance=human_joint_clearance,
                human_joints_body=human_joints_body,
                human_joint_velocity_body=human_joint_velocity_body,
                human_joint_mask=human_joint_mask,
            )
            non_human_probability = non_goal_event[
                "global_non_human_probability"]
            # Residual global classes are conditional on no Human collision:
            # continue or static collision. Goal, OOB, crash and stuck are
            # computed from explicit task geometry/memory and therefore map
            # to residual continue instead of being guessed a second time.
            destination_code = batch.get("termination_code")
            if destination_code is None:
                non_human_target = target.new_zeros(target.shape)
                non_human_valid = torch.zeros_like(valid)
            else:
                code = destination_code[:, 1:].long().squeeze(-1)
                non_human_target = torch.zeros_like(code)
                non_human_target = torch.where(
                    code.eq(3), torch.ones_like(code), non_human_target)
                if not self.transition_event.analytic_task_memory_events:
                    non_human_target = torch.where(
                        code.eq(5), torch.full_like(code, 2),
                        non_human_target)
                    non_human_target = torch.where(
                        code.eq(6), torch.full_like(code, 3),
                        non_human_target)
                non_human_valid = valid & code.ne(2)[..., None]
            non_human_class_count = (
                2 if self.transition_event.analytic_task_memory_events else 4)
            if non_human_class_count == 2:
                # This is a conditional static-contact probability over one
                # nominal 100 ms interval. Force-written terminal rows can be
                # much shorter, so use the same continuous-hazard semantics
                # as Human Event instead of treating 16 ms as a full trial.
                static_probability = rescale_interval_probability(
                    non_human_probability[..., 1:2],
                    transition_exposure_fraction,
                ).clamp(1.0e-6, 1.0 - 1.0e-6)
                non_human_raw = self._probability_binary_cross_entropy(
                    static_probability,
                    non_human_target.eq(1)[..., None].to(
                        static_probability),
                )
                non_human_event_loss = self._masked_mean(
                    non_human_raw, non_human_valid)
                non_human_probability = torch.cat((
                    1.0 - static_probability, static_probability,
                ), -1)
            else:
                non_human_ce = F.cross_entropy(
                    non_goal_event["global_non_human_logits"].reshape(
                        -1, non_human_class_count),
                    non_human_target.reshape(-1),
                    reduction="none",
                ).reshape_as(non_human_target)[..., None]
                non_human_event_loss = self._masked_mean(
                    non_human_ce, non_human_valid)

        clearance_valid = (
            next_clearance_valid[:, :-1].bool() & action_valid & source_live)
        if "sequence_valid" in batch:
            clearance_valid &= (
                batch["sequence_valid"][:, :-1].bool()
                & batch["sequence_valid"][:, 1:].bool())
        clearance_target = next_clearance[:, :-1].float()
        clearance = self._balanced_clearance_supervision(
            clearance_prediction, clearance_target, clearance_valid)
        legacy_probability = event["probability"]
        if geometry_event is not None and non_human_probability.shape[-1] == 2:
            # Primary Event metrics must match the Actor's learned stochastic
            # cause: Human collision versus no Human collision. Static safety
            # is now an analytic conservative-AABB terminal, while the
            # factual physical-static classifier remains a separately named
            # auxiliary diagnostic. Goal/boundary/crash/stuck are likewise exact
            # task geometry/memory events, not categorical guesses.
            no_human = 1.0 - geometry_human_probability
            probability = torch.cat((
                no_human,
                geometry_human_probability,
            ), -1)
            metric_target = target.eq(HUMAN_COLLISION_INDEX).long()
            metric_keys = TRANSITION_EVENT_KEYS[:2]
        else:
            probability = legacy_probability
            metric_target = target
            metric_keys = TRANSITION_EVENT_KEYS
        prediction = probability.argmax(dim=-1)
        human_target = (target == HUMAN_COLLISION_INDEX).float()[..., None]
        human_probability = (
            event["human_collision_probability"]
            if geometry_event is None else
            geometry_human_probability
        )
        brier = (human_probability - human_target).square()
        human_prediction = human_probability.ge(0.5)
        human_positive = target.eq(HUMAN_COLLISION_INDEX)[..., None]
        human_support = (human_positive & valid).sum()
        human_predicted_positive = (human_prediction & valid).sum()
        human_true_positive = (
            human_prediction & human_positive & valid).sum()
        valid_human_scores = human_probability[valid]
        valid_human_labels = human_positive[valid].float()
        if valid_human_scores.numel() and bool(human_support > 0):
            order = valid_human_scores.argsort(descending=True)
            sorted_labels = valid_human_labels[order]
            precision_curve = sorted_labels.cumsum(0) / torch.arange(
                1, sorted_labels.numel() + 1,
                device=sorted_labels.device,
                dtype=sorted_labels.dtype,
            )
            human_average_precision = (
                precision_curve * sorted_labels).sum() / human_support
        else:
            human_average_precision = probability.sum() * 0.0
        residual_valid_flat = non_human_valid.squeeze(-1)
        residual_static_label = non_human_target.eq(1)
        residual_static_probability = non_human_probability[..., 1]
        residual_static_prediction = non_human_probability.argmax(-1).eq(1)
        residual_static_support = (
            residual_static_label & residual_valid_flat).sum()
        residual_static_predicted_positive = (
            residual_static_prediction & residual_valid_flat).sum()
        residual_static_true_positive = (
            residual_static_prediction
            & residual_static_label
            & residual_valid_flat
        ).sum()
        residual_static_scores = residual_static_probability[
            residual_valid_flat]
        residual_static_labels = residual_static_label[
            residual_valid_flat].float()
        if residual_static_scores.numel() and bool(
            residual_static_support > 0
        ):
            order = residual_static_scores.argsort(descending=True)
            sorted_labels = residual_static_labels[order]
            precision_curve = sorted_labels.cumsum(0) / torch.arange(
                1, sorted_labels.numel() + 1,
                device=sorted_labels.device,
                dtype=sorted_labels.dtype,
            )
            residual_static_ap = (
                precision_curve * sorted_labels
            ).sum() / residual_static_support
        else:
            residual_static_ap = probability.sum() * 0.0
        metrics = {
            "label_available": valid.float().mean(),
            "event_accuracy": self._masked_mean(
                prediction.eq(metric_target).float()[..., None], valid),
            "event_probability_sum_error": (
                probability.sum(-1) - 1.0).abs().mean(),
            "human_event_ratio": self._masked_mean(human_target, valid),
            "human_probability_mean": self._masked_mean(
                human_probability, valid),
            "human_nominal_probability_mean": self._masked_mean(
                (
                    geometry_event["human_collision_probability"]
                    if geometry_event is not None else
                    event["human_collision_probability"]
                ),
                valid,
            ),
            "transition_exposure_fraction_mean": self._masked_mean(
                transition_exposure_fraction, valid),
            "human_brier": self._masked_mean(brier, valid),
            "geometry_human_recall": (
                human_true_positive / human_support.clamp_min(1)),
            "geometry_human_precision": (
                human_true_positive
                / human_predicted_positive.clamp_min(1)),
            "geometry_human_ap": human_average_precision,
            "geometry_human_support": human_support,
            "geometry_human_true_positive": human_true_positive,
            "geometry_human_predicted_positive": human_predicted_positive,
            "geometry_human_slot_loss": (
                slot_geometry_human_loss.detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "geometry_human_unobserved_loss": (
                unobserved_geometry_human_loss.detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "geometry_human_unobserved_positive_count": (
                (unobserved_target & valid).sum().detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "geometry_human_slot_valid_count": (
                slot_valid.sum().detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "geometry_human_slot_positive_count": (
                (slot_target & slot_valid).sum().detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "geometry_human_slot_gap_m": (
                self._masked_mean(slot_gap, slot_valid).detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "geometry_human_unattributable_positive_count": (
                (
                    human_target.bool()
                    & valid
                    & ~(slot_target & slot_valid).any(-1, keepdim=True)
                ).sum().detach()
                if geometry_event is not None else event_loss.detach() * 0.0),
            "non_human_event_accuracy": self._masked_mean(
                non_human_probability.argmax(-1).eq(
                    non_human_target).float()[..., None],
                non_human_valid,
            ),
            "non_human_event_valid_ratio": non_human_valid.float().mean(),
            "residual_static_support": residual_static_support,
            "residual_static_recall": (
                residual_static_true_positive
                / residual_static_support.clamp_min(1)),
            "residual_static_precision": (
                residual_static_true_positive
                / residual_static_predicted_positive.clamp_min(1)),
            "residual_static_ap": residual_static_ap,
            "residual_static_brier": self._masked_mean(
                (
                    residual_static_probability
                    - residual_static_label.float()
                ).square()[..., None],
                non_human_valid,
            ),
            "next_clearance_mae_m": self._masked_mean(
                clearance["absolute_error"], clearance_valid),
            "next_clearance_false_safe_ratio": (
                clearance["false_safe_mask"].sum()
                / clearance["danger_mask"].sum().clamp_min(1)),
            "next_clearance_false_danger_ratio": (
                clearance["false_danger_mask"].sum()
                / clearance["calibration_safe_mask"].sum().clamp_min(1)),
        }
        for index, key in enumerate(metric_keys):
            class_mask = valid & metric_target.eq(index)[..., None]
            class_target = metric_target.eq(index)
            class_probability = probability[..., index]
            metrics[f"event_ratio/{key}"] = self._masked_mean(
                metric_target.eq(index).float()[..., None], valid)
            metrics[f"event_recall/{key}"] = self._masked_mean(
                prediction.eq(index).float()[..., None], class_mask)
            metrics[f"event_precision/{key}"] = self._masked_mean(
                class_target.float()[..., None],
                valid & prediction.eq(index)[..., None])
            metrics[f"event_brier/{key}"] = self._masked_mean(
                (class_probability - class_target.float()).square()[..., None],
                valid)
            valid_flat = valid.squeeze(-1)
            scores = class_probability[valid_flat]
            labels = class_target[valid_flat].float()
            positive_count = labels.sum()
            if scores.numel() and bool(positive_count > 0):
                order = scores.argsort(descending=True)
                sorted_labels = labels[order]
                precision_curve = sorted_labels.cumsum(0) / torch.arange(
                    1, sorted_labels.numel() + 1,
                    device=sorted_labels.device,
                    dtype=sorted_labels.dtype)
                average_precision = (
                    precision_curve * sorted_labels).sum() / positive_count
            else:
                average_precision = probability.sum() * 0.0
            metrics[f"event_ap/{key}"] = average_precision
            metrics[f"event_support/{key}"] = positive_count
            metrics[f"event_true_positive/{key}"] = (
                (prediction.eq(index) & class_target & valid_flat).sum())
            metrics[f"event_predicted_positive/{key}"] = (
                (prediction.eq(index) & valid_flat).sum())
        return {
            "transition_event": event_loss,
            "transition_human_geometry": geometry_human_loss,
            "transition_non_human_event": non_human_event_loss,
            "next_clearance": clearance["regression_loss"],
            "next_clearance_false_safe": clearance["false_safe_loss"],
            "next_clearance_false_danger": clearance["false_danger_loss"],
        }, metrics

    @staticmethod
    def privileged_per_human_swept_collision_targets(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source-slot PhysX contact truth for every factual transition.

        Perception slots and privileged simulator slots have unrelated order.
        The dataset's carried GT identity supplies the audited one-to-one
        source correspondence through causal perception holds;
        the same physical person is then aligned independently at both segment
        endpoints before the synchronous swept surface gap is evaluated.
        Privileged tensors are targets only and never enter Actor observations.

        The returned ``slot_target`` is always the recorder's exact terminal
        contact-person label. ``gap`` is returned solely as a metric geometry
        diagnostic. A conservative swept sphere/contactOffset overlap is not a
        factual collision label: the previous implementation produced positive
        slot hazards on non-terminal transitions, directly contradicting the
        aggregate Human Event target.
        """
        required = {
            "human_gt_id", "human_gt_match_valid", "human_mask",
            "priv_human_id", "priv_human_mask",
            "priv_collision_joints_episode", "priv_collision_joint_valid",
            "privileged_geometry_available",
            "counterfactual_collision_surface_radii_m",
            "counterfactual_collision_contact_offset_m",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "per-Human swept collision targets lack " + ", ".join(missing))
        source_id = batch.get(
            "human_gt_identity_id", batch["human_gt_id"])[:, :-1].long()
        source_valid = (
            batch.get(
                "human_gt_identity_valid",
                batch["human_gt_match_valid"],
            )[:, :-1].bool()
            & batch["human_mask"][:, :-1].bool()
            & source_id.ge(0)
        )

        aligned_relative: list[torch.Tensor] = []
        aligned_joint_valid: list[torch.Tensor] = []
        geometry_rows: list[torch.Tensor] = []
        radii_rows: list[torch.Tensor] = []
        offset_rows: list[torch.Tensor] = []
        for endpoint in (slice(None, -1), slice(1, None)):
            gt_id = batch["priv_human_id"][:, endpoint].long()
            gt_mask = batch["priv_human_mask"][:, endpoint].bool()
            joints = batch[
                "priv_collision_joints_episode"][:, endpoint].float()
            joint_valid = batch[
                "priv_collision_joint_valid"][:, endpoint].bool()
            people = min(gt_id.shape[-1], joints.shape[-3])
            gt_id = gt_id[..., :people]
            gt_mask = gt_mask[..., :people]
            joints = joints[..., :people, :, :]
            joint_valid = joint_valid[..., :people, :]
            matched = (
                gt_id[..., None].eq(source_id[..., None, :])
                & gt_mask[..., None]
                & source_valid[..., None, :]
            )
            if bool(matched.sum(-2).gt(1).any()):
                raise RuntimeError(
                    "privileged Human IDs are not unique within a frame")
            aligned = (
                joints[..., :, None, :, :]
                * matched[..., None, None].to(joints)
            ).sum(-4)
            valid = (
                joint_valid[..., :, None, :]
                & matched[..., None]
            ).any(-3)
            ego = batch["ego_state"][:, endpoint, None, None, :3].float()
            aligned_relative.append(aligned - ego)
            aligned_joint_valid.append(valid)
            geometry_rows.append(
                batch["privileged_geometry_available"][:, endpoint].bool()
                .reshape(*source_id.shape[:2], -1).all(-1))
            radii_rows.append(batch[
                "counterfactual_collision_surface_radii_m"][:, endpoint].float())
            offset_rows.append(batch[
                "counterfactual_collision_contact_offset_m"][:, endpoint].float()
                .reshape(*source_id.shape[:2], -1).amax(-1))

        swept_valid = aligned_joint_valid[0] & aligned_joint_valid[1]
        geometry_valid = geometry_rows[0] & geometry_rows[1]
        swept_valid &= geometry_valid[..., None, None]
        radii = torch.maximum(radii_rows[0], radii_rows[1])
        gap = swept_relative_point_signed_gap(
            aligned_relative[0], aligned_relative[1], swept_valid,
            surface_radii_m=radii[..., None, :],
            fallback_gap_m=6.0,
            per_human=True,
        )
        geometry_slot_valid = source_valid & swept_valid.any(-1)
        contact_offset = torch.maximum(
            offset_rows[0], offset_rows[1])[..., None]
        if not bool(
            torch.isfinite(contact_offset).all()
            and contact_offset.ge(0.0).all()
        ):
            raise RuntimeError(
                "per-Human contact offsets must be finite and non-negative")
        slot_target = torch.zeros_like(source_valid)
        slot_valid = geometry_slot_valid
        collision_human_id = batch.get("priv_collision_human_id")
        termination_code = batch.get("termination_code")
        if collision_human_id is not None and termination_code is not None:
            destination_collision_id = collision_human_id[:, 1:].long()
            destination_collision_id = destination_collision_id.reshape(
                *source_id.shape[:2], -1)
            if destination_collision_id.shape[-1] != 1:
                raise ValueError(
                    "priv_collision_human_id must have one ID per frame")
            destination_collision_id = destination_collision_id[..., 0]
            exact_contact_row = destination_collision_id.ge(0)
            destination_human_collision = (
                termination_code[:, 1:].long().reshape(
                    *source_id.shape[:2], -1)
            )
            if destination_human_collision.shape[-1] != 1:
                raise ValueError(
                    "termination_code must have one value per frame")
            destination_human_collision = (
                destination_human_collision[..., 0] == 2)
            exact_contact_slot = (
                source_id.eq(destination_collision_id[..., None])
                & source_valid
            )
            if bool((exact_contact_slot.sum(-1) > 1).any()):
                raise RuntimeError(
                    "one PhysX contact person maps to multiple Actor slots")
            if bool((exact_contact_row & ~destination_human_collision).any()):
                raise RuntimeError(
                    "collision Human ID is present on a non-Human terminal")
            slot_target = exact_contact_slot & destination_human_collision[..., None]
            # On a non-Human transition every Actor-visible slot is an exact
            # negative and needs no privileged identity. On a Human terminal,
            # only identity-resolved slots are valid causes; an unresolved
            # partner is owned by the hidden branch instead of becoming a
            # fabricated visible negative/positive.
            slot_valid = torch.where(
                destination_human_collision[..., None],
                source_valid,
                batch["human_mask"][:, :-1].bool(),
            )
            unknown_contact_id = (
                destination_human_collision & ~exact_contact_row)
            slot_valid = slot_valid & ~unknown_contact_id[..., None]
        return slot_target, slot_valid, gap

    def transition_event_balanced_auxiliary_loss(
        self,
        joint_feat: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        ranking_margin: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Importance-corrected rare-event NLL plus ranking supervision.

        The auxiliary loader selects exactly one transition per sequence and
        samples event classes uniformly. ``event_importance_weight`` restores
        the natural replay prior for NLL calibration; ranking remains an
        explicit rare-event auxiliary and does not define probabilities.
        """
        if self.transition_event is None:
            zero = joint_feat.sum() * 0.0
            return zero, {"nll": zero.detach(), "ranking": zero.detach()}
        feature = joint_feat[:, :-1]
        action = batch["action"][:, 1:].float().clamp(-1.0, 1.0)
        prediction = self.transition_event(feature, action)
        logits = prediction["logits"]
        target = batch["transition_event_target"][:, :-1].long().squeeze(-1)
        valid = (
            batch["transition_event_valid"][:, :-1].bool()
            & batch["event_auxiliary_mask"][:, :-1].bool()
            & ~batch["is_last"][:, :-1].bool()
        ).squeeze(-1)
        action_valid = batch.get("action_valid")
        if action_valid is not None:
            valid = valid & action_valid[:, 1:].bool().squeeze(-1)
        importance = batch["event_importance_weight"].float().reshape(-1, 1)
        importance = importance.expand_as(target)
        ce = F.cross_entropy(
            logits.reshape(-1, len(TRANSITION_EVENT_KEYS)),
            target.reshape(-1), reduction="none").reshape_as(target)
        weighted_valid = importance * valid.float()
        nll = (ce * weighted_valid).sum() / weighted_valid.sum().clamp_min(1e-6)
        # The importance-corrected NLL above deliberately restores the
        # natural (overwhelmingly continue) replay prior for calibration. It
        # cannot by itself teach a usable rare-class decision boundary. The
        # sampler is class-balanced, so this unweighted term is a detached-head
        # discriminative auxiliary; natural replay NLL remains the authority
        # for calibrated probabilities.
        balanced_ce = ce.masked_select(valid).mean() if valid.any() else (
            logits.sum() * 0.0)
        adjustment = batch["event_logit_adjustment"].float()
        if adjustment.ndim != 2 or adjustment.shape[-1] != logits.shape[-1]:
            raise ValueError(
                "event_logit_adjustment must have shape [batch,event_classes]")
        adjusted_logits = logits + adjustment[:, None, :]
        adjusted_ce = F.cross_entropy(
            adjusted_logits.reshape(-1, len(TRANSITION_EVENT_KEYS)),
            target.reshape(-1), reduction="none").reshape_as(target)
        prior_adjusted_ce = (
            adjusted_ce.masked_select(valid).mean() if valid.any()
            else logits.sum() * 0.0)

        selected_logits = logits[valid]
        selected_target = target[valid]
        terminal = selected_target.ne(CONTINUE_INDEX)
        if terminal.any():
            # Macro-average terminal classes. A small batch may contain more
            # human events than static/goal/other even under class-balanced
            # sampling; sample averaging would silently restore that imbalance.
            continue_logit = selected_logits[..., CONTINUE_INDEX]
            per_class_ranking = []
            for event_index in range(1, len(TRANSITION_EVENT_KEYS)):
                class_mask = selected_target.eq(event_index)
                if class_mask.any():
                    per_class_ranking.append(F.softplus(
                        float(ranking_margin)
                        - selected_logits[class_mask, event_index]
                        + continue_logit[class_mask]).mean())
            terminal_ranking = torch.stack(per_class_ranking).mean()
        else:
            terminal_ranking = logits.sum() * 0.0

        human_score = selected_logits[..., HUMAN_COLLISION_INDEX]
        human_positive = selected_target.eq(HUMAN_COLLISION_INDEX)
        human_negative = ~human_positive
        if human_positive.any() and human_negative.any():
            human_ranking = F.softplus(
                float(ranking_margin)
                - human_score[human_positive].mean()
                + human_score[human_negative].mean())
        else:
            human_ranking = logits.sum() * 0.0
        ranking = terminal_ranking + human_ranking

        # Give the event/clearance heads an explicitly action-sensitive target
        # at the same posterior state. Relative body-frame joint velocity plus
        # each candidate UAV velocity provides a deterministic 0.1 s target;
        # the detached feature keeps this auxiliary out of Encoder/RSSM.
        selected = valid.nonzero(as_tuple=False)
        if selected.numel() and "skeleton" in batch and "joint_mask" in batch:
            batch_index, time_index = selected[:, 0], selected[:, 1]
            selected_feature = feature[batch_index, time_index]
            ego = batch["ego_state"][batch_index, time_index].float()
            factual_applied = batch["action"][
                batch_index, time_index + 1].float().clamp(-1.0, 1.0)
            factual_policy = self.policy_from_applied_action(factual_applied)
            candidate_policy = factual_policy[:, None].repeat(1, 4, 1)
            candidate_policy[:, 1, 0] = torch.minimum(
                candidate_policy[:, 1, 0],
                candidate_policy.new_full((candidate_policy.shape[0],), 0.10))
            candidate_policy[:, 2, 1] = 0.35
            candidate_policy[:, 3, 1] = -0.35
            expanded_ego = ego[:, None].expand(-1, 4, -1)
            candidate_applied = self.applied_from_policy_action(
                candidate_policy.reshape(-1, candidate_policy.shape[-1]),
                expanded_ego.reshape(-1, expanded_ego.shape[-1]),
            ).reshape(candidate_policy.shape[0], 4, -1)

            skeleton = batch["skeleton"][batch_index, time_index].float()
            joint_mask = batch["joint_mask"][batch_index, time_index].bool()
            action_scale = candidate_applied.new_tensor((2.15, 2.15, 1.0))
            drone_velocity = candidate_applied[..., :3] * action_scale
            relative_next = (
                skeleton[:, None, ..., :3]
                + (skeleton[:, None, ..., 3:6]
                   - drone_velocity[:, :, None, None, :]) * 0.10)
            distance = torch.linalg.vector_norm(relative_next, dim=-1)
            distance = distance.masked_fill(~joint_mask[:, None], float("inf"))
            geometry_clearance = (
                distance.flatten(2).amin(-1)
                - self.lateral_candidate_surface_radius_m
            ).clamp(-self.lateral_candidate_surface_radius_m, 2.0)
            has_human = joint_mask.flatten(1).any(-1)

            expanded_feature = selected_feature[:, None].expand(-1, 4, -1)
            counterfactual_event = self.transition_event(
                expanded_feature, candidate_applied)
            counterfactual_clearance = self.next_human_clearance(
                expanded_feature, candidate_applied).squeeze(-1)
            counterfactual_hazard = counterfactual_event[
                "human_collision_probability"].squeeze(-1)
            event_logits = counterfactual_event["logits"]
            non_human_logits = torch.cat((
                event_logits[..., :HUMAN_COLLISION_INDEX],
                event_logits[..., HUMAN_COLLISION_INDEX + 1:],
            ), dim=-1)
            counterfactual_human_log_odds = (
                event_logits[..., HUMAN_COLLISION_INDEX]
                - torch.logsumexp(non_human_logits, dim=-1))
            soft_hazard_target = torch.sigmoid(
                (0.10 - geometry_clearance) / 0.10)
            human_rows = has_human[:, None].expand_as(geometry_clearance)
            denominator = human_rows.sum().clamp_min(1)
            counterfactual_bce = (
                F.binary_cross_entropy_with_logits(
                    counterfactual_human_log_odds, soft_hazard_target,
                    reduction="none") * human_rows).sum() / denominator
            counterfactual_clearance_loss = (
                F.smooth_l1_loss(
                    counterfactual_clearance, geometry_clearance,
                    beta=0.10, reduction="none") * human_rows
            ).sum() / denominator

            best = geometry_clearance.argmax(-1)
            worst = geometry_clearance.argmin(-1)
            best_hazard = counterfactual_hazard.gather(
                -1, best[:, None]).squeeze(-1)
            worst_hazard = counterfactual_hazard.gather(
                -1, worst[:, None]).squeeze(-1)
            best_clearance = counterfactual_clearance.gather(
                -1, best[:, None]).squeeze(-1)
            worst_clearance = counterfactual_clearance.gather(
                -1, worst[:, None]).squeeze(-1)
            separable = has_human & (
                geometry_clearance.amax(-1)
                - geometry_clearance.amin(-1) >= 0.02)
            if separable.any():
                action_hazard_ranking = F.softplus(
                    0.01 + best_hazard[separable]
                    - worst_hazard[separable]).mean()
                action_clearance_ranking = F.softplus(
                    0.02 - best_clearance[separable]
                    + worst_clearance[separable]).mean()
            else:
                action_hazard_ranking = logits.sum() * 0.0
                action_clearance_ranking = logits.sum() * 0.0
            counterfactual_action = (
                counterfactual_bce + counterfactual_clearance_loss
                + 0.25 * action_hazard_ranking
                + 0.25 * action_clearance_ranking)
            action_hazard_span = self._masked_mean(
                (counterfactual_hazard.amax(-1)
                 - counterfactual_hazard.amin(-1))[:, None],
                has_human[:, None])
            action_clearance_span = self._masked_mean(
                (counterfactual_clearance.amax(-1)
                 - counterfactual_clearance.amin(-1))[:, None],
                has_human[:, None])
        else:
            counterfactual_action = logits.sum() * 0.0
            counterfactual_bce = counterfactual_action
            counterfactual_clearance_loss = counterfactual_action
            action_hazard_ranking = counterfactual_action
            action_clearance_ranking = counterfactual_action
            action_hazard_span = counterfactual_action.detach()
            action_clearance_span = counterfactual_action.detach()

        # Ordinary balanced_ce remains diagnostic only because it changes the
        # event prior. prior_adjusted_ce uses log(q/p) inside the softmax, so
        # its raw logits retain the natural p(y|x) probability semantics.
        total = (
            nll + 0.50 * prior_adjusted_ce
            + 0.25 * ranking + 0.25 * counterfactual_action)
        return total, {
            "nll": nll.detach(),
            "balanced_ce": balanced_ce.detach(),
            "prior_adjusted_ce": prior_adjusted_ce.detach(),
            "terminal_ranking": terminal_ranking.detach(),
            "human_ranking": human_ranking.detach(),
            "counterfactual_action": counterfactual_action.detach(),
            "counterfactual_hazard_bce": counterfactual_bce.detach(),
            "counterfactual_clearance": counterfactual_clearance_loss.detach(),
            "counterfactual_hazard_ranking": action_hazard_ranking.detach(),
            "counterfactual_clearance_ranking": action_clearance_ranking.detach(),
            "counterfactual_hazard_span": action_hazard_span.detach(),
            "counterfactual_clearance_span_m": action_clearance_span.detach(),
            "valid_count": valid.float().sum().detach(),
            "importance_mean": self._masked_mean(
                importance[..., None], valid[..., None]).detach(),
        }

    @staticmethod
    def _probability_binary_cross_entropy(
        probability: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate probability-space BCE in FP32 under mixed precision.

        These probabilities have already undergone continuous-time hazard
        rescaling, so there is no equivalent single pre-sigmoid logit to pass
        to ``binary_cross_entropy_with_logits``.  PyTorch deliberately rejects
        probability-space BCE inside autocast; keeping this numerically
        sensitive scalar loss in FP32 preserves the model's hazard semantics.
        """
        with torch.autocast(device_type=probability.device.type, enabled=False):
            return F.binary_cross_entropy(
                probability.float(), target.float(), reduction="none")

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(value)
        return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)

    @classmethod
    def _gt_displacement_supervision(
        cls,
        predicted_episode: torch.Tensor,
        predicted_source_episode: torch.Tensor,
        target_episode: torch.Tensor,
        target_source_episode: torch.Tensor,
        valid: torch.Tensor,
        *,
        beta: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return robust displacement loss and per-slot Euclidean error.

        Subtracting each trajectory's own source pelvis makes this label
        invariant to a constant detector-to-simulator pelvis offset.
        """
        predicted_displacement = (
            predicted_episode - predicted_source_episode)
        target_displacement = target_episode - target_source_episode
        difference = predicted_displacement - target_displacement
        raw = F.smooth_l1_loss(
            predicted_displacement, target_displacement,
            beta=float(beta), reduction="none")
        loss = cls._masked_mean(raw, valid[..., None])
        return loss, torch.linalg.vector_norm(difference, dim=-1)

    @classmethod
    def _gt_cv_regret_supervision(
        cls,
        predicted_episode: torch.Tensor,
        predicted_source_episode: torch.Tensor,
        constant_velocity_episode: torch.Tensor,
        target_episode: torch.Tensor,
        target_source_episode: torch.Tensor,
        valid: torch.Tensor,
        *,
        beta: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Penalize learned displacement only when it is worse than CV.

        The earlier regret compared body-frame roots against detector-derived
        targets. A fixed detector-to-simulator pelvis offset cancels poorly
        once occlusion changes that measurement, exactly where collision-tail
        supervision matters most. Compare source-relative episode-frame
        displacement against privileged simulator pelvis motion instead. A
        one-sided Huber tail keeps the useful constant-velocity trust region
        without allowing one long-horizon recurrent outlier to dominate the
        complete World update.  The ``2 * beta`` factor is intentional:
        PyTorch's SmoothL1 divides its quadratic basin by beta.  Rescaling it
        exactly recovers the previous squared-regret value and derivative for
        small errors, while capping the tail derivative at ``2 * beta``.
        """
        predicted_displacement = (
            predicted_episode - predicted_source_episode)
        constant_velocity_displacement = (
            constant_velocity_episode - predicted_source_episode)
        target_displacement = target_episode - target_source_episode
        predicted_error = torch.linalg.vector_norm(
            predicted_displacement - target_displacement, dim=-1)
        constant_velocity_error = torch.linalg.vector_norm(
            constant_velocity_displacement - target_displacement, dim=-1)
        positive_regret = F.relu(
            predicted_error - constant_velocity_error.detach())
        raw = (2.0 * float(beta)) * F.smooth_l1_loss(
            positive_regret,
            torch.zeros_like(positive_regret),
            beta=float(beta),
            reduction="none",
        )
        return (
            cls._masked_mean(raw, valid),
            predicted_error,
            constant_velocity_error,
        )

    def _balanced_clearance_supervision(
        self,
        prediction_m: torch.Tensor,
        target_raw_m: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Balance danger/safe regression and penalize false danger.

        V5 used one globally weighted mean. Danger-prioritized sampling plus a
        6x danger weight made a uniformly small clearance an easy solution.
        Here danger and safe examples first form separate normalized means and
        then receive equal group mass. A second calibration term specifically
        raises predictions that cross the 0.7 m decision boundary on examples
        whose true clearance is at least 1.0 m.
        """
        valid = valid.bool().expand_as(prediction_m)
        target = target_raw_m.float().clamp(
            min=0.0, max=self.risk_clearance_normalization_m)
        error = F.smooth_l1_loss(
            prediction_m / self.risk_clearance_normalization_m,
            target / self.risk_clearance_normalization_m,
            beta=0.05,
            reduction="none",
        )
        danger = valid & (target_raw_m <= self.safe_clearance_m)
        safe = valid & ~danger

        danger_weight = torch.where(
            prediction_m > target,
            error.new_tensor(self.clearance_overestimate_weight),
            error.new_ones(()),
        ) * danger.to(error.dtype)
        safe_weight = torch.where(
            target_raw_m <= 1.5,
            error.new_tensor(self.clearance_near_weight),
            error.new_tensor(self.clearance_far_weight),
        ) * safe.to(error.dtype)
        danger_loss = (error * danger_weight).sum() / danger_weight.sum().clamp_min(1.0)
        safe_loss = (error * safe_weight).sum() / safe_weight.sum().clamp_min(1.0)
        danger_active = (danger_weight.sum() > 0).to(error.dtype)
        safe_active = (safe_weight.sum() > 0).to(error.dtype)
        danger_group_weight = error.new_tensor(self.clearance_danger_weight)
        regression = (
            danger_active * danger_group_weight * danger_loss
            + safe_active * safe_loss
        ) / (
            danger_active * danger_group_weight + safe_active
        ).clamp_min(1.0)

        calibration_safe = valid & (
            target_raw_m >= self.clearance_safe_margin_m)
        false_danger_gap = F.relu(
            self.safe_clearance_m - prediction_m
        ) / self.safe_clearance_m
        false_danger_loss = self._masked_mean(
            false_danger_gap.square(), calibration_safe)
        false_safe_gap = F.relu(
            prediction_m - self.safe_clearance_m
        ) / self.safe_clearance_m
        false_safe_loss = self._masked_mean(
            false_safe_gap.square(), danger)
        absolute_error = (prediction_m - target_raw_m).abs()
        false_safe = danger & (prediction_m > self.safe_clearance_m)
        false_danger = calibration_safe & (
            prediction_m <= self.safe_clearance_m)
        return {
            "regression_loss": regression,
            "false_safe_loss": false_safe_loss,
            "false_danger_loss": false_danger_loss,
            "absolute_error": absolute_error,
            "danger_mask": danger,
            "safe_mask": safe,
            "calibration_safe_mask": calibration_safe,
            "false_safe_mask": false_safe,
            "false_danger_mask": false_danger,
        }

    def action_risk_prediction_loss(
        self,
        joint_feat: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        collision_positive_weight: float | torch.Tensor | None = None,
        normalize_collision_weights: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Supervise the risk estimator on shifted recorded actions.

        Observation ``t`` selected the command stored in row ``t+1``.  Risk
        labels at ``t`` cover the following 2.5 seconds, matching that action.
        """
        feature = joint_feat[:, :-1]
        action = batch["action"][:, 1:].float().clamp(-1.0, 1.0)
        prediction = self.action_risk(feature, action)
        valid = batch.get("action_valid")
        if valid is None:
            valid = torch.ones_like(action[..., :1], dtype=torch.bool)
        else:
            valid = valid[:, 1:].bool().reshape(*action.shape[:-1], -1).all(
                -1, keepdim=True)
        source_live = ~batch["is_last"][:, :-1].bool().reshape(
            *action.shape[:-1], -1).any(-1, keepdim=True)
        valid = valid & source_live

        collision_target = batch.get("future_collision")
        clearance_target = batch.get("future_min_human_clearance_m")
        clearance_valid = batch.get("future_min_human_clearance_valid")
        if collision_target is None or clearance_target is None or clearance_valid is None:
            zero = (
                prediction["collision_logit"].sum()
                + prediction["min_human_clearance_m"].sum()
            ) * 0.0
            return {
                "risk_collision": zero,
                "risk_clearance": zero,
                "risk_false_safe": zero,
                "risk_false_danger": zero,
            }, {
                "label_available": zero.detach(),
                "future_collision_ratio": zero.detach(),
                "clearance_target_mean_m": zero.detach(),
                "clearance_prediction_mean_m": zero.detach(),
                "collision_probability_mean": zero.detach(),
            }

        collision_target = collision_target[:, :-1].float()
        positive_weight = (
            self.risk_collision_positive_weight
            if collision_positive_weight is None
            else collision_positive_weight
        )
        positive_weight = torch.as_tensor(
            positive_weight,
            device=prediction["collision_logit"].device,
            dtype=prediction["collision_logit"].dtype,
        )
        collision_bce = F.binary_cross_entropy_with_logits(
            prediction["collision_logit"], collision_target,
            pos_weight=positive_weight,
            reduction="none",
        )
        if normalize_collision_weights:
            # ``pos_weight`` changes the overall loss magnitude as the class
            # ratio changes.  That made the live risk head's effective
            # learning rate jump by more than 20x on collision-heavy replay.
            # Divide by the active sample-weight mass so reweighting changes
            # the class trade-off, not the optimizer step size.
            sample_weight = 1.0 + (
                positive_weight - 1.0) * collision_target
            collision_loss = (
                collision_bce * valid.to(collision_bce.dtype)
            ).sum() / (
                sample_weight * valid.to(sample_weight.dtype)
            ).sum().clamp_min(1.0)
        else:
            collision_loss = self._masked_mean(collision_bce, valid)

        clearance_valid = clearance_valid[:, :-1].bool() & valid
        clearance_target_raw = clearance_target[:, :-1].float()
        clearance = self._balanced_clearance_supervision(
            prediction["min_human_clearance_m"],
            clearance_target_raw,
            clearance_valid,
        )
        probability = prediction["collision_probability"]
        positive = valid & (collision_target > 0.5)
        negative = valid & ~positive
        predicted_positive = valid & (probability >= 0.5)
        true_positive = predicted_positive & positive
        clearance_absolute_error = clearance["absolute_error"]
        danger_clearance = clearance["danger_mask"]
        safe_clearance = clearance["calibration_safe_mask"]
        false_safe_clearance = clearance["false_safe_mask"]
        false_danger_clearance = clearance["false_danger_mask"]
        return {
            "risk_collision": collision_loss,
            "risk_clearance": clearance["regression_loss"],
            "risk_false_safe": clearance["false_safe_loss"],
            "risk_false_danger": clearance["false_danger_loss"],
        }, {
            "label_available": valid.float().mean(),
            "future_collision_ratio": self._masked_mean(
                collision_target, valid),
            "clearance_target_mean_m": self._masked_mean(
                clearance_target_raw, clearance_valid),
            "clearance_prediction_mean_m": self._masked_mean(
                prediction["min_human_clearance_m"], clearance_valid),
            "collision_probability_mean": self._masked_mean(
                probability, valid),
            "collision_positive_weight": positive_weight.detach(),
            "collision_probability_positive": self._masked_mean(
                probability, positive),
            "collision_probability_negative": self._masked_mean(
                probability, negative),
            "collision_recall_at_0_5": (
                true_positive.sum() / positive.sum().clamp_min(1)
            ),
            "collision_precision_at_0_5": (
                true_positive.sum() / predicted_positive.sum().clamp_min(1)
            ),
            "clearance_mae_m": self._masked_mean(
                clearance_absolute_error,
                clearance_valid),
            "danger_clearance_mae_m": self._masked_mean(
                clearance_absolute_error, danger_clearance),
            "safe_clearance_mae_m": self._masked_mean(
                clearance_absolute_error, safe_clearance),
            "false_safe_clearance_ratio": (
                false_safe_clearance.sum()
                / danger_clearance.sum().clamp_min(1)
            ),
            "false_danger_clearance_ratio": (
                false_danger_clearance.sum()
                / safe_clearance.sum().clamp_min(1)
            ),
            # Sufficient statistics are removed by evaluate_loader after it
            # computes exact full-validation metrics. They avoid the old bug
            # where empty-positive batches contributed a spurious zero recall.
            "aggregate_valid_count": valid.sum(),
            "aggregate_positive_count": positive.sum(),
            "aggregate_negative_count": negative.sum(),
            "aggregate_true_positive_count": true_positive.sum(),
            "aggregate_predicted_positive_count": predicted_positive.sum(),
            "aggregate_probability_positive_sum": (
                probability * positive).sum(),
            "aggregate_probability_negative_sum": (
                probability * negative).sum(),
            "aggregate_clearance_count": clearance_valid.sum(),
            "aggregate_clearance_absolute_error_sum": (
                clearance_absolute_error * clearance_valid).sum(),
            "aggregate_clearance_prediction_sum": (
                prediction["min_human_clearance_m"] * clearance_valid).sum(),
            "aggregate_clearance_target_sum": (
                clearance_target_raw * clearance_valid).sum(),
            "aggregate_danger_clearance_count": danger_clearance.sum(),
            "aggregate_danger_clearance_absolute_error_sum": (
                clearance_absolute_error * danger_clearance).sum(),
            "aggregate_false_safe_clearance_count": false_safe_clearance.sum(),
            "aggregate_safe_clearance_count": safe_clearance.sum(),
            "aggregate_safe_clearance_absolute_error_sum": (
                clearance_absolute_error * safe_clearance).sum(),
            "aggregate_false_danger_clearance_count": (
                false_danger_clearance.sum()),
        }

    def prediction_loss(self, states: FactorizedState, batch: dict[str, torch.Tensor]):
        if self.prediction_heads is None:
            return {}, {}
        branch_feats = self.rssm.get_branch_feats(states)
        predictions, losses = self.prediction_heads.forward_loss(branch_feats, batch)
        return losses, predictions

    @staticmethod
    def _gather_time(
        value: torch.Tensor, indices: torch.Tensor, offset: int = 0,
    ) -> torch.Tensor:
        """Gather different time indices for each batch row and flatten B*K."""
        gather_index = indices + int(offset)
        shape = (*gather_index.shape, *((1,) * (value.ndim - 2)))
        expanded = gather_index.reshape(shape).expand(
            *gather_index.shape, *value.shape[2:])
        selected = torch.gather(value, 1, expanded)
        return selected.reshape(
            value.shape[0] * gather_index.shape[1], *value.shape[2:])

    def _aligned_privileged_collision_geometry(
        self,
        batch: dict[str, torch.Tensor],
        indices: torch.Tensor,
        offset: int,
        *,
        source_gt_id: torch.Tensor,
        source_gt_valid: torch.Tensor,
        source_sphere_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align simulator collision spheres to deployable source slots.

        Simulator people and perception slots have unrelated ordering, and the
        simulator contains people that the Actor has never observed.  A Human
        rollout loss may therefore use only the physical person identified by a
        source Actor slot.  It must also retain the source COCO12 collision-sphere
        mask: the deployed rollout deliberately never invents a joint that was
        absent at the source.  Supervising either unmatched people or unavailable
        spheres asks the Human model to reconstruct information outside its
        decision state and produces an irreducible false-safe gradient.
        """
        required = {
            "priv_human_id", "priv_human_mask",
            "priv_collision_joints_episode",
            "priv_collision_joint_valid",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(
                "privileged Human alignment lacks " + ", ".join(missing))
        if source_gt_id.shape != source_gt_valid.shape:
            raise ValueError("source GT identity tensors must share [R,N]")
        if source_sphere_valid.shape[:-1] != source_gt_id.shape:
            raise ValueError(
                "source sphere validity must match source GT slots")

        privileged_id = self._gather_time(
            batch["priv_human_id"], indices, offset).long()
        privileged_mask = self._gather_time(
            batch["priv_human_mask"], indices, offset).bool()
        privileged_joints = self._gather_time(
            batch["priv_collision_joints_episode"], indices, offset).float()
        privileged_joint_valid = self._gather_time(
            batch["priv_collision_joint_valid"], indices, offset).bool()
        people = min(privileged_id.shape[-1], privileged_joints.shape[-3])
        privileged_id = privileged_id[..., :people]
        privileged_mask = privileged_mask[..., :people]
        privileged_joints = privileged_joints[..., :people, :, :]
        privileged_joint_valid = privileged_joint_valid[..., :people, :]
        if privileged_joints.shape[-2] != source_sphere_valid.shape[-1]:
            raise ValueError(
                "privileged/deployable collision-sphere topology mismatch")

        matched = (
            privileged_id[..., None].eq(source_gt_id[:, None, :])
            & privileged_mask[..., None]
            & source_gt_valid[:, None, :]
        )
        if bool(matched.sum(-2).gt(1).any()):
            raise RuntimeError(
                "privileged Human IDs are not unique within a frame")
        aligned_joints = (
            privileged_joints[..., :, None, :, :]
            * matched[..., None, None].to(privileged_joints)
        ).sum(-4)
        aligned_valid = (
            privileged_joint_valid[..., :, None, :]
            & matched[..., None]
        ).any(-3)
        aligned_valid &= source_sphere_valid.bool()
        return aligned_joints, aligned_valid

    @classmethod
    def _gather_state_time(
        cls, states: FactorizedState, indices: torch.Tensor,
    ) -> FactorizedState:
        return {
            branch: {
                key: cls._gather_time(value, indices)
                for key, value in fields.items()
            }
            for branch, fields in states.items()
        }

    def _overshoot_start_indices(
        self, batch: dict[str, torch.Tensor], horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose ordinary and danger-prioritized factual rollout starts."""
        batch_size, sequence_length = batch["action"].shape[:2]
        count = sequence_length - int(horizon)
        if count <= 0:
            raise ValueError(
                f"sequence length {sequence_length} is too short for "
                f"overshoot horizon {horizon}")
        valid = torch.ones(
            batch_size, count, dtype=torch.bool, device=batch["action"].device)
        if "sequence_valid" in batch:
            valid &= batch["sequence_valid"][:, :count].bool().reshape(
                batch_size, count, -1).all(-1)
        for step in range(1, int(horizon) + 1):
            action_valid = batch["action_valid"][:, step:step + count].bool().reshape(
                batch_size, count, -1).all(-1)
            action_finite = torch.isfinite(
                batch["action"][:, step:step + count].float()).reshape(
                    batch_size, count, -1).all(-1)
            source_live = ~batch["is_last"][:, step - 1:step - 1 + count].bool().reshape(
                batch_size, count, -1).any(-1)
            target_contiguous = ~batch["is_first"][:, step:step + count].bool().reshape(
                batch_size, count, -1).any(-1)
            target_sequence_valid = torch.ones_like(valid)
            if "sequence_valid" in batch:
                target_sequence_valid = batch["sequence_valid"][
                    :, step:step + count].bool().reshape(
                        batch_size, count, -1).all(-1)
            transition_dt_valid = torch.ones_like(valid)
            if "dt_s" in batch:
                transition_dt = batch["dt_s"][
                    :, step:step + count].float().reshape(
                        batch_size, count, -1)
                transition_dt_valid = (
                    torch.isfinite(transition_dt).all(-1)
                    & transition_dt.gt(0.0).all(-1))
            valid &= (
                action_valid & action_finite & source_live
                & target_contiguous & target_sequence_valid
                & transition_dt_valid)
        starts = min(self.overshoot_starts_per_sequence, count)
        # Pack the actually valid candidates in chronological order, then
        # choose evenly spread quantiles from that packed set.  Sampling raw
        # sequence indices would select left padding in short episodes and
        # silently discard the few real transitions that remain.
        candidate_index = torch.arange(
            count, device=valid.device).expand(batch_size, count)
        packed_valid = candidate_index.masked_fill(~valid, count).sort(-1).values
        available = valid.sum(-1)
        selected_count = available.clamp_max(starts)
        slot = torch.arange(starts, device=valid.device).expand(
            batch_size, starts)
        denominator = (selected_count - 1).clamp_min(1)[:, None]
        packed_position = torch.round(
            slot.float()
            * (available - 1).clamp_min(0)[:, None].float()
            / denominator.float()
        ).long()
        packed_position = torch.where(
            selected_count[:, None].eq(1),
            ((available - 1).clamp_min(0) // 2)[:, None],
            packed_position,
        ).clamp_max(count - 1)
        selected_slot_valid = slot < selected_count[:, None]
        indices = torch.gather(packed_valid, 1, packed_position)
        indices = torch.where(
            selected_slot_valid, indices, torch.zeros_like(indices))

        collision = batch["future_collision"][:, :count].bool().reshape(
            batch_size, count, -1).any(-1)
        time_to_collision = batch["time_to_collision_s"][
            :, :count].float().reshape(batch_size, count, -1).amin(-1)
        time_to_collision_valid = batch["time_to_collision_valid"][
            :, :count].bool().reshape(batch_size, count, -1).any(-1)
        if "dt_s" in batch:
            horizon_duration = batch["dt_s"][:, 1:].float().unfold(
                1, int(horizon), 1)[:, :count].sum(-1).squeeze(-1)
        else:
            horizon_duration = batch["action"].new_full(
                (batch_size, count), 0.1 * float(horizon))
        endpoint_error = (
            time_to_collision - horizon_duration).abs()
        collision_danger = (
            collision & time_to_collision_valid
            & (time_to_collision <= horizon_duration + 0.15))
        collision_score = 3.0 + (
            1.0 - endpoint_error / max(0.1 * float(horizon), 0.1)
        ).clamp(0.0, 1.0)

        if {
            "priv_min_human_clearance_m",
            "priv_min_human_clearance_valid",
        }.issubset(batch):
            clearance = batch["priv_min_human_clearance_m"][
                :, horizon:horizon + count].float().reshape(
                    batch_size, count, -1).amin(-1)
            clearance_valid = batch["priv_min_human_clearance_valid"][
                :, horizon:horizon + count].bool().reshape(
                    batch_size, count, -1).any(-1)
        else:
            clearance = batch["future_min_human_clearance_m"][
                :, :count].float().reshape(batch_size, count, -1).amin(-1)
            clearance_valid = batch["future_min_human_clearance_valid"][
                :, :count].bool().reshape(batch_size, count, -1).any(-1)
        clearance_danger = (
            clearance_valid & (clearance <= self.safe_clearance_m))
        danger = valid & (collision_danger | clearance_danger)
        danger_score = (
            collision_score * collision_danger.float()
            + ((self.safe_clearance_m - clearance) / self.safe_clearance_m)
            .clamp(0.0, 1.0) * clearance_danger.float()
        ).masked_fill(~danger, -1.0)
        danger_index = danger_score.argmax(dim=1)
        has_danger = danger.any(dim=1)
        danger_slot = (selected_count - 1).clamp_min(0)
        danger_rows = has_danger.nonzero(as_tuple=False).flatten()
        if danger_rows.numel():
            indices[danger_rows, danger_slot[danger_rows]] = danger_index[
                danger_rows]
        return indices, selected_slot_valid.reshape(-1)

    def ego_multistep_overshooting_loss(
        self, states: FactorizedState, batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Train the deployed Ego imagination path over factual actions.

        Compact-v3 stores the action for ``observation_t -> observation_t+1``
        on row ``t+1``.  Each rollout therefore starts from posterior row
        ``t`` and consumes rows ``t+1 ... t+h``.  The source posterior is
        detached: this repair owns the Ego prior transition and Ego decoder,
        not the observation encoder that produced the factual start state.

        The loss is position/velocity/yaw aware instead of averaging all 14
        channels.  This prevents acceleration or altitude scale from hiding
        the metre-scale displacement error that directly corrupts Human
        relative clearance in closed-loop planning.
        """
        if not self.overshoot_horizons:
            return {}, {}
        if self.prediction_heads is None:
            raise RuntimeError("Ego overshooting requires prediction heads")
        required = {
            "ego_state", "action", "action_valid", "human_mask",
            "is_first", "is_last",
        }
        missing = required - set(batch)
        if missing:
            raise KeyError(f"Ego overshooting batch lacks {sorted(missing)}")

        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        sequence_length = int(batch["action"].shape[1])
        batch_size = int(batch["action"].shape[0])
        for horizon in self.overshoot_horizons:
            horizon = int(horizon)
            if sequence_length - horizon <= 0:
                continue
            indices, selected_valid_flat = self._overshoot_start_indices(
                batch, horizon)
            selected_valid = selected_valid_flat
            source_count = sequence_length - horizon
            if "dt_s" in batch:
                horizon_duration = batch["dt_s"][:, 1:].float().unfold(
                    1, horizon, 1)[:, :source_count].sum(-1).squeeze(-1)
            else:
                horizon_duration = batch["action"].new_full(
                    (batch_size, source_count), 0.1 * float(horizon))
            collision_critical = (
                batch["future_collision"][:, :source_count].bool().reshape(
                    batch_size, source_count, -1).any(-1)
                & batch["time_to_collision_valid"][
                    :, :source_count].bool().reshape(
                        batch_size, source_count, -1).any(-1)
                & (batch["time_to_collision_s"][
                    :, :source_count].float().reshape(
                        batch_size, source_count, -1).amin(-1)
                   <= horizon_duration + 0.15)
            )
            selected_collision_critical = self._gather_time(
                collision_critical[..., None], indices).reshape(
                    batch_size, indices.shape[1])

            rollout = self.ego_factual_rollout(
                states, batch, indices, horizon,
                rollout_valid=selected_valid)
            target_steps = torch.stack([
                self._gather_time(batch["ego_state"], indices, step).float()
                for step in range(1, horizon + 1)
            ], dim=1)
            prediction = rollout["ego_state"]
            baseline = rollout["analytic_ego_state"]
            valid = selected_valid[:, None].expand(-1, horizon)
            weight = valid.to(prediction.dtype)
            denominator = weight.sum().clamp_min(1.0)

            position_error = torch.linalg.vector_norm(
                prediction[..., :3] - target_steps[..., :3], dim=-1)
            velocity_error = torch.linalg.vector_norm(
                prediction[..., 3:6] - target_steps[..., 3:6], dim=-1)
            acceleration_error = torch.linalg.vector_norm(
                prediction[..., 6:9] - target_steps[..., 6:9], dim=-1)
            attitude_error = torch.linalg.vector_norm(
                prediction[..., 10:12] - target_steps[..., 10:12], dim=-1)
            predicted_yaw = torch.atan2(
                prediction[..., 12], prediction[..., 13])
            target_yaw = torch.atan2(
                target_steps[..., 12], target_steps[..., 13])
            yaw_error = torch.atan2(
                torch.sin(predicted_yaw - target_yaw),
                torch.cos(predicted_yaw - target_yaw),
            ).abs()
            baseline_position_error = torch.linalg.vector_norm(
                baseline[..., :3] - target_steps[..., :3], dim=-1)
            baseline_attitude_error = torch.linalg.vector_norm(
                baseline[..., 10:12] - target_steps[..., 10:12], dim=-1)

            def masked_mean(value: torch.Tensor) -> torch.Tensor:
                return (value * weight).sum() / denominator

            final_weight = valid[:, -1].to(prediction.dtype)
            final_denominator = final_weight.sum().clamp_min(1.0)

            def final_mean(value: torch.Tensor) -> torch.Tensor:
                return (value[:, -1] * final_weight).sum() / final_denominator

            # Position owns the largest weight because its error is the
            # observed source of false Human clearance.  Velocity makes the
            # correction dynamically consistent; the remaining state keeps
            # roll/pitch/yaw and acceleration from drifting during h15.
            per_step = (
                position_error
                + 0.35 * velocity_error
                + 0.05 * acceleration_error
                + 0.10 * attitude_error
                + 0.10 * yaw_error
            )
            losses[f"ego_overshoot_{horizon}"] = 0.5 * (
                masked_mean(per_step) + final_mean(per_step))
            metrics[f"h{horizon}/ego_position_ade_m"] = (
                masked_mean(position_error).detach())
            metrics[f"h{horizon}/ego_position_fde_m"] = (
                final_mean(position_error).detach())
            metrics[f"h{horizon}/ego_velocity_ade_mps"] = (
                masked_mean(velocity_error).detach())
            metrics[f"h{horizon}/ego_velocity_fde_mps"] = (
                final_mean(velocity_error).detach())
            metrics[f"h{horizon}/ego_yaw_ade_rad"] = (
                masked_mean(yaw_error).detach())
            metrics[f"h{horizon}/ego_attitude_ade_rad"] = (
                masked_mean(attitude_error).detach())
            metrics[f"h{horizon}/ego_attitude_fde_rad"] = (
                final_mean(attitude_error).detach())
            metrics[f"h{horizon}/ego_analytic_position_ade_m"] = (
                masked_mean(baseline_position_error).detach())
            metrics[f"h{horizon}/ego_analytic_position_fde_m"] = (
                final_mean(baseline_position_error).detach())
            metrics[f"h{horizon}/ego_analytic_attitude_ade_rad"] = (
                masked_mean(baseline_attitude_error).detach())
            metrics[f"h{horizon}/ego_analytic_attitude_fde_rad"] = (
                final_mean(baseline_attitude_error).detach())
            metrics[f"h{horizon}/ego_valid_start_count"] = (
                selected_valid.sum().detach())
            metrics[f"h{horizon}/ego_collision_start_available_fraction"] = (
                collision_critical.any(-1).float().mean().detach())
            metrics[f"h{horizon}/ego_collision_start_selected_fraction"] = (
                selected_collision_critical.any(-1).float().sum()
                / collision_critical.any(-1).float().sum().clamp_min(1.0)
            ).detach()
        return losses, metrics

    def ego_factual_rollout(
        self, states: FactorizedState, batch: dict[str, torch.Tensor],
        start_indices: torch.Tensor, horizon: int,
        *, rollout_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Roll out Ego with the exact factual action alignment used online."""
        state = {
            branch: {
                key: value.detach()
                for key, value in fields.items()
            }
            for branch, fields in self._gather_state_time(
                states, start_indices).items()
        }
        human_mask = self._gather_time(
            batch["human_mask"], start_indices).bool()
        ego = self._gather_time(batch["ego_state"], start_indices).float()
        decoded_ego = self.prediction_heads.decode_ego_state(
            self.rssm.get_branch_feats(state)["ego"])
        predicted, analytic = [], []
        analytic_ego = ego
        factual_row_valid = (
            torch.ones(
                start_indices.numel(), dtype=torch.bool,
                device=start_indices.device)
            if rollout_valid is None else
            rollout_valid.bool().reshape(-1)
        )
        if factual_row_valid.numel() != start_indices.numel():
            raise ValueError(
                "Ego factual rollout validity must match flattened starts")
        for step in range(1, int(horizon) + 1):
            action = self._gather_time(
                batch["action"], start_indices, step).float().clamp(-1.0, 1.0)
            transition_dt = (
                self._gather_time(
                    batch["dt_s"], start_indices, step).float()
                if "dt_s" in batch else
                action.new_full((action.shape[0], 1), 0.1)
            )
            transition_dt_valid = (
                torch.isfinite(transition_dt).reshape(
                    transition_dt.shape[0], -1).all(-1)
                & transition_dt.reshape(
                    transition_dt.shape[0], -1).gt(0.0).all(-1))
            if bool((factual_row_valid & ~transition_dt_valid).any()):
                raise ValueError(
                    "factual Ego rollout requires positive destination dt_s")
            # Fixed-size start tensors need harmless placeholders for batch
            # rows with no horizon-length factual window.  They are excluded
            # from every objective below; sanitizing them here prevents
            # padding NaNs/zero dt from contaminating a masked loss.
            action = torch.where(
                factual_row_valid[:, None], action, torch.zeros_like(action))
            transition_dt = torch.where(
                transition_dt_valid[:, None], transition_dt,
                transition_dt.new_full(transition_dt.shape, 0.1))
            next_state, _ = self.rssm.img_step(state, action, human_mask)
            next_decoded = self.prediction_heads.decode_ego_state(
                self.rssm.get_branch_feats(next_state)["ego"])
            next_ego = (
                self.anchor_ego_residual(
                    ego, decoded_ego, next_decoded, action,
                    dt_s=transition_dt,
                    velocity_response=self.analytic_ego_velocity_response)
                if self.uses_internal_action_adapter
                else self.anchor_ego_displacement(
                    ego, decoded_ego, next_decoded)
            )
            # The production contract is applied4.  Lightweight legacy unit
            # fixtures use action3 without the analytic adapter; retain a
            # stationary diagnostic baseline for those fixtures rather than
            # changing their model interface.
            if action.shape[-1] == 4:
                analytic_ego = analytic_ego_step(
                    analytic_ego, action, dt_s=transition_dt,
                    velocity_response=self.analytic_ego_velocity_response,
                    attitude_coefficients=(
                        self.analytic_ego_attitude_coefficients),
                )
            predicted.append(next_ego)
            analytic.append(analytic_ego)
            state, ego, decoded_ego = next_state, next_ego, next_decoded
        return {
            "ego_state": torch.stack(predicted, dim=1),
            "analytic_ego_state": torch.stack(analytic, dim=1),
            "final_state": state,
        }

    def multistep_overshooting_loss(
        self, states: FactorizedState, batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Human-only factual overshooting with the deployment rollout path.

        Targets are restricted to continuously tracked, measured people.  No
        future observation is supplied and no legacy risk/candidate objective
        is mixed into this loss.  Evaluation calls ``human_factual_rollout``
        as well, so Ego compensation and transition alignment cannot diverge.
        """
        if not self.overshoot_horizons:
            return {}, {}
        if self.prediction_heads is None:
            raise RuntimeError("multistep overshooting requires prediction heads")
        required = {
            "human_root", "human_ids", "action_valid", "human_mask",
            "ego_state", "action", "is_first", "is_last",
        }
        missing = required - set(batch)
        if missing:
            raise KeyError(
                f"multistep overshooting batch lacks {sorted(missing)}")

        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        sequence_length = batch["action"].shape[1]
        for horizon in self.overshoot_horizons:
            max_start = sequence_length - int(horizon) - (
                1 if self.human_only_overshooting else 2)
            if max_start < 0:
                continue
            batch_size = batch["action"].shape[0]
            count = min(
                max_start + 1, max(1, int(self.overshoot_starts_per_sequence)))
            source_count = max_start + 1
            source_indices = torch.arange(
                source_count, device=batch["action"].device)
            rollout_offsets = torch.arange(
                1, int(horizon) + 1, device=batch["action"].device)
            target_indices = source_indices[:, None] + rollout_offsets[None]
            previous_indices = target_indices - 1
            window_valid = torch.ones(
                batch_size, source_count, dtype=torch.bool,
                device=batch["action"].device)
            if "sequence_valid" in batch:
                window_valid &= batch["sequence_valid"][
                    :, :source_count].bool().reshape(
                        batch_size, source_count, -1).all(-1)
            measured_valid_all = batch.get(
                "measured_root_target_valid", batch["human_mask"])
            action_valid_windows = batch["action_valid"][
                :, target_indices].bool().reshape(
                    batch_size, source_count, int(horizon), -1).all(-1)
            finite_action_windows = torch.isfinite(
                batch["action"][:, target_indices].float()).reshape(
                    batch_size, source_count, int(horizon), -1).all(-1)
            previous_not_last = ~batch["is_last"][
                :, previous_indices].bool().reshape(
                    batch_size, source_count, int(horizon), -1).any(-1)
            target_not_first = ~batch["is_first"][
                :, target_indices].bool().reshape(
                    batch_size, source_count, int(horizon), -1).any(-1)
            window_valid &= (
                action_valid_windows
                & finite_action_windows
                & previous_not_last
                & target_not_first
            ).all(-1)
            if "sequence_valid" in batch:
                window_valid &= batch["sequence_valid"][
                    :, target_indices].bool().reshape(
                        batch_size, source_count, int(horizon), -1
                    ).all(-1).all(-1)
            if "dt_s" in batch:
                transition_dt = batch["dt_s"][
                    :, target_indices].float().reshape(
                        batch_size, source_count, int(horizon), -1)
                window_valid &= (
                    torch.isfinite(transition_dt).all(-1)
                    & transition_dt.gt(0.0).all(-1)
                ).all(-1)

            source_ids_all = batch["human_ids"][:, :source_count]
            source_present = batch["human_mask"][
                :, :source_count].bool()
            stable_step = (
                batch["human_mask"][:, target_indices].bool()
                & source_present[:, :, None]
                & batch["human_ids"][:, target_indices].eq(
                    source_ids_all[:, :, None])
                & source_ids_all[:, :, None].ge(0)
                & ~batch["human_is_first"][:, target_indices].bool()
            )
            stable_path_steps = stable_step.to(torch.int8).cumprod(
                dim=2).bool()
            measured_steps = measured_valid_all[
                :, target_indices].bool()
            valid_steps = stable_path_steps & measured_steps
            candidate_scores = valid_steps.float().sum(dim=(-1, -2))
            # Prefer starts with a measurable final target, while still using
            # intermediate supervision when a long track ends early.
            candidate_scores += 2.0 * valid_steps[:, :, -1].float().sum(-1)
            coverage_score = candidate_scores.masked_fill(
                ~window_valid, -1.0e9)
            indices = coverage_score.topk(count, dim=1).indices.long()
            selected_count = window_valid.sum(-1).clamp_max(count)
            selected_slot_valid = (
                torch.arange(count, device=indices.device)[None]
                < selected_count[:, None])

            # Reserve the final start for a collision-critical or low-clearance
            # factual rollout. Uniform replay sampling is unchanged; only the
            # two supervised starts inside each sampled sequence are chosen
            # differently.  A collision source is most useful when its contact
            # occurs near the requested rollout endpoint, so use the recorded
            # time-to-collision rather than a filename/outcome heuristic.
            source_human = batch["human_mask"][:, :source_count].bool().any(-1)
            source_joint = batch.get("joint_mask")
            if source_joint is not None:
                source_human &= source_joint[
                    :, :source_count].bool().flatten(-2).any(-1)
            source_identity_id_all = batch.get(
                "human_gt_identity_id", batch.get("human_gt_id"))
            source_identity_valid_all = batch.get(
                "human_gt_identity_valid", batch.get("human_gt_match_valid"))
            actor_source_identity: torch.Tensor | None = None
            actor_source_sphere_valid: torch.Tensor | None = None
            if (
                source_identity_id_all is not None
                and source_identity_valid_all is not None
                and source_joint is not None
            ):
                source_identity_id_all = source_identity_id_all[
                    :, :source_count].long()
                actor_source_identity = (
                    source_identity_valid_all[:, :source_count].bool()
                    & batch["human_mask"][:, :source_count].bool()
                    & source_identity_id_all.ge(0)
                )
                source_skeleton_all = batch["skeleton"][
                    :, :source_count].float()
                _, _, actor_source_sphere_valid, _ = (
                    self.deployable_human_collision_geometry(
                        source_skeleton_all[..., :3],
                        source_skeleton_all[..., 3:6],
                        source_joint[:, :source_count].bool(),
                    ))
                actor_source_identity &= actor_source_sphere_valid.any(-1)
            collision = batch.get("future_collision")
            collision_valid = batch.get("time_to_collision_valid")
            time_to_collision = batch.get("time_to_collision_s")
            collision_danger = torch.zeros_like(window_valid)
            collision_score = candidate_scores.new_zeros(window_valid.shape)
            endpoint_tolerance_s = 0.15
            if "dt_s" in batch:
                factual_horizon_duration = batch["dt_s"][:, 1:].float().unfold(
                    1, int(horizon), 1)[:, :source_count].sum(-1).squeeze(-1)
            else:
                factual_horizon_duration = candidate_scores.new_full(
                    window_valid.shape, 0.1 * float(horizon))
            if (
                collision is not None
                and collision_valid is not None
                and time_to_collision is not None
            ):
                collision = collision[:, :source_count].bool().reshape(
                    batch_size, source_count, -1).any(-1)
                collision_valid = collision_valid[
                    :, :source_count].bool().reshape(
                        batch_size, source_count, -1).any(-1)
                time_to_collision = time_to_collision[
                    :, :source_count].float().reshape(
                        batch_size, source_count, -1).amin(-1)
                endpoint_error = (
                    time_to_collision - factual_horizon_duration).abs()
                collision_danger = (
                    collision & collision_valid
                    & (time_to_collision <= factual_horizon_duration
                       + endpoint_tolerance_s))
                # An exact but currently unobserved collision belongs to the
                # hidden Event cause, not to visible-person motion repair.
                # Prioritize it here only when the recorded contact identity is
                # carried by an Actor-visible source slot.
                collision_human_id = batch.get("priv_collision_human_id")
                if (
                    collision_human_id is not None
                    and actor_source_identity is not None
                    and source_identity_id_all is not None
                ):
                    contact_id = collision_human_id[
                        :, target_indices].long().reshape(
                            batch_size, source_count, int(horizon), -1)
                    if contact_id.shape[-1] != 1:
                        raise ValueError(
                            "priv_collision_human_id must have one ID per row")
                    collision_actor_attributable = (
                        source_identity_id_all[:, :, None, :].eq(
                            contact_id[..., 0, None])
                        & actor_source_identity[:, :, None, :]
                    ).any(dim=(-1, -2))
                    collision_danger &= collision_actor_attributable
                collision_score = 3.0 + (
                    1.0 - endpoint_error / max(
                        0.1 * float(horizon), 0.1)
                ).clamp(0.0, 1.0)

            clearance_danger = torch.zeros_like(window_valid)
            clearance_score = candidate_scores.new_zeros(window_valid.shape)
            aligned_clearance_fields = {
                "priv_human_id", "priv_human_mask",
                "priv_collision_joints_episode",
                "priv_collision_joint_valid",
                "counterfactual_collision_surface_radii_m",
                "privileged_geometry_available",
            }
            if (
                aligned_clearance_fields.issubset(batch)
                and actor_source_identity is not None
                and actor_source_sphere_valid is not None
                and source_identity_id_all is not None
            ):
                all_candidate_indices = torch.arange(
                    source_count, device=indices.device
                )[None].expand(batch_size, source_count)
                aligned_endpoint, aligned_endpoint_valid = (
                    self._aligned_privileged_collision_geometry(
                        batch,
                        all_candidate_indices,
                        int(horizon),
                        source_gt_id=source_identity_id_all.reshape(
                            batch_size * source_count, -1),
                        source_gt_valid=actor_source_identity.reshape(
                            batch_size * source_count, -1),
                        source_sphere_valid=actor_source_sphere_valid.reshape(
                            batch_size * source_count,
                            actor_source_sphere_valid.shape[-2],
                            actor_source_sphere_valid.shape[-1],
                        ),
                    ))
                endpoint_ego = self._gather_time(
                    batch["ego_state"], all_candidate_indices,
                    int(horizon)).float()
                endpoint_radii = self._gather_time(
                    batch["counterfactual_collision_surface_radii_m"],
                    all_candidate_indices,
                    int(horizon),
                ).float()
                endpoint_gap = (
                    torch.linalg.vector_norm(
                        aligned_endpoint
                        - endpoint_ego[..., None, None, :3],
                        dim=-1,
                    )
                    - endpoint_radii[:, None, :]
                ).masked_fill(~aligned_endpoint_valid, torch.inf)
                endpoint_clearance = endpoint_gap.flatten(1).amin(-1).reshape(
                    batch_size, source_count)
                endpoint_clearance_valid = (
                    aligned_endpoint_valid.flatten(1).any(-1).reshape(
                        batch_size, source_count)
                    & batch["privileged_geometry_available"][
                        :, horizon:horizon + source_count
                    ].bool().reshape(batch_size, source_count, -1).all(-1)
                )
                clearance_danger = (
                    endpoint_clearance_valid
                    & (endpoint_clearance <= self.safe_clearance_m))
                clearance_score = (
                    (self.safe_clearance_m - endpoint_clearance)
                    / max(self.safe_clearance_m, 1.0e-6)
                ).clamp(0.0, 1.0)
            elif {
                "priv_min_human_clearance_m",
                "priv_min_human_clearance_valid",
            }.issubset(batch):
                # Legacy factorized callers lack simulator identity alignment.
                # Current pure-Dreamer training requires the fields above and
                # never enters this compatibility branch.
                endpoint_clearance = batch["priv_min_human_clearance_m"][
                    :, horizon:horizon + source_count].float().reshape(
                        batch_size, source_count, -1).amin(-1)
                endpoint_clearance_valid = batch[
                    "priv_min_human_clearance_valid"
                ][:, horizon:horizon + source_count].bool().reshape(
                    batch_size, source_count, -1).any(-1)
                clearance_danger = (
                    endpoint_clearance_valid
                    & (endpoint_clearance <= self.safe_clearance_m))
                clearance_score = (
                    (self.safe_clearance_m - endpoint_clearance)
                    / max(self.safe_clearance_m, 1.0e-6)
                ).clamp(0.0, 1.0)

            danger = (
                window_valid & source_human
                & (collision_danger | clearance_danger))
            danger_score = (
                collision_score + clearance_score
            ).masked_fill(~danger, -1.0e9)
            danger_index = danger_score.argmax(dim=1)
            has_danger = danger.any(dim=1)
            danger_slot = (selected_count - 1).clamp_min(0)
            danger_rows = has_danger.nonzero(as_tuple=False).flatten()
            if danger_rows.numel():
                indices[danger_rows, danger_slot[danger_rows]] = danger_index[
                    danger_rows]
            selected_window_valid = selected_slot_valid.reshape(-1)
            selected_danger = self._gather_time(
                danger[..., None], indices).reshape(
                    batch_size, count) & selected_slot_valid
            metrics[f"h{horizon}/danger_start_available_fraction"] = (
                has_danger.float().mean().detach())
            metrics[f"h{horizon}/danger_start_selected_fraction"] = (
                selected_danger.any(-1).float().sum()
                / has_danger.float().sum().clamp_min(1.0)).detach()
            selected_collision = self._gather_time(
                collision_danger[..., None], indices).reshape(
                    batch_size, count) & selected_slot_valid
            collision_available = (
                collision_danger & window_valid & source_human).any(dim=1)
            metrics[f"h{horizon}/collision_start_available_fraction"] = (
                collision_available.float().mean().detach())
            metrics[f"h{horizon}/collision_start_selected_fraction"] = (
                selected_collision.any(-1).float().sum()
                / collision_available.float().sum().clamp_min(1.0)).detach()
            metrics[f"h{horizon}/valid_start_count"] = (
                selected_window_valid.sum().detach())
            rollout = self.human_factual_rollout(
                states, batch, indices, int(horizon),
                # A Human-only repair must compare against the recorded Human
                # target under the recorded Ego motion.  Letting an already
                # frozen Ego rollout drift here makes the Human head absorb an
                # Ego error that it cannot correct at deployment.
                use_recorded_ego=self.human_only_overshooting,
                rollout_valid=selected_window_valid)
            gt_rollout_real_ego = None
            has_gt_audit = {
                "human_gt_id", "human_gt_match_valid",
                "human_gt_pelvis_body", "human_gt_pelvis_episode",
            }.issubset(batch)
            if self.require_gt_displacement_supervision and not has_gt_audit:
                raise KeyError(
                    "GT displacement supervision requires human_gt_id, "
                    "human_gt_match_valid, human_gt_pelvis_body, and "
                    "human_gt_pelvis_episode")
            if has_gt_audit:
                # Audit-only path: fixing Ego to its recorded trajectory
                # separates Human motion error from end-to-end Ego rollout
                # error. Human-only training already used that exact recorded
                # Ego path above, so reuse it instead of recursively evaluating
                # the same rollout a second time for a detached diagnostic.
                # The compatibility end-to-end mode still needs the separate
                # recorded-Ego audit and never contributes it to the loss.
                if self.human_only_overshooting:
                    gt_rollout_real_ego = rollout
                else:
                    with torch.no_grad():
                        gt_rollout_real_ego = self.human_factual_rollout(
                            states, batch, indices, int(horizon),
                            use_recorded_ego=True,
                            rollout_valid=selected_window_valid)
            source_ids = rollout["source_ids"]
            stable_path = rollout["source_mask"].clone()
            stable_path &= selected_window_valid[:, None]
            selected_near_row = self._gather_time(
                danger[..., None], indices).reshape(-1).bool()
            source_occluded = torch.zeros_like(stable_path)
            if "human_persistence_mask" in batch:
                source_occluded |= self._gather_time(
                    batch["human_persistence_mask"], indices).bool()
            if "human_joint_predicted" in batch:
                source_occluded |= self._gather_time(
                    batch["human_joint_predicted"], indices
                ).bool().any(-1)
            action_saturated_row = torch.zeros(
                source_ids.shape[0], dtype=torch.bool,
                device=source_ids.device)
            for audit_step in range(1, int(horizon) + 1):
                action_saturated_row |= self._gather_time(
                    batch["action"], indices, audit_step
                ).float().abs().ge(0.95).any(-1)
            subset_mask = {
                "ordinary": ~(
                    selected_near_row[..., None]
                    | source_occluded
                    | action_saturated_row[..., None]
                ),
                "near_collision": selected_near_row[..., None].expand_as(
                    stable_path),
                "occlusion": source_occluded,
                "action_saturation": action_saturated_row[..., None].expand_as(
                    stable_path),
            }
            if has_gt_audit:
                source_gt_id = self._gather_time(
                    batch["human_gt_id"], indices)
                gt_stable_path = self._gather_time(
                    batch["human_gt_match_valid"], indices).bool()
                gt_stable_path &= source_gt_id >= 0
                gt_stable_path &= selected_window_valid[:, None]
            else:
                source_gt_id = None
                gt_stable_path = None
            human_error_sum = rollout["root"].new_zeros(())
            human_count = rollout["root"].new_zeros(())
            final_error_sum = rollout["root"].new_zeros(())
            final_count = rollout["root"].new_zeros(())
            baseline_error_sum = rollout["root"].new_zeros(())
            baseline_final_sum = rollout["root"].new_zeros(())
            cv_regret_sum = rollout["root"].new_zeros(())
            cv_regret_final_sum = rollout["root"].new_zeros(())
            gt_cv_regret_sum = rollout["root"].new_zeros(())
            gt_cv_regret_final_sum = rollout["root"].new_zeros(())
            gt_cv_regret_count = rollout["root"].new_zeros(())
            gt_cv_regret_final_count = rollout["root"].new_zeros(())
            zero_baseline_error_sum = rollout["root"].new_zeros(())
            zero_baseline_final_sum = rollout["root"].new_zeros(())
            gt_real_ego_final_sum = rollout["root"].new_zeros(())
            gt_end_to_end_final_sum = rollout["root"].new_zeros(())
            gt_displacement_final_sum = rollout["root"].new_zeros(())
            gt_displacement_loss = rollout["root"].new_zeros(())
            gt_perception_sum = rollout["root"].new_zeros(())
            gt_final_count = rollout["root"].new_zeros(())
            joint_error_sum = rollout["joints"].new_zeros(())
            joint_baseline_error_sum = rollout["joints"].new_zeros(())
            joint_count = rollout["joints"].new_zeros(())
            joint_final_error_sum = rollout["joints"].new_zeros(())
            joint_baseline_final_sum = rollout["joints"].new_zeros(())
            joint_final_count = rollout["joints"].new_zeros(())
            subset_root_error_sum = {
                name: rollout["root"].new_zeros(()) for name in subset_mask}
            subset_root_baseline_sum = {
                name: rollout["root"].new_zeros(()) for name in subset_mask}
            subset_root_count = {
                name: rollout["root"].new_zeros(()) for name in subset_mask}
            subset_joint_error_sum = {
                name: rollout["joints"].new_zeros(()) for name in subset_mask}
            subset_joint_baseline_sum = {
                name: rollout["joints"].new_zeros(()) for name in subset_mask}
            subset_joint_count = {
                name: rollout["joints"].new_zeros(()) for name in subset_mask}
            source_joint_valid = self._gather_time(
                batch["joint_mask"], indices).bool()
            if has_gt_audit:
                source_root_body = self._gather_time(
                    batch["human_root"], indices)[..., :3].float()
                source_ego = self._gather_time(
                    batch["ego_state"], indices).float()
                predicted_source_episode = body_points_to_episode(
                    source_root_body, source_ego)
                source_gt_episode = self._gather_time(
                    batch["human_gt_pelvis_episode"], indices).float()
            else:
                predicted_source_episode = None
                source_gt_episode = None
            for step in range(1, int(horizon) + 1):
                target_ids = self._gather_time(batch["human_ids"], indices, step)
                target_mask = self._gather_time(batch["human_mask"], indices, step).bool()
                step_stable = target_mask & (source_ids == target_ids) & (source_ids >= 0)
                if "human_is_first" in batch:
                    step_stable &= ~self._gather_time(
                        batch["human_is_first"], indices, step).bool()
                stable_path &= step_stable
                if gt_stable_path is not None and source_gt_id is not None:
                    step_gt_id = self._gather_time(
                        batch["human_gt_id"], indices, step)
                    step_gt_valid = self._gather_time(
                        batch["human_gt_match_valid"], indices, step).bool()
                    gt_stable_path &= (
                        step_gt_valid & (source_gt_id == step_gt_id))
                measured_root = batch.get("measured_root_target", batch["human_root"])
                measured_valid = batch.get(
                    "measured_root_target_valid", batch["human_mask"])
                target_root = self._gather_time(
                    measured_root, indices, step)[..., :3].float()
                valid = stable_path & self._gather_time(
                    measured_valid, indices, step).bool()
                root_error = torch.linalg.vector_norm(
                    rollout["root"][:, step - 1, ..., :3] - target_root, dim=-1)
                baseline_error = torch.linalg.vector_norm(
                    rollout["constant_velocity_root"][:, step - 1, ..., :3]
                    - target_root, dim=-1)
                # Keep the old measured-root regret as a diagnostic only.
                # Detector error can change across an occluded collision tail,
                # so it is not a physical optimization target.
                cv_regret = F.relu(
                    root_error.detach() - baseline_error.detach()).square()
                zero_baseline_error = torch.linalg.vector_norm(
                    rollout["zero_velocity_root"][:, step - 1, ..., :3]
                    - target_root, dim=-1)
                human_error_sum += (root_error * valid.float()).sum()
                baseline_error_sum += (baseline_error * valid.float()).sum()
                cv_regret_sum += (cv_regret * valid.float()).sum()
                zero_baseline_error_sum += (
                    zero_baseline_error * valid.float()).sum()
                human_count += valid.sum()
                gt_cv_step_loss = None
                gt_cv_predicted_error = None
                gt_cv_baseline_error = None
                gt_valid = None
                target_gt_episode = None
                predicted_episode = None
                if (
                    has_gt_audit
                    and source_gt_id is not None
                    and gt_stable_path is not None
                    and predicted_source_episode is not None
                    and source_gt_episode is not None
                ):
                    gt_valid = gt_stable_path & valid
                    predicted_episode = body_points_to_episode(
                        rollout["root"][:, step - 1, ..., :3],
                        rollout["ego_state"][:, step - 1],
                    )
                    constant_velocity_episode = body_points_to_episode(
                        rollout["constant_velocity_root"][
                            :, step - 1, ..., :3],
                        rollout["ego_state"][:, step - 1],
                    )
                    target_gt_episode = self._gather_time(
                        batch["human_gt_pelvis_episode"], indices,
                        step).float()
                    (
                        gt_cv_step_loss,
                        gt_cv_predicted_error,
                        gt_cv_baseline_error,
                    ) = self._gt_cv_regret_supervision(
                        predicted_episode,
                        predicted_source_episode,
                        constant_velocity_episode,
                        target_gt_episode,
                        source_gt_episode,
                        gt_valid,
                    )
                    gt_step_count = gt_valid.sum()
                    gt_cv_regret_sum += gt_cv_step_loss * gt_step_count
                    gt_cv_regret_count += gt_step_count
                measured_joint = batch.get(
                    "measured_joint_target", batch["skeleton"][..., :3])
                measured_joint_valid = batch.get(
                    "measured_joint_target_valid", batch["joint_mask"])
                target_joint = self._gather_time(
                    measured_joint, indices, step).float()
                target_joint_valid = self._gather_time(
                    measured_joint_valid, indices, step).bool()
                joint_valid = (
                    valid[..., None]
                    & source_joint_valid
                    & target_joint_valid
                )
                joint_error = torch.linalg.vector_norm(
                    rollout["joints"][:, step - 1] - target_joint,
                    dim=-1,
                )
                joint_baseline_error = torch.linalg.vector_norm(
                    rollout["constant_velocity_joints"][:, step - 1]
                    - target_joint,
                    dim=-1,
                )
                joint_error_sum += (
                    joint_error * joint_valid.float()).sum()
                joint_baseline_error_sum += (
                    joint_baseline_error * joint_valid.float()).sum()
                joint_count += joint_valid.sum()
                if step == int(horizon):
                    final_error_sum = (root_error * valid.float()).sum()
                    baseline_final_sum = (baseline_error * valid.float()).sum()
                    cv_regret_final_sum = (
                        cv_regret * valid.float()).sum()
                    zero_baseline_final_sum = (
                        zero_baseline_error * valid.float()).sum()
                    final_count = valid.sum()
                    joint_final_error_sum = (
                        joint_error * joint_valid.float()).sum()
                    joint_baseline_final_sum = (
                        joint_baseline_error * joint_valid.float()).sum()
                    joint_final_count = joint_valid.sum()
                    for subset_name, source_subset in subset_mask.items():
                        subset_valid = valid & source_subset
                        subset_joint_valid = (
                            joint_valid & source_subset[..., None])
                        subset_root_error_sum[subset_name] = (
                            root_error * subset_valid.float()).sum()
                        subset_root_baseline_sum[subset_name] = (
                            baseline_error * subset_valid.float()).sum()
                        subset_root_count[subset_name] = subset_valid.sum()
                        subset_joint_error_sum[subset_name] = (
                            joint_error * subset_joint_valid.float()).sum()
                        subset_joint_baseline_sum[subset_name] = (
                            joint_baseline_error
                            * subset_joint_valid.float()).sum()
                        subset_joint_count[subset_name] = (
                            subset_joint_valid.sum())
                    if has_gt_audit and gt_rollout_real_ego is not None:
                        assert gt_valid is not None
                        assert predicted_episode is not None
                        assert target_gt_episode is not None
                        assert predicted_source_episode is not None
                        assert source_gt_episode is not None
                        assert gt_cv_step_loss is not None
                        assert gt_cv_predicted_error is not None
                        assert gt_cv_baseline_error is not None
                        target_gt_body = self._gather_time(
                            batch["human_gt_pelvis_body"], indices,
                            step).float()
                        gt_real_error = torch.linalg.vector_norm(
                            gt_rollout_real_ego["root"][
                                :, step - 1, ..., :3] - target_gt_body,
                            dim=-1)
                        gt_end_error = torch.linalg.vector_norm(
                            predicted_episode - target_gt_episode, dim=-1)
                        gt_displacement_loss, displacement_error = (
                            self._gt_displacement_supervision(
                                predicted_episode,
                                predicted_source_episode,
                                target_gt_episode,
                                source_gt_episode,
                                gt_valid,
                            ))
                        perception_error = torch.linalg.vector_norm(
                            target_root - target_gt_body, dim=-1)
                        gt_weight = gt_valid.float()
                        gt_real_ego_final_sum = (
                            gt_real_error * gt_weight).sum()
                        gt_end_to_end_final_sum = (
                            gt_end_error * gt_weight).sum()
                        gt_displacement_final_sum = (
                            displacement_error * gt_weight).sum()
                        gt_perception_sum = (
                            perception_error * gt_weight).sum()
                        gt_final_count = gt_valid.sum()
                        gt_cv_regret_final_sum = (
                            gt_cv_step_loss * gt_final_count)
                        gt_cv_regret_final_count = gt_final_count
            human_ade = human_error_sum / human_count.clamp_min(1)
            human_fde = final_error_sum / final_count.clamp_min(1)
            baseline_ade = baseline_error_sum / human_count.clamp_min(1)
            baseline_fde = baseline_final_sum / final_count.clamp_min(1)
            zero_baseline_ade = (
                zero_baseline_error_sum / human_count.clamp_min(1))
            zero_baseline_fde = (
                zero_baseline_final_sum / final_count.clamp_min(1))
            losses[f"human_overshoot_{horizon}"] = 0.5 * (human_ade + human_fde)
            if self.collision_tail_regret_supervision:
                if has_gt_audit:
                    losses[f"human_cv_regret_{horizon}"] = 0.5 * (
                        gt_cv_regret_sum
                        / gt_cv_regret_count.clamp_min(1)
                        + gt_cv_regret_final_sum
                        / gt_cv_regret_final_count.clamp_min(1)
                    )
                else:
                    # Compatibility for generic factorized configurations.
                    # Production Pure Dreamer requires privileged displacement
                    # labels and therefore never takes this fallback.
                    losses[f"human_cv_regret_{horizon}"] = 0.5 * (
                        cv_regret_sum / human_count.clamp_min(1)
                        + cv_regret_final_sum / final_count.clamp_min(1)
                    )
            joint_ade = joint_error_sum / joint_count.clamp_min(1)
            joint_fde = (
                joint_final_error_sum / joint_final_count.clamp_min(1))
            joint_baseline_ade = (
                joint_baseline_error_sum / joint_count.clamp_min(1))
            joint_baseline_fde = (
                joint_baseline_final_sum
                / joint_final_count.clamp_min(1))
            losses[f"human_joint_overshoot_{horizon}"] = 0.5 * (
                joint_ade + joint_fde)
            metrics[f"h{horizon}/human_ade_m"] = human_ade.detach()
            metrics[f"h{horizon}/human_fde_m"] = human_fde.detach()
            metrics[f"h{horizon}/constant_velocity_ade_m"] = baseline_ade.detach()
            metrics[f"h{horizon}/constant_velocity_fde_m"] = baseline_fde.detach()
            metrics[f"h{horizon}/zero_velocity_ade_m"] = (
                zero_baseline_ade.detach())
            metrics[f"h{horizon}/zero_velocity_fde_m"] = (
                zero_baseline_fde.detach())
            metrics[f"h{horizon}/fde_to_baseline_ratio"] = (
                human_fde / baseline_fde.clamp_min(1.0e-4)).detach()
            if self.collision_tail_regret_supervision:
                metrics[f"h{horizon}/human_cv_regret_m2"] = (
                    cv_regret_sum / human_count.clamp_min(1)).detach()
                if has_gt_audit:
                    metrics[f"h{horizon}/human_gt_cv_regret_m2"] = losses[
                        f"human_cv_regret_{horizon}"].detach()
                    metrics[f"h{horizon}/human_gt_cv_regret_count"] = (
                        gt_cv_regret_final_count.detach())
            metrics[f"h{horizon}/stable_measured_count"] = final_count.detach()
            metrics[f"h{horizon}/joint_ade_m"] = joint_ade.detach()
            metrics[f"h{horizon}/joint_fde_m"] = joint_fde.detach()
            metrics[f"h{horizon}/constant_velocity_joint_ade_m"] = (
                joint_baseline_ade.detach())
            metrics[f"h{horizon}/constant_velocity_joint_fde_m"] = (
                joint_baseline_fde.detach())
            metrics[f"h{horizon}/joint_fde_to_baseline_ratio"] = (
                joint_fde / joint_baseline_fde.clamp_min(1.0e-4)).detach()
            metrics[f"h{horizon}/stable_measured_joint_count"] = (
                joint_final_count.detach())
            final_root_values = root_error[valid]
            final_root_baseline_values = baseline_error[valid]
            final_joint_values = joint_error[joint_valid]
            final_joint_baseline_values = joint_baseline_error[joint_valid]

            def audited_quantile(
                value: torch.Tensor, quantile: float,
            ) -> torch.Tensor:
                return (
                    torch.quantile(value.float(), quantile)
                    if value.numel() else rollout["root"].new_zeros(())
                )

            metrics[f"h{horizon}/human_fde_p95_m"] = audited_quantile(
                final_root_values, 0.95).detach()
            metrics[f"h{horizon}/human_fde_max_m"] = (
                final_root_values.amax().detach()
                if final_root_values.numel() else
                rollout["root"].new_zeros(()))
            metrics[f"h{horizon}/constant_velocity_fde_p95_m"] = (
                audited_quantile(
                    final_root_baseline_values, 0.95).detach())
            metrics[f"h{horizon}/joint_fde_p95_m"] = audited_quantile(
                final_joint_values, 0.95).detach()
            metrics[f"h{horizon}/constant_velocity_joint_fde_p95_m"] = (
                audited_quantile(
                    final_joint_baseline_values, 0.95).detach())
            lateral_direction_valid = valid & target_root[..., 1].abs().ge(0.10)
            learned_wrong_side = lateral_direction_valid & (
                rollout["root"][:, -1, ..., 1].sign()
                != target_root[..., 1].sign())
            cv_wrong_side = lateral_direction_valid & (
                rollout["constant_velocity_root"][:, -1, ..., 1].sign()
                != target_root[..., 1].sign())
            metrics[f"h{horizon}/lateral_direction_valid_count"] = (
                lateral_direction_valid.sum().detach())
            metrics[f"h{horizon}/lateral_direction_error_ratio"] = (
                learned_wrong_side.sum()
                / lateral_direction_valid.sum().clamp_min(1)).detach()
            metrics[
                f"h{horizon}/constant_velocity_lateral_direction_error_ratio"
            ] = (
                cv_wrong_side.sum()
                / lateral_direction_valid.sum().clamp_min(1)).detach()
            # Section-10 acceptance is distributional: a mean FDE can hide
            # exactly the collision, occlusion, or saturated-action rows that
            # determine closed-loop safety.  Report learned and CV on every
            # required subset from the identical selected starts/targets.
            for subset_name in subset_mask:
                source_subset = subset_mask[subset_name]
                root_denominator = subset_root_count[
                    subset_name].clamp_min(1.0)
                joint_denominator = subset_joint_count[
                    subset_name].clamp_min(1.0)
                subset_fde = (
                    subset_root_error_sum[subset_name] / root_denominator)
                subset_cv_fde = (
                    subset_root_baseline_sum[subset_name] / root_denominator)
                subset_joint_fde = (
                    subset_joint_error_sum[subset_name] / joint_denominator)
                subset_cv_joint_fde = (
                    subset_joint_baseline_sum[subset_name]
                    / joint_denominator)
                subset_prefix = f"h{horizon}/subset_{subset_name}"
                metrics[f"{subset_prefix}_human_fde_m"] = (
                    subset_fde.detach())
                metrics[f"{subset_prefix}_constant_velocity_fde_m"] = (
                    subset_cv_fde.detach())
                metrics[f"{subset_prefix}_fde_to_baseline_ratio"] = (
                    subset_fde / subset_cv_fde.clamp_min(1.0e-4)).detach()
                metrics[f"{subset_prefix}_human_count"] = (
                    subset_root_count[subset_name].detach())
                metrics[f"{subset_prefix}_joint_fde_m"] = (
                    subset_joint_fde.detach())
                metrics[f"{subset_prefix}_constant_velocity_joint_fde_m"] = (
                    subset_cv_joint_fde.detach())
                metrics[f"{subset_prefix}_joint_fde_to_baseline_ratio"] = (
                    subset_joint_fde
                    / subset_cv_joint_fde.clamp_min(1.0e-4)).detach()
                metrics[f"{subset_prefix}_joint_count"] = (
                    subset_joint_count[subset_name].detach())
                subset_final_root = root_error[valid & source_subset]
                subset_final_cv_root = baseline_error[valid & source_subset]
                metrics[f"{subset_prefix}_human_fde_p95_m"] = (
                    audited_quantile(subset_final_root, 0.95).detach())
                metrics[
                    f"{subset_prefix}_constant_velocity_fde_p95_m"
                ] = audited_quantile(
                    subset_final_cv_root, 0.95).detach()
            if has_gt_audit:
                losses[f"human_gt_displacement_{horizon}"] = (
                    gt_displacement_loss)
                gt_denominator = gt_final_count.clamp_min(1)
                metrics[f"h{horizon}/human_gt_fde_real_ego_m"] = (
                    gt_real_ego_final_sum / gt_denominator).detach()
                metrics[f"h{horizon}/human_gt_fde_end_to_end_m"] = (
                    gt_end_to_end_final_sum / gt_denominator).detach()
                metrics[f"h{horizon}/human_gt_displacement_fde_m"] = (
                    gt_displacement_final_sum / gt_denominator).detach()
                metrics[f"h{horizon}/perception_pelvis_error_m"] = (
                    gt_perception_sum / gt_denominator).detach()
                metrics[f"h{horizon}/human_gt_matched_count"] = (
                    gt_final_count.detach())
            clearance_required = {
                "skeleton", "joint_mask", "ego_state",
                "human_gt_id", "human_gt_match_valid",
                "priv_human_id", "priv_human_mask",
                "priv_collision_joints_episode",
                "priv_collision_joint_valid",
                "privileged_geometry_available",
            }
            if clearance_required.issubset(batch):
                # The deployed Event consumes clearance derived from imagined
                # articulated joints, not just pelvis/root.  Supervise that
                # exact rollout quantity against simulator collision geometry
                # over every 0.1 s transition up to this horizon.
                target_joint_radii = rollout["joints"].new_tensor((
                    0.14, 0.08, 0.08, 0.09, 0.09,
                    0.09, 0.09, 0.08, 0.08, 0.12,
                ))
                source_ego = self._gather_time(
                    batch["ego_state"], indices).float()
                source_skeleton = self._gather_time(
                    batch["skeleton"], indices).float()
                source_joint_mask = self._gather_time(
                    batch["joint_mask"], indices).bool()
                (
                    previous_predicted_body,
                    _,
                    predicted_joint_valid,
                    predicted_joint_radii,
                ) = self.deployable_human_collision_geometry(
                    source_skeleton[..., :3],
                    source_skeleton[..., 3:6],
                    source_joint_mask,
                )
                source_identity_id = self._gather_time(
                    batch.get("human_gt_identity_id", batch["human_gt_id"]),
                    indices,
                ).long()
                source_identity_valid = (
                    self._gather_time(
                        batch.get(
                            "human_gt_identity_valid",
                            batch["human_gt_match_valid"],
                        ),
                        indices,
                    ).bool()
                    & rollout["source_mask"]
                    & source_identity_id.ge(0)
                    & selected_window_valid[:, None]
                )
                predicted_joint_valid &= source_identity_valid[..., None]
                previous_predicted_episode = body_points_to_episode(
                    previous_predicted_body, source_ego)
                previous_predicted_relative = (
                    previous_predicted_episode
                    - source_ego[..., None, None, :3])
                previous_baseline_relative = previous_predicted_relative
                previous_baseline_valid = predicted_joint_valid
                previous_target, previous_target_valid = (
                    self._aligned_privileged_collision_geometry(
                        batch,
                        indices,
                        0,
                        source_gt_id=source_identity_id,
                        source_gt_valid=source_identity_valid,
                        source_sphere_valid=predicted_joint_valid,
                    ))
                clearance_loss_sum = rollout["joints"].new_zeros(())
                clearance_weight_sum = rollout["joints"].new_zeros(())
                clearance_error_sum = rollout["joints"].new_zeros(())
                false_safe_sum = rollout["joints"].new_zeros(())
                baseline_clearance_error_sum = rollout["joints"].new_zeros(())
                baseline_false_safe_sum = rollout["joints"].new_zeros(())
                danger_count = rollout["joints"].new_zeros(())
                valid_count = rollout["joints"].new_zeros(())
                false_safe_values: list[torch.Tensor] = []
                baseline_false_safe_values: list[torch.Tensor] = []
                collision_false_safe_values: list[torch.Tensor] = []
                baseline_collision_false_safe_values: list[torch.Tensor] = []
                collision_tail_loss_sum = rollout["joints"].new_zeros(())
                collision_tail_count = rollout["joints"].new_zeros(())
                for step in range(1, int(horizon) + 1):
                    current_ego = rollout["ego_state"][:, step - 1]
                    predicted_body, _, current_predicted_valid, _ = (
                        self.deployable_human_collision_geometry(
                            rollout["joints"][:, step - 1],
                            rollout["joint_velocity"][:, step - 1],
                            source_joint_mask,
                        ))
                    current_predicted_valid &= source_identity_valid[..., None]
                    predicted_pair_valid = (
                        predicted_joint_valid & current_predicted_valid)
                    predicted_episode = body_points_to_episode(
                        predicted_body, current_ego)
                    predicted_relative = (
                        predicted_episode
                        - current_ego[..., None, None, :3])
                    # Preserve the source Human identity through the loss.
                    # Reducing over people here lets a prediction for person
                    # B explain a PhysX contact caused by person A.  That can
                    # fit the scalar scene minimum while teaching the rollout
                    # the wrong avoidance side, even though the Actor/Event
                    # paths consume per-person geometry.
                    predicted_gap = swept_relative_point_signed_gap(
                        previous_predicted_relative,
                        predicted_relative,
                        predicted_pair_valid,
                        surface_radii_m=(0.17 + predicted_joint_radii),
                        per_human=True,
                    )
                    baseline_body, _, baseline_valid, _ = (
                        self.deployable_human_collision_geometry(
                            rollout["constant_velocity_joints"][:, step - 1],
                            rollout[
                                "constant_velocity_joint_velocity"
                            ][:, step - 1],
                            source_joint_mask,
                        ))
                    baseline_valid &= source_identity_valid[..., None]
                    baseline_pair_valid = (
                        previous_baseline_valid & baseline_valid)
                    baseline_episode = body_points_to_episode(
                        baseline_body, current_ego)
                    baseline_relative = (
                        baseline_episode
                        - current_ego[..., None, None, :3])
                    baseline_gap = swept_relative_point_signed_gap(
                        previous_baseline_relative,
                        baseline_relative,
                        baseline_pair_valid,
                        surface_radii_m=(0.17 + predicted_joint_radii),
                        per_human=True,
                    )
                    target, target_valid = (
                        self._aligned_privileged_collision_geometry(
                            batch,
                            indices,
                            step,
                            source_gt_id=source_identity_id,
                            source_gt_valid=source_identity_valid,
                            source_sphere_valid=predicted_joint_valid,
                        ))
                    target_pair_valid = previous_target_valid & target_valid
                    previous_target_relative = (
                        previous_target
                        - rollout["ego_state"][:, max(0, step - 2)][
                            ..., None, None, :3]
                        if step > 1 else
                        previous_target
                        - source_ego[..., None, None, :3]
                    )
                    target_relative = (
                        target - current_ego[..., None, None, :3])
                    target_gap = swept_relative_point_signed_gap(
                        previous_target_relative,
                        target_relative,
                        target_pair_valid,
                        surface_radii_m=(0.17 + target_joint_radii),
                        per_human=True,
                    )
                    identity_valid = target_pair_valid.any(-1)
                    identity_valid &= selected_window_valid[:, None]
                    identity_valid &= self._gather_time(
                        batch["privileged_geometry_available"],
                        indices, step).bool().reshape(-1, 1)
                    if "sequence_valid" in batch:
                        identity_valid &= self._gather_time(
                            batch["sequence_valid"],
                            indices, step).bool().reshape(-1, 1)
                    prediction = predicted_gap.clamp(-0.5, 4.0)
                    truth = target_gap.clamp(-0.5, 4.0)
                    error = prediction - truth
                    danger_weight = (
                        1.0
                        + 3.0 * truth.lt(1.0).to(error)
                        + 4.0 * truth.lt(0.35).to(error)
                    )
                    false_safe = F.relu(error - 0.05)
                    baseline_error = baseline_gap.clamp(-0.5, 4.0) - truth
                    baseline_false_safe = F.relu(baseline_error - 0.05)
                    per_row = (
                        F.smooth_l1_loss(
                            prediction, truth, reduction="none", beta=0.20)
                        + 2.0 * false_safe.square()
                    )
                    valid_weight = danger_weight * identity_valid.to(error)
                    clearance_loss_sum += (per_row * valid_weight).sum()
                    clearance_weight_sum += valid_weight.sum()
                    clearance_error_sum += (
                        error.abs() * identity_valid.to(error)).sum()
                    false_safe_sum += (
                        false_safe * identity_valid.to(error)).sum()
                    baseline_clearance_error_sum += (
                        baseline_error.abs() * identity_valid.to(error)).sum()
                    baseline_false_safe_sum += (
                        baseline_false_safe
                        * identity_valid.to(error)).sum()
                    false_safe_values.append(false_safe[identity_valid])
                    baseline_false_safe_values.append(
                        baseline_false_safe[identity_valid])
                    contact_offset = (
                        self._gather_time(
                            batch[
                                "counterfactual_collision_contact_offset_m"],
                            indices,
                            step,
                        ).float().reshape(-1, 1).expand_as(truth)
                        if "counterfactual_collision_contact_offset_m" in batch
                        else truth.new_full(truth.shape, 0.02)
                    )
                    collision_row = (
                        identity_valid & truth.le(contact_offset))
                    # Give rare true-contact false-safe errors independent
                    # group mass.  The second term is a differentiable
                    # training regret against constant velocity, not a
                    # deployment-time fallback or hard action veto.
                    collision_tail = (
                        false_safe.square()
                        + F.relu(
                            false_safe
                            - baseline_false_safe.detach()
                        ).square()
                    )
                    collision_tail_loss_sum += collision_tail[
                        collision_row].sum()
                    collision_tail_count += collision_row.sum()
                    collision_false_safe_values.append(
                        false_safe[collision_row])
                    baseline_collision_false_safe_values.append(
                        baseline_false_safe[collision_row])
                    danger_count += (
                        identity_valid & truth.lt(1.0)).to(error).sum()
                    valid_count += identity_valid.to(error).sum()
                    previous_predicted_relative = predicted_relative
                    predicted_joint_valid = current_predicted_valid
                    previous_baseline_relative = baseline_relative
                    previous_baseline_valid = baseline_valid
                    previous_target = target
                    previous_target_valid = target_valid
                    if step == int(horizon):
                        final_clearance_prediction = prediction
                        final_clearance_baseline = baseline_gap.clamp(-0.5, 4.0)
                        final_clearance_truth = truth
                        final_clearance_valid = identity_valid
                losses[f"human_joint_clearance_{horizon}"] = (
                    clearance_loss_sum / clearance_weight_sum.clamp_min(1.0))
                if self.collision_tail_regret_supervision:
                    losses[f"human_joint_collision_tail_{horizon}"] = (
                        collision_tail_loss_sum
                        / collision_tail_count.clamp_min(1.0))
                metrics[f"h{horizon}/joint_clearance_mae_m"] = (
                    clearance_error_sum / valid_count.clamp_min(1.0)).detach()
                metrics[f"h{horizon}/joint_clearance_false_safe_m"] = (
                    false_safe_sum / valid_count.clamp_min(1.0)).detach()
                metrics[
                    f"h{horizon}/constant_velocity_joint_clearance_mae_m"
                ] = (
                    baseline_clearance_error_sum
                    / valid_count.clamp_min(1.0)).detach()
                metrics[
                    f"h{horizon}/constant_velocity_joint_clearance_false_safe_m"
                ] = (
                    baseline_false_safe_sum
                    / valid_count.clamp_min(1.0)).detach()
                metrics[f"h{horizon}/joint_clearance_valid_count"] = (
                    valid_count.detach())
                metrics[f"h{horizon}/joint_clearance_danger_count"] = (
                    danger_count.detach())
                all_false_safe = torch.cat(false_safe_values)
                all_baseline_false_safe = torch.cat(
                    baseline_false_safe_values)
                all_collision_false_safe = torch.cat(
                    collision_false_safe_values)
                all_baseline_collision_false_safe = torch.cat(
                    baseline_collision_false_safe_values)
                metrics[f"h{horizon}/joint_clearance_false_safe_p95_m"] = (
                    audited_quantile(all_false_safe, 0.95).detach())
                metrics[f"h{horizon}/joint_clearance_false_safe_max_m"] = (
                    all_false_safe.amax().detach()
                    if all_false_safe.numel() else
                    rollout["joints"].new_zeros(()))
                metrics[
                    f"h{horizon}/constant_velocity_joint_clearance_false_safe_p95_m"
                ] = audited_quantile(
                    all_baseline_false_safe, 0.95).detach()
                metrics[
                    f"h{horizon}/collision_joint_clearance_false_safe_p95_m"
                ] = audited_quantile(
                    all_collision_false_safe, 0.95).detach()
                metrics[
                    f"h{horizon}/constant_velocity_collision_joint_clearance_false_safe_p95_m"
                ] = audited_quantile(
                    all_baseline_collision_false_safe, 0.95).detach()
                metrics[f"h{horizon}/collision_clearance_count"] = (
                    all_collision_false_safe.new_tensor(
                        float(all_collision_false_safe.numel())).detach())
                if self.collision_tail_regret_supervision:
                    metrics[f"h{horizon}/collision_tail_loss_m2"] = losses[
                        f"human_joint_collision_tail_{horizon}"].detach()
                for subset_name, source_subset in subset_mask.items():
                    subset_row = source_subset & final_clearance_valid
                    subset_prefix = f"h{horizon}/subset_{subset_name}"
                    subset_error = (
                        final_clearance_prediction - final_clearance_truth
                    )[subset_row]
                    subset_cv_error = (
                        final_clearance_baseline - final_clearance_truth
                    )[subset_row]
                    metrics[f"{subset_prefix}_clearance_mae_m"] = (
                        subset_error.abs().mean().detach()
                        if subset_error.numel() else
                        rollout["joints"].new_zeros(()))
                    metrics[
                        f"{subset_prefix}_constant_velocity_clearance_mae_m"
                    ] = (
                        subset_cv_error.abs().mean().detach()
                        if subset_cv_error.numel() else
                        rollout["joints"].new_zeros(()))
                    metrics[f"{subset_prefix}_clearance_false_safe_p95_m"] = (
                        audited_quantile(
                            F.relu(subset_error - 0.05), 0.95).detach())
                    metrics[
                        f"{subset_prefix}_constant_velocity_clearance_false_safe_p95_m"
                    ] = audited_quantile(
                        F.relu(subset_cv_error - 0.05), 0.95).detach()
            if not self.human_only_overshooting:
                legacy_required = {
                    "goal_position", "future_collision",
                    "future_min_human_clearance_m",
                    "future_min_human_clearance_valid",
                }
                if legacy_required.issubset(batch):
                    final_ego = rollout["ego_state"][:, -1]
                    goal_position = self._gather_time(
                        batch["goal_position"], indices, horizon).float()
                    goal = goal_features_torch(final_ego, goal_position)
                    joint, _ = self.rssm.get_joint_feat(
                        rollout["final_state"], rollout["source_mask"], goal)
                    target_action = self._gather_time(
                        batch["action"], indices, horizon + 1).float().clamp(-1.0, 1.0)
                    risk = self.action_risk(joint, target_action)
                    collision_target = self._gather_time(
                        batch["future_collision"], indices, horizon).float()
                    risk_valid = torch.ones_like(collision_target, dtype=torch.bool)
                    collision_bce = F.binary_cross_entropy_with_logits(
                        risk["collision_logit"], collision_target,
                        pos_weight=risk["collision_logit"].new_tensor(
                            self.risk_collision_positive_weight), reduction="none")
                    losses[f"risk_collision_overshoot_{horizon}"] = self._masked_mean(
                        collision_bce, risk_valid)
                    clearance = self._balanced_clearance_supervision(
                        risk["min_human_clearance_m"],
                        self._gather_time(
                            batch["future_min_human_clearance_m"], indices,
                            horizon).float(),
                        self._gather_time(
                            batch["future_min_human_clearance_valid"], indices,
                            horizon).bool() & risk_valid,
                    )
                    losses[f"risk_clearance_overshoot_{horizon}"] = clearance[
                        "regression_loss"]
                    losses[f"risk_false_safe_overshoot_{horizon}"] = clearance[
                        "false_safe_loss"]
                    losses[f"risk_false_danger_overshoot_{horizon}"] = clearance[
                        "false_danger_loss"]
        return losses, metrics

    def human_factual_rollout(
        self, states: FactorizedState, batch: dict[str, torch.Tensor],
        start_indices: torch.Tensor, horizon: int,
        *, use_recorded_ego: bool = False,
        rollout_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Shared action-aligned Human rollout used by training and evaluation."""
        state = self._gather_state_time(states, start_indices)
        human_mask = self._gather_time(batch["human_mask"], start_indices).bool()
        source_ids = self._gather_time(batch["human_ids"], start_indices)
        root = self._gather_time(batch["human_root"], start_indices).float()
        source_skeleton = self._gather_time(
            batch["skeleton"], start_indices).float()
        joint_state = source_skeleton[..., :3].clone()
        joint_velocity_state = source_skeleton[..., 3:6].clone()
        joint_mask_state = self._gather_time(
            batch["joint_mask"], start_indices).bool()
        ego = self._gather_time(batch["ego_state"], start_indices).float()
        decoded_ego = self.prediction_heads.decode_ego_state(
            self.rssm.get_branch_feats(state)["ego"])
        roots, joints, joint_velocities, baseline_roots, zero_velocity_roots, egos, survival = (
            [], [], [], [], [], [], [])
        baseline_joints, baseline_joint_velocities = [], []
        baseline_root = root.clone()
        baseline_joint = joint_state.clone()
        baseline_joint_velocity = joint_velocity_state.clone()
        zero_velocity_root = root.clone()
        zero_velocity_root[..., 3:6] = 0.0
        factual_row_valid = (
            torch.ones(
                start_indices.numel(), dtype=torch.bool,
                device=start_indices.device)
            if rollout_valid is None else
            rollout_valid.bool().reshape(-1)
        )
        if factual_row_valid.numel() != start_indices.numel():
            raise ValueError(
                "Human factual rollout validity must match flattened starts")
        for step in range(1, int(horizon) + 1):
            branch = self.rssm.get_branch_feats(state)
            action = self._gather_time(batch["action"], start_indices, step).float()
            transition_dt = (
                self._gather_time(
                    batch["dt_s"], start_indices, step).float()
                if "dt_s" in batch else
                action.new_full((action.shape[0], 1), 0.1)
            )
            transition_dt_valid = (
                torch.isfinite(transition_dt).reshape(
                    transition_dt.shape[0], -1).all(-1)
                & transition_dt.reshape(
                    transition_dt.shape[0], -1).gt(0.0).all(-1))
            if bool((factual_row_valid & ~transition_dt_valid).any()):
                raise ValueError(
                    "factual Human rollout requires positive destination dt_s")
            action = torch.where(
                factual_row_valid[:, None], action, torch.zeros_like(action))
            transition_dt = torch.where(
                transition_dt_valid[:, None], transition_dt,
                transition_dt.new_full(transition_dt.shape, 0.1))
            next_state, _ = self.rssm.img_step(state, action, human_mask)
            next_branch = self.rssm.get_branch_feats(next_state)
            next_decoded = self.prediction_heads.decode_ego_state(
                self.rssm.get_branch_feats(next_state)["ego"])
            if use_recorded_ego:
                next_ego = self._gather_time(
                    batch["ego_state"], start_indices, step).float()
            else:
                next_ego = (
                    self.anchor_ego_residual(
                        ego, decoded_ego, next_decoded, action,
                        dt_s=transition_dt,
                        velocity_response=self.analytic_ego_velocity_response)
                    if self.uses_internal_action_adapter
                    else self.anchor_ego_displacement(
                        ego, decoded_ego, next_decoded)
                )
            human = self.prediction_heads.human(
                next_branch["human"], root, current_ego=ego,
                next_ego=next_ego.detach(),
                current_joints=joint_state,
                current_joint_velocity=joint_velocity_state,
                current_joint_mask=joint_mask_state,
                lifecycle_feat=branch["human"],
                dt_s=transition_dt)
            root = torch.cat((
                human["root"], human["root_velocity"], root[..., 6:]), -1)
            baseline_velocity_source = (
                baseline_root[..., 3:6]
                if baseline_root.shape[-1] >= 6
                else torch.zeros_like(baseline_root[..., :3])
            )
            baseline_position, baseline_velocity = analytic_relative_human_step(
                baseline_root[..., :3], baseline_velocity_source,
                ego, next_ego.detach(), dt_s=transition_dt,
            )
            baseline_root = torch.cat((
                baseline_position, baseline_velocity,
                baseline_root[..., 6:] if baseline_root.shape[-1] >= 6
                else baseline_root[..., 3:]), -1)
            baseline_shape = baseline_joint.shape
            baseline_joint, baseline_joint_velocity = (
                analytic_relative_human_step(
                    baseline_joint.reshape(baseline_shape[0], -1, 3),
                    baseline_joint_velocity.reshape(
                        baseline_shape[0], -1, 3),
                    ego,
                    next_ego.detach(),
                    dt_s=transition_dt,
                ))
            baseline_joint = baseline_joint.reshape(baseline_shape)
            baseline_joint_velocity = baseline_joint_velocity.reshape(
                baseline_shape)
            zero_position, zero_velocity = analytic_relative_human_step(
                zero_velocity_root[..., :3],
                torch.zeros_like(zero_velocity_root[..., :3]),
                ego, next_ego.detach(), dt_s=transition_dt,
            )
            zero_velocity_root = torch.cat((
                zero_position, zero_velocity,
                zero_velocity_root[..., 6:]
                if zero_velocity_root.shape[-1] >= 6
                else zero_velocity_root[..., 3:]), -1)
            roots.append(root)
            joints.append(human["joints"])
            joint_velocities.append(human["joint_velocity"])
            joint_state = human["joints"]
            joint_velocity_state = human["joint_velocity"]
            baseline_roots.append(baseline_root)
            baseline_joints.append(baseline_joint)
            baseline_joint_velocities.append(baseline_joint_velocity)
            zero_velocity_roots.append(zero_velocity_root)
            egos.append(next_ego)
            survival.append(torch.sigmoid(human["survival_logit"]))
            # Human overshooting must not modify the already audited Ego
            # dynamics. Coupling still conditions Human priors on Ego state.
            next_state = {
                "ego": {key: value.detach() for key, value in next_state["ego"].items()},
                "human": next_state["human"],
            }
            state, ego, decoded_ego = next_state, next_ego.detach(), next_decoded.detach()
        return {
            "root": torch.stack(roots, dim=1),
            "joints": torch.stack(joints, dim=1),
            "joint_velocity": torch.stack(joint_velocities, dim=1),
            "constant_velocity_root": torch.stack(baseline_roots, dim=1),
            "constant_velocity_joints": torch.stack(
                baseline_joints, dim=1),
            "constant_velocity_joint_velocity": torch.stack(
                baseline_joint_velocities, dim=1),
            "zero_velocity_root": torch.stack(zero_velocity_roots, dim=1),
            "ego_state": torch.stack(egos, dim=1),
            "survival_probability": torch.stack(survival, dim=1),
            "source_ids": source_ids,
            "source_mask": human_mask,
            "final_state": state,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        initial: FactorizedState | None = None,
        imag_horizon: int = 15,
        world_model_only: bool = False,
    ):
        """Complete training forward used by DistributedDataParallel.

        DDP must observe the forward that creates every optimized loss tensor;
        calling methods directly on ``ddp.module`` would bypass reducer setup
        and silently leave gradients unsynchronized.
        """
        world_losses, states, aux = self.world_model_loss(batch, initial)
        prediction_losses, predictions = self.prediction_loss(states, batch)
        if world_model_only:
            actor_losses: dict[str, torch.Tensor] = {}
            imag_metrics: dict[str, torch.Tensor] = {}
        else:
            actor_losses, imag_metrics = self.actor_critic_loss(
                states, aux, batch, int(imag_horizon))
        return (
            world_losses,
            states,
            aux,
            prediction_losses,
            predictions,
            actor_losses,
            imag_metrics,
        )

    @staticmethod
    def last_state(states: FactorizedState) -> FactorizedState:
        return {
            branch: {key: value[:, -1] for key, value in values.items()}
            for branch, values in states.items()
        }

    def imagine(
        self,
        initial: FactorizedState,
        horizon: int,
        human_mask: torch.Tensor,
        *,
        goal_position: torch.Tensor,
        ego_state: torch.Tensor | None = None,
        sample: bool = True,
    ):
        return self._imagine_goal_conditioned(
            initial, horizon, human_mask, goal_position,
            initial_ego_state=ego_state, sample=sample,
        )

    @staticmethod
    def _r2_planner_goal_aligned_action(
        policy_action: torch.Tensor,
        ego_state: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate horizontal speed onto the deployable body-goal direction."""
        goal_xy = goal_features_torch(
            ego_state.float(), goal_position.float())[..., :2]
        goal_norm = torch.linalg.vector_norm(goal_xy, dim=-1, keepdim=True)
        goal_unit = goal_xy / goal_norm.clamp_min(1.0e-6)
        speed = torch.linalg.vector_norm(
            policy_action[..., :2].float(), dim=-1, keepdim=True)
        aligned = goal_unit * speed
        bound_scale = (
            1.0 / aligned.abs().amax(-1, keepdim=True).clamp_min(1.0)
        ).clamp(max=1.0)
        aligned = aligned * bound_scale
        aligned = torch.where(
            goal_norm.gt(1.0e-6), aligned,
            policy_action[..., :2].float())
        return torch.cat((
            aligned.to(policy_action), policy_action[..., 2:3],
        ), dim=-1)

    @classmethod
    def _r2_structured_candidate_policy_actions(
        cls,
        actor_policy_action: torch.Tensor,
        ego_state: torch.Tensor,
        goal_position: torch.Tensor,
        *,
        forward_delta: float,
        lateral_delta: float,
    ) -> torch.Tensor:
        """Build ten distinct control intents without increasing WM batch.

        The former 3x3 grid differed only by one small offset around the Actor
        mode.  The stateful action smoother compressed that grid by roughly
        four times on the first executable tick.  These candidates instead
        use absolute, sustained targets for braking, reversing and crossing.
        They still pass through the exact deployed smoother and action adapter
        before support checks or imagination, so no actuator shortcut is
        introduced.
        """
        centre = actor_policy_action.float().clamp(-1.0, 1.0)
        if centre.ndim != 2 or centre.shape[-1] != 3:
            raise ValueError("structured Actor policy action must be [B,3]")
        if ego_state.shape != (centre.shape[0], 14):
            raise ValueError("structured candidate Ego state must be [B,14]")
        if goal_position.shape != (centre.shape[0], 3):
            raise ValueError("structured candidate goal must be [B,3]")
        if not 0.0 < float(forward_delta) <= 1.0:
            raise ValueError("structured forward delta must be within (0,1]")
        if not 0.0 < float(lateral_delta) <= 1.0:
            raise ValueError("structured lateral delta must be within (0,1]")

        goal_aligned = cls._r2_planner_goal_aligned_action(
            centre, ego_state, goal_position)
        yaw = centre[:, 2]
        zero = torch.zeros_like(yaw)
        # Twice the old local radius gives a materially different sustained
        # lateral intent while remaining bounded and checkpoint-compatible.
        lateral = torch.full_like(
            yaw, min(1.0, 2.0 * float(lateral_delta)))
        reverse = torch.full_like(yaw, -float(forward_delta))
        accelerate = (
            centre[:, 0] + 2.0 * float(forward_delta)
        ).clamp(max=1.0)

        def action(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.stack((x, y, yaw), dim=-1)

        candidates = torch.stack((
            centre,
            goal_aligned,
            action(zero, zero),
            action(reverse, zero),
            action(centre[:, 0], lateral),
            action(centre[:, 0], -lateral),
            action(zero, lateral),
            action(zero, -lateral),
            action(accelerate, lateral),
            action(accelerate, -lateral),
        ), dim=1).clamp(-1.0, 1.0)
        if candidates.shape[1] != len(R2_STRUCTURED_CANDIDATE_NAMES):
            raise AssertionError("structured R2 candidate count changed")
        return candidates

    @staticmethod
    def _r2_planner_behavior_descriptor(
        ego_state: torch.Tensor,
        goal_position: torch.Tensor,
        human_root: torch.Tensor,
        human_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match the deployable descriptor frozen into the training support."""
        ego = ego_state.float()
        goal = goal_position.float()
        root = human_root.float()[..., :3]
        mask = human_mask.bool()
        distance = torch.linalg.vector_norm(root, dim=-1).masked_fill(
            ~mask, torch.inf)
        nearest_index = distance.argmin(-1)
        nearest = root.gather(
            -2,
            nearest_index[..., None, None].expand(
                *nearest_index.shape, 1, 3),
        ).squeeze(-2)
        nearest = torch.where(
            mask.any(-1)[..., None], nearest, torch.zeros_like(nearest))
        relative_goal = goal - ego[..., :3]
        return torch.cat((
            ego[..., 3:6] / ego.new_tensor((2.15, 2.15, 1.0)),
            ego[..., 10:14],
            relative_goal / 10.0,
            nearest / 6.0,
            mask.float().sum(-1, keepdim=True) / max(1, mask.shape[-1]),
        ), -1)

    @staticmethod
    def _r2_planner_candidate_support(
        descriptor: torch.Tensor,
        candidate_action: torch.Tensor,
        support: dict[str, Any],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        required = (
            "descriptors", "actions", "mean", "std", "state_radius",
            "action_radius", "neighbors",
        )
        if any(name not in support for name in required):
            raise ValueError("R2 planner behavior support is incomplete")
        descriptors = support["descriptors"].to(descriptor).float()
        actions = support["actions"].to(candidate_action).float()
        mean = support["mean"].to(descriptor).float()
        std = support["std"].to(descriptor).float()
        normalized = (descriptor.float() - mean) / std.clamp_min(1.0e-3)
        distance = torch.cdist(normalized, descriptors)
        state_distance = distance.amin(-1)
        neighbors = min(int(support["neighbors"]), distance.shape[-1])
        indices = distance.topk(neighbors, largest=False).indices
        local_actions = actions[indices]
        action_distance = (
            candidate_action.float()[:, :, None, :]
            - local_actions[:, None, :, :]
        ).abs().amax(-1).amin(-1)
        state_supported = (
            state_distance <= float(support["state_radius"]))
        action_supported = (
            action_distance <= float(support["action_radius"]))
        event_supported = state_supported[:, None] & action_supported
        return (
            event_supported, action_supported, action_distance,
            state_distance,
        )

    @torch.no_grad()
    def plan_pure_r2_candidates(
        self,
        state: FactorizedState,
        human_mask: torch.Tensor,
        actor_policy_action: torch.Tensor,
        previous_applied_action: torch.Tensor,
        ego_state: torch.Tensor,
        goal_position: torch.Tensor,
        human_root: torch.Tensor,
        human_quality: torch.Tensor,
        human_joint_mask: torch.Tensor,
        human_joints_body: torch.Tensor,
        behavior_support: dict[str, Any],
        *,
        horizon_steps: int = 5,
        forward_delta: float = 0.25,
        lateral_delta: float = 0.35,
        risk_tolerance: float = 0.002,
        safe_risk_cap: float = 0.08,
        safe_risk_margin: float = 0.02,
        minimum_risk_improvement: float = 0.01,
        minimum_progress_improvement: float = 0.005,
        commitment_steps: int = 8,
        boundary_position_world_xy: torch.Tensor | None = None,
        boundary_yaw_rad: torch.Tensor | None = None,
        boundary_flight_bounds_xy: torch.Tensor | None = None,
        boundary_buffer_m: float = 0.35,
        return_rollout_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Receding-horizon v10 policy improvement without ``action_risk``.

        The Actor supplies the nominal proposal. Ten structured policy intents
        are smoothed, converted to deployed actions, rolled through the frozen
        coupled priors, and ranked by learned multi-step Event probability.
        Static flight-boundary risk remains an independent hard task
        constraint; no swept-human geometry or constant-velocity CPA is used
        to rank candidates. Analytic predicted goal progress breaks Event ties.
        """
        if not self.uses_internal_action_adapter:
            raise RuntimeError("pure R2 candidate planning requires ActionAdapter")
        if self.transition_event is None or not (
            self.transition_event.explicit_human_geometry
        ):
            raise RuntimeError(
                "pure R2 candidate planning requires explicit Human geometry")
        batch = actor_policy_action.shape[0]
        if actor_policy_action.shape != (batch, 3):
            raise ValueError("Actor policy action must be [B,3]")
        if previous_applied_action.shape != (batch, 4):
            raise ValueError("previous applied action must be [B,4]")
        if ego_state.shape != (batch, 14):
            raise ValueError("planner Ego state must be [B,14]")
        if goal_position.shape != (batch, 3):
            raise ValueError("planner goal position must be [B,3]")
        if int(horizon_steps) <= 0:
            raise ValueError("planner horizon must be positive")
        if int(commitment_steps) <= 0:
            raise ValueError("planner commitment must be positive")
        if min(
            float(forward_delta), float(lateral_delta),
            float(boundary_buffer_m),
        ) <= 0.0:
            raise ValueError("planner deltas/boundary buffer must be positive")
        if min(
            float(risk_tolerance), float(minimum_risk_improvement),
            float(minimum_progress_improvement), float(safe_risk_margin),
        ) < 0.0:
            raise ValueError("planner tolerances must be non-negative")
        if not 0.0 < float(safe_risk_cap) < 1.0:
            raise ValueError("planner safe-risk cap must be within (0,1)")

        centre = actor_policy_action.float().clamp(-1.0, 1.0)
        candidates = self._r2_structured_candidate_policy_actions(
            centre, ego_state, goal_position,
            forward_delta=forward_delta,
            lateral_delta=lateral_delta,
        )
        candidate_count = candidates.shape[1]

        previous_policy = self.policy_from_applied_action(
            previous_applied_action.float().clamp(-1.0, 1.0))
        if self.action_smoother is None:
            smoothed = candidates
        else:
            smoothed = self.action_smoother(
                candidates,
                previous_policy[:, None].expand_as(candidates),
            )
        expanded_ego = ego_state[:, None].expand(
            -1, candidate_count, -1)
        applied = self.applied_from_policy_action(
            smoothed.reshape(batch * candidate_count, -1),
            expanded_ego.reshape(batch * candidate_count, -1),
        ).reshape(batch, candidate_count, -1)
        candidate_override = candidates.reshape(batch * candidate_count, -1)
        # Candidate zero remains the evolving Actor policy after the first
        # applied action.  Every other intent is sustained through the
        # commitment window, producing a true temporal brake/reverse/crossing
        # sequence under the same smoother used online.
        candidate_override_mask = torch.ones(
            batch, candidate_count, 1, dtype=torch.bool,
            device=candidates.device)
        candidate_override_mask[:, 0] = False

        descriptor = self._r2_planner_behavior_descriptor(
            ego_state, goal_position, human_root, human_mask)
        event_supported, action_supported, action_distance, state_distance = (
            self._r2_planner_candidate_support(
                descriptor, smoothed, behavior_support))
        state_supported = event_supported.any(-1) | (
            state_distance <= float(behavior_support["state_radius"]))

        repeated_state = self._repeat_state_candidates(
            state, candidate_count)

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand(
                -1, candidate_count, *value.shape[1:]
            ).reshape(batch * candidate_count, *value.shape[1:])

        initial_clearance = analytic_joint_surface_gap(
            human_joints_body.float(), human_joint_mask.bool(),
            surface_radius_m=self.lateral_candidate_surface_radius_m,
            maximum_gap_m=6.0,
            per_human=True,
        )
        _, imagined = self._imagine_goal_conditioned(
            repeated_state,
            int(horizon_steps) + 1,
            repeat(human_mask),
            repeat(goal_position),
            sample=False,
            collect_states=False,
            first_applied_action=applied.reshape(
                batch * candidate_count, -1),
            policy_action_override=candidate_override,
            policy_action_override_mask=(
                candidate_override_mask.reshape(
                    batch * candidate_count, 1)),
            policy_action_override_steps=min(
                int(commitment_steps), int(horizon_steps)),
            initial_ego_state=repeat(ego_state),
            initial_previous_applied_action=repeat(previous_applied_action),
            initial_human_root=repeat(human_root),
            initial_human_quality=repeat(human_quality),
            initial_human_joint_clearance=repeat(initial_clearance),
            initial_human_joint_mask=repeat(human_joint_mask),
            initial_human_joints_body=repeat(human_joints_body),
        )
        feature = imagined["joint_feat"]
        rollout_mask = repeat(human_mask)[:, None].expand(
            -1, feature.shape[1] - 1, -1)
        event = self.transition_event.forward_human_hazard(
            feature[:, :-1],
            imagined["action"][:, :-1],
            ego_feature=imagined["ego_branch_feature"][:, :-1],
            human_feature=imagined["human_branch_feature"][:, :-1],
            human_root=imagined["human_root"][:, :-1],
            human_quality=imagined["human_observation_quality"][:, :-1],
            human_mask=rollout_mask,
            human_presence=imagined["human_presence"][:, :-1],
            human_joint_clearance=imagined[
                "human_joint_clearance"][:, :-1],
        )
        learned_hazard = event["human_collision_probability"].squeeze(-1)
        learned_risk = (
            1.0 - torch.prod(1.0 - learned_hazard.float(), dim=1)
        )
        human_risk = learned_risk

        progress = analytic_progress_reward(
            imagined["ego_state"],
            imagined["goal_position"],
            dt_s=0.1,
            progress_weight_per_m=2.0,
            max_progress_speed_mps=3.0,
        )[:, 1:].sum(1).squeeze(-1)

        boundary_available = all(value is not None for value in (
            boundary_position_world_xy,
            boundary_yaw_rad,
            boundary_flight_bounds_xy,
        ))
        if boundary_available:
            assert boundary_position_world_xy is not None
            assert boundary_yaw_rad is not None
            assert boundary_flight_bounds_xy is not None
            position = repeat(boundary_position_world_xy.float())
            yaw_world = repeat(boundary_yaw_rad.float().reshape(batch, 1))[
                :, 0]
            bounds = repeat(boundary_flight_bounds_xy.float())
            source_ego = repeat(ego_state)
            relative_yaw = torch.atan2(
                source_ego[:, 12], source_ego[:, 13])
            episode_yaw_world = yaw_world - relative_yaw
            cosine = torch.cos(episode_yaw_world)[:, None]
            sine = torch.sin(episode_yaw_world)[:, None]
            delta = imagined["ego_state"][:, 1:, :2] - source_ego[:, None, :2]
            future_x = position[:, None, 0] + (
                cosine * delta[..., 0] - sine * delta[..., 1])
            future_y = position[:, None, 1] + (
                sine * delta[..., 0] + cosine * delta[..., 1])
            inward = torch.stack((
                future_x - bounds[:, None, 0],
                bounds[:, None, 1] - future_x,
                future_y - bounds[:, None, 2],
                bounds[:, None, 3] - future_y,
            ), dim=-1)
            minimum_boundary_clearance = inward.amin(dim=(-1, -2))
            boundary_risk = torch.sigmoid((
                float(boundary_buffer_m) - inward.amin(-1)
            ) / 0.05).amax(-1)
        else:
            boundary_risk = human_risk.new_zeros(human_risk.shape)
            minimum_boundary_clearance = human_risk.new_full(
                human_risk.shape, torch.inf)
        # Event is deliberately the human-safety selector both inside and
        # outside the frozen Replay state radius. The radius remains visible
        # as a diagnostic and local action support is still enforced, but it
        # no longer switches the planner to a hand-written geometry fallback.
        total_risk = torch.maximum(human_risk, boundary_risk)
        selection_cost = total_risk

        risk = selection_cost.reshape(batch, candidate_count)
        candidate_progress = progress.reshape(batch, candidate_count)
        supported = action_supported
        masked_risk = risk.masked_fill(~supported, torch.inf)
        available = supported.any(-1)
        minimum_risk = masked_risk.amin(-1)
        # In calibrated low-risk states, Event is a feasibility constraint:
        # candidates must satisfy both the absolute h15 risk budget and a
        # calibrated margin above the safest supported option before they may
        # compete on goal progress.  The relative guard prevents a 0.078-risk
        # action from being treated as equivalent to a 0.045-risk action just
        # because both happen to lie below the absolute cap.
        within_safe_budget = (
            supported
            & risk.le(float(safe_risk_cap))
            & risk.le(minimum_risk[:, None] + float(safe_risk_margin))
        )
        safe_candidate_available = within_safe_budget.any(-1)
        strict_minimum_risk_band = supported & risk.le(
            minimum_risk[:, None] + float(risk_tolerance))
        eligible = torch.where(
            safe_candidate_available[:, None],
            within_safe_budget,
            strict_minimum_risk_band,
        )
        masked_progress = candidate_progress.masked_fill(
            ~eligible, -torch.inf)
        proposed_index = masked_progress.argmax(-1)
        row = torch.arange(batch, device=risk.device)
        proposed_risk = risk[row, proposed_index]
        proposed_progress = candidate_progress[row, proposed_index]
        base_risk = risk[:, 0]
        base_progress = candidate_progress[:, 0]
        base_supported = supported[:, 0]
        base_in_safe_band = base_supported & eligible[:, 0]
        base_within_safe_risk_budget = (
            base_supported & base_risk.le(float(safe_risk_cap)))
        risk_improvement = base_risk - proposed_risk
        progress_improvement = proposed_progress - base_progress
        intervene = available & proposed_index.ne(0) & (
            ~base_in_safe_band
            | progress_improvement.ge(
                float(minimum_progress_improvement))
        )
        selected_index = torch.where(
            intervene, proposed_index, torch.zeros_like(proposed_index))
        selected_policy = candidates[row, selected_index]
        selected_smoothed = smoothed[row, selected_index]
        selected_applied = applied[row, selected_index]
        result = {
            "candidate_scheme": "structured_temporal_v2",
            "candidate_names": R2_STRUCTURED_CANDIDATE_NAMES,
            "action": selected_applied,
            "policy_action": selected_policy,
            "smoothed_policy_action": selected_smoothed,
            "actor_action": applied[:, 0],
            "actor_policy_action": candidates[:, 0],
            "actor_smoothed_policy_action": smoothed[:, 0],
            "candidate_policy_action": candidates,
            "candidate_smoothed_policy_action": smoothed,
            "candidate_action": applied,
            "candidate_supported": supported,
            "candidate_event_supported": event_supported,
            "state_supported": state_supported,
            "candidate_action_support_distance": action_distance,
            "state_support_distance": state_distance,
            "candidate_risk": risk,
            "candidate_fused_risk": total_risk.reshape(
                batch, candidate_count),
            "candidate_human_risk": human_risk.reshape(
                batch, candidate_count),
            "candidate_learned_risk": learned_risk.reshape(
                batch, candidate_count),
            "candidate_boundary_risk": boundary_risk.reshape(
                batch, candidate_count),
            "candidate_progress": candidate_progress,
            "candidate_minimum_boundary_clearance_m": (
                minimum_boundary_clearance.reshape(batch, candidate_count)),
            "boundary_available": risk.new_full(
                (batch,), float(boundary_available)),
            "selected_index": selected_index,
            "proposed_index": proposed_index,
            "intervened": intervene,
            "base_supported": base_supported,
            "base_in_safe_band": base_in_safe_band,
            "base_within_safe_risk_budget": (
                base_within_safe_risk_budget),
            "available": available,
            "safe_candidate_available": safe_candidate_available,
            "candidate_within_safe_risk_budget": within_safe_budget,
            "safe_risk_cap": risk.new_full(
                (batch,), float(safe_risk_cap)),
            "safe_risk_margin": risk.new_full(
                (batch,), float(safe_risk_margin)),
            "risk_improvement": risk_improvement,
            "progress_improvement": progress_improvement,
        }
        if return_rollout_diagnostics:
            # Keep high-volume rollout tensors out of the live JSON protocol.
            # They are exposed only to in-process replay diagnostics so a
            # late Event warning can be attributed to Human prediction versus
            # Event classification without introducing geometric selection.
            result.update({
                "diagnostic_origin_human_root": human_root,
                "diagnostic_origin_human_joint_clearance": (
                    initial_clearance),
                "candidate_event_step_probability": learned_hazard.reshape(
                    batch, candidate_count, int(horizon_steps)),
                "candidate_predicted_human_root": imagined[
                    "human_root"][:, 1:int(horizon_steps) + 1].reshape(
                        batch, candidate_count, int(horizon_steps),
                        *human_root.shape[1:]),
                "candidate_predicted_ego_state": imagined[
                    "ego_state"][:, 1:int(horizon_steps) + 1].reshape(
                        batch, candidate_count, int(horizon_steps),
                        ego_state.shape[-1]),
                "candidate_predicted_human_joint_clearance": imagined[
                    "human_joint_clearance"][
                        :, 1:int(horizon_steps) + 1].reshape(
                            batch, candidate_count, int(horizon_steps),
                            *imagined["human_joint_clearance"].shape[2:]),
            })
        return result

    @torch.no_grad()
    def shadow_imagination(
        self,
        state: FactorizedState,
        human_mask: torch.Tensor,
        goal_position: torch.Tensor,
        ego_state: torch.Tensor,
        actor_action: torch.Tensor,
        *,
        horizon_steps: int = 15,
        human_root: torch.Tensor | None = None,
        human_quality: torch.Tensor | None = None,
        human_joint_mask: torch.Tensor | None = None,
        human_joints_body: torch.Tensor | None = None,
        geometry_warning_buffer_m: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Predict a deterministic future without changing the live action.

        The returned rollout is diagnostic only.  It starts from the same
        posterior latent consumed by the online Actor, advances the coupled
        Ego/Human priors with the Actor distribution mode, and exposes the
        learned reward, value, continuation, and action-risk predictions.
        Future index zero is one RSSM transition after ``actor_action``.
        """
        horizon_steps = int(horizon_steps)
        if horizon_steps <= 0:
            raise ValueError("shadow horizon_steps must be positive")
        if ego_state.ndim != 2 or ego_state.shape[-1] != 14:
            raise ValueError("shadow ego_state must be [B,14]")
        if goal_position.shape != (ego_state.shape[0], 3):
            raise ValueError("shadow goal_position must be [B,3]")
        if actor_action.ndim != 2 or actor_action.shape[0] != ego_state.shape[0]:
            raise ValueError("shadow actor_action must be [B,A]")

        explicit_geometry = (
            human_root is not None
            and human_quality is not None
            and human_joint_mask is not None
            and human_joints_body is not None
        )
        if any(value is not None for value in (
            human_root, human_quality, human_joint_mask, human_joints_body,
        )) and not explicit_geometry:
            raise ValueError(
                "shadow explicit Human geometry inputs must be supplied "
                "together")
        if not explicit_geometry:
            raise RuntimeError(
                "shadow imagination requires explicit Human geometry; "
                "the untrained legacy action_risk fallback is disabled")
        if float(geometry_warning_buffer_m) < 0.0:
            raise ValueError(
                "shadow geometry warning buffer must be non-negative")
        initial_human_clearance = None
        if explicit_geometry:
            assert human_joint_mask is not None
            assert human_joints_body is not None
            initial_human_clearance = analytic_joint_surface_gap(
                human_joints_body.float(),
                human_joint_mask.bool(),
                surface_radius_m=self.lateral_candidate_surface_radius_m,
                maximum_gap_m=6.0,
                per_human=True,
            )

        # Include the current posterior at index zero so the first transition
        # is driven by the exact action that the controller will execute.
        # The explicit geometry arguments are the same deployable state used
        # by the v17 Actor/Critic objective.  Without them, ``decision_feat``
        # degenerates to the legacy joint feature and makes a trained Critic
        # look almost constant online.
        _, imagined = self._imagine_goal_conditioned(
            state,
            horizon_steps + 1,
            human_mask,
            goal_position,
            initial_ego_state=ego_state,
            sample=False,
            collect_states=False,
            first_applied_action=actor_action,
            initial_human_root=human_root,
            initial_human_quality=human_quality,
            initial_human_joint_clearance=initial_human_clearance,
            initial_human_joint_mask=human_joint_mask,
            initial_human_joints_body=human_joints_body,
        )
        feature = imagined["joint_feat"]
        decision_feature = imagined["decision_feat"]
        rollout_action = imagined["action"]
        future_feature = feature[:, 1:]
        future_decision_feature = decision_feature[:, 1:]
        future_action = rollout_action[:, 1:]
        assert self.transition_event is not None
        rollout_human_mask = human_mask[:, None].expand(
            -1, feature.shape[1], -1)
        event = self.transition_event.forward_human_hazard(
            feature,
            rollout_action,
            ego_feature=imagined["ego_branch_feature"],
            human_feature=imagined["human_branch_feature"],
            human_root=imagined["human_root"],
            human_quality=imagined["human_observation_quality"],
            human_mask=rollout_human_mask,
            human_presence=imagined["human_presence"],
            human_joint_clearance=imagined["human_joint_clearance"],
        )
        quality = imagined["human_observation_quality"].float()
        presence = imagined["human_presence"].float()
        active = rollout_human_mask & presence.gt(0.05)
        uncertainty_margin = (
            0.02
            + quality[..., 3].clamp_min(0.0)
            * quality[..., 5].clamp_min(0.0)
        ).clamp(max=0.35)
        conservative_clearance = (
            imagined["human_joint_clearance"].float()
            - uncertainty_margin
        ).masked_fill(~active, float("inf"))
        minimum_clearance = conservative_clearance.amin(
            dim=-1, keepdim=True)
        geometry_hazard = torch.sigmoid((
            float(geometry_warning_buffer_m) - minimum_clearance
        ) / 0.05).masked_fill(
            ~active.any(dim=-1, keepdim=True), 0.0)
        learned_hazard = event["human_collision_probability"]
        fused_hazard = torch.maximum(learned_hazard, geometry_hazard)
        risk_details = {
            "collision_probability": fused_hazard,
            "learned_collision_probability": learned_hazard,
            "geometry_collision_probability": geometry_hazard,
            "min_human_clearance_m": minimum_clearance,
            "raw_min_human_clearance_m": imagined[
                "human_joint_clearance"].float().masked_fill(
                    ~active, float("inf")).amin(dim=-1, keepdim=True),
            "event": event,
        }

        future_reward = self.policy_reward(future_feature)
        future_value = self._dist_mode(self.value(future_decision_feature))
        origin_value = self._dist_mode(self.value(decision_feature[:, 0]))
        future_continue = self.cont(future_feature).mean
        future_ego = imagined["ego_state"][:, 1:]
        future_goal_distance = torch.linalg.vector_norm(
            goal_position[:, None, :] - future_ego[..., :3], dim=-1)

        result = {
            "future_steps": torch.arange(
                1, horizon_steps + 1, device=ego_state.device),
            "ego_state": future_ego,
            "goal_distance_m": future_goal_distance,
            "action": future_action,
            "collision_probability": risk_details[
                "collision_probability"][:, 1:].squeeze(-1),
            "learned_collision_probability": risk_details[
                "learned_collision_probability"][:, 1:].squeeze(-1),
            "geometry_collision_probability": risk_details[
                "geometry_collision_probability"][:, 1:].squeeze(-1),
            "predicted_min_human_clearance_m": risk_details[
                "min_human_clearance_m"][:, 1:].squeeze(-1),
            "raw_min_human_clearance_m": risk_details[
                "raw_min_human_clearance_m"][:, 1:].squeeze(-1),
            "reward": future_reward.squeeze(-1),
            "critic_value": future_value.squeeze(-1),
            "continuation_probability": future_continue.squeeze(-1),
            "origin_collision_probability": risk_details[
                "collision_probability"][:, 0].squeeze(-1),
            "origin_learned_collision_probability": risk_details[
                "learned_collision_probability"][:, 0].squeeze(-1),
            "origin_geometry_collision_probability": risk_details[
                "geometry_collision_probability"][:, 0].squeeze(-1),
            "origin_action": rollout_action[:, 0],
            "origin_predicted_min_human_clearance_m": risk_details[
                "min_human_clearance_m"][:, 0].squeeze(-1),
            "origin_raw_min_human_clearance_m": risk_details[
                "raw_min_human_clearance_m"][:, 0].squeeze(-1),
            "origin_critic_value": origin_value.squeeze(-1),
        }
        event = risk_details["event"]
        if event is not None:
            # ``forward_human_hazard`` is the binary pure-R2 deploy contract;
            # unlike the legacy multi-event head it does not expose the
            # generic ``probability``/``continue_probability`` keys.  Keep a
            # compact two-column diagnostic for existing response consumers:
            # [human_survival, human_collision].
            human_survival = event["human_survival_probability"]
            human_collision = event["human_collision_probability"]
            result.update({
                "transition_event_probability": torch.cat((
                    human_survival[:, 1:], human_collision[:, 1:],
                ), dim=-1),
                "one_step_human_hazard": human_collision[
                    :, 1:].squeeze(-1),
                "one_step_continue_probability": human_survival[
                    :, 1:].squeeze(-1),
            })
        return result

    def task_predictions(self, joint_feat: torch.Tensor) -> dict[str, Any]:
        """Shared task heads all read the same joint latent feature."""
        return {
            "actor": self.actor(joint_feat),
            "value": self.value(joint_feat),
            "reward": self.reward(joint_feat),
            "continue": self.cont(joint_feat),
        }

    def policy_from_applied_action(
        self, applied_action: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_adapter is None:
            return applied_action
        return self.action_adapter.policy_from_applied(applied_action)

    def applied_from_policy_action(
        self, policy_action: torch.Tensor, ego_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_adapter is None:
            return policy_action
        return self.action_adapter(policy_action, ego_state)

    @torch.no_grad()
    def update_safety_constraint(
        self,
        human_collision_rate: float,
        *,
        beta: float = 0.8,
        budget: float = 0.20,
        step_size: float = 0.20,
        minimum: float = 0.5,
        maximum: float = 5.0,
    ) -> float:
        """Update the bounded dual variable from deterministic evaluation."""
        rate = float(human_collision_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("human collision rate must lie in [0, 1]")
        if not 0.0 <= beta < 1.0:
            raise ValueError("EWMA beta must lie in [0, 1)")
        self.safety_cost_ewma.mul_(beta).add_((1.0 - beta) * rate)
        updated = (
            self.safety_lambda
            + float(step_size) * (self.safety_cost_ewma - float(budget)))
        self.safety_lambda.copy_(updated.clamp(float(minimum), float(maximum)))
        self.safety_lambda_update_count.add_(1)
        return float(self.safety_lambda)

    def initial_agent_state(self, batch_size: int, max_people: int, action_dim: int) -> dict[str, Any]:
        return {
            "latent": self.rssm.initial(batch_size, max_people),
            "prev_action": torch.zeros(batch_size, int(action_dim), device=next(self.parameters()).device),
            # The transition into the next observation may only couple Humans
            # visible in this source observation.  Keeping the mask in recurrent
            # agent state prevents destination-frame visibility look-ahead.
            "previous_human_mask": torch.zeros(
                batch_size, int(max_people), dtype=torch.bool,
                device=next(self.parameters()).device),
            "smoothed_policy_action": torch.zeros(
                batch_size,
                int(self.action_smoother.action_dim)
                if self.action_smoother is not None else int(action_dim),
                device=next(self.parameters()).device,
            ),
        }

    @staticmethod
    def replay_state_fields(state: FactorizedState) -> dict[str, torch.Tensor]:
        """Flatten a structured latent for TensorDict/ReplayBuffer storage."""
        return {
            "ego_stoch": state["ego"]["stoch"], "ego_deter": state["ego"]["deter"],
            "human_stoch": state["human"]["stoch"], "human_deter": state["human"]["deter"],
        }

    @staticmethod
    def state_from_replay_fields(fields: dict[str, torch.Tensor]) -> FactorizedState:
        return {
            "ego": {"stoch": fields["ego_stoch"], "deter": fields["ego_deter"]},
            "human": {"stoch": fields["human_stoch"], "deter": fields["human_deter"]},
        }

    @torch.no_grad()
    def act_step(
        self,
        observation: dict[str, torch.Tensor],
        agent_state: dict[str, Any],
        *,
        evaluation: bool = False,
        use_planner: bool = False,
        exploration_scale: float = 1.0,
        exploration_delta_limits: torch.Tensor | None = None,
        sample_actor_distribution: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """One online posterior update followed by one joint policy action.

        Online observations may be ``[B,...]`` or ``[B,T,...]``.  The human
        encoder consumes the causal pose history, while the RSSM advances only
        once from the newest observation in that history.
        """
        encoded = self.encoder(observation)
        embeds = self._embeddings(encoded)
        if embeds["ego"].ndim == 3:
            embeds = {key: value[:, -1] for key, value in embeds.items()}
        human_mask = observation.get("human_mask", encoded.get("human_token_mask"))
        human_first = observation.get("human_is_first")
        is_first = observation["is_first"]
        for name, value in (("human_mask", human_mask), ("human_is_first", human_first)):
            if value is None:
                raise KeyError(f"online observation must provide {name}")
        if human_mask.ndim == 3:
            human_mask = human_mask[:, -1]
        if human_first.ndim == 3:
            human_first = human_first[:, -1]
        if is_first.ndim > 1:
            is_first = is_first.reshape(is_first.shape[0], -1)[:, -1]
        goal = observation.get("goal")
        if goal is None:
            raise KeyError("online observation must provide goal")
        if goal.ndim == 3:
            goal = goal[:, -1]
        latent, aux = self.rssm.obs_step(
            agent_state["latent"], agent_state["prev_action"], embeds, is_first,
            human_mask, human_first, goal=goal,
            previous_human_mask=agent_state.get("previous_human_mask"),
            # Evaluation means a deterministic complete policy state, not
            # merely the mode of an Actor fed a newly sampled posterior.
            sample_state=not (
                evaluation and self.deterministic_evaluation_state_enabled),
        )
        task_geometry = observation.get("task_geometry")
        if task_geometry is not None and task_geometry.ndim == 3:
            task_geometry = task_geometry[:, -1]
        task_memory = observation.get("task_memory")
        if task_memory is not None and task_memory.ndim == 3:
            task_memory = task_memory[:, -1]
        task_physical_state = observation.get("task_physical_state")
        if task_physical_state is not None and task_physical_state.ndim == 3:
            task_physical_state = task_physical_state[:, -1]
        joint_feature, augmented_actor_feature, task_feature = (
            self._augment_task_geometry(
                aux["joint_feat"], aux["actor_feat"], task_geometry,
                task_memory))
        aux["joint_feat"] = joint_feature
        aux["actor_feat"] = augmented_actor_feature
        actor_feature = (
            aux["actor_feat"] if self.uses_internal_action_adapter
            else aux["joint_feat"])
        if (
            self.uses_internal_action_adapter
            and self.transition_event is not None
            and self.transition_event.explicit_human_geometry
            and self.actor_explicit_human_geometry_enabled
        ):
            human_root = observation.get("human_root")
            human_quality = observation.get("human_observation_quality")
            if human_root is not None and human_quality is not None:
                if human_root.ndim == 4:
                    human_root = human_root[:, -1]
                if human_quality.ndim == 4:
                    human_quality = human_quality[:, -1]
                human_joint_clearance = None
                skeleton = observation.get("skeleton")
                joint_mask = observation.get("joint_mask")
                human_joints_body = None
                human_joint_velocity_body = None
                if skeleton is not None and joint_mask is not None:
                    if skeleton.ndim == 5:
                        skeleton = skeleton[:, -1]
                    if joint_mask.ndim == 4:
                        joint_mask = joint_mask[:, -1]
                    human_joints_body = skeleton[..., :3].float()
                    human_joint_velocity_body = skeleton[..., 3:6].float()
                    human_joint_clearance = self.deployable_human_clearance(
                        human_joints_body,
                        human_joint_velocity_body,
                        joint_mask.bool(),
                        per_human=True,
                    )
                branch = self.rssm.get_branch_feats(latent)
                geometry_context = self.transition_event.human_geometry_pool(
                    branch["human"], human_root.float(),
                    human_quality.float(), human_mask,
                    human_joint_clearance=human_joint_clearance,
                    human_joints_body=human_joints_body,
                    human_joint_velocity_body=human_joint_velocity_body,
                    human_joint_mask=(
                        None if joint_mask is None else joint_mask.bool()),
                )
                geometry = geometry_context["actor_human_state"]
                latent_attention = aux["latent_attention"]
                observed_ego_state = observation.get("ego_state")
                if observed_ego_state is None:
                    raise KeyError("online observation must provide ego_state")
                if observed_ego_state.ndim == 3:
                    observed_ego_state = observed_ego_state[:, -1]
                ego_token = self.actor_ego_task_token(
                    observed_ego_state,
                    latent_attention["private_ego_token"],
                    task_geometry,
                    task_memory,
                )
                actor_feature = torch.cat((
                    latent_attention["private_goal_token"],
                    ego_token,
                    aux["joint_feat"], geometry,
                ), -1)
                if self.actor_task_physical_state_enabled:
                    if task_physical_state is None:
                        raise KeyError(
                            "full-state Actor requires task_physical_state")
                    expected = int(getattr(
                        self.actor, "task_geometry_dim", 0))
                    if task_physical_state.shape != (
                        actor_feature.shape[0], expected
                    ):
                        raise ValueError(
                            "task physical state does not match Actor rows")
                    actor_feature = torch.cat((
                        actor_feature,
                        task_physical_state.float().to(actor_feature),
                    ), -1)
        dist = self.actor(actor_feature)
        macro_index = agent_state.get("exploration_macro_index")
        macro_remaining = agent_state.get("exploration_macro_remaining")
        macro_delta = agent_state.get("exploration_macro_delta")
        if torch.is_tensor(dist):
            policy_action = dist
        elif evaluation:
            policy_action = self._dist_mode(dist)
        elif sample_actor_distribution:
            # Pure R2-Dreamer collection samples the exact Actor distribution
            # that is optimized in imagination.  The historical online-v6.x
            # path keeps using temporally coherent macro probes by default.
            policy_action = (
                dist.rsample() if hasattr(dist, "rsample") else dist.sample())
        elif self.uses_internal_action_adapter:
            # Temporally coherent macro exploration supplies identifiable
            # action consequences to replay; frame-wise white noise does not.
            mode_action = self._dist_mode(dist)
            batch_size, policy_dim = mode_action.shape
            if batch_size != 1 or policy_dim != 3:
                raise ValueError(
                    "v6.2 structured online exploration expects [1,3]")
            limits = (
                mode_action.new_tensor((0.05, 0.12, 0.03))
                if exploration_delta_limits is None
                else torch.as_tensor(
                    exploration_delta_limits, device=mode_action.device,
                    dtype=mode_action.dtype).reshape(-1)
            )
            if limits.shape != (3,):
                raise ValueError(
                    "v6.2 exploration limits must be [vx,vy,yaw]")
            if macro_remaining is None or int(macro_remaining) <= 0:
                macro_index = int(torch.randint(
                    0, 9, (), device=mode_action.device).item())
                table = mode_action.new_tensor((
                    (0.0, 0.0, 0.0),       # Actor mode / hold macro
                    (-1.0, 0.0, 0.0),      # slow
                    (1.0, 0.0, 0.0),       # small forward-speed probe
                    (0.0, 1.0, 0.0),       # left
                    (0.0, -1.0, 0.0),      # right
                    (-0.5, 1.0, 0.0),      # slow-left
                    (-0.5, -1.0, 0.0),     # slow-right
                    (0.0, 0.0, 1.0),       # small yaw probe
                    (0.0, 0.0, -1.0),      # opposite yaw probe
                ))
                macro_delta = table[macro_index:macro_index + 1] * limits
                # At 10 Hz, hold one intervention for 0.3--0.5 s. Longer
                # probes can dominate a local encounter before feedback is
                # observed and no longer resemble controlled Dreamer data
                # collection.
                macro_remaining = int(torch.randint(
                    3, 6, (), device=mode_action.device).item())
            scale = max(0.0, float(exploration_scale))
            policy_action = (
                mode_action + scale * macro_delta).clamp(-1.0, 1.0)
            macro_remaining = int(macro_remaining) - 1
        elif exploration_delta_limits is not None:
            mode_action = self._dist_mode(dist)
            sampled_action = (
                dist.rsample() if hasattr(dist, "rsample") else dist.sample())
            limits = torch.as_tensor(
                exploration_delta_limits,
                device=mode_action.device,
                dtype=mode_action.dtype,
            ).reshape(1, -1)
            if limits.shape[-1] != mode_action.shape[-1]:
                raise ValueError(
                    "exploration_delta_limits must match action dimension")
            scale = max(0.0, float(exploration_scale))
            delta = (sampled_action - mode_action).clamp(-limits, limits)
            policy_action = (mode_action + scale * delta).clamp(-1.0, 1.0)
        elif hasattr(dist, "rsample"):
            policy_action = dist.rsample()
        else:
            policy_action = dist.sample()
        smoothed_policy_action = policy_action
        if self.action_smoother is not None:
            previous_smoothed = agent_state.get("smoothed_policy_action")
            if previous_smoothed is None:
                previous_smoothed = torch.zeros_like(policy_action)
            smoothed_policy_action = self.action_smoother(
                policy_action, previous_smoothed)
        if self.uses_internal_action_adapter:
            ego_state = observation.get("ego_state")
            if ego_state is None:
                raise KeyError("v6.2 ActionAdapter requires ego_state")
            if ego_state.ndim == 3:
                ego_state = ego_state[:, -1]
            action = self.applied_from_policy_action(
                smoothed_policy_action, ego_state.float())
        else:
            action = policy_action.clamp(-1.0, 1.0)
        planner = None
        if use_planner:
            if self.uses_internal_action_adapter:
                raise RuntimeError(
                    "v6.2 raw Actor training forbids the external online planner")
            ego_state = observation.get("ego_state")
            goal_position = observation.get("goal_position")
            if ego_state is None or goal_position is None:
                raise KeyError(
                    "world-model planner requires ego_state and goal_position")
            if ego_state.ndim == 3:
                ego_state = ego_state[:, -1]
            if goal_position.ndim == 3:
                goal_position = goal_position[:, -1]
            observed_clearance = observation.get(
                "observed_human_clearance_m")
            current_observed_clearance = observation.get(
                "current_observed_human_clearance_m")
            observed_away_xy = observation.get("observed_human_away_xy")
            observed_nearest_xy = observation.get(
                "observed_nearest_human_xy")
            if observed_clearance is None:
                raise KeyError(
                    "world-model planner requires "
                    "observed_human_clearance_m")
            if observed_away_xy is None:
                raise KeyError(
                    "world-model planner requires observed_human_away_xy")
            if observed_clearance.ndim >= 2:
                observed_clearance = observed_clearance.reshape(
                    observed_clearance.shape[0], -1)[:, -1]
            if observed_away_xy.ndim == 3:
                observed_away_xy = observed_away_xy[:, -1]
            if current_observed_clearance is not None:
                if current_observed_clearance.ndim >= 2:
                    current_observed_clearance = (
                        current_observed_clearance.reshape(
                            current_observed_clearance.shape[0], -1
                        )[:, -1]
                    )
            if observed_nearest_xy is not None and observed_nearest_xy.ndim == 3:
                observed_nearest_xy = observed_nearest_xy[:, -1]
            action, planner = self.plan_online_action(
                latent,
                aux["joint_feat"],
                human_mask,
                action,
                agent_state["prev_action"],
                ego_state.float(),
                goal_position.float(),
                observed_clearance.float(),
                observed_away_xy.float(),
                current_observed_clearance_m=(
                    None
                    if current_observed_clearance is None
                    else current_observed_clearance.float()
                ),
                observed_nearest_xy=(
                    None
                    if observed_nearest_xy is None
                    else observed_nearest_xy.float()
                ),
            )
        return action, {
            "latent": latent,
            "prev_action": action,
            "previous_human_mask": human_mask.bool(),
            "policy_action": policy_action,
            "smoothed_policy_action": smoothed_policy_action,
            "planner": planner,
            "exploration_scale": float(
                0.0 if evaluation else max(0.0, exploration_scale)),
            "exploration_macro_index": macro_index,
            "exploration_macro_remaining": macro_remaining,
            "exploration_macro_delta": macro_delta,
        }

    @torch.no_grad()
    def plan_online_action(
        self,
        state: FactorizedState,
        joint_feature: torch.Tensor,
        human_mask: torch.Tensor,
        actor_action: torch.Tensor,
        previous_action: torch.Tensor,
        ego_state: torch.Tensor,
        goal_position: torch.Tensor,
        observed_clearance_m: torch.Tensor | None = None,
        observed_away_xy: torch.Tensor | None = None,
        current_observed_clearance_m: torch.Tensor | None = None,
        observed_nearest_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Select a short-horizon safe action without privileged inputs.

        Four candidates (Actor, slow, left, right) are rolled through the
        learned coupled priors. Planning activates only when the calibrated
        risk head predicts danger. During initial online training the speed
        governor executes only the slow candidate, because the current world
        model detects danger more reliably than it ranks left versus right.
        Candidate scores remain available for diagnostics and later planner
        re-enablement. This is a conservative online-data teacher, not a
        hidden source of ground-truth geometry.
        """
        actor_action = actor_action.float().clamp(-1.0, 1.0)
        previous_action = previous_action.float().clamp(-1.0, 1.0)
        if previous_action.shape != actor_action.shape:
            raise ValueError(
                "previous_action must have the same shape as actor_action")
        current_risk = self.action_risk(
            joint_feature, actor_action, detach_parameters=True)
        current_collision = current_risk["collision_probability"]
        current_clearance = current_risk["min_human_clearance_m"]
        visible_human = human_mask.bool().any(dim=-1)
        if observed_clearance_m is None:
            observed_clearance_m = torch.full_like(
                current_collision.reshape(-1), float("inf"))
        observed_clearance_m = observed_clearance_m.reshape(-1)
        if observed_clearance_m.shape[0] != actor_action.shape[0]:
            raise ValueError("observed clearance batch does not match action")
        if observed_away_xy is None:
            observed_away_xy = actor_action.new_zeros(
                actor_action.shape[0], 2)
        if observed_away_xy.shape != (actor_action.shape[0], 2):
            raise ValueError("observed away direction must have shape [B, 2]")
        observed_away_xy = F.normalize(
            observed_away_xy.float(), dim=-1, eps=1.0e-6)
        if current_observed_clearance_m is None:
            current_observed_clearance_m = observed_clearance_m
        current_observed_clearance_m = current_observed_clearance_m.reshape(-1)
        if current_observed_clearance_m.shape != observed_clearance_m.shape:
            raise ValueError(
                "current observed clearance batch does not match action")
        if observed_nearest_xy is None:
            observed_nearest_xy = actor_action.new_zeros(
                actor_action.shape[0], 2)
        if observed_nearest_xy.shape != (actor_action.shape[0], 2):
            raise ValueError(
                "nearest observed human must have shape [B, 2]")
        nearest_direction = F.normalize(
            observed_nearest_xy.float(), dim=-1, eps=1.0e-6)
        has_nearest = (
            observed_nearest_xy.float().square().sum(dim=-1) > 1.0e-6)
        has_escape = observed_away_xy.square().sum(dim=-1) > 0.25
        observed_caution = (
            observed_clearance_m
            <= self.planner_observed_caution_clearance_m
        )
        observed_emergency = (
            observed_clearance_m
            <= self.planner_observed_emergency_clearance_m
        )
        predicted_emergency = (
            (current_collision >= self.planner_emergency_collision_trigger)
            | (current_clearance <= self.planner_emergency_clearance_m)
        ).reshape(-1)
        forced_escape = (
            observed_clearance_m
            <= self.planner_observed_forced_escape_clearance_m
        )
        # Live skeleton geometry is the minimum online safety teacher. The
        # learned world model can intervene earlier, but cannot veto slowing
        # or sidestepping after an observed close-range threshold is crossed.
        trigger = visible_human & observed_caution
        emergency = visible_human & (
            forced_escape | (predicted_emergency & observed_emergency)
        )

        batch_size, action_dim = actor_action.shape
        candidate_count = 4
        candidates = actor_action[:, None].repeat(1, candidate_count, 1)
        candidates[:, 1, 0] = torch.minimum(
            candidates[:, 1, 0],
            candidates.new_tensor(self.lateral_candidate_slow_forward_action),
        )
        candidates[:, 2:, 0] = torch.minimum(
            candidates[:, 2:, 0],
            candidates.new_tensor(self.safe_forward_action),
        )
        if action_dim > 1:
            candidates[:, 2, 1] = self.lateral_candidate_action
            candidates[:, 3, 1] = -self.lateral_candidate_action
        candidates.clamp_(-1.0, 1.0)

        imagined_state = self._repeat_state_candidates(
            state, candidate_count)
        imagined_mask = human_mask[:, None].expand(
            -1, candidate_count, -1).reshape(
                batch_size * candidate_count, human_mask.shape[-1])
        imagined_goal = goal_position[:, None].expand(
            -1, candidate_count, -1).reshape(-1, 3)
        imagined_action = candidates.reshape(-1, action_dim)
        imagined_ego = ego_state[:, None].expand(
            -1, candidate_count, -1).reshape(-1, ego_state.shape[-1])
        decoded_ego = self.prediction_heads.decode_ego_state(
            self.rssm.get_branch_feats(imagined_state)["ego"])
        initial_distance = torch.linalg.vector_norm(
            imagined_goal - imagined_ego[..., :3], dim=-1, keepdim=True)
        collision_costs = []
        clearance_costs = []
        final_distance = initial_distance
        for _ in range(self.lateral_candidate_rollout_steps):
            imagined_state, _ = self.rssm.img_step(
                imagined_state, imagined_action, imagined_mask)
            next_decoded_ego = self.prediction_heads.decode_ego_state(
                self.rssm.get_branch_feats(imagined_state)["ego"])
            imagined_ego = self.anchor_ego_displacement(
                imagined_ego, decoded_ego, next_decoded_ego)
            decoded_ego = next_decoded_ego
            imagined_goal_feature = goal_features_torch(
                imagined_ego, imagined_goal)
            imagined_joint, _ = self.rssm.get_joint_feat(
                imagined_state, imagined_mask, imagined_goal_feature)
            risk = self.action_risk(
                imagined_joint,
                imagined_action,
                detach_parameters=True,
            )
            collision_costs.append(risk["collision_probability"])
            clearance_costs.append(
                F.softplus(
                    (self.planning_safe_clearance_m
                     - risk["min_human_clearance_m"])
                    / self.clearance_temperature_m
                )
                * self.clearance_temperature_m
                / self.planning_safe_clearance_m
            )
            final_distance = torch.linalg.vector_norm(
                imagined_goal - imagined_ego[..., :3],
                dim=-1,
                keepdim=True,
            )

        collision_cost = torch.stack(collision_costs, dim=1).amax(dim=1)
        clearance_cost = torch.stack(clearance_costs, dim=1).amax(dim=1)
        progress_cost = final_distance - initial_distance
        change_cost = (
            candidates[..., :2] - previous_action[:, None, :2]
        ).square().mean(dim=-1, keepdim=True).reshape(-1, 1)
        score = (
            self.lateral_candidate_collision_cost * collision_cost
            + self.lateral_candidate_clearance_cost * clearance_cost
            + self.lateral_candidate_progress_cost * progress_cost
            + self.lateral_candidate_action_change_cost * change_cost
        ).reshape(batch_size, candidate_count)
        imagined_best_index = score.argmin(dim=1)
        best_score = score.gather(
            1, imagined_best_index[:, None]).squeeze(1)
        improvement = score[:, 0] - best_score
        governor_cap = actor_action[:, 0]
        if self.planner_speed_governor:
            # The risk/clearance heads are already useful danger detectors,
            # while short-horizon directional contrast is not yet calibrated
            # well enough for closed-loop left/right control. Cap only forward
            # speed and preserve the Actor's lateral, vertical and yaw command.
            caution_depth = (
                self.planner_observed_caution_clearance_m
                - observed_clearance_m
            ) / max(
                self.planner_observed_caution_clearance_m
                - self.planner_observed_emergency_clearance_m,
                1.0e-6,
            )
            outer_forward_cap = max(
                self.safe_forward_action,
                self.lateral_candidate_slow_forward_action,
            )
            caution_cap = (
                actor_action.new_full((batch_size,), outer_forward_cap)
                + caution_depth.clamp(0.0, 1.0)
                * (
                    self.lateral_candidate_slow_forward_action
                    - outer_forward_cap
                )
            )
            governor_cap = torch.where(
                emergency,
                actor_action.new_full(
                    (batch_size,), self.planner_emergency_forward_action),
                caution_cap,
            )
            forward_capped = trigger & (
                actor_action[:, 0]
                > governor_cap + torch.finfo(actor_action.dtype).eps
            )
            # Emergency escape must not depend on the Actor still requesting
            # positive forward speed.  A stopping/reversing Actor can remain
            # on a collision course laterally, and previously bypassed the
            # only command that creates observed geometric clearance.
            use_candidate = forward_capped | (emergency & has_escape)
            executed_index = torch.where(
                use_candidate,
                torch.ones_like(imagined_best_index),
                torch.zeros_like(imagined_best_index),
            )
        else:
            use_candidate = (
                trigger
                & imagined_best_index.ne(0)
                & (improvement >= self.planner_min_score_improvement)
            )
            executed_index = torch.where(
                use_candidate,
                imagined_best_index,
                torch.zeros_like(imagined_best_index),
            )
        selected = candidates[
            torch.arange(batch_size, device=candidates.device), executed_index]
        if self.planner_speed_governor:
            selected = actor_action.clone()
            selected[:, 0] = torch.minimum(actor_action[:, 0], governor_cap)
            if action_dim > 1:
                escape_direction = observed_away_xy
                hard_stop = (
                    current_observed_clearance_m
                    <= self.planner_observed_hard_stop_clearance_m
                ) & has_nearest
                critical = (
                    current_observed_clearance_m
                    <= self.planner_observed_critical_clearance_m
                ) & has_nearest
                # Near contact, keep the latched velocity-obstacle side but
                # remove any component that moves toward the closest observed
                # body capsule. If no tangent remains, retreat directly. At
                # critical clearance, blend in additional repulsion before
                # renormalizing to the configured emergency speed.
                approach = (
                    escape_direction * nearest_direction
                ).sum(dim=-1, keepdim=True).clamp_min(0.0)
                tangent = escape_direction - approach * nearest_direction
                tangent_norm = torch.linalg.vector_norm(
                    tangent, dim=-1, keepdim=True)
                tangent = torch.where(
                    tangent_norm > 1.0e-4,
                    tangent / tangent_norm.clamp_min(1.0e-6),
                    -nearest_direction,
                )
                critical_direction = F.normalize(
                    tangent
                    - self.planner_critical_repulsion_blend
                    * nearest_direction,
                    dim=-1,
                    eps=1.0e-6,
                )
                hard_direction = torch.where(
                    critical[:, None], critical_direction, tangent)
                escape_direction = torch.where(
                    hard_stop[:, None], hard_direction, escape_direction)
                # The former fixed 0.60 normalized escape immediately
                # commanded about 1.29 m/s even at the outer edge of the
                # emergency band.  That overwhelmed the Actor and accumulated
                # large lateral detours before online learning could receive
                # useful near-crowd transitions.  Scale continuously from a
                # modest sidestep at the emergency boundary to the full escape
                # command only at critical clearance.
                escape_depth = (
                    self.planner_observed_emergency_clearance_m
                    - current_observed_clearance_m
                ) / max(
                    self.planner_observed_emergency_clearance_m
                    - self.planner_observed_critical_clearance_m,
                    1.0e-6,
                )
                escape_scale = (
                    self.planner_emergency_escape_min_action
                    + escape_depth.clamp(0.0, 1.0)
                    * (
                        self.planner_emergency_escape_action
                        - self.planner_emergency_escape_min_action
                    )
                )
                escape_xy = escape_direction * escape_scale[:, None]
                use_escape = emergency & has_escape
                selected[:, :2] = torch.where(
                    use_escape[:, None], escape_xy, selected[:, :2])
        action = torch.where(
            use_candidate[:, None], selected, actor_action)
        return action, {
            "triggered": trigger,
            "emergency": emergency,
            "visible_human": visible_human,
            "intervened": use_candidate,
            "best_index": executed_index,
            "imagined_best_index": imagined_best_index,
            "score_improvement": improvement,
            "collision_probability": current_collision.reshape(-1),
            "predicted_clearance_m": current_clearance.reshape(-1),
            "observed_clearance_m": observed_clearance_m,
            "current_observed_clearance_m": current_observed_clearance_m,
            "hard_stop": (
                current_observed_clearance_m
                <= self.planner_observed_hard_stop_clearance_m
            ),
            "forward_cap_action": governor_cap,
            "scores": score,
            "actor_action": actor_action,
        }

    def checkpoint_state(self, optimizer=None, scheduler=None, scaler=None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.state_dict()}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        return payload

    def load_checkpoint_state(self, payload, optimizer=None, scheduler=None, scaler=None, strict=True) -> None:
        self.load_state_dict(payload["model"], strict=strict)
        for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            if obj is not None and name in payload:
                obj.load_state_dict(payload[name])

    @torch.no_grad()
    def update_slow_value(self, fraction: float | None = None) -> None:
        mix = self.slow_target_fraction if fraction is None else float(fraction)
        for source, target in zip(self.value.parameters(), self.slow_value.parameters()):
            target.data.copy_(mix * source.data + (1.0 - mix) * target.data)
        for source, target in zip(
            self.safety_value.parameters(),
            self.slow_safety_value.parameters(),
        ):
            target.data.copy_(mix * source.data + (1.0 - mix) * target.data)

    @staticmethod
    def _flatten_state_time(states: FactorizedState) -> FactorizedState:
        return {
            branch: {
                key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
                for key, value in values.items()
            }
            for branch, values in states.items()
        }

    @staticmethod
    def _dist_mode(dist):
        mode = getattr(dist, "mode", None)
        if mode is not None:
            return mode() if callable(mode) else mode
        return dist.mean

    def policy_reward_components(self, feature: torch.Tensor) -> torch.Tensor:
        """Decode supervised reward components into the task's value domain."""
        return symexp(self.reward_components(feature))

    def policy_reward(self, feature: torch.Tensor) -> torch.Tensor:
        """Compose the Actor's task reward from supervised components.

        Human clearance is intentionally a separate constrained objective.
        Keeping it out of task return prevents the predictive corridor term
        from rewarding long routes around the crowd while collision and
        clearance costs still supply the full safety gradient.
        """
        component_value = self.policy_reward_components(feature)
        weights = component_value.new_tensor(
            self.policy_reward_component_weights)
        return (component_value * weights).sum(dim=-1, keepdim=True)

    def policy_reward_v63(
        self,
        next_feature: torch.Tensor,
        event_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Task reward with human collision excluded from the dual objective.

        The non-event components remain learned from replay. Mutually exclusive
        terminal event reward is composed from the calibrated event head, while
        the human-collision class is deliberately zero here and handled only by
        the physical-probability safety constraint.
        """
        component_symlog = self.reward_components(next_feature)
        component_value = symexp(component_symlog)
        weights = component_value.new_tensor(
            self.policy_reward_component_weights)
        weights = weights.clone()
        weights[0] = 0.0
        dense_reward = (component_value * weights).sum(dim=-1, keepdim=True)
        terminal_reward = (
            self.task_goal_event_reward * event_probability[
                ..., _EVENT_REACHED_GOAL_INDEX:_EVENT_REACHED_GOAL_INDEX + 1]
            + self.task_static_collision_reward * event_probability[
                ..., _EVENT_STATIC_COLLISION_INDEX:_EVENT_STATIC_COLLISION_INDEX + 1]
            + self.task_other_terminal_reward * event_probability[
                ..., _EVENT_OTHER_TERMINAL_INDEX:_EVENT_OTHER_TERMINAL_INDEX + 1]
        )
        return dense_reward + terminal_reward

    @staticmethod
    def _v63_task_lambda_return(
        reward: torch.Tensor,
        continue_probability: torch.Tensor,
        next_slow_value: torch.Tensor,
        *,
        discount: float,
        lamb: float,
    ) -> torch.Tensor:
        """Dreamer lambda-return aligned to one-step transition quantities."""
        if not (
            reward.shape == continue_probability.shape == next_slow_value.shape
        ):
            raise ValueError("v6.3 task return inputs must have identical shapes")
        bootstrap = next_slow_value[:, -1]
        outputs = []
        for index in range(reward.shape[1] - 1, -1, -1):
            mixed_bootstrap = (
                (1.0 - float(lamb)) * next_slow_value[:, index]
                + float(lamb) * bootstrap
            )
            bootstrap = reward[:, index] + (
                float(discount) * continue_probability[:, index]
                * mixed_bootstrap
            )
            outputs.append(bootstrap)
        return torch.stack(tuple(reversed(outputs)), dim=1)

    def policy_reward_target(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        components = batch.get("reward_components")
        if components is None:
            return batch["reward"].float()
        weights = components.new_tensor(
            self.policy_reward_component_weights, dtype=torch.float32)
        return (components.float() * weights).sum(dim=-1, keepdim=True)

    @staticmethod
    def lambda_return(last, terminal, reward, value, bootstrap, discount, lamb):
        if not (last.shape == terminal.shape == reward.shape == value.shape == bootstrap.shape):
            raise ValueError("lambda-return inputs must have identical shapes")
        live = (1.0 - terminal.float())[:, 1:] * discount
        cont = (1.0 - last.float())[:, 1:] * lamb
        interm = reward[:, 1:] + (1.0 - cont) * live * bootstrap[:, 1:]
        outputs = [bootstrap[:, -1]]
        for index in reversed(range(live.shape[1])):
            outputs.append(interm[:, index] + live[:, index] * cont[:, index] * outputs[-1])
        return torch.stack(list(reversed(outputs))[:-1], dim=1)

    def _imagine_goal_conditioned(
        self,
        initial: FactorizedState,
        horizon: int,
        human_mask: torch.Tensor,
        goal_position: torch.Tensor,
        *,
        initial_ego_state: torch.Tensor | None = None,
        sample: bool = True,
        collect_states: bool = True,
        first_applied_action: torch.Tensor | None = None,
        first_policy_action: torch.Tensor | None = None,
        policy_action_delta: torch.Tensor | None = None,
        policy_action_delta_steps: int = 0,
        policy_action_override: torch.Tensor | None = None,
        policy_action_override_mask: torch.Tensor | None = None,
        policy_action_override_steps: int = 0,
        initial_previous_applied_action: torch.Tensor | None = None,
        initial_human_root: torch.Tensor | None = None,
        initial_human_quality: torch.Tensor | None = None,
        initial_human_joint_clearance: torch.Tensor | None = None,
        initial_human_joint_mask: torch.Tensor | None = None,
        initial_human_joints_body: torch.Tensor | None = None,
        initial_human_joint_velocity_body: torch.Tensor | None = None,
        initial_human_presence: torch.Tensor | None = None,
        initial_task_geometry: torch.Tensor | None = None,
        initial_task_physical_state: torch.Tensor | None = None,
        initial_task_memory: torch.Tensor | None = None,
        task_progress_epsilon_m: float = 0.25,
        task_stuck_max_horizontal_speed_mps: float = 0.15,
        task_acceleration_filter_alpha: float = 0.25,
        task_origin_xyz: torch.Tensor | None = None,
        task_origin_yaw: torch.Tensor | None = None,
        task_flight_bounds_xy: torch.Tensor | None = None,
        task_static_obstacle_aabbs_xy: torch.Tensor | None = None,
        task_static_obstacle_valid: torch.Tensor | None = None,
        task_bounds_valid: torch.Tensor | None = None,
        task_maximum_altitude_m: torch.Tensor | None = None,
        task_maximum_altitude_valid: torch.Tensor | None = None,
        task_static_terminal_valid: torch.Tensor | None = None,
        authoritative_analytic_ego: bool = False,
        antithetic_policy_pairs: bool = False,
    ):
        """Roll out coupled priors and recompute Goal token at every step.

        The fixed target remains in the episode-start local frame.  Ego14 is
        decoded from each imagined Ego latent, then converted to an updated
        body-relative Goal feature.  Goal therefore conditions the policy but
        never enters either branch's physical transition.
        """
        if self.prediction_heads is None or not hasattr(self.prediction_heads, "decode_ego_state"):
            raise RuntimeError("goal-conditioned imagination requires the Ego state decoder")
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        if goal_position.ndim != 2 or goal_position.shape[-1] != 3:
            raise ValueError("goal_position must be [B,3]")
        state = initial
        initial_ego_feature = self.rssm.get_branch_feats(state)["ego"]
        decoded_ego = self.prediction_heads.decode_ego_state(
            initial_ego_feature)
        ego_state = (
            decoded_ego if initial_ego_state is None
            else initial_ego_state.float()
        )
        if ego_state.shape != decoded_ego.shape:
            raise ValueError(
                "initial_ego_state must match the imagined batch as [B,14]")
        states, joints, decision_feats, actor_feats = [], [], [], []
        policy_actions, policy_pre_tanh, smoothed_policy_actions, actions = (
            [], [], [], [])
        action_smoother_local_scales = []
        goals, ego_states, analytic_ego_states, learned_ego_states = (
            [], [], [], [])
        task_geometries, task_physical_states, task_memories, human_masks = (
            [], [], [], [])
        task_geometry_clearances = []
        human_roots, human_qualities, human_presences = [], [], []
        human_birth_probabilities = []
        human_joint_clearances, human_joints_body = [], []
        human_joint_velocities_body = []
        learned_human_joint_clearances, cv_human_joint_clearances = [], []
        cv_human_joints_body = []
        ego_branch_features, human_branch_features = [], []
        human_root_state = (
            None if initial_human_root is None
            else initial_human_root.float().clone())
        if human_root_state is not None and human_root_state.shape[-1] < 6:
            human_root_state = F.pad(
                human_root_state, (0, 6 - human_root_state.shape[-1]))
        human_quality_state = (
            None if initial_human_quality is None
            else initial_human_quality.float().clone())
        if (human_root_state is None) != (human_quality_state is None):
            raise ValueError(
                "initial Human root and quality must be supplied together")
        human_joint_mask_state = None
        if human_root_state is not None:
            if initial_human_joint_mask is None:
                raise ValueError(
                    "explicit Human geometry imagination requires "
                    "initial_human_joint_mask")
            human_joint_mask_state = persistent_rollout_joint_mask(
                human_mask, initial_human_joint_mask)
        human_joint_clearance_state = (
            None if initial_human_joint_clearance is None
            else initial_human_joint_clearance.float().clone())
        if human_root_state is not None and human_joint_clearance_state is None:
            # Compatibility fallback for callers without measured joints.
            human_joint_clearance_state = (
                torch.linalg.vector_norm(
                    human_root_state[..., :3], dim=-1)
                - self.lateral_candidate_surface_radius_m
            ).clamp(0.0, 6.0)
        if human_joint_clearance_state is not None and (
            human_joint_clearance_state.shape != human_mask.shape
        ):
            raise ValueError(
                "initial Human joint clearance must match human_mask")
        human_joints_body_state = (
            None if initial_human_joints_body is None
            else initial_human_joints_body.float().clone())
        if human_joints_body_state is not None:
            if human_joint_mask_state is None:
                raise ValueError(
                    "initial Human joints require an initial joint mask")
            if (
                human_joints_body_state.shape[:-1]
                != human_joint_mask_state.shape
                or human_joints_body_state.shape[-1] != 3
            ):
                raise ValueError(
                    "initial Human joints must match [B,N,J,3]")
        human_joint_velocity_body_state = (
            None if initial_human_joint_velocity_body is None
            else initial_human_joint_velocity_body.float().clone())
        if human_joints_body_state is not None:
            if human_joint_velocity_body_state is None:
                human_joint_velocity_body_state = torch.zeros_like(
                    human_joints_body_state)
            if human_joint_velocity_body_state.shape != human_joints_body_state.shape:
                raise ValueError(
                    "initial Human joint velocities must match [B,N,J,3]")
        human_cv_joints_body_state = (
            None if human_joints_body_state is None
            else human_joints_body_state.clone())
        human_cv_joint_velocity_body_state = (
            None if initial_human_joint_velocity_body is None
            else initial_human_joint_velocity_body.float().clone())
        if human_cv_joints_body_state is not None:
            if human_cv_joint_velocity_body_state is None:
                # Compatibility for older diagnostics. Production pure
                # Dreamer supplies measured per-joint velocities.
                human_cv_joint_velocity_body_state = torch.zeros_like(
                    human_cv_joints_body_state)
            if (
                human_cv_joint_velocity_body_state.shape
                != human_cv_joints_body_state.shape
            ):
                raise ValueError(
                    "initial Human joint velocities must match [B,N,J,3]")
        human_presence_state = (
            human_mask.to(ego_state.dtype)
            if initial_human_presence is None
            else initial_human_presence.to(ego_state.dtype).clone()
        )
        if human_presence_state.shape != human_mask.shape:
            raise ValueError("initial Human presence must match human_mask")
        # A source-observed person remains a physical obstacle throughout the
        # short 1.5 s imagination even if the learned visibility/survival head
        # expects the track to leave camera view. Constant-velocity geometry
        # moves that person away naturally; visibility loss must not erase it.
        human_safety_presence_state = human_mask.to(ego_state.dtype)
        human_rollout_mask = human_mask.bool().clone()
        task_memory_state = (
            None if initial_task_memory is None
            else initial_task_memory.float().clone())
        if task_memory_state is not None and task_memory_state.shape != (
            ego_state.shape[0], TASK_MEMORY_DIM
        ):
            raise ValueError(
                f"initial task memory must be [B,{TASK_MEMORY_DIM}]")
        analytic_ego_state = ego_state.clone()
        if first_applied_action is not None and first_policy_action is not None:
            raise ValueError(
                "first applied and first policy action are mutually exclusive")
        expected_action_dim = int(getattr(
            self.rssm, "action_dim", first_applied_action.shape[-1]
            if first_applied_action is not None else 0))
        if first_applied_action is not None and (
            first_applied_action.shape
            != (ego_state.shape[0], expected_action_dim)
        ):
            raise ValueError(
                "first_applied_action must match RSSM action dimension")
        policy_action_dim = (
            self.action_smoother.action_dim
            if self.action_smoother is not None
            else int(getattr(self.actor, "act_dim", expected_action_dim))
        )
        if first_policy_action is not None and (
            first_policy_action.shape
            != (ego_state.shape[0], policy_action_dim)
        ):
            raise ValueError(
                "first_policy_action must match the policy action dimension")
        policy_action_delta_steps = int(policy_action_delta_steps)
        if policy_action_delta_steps < 0:
            raise ValueError("policy_action_delta_steps must be non-negative")
        if policy_action_delta is not None and (
            policy_action_delta.shape
            != (ego_state.shape[0], policy_action_dim)
        ):
            raise ValueError(
                "policy_action_delta must match the policy action dimension")
        if (policy_action_delta is None) != (policy_action_delta_steps == 0):
            raise ValueError(
                "policy_action_delta and positive delta steps are required "
                "together")
        policy_action_override_steps = int(policy_action_override_steps)
        if policy_action_override_steps < 0:
            raise ValueError(
                "policy_action_override_steps must be non-negative")
        if policy_action_override is not None and (
            policy_action_override.shape
            != (ego_state.shape[0], policy_action_dim)
        ):
            raise ValueError(
                "policy_action_override must match the policy action dimension")
        if policy_action_override_mask is not None and (
            policy_action_override_mask.shape != (ego_state.shape[0], 1)
        ):
            raise ValueError(
                "policy_action_override_mask must be [B,1]")
        if (policy_action_override is None) != (
            policy_action_override_steps == 0
        ):
            raise ValueError(
                "policy_action_override and positive override steps are "
                "required together")
        if policy_action_override is None and policy_action_override_mask is not None:
            raise ValueError(
                "policy_action_override_mask requires an override")
        if policy_action_delta is not None and policy_action_override is not None:
            raise ValueError(
                "policy action delta and absolute override are mutually exclusive")
        if initial_previous_applied_action is not None and (
            initial_previous_applied_action.shape
            != (ego_state.shape[0], expected_action_dim)
        ):
            raise ValueError(
                "initial_previous_applied_action must match RSSM action dimension")
        if self.action_smoother is None:
            previous_smoothed_policy = None
        elif initial_previous_applied_action is None:
            previous_smoothed_policy = ego_state.new_zeros(
                ego_state.shape[0], self.action_smoother.action_dim)
        else:
            previous_smoothed_policy = self.policy_from_applied_action(
                initial_previous_applied_action.float().clamp(-1.0, 1.0))
        for rollout_step in range(int(horizon)):
            # The RSSM's learned Ego state remains recursively imagined and
            # audited below.  Pure Dreamer can nevertheless make the known
            # applied-action kinematics authoritative for all task geometry.
            # Both paths are differentiable with respect to Actor actions.
            task_ego_state = (
                analytic_ego_state
                if authoritative_analytic_ego else ego_state)
            goal = goal_features_torch(task_ego_state, goal_position)
            branch = self.rssm.get_branch_feats(state)
            joint, latent = self.rssm.get_joint_feat(
                state, human_rollout_mask, goal)
            actor_feature_base = (
                latent["actor_feat"] if self.uses_internal_action_adapter
                else joint)
            geometry_context = (
                task_origin_xyz, task_origin_yaw, task_flight_bounds_xy,
                task_static_obstacle_aabbs_xy, task_static_obstacle_valid,
            )
            if all(value is not None for value in geometry_context):
                assert task_origin_xyz is not None
                assert task_origin_yaw is not None
                assert task_flight_bounds_xy is not None
                assert task_static_obstacle_aabbs_xy is not None
                assert task_static_obstacle_valid is not None
                task_geometry_state = episode_to_world_geometry_torch(
                    task_ego_state,
                    task_origin_xyz,
                    task_origin_yaw,
                    task_flight_bounds_xy,
                    task_static_obstacle_aabbs_xy,
                    task_static_obstacle_valid,
                )
                task_geometry_clearance_state = (
                    episode_to_world_task_clearance_torch(
                        task_ego_state,
                        task_origin_xyz,
                        task_origin_yaw,
                        task_flight_bounds_xy,
                        task_static_obstacle_aabbs_xy,
                        task_static_obstacle_valid,
                    )
                )
                if self.actor_task_physical_state_enabled:
                    if (
                        task_bounds_valid is None
                        or task_maximum_altitude_m is None
                        or task_maximum_altitude_valid is None
                        or task_static_terminal_valid is None
                    ):
                        raise ValueError(
                            "full-state Actor requires boundary/altitude "
                            "validity and maximum altitude")
                    task_physical_state = (
                        episode_to_world_task_physical_state_torch(
                            task_ego_state,
                            task_origin_xyz,
                            task_origin_yaw,
                            task_flight_bounds_xy,
                            task_static_obstacle_aabbs_xy,
                            task_static_obstacle_valid,
                            maximum_altitude_m=task_maximum_altitude_m,
                            bounds_valid=task_bounds_valid,
                            maximum_altitude_valid=(
                                task_maximum_altitude_valid),
                            static_terminal_valid=(
                                task_static_terminal_valid),
                            maximum_obstacles=int(
                                self.actor_task_physical_obstacle_slots),
                        )
                    )
                else:
                    task_physical_state = None
            elif any(value is not None for value in geometry_context):
                raise ValueError("incomplete imagined task-geometry context")
            else:
                task_geometry_state = initial_task_geometry
                task_geometry_clearance_state = None
                task_physical_state = initial_task_physical_state
            joint, actor_feature, task_feature = self._augment_task_geometry(
                joint, actor_feature_base, task_geometry_state,
                task_memory_state)
            decision_feature = joint
            if (
                self.uses_internal_action_adapter
                and self.transition_event is not None
                and self.transition_event.explicit_human_geometry
                and self.actor_explicit_human_geometry_enabled
                and human_root_state is not None
                and human_quality_state is not None
            ):
                geometry_context = self.transition_event.human_geometry_pool(
                    branch["human"], human_root_state,
                    human_quality_state, human_rollout_mask,
                    human_joint_clearance=human_joint_clearance_state,
                    human_joints_body=human_joints_body_state,
                    human_joint_velocity_body=(
                        human_joint_velocity_body_state),
                    human_joint_mask=human_joint_mask_state,
                    human_presence=human_safety_presence_state,
                    detach_parameters=True,
                )
                geometry = geometry_context["actor_human_state"]
                ego_token = self.actor_ego_task_token(
                    task_ego_state,
                    latent["private_ego_token"],
                    task_geometry_state,
                    task_memory_state,
                )
                actor_feature = torch.cat((
                    latent["private_goal_token"], ego_token,
                    joint, geometry,
                ), -1)
                if self.actor_task_physical_state_enabled:
                    if task_physical_state is None:
                        raise ValueError(
                            "full-state Actor requires exact task geometry")
                    if task_physical_state.shape[-1] != int(
                        getattr(self.actor, "task_geometry_dim", 0)
                    ):
                        raise ValueError(
                            "imagined task physical state width is invalid")
                    actor_feature = torch.cat((
                        actor_feature,
                        task_physical_state.float().to(actor_feature),
                    ), -1)
                if self.critic_full_state_enabled:
                    decision_feature = actor_feature
                elif geometry.shape == joint.shape:
                    decision_feature = self._merge_decision_geometry(
                        joint, geometry)
            if rollout_step == 0 and first_applied_action is not None:
                action = first_applied_action.float().clamp(-1.0, 1.0)
                policy_action = self.policy_from_applied_action(action)
                policy_latent = torch.atanh(
                    policy_action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
                smoothed_policy_action = policy_action
            elif rollout_step == 0 and first_policy_action is not None:
                # Counterfactual gradient audits must exercise the same
                # policy3 -> stateful smoother -> deterministic altitude
                # adapter path as training and deployment.  Injecting an
                # applied4 tensor bypasses both state variables and can assign
                # an apparent policy gradient to the non-learned vertical
                # controller axis.
                policy_action = first_policy_action.float().clamp(-1.0, 1.0)
                policy_latent = torch.atanh(
                    policy_action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
                smoothed_policy_action = policy_action
                if self.action_smoother is not None:
                    assert previous_smoothed_policy is not None
                    smoothed_policy_action = self.action_smoother(
                        policy_action, previous_smoothed_policy)
                action = self.applied_from_policy_action(
                    smoothed_policy_action, task_ego_state)
            elif (
                policy_action_override is not None
                and rollout_step < policy_action_override_steps
                and policy_action_override_mask is None
            ):
                # A full-row counterfactual override is independent of the
                # Actor. Bypass its distribution instead of computing an
                # output that the override discards for every row. Standard
                # Actor imagination has no override and remains unchanged.
                policy_action = policy_action_override.float().clamp(
                    -1.0, 1.0)
                policy_latent = torch.atanh(policy_action.clamp(
                    -1.0 + 1.0e-6, 1.0 - 1.0e-6))
                smoothed_policy_action = policy_action
                if self.action_smoother is not None:
                    assert previous_smoothed_policy is not None
                    smoothed_policy_action = self.action_smoother(
                        policy_action, previous_smoothed_policy)
                action = self.applied_from_policy_action(
                    smoothed_policy_action, task_ego_state)
            else:
                actor_distribution = self.actor(actor_feature)
                if sample and not torch.is_tensor(actor_distribution) and hasattr(
                    actor_distribution, "rsample_with_pre_tanh"
                ):
                    if antithetic_policy_pairs:
                        antithetic_sample = getattr(
                            actor_distribution,
                            "rsample_with_pre_tanh_antithetic_pairs",
                            None,
                        )
                        if not callable(antithetic_sample):
                            raise TypeError(
                                "antithetic Actor imagination requires a "
                                "paired reparameterized distribution")
                        policy_action, policy_latent = antithetic_sample()
                    else:
                        policy_action, policy_latent = (
                            actor_distribution.rsample_with_pre_tanh())
                else:
                    # Preserve the generic/mocked RSSM sampling contract for
                    # discrete policies, deterministic planning and tests.
                    policy_action = self.rssm._sample_actor(
                        self.actor, actor_feature, sample)
                    policy_latent = getattr(
                        actor_distribution, "pre_tanh_mean", None)
                # Deployment keeps the selected Actor-centred candidate
                # offset for several receding-horizon control ticks.  Apply
                # that same offset in imagination so the scored action chain
                # is the one the online controller will actually execute.
                if (
                    policy_action_delta is not None
                    and rollout_step < policy_action_delta_steps
                ):
                    policy_action = (
                        policy_action + policy_action_delta
                    ).clamp(-1.0, 1.0)
                if (
                    policy_action_override is not None
                    and rollout_step < policy_action_override_steps
                ):
                    override_mask = (
                        torch.ones_like(policy_action[..., :1], dtype=torch.bool)
                        if policy_action_override_mask is None
                        else policy_action_override_mask.bool()
                    )
                    policy_action = torch.where(
                        override_mask,
                        policy_action_override.float().clamp(-1.0, 1.0),
                        policy_action,
                    )
                if (
                    policy_latent is not None
                    and (
                        policy_action_delta is not None
                        or policy_action_override is not None
                    )
                ):
                    # Candidate modifications are not samples from the Actor;
                    # retain only a numerically valid diagnostic inverse.
                    policy_latent = torch.atanh(policy_action.clamp(
                        -1.0 + 1.0e-6, 1.0 - 1.0e-6))
                smoothed_policy_action = policy_action
                if self.action_smoother is not None:
                    assert previous_smoothed_policy is not None
                    smoothed_policy_action = self.action_smoother(
                        policy_action, previous_smoothed_policy)
                action = self.applied_from_policy_action(
                    smoothed_policy_action, task_ego_state)
            if self.action_smoother is None:
                action_smoother_local_scale = torch.ones_like(policy_action)
            elif rollout_step == 0 and first_applied_action is not None:
                # This explicit compatibility path bypasses the Actor and its
                # smoother; it must not masquerade as a trainable Jacobian.
                action_smoother_local_scale = torch.zeros_like(policy_action)
            else:
                assert previous_smoothed_policy is not None
                action_smoother_local_scale = (
                    self.action_smoother.local_action_scale(
                        policy_action, previous_smoothed_policy))
            if self.action_smoother is not None:
                previous_smoothed_policy = smoothed_policy_action
            if collect_states:
                states.append(state)
            joints.append(joint)
            decision_feats.append(decision_feature)
            actor_feats.append(actor_feature)
            policy_actions.append(policy_action)
            policy_pre_tanh.append(policy_latent)
            smoothed_policy_actions.append(smoothed_policy_action)
            action_smoother_local_scales.append(
                action_smoother_local_scale)
            actions.append(action)
            goals.append(goal)
            ego_states.append(task_ego_state)
            analytic_ego_states.append(analytic_ego_state)
            learned_ego_states.append(ego_state)
            human_masks.append(human_rollout_mask)
            if task_geometry_state is not None:
                task_geometries.append(task_geometry_state)
            if task_physical_state is not None:
                task_physical_states.append(task_physical_state)
            if task_memory_state is not None:
                task_memories.append(task_memory_state)
            if task_geometry_clearance_state is not None:
                task_geometry_clearances.append(task_geometry_clearance_state)
            ego_branch_features.append(branch["ego"])
            human_branch_features.append(branch["human"])
            if human_root_state is not None and human_quality_state is not None:
                human_roots.append(human_root_state)
                human_qualities.append(human_quality_state)
                human_presences.append(human_safety_presence_state)
                human_joint_clearances.append(human_joint_clearance_state)
                if human_joints_body_state is not None:
                    human_joints_body.append(human_joints_body_state)
                    assert human_joint_velocity_body_state is not None
                    human_joint_velocities_body.append(
                        human_joint_velocity_body_state)
                if human_cv_joints_body_state is not None:
                    cv_human_joints_body.append(human_cv_joints_body_state)
            next_state, _ = self.rssm.img_step(
                state, action, human_rollout_mask,
                sample_state=sample)
            next_branch = self.rssm.get_branch_feats(next_state)
            next_ego_feature = next_branch["ego"]
            next_decoded_ego = self.prediction_heads.decode_ego_state(
                next_ego_feature)
            next_learned_ego_state = (
                self.anchor_ego_residual(
                    ego_state, decoded_ego, next_decoded_ego, action,
                    velocity_response=self.analytic_ego_velocity_response)
                if self.uses_internal_action_adapter
                else self.anchor_ego_displacement(
                    ego_state, decoded_ego, next_decoded_ego)
            )
            next_analytic_ego_state = (
                analytic_ego_step(
                    analytic_ego_state, action,
                    dt_s=self.lateral_candidate_step_duration_s,
                    velocity_response=self.analytic_ego_velocity_response,
                    attitude_coefficients=(
                        self.analytic_ego_attitude_coefficients),
                )
                if action.shape[-1] == 4 else next_learned_ego_state
            )
            if task_memory_state is not None:
                task_memory_state = task_memory_step_torch(
                    task_memory_state,
                    next_analytic_ego_state
                    if authoritative_analytic_ego
                    else next_learned_ego_state,
                    goal_position,
                    smoothed_policy_action,
                    dt_s=self.lateral_candidate_step_duration_s,
                    progress_epsilon_m=task_progress_epsilon_m,
                    stuck_max_horizontal_speed_mps=(
                        task_stuck_max_horizontal_speed_mps),
                    acceleration_filter_alpha=(
                        task_acceleration_filter_alpha),
                )
            if human_root_state is not None and human_quality_state is not None:
                human_current_ego = (
                    analytic_ego_state
                    if authoritative_analytic_ego else ego_state)
                human_next_ego = (
                    next_analytic_ego_state
                    if authoritative_analytic_ego
                    else next_learned_ego_state)
                predicted_human = self.prediction_heads.human(
                    next_branch["human"], human_root_state,
                    current_ego=human_current_ego,
                    # During Actor imagination the World parameters are
                    # frozen, but the action -> Ego -> relative-Human chain
                    # must remain differentiable.  Detaching here made all
                    # future Human geometry blind to the UAV action's induced
                    # motion and broke Dreamer's dynamics gradient.
                    next_ego=human_next_ego,
                    current_joints=human_joints_body_state,
                    current_joint_velocity=human_joint_velocity_body_state,
                    current_joint_mask=human_joint_mask_state,
                    lifecycle_feat=branch["human"],
                    dt_s=self.lateral_candidate_step_duration_s,
                )
                birth_per_empty_slot = (
                    torch.sigmoid(predicted_human["birth_logit"])
                    * (~human_rollout_mask).to(human_presence_state)
                )
                # Probability that at least one currently empty slot acquires
                # a track at the next step. Direction is unknowable until an
                # observation arrives, so this remains an audited world-model
                # diagnostic only. It must not be described as an Actor-return
                # term until a deployable directional birth state exists.
                human_birth_probabilities.append(
                    1.0 - torch.prod(
                        1.0 - birth_per_empty_slot.clamp(0.0, 1.0),
                        dim=-1,
                        keepdim=True,
                    )
                )
                human_root_state = torch.cat((
                    predicted_human["root"],
                    predicted_human["root_velocity"],
                    human_root_state[..., 6:],
                ), -1)
                survival = torch.sigmoid(
                    predicted_human["survival_logit"])
                human_presence_state = (
                    human_presence_state * survival
                ).clamp(0.0, 1.0)
                assert human_joint_mask_state is not None
                predicted_joint_mask = human_joint_mask_state
                human_joints_body_state = predicted_human["joints"]
                human_joint_velocity_body_state = predicted_human[
                    "joint_velocity"]
                learned_clearance = self.deployable_human_clearance(
                    human_joints_body_state,
                    human_joint_velocity_body_state,
                    predicted_joint_mask,
                    per_human=True,
                )
                if (
                    human_cv_joints_body_state is None
                    or human_cv_joint_velocity_body_state is None
                ):
                    cv_clearance = learned_clearance
                else:
                    cv_shape = human_cv_joints_body_state.shape
                    flat_cv_joints = human_cv_joints_body_state.reshape(
                        cv_shape[0], -1, 3)
                    flat_cv_velocity = human_cv_joint_velocity_body_state.reshape(
                        cv_shape[0], -1, 3)
                    flat_cv_joints, flat_cv_velocity = (
                        analytic_relative_human_step(
                            flat_cv_joints,
                            flat_cv_velocity,
                            analytic_ego_state,
                            next_analytic_ego_state,
                            dt_s=self.lateral_candidate_step_duration_s,
                        )
                    )
                    human_cv_joints_body_state = flat_cv_joints.reshape(cv_shape)
                    human_cv_joint_velocity_body_state = (
                        flat_cv_velocity.reshape(cv_shape))
                    cv_clearance = self.deployable_human_clearance(
                        human_cv_joints_body_state,
                        human_cv_joint_velocity_body_state,
                        predicted_joint_mask,
                        per_human=True,
                    )
                learned_human_joint_clearances.append(learned_clearance)
                cv_human_joint_clearances.append(cv_clearance)
                human_joint_clearance_state = (
                    torch.minimum(learned_clearance, cv_clearance)
                    if self.conservative_human_clearance_enabled
                    else learned_clearance
                )
                # Compact-v3 quality is ordered as measured-joint fraction,
                # predicted-joint fraction, track age, measurement age,
                # prediction run, velocity sigma, and velocity-valid times
                # identity confidence.  A recursive model step predicts every
                # source-valid joint; learned camera survival is not a joint
                # fraction and must not be written into field 1.  Carry that
                # visibility uncertainty through the confidence fields while
                # retaining the person as a physical obstacle for the full
                # short horizon.
                human_root_state = human_root_state.clone()
                if human_root_state.shape[-1] >= 10:
                    human_root_state[..., 9] *= survival
                human_quality_state = human_quality_state.clone()
                human_quality_state[..., 0] = 0.0
                human_quality_state[..., 1] = (
                    human_joint_mask_state.float().mean(-1)
                    * human_safety_presence_state)
                human_quality_state[..., 2] += (
                    self.lateral_candidate_step_duration_s)
                human_quality_state[..., 3] += (
                    self.lateral_candidate_step_duration_s)
                human_quality_state[..., 4] += 1.0
                human_quality_state[..., 6] *= survival
            ego_state = next_learned_ego_state
            analytic_ego_state = next_analytic_ego_state
            state = next_state
            decoded_ego = next_decoded_ego
        stacked_states = self._stack_imagined_states(states) if collect_states else None
        result = {
            "joint_feat": torch.stack(joints, 1),
            "decision_feat": torch.stack(decision_feats, 1),
            "actor_feat": torch.stack(actor_feats, 1),
            "policy_action": torch.stack(policy_actions, 1),
            "smoothed_policy_action": torch.stack(
                smoothed_policy_actions, 1),
            "action_smoother_local_scale": torch.stack(
                action_smoother_local_scales, 1),
            "action": torch.stack(actions, 1),
            "goal": torch.stack(goals, 1),
            "goal_position": goal_position,
            "ego_state": torch.stack(ego_states, 1),
            "analytic_ego_state": torch.stack(analytic_ego_states, 1),
            "learned_ego_state": torch.stack(learned_ego_states, 1),
            "authoritative_analytic_ego": torch.as_tensor(
                float(authoritative_analytic_ego),
                dtype=ego_state.dtype, device=ego_state.device),
            "ego_branch_feature": torch.stack(ego_branch_features, 1),
            "human_branch_feature": torch.stack(human_branch_features, 1),
            "human_mask": human_mask,
            "human_mask_rollout": torch.stack(human_masks, 1),
            "final_state": state,
        }
        if task_geometries:
            result["task_geometry"] = torch.stack(task_geometries, 1)
        if task_physical_states:
            result["task_physical_state"] = torch.stack(
                task_physical_states, 1)
        if task_memories:
            result["task_memory"] = torch.stack(task_memories, 1)
        if task_geometry_clearances:
            result["task_geometry_clearance_m"] = torch.stack(
                task_geometry_clearances, 1)[..., None]
        if human_birth_probabilities:
            result["human_birth_probability"] = torch.stack(
                human_birth_probabilities, 1)
        if all(item is not None for item in policy_pre_tanh):
            result["policy_pre_tanh"] = torch.stack(policy_pre_tanh, 1)
        if human_roots:
            result.update({
                "human_root": torch.stack(human_roots, 1),
                "human_observation_quality": torch.stack(human_qualities, 1),
                "human_presence": torch.stack(human_presences, 1),
                "human_joint_mask": human_joint_mask_state,
                "human_joint_clearance": torch.stack(
                    human_joint_clearances, 1),
            })
            if human_joints_body:
                result["human_joints_body"] = torch.stack(
                    human_joints_body, 1)
                # The kinematic Human head carries velocity as part of the
                # recursive decision state; Event and Actor must see the same
                # directional pose derivative used to generate the next pose.
                result["human_joint_velocity_body"] = torch.stack(
                    human_joint_velocities_body, 1)
            if cv_human_joints_body:
                result["constant_velocity_human_joints_body"] = torch.stack(
                    cv_human_joints_body, 1)
            if learned_human_joint_clearances:
                result["learned_human_joint_clearance_next"] = torch.stack(
                    learned_human_joint_clearances, 1)
                result["constant_velocity_human_joint_clearance_next"] = (
                    torch.stack(cv_human_joint_clearances, 1))
        return stacked_states, result

    @staticmethod
    def anchor_ego_displacement(
        source_truth: torch.Tensor,
        source_decoded: torch.Tensor,
        target_decoded: torch.Tensor,
    ) -> torch.Tensor:
        """Carry action-conditioned latent displacement from an exact pose.

        Absolute position decoded independently from each latent has a stable
        scene-wide bias.  Subtracting consecutive decodes cancels most of that
        bias while retaining the RSSM transition's dependence on the candidate
        action. Yaw is accumulated as a wrapped angular displacement.
        """
        anchored = target_decoded.clone()
        anchored[..., :3] = (
            source_truth[..., :3]
            + target_decoded[..., :3] - source_decoded[..., :3]
        )
        source_truth_yaw = torch.atan2(
            source_truth[..., 12], source_truth[..., 13])
        source_decoded_yaw = torch.atan2(
            source_decoded[..., 12], source_decoded[..., 13])
        target_decoded_yaw = torch.atan2(
            target_decoded[..., 12], target_decoded[..., 13])
        yaw_delta = torch.atan2(
            torch.sin(target_decoded_yaw - source_decoded_yaw),
            torch.cos(target_decoded_yaw - source_decoded_yaw),
        )
        anchored_yaw = source_truth_yaw + yaw_delta
        anchored[..., 12] = torch.sin(anchored_yaw)
        anchored[..., 13] = torch.cos(anchored_yaw)
        return anchored

    def anchor_ego_residual(
        self,
        source_truth: torch.Tensor,
        source_decoded: torch.Tensor,
        target_decoded: torch.Tensor,
        applied_action: torch.Tensor,
        *,
        dt_s: float | torch.Tensor = 0.1,
        residual_scale: float = 0.25,
        velocity_response: float | tuple[float, float, float] = 1.0,
    ) -> torch.Tensor:
        """Apply a bounded learned correction around analytic action dynamics.

        A full residual would algebraically cancel the analytic baseline for a
        legacy RSSM that already learned the complete displacement.  Keeping
        the correction small preserves action causality during migration while
        still representing aerodynamic/model mismatch.
        """
        if not 0.0 <= float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must be in [0, 1]")
        truth_baseline = analytic_ego_step(
            source_truth, applied_action, dt_s=dt_s,
            velocity_response=velocity_response,
            attitude_coefficients=self.analytic_ego_attitude_coefficients)
        decoded_baseline = analytic_ego_step(
            source_decoded, applied_action, dt_s=dt_s,
            velocity_response=velocity_response,
            attitude_coefficients=self.analytic_ego_attitude_coefficients)
        residual = (target_decoded - decoded_baseline) * float(residual_scale)
        anchored = truth_baseline + residual
        target_yaw = torch.atan2(
            truth_baseline[..., 12], truth_baseline[..., 13])
        decoded_baseline_yaw = torch.atan2(
            decoded_baseline[..., 12], decoded_baseline[..., 13])
        decoded_target_yaw = torch.atan2(
            target_decoded[..., 12], target_decoded[..., 13])
        yaw_residual = torch.atan2(
            torch.sin(decoded_target_yaw - decoded_baseline_yaw),
            torch.cos(decoded_target_yaw - decoded_baseline_yaw))
        yaw = target_yaw + yaw_residual * float(residual_scale)
        anchored[..., 12] = torch.sin(yaw)
        anchored[..., 13] = torch.cos(yaw)
        return anchored

    @staticmethod
    def _stack_imagined_states(states: list[FactorizedState]) -> FactorizedState:
        return {
            branch: {
                key: torch.stack([state[branch][key] for state in states], dim=1)
                for key in ("stoch", "deter")
            }
            for branch in ("ego", "human")
        }

    def actor_critic_loss(
        self,
        posterior_states: FactorizedState,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
        imag_horizon: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """DreamerV3 imagination and replay-value objectives on joint features."""

        if self.uses_internal_action_adapter:
            return self.actor_critic_loss_v62(
                posterior_states, posterior_aux, batch, imag_horizon)

        b, t = batch["action"].shape[:2]
        start = self._flatten_state_time(posterior_states)
        mask = posterior_aux["human_mask"].reshape(b * t, -1)
        if "goal_position" not in batch:
            raise KeyError("actor-critic imagination requires goal_position")
        goal_position = batch["goal_position"]
        if goal_position.ndim == 2:
            goal_position = goal_position[:, None].expand(-1, t, -1)
        goal_position = goal_position.reshape(b * t, 3)
        initial_ego_state = batch["ego_state"].reshape(b * t, -1).float()

        # World-model rollout and sampled actions are targets for policy/value
        # optimization, matching the detached imagination path in DreamerV3.
        with torch.no_grad():
            _, imagined = self._imagine_goal_conditioned(
                start, int(imag_horizon) + 1, mask, goal_position, sample=True,
                initial_ego_state=initial_ego_state,
                collect_states=False,
            )
            imag_feat = imagined["joint_feat"].detach()
            imag_actor_feat = imagined["actor_feat"].detach()
            imag_action = imagined["action"].detach()
            imag_policy_action = imagined["policy_action"].detach()
            imag_reward = self.policy_reward(imag_feat)
            imag_cont = self.cont(imag_feat).mean
            imag_value = self._dist_mode(self.value(imag_feat))
            imag_slow_value = self._dist_mode(self.slow_value(imag_feat))
            discount = 1.0 - 1.0 / self.horizon
            weight = torch.cumprod(imag_cont * discount, dim=1)
            ret = self.lambda_return(
                torch.zeros_like(imag_cont), 1.0 - imag_cont, imag_reward,
                imag_value, imag_value, discount, self.lamb,
            )
            ret_offset, ret_scale = self.return_ema(ret)
            advantage = (ret - imag_value[:, :-1]) / ret_scale
            safety_rollouts = [{
                "feature": imag_feat,
                "actor_feature": imag_actor_feat,
                "action": imag_action,
                "policy_action": imag_policy_action,
                "continuation": imag_cont,
                "weight": weight,
            }]
            for _ in range(1, self.safety_rollout_samples):
                _, alternative = self._imagine_goal_conditioned(
                    start, int(imag_horizon) + 1, mask, goal_position,
                    initial_ego_state=initial_ego_state,
                    sample=True, collect_states=False)
                alternative_feature = alternative["joint_feat"].detach()
                alternative_actor_feature = alternative[
                    "actor_feat"].detach()
                alternative_action = alternative["action"].detach()
                alternative_policy_action = alternative[
                    "policy_action"].detach()
                alternative_continuation = self.cont(
                    alternative_feature).mean
                alternative_weight = torch.cumprod(
                    alternative_continuation * discount, dim=1)
                safety_rollouts.append({
                    "feature": alternative_feature,
                    "actor_feature": alternative_actor_feature,
                    "action": alternative_action,
                    "policy_action": alternative_policy_action,
                    "continuation": alternative_continuation,
                    "weight": alternative_weight,
                })
            safety_components = [
                self.safety_cost_components(
                    rollout["feature"], rollout["action"],
                    detach_risk_parameters=True)
                for rollout in safety_rollouts
            ]
            safety_returns = torch.stack([
                self.discounted_safety_return(
                    components["total"], rollout["continuation"], discount,
                    self.safety_return_lambda)
                for components, rollout in zip(
                    safety_components, safety_rollouts)
            ], dim=1)
            # Baseline and scale are computed only across alternative futures
            # from the same latent start, never across unrelated states.
            safety_center = safety_returns.mean(dim=1, keepdim=True)
            safety_scale = safety_returns.std(
                dim=1, keepdim=True, unbiased=False).clamp_min(0.05)
            safety_advantage = (
                safety_returns - safety_center
            ) / safety_scale

        policy_dist = self.actor(imag_actor_feat)
        if not hasattr(policy_dist, "log_prob"):
            raise TypeError("Actor must return a distribution for Dreamer policy loss")
        log_prob = policy_dist.log_prob(imag_policy_action)[:, :-1]
        entropy = policy_dist.entropy()[:, :-1]
        if log_prob.ndim == 2:
            log_prob = log_prob[..., None]
        if entropy.ndim == 2:
            entropy = entropy[..., None]
        policy_loss = (
            weight[:, :-1].detach()
            * -(log_prob * advantage.detach() + self.act_entropy * entropy)
        ).mean()
        # Minimize the future state-safety cost. The cost at imagined state
        # t+1 is assigned to action t, so action causality flows through the
        # RSSM transition even when the risk estimator mostly reads state.
        safety_log_probs = [log_prob]
        for rollout in safety_rollouts[1:]:
            rollout_distribution = self.actor(rollout["actor_feature"])
            rollout_log_prob = rollout_distribution.log_prob(
                rollout["policy_action"])[:, :-1]
            if rollout_log_prob.ndim == 2:
                rollout_log_prob = rollout_log_prob[..., None]
            safety_log_probs.append(rollout_log_prob)
        stacked_safety_log_prob = torch.stack(safety_log_probs, dim=1)
        stacked_safety_weight = torch.stack([
            rollout["weight"][:, :-1] for rollout in safety_rollouts
        ], dim=1)
        safety_policy_loss = (
            stacked_safety_weight.detach()
            * stacked_safety_log_prob * safety_advantage.detach()
        ).mean()

        value_dist = self.value(imag_feat)
        padded_return = torch.cat((ret, torch.zeros_like(ret[:, -1:])), dim=1)
        value_nll = -value_dist.log_prob(padded_return.detach())
        slow_nll = -value_dist.log_prob(imag_slow_value.detach())
        if value_nll.ndim == 2:
            value_nll = value_nll[..., None]
            slow_nll = slow_nll[..., None]
        value_loss = (weight[:, :-1].detach() * (value_nll + slow_nll)[:, :-1]).mean()

        # Replay value keeps gradients through posterior joint features and thus
        # through both world-model branches and their policy interaction.
        replay_feat = posterior_aux["joint_feat"]
        replay_value = self._dist_mode(self.value(replay_feat))
        replay_slow = self._dist_mode(self.slow_value(replay_feat))
        boot = ret[:, 0].reshape(b, t, *ret.shape[2:])
        replay_return = self.lambda_return(
            batch["is_last"].float(), batch["is_terminal"].float(),
            self.policy_reward_target(batch),
            replay_value.detach(), boot.detach(), 1.0 - 1.0 / self.horizon, self.lamb,
        )
        replay_padded = torch.cat((replay_return, torch.zeros_like(replay_return[:, -1:])), dim=1)
        replay_dist = self.value(replay_feat)
        replay_nll = -replay_dist.log_prob(replay_padded.detach())
        replay_slow_nll = -replay_dist.log_prob(replay_slow.detach())
        if replay_nll.ndim == 2:
            replay_nll = replay_nll[..., None]
            replay_slow_nll = replay_slow_nll[..., None]
        replay_weight = 1.0 - batch["is_last"].float()
        repval_loss = (replay_weight[:, :-1] * (replay_nll + replay_slow_nll)[:, :-1]).mean()
        behavior_clone, behavior_metrics = self.behavior_clone_objective(
            posterior_aux, batch)
        safety_losses, safety_metrics = self.actor_safety_objective(imag_feat)
        replay_speed_loss, replay_speed_metrics = self.replay_unsafe_speed_objective(
            posterior_aux, batch)
        lateral_candidate_loss, lateral_candidate_metrics = (
            self.lateral_candidate_objective(
                posterior_states, posterior_aux, batch)
        )
        goal_directed_loss, goal_directed_metrics = (
            self.goal_directed_objective(posterior_aux, batch)
        )
        return {
            "policy": policy_loss,
            "safety_policy": safety_policy_loss,
            "behavior_clone": behavior_clone,
            **safety_losses,
            "replay_unsafe_speed": replay_speed_loss,
            "lateral_candidate": lateral_candidate_loss,
            "goal_directed": goal_directed_loss,
            "value": value_loss,
            "repval": repval_loss,
        }, {
            "imag_reward": imag_reward.mean(), "imag_continue": imag_cont.mean(),
            "imag_value": imag_value.mean(), "action_entropy": entropy.mean(),
            "imag_return": ret.mean(), "return_normalized": ((ret - ret_offset) / ret_scale).mean(),
            "return_005": self.return_ema.ema_vals[0], "return_095": self.return_ema.ema_vals[1],
            "advantage": advantage.mean(), "advantage_std": advantage.std(),
            "safety_cost": torch.stack([
                components["total"].mean()
                for components in safety_components]).mean(),
            "safety_collision_cost": torch.stack([
                components["collision"].mean()
                for components in safety_components]).mean(),
            "safety_clearance_cost": torch.stack([
                components["clearance"].mean()
                for components in safety_components]).mean(),
            "safety_speed_cost": torch.stack([
                components["speed"].mean()
                for components in safety_components]).mean(),
            "safety_return": safety_returns.mean(),
            "safety_return_std": safety_returns.std(),
            "safety_counterfactual_cost_span": (
                safety_returns.max(dim=1).values
                - safety_returns.min(dim=1).values).mean(),
            "safety_advantage": safety_advantage.mean(),
            **{f"bc_{key}": value for key, value in behavior_metrics.items()},
            **{f"safety_{key}": value for key, value in safety_metrics.items()},
            **{
                f"replay_safety_{key}": value
                for key, value in replay_speed_metrics.items()
            },
            **{
                f"candidate_{key}": value
                for key, value in lateral_candidate_metrics.items()
            },
            **{
                f"goal_directed_{key}": value
                for key, value in goal_directed_metrics.items()
            },
        }

    def actor_critic_loss_v62(
        self,
        posterior_states: FactorizedState,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
        imag_horizon: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Dual-critic score-function Dreamer objective for v6.2.

        The frozen imagined trajectory provides task and safety return targets.
        Actor gradients flow only through the exact Tanh-Normal log probability;
        they never backpropagate into RSSM or the detached risk diagnostic head.
        """
        b, t = batch["action"].shape[:2]
        start = self._flatten_state_time(posterior_states)
        human_mask = posterior_aux["human_mask"].reshape(b * t, -1)
        goal_position = batch["goal_position"]
        if goal_position.ndim == 2:
            goal_position = goal_position[:, None].expand(-1, t, -1)
        goal_position = goal_position.reshape(b * t, 3)
        initial_ego_state = batch["ego_state"].reshape(b * t, -1).float()
        discount = 1.0 - 1.0 / self.horizon

        with torch.no_grad():
            _, imagined = self._imagine_goal_conditioned(
                start, int(imag_horizon) + 1, human_mask, goal_position,
                sample=True, initial_ego_state=initial_ego_state,
                collect_states=False)
            feature = imagined["joint_feat"].detach()
            actor_feature = imagined["actor_feat"].detach()
            policy_action = imagined["policy_action"].detach()
            applied_action = imagined["action"].detach()
            continuation = self.cont(feature).mean
            weight = torch.cumprod(continuation * discount, dim=1)

            task_reward = self.policy_reward(feature)
            task_value = self._dist_mode(self.value(feature))
            task_return = self.lambda_return(
                torch.zeros_like(continuation),
                1.0 - continuation,
                task_reward, task_value, task_value,
                discount, self.lamb)
            _, task_scale = self.return_ema(task_return)
            task_advantage = (
                task_return - task_value[:, :-1]
            ) / task_scale

            safety_components = self.safety_cost_components(
                feature, applied_action, detach_risk_parameters=True)
            safety_cost = safety_components["total"]
            safety_value = self._dist_mode(self.safety_value(feature))
            safety_return = self.lambda_return(
                torch.zeros_like(continuation),
                1.0 - continuation,
                safety_cost, safety_value, safety_value,
                discount, self.safety_return_lambda)
            _, safety_scale = self.safety_return_ema(safety_return)
            safety_advantage = (
                safety_return - safety_value[:, :-1]
            ) / safety_scale
            combined_advantage = (
                task_advantage
                - self.safety_lambda.detach() * safety_advantage)

        distribution = self.actor(actor_feature)
        log_prob = distribution.log_prob(policy_action)[:, :-1]
        entropy = distribution.entropy()[:, :-1]
        if log_prob.ndim == 2:
            log_prob = log_prob[..., None]
            entropy = entropy[..., None]
        policy_loss = (
            weight[:, :-1].detach()
            * -(
                log_prob * combined_advantage.detach()
                + self.act_entropy * entropy)
        ).mean()

        task_value_dist = self.value(feature)
        task_target = torch.cat((
            task_return, torch.zeros_like(task_return[:, -1:])), dim=1)
        task_value_nll = -task_value_dist.log_prob(task_target.detach())
        if task_value_nll.ndim == 2:
            task_value_nll = task_value_nll[..., None]
        value_loss = (
            weight[:, :-1].detach() * task_value_nll[:, :-1]).mean()

        safety_value_dist = self.safety_value(feature)
        safety_target = torch.cat((
            safety_return, torch.zeros_like(safety_return[:, -1:])), dim=1)
        safety_value_nll = -safety_value_dist.log_prob(
            safety_target.detach())
        if safety_value_nll.ndim == 2:
            safety_value_nll = safety_value_nll[..., None]
        safety_value_loss = (
            weight[:, :-1].detach()
            * safety_value_nll[:, :-1]).mean()

        behavior_clone, behavior_metrics = self.behavior_clone_objective(
            posterior_aux, batch)
        posterior_distribution = self.actor(
            posterior_aux["actor_feat"][:, :-1].detach())
        gate = posterior_distribution.conflict_gate
        danger = self._local_human_danger_mask(
            batch, gate.shape[1]).to(gate.dtype)
        gate_teacher_loss = F.binary_cross_entropy_with_logits(
            posterior_distribution.conflict_gate_logit, danger)

        zero = policy_loss.detach() * 0.0
        losses = {
            "policy": policy_loss,
            "value": value_loss,
            "safety_value": safety_value_loss,
            "behavior_clone": behavior_clone,
            "gate_teacher": gate_teacher_loss,
            # Explicitly retire the legacy direct action-shaping objectives.
            "safety_policy": zero,
            "collision_risk": zero,
            "clearance_barrier": zero,
            "unsafe_speed": zero,
            "replay_unsafe_speed": zero,
            "vertical_action": zero,
            "lateral_candidate": zero,
            "action_smoothness": zero,
            "goal_directed": zero,
            "repval": zero,
        }
        metrics = {
            "imag_reward": task_reward.mean(),
            "imag_continue": continuation.mean(),
            "imag_value": task_value.mean(),
            "imag_safety_value": safety_value.mean(),
            "imag_return": task_return.mean(),
            "safety_return": safety_return.mean(),
            "task_advantage_std": task_advantage.std(),
            "safety_advantage_std": safety_advantage.std(),
            "combined_advantage_mean": combined_advantage.mean(),
            "safety_lambda": self.safety_lambda.detach(),
            "action_entropy": entropy.mean(),
            "conflict_gate_mean": gate.mean(),
            "conflict_gate_danger_mean": self._masked_mean(
                gate, danger.bool()),
            "conflict_gate_safe_mean": self._masked_mean(
                gate, ~danger.bool()),
            **{f"bc_{key}": value for key, value in behavior_metrics.items()},
        }
        return losses, metrics

    def actor_critic_loss_v63(
        self,
        posterior_states: FactorizedState,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
        imag_horizon: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """v6.3 score-function objective with one physical safety contract."""
        if (
            not self.v63_enabled
            or self.transition_event is None
            or self.next_human_clearance is None
            or not isinstance(self.safety_value, HumanSafetyCritic)
            or not isinstance(self.slow_safety_value, HumanSafetyCritic)
        ):
            raise RuntimeError("v6.3 actor loss requires all v6.3 safety modules")
        b, t = batch["action"].shape[:2]
        flat_state = self._flatten_state_time(posterior_states)
        start_valid = ~batch["is_last"].bool().reshape(b * t, -1).any(
            -1)
        start_indices = start_valid.nonzero(as_tuple=False).squeeze(-1)
        if start_indices.numel() == 0:
            raise ValueError("v6.3 batch contains no non-terminal imagination starts")
        start = self._select_state_rows(flat_state, start_indices)
        human_mask = posterior_aux["human_mask"].reshape(
            b * t, -1).index_select(0, start_indices)
        goal_position = batch["goal_position"]
        if goal_position.ndim == 2:
            goal_position = goal_position[:, None].expand(-1, t, -1)
        goal_position = goal_position.reshape(b * t, 3).index_select(
            0, start_indices)
        initial_ego_state = batch["ego_state"].reshape(
            b * t, -1).float().index_select(0, start_indices)
        discount = 1.0 - 1.0 / self.horizon
        rollout_count = max(2, int(self.safety_rollout_samples))
        rollouts: list[dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for _ in range(rollout_count):
                _, imagined = self._imagine_goal_conditioned(
                    start,
                    int(imag_horizon) + 1,
                    human_mask,
                    goal_position,
                    sample=True,
                    initial_ego_state=initial_ego_state,
                    collect_states=False,
                )
                feature = imagined["joint_feat"].detach()
                actor_feature = imagined["actor_feat"].detach()
                policy_action = imagined["policy_action"].detach()
                applied_action = imagined["action"].detach()
                current_feature = feature[:, :-1]
                next_feature = feature[:, 1:]
                event = self.transition_event(
                    current_feature,
                    applied_action[:, :-1],
                    detach_parameters=True,
                )
                continue_probability = event["continue_probability"]
                human_hazard = event["human_collision_probability"]
                task_reward = self.policy_reward_v63(
                    next_feature, event["probability"])
                next_slow_value = self._dist_mode(
                    self.slow_value(next_feature))
                task_return = self._v63_task_lambda_return(
                    task_reward,
                    continue_probability,
                    next_slow_value,
                    discount=discount,
                    lamb=self.lamb,
                )
                safety_bootstrap = self.slow_safety_value(
                    feature[:, -1])
                safety_return = survival_human_collision_return(
                    human_hazard,
                    continue_probability,
                    safety_bootstrap,
                )
                prefix = torch.cat((
                    torch.ones_like(continue_probability[:, :1]),
                    torch.cumprod(
                        continue_probability[:, :-1] * discount,
                        dim=1,
                    ),
                ), dim=1)
                rollouts.append({
                    "feature": current_feature,
                    "actor_feature": actor_feature[:, :-1],
                    "policy_action": policy_action[:, :-1],
                    "task_reward": task_reward,
                    "task_return": task_return,
                    "safety_return": safety_return,
                    "human_hazard": human_hazard,
                    "continue_probability": continue_probability,
                    "weight": prefix,
                    "event_probability": event["probability"],
                })

            task_return = torch.cat(
                [item["task_return"] for item in rollouts], dim=0)
            safety_return = torch.cat(
                [item["safety_return"] for item in rollouts], dim=0)
            feature = torch.cat(
                [item["feature"] for item in rollouts], dim=0)
            task_value = self._dist_mode(self.value(feature))
            safety_value = self.safety_value(feature)
            _, task_scale = self.return_ema(task_return)
            task_advantage = (task_return - task_value) / task_scale
            # Safety remains in physical episode-collision probability units.
            safety_advantage = safety_return - safety_value
            combined_advantage = (
                task_advantage
                - self.safety_lambda.detach() * safety_advantage)

        actor_feature = torch.cat(
            [item["actor_feature"] for item in rollouts], dim=0)
        sampled_policy_action = torch.cat(
            [item["policy_action"] for item in rollouts], dim=0)
        weight = torch.cat([item["weight"] for item in rollouts], dim=0)
        distribution = self.actor(actor_feature)
        log_prob = distribution.log_prob(sampled_policy_action)[..., None]
        entropy = distribution.entropy()[..., None]
        support_limits = sampled_policy_action.new_tensor(
            self.imagination_support_delta_limits)
        if support_limits.numel() != sampled_policy_action.shape[-1]:
            raise ValueError("support limits must match policy action dimension")
        mode_action = self._dist_mode(distribution)
        sampled_delta = (sampled_policy_action - mode_action).abs()
        support_violation = sampled_delta > support_limits
        support_mask = (~support_violation.any(-1, keepdim=True)).to(
            weight.dtype)
        policy_weight = weight.detach() * support_mask
        policy_loss = (
            policy_weight
            * -(
                log_prob * combined_advantage.detach()
                + self.act_entropy * entropy
            )
        ).sum() / policy_weight.sum().clamp_min(1.0)
        supported_std = support_limits / 1.645
        support_policy_loss = F.relu(
            distribution.stddev - supported_std).square().mean()

        task_value_dist = self.value(feature)
        task_value_nll = -task_value_dist.log_prob(task_return.detach())
        if task_value_nll.ndim == 2:
            task_value_nll = task_value_nll[..., None]
        value_loss = (weight.detach() * task_value_nll).mean()

        safety_logit = self.safety_value.logits(feature)
        safety_value_bce = F.binary_cross_entropy_with_logits(
            safety_logit, safety_return.detach(), reduction="none")
        safety_value_loss = (weight.detach() * safety_value_bce).mean()

        behavior_clone, behavior_metrics = self.behavior_clone_objective(
            posterior_aux, batch)
        mode_support_loss, mode_support_metrics = (
            self.behavior_mode_support_objective(posterior_aux, batch))
        posterior_distribution = self.actor(
            posterior_aux["actor_feat"][:, :-1].detach())
        gate = posterior_distribution.conflict_gate
        danger = self._local_human_danger_mask(
            batch, gate.shape[1]).to(gate.dtype)
        gate_teacher_loss = F.binary_cross_entropy_with_logits(
            posterior_distribution.conflict_gate_logit, danger)

        zero = policy_loss.detach() * 0.0
        losses = {
            "policy": policy_loss,
            "value": value_loss,
            "safety_value": safety_value_loss,
            "behavior_clone": behavior_clone,
            "gate_teacher": gate_teacher_loss,
            "support_policy": support_policy_loss,
            "mode_support": mode_support_loss,
            "safety_policy": zero,
            "collision_risk": zero,
            "clearance_barrier": zero,
            "unsafe_speed": zero,
            "replay_unsafe_speed": zero,
            "vertical_action": zero,
            "lateral_candidate": zero,
            "action_smoothness": zero,
            "goal_directed": zero,
            "repval": zero,
        }
        safety_by_rollout = torch.stack(
            [item["safety_return"] for item in rollouts], dim=1)
        task_by_rollout = torch.stack(
            [item["task_return"] for item in rollouts], dim=1)
        event_probability = torch.cat(
            [item["event_probability"] for item in rollouts], dim=0)
        metrics = {
            "imag_reward": torch.cat(
                [item["task_reward"] for item in rollouts], dim=0).mean(),
            "imag_continue": torch.cat(
                [item["continue_probability"] for item in rollouts], dim=0).mean(),
            "imag_human_hazard": torch.cat(
                [item["human_hazard"] for item in rollouts], dim=0).mean(),
            "imag_value": task_value.mean(),
            "imag_safety_value": safety_value.mean(),
            "imag_return": task_return.mean(),
            "safety_return_probability": safety_return.mean(),
            "safety_return_min": safety_return.min(),
            "safety_return_max": safety_return.max(),
            "task_advantage_std": task_advantage.std(),
            "safety_advantage_std": safety_advantage.std(),
            "combined_advantage_mean": combined_advantage.mean(),
            "safety_lambda": self.safety_lambda.detach(),
            "action_entropy": entropy.mean(),
            "rollout_samples": weight.new_tensor(float(rollout_count)),
            "counterfactual_safety_span": (
                safety_by_rollout.max(dim=1).values
                - safety_by_rollout.min(dim=1).values).mean(),
            "counterfactual_task_span": (
                task_by_rollout.max(dim=1).values
                - task_by_rollout.min(dim=1).values).mean(),
            "terminal_start_excluded_ratio": 1.0 - start_valid.float().mean(),
            "event_probability_sum_error": (
                event_probability.sum(-1) - 1.0).abs().mean(),
            "support_violation_ratio": support_violation.float().mean(),
            "support_accepted_sample_ratio": support_mask.mean(),
            "support_rejected_sample_ratio": 1.0 - support_mask.mean(),
            "support_violation_excess": F.relu(
                support_violation.float().mean()
                - self.imagination_support_violation_limit),
            "conflict_gate_mean": gate.mean(),
            "conflict_gate_danger_mean": self._masked_mean(
                gate, danger.bool()),
            "conflict_gate_safe_mean": self._masked_mean(
                gate, ~danger.bool()),
            **{f"bc_{key}": value for key, value in behavior_metrics.items()},
            **{
                f"mode_support_{key}": value
                for key, value in mode_support_metrics.items()
            },
        }
        action_names = ("vx", "vy", "yaw")
        for index in range(sampled_policy_action.shape[-1]):
            name = action_names[index] if index < len(action_names) else str(index)
            metrics[f"actor_std/{name}"] = distribution.stddev[..., index].mean()
            metrics[f"support_violation/{name}"] = (
                support_violation[..., index].float().mean())
        return losses, metrics

    @torch.no_grad()
    def actor_imagination_diagnostics(
        self,
        posterior_states: FactorizedState,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
        imag_horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Validate the Actor deterministically inside the learned world.

        Unlike the stochastic training objective, this path does not update
        return normalization or the slow critic. It exists so TensorBoard and
        best-checkpoint selection measure the internal imagination mechanism,
        instead of selecting an Actor only by behavior-cloning error.
        """
        b, t = batch["action"].shape[:2]
        start = self._flatten_state_time(posterior_states)
        human_mask = posterior_aux["human_mask"].reshape(b * t, -1)
        goal_position = batch["goal_position"]
        if goal_position.ndim == 2:
            goal_position = goal_position[:, None].expand(-1, t, -1)
        goal_position = goal_position.reshape(b * t, 3)
        ego_state = batch["ego_state"].reshape(b * t, -1).float()
        _, imagined = self._imagine_goal_conditioned(
            start,
            int(imag_horizon),
            human_mask,
            goal_position,
            initial_ego_state=ego_state,
            sample=False,
            collect_states=False,
        )
        feature = imagined["joint_feat"]
        action = imagined["action"]
        discount = 1.0 - 1.0 / self.horizon
        if self.v63_enabled:
            current_feature = feature[:, :-1]
            next_feature = feature[:, 1:]
            current_action = action[:, :-1]
            event = self.transition_event(
                current_feature, current_action,
                detach_parameters=True)
            continuation = event["continue_probability"]
            reward = self.policy_reward_v63(
                next_feature, event["probability"])
            value = self._dist_mode(self.value(next_feature))
            safety_return = survival_human_collision_return(
                event["human_collision_probability"],
                continuation,
                self.slow_safety_value(feature[:, -1]),
            )
            collision_probability = event["human_collision_probability"]
            clearance = self.next_human_clearance(
                current_feature, current_action,
                detach_parameters=True)
            diagnostic_action = current_action
        else:
            reward = self.policy_reward(feature)
            continuation = self.cont(feature).mean
            value = self._dist_mode(self.value(feature))
            safety = self.safety_cost_components(
                feature, action, detach_risk_parameters=True)
            safety_return = safety["total"]
            collision_probability = safety["collision"]
            clearance = safety["clearance_m"]
            diagnostic_action = action
        weights = torch.cat((
            torch.ones_like(continuation[:, :1]),
            torch.cumprod(continuation[:, :-1] * discount, dim=1),
        ), dim=1)
        denominator = weights.sum().clamp_min(1.0)
        task_return = (
            weights * reward
        ).sum(dim=1) + weights[:, -1] * discount * value[:, -1]
        return {
            "reward_mean": (weights * reward).sum() / denominator,
            "task_return_mean": task_return.mean(),
            "safety_cost_mean": (
                weights * safety_return).sum() / denominator,
            "collision_probability_mean": (
                weights * collision_probability).sum() / denominator,
            "clearance_m_mean": (
                weights * clearance).sum() / denominator,
            "forward_action_mean": (
                weights * diagnostic_action[..., :1]).sum() / denominator,
            "lateral_action_abs_mean": (
                (weights * diagnostic_action[..., 1:2].abs()).sum()
                / denominator
                if diagnostic_action.shape[-1] > 1
                else diagnostic_action.new_zeros(())
            ),
            "continuation_mean": continuation.mean(),
            "value_mean": value.mean(),
        }

    @staticmethod
    def discounted_safety_return(
        cost: torch.Tensor,
        continuation: torch.Tensor,
        discount: float,
        trace_lambda: float,
    ) -> torch.Tensor:
        """Assign future imagined-state costs to the preceding action."""
        if cost.shape != continuation.shape:
            raise ValueError("cost and continuation must have identical shapes")
        next_cost = cost[:, 1:]
        next_live = continuation[:, 1:] * float(discount) * float(trace_lambda)
        accumulator = torch.zeros_like(next_cost[:, -1])
        outputs = []
        for index in reversed(range(next_cost.shape[1])):
            accumulator = next_cost[:, index] + next_live[:, index] * accumulator
            outputs.append(accumulator)
        return torch.stack(list(reversed(outputs)), dim=1)

    def safety_cost_components(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        detach_risk_parameters: bool,
    ) -> dict[str, torch.Tensor]:
        risk = self.action_risk(
            feature, action, detach_parameters=detach_risk_parameters)
        collision = risk["collision_probability"]
        clearance_m = risk["min_human_clearance_m"]
        clearance = F.softplus(
            (self.planning_safe_clearance_m - clearance_m)
            / self.clearance_temperature_m
        ) * self.clearance_temperature_m / self.planning_safe_clearance_m
        risk_gate = torch.maximum(
            collision,
            torch.sigmoid(
                (self.planning_safe_clearance_m - clearance_m)
                / self.clearance_temperature_m),
        )
        forward_excess = F.relu(action[..., :1] - self.safe_forward_action)
        speed = risk_gate * forward_excess.square()
        total = (
            self.safety_collision_cost * collision
            + self.safety_clearance_cost * clearance
            + self.safety_speed_cost * speed
        )
        return {
            "total": total,
            "collision": collision,
            "clearance": clearance,
            "speed": speed,
            "clearance_m": clearance_m,
            "risk_gate": risk_gate,
        }

    def actor_safety_objective(
        self,
        imagined_feature: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Penalize unsafe candidate actions using a frozen risk estimator."""
        if self.uses_internal_action_adapter:
            # v6.2 obtains safety credit from the dedicated cost Critic.  Keep
            # the old direct risk penalties disabled: they were an external
            # action-shaping shortcut and cannot supply long-horizon causal
            # credit to the factorized Actor.
            zero = imagined_feature.sum() * 0.0
            return {
                name: zero for name in (
                    "collision_risk", "clearance_barrier", "unsafe_speed",
                    "vertical_action", "action_smoothness")
            }, {
                name: zero.detach() for name in (
                    "collision_probability", "predicted_clearance_m",
                    "unsafe_forward_ratio", "vertical_action_abs",
                    "lateral_action_delta", "smoothness_gate_mean")
            }
        feature = imagined_feature[:, :-1].detach()
        distribution = self.actor(feature)
        action = self._dist_mode(distribution).float().clamp(-1.0, 1.0)
        safety = self.safety_cost_components(
            feature, action, detach_risk_parameters=True)
        collision_probability = safety["collision"]
        clearance_m = safety["clearance_m"]
        collision_risk = collision_probability.mean()
        clearance_barrier = safety["clearance"].mean()

        forward_excess = F.relu(
            action[..., :1] - self.safe_forward_action)
        # Direct speed control intentionally treats state risk as a fixed gate.
        # Its useful gradient is d speed / d vx, not the observationally
        # confounded d risk / d action that failed the first 500-step check.
        near_human_gate = safety["risk_gate"].detach()
        unsafe_speed = (near_human_gate * forward_excess.square()).mean()

        # The deployed PX4 interface interprets the third Actor dimension as
        # vertical velocity, not an altitude setpoint.  A seemingly harmless
        # persistent bias of -0.02 therefore accumulates into tens of
        # centimetres of descent over a one-minute episode.  Keep the learned
        # navigation policy centred on zero vertical speed; takeoff is handled
        # before Actor control begins and PX4 holds altitude at zero command.
        if action.shape[-1] > 2:
            vertical_action = action[..., 2:3].square().mean()
            vertical_action_abs = action[..., 2:3].abs().mean()
        else:
            vertical_action = action.sum() * 0.0
            vertical_action_abs = vertical_action.detach()

        if action.shape[1] > 1:
            delta = action[:, 1:] - action[:, :-1]
            dimension_weight = torch.ones(
                action.shape[-1], device=action.device, dtype=action.dtype)
            if action.shape[-1] > 1:
                dimension_weight[1] = 2.0
            # Keep commands smooth in open space, but allow a timely lateral
            # turn when either side of the transition is predicted dangerous.
            transition_risk = torch.maximum(
                safety["risk_gate"][:, 1:], safety["risk_gate"][:, :-1]
            ).detach()
            smoothness_gate = 1.0 - transition_risk * (
                1.0 - self.danger_lateral_smoothness_scale)
            weighted_delta = (
                delta.square() * dimension_weight * smoothness_gate)
            action_smoothness = weighted_delta.sum() / (
                smoothness_gate.sum() * dimension_weight.sum()
            ).clamp_min(1.0)
            lateral_delta = delta[..., 1].abs().mean() if action.shape[-1] > 1 else delta.abs().mean()
        else:
            action_smoothness = action.sum() * 0.0
            lateral_delta = action_smoothness.detach()
            smoothness_gate = action_smoothness.detach().reshape(1)
        return {
            "collision_risk": collision_risk,
            "clearance_barrier": clearance_barrier,
            "unsafe_speed": unsafe_speed,
            "vertical_action": vertical_action,
            "action_smoothness": action_smoothness,
        }, {
            "collision_probability": collision_probability.mean(),
            "predicted_clearance_m": clearance_m.mean(),
            "unsafe_forward_ratio": (
                (near_human_gate > 0.5) & (forward_excess > 0.0)
            ).float().mean(),
            "vertical_action_abs": vertical_action_abs,
            "lateral_action_delta": lateral_delta,
            "smoothness_gate_mean": smoothness_gate.mean(),
        }

    def replay_unsafe_speed_objective(
        self,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.uses_internal_action_adapter:
            zero = posterior_aux["joint_feat"].sum() * 0.0
            return zero, {"active_transition_ratio": zero.detach()}
        """Enforce a real-data speed floor in observed dangerous states.

        This objective intentionally makes no claim about the correct lateral
        direction. It supplies the causal invariant that was missing from the
        observational action-risk head: when recorded future collision is
        imminent or surface clearance is already unsafe, do not keep a high
        forward command.
        """
        feature = posterior_aux["joint_feat"][:, :-1].detach()
        action = self._dist_mode(self.actor(feature)).float().clamp(-1.0, 1.0)
        valid = ~batch["is_last"][:, :-1].bool()
        collision = batch.get("future_collision")
        if collision is None:
            collision_gate = torch.zeros_like(valid, dtype=action.dtype)
        else:
            collision_gate = collision[:, :-1].float()
        clearance = batch.get("future_min_human_clearance_m")
        clearance_valid = batch.get("future_min_human_clearance_valid")
        if clearance is None or clearance_valid is None:
            clearance_gate = torch.zeros_like(valid, dtype=action.dtype)
        else:
            clearance_gate = (
                (self.safe_clearance_m - clearance[:, :-1].float())
                / self.safe_clearance_m
            ).clamp(0.0, 1.0) * clearance_valid[:, :-1].float()
        danger_gate = torch.maximum(
            collision_gate, clearance_gate) * valid.float()
        forward_excess = F.relu(
            action[..., :1] - self.safe_forward_action)
        loss = (
            danger_gate * forward_excess.square()
        ).sum() / danger_gate.sum().clamp_min(1.0)
        active = danger_gate > 0.0
        return loss, {
            "active_transition_ratio": active.float().mean(),
            "forward_action_mean": self._masked_mean(
                action[..., :1], active),
            "forward_excess_mean": self._masked_mean(
                forward_excess, active),
        }

    @staticmethod
    def _select_state_rows(
        state: FactorizedState, indices: torch.Tensor,
    ) -> FactorizedState:
        return {
            branch: {
                key: value.index_select(0, indices)
                for key, value in values.items()
            }
            for branch, values in state.items()
        }

    @staticmethod
    def _repeat_state_candidates(
        state: FactorizedState, candidate_count: int,
    ) -> FactorizedState:
        return {
            branch: {
                key: value[:, None].expand(
                    value.shape[0], candidate_count, *value.shape[1:]
                ).reshape(value.shape[0] * candidate_count, *value.shape[1:])
                for key, value in values.items()
            }
            for branch, values in state.items()
        }

    def _local_human_danger_mask(
        self,
        batch: dict[str, torch.Tensor],
        source_length: int,
    ) -> torch.Tensor:
        """Return local/imminent danger, excluding a distant crowd ahead.

        The recorded 2.5 s collision label is useful for risk calibration but
        is too broad for directional Actor supervision: it taught the policy
        to leave the whole crowd long before a particular pedestrian became a
        conflict. Lateral actions are enabled only by current clearance or an
        imminent factual collision/TTC.
        """
        reference = batch["is_last"][:, :source_length].bool()
        reference = reference.reshape(*reference.shape[:2], -1).any(
            -1, keepdim=True)
        danger = torch.zeros_like(reference)

        clearance = batch.get("priv_min_human_clearance_m")
        clearance_valid = batch.get("priv_min_human_clearance_valid")
        if clearance is not None and clearance_valid is not None:
            current_clearance = clearance[:, :source_length].float().reshape(
                *reference.shape[:2], -1)
            current_valid = clearance_valid[:, :source_length].bool().reshape(
                *reference.shape[:2], -1).all(-1, keepdim=True)
            danger = danger | (
                current_valid
                & (current_clearance.amin(-1, keepdim=True)
                   <= self.lateral_candidate_local_clearance_m)
            )
        else:
            current_clearance = None
            current_valid = None

        collision = batch.get("future_collision")
        collision_ttc = batch.get("time_to_collision_s")
        collision_ttc_valid = batch.get("time_to_collision_valid")
        if (
            collision is not None
            and collision_ttc is not None
            and collision_ttc_valid is not None
        ):
            collision = collision[:, :source_length].bool().reshape(
                *reference.shape[:2], -1).any(-1, keepdim=True)
            collision_ttc = collision_ttc[:, :source_length].float().reshape(
                *reference.shape[:2], -1).amin(-1, keepdim=True)
            collision_ttc_valid = collision_ttc_valid[
                :, :source_length].bool().reshape(
                    *reference.shape[:2], -1).all(-1, keepdim=True)
            danger = danger | (
                collision & collision_ttc_valid
                & (collision_ttc <= self.lateral_candidate_imminent_ttc_s)
            )

        current_ttc = batch.get("priv_min_human_ttc_s")
        current_ttc_valid = batch.get("priv_min_human_ttc_valid")
        if (
            current_ttc is not None
            and current_ttc_valid is not None
            and current_clearance is not None
            and current_valid is not None
        ):
            current_ttc = current_ttc[:, :source_length].float().reshape(
                *reference.shape[:2], -1).amin(-1, keepdim=True)
            current_ttc_valid = current_ttc_valid[
                :, :source_length].bool().reshape(
                    *reference.shape[:2], -1).all(-1, keepdim=True)
            danger = danger | (
                current_valid & current_ttc_valid
                & (current_ttc <= self.lateral_candidate_imminent_ttc_s)
                & (current_clearance.amin(-1, keepdim=True)
                   <= 1.5 * self.lateral_candidate_local_clearance_m)
            )
        return danger

    def _kinematic_candidate_clearance(
        self,
        skeleton: torch.Tensor,
        joint_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Short-horizon constant-velocity clearance for candidate actions.

        Skeleton velocity is human world motion expressed in current body
        axes. Subtracting the candidate UAV velocity therefore yields future
        relative joint positions. This gives an explicitly action-sensitive
        target without branching or resetting Isaac Sim.
        """
        if skeleton.ndim != 4 or candidates.ndim != 3:
            raise ValueError("expected skeleton [B,N,J,F] and candidates [B,C,A]")
        xyz = skeleton[..., :3].float()
        velocity = skeleton[..., 3:6].float()
        valid = joint_mask.bool()
        has_human = valid.reshape(valid.shape[0], -1).any(-1)
        action_scale = candidates.new_tensor((2.15, 2.15, 1.0))
        drone_velocity = candidates.new_zeros(
            *candidates.shape[:-1], 3)
        used_dimensions = min(3, candidates.shape[-1])
        drone_velocity[..., :used_dimensions] = (
            candidates[..., :used_dimensions]
            * action_scale[:used_dimensions]
        )
        minimum = candidates.new_full(
            candidates.shape[:2], float("inf"))
        for step in range(1, self.lateral_candidate_geometry_steps + 1):
            delta_t = step * self.lateral_candidate_step_duration_s
            relative = (
                xyz[:, None]
                + velocity[:, None] * delta_t
                - drone_velocity[:, :, None, None, :] * delta_t
            )
            distance = torch.linalg.vector_norm(relative, dim=-1)
            distance = distance.masked_fill(~valid[:, None], float("inf"))
            minimum = torch.minimum(
                minimum, distance.flatten(2).amin(-1))
        clearance = minimum - self.lateral_candidate_surface_radius_m
        clearance = torch.where(
            has_human[:, None], clearance,
            clearance.new_full(clearance.shape, self.risk_clearance_normalization_m),
        )
        return clearance.clamp(
            min=-self.lateral_candidate_surface_radius_m,
            max=self.risk_clearance_normalization_m,
        ), has_human

    def counterfactual_action_risk_objective(
        self,
        joint_feature: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Teach the risk head that left/right actions have different effects."""
        feature = joint_feature[:, :-1].detach()
        skeleton = batch["skeleton"][:, :-1].float()
        joint_mask = batch["joint_mask"][:, :-1].bool()
        flat_feature = feature.reshape(-1, feature.shape[-1])
        flat_skeleton = skeleton.reshape(
            -1, *skeleton.shape[2:])
        flat_mask = joint_mask.reshape(-1, *joint_mask.shape[2:])
        current_distance = torch.linalg.vector_norm(
            flat_skeleton[..., :3], dim=-1)
        current_distance = current_distance.masked_fill(
            ~flat_mask, float("inf"))
        current_clearance = (
            current_distance.flatten(1).amin(-1)
            - self.lateral_candidate_surface_radius_m)
        action_valid = batch.get("action_valid")
        if action_valid is None:
            source_valid = torch.ones_like(
                batch["is_last"][:, :-1], dtype=torch.bool)
        else:
            source_valid = action_valid[:, 1:].bool()
        source_valid = source_valid.reshape(
            source_valid.shape[0], source_valid.shape[1], -1).all(-1)
        source_live = ~batch["is_last"][:, :-1].bool().reshape(
            batch["is_last"].shape[0], batch["is_last"].shape[1] - 1,
            -1).any(-1)
        active = (
            (current_clearance
             <= 1.5 * self.lateral_candidate_local_clearance_m)
            & source_valid.reshape(-1)
            & source_live.reshape(-1)
        )
        indices = active.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            zero = flat_feature.sum() * 0.0
            return zero, {"active_transition_ratio": active.float().mean()}

        selected_feature = flat_feature.index_select(0, indices)
        selected_skeleton = flat_skeleton.index_select(0, indices)
        selected_mask = flat_mask.index_select(0, indices)
        candidate_count = 4
        actions = selected_feature.new_zeros(
            selected_feature.shape[0], candidate_count,
            batch["action"].shape[-1])
        actions[:, 0, 0] = self.goal_directed_forward_action
        actions[:, 1, 0] = self.lateral_candidate_slow_forward_action
        actions[:, 2:, 0] = self.safe_forward_action
        actions[:, 2, 1] = self.lateral_candidate_action
        actions[:, 3, 1] = -self.lateral_candidate_action
        target_clearance, _ = self._kinematic_candidate_clearance(
            selected_skeleton, selected_mask, actions)
        repeated_feature = selected_feature[:, None].expand(
            -1, candidate_count, -1)
        prediction = self.action_risk(repeated_feature, actions)
        predicted_clearance = prediction["min_human_clearance_m"].squeeze(-1)
        clearance_target = target_clearance.detach().clamp_min(0.0)
        clearance_loss = F.smooth_l1_loss(
            predicted_clearance, clearance_target,
            beta=self.clearance_temperature_m)
        collision_target = (target_clearance <= 0.0).float()
        collision_loss = F.binary_cross_entropy_with_logits(
            prediction["collision_logit"].squeeze(-1), collision_target)
        best_target = target_clearance.argmax(dim=1)
        ranking_loss = F.cross_entropy(
            predicted_clearance / self.clearance_temperature_m,
            best_target)
        loss = clearance_loss + 0.25 * collision_loss + 0.25 * ranking_loss
        return loss, {
            "active_transition_ratio": active.float().mean().detach(),
            "clearance_loss": clearance_loss.detach(),
            "collision_loss": collision_loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "target_action_clearance_span_m": (
                target_clearance.amax(1) - target_clearance.amin(1)
            ).mean().detach(),
            "predicted_action_clearance_span_m": (
                predicted_clearance.amax(1) - predicted_clearance.amin(1)
            ).mean().detach(),
            "best_left_ratio": (best_target == 2).float().mean().detach(),
            "best_right_ratio": (best_target == 3).float().mean().detach(),
        }

    def goal_directed_objective(
        self,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Keep the raw Actor on the shortest goal path outside local danger."""
        feature = posterior_aux[
            "actor_feat" if self.uses_internal_action_adapter
            else "joint_feat"
        ][:, :-1].detach()
        actor_action = self._dist_mode(self.actor(feature)).float().clamp(
            -1.0, 1.0)
        goal = batch["goal"][:, :-1].float()
        distance = goal[..., 3:4]
        unit_x = goal[..., 4:5].clamp(min=0.0, max=1.0)
        unit_y = goal[..., 5:6].clamp(min=-1.0, max=1.0)
        unit_y = torch.where(
            unit_y.abs() < self.goal_directed_deadband,
            torch.zeros_like(unit_y), unit_y)
        speed_scale = ((distance - 1.0) / max(
            self.goal_directed_slow_radius_m - 1.0, 1.0e-3)).clamp(0.0, 1.0)
        target = torch.cat((
            self.goal_directed_forward_action * speed_scale * unit_x,
            self.goal_directed_lateral_action * speed_scale * unit_y,
        ), dim=-1)
        valid = batch.get("action_valid")
        if valid is None:
            valid = torch.ones_like(distance, dtype=torch.bool)
        else:
            valid = valid[:, 1:].bool().reshape(
                *distance.shape[:-1], -1).all(-1, keepdim=True)
        source_live = ~batch["is_last"][:, :-1].bool().reshape(
            *distance.shape[:-1], -1).any(-1, keepdim=True)
        local_danger = self._local_human_danger_mask(
            batch, feature.shape[1])
        active = valid & source_live & ~local_danger & (distance > 1.0)
        regression = F.smooth_l1_loss(
            actor_action[..., :2], target,
            beta=self.behavior_clone_beta, reduction="none")
        weights = regression.new_tensor((0.5, 1.0))
        supervision = active.to(regression.dtype)
        loss = (regression * weights * supervision).sum() / (
            supervision.sum() * weights.sum()).clamp_min(1.0)
        aligned = active & (unit_y.abs() <= self.goal_directed_deadband)
        return loss, {
            "active_transition_ratio": active.float().mean().detach(),
            "local_danger_ratio": local_danger.float().mean().detach(),
            "target_forward_mean": (
                target[..., :1] * active).sum().detach()
                / active.sum().clamp_min(1),
            "target_lateral_abs_mean": (
                target[..., 1:2].abs() * active).sum().detach()
                / active.sum().clamp_min(1),
            "actor_lateral_abs_mean": (
                actor_action[..., 1:2].abs() * active).sum().detach()
                / active.sum().clamp_min(1),
            "aligned_unnecessary_lateral_ratio": (
                aligned & (actor_action[..., 1:2].abs()
                           > self.goal_directed_deadband)
            ).float().sum().detach() / aligned.float().sum().clamp_min(1.0),
        }

    def lateral_candidate_objective(
        self,
        posterior_states: FactorizedState,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Distill a world-model-ranked directional action in danger.

        For each replay state whose privileged 2.5-second label is unsafe,
        compare four commands: hold, slow, left and right. Each command is
        committed only for a short prefix and then the frozen Actor continues
        that imagined branch; the risk head at every imagined state still
        forecasts the full 2.5-second collision horizon.
        The Actor is supervised only when the best and second-best candidates
        have a meaningful score margin. This supplies the left/right target
        that collision labels alone cannot identify.
        """
        if self.uses_internal_action_adapter:
            zero = posterior_aux["actor_feat"].sum() * 0.0
            return zero, {
                "active_transition_ratio": zero.detach(),
                "supervised_danger_ratio": zero.detach(),
                "selected_lateral_ratio": zero.detach(),
            }
        feature = posterior_aux["joint_feat"][:, :-1].detach()
        if feature.shape[-1] == 0:
            zero = feature.sum() * 0.0
            return zero, {"active_transition_ratio": zero.detach()}

        target_action = batch["action"][:, 1:].float()
        if target_action.shape[-1] < 2:
            zero = feature.sum() * 0.0
            return zero, {"active_transition_ratio": zero.detach()}
        valid = batch.get("action_valid")
        if valid is None:
            valid = torch.ones_like(target_action[..., :1], dtype=torch.bool)
        else:
            valid = valid[:, 1:].bool().reshape(
                *target_action.shape[:-1], -1).all(-1, keepdim=True)
        valid = valid & ~batch["is_last"][:, :-1].bool().reshape(
            *target_action.shape[:-1], -1).any(-1, keepdim=True)

        danger = valid & self._local_human_danger_mask(
            batch, feature.shape[1])
        flat_danger = danger.reshape(-1)
        indices = flat_danger.nonzero(as_tuple=False).squeeze(-1)
        active_ratio = danger.float().mean()
        if indices.numel() == 0:
            zero = self._dist_mode(self.actor(feature)).sum() * 0.0
            return zero, {
                "active_transition_ratio": active_ratio.detach(),
                "supervised_danger_ratio": zero.detach(),
                "selected_lateral_ratio": zero.detach(),
            }

        actor_action = self._dist_mode(self.actor(feature)).float().clamp(
            -1.0, 1.0)
        selected_actor_action = actor_action.reshape(
            -1, actor_action.shape[-1]).index_select(0, indices)

        source_states = {
            branch: {
                key: value[:, :-1].reshape(
                    value.shape[0] * (value.shape[1] - 1), *value.shape[2:]
                ).detach()
                for key, value in values.items()
            }
            for branch, values in posterior_states.items()
        }
        selected_state = self._select_state_rows(source_states, indices)
        human_mask = posterior_aux["human_mask"][:, :-1].reshape(
            -1, posterior_aux["human_mask"].shape[-1]
        ).index_select(0, indices).detach()
        goal_position = batch["goal_position"]
        if goal_position.ndim == 2:
            goal_position = goal_position[:, None].expand(
                -1, feature.shape[1], -1)
        else:
            goal_position = goal_position[:, :-1]
        goal_position = goal_position.reshape(-1, 3).index_select(
            0, indices).detach()
        selected_ego_state = batch["ego_state"][:, :-1].reshape(
            -1, batch["ego_state"].shape[-1]
        ).index_select(0, indices).detach()
        selected_skeleton = batch["skeleton"][:, :-1].reshape(
            -1, *batch["skeleton"].shape[2:]
        ).index_select(0, indices).detach()
        selected_joint_mask = batch["joint_mask"][:, :-1].reshape(
            -1, *batch["joint_mask"].shape[2:]
        ).index_select(0, indices).detach()

        candidate_count = 4
        with torch.no_grad():
            base = selected_actor_action.detach()
            candidates = base[:, None].repeat(1, candidate_count, 1)
            # 0=hold, 1=slow, 2=left, 3=right. Lateral candidates also cap
            # forward speed so the teacher does not combine a useful steering
            # direction with the already unsafe high-speed command.
            candidates[:, 1, 0] = torch.minimum(
                candidates[:, 1, 0],
                candidates.new_tensor(
                    self.lateral_candidate_slow_forward_action),
            )
            candidates[:, 2:, 0] = torch.minimum(
                candidates[:, 2:, 0],
                candidates.new_tensor(self.safe_forward_action),
            )
            candidates[:, 2, 1] = self.lateral_candidate_action
            candidates[:, 3, 1] = -self.lateral_candidate_action
            candidates.clamp_(-1.0, 1.0)

            imagined_state = self._repeat_state_candidates(
                selected_state, candidate_count)
            imagined_mask = human_mask[:, None].expand(
                -1, candidate_count, -1).reshape(
                    human_mask.shape[0] * candidate_count,
                    human_mask.shape[-1])
            imagined_goal = goal_position[:, None].expand(
                -1, candidate_count, -1).reshape(-1, 3)
            imagined_action = candidates.reshape(-1, candidates.shape[-1])
            ego_state = selected_ego_state[:, None].expand(
                -1, candidate_count, -1).reshape(
                    selected_ego_state.shape[0] * candidate_count, -1)

            initial_ego_feature = self.rssm.get_branch_feats(
                imagined_state)["ego"]
            decoded_ego = self.prediction_heads.decode_ego_state(
                initial_ego_feature)
            initial_distance = torch.linalg.vector_norm(
                imagined_goal - ego_state[..., :3], dim=-1, keepdim=True)
            collision_costs = []
            clearance_costs = []
            final_distance = initial_distance
            for rollout_step in range(self.lateral_candidate_rollout_steps):
                imagined_state, _ = self.rssm.img_step(
                    imagined_state, imagined_action, imagined_mask)
                ego_feature = self.rssm.get_branch_feats(
                    imagined_state)["ego"]
                next_decoded_ego = self.prediction_heads.decode_ego_state(
                    ego_feature)
                ego_state = self.anchor_ego_displacement(
                    ego_state, decoded_ego, next_decoded_ego)
                decoded_ego = next_decoded_ego
                goal = goal_features_torch(ego_state, imagined_goal)
                joint, _ = self.rssm.get_joint_feat(
                    imagined_state, imagined_mask, goal)
                risk = self.action_risk(
                    joint, imagined_action, detach_parameters=True)
                collision_costs.append(risk["collision_probability"])
                clearance_costs.append(
                    F.softplus(
                        (self.planning_safe_clearance_m
                         - risk["min_human_clearance_m"])
                        / self.clearance_temperature_m
                    ) * self.clearance_temperature_m / self.planning_safe_clearance_m
                )
                final_distance = torch.linalg.vector_norm(
                    imagined_goal - ego_state[..., :3],
                    dim=-1, keepdim=True)
                if rollout_step + 1 >= self.lateral_candidate_commit_steps:
                    # Only the first command is the counterfactual decision.
                    # Thereafter score its consequence under the policy that
                    # will actually continue the trajectory online.
                    imagined_action = self._dist_mode(
                        self.actor(joint)).float().clamp(-1.0, 1.0)

            collision_cost = torch.stack(collision_costs, dim=1).amax(dim=1)
            clearance_cost = torch.stack(clearance_costs, dim=1).amax(dim=1)
            progress_cost = final_distance - initial_distance
            base_expanded = base[:, None].expand_as(candidates)
            action_change_cost = (
                candidates[..., :2] - base_expanded[..., :2]
            ).square().mean(dim=-1, keepdim=True).reshape(-1, 1)
            geometric_clearance, _ = self._kinematic_candidate_clearance(
                selected_skeleton, selected_joint_mask, candidates)
            geometric_cost = F.softplus(
                (self.planning_safe_clearance_m - geometric_clearance)
                / self.clearance_temperature_m
            ) * self.clearance_temperature_m / self.planning_safe_clearance_m
            score = (
                self.lateral_candidate_collision_cost * collision_cost
                + self.lateral_candidate_clearance_cost * clearance_cost
                + self.lateral_candidate_geometry_cost
                * geometric_cost.reshape(-1, 1)
                + self.lateral_candidate_progress_cost * progress_cost
                + self.lateral_candidate_action_change_cost
                * action_change_cost
            ).reshape(-1, candidate_count)
            ranked = torch.topk(score, k=2, dim=1, largest=False).values
            score_margin = ranked[:, 1] - ranked[:, 0]
            best_index = score.argmin(dim=1)
            hold_improvement = score[:, 0] - score.gather(
                1, best_index[:, None]).squeeze(1)
            selected_target = candidates[
                torch.arange(candidates.shape[0], device=candidates.device),
                best_index,
            ]
            # Do not manufacture supervision merely because the two best
            # non-hold candidates differ. The selected command must improve
            # on the raw Actor command by a separately configured margin.
            confident = (
                best_index.ne(0)
                & (score_margin >= self.lateral_candidate_min_score_margin)
                & (
                    hold_improvement
                    >= self.lateral_candidate_min_hold_improvement
                )
            )

        regression = F.smooth_l1_loss(
            selected_actor_action[..., :2], selected_target[..., :2],
            beta=self.behavior_clone_beta, reduction="none")
        dimension_weight = regression.new_tensor((0.5, 1.0))
        supervision = confident[:, None].to(regression.dtype)
        loss = (
            regression * dimension_weight * supervision
        ).sum() / (
            supervision.sum() * dimension_weight.sum()
        ).clamp_min(1.0)
        active_count = best_index.numel()
        confident_count = confident.sum().clamp_min(1)
        confident_float = confident.to(regression.dtype)
        lateral = best_index >= 2
        return loss, {
            "active_transition_ratio": active_ratio.detach(),
            "supervised_danger_ratio": confident_float.mean().detach(),
            "selected_hold_ratio": (best_index == 0).float().mean(),
            "selected_slow_ratio": (best_index == 1).float().mean(),
            "selected_left_ratio": (best_index == 2).float().mean(),
            "selected_right_ratio": (best_index == 3).float().mean(),
            "selected_lateral_ratio": lateral.float().mean(),
            "supervised_lateral_ratio": (
                (lateral & confident).sum().to(regression.dtype)
                / confident_count.to(regression.dtype)
            ),
            "score_margin": score_margin.mean(),
            "hold_improvement": hold_improvement.mean(),
            "actor_lateral_abs": selected_actor_action[:, 1].abs().mean(),
            "target_lateral_abs": selected_target[:, 1].abs().mean(),
            "score_hold": score[:, 0].mean(),
            "score_slow": score[:, 1].mean(),
            "score_left": score[:, 2].mean(),
            "score_right": score[:, 3].mean(),
            "geometric_clearance_hold_m": geometric_clearance[:, 0].mean(),
            "geometric_clearance_slow_m": geometric_clearance[:, 1].mean(),
            "geometric_clearance_left_m": geometric_clearance[:, 2].mean(),
            "geometric_clearance_right_m": geometric_clearance[:, 3].mean(),
            "active_count": regression.new_tensor(float(active_count)),
        }

    def behavior_mode_support_objective(
        self,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Keep the Actor mode inside conditional replay support.

        Unlike behavior cloning, this is an inequality barrier and uses every
        valid factual transition, including failures. It does not claim that
        the behavior action is optimal; it only prevents imagination from
        moving its mean into an action region for which the same latent state
        has no factual support.
        """
        feature = posterior_aux["actor_feat"][:, :-1].detach()
        target = self.policy_from_applied_action(
            batch["action"][:, 1:].float().clamp(-1.0, 1.0))
        valid = batch.get("action_valid")
        if valid is None:
            valid = torch.ones_like(target[..., :1], dtype=torch.bool)
        else:
            valid = valid[:, 1:].bool().reshape(
                *target.shape[:-1], -1).all(-1, keepdim=True)
        source_live = ~batch["is_last"][:, :-1].bool().reshape(
            *target.shape[:-1], -1).any(-1, keepdim=True)
        valid = valid & source_live
        mode = self._dist_mode(self.actor(feature)).float().clamp(-1.0, 1.0)
        limits = mode.new_tensor(self.behavior_mode_support_delta_limits)
        if limits.numel() != mode.shape[-1]:
            raise ValueError("behavior mode support limits must match Actor action")
        delta = (mode - target).abs()
        violation = delta > limits
        excess = F.relu(delta - limits)
        element_mask = valid.expand_as(excess)
        loss = excess.square().masked_select(element_mask).sum() / (
            element_mask.sum().clamp_min(1))
        sample_violation = violation.any(-1, keepdim=True) & valid
        valid_count = valid.sum().clamp_min(1)
        metrics = {
            "loss": loss.detach(),
            "accepted_sample_ratio": (
                ((~sample_violation) & valid).sum() / valid_count),
            "rejected_sample_ratio": sample_violation.sum() / valid_count,
            "mean_excess": excess.masked_select(element_mask).sum()
            / element_mask.sum().clamp_min(1),
        }
        names = ("vx", "vy", "yaw_rate")
        for index in range(mode.shape[-1]):
            name = names[index] if index < len(names) else str(index)
            metrics[f"violation_{name}"] = (
                (violation[..., index:index + 1] & valid).sum() / valid_count)
        return loss, metrics

    def behavior_clone_objective(
        self,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Constrain the offline actor to actions supported by the dataset.

        Dataset row ``t`` stores ``observation_t`` and the action that produced
        that observation.  The action selected from ``observation_t`` is thus
        stored in row ``t+1``.  Pairing features and actions without this shift
        would supervise the policy with the previous observation's command.

        Posterior features are detached so behavior cloning trains the Actor,
        not the world model.  This prevents a large BC coefficient from
        distorting the learned dynamics merely to make actions easier to fit.
        """
        feature = posterior_aux[
            "actor_feat" if self.uses_internal_action_adapter
            else "joint_feat"
        ][:, :-1].detach()
        applied_target = batch["action"][:, 1:].float().clamp(-1.0, 1.0)
        target = self.policy_from_applied_action(applied_target)
        valid = batch.get("action_valid")
        if valid is None:
            valid = torch.ones_like(target[..., :1], dtype=torch.bool)
        else:
            valid = valid[:, 1:].bool().reshape(*target.shape[:-1], -1).all(-1, keepdim=True)
        # A terminal observation cannot issue a next command even if malformed
        # legacy data happens to mark the following row valid.
        source_live = ~batch["is_last"][:, :-1].bool().reshape(
            *target.shape[:-1], -1).any(-1, keepdim=True)
        valid = valid & source_live

        episode_success = batch.get("episode_success")
        if episode_success is None:
            # Compatibility for synthetic/online batches predating the compact
            # dataset outcome field. Offline v3 training always provides it.
            episode_success = torch.ones_like(batch["is_last"], dtype=torch.bool)
        success_mask = episode_success[:, :-1].bool().reshape(
            *target.shape[:-1], -1).all(-1, keepdim=True)
        valid = valid & success_mask

        # A reached-goal label does not make every action in that episode a
        # safe demonstration.  In particular, planner-assisted online runs can
        # finish by chance with only centimetres of propeller clearance. Keep
        # their earlier useful avoidance actions, but do not clone a command
        # whose factual 2.5 s future is already a near miss.
        clearance = batch.get("future_min_human_clearance_m")
        clearance_valid = batch.get("future_min_human_clearance_valid")
        if clearance is None or clearance_valid is None:
            safe_demonstration = torch.ones_like(valid)
        else:
            clearance = clearance[:, :-1].float().reshape(
                *target.shape[:-1], -1)
            clearance_valid = clearance_valid[:, :-1].bool().reshape(
                *target.shape[:-1], -1)
            safe_demonstration = (
                ~clearance_valid
                | (clearance >= self.behavior_clone_min_clearance_m)
            ).all(-1, keepdim=True)
        valid = valid & safe_demonstration

        distribution = self.actor(feature)
        if not hasattr(distribution, "log_prob"):
            raise TypeError("Actor must return a distribution for behavior cloning")
        mode = self._dist_mode(distribution).float().clamp(-1.0, 1.0)
        regression = F.smooth_l1_loss(
            mode, target, beta=self.behavior_clone_beta, reduction="none")
        action_mask = valid.expand_as(target)
        action_count = action_mask.sum().clamp_min(1)
        loss = regression.masked_select(action_mask).sum() / action_count
        absolute_error = (mode - target).abs()
        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            selected = value.masked_select(mask)
            return selected.sum() / mask.sum().clamp_min(1).to(selected.dtype)

        unsafe_vertical = (
            mode.new_zeros(()) if self.uses_internal_action_adapter else
            masked_mean((mode[..., 2:3].abs() > 0.50).float(), valid)
            if mode.shape[-1] > 2 else mode.new_zeros(())
        )
        metrics: dict[str, torch.Tensor] = {
            "action_mae": absolute_error.masked_select(action_mask).sum()
            / action_count,
            "saturation_ratio": masked_mean(
                (mode.abs() > 0.90).float(), action_mask),
            "unsafe_vertical_ratio": unsafe_vertical,
            "valid_transition_ratio": valid.float().mean(),
            "episode_success_ratio": success_mask.float().mean(),
            "safe_demonstration_ratio": safe_demonstration.float().mean(),
            "near_miss_excluded_ratio": (
                success_mask & ~safe_demonstration).float().mean(),
            "bc_transition_ratio": valid.float().mean(),
        }
        names = (
            ("vx", "vy", "yaw_rate")
            if self.uses_internal_action_adapter
            else ("vx", "vy", "vz", "yaw_rate")
        )
        for index in range(target.shape[-1]):
            name = names[index] if index < len(names) else f"action_{index}"
            dim_valid = action_mask[..., index]
            metrics[f"mae_{name}"] = masked_mean(
                absolute_error[..., index], dim_valid)
            metrics[f"pred_mean_{name}"] = masked_mean(
                mode[..., index], dim_valid)
            metrics[f"target_mean_{name}"] = masked_mean(
                target[..., index], dim_valid)
        return loss, metrics

    def train(self, mode: bool = True):
        super().train(mode)
        self.slow_value.train(False)
        self.slow_safety_value.train(False)
        return self
