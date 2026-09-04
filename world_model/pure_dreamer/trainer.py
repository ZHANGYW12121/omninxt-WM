"""Joint world-model and imagined Actor-Critic training.

One call performs the three updates used by Dreamer:

1. learn the posterior/prior, observation predictions, reward and continuation;
2. imagine futures from replay posteriors and improve the Actor;
3. fit the Critic to lambda returns from the same imagined futures.

The optional Event head is an auxiliary world-model output and an imagined
risk cost.  It is trained on every update together with the world model; it
does not introduce an Event-only phase, candidate teacher, or runtime planner.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field
import math
import os
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from modules.action_smoother import SOFT_ABSOLUTE_TARGET
from modules.transition_event_head import (
    HUMAN_COLLISION_INDEX,
    TRANSITION_EVENT_KEYS,
)
from modules.reward_components import REWARD_COMPONENT_KEYS
from modules.reward_components import symexp as reward_component_symexp
from modules.reward_components import symlog as reward_component_symlog
from modules.skeleton_topology import COCO12_GEOMETRY_SANITATION_CONTRACT
from modules.task_geometry import (
    episode_to_world_flight_bounds_signed_clearance_torch,
    episode_to_world_static_clearance_torch,
    task_physical_state_dim,
)
from modules.se2_relative_dynamics import (
    ACTION_SCALE,
    COCO12_COLLISION_SPHERE_RADII_M,
    DEPLOYABLE_DRONE_COLLISION_RADIUS_M,
    analytic_ego_step,
    analytic_joint_signed_surface_gap,
    analytic_joint_surface_gap,
    body_points_to_episode,
    ego_velocity_full_body,
    interpolate_body_kinematics,
    interpolate_ego_state,
    swept_relative_point_first_contact_fraction,
    swept_relative_point_signed_gap,
    swept_scalar_exit_interval_fraction,
    swept_scalar_enter_lower_bound_fraction,
    swept_scalar_enter_upper_bound_fraction,
    swept_segment_aabb_first_contact_fraction_2d,
    swept_segment_exit_aabb_fraction_2d,
    swept_vector_exit_ball_fraction,
)
from modules.task_reward import (
    analytic_boundary_proximity_reward,
    analytic_fractional_smoothness_reward,
    analytic_fractional_progress_reward,
    analytic_human_risk_reward,
    analytic_progress_reward,
    analytic_route_deviation_reward,
    analytic_smoothness_reward,
    compose_uncertain_swept_human_analytic_events,
    cross_track_potential_reward,
    first_analytic_task_event,
    route_heading_potential,
    point_goal_distance,
    rescale_interval_probability,
    swept_human_contact_probability,
    terminal_aware_potential_reward,
    trajectory_correlated_human_contact_hazard,
)
from modules.task_memory import TASK_MEMORY_DIM


def _mode(distribution: Any) -> torch.Tensor:
    value = getattr(distribution, "mode", None)
    if value is not None:
        return value() if callable(value) else value
    value = getattr(distribution, "mean", None)
    if value is None:
        raise TypeError("distribution has neither mode nor mean")
    return value() if callable(value) else value


def _tree_detach(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _tree_detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_detach(item) for item in value)
    return value


@torch.no_grad()
def _all_floating_tensors_finite(tensors: Sequence[torch.Tensor]) -> bool:
    """Check many tensors with one device synchronization per dtype group.

    Calling ``bool(isfinite(tensor).all())`` once per parameter serialized
    hundreds of tiny GPU reductions after every joint update. A foreach norm
    detects NaN/Inf just as strictly for model/Adam tensors while preserving
    the asynchronous batch and synchronizing only the grouped result.
    Integer optimizer counters are finite by construction and are ignored.
    """
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for value in tensors:
        if not torch.is_tensor(value) or not (
            value.is_floating_point() or value.is_complex()
        ):
            continue
        grouped.setdefault((value.device, value.dtype), []).append(
            value.detach())
    for values in grouped.values():
        norms = torch._foreach_norm(values, 2.0)
        if not bool(torch.isfinite(torch.stack(norms)).all()):
            return False
    return True


@torch.no_grad()
def _clip_grad_norm_with_float64_fallback_(
    parameters: Sequence[nn.Parameter], max_norm: float,
) -> tuple[torch.Tensor, bool]:
    """Clip finite gradients even when a float32 p=2 norm overflows.

    PyTorch accumulates the ordinary global norm in the gradient dtype.  The
    gradients can therefore all be finite while their squared float32 norm is
    infinite.  Recompute only that exceptional scalar in float64; actual
    non-finite gradient elements remain an error.
    """
    parameters = tuple(parameters)
    try:
        return (
            torch.nn.utils.clip_grad_norm_(
                parameters, float(max_norm), error_if_nonfinite=True),
            False,
        )
    except RuntimeError:
        gradients = [
            parameter.grad.detach()
            for parameter in parameters if parameter.grad is not None
        ]
        if any(
            not bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ):
            raise
        parameter_norms = [
            torch.linalg.vector_norm(
                gradient.float(), ord=2, dtype=torch.float64)
            for gradient in gradients
        ]
        total_norm = (
            torch.linalg.vector_norm(
                torch.stack(parameter_norms), ord=2, dtype=torch.float64)
            if parameter_norms else torch.zeros((), dtype=torch.float64)
        )
        if not bool(torch.isfinite(total_norm)):
            raise
        coefficient = min(
            1.0, float(max_norm) / (float(total_norm) + 1.0e-6))
        for gradient in gradients:
            gradient.mul_(coefficient)
        return total_norm, True


@dataclass(frozen=True)
class PureDreamerConfig:
    """Algorithm-only configuration; architecture remains in Hydra YAML."""

    # Ada/Ampere BF16 accelerates neural linear/attention kernels while its
    # FP32-sized exponent range keeps Dreamer gradients stable. Explicit
    # geometric state/reward calculations call float() and remain FP32.
    amp_dtype: str = "bfloat16"
    imagination_horizon: int = 15
    max_imagination_starts: int = 128
    # Reserve coverage for collision precursors with and without enough
    # actuator response time. Per-start Horvitz-Thompson weights below keep
    # the optimized expectation equal to uniform valid replay-state Dreamer;
    # these are variance/coverage controls, not a risk-weighted objective.
    actionable_collision_start_target: int = 16
    imminent_collision_start_target: int = 16
    # A 15-step rollout only exposes a collision once the command chain has
    # almost no lateral response margin.  Reserve starts whose factual
    # collision lies just beyond imagination so the learned articulated/CV
    # risk can teach an early lateral-plus-forward response instead of making
    # retreat the first action that is still physically safe.
    preemptive_collision_start_target: int = 16
    near_goal_start_target: int = 16
    near_goal_start_distance_m: float = 5.0
    # Exact deployable flight-boundary clearance, not an outcome label.  This
    # stratum covers both the upper goal strip and lateral escape failures.
    near_boundary_start_target: int = 16
    near_boundary_start_clearance_m: float = 1.5
    actor_imagination_samples: int = 1
    imagination_window_burn_in: int = 8
    discount: float = 0.997
    return_lambda: float = 0.95
    actor_entropy: float = 3.0e-4
    # tanh(4)=0.999329 while its local derivative is still 1.34e-3.  Larger
    # logits cannot produce a materially stronger command, but rapidly erase
    # the reparameterized dynamics gradient that must train this Actor.
    actor_pre_tanh_abs_max_limit: float = 4.0
    # Regularize only the redundant raw parameterization beyond the
    # replay-calibrated range.  At the default algebraic bound, raw=6 already
    # produces an approximately 0.994 normalized actuator command.
    actor_raw_pre_tanh_soft_limit: float = 6.0
    actor_raw_pre_tanh_headroom_scale: float = 3.0e-3
    # The audited 10 Hz replay is inside roughly [-121, 101] per transition.
    # This wider margin is a fail-fast check for model exploitation, not a
    # clamp and therefore does not alter valid Dreamer gradients.
    imagined_reward_abs_max_limit: float = 150.0
    # The explicit per-person Event head remains joint World supervision and
    # a calibration diagnostic. Actor/Critic termination is instead obtained
    # from recursively predicted articulated geometry and the known swept
    # PhysX contact rule, so an observational action classifier cannot invent
    # a counterfactual left/right advantage.
    imagined_explicit_human_event: bool = True
    # Compact-v3 replay currently has no positive in-bounds static contact and
    # records static AABBs as an explicitly inexact conservative proxy. Keep
    # the learned static classifier diagnostic until positive support exists;
    # Actor safety uses the proxy potential, not an unsupported probability.
    imagined_learned_static_event: bool = False
    # Pure Dreamer may use known, differentiable applied-action kinematics as
    # the authoritative task state while continuing to train/audit the learned
    # Ego residual.  This prevents Actor optimization from exploiting a
    # learned correction that is empirically worse than the analytic baseline.
    authoritative_analytic_ego: bool = True
    require_multistep_geometry_supervision: bool = True
    progress_weight_per_m: float = 2.0
    max_progress_speed_mps: float = 3.0
    # Goal radius changes both reward and termination while it is deliberately
    # absent from the fixed-task Actor state. It must therefore be global.
    goal_radius_m: float = 1.0
    # Current replay is one fixed warehouse traversal task, not arbitrary
    # point-to-point navigation.  These are simulator generator bounds, not
    # empirical extrema, so unseen random draws inside the declared regions
    # remain valid while reverse/out-of-layout tasks fail closed.
    task_goal_world_x_min_m: float = -3.0
    task_goal_world_x_max_m: float = 4.0
    task_goal_world_y_min_m: float = 27.0
    task_goal_world_y_max_m: float = 29.0
    task_initial_world_x_min_m: float = -3.0
    task_initial_world_x_max_m: float = 4.0
    task_initial_world_y_min_m: float = -8.5
    task_initial_world_y_max_m: float = -5.0
    # Valid nonterminal replay reaches 3.0712 m/s horizontally.  This bound is
    # part of the analytic reachable-set contract used by boundary shaping.
    task_maximum_horizontal_speed_mps: float = 3.1
    human_clearance_weight_per_sec: float = 4.0
    human_hard_clearance_m: float = 0.10
    human_safe_clearance_m: float = 0.70
    human_predictive_safe_clearance_m: float = 0.90
    human_ttc_horizon_s: float = 2.5
    human_corridor_half_width_m: float = 0.75
    human_corridor_lookahead_m: float = 4.0
    drone_collision_radius_m: float = 0.17
    pelvis_collision_radius_m: float = 0.14
    human_collision_contact_offset_m: float = 0.02
    # Logistic residual model for learned articulated swept clearance.  These
    # are model-error parameters, not additional physical collision radii.
    human_collision_prediction_bias_m: float = 0.03
    human_collision_prediction_scale_m: float = 0.07
    # Recursive articulated prediction error grows with rollout depth. v88
    # measured near-collision false-safe p95 from 0.046 m at h1 to 0.287 m at
    # h15; a horizon-invariant residual distribution was overconfident.
    human_collision_prediction_bias_growth_m_per_step: float = 0.004
    imagination_dt_s: float = 0.10
    # Absolute warehouse floor shared by replay, reward and deployment AGL.
    task_static_ground_z_world: float = 0.0
    # Crash altitude affects termination but is not an episode-varying Actor
    # input, so it must remain a versioned global task constant.
    task_crash_altitude_world_m: float = 0.15
    # This is not a guessed wall margin.  It is the exact lateral room needed
    # for the simulator's most conservative predictive Human clearance:
    # 0.90 m surface gap + 0.17 m UAV radius + 0.14 m pelvis radius.
    route_cross_track_tolerance_m: float = 1.21
    # Linear discounted potential uses the same reward/metre coefficient as
    # factual goal progress.  It changes optimization geometry, not Dreamer's
    # RSSM, lambda return, or dynamics-gradient Actor update.
    route_potential_scale: float = 2.0
    # Unlike the potential above, this is an actual task cost. One metre of
    # excess deviation sustained for one second costs the same reward as one
    # metre of goal progress gains, while the complete 1.21 m physical Human
    # avoidance allowance remains free.
    route_deviation_weight_per_m_per_sec: float = 2.0
    # Bounded discounted potential for keeping the body aligned with the fixed
    # episode route. The vehicle is holonomic, so pointing at an off-axis point
    # goal incorrectly traded lateral correction for yaw near the destination.
    # Fixed-route alignment still exposes circular trajectories inside H15
    # without prescribing how the Actor translates around a person.
    route_heading_potential_scale: float = 8.0
    # Actual near-boundary state cost.  A discounted potential difference
    # refunds approach when the rollout moves inward and, at an absorbing
    # terminal, can return a positive reward for leaving a negative potential.
    # This non-refundable cost instead supplies a continuous pre-exit gradient
    # while the exact boundary Event remains the termination authority.
    boundary_proximity_weight_per_sec: float = 2.0
    static_proxy_clearance_margin_m: float = 1.0
    static_proxy_potential_scale: float = 2.0
    cruise_height_m: float = 1.0
    height_tolerance_m: float = 0.15
    height_scale_m: float = 0.50
    height_weight_per_sec: float = 0.5
    time_cost_per_sec: float = 0.10
    acceleration_weight_per_sec: float = 0.15
    acceleration_scale_mps2: float = 3.0
    jerk_weight_per_sec: float = 0.10
    jerk_scale_mps3: float = 10.0
    yaw_rate_weight_per_sec: float = 0.05
    yaw_rate_scale_rps: float = 1.5
    smooth_term_clip: float = 4.0
    acceleration_filter_alpha: float = 0.25
    crash_min_elapsed_s: float = 0.0
    watchdog_min_elapsed_s: float = 5.0
    watchdog_no_progress_timeout_s: float = 30.0
    watchdog_progress_epsilon_m: float = 0.25
    watchdog_stuck_max_horizontal_speed_mps: float = 0.15
    # World-model-only continuous-action perturbation supervision.  The
    # longitudinal probes distinguish accelerating through a crossing window
    # from mild positive-speed deceleration; neither is a policy label.  The
    # lateral probes expose both pass-behind directions while factual replay
    # remains the baseline candidate.  These candidates supervise geometry
    # only and never enter Actor inference or action selection.
    counterfactual_safety_aux_scale: float = 1.0
    counterfactual_safety_max_starts: int = 2
    # The all-pair mean can improve while a few state-local left/right
    # reversals grow catastrophically.  Mix in an equal-per-state
    # worst-quartile tail without increasing the total ranking-loss mass or
    # prescribing which direction is safe.  The historical ``_scale`` field
    # name is retained for checkpoint/config stability; in v66 it is the
    # convex tail weight and must lie in [0, 1].
    counterfactual_pairwise_cvar_fraction: float = 0.25
    counterfactual_pairwise_cvar_scale: float = 0.25
    # Pedestrians in the present simulator are exogenous to the UAV. Human
    # RSSM dynamics still receive action because observations live in the
    # moving body frame, but after transforming a rollout back into the fixed
    # episode frame, different candidate UAV commands must not invent
    # different root or articulated-joint trajectories for the same person.
    # The loss is World-only, symmetric over candidates, and normalized by
    # the existing 0.10 m hard-clearance scale.
    counterfactual_human_exogeneity_scale: float = 1.0
    counterfactual_lateral_policy: float = 0.50
    counterfactual_fast_forward_policy: float = 0.65
    counterfactual_mild_forward_policy: float = 0.20
    rare_event_aux_scale: float = 0.0
    # Natural replay NLL remains the calibrated reward objective.  This
    # destination-row-aligned auxiliary gives ordinary transitions half the
    # mass and a terminal-class macro-average the other half.
    rare_reward_aux_scale: float = 0.25
    rare_event_focal_gamma: float = 2.0
    # This auxiliary changes discrimination, while the unweighted Event NLL
    # remains present for probability calibration.
    rare_event_class_weights: tuple[float, ...] = (1.0, 8.0, 4.0, 2.0, 2.0)
    world_lr: float = 3.0e-4
    actor_lr: float = 4.0e-5
    critic_lr: float = 4.0e-5
    world_weight_decay: float = 0.0
    actor_weight_decay: float = 0.0
    critic_weight_decay: float = 0.0
    # Keep the historical common ceiling for checkpoint compatibility, but
    # isolate optimizer shocks below it.  A single H15 Human-regret outlier
    # produced a 3.14e8 World norm and was followed by a 6.50 Actor norm plus
    # one collision/two xMax exits from the same published policy snapshot.
    # The observed steady bands were World 26--35 and Actor about 1, so these
    # caps reject the discontinuous shock without clipping ordinary updates.
    grad_clip: float = 100.0
    world_grad_clip: float = 50.0
    actor_grad_clip: float = 5.0
    critic_grad_clip: float = 100.0
    slow_value_fraction: float = 0.02
    world_loss_scales: Mapping[str, float] = field(default_factory=lambda: {
        "dyn_ego": 1.0,
        "rep_ego": 0.1,
        "dyn_human": 1.0,
        "rep_human": 0.1,
        "ego_recon": 1.0,
        # The legacy delta head has no next action and is not used by rollout;
        # training it against terminal acceleration spikes is underconditioned.
        "ego_pred": 0.0,
        "yaw_unit": 0.1,
        "human_root": 1.0,
        "human_velocity": 0.5,
        "human_mpjpe": 1.0,
        "human_presence": 0.2,
        "human_birth": 0.05,
        "human_overshoot_1": 0.25,
        "human_overshoot_5": 0.5,
        "human_overshoot_10": 0.75,
        "human_overshoot_15": 1.0,
        "human_cv_regret_1": 0.25,
        "human_cv_regret_5": 0.5,
        "human_cv_regret_10": 0.75,
        "human_cv_regret_15": 1.0,
        "human_joint_overshoot_1": 0.25,
        "human_joint_overshoot_5": 0.5,
        "human_joint_overshoot_10": 0.75,
        "human_joint_overshoot_15": 1.0,
        # Privileged simulator identity is never policy-visible.  It supplies
        # a world-only displacement target that separates actual Human motion
        # from detector bias and is especially important on the collision
        # subset where measured pose can be partially occluded.
        "human_gt_displacement_1": 0.25,
        "human_gt_displacement_5": 0.5,
        "human_gt_displacement_10": 0.75,
        "human_gt_displacement_15": 1.0,
        # These losses were already implemented by the factorized world
        # model, but the original pure-Dreamer owner did not include them in
        # its selected loss dictionary.  They supervise the exact recursive
        # Ego path and articulated Human clearance consumed by imagination.
        "ego_overshoot_1": 0.25,
        "ego_overshoot_5": 0.5,
        "ego_overshoot_10": 0.75,
        "ego_overshoot_15": 1.0,
        "human_joint_clearance_1": 0.25,
        "human_joint_clearance_5": 0.5,
        "human_joint_clearance_10": 0.75,
        "human_joint_clearance_15": 1.0,
        "human_joint_collision_tail_1": 0.25,
        "human_joint_collision_tail_5": 0.5,
        "human_joint_collision_tail_10": 0.75,
        "human_joint_collision_tail_15": 1.0,
        "rew": 1.0,
        "rew_components": 0.2,
        "con": 1.0,
        "transition_event": 0.0,
        "transition_human_geometry": 1.0,
        "transition_non_human_event": 1.0,
        # Auxiliary factual world-model supervision only. Actor safety return
        # below uses recursively imagined articulated/CV geometry and the
        # explicit Human Event head; these one-step clearance heads are not
        # silently substituted into the policy reward.
        "next_clearance": 1.0,
        "next_clearance_false_safe": 1.0,
        "next_clearance_false_danger": 0.25,
    })

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"bfloat16", "float32"}:
            raise ValueError(
                "pure Dreamer amp_dtype must be bfloat16 or float32")
        if self.imagination_horizon < 2:
            raise ValueError("imagination_horizon must be at least two")
        if self.max_imagination_starts <= 0:
            raise ValueError("max_imagination_starts must be positive")
        if min(
            self.actionable_collision_start_target,
            self.imminent_collision_start_target,
            self.preemptive_collision_start_target,
            self.near_goal_start_target,
            self.near_boundary_start_target,
        ) < 0:
            raise ValueError(
                "imagination-start targets must be non-negative")
        if self.near_goal_start_distance_m <= 0.0:
            raise ValueError("near-goal start distance must be positive")
        if self.near_boundary_start_clearance_m <= 0.0:
            raise ValueError("near-boundary start clearance must be positive")
        if (
            self.actor_imagination_samples <= 0
            or (
                self.actor_imagination_samples > 1
                and self.actor_imagination_samples % 2 != 0
            )
        ):
            raise ValueError(
                "Actor imagination samples must be one or a positive even count")
        if self.imagination_window_burn_in < 0:
            raise ValueError("imagination_window_burn_in must be non-negative")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0,1]")
        if not 0.0 <= self.return_lambda <= 1.0:
            raise ValueError("return_lambda must lie in [0,1]")
        if self.actor_entropy < 0.0:
            raise ValueError("actor_entropy must be non-negative")
        if min(self.world_lr, self.actor_lr, self.critic_lr) <= 0.0:
            raise ValueError("all learning rates must be positive")
        if min(
            self.world_weight_decay,
            self.actor_weight_decay,
            self.critic_weight_decay,
        ) < 0.0:
            raise ValueError("optimizer weight decay must be non-negative")
        if min(
            self.progress_weight_per_m,
            self.max_progress_speed_mps,
            self.human_clearance_weight_per_sec,
            self.human_hard_clearance_m,
            self.route_cross_track_tolerance_m,
            self.route_potential_scale,
            self.route_deviation_weight_per_m_per_sec,
            self.route_heading_potential_scale,
            self.boundary_proximity_weight_per_sec,
            self.static_proxy_clearance_margin_m,
            self.static_proxy_potential_scale,
            self.cruise_height_m,
            self.height_tolerance_m,
            self.height_weight_per_sec,
            self.time_cost_per_sec,
        ) < 0.0:
            raise ValueError("analytic task reward parameters must be non-negative")
        if (
            self.max_progress_speed_mps == 0.0
            or not math.isfinite(self.goal_radius_m)
            or self.goal_radius_m <= 0.0
            or not math.isfinite(self.task_maximum_horizontal_speed_mps)
            or self.task_maximum_horizontal_speed_mps <= 0.0
            or self.imagination_dt_s <= 0.0
        ):
            raise ValueError(
                "progress speed, goal radius, horizontal speed contract, and "
                "imagination dt must be positive")
        if not math.isfinite(self.task_static_ground_z_world):
            raise ValueError("task_static_ground_z_world must be finite")
        if not math.isfinite(self.task_crash_altitude_world_m):
            raise ValueError("task_crash_altitude_world_m must be finite")
        task_regions = (
            (self.task_goal_world_x_min_m, self.task_goal_world_x_max_m),
            (self.task_goal_world_y_min_m, self.task_goal_world_y_max_m),
            (self.task_initial_world_x_min_m,
             self.task_initial_world_x_max_m),
            (self.task_initial_world_y_min_m,
             self.task_initial_world_y_max_m),
        )
        if any(
            not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) >= float(upper)
            for lower, upper in task_regions
        ):
            raise ValueError("fixed traversal goal/initial regions are invalid")
        maximum_commanded_horizontal_speed = math.hypot(
            float(ACTION_SCALE[0]), float(ACTION_SCALE[1]))
        if self.task_maximum_horizontal_speed_mps + 1.0e-6 < (
            maximum_commanded_horizontal_speed
        ):
            raise ValueError(
                "horizontal-speed contract is below the simultaneous-axis "
                "action command norm")
        if not math.isclose(
            self.drone_collision_radius_m,
            DEPLOYABLE_DRONE_COLLISION_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ) or not math.isclose(
            self.pelvis_collision_radius_m,
            float(COCO12_COLLISION_SPHERE_RADII_M[0]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "task Human radii differ from deployable collision geometry")
        if (
            not math.isfinite(self.human_collision_contact_offset_m)
            or self.human_collision_contact_offset_m < 0.0
        ):
            raise ValueError("Human contact offset must be finite/non-negative")
        if (
            not math.isfinite(self.human_collision_prediction_bias_m)
            or self.human_collision_prediction_bias_m < 0.0
            or not math.isfinite(self.human_collision_prediction_scale_m)
            or self.human_collision_prediction_scale_m <= 0.0
            or not math.isfinite(
                self.human_collision_prediction_bias_growth_m_per_step)
            or self.human_collision_prediction_bias_growth_m_per_step < 0.0
        ):
            raise ValueError(
                "Human collision prediction bias/scale must be valid")
        if self.human_safe_clearance_m <= self.human_hard_clearance_m:
            raise ValueError("safe Human clearance must exceed hard clearance")
        if self.human_predictive_safe_clearance_m <= (
            self.human_hard_clearance_m
        ):
            raise ValueError(
                "predictive Human clearance must exceed hard clearance")
        if min(
            self.human_ttc_horizon_s,
            self.human_corridor_half_width_m,
            self.human_corridor_lookahead_m,
            self.drone_collision_radius_m,
            self.pelvis_collision_radius_m,
            self.height_scale_m,
            self.acceleration_scale_mps2,
            self.jerk_scale_mps3,
            self.yaw_rate_scale_rps,
            self.smooth_term_clip,
            self.watchdog_no_progress_timeout_s,
            self.watchdog_progress_epsilon_m,
            self.watchdog_stuck_max_horizontal_speed_mps,
        ) <= 0.0:
            raise ValueError("task geometry/reward scales must be positive")
        if min(
            self.acceleration_weight_per_sec,
            self.jerk_weight_per_sec,
            self.yaw_rate_weight_per_sec,
            self.crash_min_elapsed_s,
            self.watchdog_min_elapsed_s,
            self.rare_event_aux_scale,
            self.rare_event_focal_gamma,
        ) < 0.0:
            raise ValueError("task/auxiliary weights must be non-negative")
        if not 0.0 <= self.acceleration_filter_alpha <= 1.0:
            raise ValueError("acceleration filter alpha must lie in [0,1]")
        if self.actor_pre_tanh_abs_max_limit <= 0.0:
            raise ValueError(
                "actor_pre_tanh_abs_max_limit must be positive")
        if self.actor_raw_pre_tanh_soft_limit <= 0.0:
            raise ValueError(
                "actor_raw_pre_tanh_soft_limit must be positive")
        if self.actor_raw_pre_tanh_headroom_scale <= 0.0:
            raise ValueError(
                "actor_raw_pre_tanh_headroom_scale must be positive")
        if self.imagined_reward_abs_max_limit <= 0.0:
            raise ValueError(
                "imagined_reward_abs_max_limit must be positive")
        gradient_caps = {
            "grad_clip": self.grad_clip,
            "world_grad_clip": self.world_grad_clip,
            "actor_grad_clip": self.actor_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in gradient_caps.values()
        ):
            raise ValueError("gradient clip limits must be finite and positive")
        excessive_caps = {
            name: value for name, value in gradient_caps.items()
            if name != "grad_clip" and float(value) > float(self.grad_clip)
        }
        if excessive_caps:
            raise ValueError(
                "optimizer-specific gradient clips cannot exceed grad_clip: "
                f"{excessive_caps}")
        if not 0.0 < self.slow_value_fraction <= 1.0:
            raise ValueError("slow_value_fraction must lie in (0,1]")
        if len(self.rare_event_class_weights) != len(TRANSITION_EVENT_KEYS):
            raise ValueError("rare Event weights must match Event classes")
        if min(self.rare_event_class_weights) <= 0.0:
            raise ValueError("rare Event class weights must be positive")
        if self.rare_reward_aux_scale < 0.0:
            raise ValueError("rare reward auxiliary scale must be non-negative")
        if self.counterfactual_safety_aux_scale < 0.0:
            raise ValueError(
                "counterfactual safety auxiliary scale must be non-negative")
        if self.counterfactual_safety_max_starts <= 0:
            raise ValueError(
                "counterfactual safety maximum starts must be positive")
        if not 0.0 < self.counterfactual_pairwise_cvar_fraction <= 1.0:
            raise ValueError(
                "counterfactual pairwise CVaR fraction must lie in (0,1]")
        if not 0.0 <= self.counterfactual_pairwise_cvar_scale <= 1.0:
            raise ValueError(
                "counterfactual pairwise CVaR blend must lie in [0,1]")
        if (
            not math.isfinite(self.counterfactual_human_exogeneity_scale)
            or self.counterfactual_human_exogeneity_scale < 0.0
        ):
            raise ValueError(
                "counterfactual Human exogeneity scale must be finite and "
                "non-negative")
        if not 0.0 < self.counterfactual_lateral_policy <= 1.0:
            raise ValueError(
                "counterfactual lateral policy must lie in (0,1]")
        if not 0.0 < self.counterfactual_mild_forward_policy < (
            self.counterfactual_fast_forward_policy
        ) <= 1.0:
            raise ValueError(
                "counterfactual forward probes must satisfy "
                "0 < mild < fast <= 1")


class PureDreamerTrainer:
    """Minimal Dreamer training owner for an existing FactorizedDreamer."""

    ARCHITECTURE_VERSION = "factorized_dreamer_v8.7"
    OBJECTIVE_VERSION = (
        "factorized_pure_dreamer_v67_action_exogenous_human_trajectory")
    # This changes only which uniformly replayed posterior rows receive the
    # fixed imagination compute budget. Horvitz-Thompson correction below
    # preserves the same standard Dreamer objective, so an older v50 joint
    # checkpoint can resume without resetting parameters or optimizer state.
    IMAGINATION_START_SAMPLING_CONTRACT_VERSION = (
        "disjoint_collision_preemptive_near_goal_ymax_route_misaligned_"
        "side_boundary_front_human_h15_v3")
    COUNTERFACTUAL_SAFETY_SAMPLING_CONTRACT_VERSION = (
        "distinct_sequence_early_warning_then_severity_v2")
    LEGACY_IMAGINATION_START_SAMPLING_CONTRACTS = frozenset((
        "disjoint_collision_near_goal_ymax_misaligned_"
        "side_boundary_front_human_v1",
        "disjoint_collision_near_goal_ymax_misaligned_"
        "side_boundary_front_human_v2",
    ))
    NEAR_GOAL_ROUTE_HEADING_ERROR_THRESHOLD_RAD = math.radians(20.0)
    ACTOR_HORIZONTAL_STEP_RESPONSE_TARGET = 0.80
    # Counterfactual replay shows the largest forward+lateral return gap in
    # the 1.0--1.3 s TTC band. The exact deployed command chain reaches 60%
    # horizontal response at 1.0 s, so this lower measured crossing separates
    # still-actionable precursors from genuinely imminent rows.
    ACTOR_ACTIONABLE_HORIZONTAL_RESPONSE_TARGET = 0.60
    HUMAN_EVENT_DENSE_REWARD_CONTRACT_VERSION = (
        "learned_articulated_recursive_uncertain_swept_physx_contact_v6")
    HUMAN_KINEMATIC_RESIDUAL_CONTRACT_VERSION = (
        "root_dv0p70_vmax2p50_joint_dv1p50_vmax6p00_per_nominal100ms_v2")
    HUMAN_GEOMETRY_SANITATION_CONTRACT_VERSION = (
        COCO12_GEOMETRY_SANITATION_CONTRACT)
    ACTION_SMOOTHER_CONTRACT_VERSION = (
        "soft_absolute_target_mean_reverting_slew_bounded_v1")
    ACTION_SMOOTHER_PARAMETERIZATION = SOFT_ABSOLUTE_TARGET
    CONTROL_TRANSITION_CONTRACT_VERSION = (
        "acknowledged_physical_control_step_v1")
    RSSM_INITIAL_STATE_CONTRACT_VERSION = (
        "learned_prior_categorical_reset_deterministic_eval_v2")
    LEGACY_MIGRATION_OBJECTIVE = (
        "factorized_pure_dreamer_v4_stable_pre_tanh_score")

    _UNUSED_PREFIXES = (
        "action_risk.",
        "safety_value.",
        "slow_safety_value.",
        "transition_event.network.",
    )
    _ACTOR_PREFIXES = ("actor.",)
    _CRITIC_PREFIXES = ("value.",)
    _TARGET_PREFIXES = ("slow_value.",)
    # These compatibility heads are neither optimized nor consumed by the
    # current Actor return.  Keep the list exact so a newly emitted world loss
    # cannot disappear merely because nobody added a YAML weight for it.
    _INTENTIONALLY_IGNORED_WORLD_LOSSES = frozenset({
        "risk_collision",
        "risk_clearance",
        "risk_false_safe",
        "risk_false_danger",
    })

    def __init__(
        self,
        model: nn.Module,
        config: PureDreamerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PureDreamerConfig()
        parameter_device = next(iter(model.parameters())).device
        self.amp_dtype = (
            torch.bfloat16
            if self.config.amp_dtype == "bfloat16" else torch.float32)
        self.autocast_enabled = bool(
            parameter_device.type == "cuda"
            and self.amp_dtype is torch.bfloat16)
        if getattr(model, "prediction_heads", None) is None:
            raise ValueError("pure Dreamer requires the factorized state decoder")
        if getattr(model, "transition_event", None) is None:
            raise ValueError("pure Dreamer requires the Event auxiliary head")
        try:
            model.transition_event.require_identity_human_probability_calibration()
        except RuntimeError as error:
            raise ValueError(
                "joint pure-Dreamer training requires its raw natural-NLL "
                "Human hazard") from error
        if (
            self.config.imagined_explicit_human_event
            and not bool(model.transition_event.explicit_human_geometry)
        ):
            raise ValueError(
                "explicit imagined Human Event requires the per-person "
                "geometry head")
        actor_config = getattr(model.actor, "config", None)
        critic_input_layer = next((
            module for module in model.value.modules()
            if isinstance(module, nn.Linear)
        ), None)
        critic_external_input_dim = int(getattr(
            model.value, "input_dim",
            -1 if critic_input_layer is None else critic_input_layer.in_features,
        ))
        expected_decision_width = (
            3 * int(model.rssm.joint_feat_size)
            + int(getattr(model.actor, "human_geometry_dim", -1))
            + int(getattr(model.actor, "task_geometry_dim", -1))
        )
        required_contracts = {
            "learned Ego RSSM initial prior": bool(
                getattr(model.rssm.ego_rssm, "_initial", "") == "learned"
                and isinstance(
                    getattr(model.rssm.ego_rssm, "_initial_deter", None),
                    nn.Parameter,
                )),
            "learned Human RSSM initial prior": bool(
                getattr(model.rssm.human_rssm, "_initial", "") == "learned"
                and isinstance(
                    getattr(model.rssm.human_rssm, "_initial_deter", None),
                    nn.Parameter,
                )),
            "one full-state stochastic Actor": bool(
                getattr(actor_config, "unified_policy", False)),
            "headroom-regularized reachable-gradient algebraic Actor mean": bool(
                0.0 < float(getattr(
                    actor_config, "pre_tanh_mean_bound", 0.0))
                < self.config.actor_pre_tanh_abs_max_limit
                and getattr(
                    actor_config,
                    "pre_tanh_mean_parameterization",
                    "",
                ) == "algebraic_sqrt"
                and self.config.actor_raw_pre_tanh_soft_limit > 0.0
                and self.config.actor_raw_pre_tanh_headroom_scale > 0.0),
            "learned bounded Actor standard deviation": bool(
                getattr(actor_config, "std_parameterization", "")
                == "learned_bounded"),
            "Actor/Critic identical decision state": bool(
                getattr(model, "critic_full_state_enabled", False)),
            "Actor/Critic identical decision-state width": bool(
                int(getattr(model.actor, "input_dim", -1))
                == expected_decision_width
                and critic_external_input_dim == expected_decision_width),
            "permutation-invariant Human/obstacle Actor and Critic": bool(
                getattr(
                    actor_config,
                    "permutation_invariant_entities",
                    False,
                )
                and getattr(model.value, "input_dim", -1)
                == getattr(model.actor, "input_dim", -2)
                and hasattr(model.actor, "decision_encoder")
                and hasattr(model.value, "encoder")),
            "deterministic evaluation posterior state": bool(getattr(
                model, "deterministic_evaluation_state_enabled", False)),
            "authoritative deployable Ego token": bool(
                getattr(model, "actor_authoritative_ego_token_enabled", False)),
            "injective direct Ego/task state": bool(
                getattr(model, "actor_direct_ego_task_state_enabled", False)),
            "explicit Actor Human geometry": bool(
                getattr(model, "actor_explicit_human_geometry_enabled", False)),
            "task geometry": bool(
                getattr(model, "task_geometry_enabled", False)),
            "direct task geometry": bool(
                getattr(model, "direct_task_geometry_enabled", False)),
            "complete exact task geometry in Actor/Critic": bool(
                getattr(model, "actor_task_physical_state_enabled", False)
                and getattr(actor_config, "task_geometry_dim", 0)
                == task_physical_state_dim(int(getattr(
                    model, "actor_task_physical_obstacle_slots", -1)))
                and int(getattr(
                    model, "actor_task_physical_obstacle_slots", 0)) == 256),
            "task/reward/actuator memory": bool(
                getattr(model, "task_memory_enabled", False)),
            "direct task memory": bool(
                getattr(model, "direct_task_memory_enabled", False)),
            "analytic crash/stuck events": bool(
                model.transition_event.analytic_task_memory_events),
            "explicit articulated Human kinematics": bool(
                model.transition_event.explicit_joint_kinematics),
            "Event reads complete articulated Human state": bool(
                model.transition_event.event_full_articulated_state),
            "explicit per-person lifecycle probability": bool(
                model.transition_event.explicit_human_presence_physical),
            "Actor per-person lifecycle probability": bool(
                getattr(actor_config, "human_physical_presence_state", False)),
            "all articulated Human fields in Actor/Critic": bool(
                getattr(model.transition_event, "actor_full_state_slots", 0)
                == getattr(actor_config, "human_physical_slots", -1)
                and getattr(
                    model.transition_event,
                    "actor_full_fields_per_slot", 0,
                ) == getattr(
                    actor_config, "human_physical_fields_per_slot", -1,
                )
                and getattr(
                    model.transition_event, "actor_full_state_dim", 0,
                ) == getattr(actor_config, "human_geometry_dim", -1)),
            "per-person learned Human state in Actor/Critic": bool(
                getattr(
                    model.transition_event,
                    "actor_full_learned_per_slot", False,
                )
                and getattr(
                    model.transition_event,
                    "actor_full_learned_fields_per_slot", 0,
                ) == (
                    getattr(model.transition_event, "geometry_dim", -1)
                    - getattr(
                        model.transition_event,
                        "geometry_slot_physical_dim", 0,
                    )
                )),
            "single learned Human rollout": not bool(
                getattr(model, "conservative_human_clearance_enabled", True)),
            "audited root/joint kinematic residual support": bool(
                math.isclose(
                    float(model.prediction_heads.config.
                          max_velocity_residual_mps),
                    0.70, rel_tol=0.0, abs_tol=1.0e-9,
                )
                and math.isclose(
                    float(model.prediction_heads.config.
                          max_joint_velocity_residual_mps),
                    1.50, rel_tol=0.0, abs_tol=1.0e-9,
                )
                and math.isclose(
                    float(model.prediction_heads.config.max_root_speed_mps),
                    2.50, rel_tol=0.0, abs_tol=1.0e-9,
                )
                and math.isclose(
                    float(model.prediction_heads.config.max_joint_speed_mps),
                    6.00, rel_tol=0.0, abs_tol=1.0e-9,
                )),
            "unsupported learned static Event excluded from return": not bool(
                self.config.imagined_learned_static_event),
            "h15-covered absolute-target action response": bool(
                getattr(model, "action_smoother", None) is not None
                and getattr(
                    model.action_smoother, "parameterization", "")
                == self.ACTION_SMOOTHER_PARAMETERIZATION
                and torch.allclose(
                    model.action_smoother.filter_alpha.detach().cpu(),
                    torch.tensor(1.0 - math.exp(-1.0)),
                    rtol=0.0, atol=1.0e-7,
                )
                and torch.allclose(
                    model.action_smoother.maximum_step_delta.detach().cpu(),
                    torch.tensor((0.30, 0.30, 0.045)),
                    rtol=0.0, atol=1.0e-7,
                )),
            "horizontal ActionAdapter": getattr(
                model, "action_adapter", None) is not None,
            "explicit fixed-altitude task boundary": bool(
                int(getattr(actor_config, "policy_action_dim", -1)) == 3
                and math.isclose(
                    float(model.action_adapter.config.
                          target_altitude_agl_m),
                    float(self.config.cruise_height_m),
                    rel_tol=0.0, abs_tol=1.0e-9,
                )),
        }
        missing_contracts = tuple(
            name for name, enabled in required_contracts.items() if not enabled)
        if missing_contracts:
            raise ValueError(
                "complete-state pure Dreamer contract is incomplete: "
                + ", ".join(missing_contracts))
        if bool(getattr(actor_config, "human_conditioned_residual", False)) \
                or bool(getattr(
                    actor_config, "human_reflection_equivariant", False)):
            raise ValueError(
                "unified Actor forbids residual/gate/reflection policy branches")
        encoder_config = getattr(model.encoder, "config", None)
        if not all((
            bool(getattr(encoder_config, "ego_metric_scaling", False)),
            bool(getattr(encoder_config, "human_root_metric_scaling", False)),
            bool(getattr(encoder_config, "human_quality_metric_scaling", False)),
            bool(getattr(
                model.rssm.latent_policy_attention,
                "goal_metric_scaling", False)),
        )):
            raise ValueError(
                "metric Ego/Human/Goal encoders are required by v24")
        if not math.isclose(
            float(model.lateral_candidate_step_duration_s),
            float(self.config.imagination_dt_s),
            rel_tol=0.0, abs_tol=1.0e-9,
        ):
            raise ValueError(
                "model dynamics dt and pure-Dreamer imagination dt differ")
        required_horizons = {1, 5, 10, 15}
        if self.config.require_multistep_geometry_supervision and not (
            required_horizons.issubset(set(model.overshoot_horizons))
        ):
            raise ValueError(
                "v24 requires Human/Ego geometry supervision at h1/5/10/15")
        if (
            self.config.require_multistep_geometry_supervision
            and not bool(getattr(
                model, "collision_tail_regret_supervision", False))
        ):
            raise ValueError(
                "v49 requires Human CV-regret and collision-tail supervision")
        if (
            self.config.require_multistep_geometry_supervision
            and not bool(getattr(
                model, "require_gt_displacement_supervision", False))
        ):
            raise ValueError(
                "v61 requires privileged Human displacement labels for "
                "offset-invariant CV regret")
        invalid_world_scales = {
            str(name): value
            for name, value in self.config.world_loss_scales.items()
            if not math.isfinite(float(value)) or float(value) < 0.0
        }
        if invalid_world_scales:
            raise ValueError(
                "world loss scales must be finite and non-negative: "
                f"{invalid_world_scales}")
        accidentally_enabled_legacy = (
            self._INTENTIONALLY_IGNORED_WORLD_LOSSES
            & set(self.config.world_loss_scales)
        )
        if accidentally_enabled_legacy:
            raise ValueError(
                "pure Dreamer cannot configure removed risk-head losses: "
                f"{sorted(accidentally_enabled_legacy)}")
        required_world_losses = {
            "dyn_ego", "rep_ego", "dyn_human", "rep_human",
            "ego_recon", "yaw_unit", "human_root", "human_velocity",
            "human_mpjpe", "human_presence", "human_birth", "rew",
            "rew_components", "con", "transition_human_geometry",
            "transition_non_human_event", "next_clearance",
            "next_clearance_false_safe", "next_clearance_false_danger",
        }
        for horizon in required_horizons:
            required_world_losses.update({
                f"human_overshoot_{horizon}",
                f"human_cv_regret_{horizon}",
                f"human_joint_overshoot_{horizon}",
                f"human_gt_displacement_{horizon}",
                f"ego_overshoot_{horizon}",
                f"human_joint_clearance_{horizon}",
                f"human_joint_collision_tail_{horizon}",
            })
        missing_world_scales = required_world_losses.difference(
            self.config.world_loss_scales)
        if missing_world_scales:
            raise ValueError(
                "complete-state world objective lacks explicit scales: "
                f"{sorted(missing_world_scales)}")
        self._required_world_losses = frozenset(required_world_losses)
        (
            self.actor_human_action_response_time_s,
            self.actor_human_action_horizon_response_fraction,
            self.actor_actionable_response_time_s,
        ) = self._measure_actor_human_action_response()

        for name, parameter in model.named_parameters():
            if name.startswith(self._UNUSED_PREFIXES + self._TARGET_PREFIXES):
                parameter.requires_grad_(False)

        named = dict(model.named_parameters())
        self.actor_parameters = self._parameters_with_prefix(
            named, self._ACTOR_PREFIXES)
        self.critic_parameters = self._parameters_with_prefix(
            named, self._CRITIC_PREFIXES)
        owned = {id(item) for item in self.actor_parameters + self.critic_parameters}
        self.world_parameters = [
            parameter for name, parameter in named.items()
            if parameter.requires_grad
            and id(parameter) not in owned
            and not name.startswith(self._UNUSED_PREFIXES + self._TARGET_PREFIXES)
        ]
        if not self.world_parameters or not self.actor_parameters or not self.critic_parameters:
            raise ValueError("world, Actor and Critic parameter groups must be non-empty")
        all_owned = self.world_parameters + self.actor_parameters + self.critic_parameters
        if len({id(item) for item in all_owned}) != len(all_owned):
            raise RuntimeError("a parameter has more than one optimizer owner")

        self.world_optimizer = torch.optim.AdamW(
            self.world_parameters, lr=self.config.world_lr, eps=1.0e-8,
            weight_decay=self.config.world_weight_decay)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_parameters, lr=self.config.actor_lr, eps=1.0e-8,
            weight_decay=self.config.actor_weight_decay)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_parameters, lr=self.config.critic_lr, eps=1.0e-8,
            weight_decay=self.config.critic_weight_decay)
        self.update_count = 0
        # Installed from transition-unique replay immediately before the first
        # optimizer call.  It is checkpointed so a resume never recomputes or
        # silently changes the prior after online replay has grown.
        self.event_prior: dict[str, Any] | None = None
        self.checkpoint_safe = True
        self._optimizer_step_started = False

    @property
    def event_prior_initialized(self) -> bool:
        return self.event_prior is not None

    @staticmethod
    @torch.no_grad()
    def _set_categorical_output_prior(
        layer: nn.Module,
        probability: torch.Tensor,
    ) -> None:
        if not isinstance(layer, nn.Linear):
            raise TypeError("Event output layer must be nn.Linear")
        if layer.bias is None or layer.bias.numel() != probability.numel():
            raise ValueError("Event output layer/prior shape mismatch")
        layer.weight.zero_()
        layer.bias.copy_(probability.log().to(layer.bias))

    @staticmethod
    @torch.no_grad()
    def _set_binary_output_prior(layer: nn.Module, probability: float) -> None:
        if not isinstance(layer, nn.Linear):
            raise TypeError("binary output layer must be nn.Linear")
        if layer.bias is None or layer.bias.numel() != 1:
            raise ValueError("binary output layer must have one output")
        value = float(probability)
        if not 0.0 < value < 1.0:
            raise ValueError("binary prior must lie strictly in (0,1)")
        layer.weight.zero_()
        layer.bias.fill_(math.log(value / (1.0 - value)))

    @torch.no_grad()
    def initialize_empirical_event_priors(
        self,
        event_counts: Mapping[str, int | float],
        *,
        dense_reward_component_symlog_mean: Mapping[
            str, int | float] | None = None,
        source: str = "transition_unique_replay",
    ) -> dict[str, Any]:
        """Initialize terminal heads from deduplicated factual transitions.

        A symmetric one-count Laplace prior is used only to keep unseen event
        logits finite.  The resulting probabilities otherwise come directly
        from replay, and final weights are zero so random hidden features
        cannot manufacture a large terminal reward before learning begins.
        """
        if self.update_count != 0:
            raise RuntimeError(
                "Event priors may only be initialized before update zero")
        ordered_counts = torch.tensor([
            float(event_counts.get(key, 0.0))
            for key in TRANSITION_EVENT_KEYS
        ], dtype=torch.float64)
        if (
            not torch.isfinite(ordered_counts).all()
            or bool((ordered_counts < 0.0).any())
            or float(ordered_counts.sum()) <= 0.0
        ):
            raise ValueError("Event prior counts must be finite and non-empty")
        smoothed = ordered_counts + 1.0
        probability = smoothed / smoothed.sum()
        self._set_categorical_output_prior(
            self.model.transition_event.network[-1], probability)

        total = float(ordered_counts.sum())
        human_count = float(ordered_counts[HUMAN_COLLISION_INDEX])
        nominal_step_s = float(self.config.imagination_dt_s)
        transition_exposure_s = float(event_counts.get(
            "transition_exposure_s", total * nominal_step_s))
        if not math.isfinite(transition_exposure_s) \
                or transition_exposure_s <= 0.0:
            raise ValueError("Event transition exposure must be positive")
        human_rate_per_s = (
            (human_count + 1.0)
            / (transition_exposure_s + 2.0 * nominal_step_s)
        )
        human_probability = 1.0 - math.exp(
            -human_rate_per_s * nominal_step_s)
        visible_human_collision_count = float(event_counts.get(
            "actor_observable_human_collision",
            event_counts.get("visible_human_collision", human_count),
        ))
        unobserved_human_collision_count = float(event_counts.get(
            "unobserved_human_collision",
            human_count - visible_human_collision_count,
        ))
        if (
            unobserved_human_collision_count < 0.0
            or unobserved_human_collision_count > human_count
        ):
            raise ValueError(
                "unobserved Human collision count is inconsistent")
        per_slot_human_probability = human_probability
        mean_visible_humans = 1.0
        visible_human_rows = total
        conditional_human_probability = human_probability
        if bool(self.model.transition_event.explicit_human_geometry):
            histogram = {
                int(str(key).rsplit("_", 1)[-1]): float(value)
                for key, value in event_counts.items()
                if str(key).startswith("visible_human_count_")
                and int(str(key).rsplit("_", 1)[-1]) > 0
            }
            if histogram:
                visible_human_rows = sum(histogram.values())
                visible_exposure = sum(
                    count * rows for count, rows in histogram.items())
            else:
                visible_exposure = float(event_counts.get(
                    "visible_human_exposure", 0.0))
                all_rows = float(event_counts.get(
                    "visible_human_transition_rows", 0.0))
                empty_rows = float(event_counts.get(
                    "visible_human_count_0", 0.0))
                visible_human_rows = max(0.0, all_rows - empty_rows)
                histogram = (
                    {1: visible_human_rows}
                    if visible_human_rows > 0.0 else {})
            visible_human_exposure_s = float(event_counts.get(
                "visible_human_exposure_s",
                visible_exposure * nominal_step_s,
            ))
            if not math.isfinite(visible_human_exposure_s) \
                    or visible_human_exposure_s < 0.0:
                raise ValueError("visible Human exposure must be non-negative")
            if visible_human_rows <= 0.0 or visible_human_exposure_s <= 0.0:
                if visible_human_collision_count > 0.0:
                    raise ValueError(
                        "visible-Human collision count has no visible rows")
                conditional_human_probability = 1.0e-6
                mean_visible_humans = 0.0
                per_slot_human_probability = 1.0e-6
            else:
                if visible_human_collision_count > visible_human_rows:
                    raise ValueError(
                        "visible-Human collisions exceed visible transition rows")
                mean_visible_humans = visible_exposure / visible_human_rows
                # Cause-specific exposure gives each visible person-time the
                # same initial rate. Count-only noisy-OR initialization treats
                # a 16 ms forced terminal row as if it exposed the model for a
                # complete 100 ms interval and is not a calibrated hazard.
                visible_rate_per_s = (
                    (visible_human_collision_count + 1.0)
                    / (visible_human_exposure_s + 2.0 * nominal_step_s)
                )
                per_slot_human_probability = 1.0 - math.exp(
                    -visible_rate_per_s * nominal_step_s)
                conditional_human_probability = 1.0 - math.exp(
                    -visible_rate_per_s * nominal_step_s
                    * max(mean_visible_humans, 0.0))
            self._set_binary_output_prior(
                self.model.transition_event.per_human_network[-1],
                per_slot_human_probability,
            )
            unobserved_rate_per_s = (
                (unobserved_human_collision_count + 1.0)
                / (transition_exposure_s + 2.0 * nominal_step_s)
            )
            unobserved_human_probability = 1.0 - math.exp(
                -unobserved_rate_per_s * nominal_step_s)
            self._set_binary_output_prior(
                self.model.transition_event.unobserved_human_network[-1],
                unobserved_human_probability,
            )
            residual_keys = (
                "residual_continue",
                "residual_static_collision",
            )
            residual_counts = torch.tensor([
                float(event_counts.get(
                    key,
                    (
                        event_counts.get("continue", 0.0)
                        + event_counts.get("reached_goal", 0.0)
                        + event_counts.get("other_task_terminal", 0.0)
                        if index == 0 else
                        event_counts.get("static_collision", 0.0)
                        if index == 1 else 0.0
                    ),
                ))
                for index, key in enumerate(residual_keys)
            ], dtype=torch.float64)
            residual_probability = (
                residual_counts + 1.0
            ) / (residual_counts.sum() + len(residual_keys))
            self._set_categorical_output_prior(
                self.model.transition_event.non_goal_network[-1],
                residual_probability,
            )
        else:
            residual_keys = ()
            residual_counts = torch.zeros(0, dtype=torch.float64)
            residual_probability = torch.zeros(0, dtype=torch.float64)
            unobserved_human_probability = 0.0

        continue_count = float(ordered_counts[0])
        continuation_probability = (continue_count + 1.0) / (total + 2.0)
        continuation_layer = getattr(
            self.model.cont, "last", getattr(self.model.cont, "linear", None))
        self._set_binary_output_prior(
            continuation_layer, continuation_probability)

        # Component heads are factual diagnostics only in this objective.
        # Give their dense rows replay means so update-zero audits are stable;
        # Actor reward below uses structured height/time terms and never reads
        # these learned outputs.
        dense_keys = ("smoothness", "height", "time")
        supplied_dense = (
            {} if dense_reward_component_symlog_mean is None
            else dict(dense_reward_component_symlog_mean))
        dense_prior = {
            key: float(supplied_dense.get(key, 0.0)) for key in dense_keys}
        if not all(math.isfinite(value) for value in dense_prior.values()):
            raise ValueError("dense reward-component priors must be finite")
        component_layer = self.model.reward_components.net[-1]
        if (
            not isinstance(component_layer, nn.Linear)
            or component_layer.bias is None
            or component_layer.bias.numel() != len(REWARD_COMPONENT_KEYS)
        ):
            raise TypeError("reward-component output must be a six-row Linear")
        for key, value in dense_prior.items():
            index = REWARD_COMPONENT_KEYS.index(key)
            component_layer.weight[index].zero_()
            component_layer.bias[index].fill_(value)
        self.event_prior = {
            "source": str(source),
            "laplace_pseudocount_per_class": 1.0,
            "counts": {
                key: int(ordered_counts[index].item())
                for index, key in enumerate(TRANSITION_EVENT_KEYS)
            },
            "categorical_probability": {
                key: float(probability[index])
                for index, key in enumerate(TRANSITION_EVENT_KEYS)
            },
            "explicit_human_probability": float(human_probability),
            "human_rate_per_s": float(human_rate_per_s),
            "transition_exposure_s": float(transition_exposure_s),
            "visible_conditional_human_probability": float(
                conditional_human_probability),
            "visible_human_collision_count": int(
                visible_human_collision_count),
            "unobserved_human_collision_count": int(
                unobserved_human_collision_count),
            "unobserved_human_probability": float(
                unobserved_human_probability),
            "explicit_per_slot_human_probability": float(
                per_slot_human_probability),
            "mean_visible_humans": float(mean_visible_humans),
            "visible_human_exposure_s": float(
                event_counts.get(
                    "visible_human_exposure_s",
                    float(event_counts.get(
                        "visible_human_exposure", 0.0)) * nominal_step_s,
                )),
            "visible_human_rows": int(visible_human_rows),
            "residual_non_human_counts": {
                key: int(residual_counts[index].item())
                for index, key in enumerate(residual_keys)
            },
            "residual_non_human_probability": {
                key: float(residual_probability[index])
                for index, key in enumerate(residual_keys)
            },
            "continuation_probability": float(continuation_probability),
            "dense_reward_component_symlog_mean": dense_prior,
        }
        return copy.deepcopy(self.event_prior)

    def _initialize_event_priors_from_batch(
        self, batch: Mapping[str, torch.Tensor],
    ) -> None:
        """Unit/offline fallback; production installs a full-replay prior."""
        target = batch.get("transition_event_target")
        target_valid = batch.get("transition_event_valid")
        if target is None or target_valid is None:
            raise RuntimeError(
                "pure Dreamer needs Event labels before its first update")
        target = target[:, :-1].long().squeeze(-1)
        valid = target_valid[:, :-1].bool().squeeze(-1).clone()
        valid &= ~batch["is_last"][:, :-1].bool().reshape_as(valid)
        if "action_valid" in batch:
            valid &= batch["action_valid"][:, 1:].bool().reshape_as(valid)
        if "sequence_valid" in batch:
            valid &= (
                batch["sequence_valid"][:, :-1].bool().reshape_as(valid)
                & batch["sequence_valid"][:, 1:].bool().reshape_as(valid))
        counts = torch.bincount(
            target[valid].detach().cpu(), minlength=len(TRANSITION_EVENT_KEYS))
        residual_counts = torch.zeros(2, dtype=torch.long)
        termination_code = batch.get("termination_code")
        if termination_code is not None:
            code = termination_code[:, 1:].long().squeeze(-1)
            residual_valid = valid & code.ne(2)
            residual_target = torch.zeros_like(code)
            residual_target = torch.where(
                code.eq(3), torch.ones_like(code), residual_target)
            residual_counts = torch.bincount(
                residual_target[residual_valid].detach().cpu(), minlength=2)
        visible = batch.get("human_mask")
        visible_exposure = 0
        visible_rows = 0
        visible_human_collision = 0
        actor_observable_human_collision = 0
        transition_exposure_s = float(valid.sum().detach().cpu()) * float(
            self.config.imagination_dt_s)
        visible_human_exposure_s = float(visible_exposure) * float(
            self.config.imagination_dt_s)
        if "dt_s" in batch:
            destination_dt = batch["dt_s"][:, 1:].float().squeeze(-1)
            if destination_dt.shape != valid.shape:
                raise ValueError("fallback Event dt_s shape is invalid")
            if bool((destination_dt[valid] <= 0.0).any()):
                raise ValueError("fallback Event exposure must be positive")
            transition_exposure_s = float(
                destination_dt[valid].sum().detach().cpu())
        visible_histogram: dict[int, int] = {}
        if visible is not None:
            source_visible = visible[:, :-1].bool().sum(-1)
            visible_exposure = int(source_visible[valid].sum().detach().cpu())
            visible_human_exposure_s = (
                float((
                    destination_dt * source_visible.to(destination_dt)
                )[valid].sum().detach().cpu())
                if "dt_s" in batch else
                float(visible_exposure) * float(
                    self.config.imagination_dt_s)
            )
            visible_rows = int(valid.sum().detach().cpu())
            visible_human_collision = int((
                target.eq(HUMAN_COLLISION_INDEX)
                & valid
                & source_visible.gt(0)
            ).sum().detach().cpu())
            actor_observable_human_collision = visible_human_collision
            privileged_slot_fields = {
                "human_gt_id", "human_gt_match_valid", "human_mask",
                "priv_human_id", "priv_human_mask",
                "priv_collision_joints_episode",
                "priv_collision_joint_valid",
                "privileged_geometry_available",
                "counterfactual_collision_surface_radii_m",
                "counterfactual_collision_contact_offset_m",
            }
            if privileged_slot_fields.issubset(batch):
                slot_target, slot_valid, _ = (
                    self.model.privileged_per_human_swept_collision_targets(
                        batch))
                attributable = (slot_target & slot_valid).any(-1)
                actor_observable_human_collision = int((
                    target.eq(HUMAN_COLLISION_INDEX)
                    & valid
                    & attributable
                ).sum().detach().cpu())
            for count in source_visible[valid].detach().cpu().tolist():
                count = int(count)
                visible_histogram[count] = visible_histogram.get(count, 0) + 1
        dense_prior: dict[str, float] = {}
        components = batch.get("reward_components")
        if components is not None:
            destination_valid = valid
            for key in ("smoothness", "height", "time"):
                index = REWARD_COMPONENT_KEYS.index(key)
                target_symlog = reward_component_symlog(
                    components[:, 1:, index].float())
                dense_prior[key] = float(
                    target_symlog[destination_valid].mean().detach().cpu())
        self.initialize_empirical_event_priors(
            {
                key: int(counts[index])
                for index, key in enumerate(TRANSITION_EVENT_KEYS)
            } | {
                key: int(residual_counts[index])
                for index, key in enumerate((
                    "residual_continue", "residual_static_collision",
                ))
            } | {
                "visible_human_exposure": visible_exposure,
                "transition_exposure_s": transition_exposure_s,
                "visible_human_exposure_s": visible_human_exposure_s,
                "visible_human_transition_rows": visible_rows,
                "visible_human_collision": visible_human_collision,
                "actor_observable_human_collision": (
                    actor_observable_human_collision),
                **{
                    f"visible_human_count_{count}": rows
                    for count, rows in visible_histogram.items()
                },
            },
            dense_reward_component_symlog_mean=dense_prior,
            source="first_batch_fallback",
        )

    @torch.no_grad()
    def _normalize_returns(
        self,
        returns: torch.Tensor,
        start_importance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Dreamer EMA offset/scale with a stable first observation.

        A migrated v2 checkpoint has no ReturnEMA history because its old
        Actor objective never used the module.  Initializing that history from
        the first imagined-return quantiles avoids hundreds of updates with a
        falsely clipped unit scale.  Subsequent calls use the repository's
        standard ReturnEMA update unchanged.
        """
        ema = self.model.return_ema
        flattened = returns.detach().flatten()
        if start_importance is None:
            quantiles = torch.quantile(flattened, ema.range)
        else:
            importance = start_importance.detach().to(returns).reshape(-1)
            if importance.numel() != returns.shape[0]:
                raise ValueError(
                    "return-normalizer start weights must match starts")
            if not bool(torch.isfinite(importance).all()) or bool(
                (importance <= 0.0).any()
            ):
                raise ValueError(
                    "return-normalizer start weights must be finite/positive")
            expanded = importance[:, None].expand(
                returns.shape[0], returns[0].numel()).reshape(-1)
            if bool(torch.allclose(
                expanded, torch.ones_like(expanded), rtol=0.0, atol=1.0e-7,
            )):
                quantiles = torch.quantile(flattened, ema.range)
            else:
                order = torch.argsort(flattened)
                ordered_value = flattened.index_select(0, order)
                ordered_weight = expanded.index_select(0, order)
                cumulative = ordered_weight.cumsum(0)
                # Centred weighted empirical CDF avoids assigning an entire
                # stratum weight jump to only one endpoint.
                cdf = (
                    cumulative - 0.5 * ordered_weight
                ) / ordered_weight.sum().clamp_min(1.0e-12)
                targets = ema.range.to(cdf)
                upper = torch.searchsorted(cdf, targets).clamp(
                    max=ordered_value.numel() - 1)
                lower = (upper - 1).clamp(min=0)
                lower_cdf = cdf.index_select(0, lower)
                upper_cdf = cdf.index_select(0, upper)
                fraction = (
                    (targets - lower_cdf)
                    / (upper_cdf - lower_cdf).clamp_min(1.0e-12)
                ).clamp(0.0, 1.0)
                lower_value = ordered_value.index_select(0, lower)
                upper_value = ordered_value.index_select(0, upper)
                quantiles = lower_value + fraction * (
                    upper_value - lower_value)
        bootstrapped = torch.all(ema.ema_vals == 0.0)
        if bool(bootstrapped):
            ema.ema_vals.copy_(quantiles)
        else:
            ema.ema_vals.copy_(
                ema.alpha * quantiles.detach()
                + (1.0 - ema.alpha) * ema.ema_vals)
        scale = torch.clip(ema.ema_vals[1] - ema.ema_vals[0], min=1.0)
        offset = ema.ema_vals[0]
        return offset, scale, torch.as_tensor(
            bootstrapped, device=returns.device, dtype=returns.dtype)

    @staticmethod
    def _parameters_with_prefix(
        named: Mapping[str, nn.Parameter], prefixes: Sequence[str],
    ) -> list[nn.Parameter]:
        return [
            parameter for name, parameter in named.items()
            if parameter.requires_grad and name.startswith(tuple(prefixes))
        ]

    @contextmanager
    def _temporarily_frozen(self, parameters: Sequence[nn.Parameter]):
        previous = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, enabled in zip(parameters, previous, strict=True):
                parameter.requires_grad_(enabled)

    @staticmethod
    def lambda_return(
        reward: torch.Tensor,
        continuation: torch.Tensor,
        value: torch.Tensor,
        *,
        discount: float,
        lamb: float,
    ) -> torch.Tensor:
        """Dreamer lambda return for H rewards and H+1 values."""
        if reward.shape != continuation.shape:
            raise ValueError("reward and continuation shapes differ")
        if value.shape[:-2] != reward.shape[:-2] or value.shape[-2] != reward.shape[-2] + 1:
            raise ValueError("value must contain one bootstrap step")
        if value.shape[-1] != reward.shape[-1]:
            raise ValueError("value and reward event dimensions differ")
        next_return = value[:, -1]
        outputs: list[torch.Tensor] = []
        for index in reversed(range(reward.shape[1])):
            bootstrap = (
                (1.0 - float(lamb)) * value[:, index + 1]
                + float(lamb) * next_return
            )
            next_return = reward[:, index] + (
                float(discount) * continuation[:, index] * bootstrap)
            outputs.append(next_return)
        return torch.stack(list(reversed(outputs)), dim=1)

    def _require_current_static_terminal_contract(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        context: str,
    ) -> None:
        """Reject Actor starts from a different static-obstacle MDP."""
        if not bool(getattr(
            self.model, "actor_task_physical_state_enabled", False,
        )):
            return
        valid = values.get("task_static_terminal_valid")
        if valid is None:
            raise KeyError(
                f"{context} lacks task_static_terminal_valid for the "
                "current analytic-static-terminal Actor")
        if not bool(valid.bool().all()):
            raise ValueError(
                f"{context} mixes proximity-only replay with the current "
                "analytic-static-terminal deployment MDP")

    def _require_task_memory_action_contract(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        context: str,
    ) -> None:
        """Require one unambiguous actuator state at an imagination start.

        Compact-v3 row ``t`` stores the applied action that produced state
        ``t``. Task-memory fields 6:9 store the corresponding previous
        smoothed policy target. They are two representations of the same
        Markov variable, so accepting a disagreement would let Actor/Critic
        condition on an impossible state while the rollout smoother follows a
        different value.
        """
        if not bool(getattr(self.model, "task_memory_enabled", False)):
            return
        memory = values.get("task_memory")
        action = values.get("previous_action")
        if memory is None or action is None:
            raise KeyError(
                f"{context} lacks task memory or previous applied action")
        if memory.shape[:-1] != action.shape[:-1] or memory.shape[-1] != (
            TASK_MEMORY_DIM
        ):
            raise ValueError(
                f"{context} task-memory/action row shapes disagree")
        expected = self.model.policy_from_applied_action(action.float())
        actual = memory[..., 6:9].float()
        if expected.shape != actual.shape:
            raise ValueError(
                f"{context} previous policy-action layout is invalid")
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError(
                f"{context} previous actuator state contains NaN/Inf")
        if not torch.allclose(expected, actual, rtol=0.0, atol=2.0e-6):
            maximum_error = float((expected - actual).abs().amax())
            raise ValueError(
                f"{context} task memory and applied action encode different "
                f"previous smoother states (max error {maximum_error:.6g})")

    def _rare_event_loss(
        self,
        joint_feature: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        states: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = joint_feature.sum() * 0.0
        required = ("transition_event_target", "transition_event_valid")
        if any(key not in batch for key in required):
            return zero, {"valid_ratio": zero.detach(), "rare_ratio": zero.detach()}
        feature = joint_feature[:, :-1]
        action = batch["action"][:, 1:].float().clamp(-1.0, 1.0)
        target = batch["transition_event_target"][:, :-1].long().squeeze(-1)
        # Clone before in-place mask refinement: ``Tensor.bool()`` is an alias
        # when the source is already boolean, and mutating replay labels after
        # another loss saved them for backward invalidates autograd.
        valid = batch[
            "transition_event_valid"][:, :-1].bool().squeeze(-1).clone()
        valid &= ~batch["is_last"][:, :-1].bool().reshape_as(valid)
        if "action_valid" in batch:
            valid &= batch["action_valid"][:, 1:].bool().reshape_as(valid)
        if not bool(valid.any()):
            return zero, {"valid_ratio": valid.float().mean(), "rare_ratio": zero.detach()}
        explicit_context = (
            self.config.imagined_explicit_human_event
            and states is not None
            and all(key in batch for key in (
                "human_root", "human_observation_quality", "human_mask",
                "joint_mask", "skeleton", "termination_code"))
        )
        if not explicit_context:
            return zero, {
                "valid_ratio": valid.float().mean(),
                "rare_ratio": zero.detach(),
                "human_ratio": zero.detach(),
                "explicit_human_focal": zero.detach(),
                "residual_non_human_focal": zero.detach(),
            }
        branch = self.model.rssm.get_branch_feats(states)
        human_joints = batch["skeleton"][:, :-1, ..., :3].float()
        human_velocity = batch["skeleton"][:, :-1, ..., 3:6].float()
        human_joint_mask = batch["joint_mask"][:, :-1].bool()
        human_joint_clearance = self.model.deployable_human_clearance(
            human_joints, human_velocity, human_joint_mask, per_human=True)
        prediction = self.model.transition_event.forward_non_goal(
            feature,
            action,
            ego_feature=branch["ego"][:, :-1],
            human_feature=branch["human"][:, :-1],
            human_root=batch["human_root"][:, :-1].float(),
            human_quality=batch[
                "human_observation_quality"][:, :-1].float(),
            human_mask=batch["human_mask"][:, :-1].bool(),
            human_joint_clearance=human_joint_clearance,
            human_joints_body=human_joints,
            human_joint_velocity_body=human_velocity,
            human_joint_mask=human_joint_mask,
        )
        exposure_fraction = action.new_ones((*action.shape[:-1], 1))
        if "dt_s" in batch:
            transition_dt = batch["dt_s"][:, 1:].float()
            if transition_dt.shape != exposure_fraction.shape:
                raise ValueError(
                    "destination dt_s must match rare Event transitions")
            if not torch.isfinite(transition_dt).all() \
                    or bool((transition_dt < 0.0).any()):
                raise ValueError("rare Event dt_s must be finite/non-negative")
            if bool((transition_dt.le(0.0) & valid[..., None]).any()):
                raise ValueError("valid rare Event transitions require positive dt_s")
            exposure_fraction = (
                transition_dt / float(self.config.imagination_dt_s))
        code = batch["termination_code"][:, 1:].long().squeeze(-1)
        residual_target = torch.zeros_like(code)
        residual_target = torch.where(
            code.eq(3), torch.ones_like(code), residual_target)
        residual_valid = valid & code.ne(2)
        residual_probability = prediction[
            "global_non_human_probability"].gather(
                -1, residual_target[..., None]).squeeze(-1).clamp_min(1.0e-6)
        source_weights = residual_probability.new_tensor(
            self.config.rare_event_class_weights)
        residual_weights = torch.stack((
            source_weights[0], source_weights[2],
        ))
        residual_focal = (
            -residual_weights[residual_target]
            * (1.0 - residual_probability).pow(
                self.config.rare_event_focal_gamma)
            * residual_probability.log()
        )
        residual_non_human_focal = (
            residual_focal[residual_valid].mean()
            if bool(residual_valid.any()) else zero)
        human_probability = rescale_interval_probability(
            prediction["human_collision_probability"], exposure_fraction,
        ).clamp(1.0e-6, 1.0 - 1.0e-6)
        human_target = target.eq(HUMAN_COLLISION_INDEX)[..., None]
        target_probability = torch.where(
            human_target, human_probability, 1.0 - human_probability)
        human_alpha = torch.where(
            human_target,
            human_probability.new_tensor(
                self.config.rare_event_class_weights[HUMAN_COLLISION_INDEX]),
            human_probability.new_ones(()),
        )
        human_focal = (
            -human_alpha
            * (1.0 - target_probability).pow(
                self.config.rare_event_focal_gamma)
            * target_probability.log()
        )
        aggregate_human_valid = valid
        slot_human_focal = zero
        unobserved_human_focal = zero
        privileged_slot_fields = {
            "human_gt_id", "human_gt_match_valid", "human_mask",
            "priv_human_id", "priv_human_mask",
            "priv_collision_joints_episode", "priv_collision_joint_valid",
            "privileged_geometry_available",
            "counterfactual_collision_surface_radii_m",
            "counterfactual_collision_contact_offset_m",
            "priv_collision_human_id",
        }
        slot_positive_count = zero.detach()
        if privileged_slot_fields.issubset(batch):
            slot_target, slot_valid, _ = (
                self.model.privileged_per_human_swept_collision_targets(batch))
            slot_valid &= valid[..., None]
            attributable_positive = (slot_target & slot_valid).any(-1)
            slot_probability = rescale_interval_probability(
                prediction["slot_hazard"], exposure_fraction,
            ).clamp(1.0e-6, 1.0 - 1.0e-6)
            slot_target_probability = torch.where(
                slot_target, slot_probability, 1.0 - slot_probability)
            slot_alpha = torch.where(
                slot_target,
                slot_probability.new_tensor(
                    self.config.rare_event_class_weights[
                        HUMAN_COLLISION_INDEX]),
                slot_probability.new_ones(()),
            )
            slot_focal = (
                -slot_alpha
                * (1.0 - slot_target_probability).pow(
                    self.config.rare_event_focal_gamma)
                * slot_target_probability.log()
            )
            if bool(slot_valid.any()):
                slot_human_focal = slot_focal[slot_valid].mean()
            slot_positive_count = (slot_target & slot_valid).sum().to(zero)
            unobserved_target = (
                human_target.squeeze(-1) & ~attributable_positive)
            unobserved_probability = rescale_interval_probability(
                prediction["unobserved_human_collision_probability"],
                exposure_fraction,
            ).squeeze(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
            unobserved_target_probability = torch.where(
                unobserved_target,
                unobserved_probability,
                1.0 - unobserved_probability,
            )
            unobserved_alpha = torch.where(
                unobserved_target,
                unobserved_probability.new_tensor(
                    self.config.rare_event_class_weights[
                        HUMAN_COLLISION_INDEX]),
                unobserved_probability.new_ones(()),
            )
            unobserved_focal = (
                -unobserved_alpha
                * (1.0 - unobserved_target_probability).pow(
                    self.config.rare_event_focal_gamma)
                * unobserved_target_probability.log()
            )
            unobserved_human_focal = unobserved_focal[valid].mean()
        explicit_human_focal = (
            human_focal[aggregate_human_valid[..., None]].mean()
            if bool(aggregate_human_valid.any()) else zero)
        if privileged_slot_fields.issubset(batch):
            explicit_human_focal = (
                explicit_human_focal
                + slot_human_focal
                + unobserved_human_focal
            ) / 3.0
        loss = residual_non_human_focal + explicit_human_focal
        rare = target.ne(0) & valid
        return loss, {
            "valid_ratio": valid.float().mean(),
            "rare_ratio": rare.float().sum() / valid.float().sum().clamp_min(1.0),
            "human_ratio": (
                (target.eq(HUMAN_COLLISION_INDEX) & valid).float().sum()
                / valid.float().sum().clamp_min(1.0)
            ),
            "explicit_human_focal": explicit_human_focal.detach(),
            "slot_human_focal": slot_human_focal.detach(),
            "unobserved_human_focal": unobserved_human_focal.detach(),
            "slot_human_positive_count": slot_positive_count.detach(),
            "residual_non_human_focal": (
                residual_non_human_focal.detach()),
        }

    def _rare_reward_loss(
        self,
        joint_feature: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Balance factual ordinary and terminal reward supervision.

        Compact-v3 stores the reward caused by source row ``t``'s action on
        destination row ``t+1``.  ``transition_event_target[t]`` uses that
        same transition.  Pairing ``joint[t+1]``/``reward[t+1]`` with the
        source event label is therefore the only source-correct alignment.

        The ordinary world loss still trains every reward row under the
        natural replay distribution.  This auxiliary assigns half its mass to
        ordinary transitions and half to a macro-average of the terminal
        classes present in the minibatch.  Including the ordinary half is
        essential: a terminal-only auxiliary can create a globally negative
        Event-component bias before the latent is discriminative, because the
        natural component loss is averaged over all six reward components.
        """
        zero = joint_feature.sum() * 0.0
        required = (
            "reward", "reward_components", "transition_event_target",
            "transition_event_valid", "is_last",
        )
        if any(key not in batch for key in required):
            return zero, {
                "class_count": zero.detach(),
                "transition_count": zero.detach(),
                "ordinary_transition_count": zero.detach(),
                "scalar_nll": zero.detach(),
                "scalar_terminal_nll": zero.detach(),
                "scalar_ordinary_nll": zero.detach(),
                "component_event": zero.detach(),
                "component_event_terminal": zero.detach(),
                "component_event_ordinary": zero.detach(),
                "target_mean": zero.detach(),
                "scalar_mae": zero.detach(),
                "component_event_mae": zero.detach(),
            }

        destination_feature = joint_feature[:, 1:]
        target_class = batch[
            "transition_event_target"][:, :-1].long().squeeze(-1)
        valid = batch[
            "transition_event_valid"][:, :-1].bool().squeeze(-1).clone()
        valid &= ~batch["is_last"][:, :-1].bool().reshape_as(valid)
        if "action_valid" in batch:
            valid &= batch["action_valid"][:, 1:].bool().reshape_as(valid)
        if "sequence_valid" in batch:
            sequence_valid = batch["sequence_valid"].bool()
            source_valid = sequence_valid[:, :-1].reshape_as(valid)
            destination_valid = sequence_valid[:, 1:].reshape_as(valid)
            valid &= source_valid & destination_valid
        rare = valid & target_class.ne(0)
        ordinary = valid & target_class.eq(0)
        if not bool(rare.any()) or not bool(ordinary.any()):
            return zero, {
                "class_count": zero.detach(),
                "transition_count": zero.detach(),
                "ordinary_transition_count": zero.detach(),
                "scalar_nll": zero.detach(),
                "scalar_terminal_nll": zero.detach(),
                "scalar_ordinary_nll": zero.detach(),
                "component_event": zero.detach(),
                "component_event_terminal": zero.detach(),
                "component_event_ordinary": zero.detach(),
                "target_mean": zero.detach(),
                "scalar_mae": zero.detach(),
                "component_event_mae": zero.detach(),
            }

        reward_target = batch["reward"][:, 1:].float()
        scalar_distribution = self.model.reward(destination_feature)
        scalar_nll = -scalar_distribution.log_prob(reward_target)
        if scalar_nll.ndim == rare.ndim + 1:
            scalar_nll = scalar_nll.squeeze(-1)
        scalar_prediction = _mode(scalar_distribution)
        scalar_error = (scalar_prediction - reward_target).abs().squeeze(-1)

        component_target = batch[
            "reward_components"][:, 1:, 0].float()
        component_prediction_symlog = self.model.reward_components(
            destination_feature)[..., 0]
        component_error = F.smooth_l1_loss(
            component_prediction_symlog,
            reward_component_symlog(component_target),
            reduction="none",
        )
        component_prediction = reward_component_symexp(
            component_prediction_symlog)
        component_absolute_error = (
            component_prediction - component_target).abs()

        scalar_by_class: list[torch.Tensor] = []
        component_by_class: list[torch.Tensor] = []
        for event_index in range(1, len(TRANSITION_EVENT_KEYS)):
            class_mask = rare & target_class.eq(event_index)
            if bool(class_mask.any()):
                scalar_by_class.append(scalar_nll[class_mask].mean())
                component_by_class.append(
                    component_error[class_mask].mean())
        scalar_terminal_macro = torch.stack(scalar_by_class).mean()
        component_terminal_macro = torch.stack(component_by_class).mean()
        scalar_ordinary = scalar_nll[ordinary].mean()
        component_ordinary = component_error[ordinary].mean()
        scalar_balanced = 0.5 * (
            scalar_ordinary + scalar_terminal_macro)
        component_balanced = 0.5 * (
            component_ordinary + component_terminal_macro)
        component_scale = float(
            self.config.world_loss_scales.get("rew_components", 0.0))
        loss = scalar_balanced + component_scale * component_balanced
        return loss, {
            "class_count": scalar_balanced.new_tensor(
                float(len(scalar_by_class))).detach(),
            "transition_count": rare.float().sum().detach(),
            "ordinary_transition_count": ordinary.float().sum().detach(),
            "scalar_nll": scalar_balanced.detach(),
            "scalar_terminal_nll": scalar_terminal_macro.detach(),
            "scalar_ordinary_nll": scalar_ordinary.detach(),
            "component_event": component_balanced.detach(),
            "component_event_terminal": component_terminal_macro.detach(),
            "component_event_ordinary": component_ordinary.detach(),
            "target_mean": reward_target.squeeze(-1)[rare].mean().detach(),
            "scalar_mae": scalar_error[rare].mean().detach(),
            "component_event_mae": (
                component_absolute_error[rare].mean().detach()),
        }

    def _world_loss(
        self, batch: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor, dict[str, torch.Tensor], Any, Mapping[str, Any],
    ]:
        profile_runtime = os.environ.get(
            "PURE_DREAMER_PROFILE_UPDATE_TIMES", "0") == "1"
        profile_previous = time.perf_counter()
        profile_metrics: dict[str, torch.Tensor] = {}

        def profile_mark(name: str) -> None:
            nonlocal profile_previous
            if not profile_runtime:
                return
            device = next(iter(self.model.parameters())).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            profile_metrics[f"runtime/profile_world_{name}_ms"] = (
                batch["action"].new_tensor(
                    (now - profile_previous) * 1000.0))
            profile_previous = now

        if (
            self.config.require_multistep_geometry_supervision
            and getattr(self.model, "overshoot_horizons", ())
        ):
            required = {
                "ego_state", "action", "action_valid", "human_mask",
                "human_root", "human_ids", "skeleton", "joint_mask",
                "human_gt_id", "human_gt_match_valid",
                "human_gt_identity_id", "human_gt_identity_valid",
                "human_gt_pelvis_body", "human_gt_pelvis_episode",
                "priv_human_id", "priv_human_mask",
                "priv_collision_human_id",
                "priv_collision_joints_episode",
                "priv_collision_joint_valid",
                "privileged_geometry_available",
                "counterfactual_collision_surface_radii_m",
                "counterfactual_collision_contact_offset_m",
            }
            missing = sorted(required.difference(batch))
            if missing:
                raise KeyError(
                    "pure Dreamer multistep geometry batch lacks "
                    f"{missing}")
        losses, states, auxiliary = self.model.world_model_loss(dict(batch))
        profile_mark("model_loss_forward")
        ego_overshoot_losses, ego_overshoot_metrics = (
            self.model.ego_multistep_overshooting_loss(
                states, dict(batch)))
        profile_mark("ego_overshoot_forward")
        losses.update(ego_overshoot_losses)
        prediction_losses, _ = self.model.prediction_loss(states, dict(batch))
        profile_mark("prediction_forward")
        losses.update(prediction_losses)
        missing_losses = self._required_world_losses.difference(losses)
        if missing_losses:
            raise KeyError(
                "world model failed to emit required pure-Dreamer losses: "
                f"{sorted(missing_losses)}")
        unknown_losses = (
            set(losses)
            - set(self.config.world_loss_scales)
            - self._INTENTIONALLY_IGNORED_WORLD_LOSSES
        )
        if unknown_losses:
            raise KeyError(
                "world model emitted losses without an explicit pure-Dreamer "
                f"weight or ignore contract: {sorted(unknown_losses)}")
        selected = {
            name: loss for name, loss in losses.items()
            if name in self.config.world_loss_scales
        }
        if not selected:
            raise RuntimeError("no configured world-model loss was produced")
        total = sum(
            float(self.config.world_loss_scales[name]) * loss
            for name, loss in selected.items()
        )
        if self.config.rare_event_aux_scale > 0.0:
            rare_loss, rare_metrics = self._rare_event_loss(
                auxiliary["joint_feat"], batch, states=states)
        else:
            rare_loss = auxiliary["joint_feat"].sum() * 0.0
            rare_metrics = {
                "valid_ratio": rare_loss.detach(),
                "rare_ratio": rare_loss.detach(),
            }
        rare_reward_loss, rare_reward_metrics = self._rare_reward_loss(
            auxiliary["joint_feat"], batch)
        profile_mark("rare_reward_forward")
        counterfactual_loss, counterfactual_metrics = (
            self._counterfactual_safety_loss(
                states, auxiliary, batch))
        profile_mark("counterfactual_forward")
        total = (
            total
            + self.config.rare_event_aux_scale * rare_loss
            + self.config.rare_reward_aux_scale * rare_reward_loss
            + self.config.counterfactual_safety_aux_scale
            * counterfactual_loss
        )
        metrics = {f"world/{name}": loss.detach() for name, loss in selected.items()}
        metrics.update({
            "world/rare_event_aux": rare_loss.detach(),
            "world/rare_reward_aux": rare_reward_loss.detach(),
            "world/counterfactual_safety_aux": (
                counterfactual_loss.detach()),
            **{f"event/{name}": value.detach() for name, value in rare_metrics.items()},
            **{
                f"reward_terminal/{name}": value.detach()
                for name, value in rare_reward_metrics.items()
            },
            **{
                f"counterfactual_safety/{name}": value.detach()
                for name, value in counterfactual_metrics.items()
            },
        })
        metrics.update(profile_metrics)
        # Log only heads that the current objective actually trains/uses. The
        # compatibility five-way network is frozen and its categorical metrics
        # would misleadingly look like current Event quality.
        transition_metrics = auxiliary.get("transition_metrics", {})
        current_event_metric_prefixes = (
            "human_", "geometry_human_", "non_human_event_",
            "residual_static_", "next_clearance_",
        )
        metrics.update({
            f"event_prediction/{name}": value.detach()
            for name, value in transition_metrics.items()
            if torch.is_tensor(value) and value.numel() == 1
            and (
                name == "label_available"
                or name.startswith(current_event_metric_prefixes)
            )
        })
        rollout_metrics = {
            **auxiliary.get("overshoot_metrics", {}),
            **ego_overshoot_metrics,
        }
        metrics.update({
            f"world_rollout/{name}": value.detach()
            for name, value in rollout_metrics.items()
            if torch.is_tensor(value) and value.numel() == 1
        })
        # The detached replay posterior is the standard Dreamer start state
        # for Actor/Critic imagination. Returning this exact posterior avoids
        # a second encoder/RSSM pass and keeps the indivisible update anchored
        # to one sampled latent state.
        return total, metrics, states, auxiliary

    def _matched_gt_future_clearance(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        source_count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source-matched future minimum and per-depth clearance.

        Perception slots and privileged simulator slots have unrelated ordering.
        The carried GT identity extends the dataset's one-to-one Hungarian
        correspondence through the causal Actor-visible track hold. Matching
        by that stable ID at every future row prevents the loss from comparing
        one predicted person with another person's simulator truth.
        """
        source_id = batch.get(
            "human_gt_identity_id", batch["human_gt_id"]
        )[:, :source_count].long()
        source_valid = (
            batch.get(
                "human_gt_identity_valid",
                batch["human_gt_match_valid"],
            )[:, :source_count].bool()
            & batch["human_mask"][:, :source_count].bool()
            & source_id.ge(0)
        )
        source_joint_mask = batch["joint_mask"][:, :source_count].bool()
        source_joint_zeros = batch["ego_state"].new_zeros(
            (*source_joint_mask.shape, 3))
        _, _, source_sphere_valid, _ = (
            self.model.deployable_human_collision_geometry(
                source_joint_zeros,
                source_joint_zeros,
                source_joint_mask,
            ))
        source_valid &= source_sphere_valid.any(-1)
        clearance_steps: list[torch.Tensor] = []
        previous_relative: torch.Tensor | None = None
        previous_valid: torch.Tensor | None = None
        previous_radii: torch.Tensor | None = None
        for step in range(0, horizon + 1):
            target_slice = slice(step, step + source_count)
            gt_id = batch["priv_human_id"][:, target_slice].long()
            gt_mask = batch["priv_human_mask"][:, target_slice].bool()
            joints = batch[
                "priv_collision_joints_episode"][:, target_slice].float()
            joint_valid = batch[
                "priv_collision_joint_valid"][:, target_slice].bool()
            people = min(gt_id.shape[-1], joints.shape[-3])
            gt_id = gt_id[..., :people]
            gt_mask = gt_mask[..., :people]
            joints = joints[..., :people, :, :]
            joint_valid = joint_valid[..., :people, :]
            matched_person = (
                gt_id[..., None].eq(source_id[:, :, None, :])
                & source_valid[:, :, None, :]
                & gt_mask[..., None]
            )
            if bool(matched_person.sum(-2).gt(1).any()):
                raise RuntimeError(
                    "privileged Human IDs are not unique within a frame")
            aligned_joints = (
                joints[..., :, None, :, :]
                * matched_person[..., None, None].to(joints)
            ).sum(-4)
            valid = (
                joint_valid[..., :, None, :]
                & matched_person[..., None]
            ).any(-3)
            # Actor imagination keeps only source-observed deployable spheres.
            # Future simulator joints that were unavailable at the source are
            # outside the modeled state and cannot be valid supervision.
            valid &= source_sphere_valid
            ego = batch["ego_state"][:, target_slice, None, None, :3].float()
            radii = batch[
                "counterfactual_collision_surface_radii_m"
            ][:, target_slice].float()
            relative = aligned_joints - ego
            if previous_relative is not None:
                if previous_valid is None or previous_radii is None:
                    raise AssertionError("incomplete swept-clearance state")
                swept_valid = previous_valid & valid
                signed_gap = swept_relative_point_signed_gap(
                    previous_relative, relative, swept_valid,
                    surface_radii_m=torch.maximum(
                        previous_radii, radii)[..., None, :],
                    fallback_gap_m=6.0,
                    per_human=False,
                )
                step_valid = swept_valid.flatten(-2).any(-1)
                clearance_steps.append(signed_gap.masked_fill(
                    ~step_valid, torch.inf))
            previous_relative = relative
            previous_valid = valid
            previous_radii = radii
        future_clearance = torch.stack(clearance_steps, -1)
        future_valid = torch.isfinite(future_clearance).any(-1)
        future_minimum = future_clearance.amin(-1)
        return future_minimum, future_valid, future_clearance

    def _counterfactual_danger_start_indices(
        self, batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select both early-warning and severe replay-supported starts."""
        required = (
            "privileged_geometry_available",
            "priv_human_id",
            "priv_human_mask",
            "priv_collision_joint_valid",
            "priv_collision_joints_episode",
            "counterfactual_collision_surface_radii_m",
            "counterfactual_collision_contact_offset_m",
            "human_gt_id",
            "human_gt_match_valid",
            "human_gt_match_error_m",
            "human_mask",
            "joint_mask",
            "is_last",
        )
        if any(name not in batch for name in required):
            empty = batch["action"].new_empty((0,), dtype=torch.long)
            return empty, empty, empty, empty
        batch_size, sequence_length = batch["action"].shape[:2]
        horizon = int(self.config.imagination_horizon)
        source_count = sequence_length - horizon
        if source_count <= 0:
            empty = batch["action"].new_empty((0,), dtype=torch.long)
            return empty, empty, empty, empty

        future_minimum, valid, future_clearance = (
            self._matched_gt_future_clearance(
                batch, source_count=source_count, horizon=horizon))
        geometry_valid = batch[
            "privileged_geometry_available"].bool().reshape(
                batch_size, sequence_length, -1).any(-1)
        valid &= geometry_valid.unfold(
            1, horizon + 1, 1).all(-1)

        source_human = batch["human_mask"][:, :source_count].bool().any(-1)
        source_joint = batch["joint_mask"][:, :source_count].bool().flatten(
            -2).any(-1)
        valid &= source_human & source_joint
        terminal = batch["is_last"].bool().reshape(
            batch_size, sequence_length, -1).any(-1)
        valid &= ~terminal.unfold(1, horizon, 1)[:, :source_count].any(-1)
        if "sequence_valid" in batch:
            sequence_valid = batch["sequence_valid"].bool().reshape(
                batch_size, sequence_length, -1).all(-1)
            valid &= sequence_valid.unfold(
                1, horizon + 1, 1).all(-1)
        if "action_valid" in batch:
            action_valid = batch["action_valid"].bool().reshape(
                batch_size, sequence_length, -1).all(-1)
            # Row t stores the action that produced state t.  A
            # counterfactual starting at t needs that previous smoother state
            # (unless t is a physical reset) and the factual destination
            # action at t+1 that defines candidate zero.
            previous_valid = action_valid[:, :source_count]
            if "physical_is_first" in batch:
                previous_valid |= batch["physical_is_first"][
                    :, :source_count].bool().reshape(
                        batch_size, source_count, -1).any(-1)
            valid &= previous_valid
            valid &= action_valid[:, 1:source_count + 1]

        if self.model.action_smoother is not None:
            previous_smoothed = self.model.policy_from_applied_action(
                batch["action"][:, :source_count].float())
            desired_smoothed = self.model.policy_from_applied_action(
                batch["action"][:, 1:source_count + 1].float())
            _, baseline_reachable = (
                self.model.action_smoother.raw_target_for_smoothed(
                    desired_smoothed, previous_smoothed))
            valid &= baseline_reachable.squeeze(-1)

        # The same 0.70 m threshold activates the analytic Human-clearance
        # reward.  This selects relevant starts without creating a separate
        # hand-labelled danger definition.
        valid &= future_minimum <= float(self.config.human_safe_clearance_m)
        flat_valid = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if flat_valid.numel() == 0:
            empty = batch["action"].new_empty((0,), dtype=torch.long)
            return empty, empty, empty, empty
        minimum_score = future_minimum.reshape(-1).index_select(0, flat_valid)
        depth = torch.arange(
            1, horizon + 1,
            device=future_clearance.device,
            dtype=torch.long,
        ).view(1, 1, horizon)
        first_danger_depth = torch.where(
            future_clearance <= float(self.config.human_safe_clearance_m),
            depth,
            torch.full_like(depth, horizon + 1),
        ).amin(-1)
        valid_depth = first_danger_depth.reshape(-1).index_select(
            0, flat_valid)
        count = min(
            int(self.config.counterfactual_safety_max_starts),
            int(flat_valid.numel()),
        )
        # Minimum-clearance-only top-k is ambiguous for overlapping windows:
        # every source in the 1.5 s before one collision can share the same
        # eventual minimum.  A global early/severe pair also starves other
        # dangerous sequences in the same batch.  First choose one earliest-
        # warning source from as many distinct sequences as capacity permits;
        # randomly rotating those sequences avoids a fixed batch-order bias.
        # Remaining capacity follows global severity.  This is World-only
        # supervision, not an Actor target or online action search.
        remaining = torch.ones(
            flat_valid.numel(), dtype=torch.bool, device=flat_valid.device)
        chosen_positions: list[torch.Tensor] = []
        valid_batch_index = torch.div(
            flat_valid, source_count, rounding_mode="floor")
        active_sequences = torch.unique(valid_batch_index, sorted=True)
        sequence_order = active_sequences.index_select(
            0, torch.randperm(
                active_sequences.numel(), device=active_sequences.device))
        for sequence_index in sequence_order:
            if len(chosen_positions) >= count:
                break
            candidates = (
                remaining & valid_batch_index.eq(sequence_index)
            ).nonzero(as_tuple=False).squeeze(-1)
            candidate_depth = valid_depth.index_select(0, candidates)
            earliest = candidates.index_select(
                0, candidate_depth.eq(candidate_depth.amax()).nonzero(
                    as_tuple=False).squeeze(-1))
            # If several overlapping sources have the same warning depth,
            # retain the one with the smaller eventual gap.
            early_position = earliest.index_select(
                0, minimum_score.index_select(0, earliest).argmin().view(1)
            ).squeeze(0)
            chosen_positions.append(early_position)
            remaining[early_position] = False
        while len(chosen_positions) < count:
            available = remaining.nonzero(as_tuple=False).squeeze(-1)
            severe_position = available.index_select(
                0, minimum_score.index_select(
                    0, available).argmin().view(1)
            ).squeeze(0)
            chosen_positions.append(severe_position)
            remaining[severe_position] = False
        chosen_position = torch.stack(chosen_positions)
        chosen = flat_valid.index_select(0, chosen_position)
        chosen_depth = valid_depth.index_select(0, chosen_position)
        batch_index = torch.div(chosen, source_count, rounding_mode="floor")
        time_index = chosen.remainder(source_count)
        flat_state_index = batch_index * sequence_length + time_index
        return (
            flat_state_index.long(), batch_index.long(), time_index.long(),
            chosen_depth.long(),
        )

    @staticmethod
    def _all_pair_counterfactual_clearance_ranking(
        predicted_minimum: torch.Tensor,
        truth_minimum: torch.Tensor,
        candidate_valid: torch.Tensor,
        *,
        minimum_truth_margin_m: float,
        required_prediction_margin_m: float,
        normalization_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank every physically distinguishable candidate clearance pair.

        Absolute endpoint and swept-clearance losses fit metric distance. This
        term supplies the missing action ordering: whenever privileged
        geometry separates two candidate command chains, the recursively
        predicted geometry must give the safer chain a positive, robust
        margin. It uses no preferred direction and provides no Actor target.
        """
        if (
            predicted_minimum.ndim != 2
            or truth_minimum.shape != predicted_minimum.shape
            or candidate_valid.shape != predicted_minimum.shape
        ):
            raise ValueError(
                "counterfactual ranking tensors must share [start,candidate]"
            )
        if minimum_truth_margin_m <= 0.0:
            raise ValueError("minimum truth ranking margin must be positive")
        if required_prediction_margin_m <= 0.0:
            raise ValueError(
                "required prediction ranking margin must be positive")
        if normalization_m <= 0.0:
            raise ValueError("ranking normalization must be positive")

        finite_valid = (
            candidate_valid.bool()
            & torch.isfinite(predicted_minimum)
            & torch.isfinite(truth_minimum)
        )
        prediction = torch.where(
            finite_valid, predicted_minimum, torch.zeros_like(predicted_minimum)
        )
        truth = torch.where(
            finite_valid, truth_minimum.detach(), torch.zeros_like(truth_minimum)
        )
        truth_delta = truth[:, :, None] - truth[:, None, :]
        prediction_delta = prediction[:, :, None] - prediction[:, None, :]
        upper_triangle = torch.triu(
            torch.ones(
                predicted_minimum.shape[-1], predicted_minimum.shape[-1],
                dtype=torch.bool, device=predicted_minimum.device,
            ),
            diagonal=1,
        )[None]
        pair_valid = (
            upper_triangle
            & finite_valid[:, :, None]
            & finite_valid[:, None, :]
            & truth_delta.abs().ge(float(minimum_truth_margin_m))
        )
        pair_count = pair_valid.sum().to(predicted_minimum)
        zero = prediction.sum() * 0.0
        if not bool(pair_valid.any()):
            return zero, pair_count, zero.detach(), zero.detach()

        truth_sign = torch.where(
            truth_delta >= 0.0,
            torch.ones_like(truth_delta),
            -torch.ones_like(truth_delta),
        )
        predicted_safer_advantage = truth_sign * prediction_delta
        required_margin = truth_delta.abs().clamp_max(
            float(required_prediction_margin_m))
        shortfall_m = F.relu(
            required_margin - predicted_safer_advantage)
        normalized_shortfall = shortfall_m / float(normalization_m)
        per_pair = F.smooth_l1_loss(
            normalized_shortfall,
            torch.zeros_like(normalized_shortfall),
            reduction="none",
            beta=1.0,
        )
        pair_weight = pair_valid.to(per_pair)
        loss = (per_pair * pair_weight).sum() / pair_count.clamp_min(1.0)
        pairwise_accuracy = (
            predicted_safer_advantage.gt(0.0).to(per_pair) * pair_weight
        ).sum() / pair_count.clamp_min(1.0)
        mean_shortfall_m = (
            shortfall_m * pair_weight
        ).sum() / pair_count.clamp_min(1.0)
        return loss, pair_count, pairwise_accuracy, mean_shortfall_m

    @staticmethod
    def _statewise_counterfactual_clearance_cvar(
        predicted_minimum: torch.Tensor,
        truth_minimum: torch.Tensor,
        candidate_valid: torch.Tensor,
        *,
        minimum_truth_margin_m: float,
        required_prediction_margin_m: float,
        normalization_m: float,
        tail_fraction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Average each state's worst pairwise-clearance loss tail.

        The ordinary all-pair mean weights every valid pair equally.  That is
        useful for population calibration, but it allowed a few severe
        state-local action-order reversals to be diluted by many easy pairs.
        This complementary CVaR term first takes the worst ``tail_fraction``
        of valid pairs *inside each state*, then averages states equally.  It
        remains candidate-permutation invariant and direction agnostic.
        """
        if (
            predicted_minimum.ndim != 2
            or truth_minimum.shape != predicted_minimum.shape
            or candidate_valid.shape != predicted_minimum.shape
        ):
            raise ValueError(
                "counterfactual CVaR tensors must share [start,candidate]")
        if minimum_truth_margin_m <= 0.0:
            raise ValueError("minimum truth ranking margin must be positive")
        if required_prediction_margin_m <= 0.0:
            raise ValueError(
                "required prediction ranking margin must be positive")
        if normalization_m <= 0.0:
            raise ValueError("ranking normalization must be positive")
        if not 0.0 < float(tail_fraction) <= 1.0:
            raise ValueError("counterfactual CVaR tail fraction must lie in (0,1]")

        finite_valid = (
            candidate_valid.bool()
            & torch.isfinite(predicted_minimum)
            & torch.isfinite(truth_minimum)
        )
        prediction = torch.where(
            finite_valid, predicted_minimum, torch.zeros_like(predicted_minimum)
        )
        truth = torch.where(
            finite_valid, truth_minimum.detach(), torch.zeros_like(truth_minimum)
        )
        truth_delta = truth[:, :, None] - truth[:, None, :]
        prediction_delta = prediction[:, :, None] - prediction[:, None, :]
        upper_triangle = torch.triu(
            torch.ones(
                predicted_minimum.shape[-1], predicted_minimum.shape[-1],
                dtype=torch.bool, device=predicted_minimum.device,
            ),
            diagonal=1,
        )[None]
        pair_valid = (
            upper_triangle
            & finite_valid[:, :, None]
            & finite_valid[:, None, :]
            & truth_delta.abs().ge(float(minimum_truth_margin_m))
        )
        truth_sign = torch.where(
            truth_delta >= 0.0,
            torch.ones_like(truth_delta),
            -torch.ones_like(truth_delta),
        )
        safer_advantage = truth_sign * prediction_delta
        shortfall_m = F.relu(
            truth_delta.abs().clamp_max(float(required_prediction_margin_m))
            - safer_advantage
        )
        normalized_shortfall = shortfall_m / float(normalization_m)
        per_pair = F.smooth_l1_loss(
            normalized_shortfall,
            torch.zeros_like(normalized_shortfall),
            reduction="none",
            beta=1.0,
        )

        state_losses: list[torch.Tensor] = []
        state_shortfalls: list[torch.Tensor] = []
        for state_index in range(predicted_minimum.shape[0]):
            valid_loss = per_pair[state_index][pair_valid[state_index]]
            if valid_loss.numel() == 0:
                continue
            tail_count = max(
                1, int(math.ceil(
                    float(tail_fraction) * int(valid_loss.numel()))))
            tail_values, tail_indices = valid_loss.topk(
                tail_count, largest=True, sorted=False)
            valid_shortfall = shortfall_m[state_index][
                pair_valid[state_index]]
            state_losses.append(tail_values.mean())
            state_shortfalls.append(
                valid_shortfall.index_select(0, tail_indices).mean())
        zero = prediction.sum() * 0.0
        if not state_losses:
            return zero, zero.detach(), zero.detach()
        return (
            torch.stack(state_losses).mean(),
            prediction.new_tensor(float(len(state_losses))),
            torch.stack(state_shortfalls).mean(),
        )

    @staticmethod
    def _convex_counterfactual_clearance_ranking(
        population_mean: torch.Tensor,
        statewise_cvar: torch.Tensor,
        *,
        tail_weight: float,
    ) -> torch.Tensor:
        """Keep ranking-loss mass fixed while exposing state-local tails.

        V65 added the full population mean and a second full-scale CVaR term.
        The fixed collision audit showed lower mean false-safe bias but worse
        clearance/return ordering and larger World gradients.  A convex risk
        mixture retains the population objective, exposes the same
        direction-agnostic tail, and cannot silently multiply the ranking
        coefficient when the tail is enabled.
        """
        if population_mean.ndim != 0 or statewise_cvar.ndim != 0:
            raise ValueError("counterfactual ranking losses must be scalars")
        weight = float(tail_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                "counterfactual ranking tail weight must lie in [0,1]")
        return population_mean * (1.0 - weight) + statewise_cvar * weight

    @staticmethod
    def _candidate_episode_trajectory_exogeneity(
        positions_episode: torch.Tensor,
        valid: torch.Tensor,
        *,
        normalization_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Penalize candidate-dependent motion of an exogenous Human.

        ``positions_episode`` is ``[state,candidate,time,...,xyz]`` and
        ``valid`` has the same leading dimensions without ``xyz``.  The
        candidate centroid is a symmetric, direction-free consensus target;
        no candidate acts as a teacher.  A point contributes only when it is
        valid for every candidate so presence differences cannot bias the
        consensus.  The radial Huber loss is expressed in the task's physical
        clearance scale and remains invariant to candidate permutation or a
        common episode-frame translation.
        """
        if (
            positions_episode.ndim < 4
            or positions_episode.shape[-1] != 3
            or valid.shape != positions_episode.shape[:-1]
        ):
            raise ValueError(
                "candidate Human trajectories must be "
                "[state,candidate,time,...,xyz] with matching validity")
        if positions_episode.shape[1] < 2:
            raise ValueError(
                "Human exogeneity requires at least two action candidates")
        if not math.isfinite(float(normalization_m)) or normalization_m <= 0.0:
            raise ValueError(
                "Human exogeneity normalization must be finite and positive")
        common_valid = valid.bool().all(dim=1)
        expanded_valid = common_valid[:, None].expand_as(valid)
        if bool((expanded_valid & ~torch.isfinite(
            positions_episode).all(-1)).any()
        ):
            raise ValueError(
                "valid candidate Human trajectories must be finite")
        centroid = positions_episode.mean(dim=1, keepdim=True)
        deviation_m = torch.linalg.vector_norm(
            positions_episode - centroid, dim=-1)
        per_point = F.smooth_l1_loss(
            deviation_m / float(normalization_m),
            torch.zeros_like(deviation_m),
            reduction="none",
            beta=1.0,
        )
        weight = expanded_valid.to(per_point)
        candidate_point_count = weight.sum()
        loss = (
            (per_point * weight).sum()
            / candidate_point_count.clamp_min(1.0)
        )
        valid_deviation = deviation_m.masked_select(expanded_valid)
        zero = positions_episode.sum() * 0.0
        mean_deviation_m = (
            valid_deviation.mean() if valid_deviation.numel() else zero)
        maximum_deviation_m = (
            valid_deviation.amax() if valid_deviation.numel() else zero)
        return (
            loss,
            candidate_point_count.detach(),
            mean_deviation_m.detach(),
            maximum_deviation_m.detach(),
        )

    def _counterfactual_safety_loss(
        self,
        states: Any,
        auxiliary: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise the exact h15 articulated-geometry path used by Actor.

        Privileged simulator joints provide only world-model targets.  The
        Actor is neither cloned nor assigned a candidate action: its update
        remains the standard reparameterized Dreamer dynamics gradient.
        """
        zero = auxiliary["joint_feat"].sum() * 0.0
        required = (
            "priv_collision_joints_episode",
            "priv_collision_joint_valid",
            "privileged_geometry_available",
            "priv_human_id",
            "priv_human_mask",
            "counterfactual_collision_surface_radii_m",
            "counterfactual_collision_contact_offset_m",
            "human_gt_id",
            "human_gt_match_valid",
            "human_gt_match_error_m",
        )
        if (
            self.config.counterfactual_safety_aux_scale == 0.0
            or any(name not in batch for name in required)
        ):
            return zero, {"start_count": zero.detach()}
        flat_indices, batch_index, time_index, first_danger_depth = (
            self._counterfactual_danger_start_indices(batch))
        if flat_indices.numel() == 0:
            return zero, {"start_count": zero.detach()}

        horizon = int(self.config.imagination_horizon)
        source_gt_id = self._flat_rows(
            batch.get("human_gt_identity_id", batch["human_gt_id"]),
            flat_indices,
        ).long()
        source_match_mask = (
            self._flat_rows(
                batch.get(
                    "human_gt_identity_valid",
                    batch["human_gt_match_valid"],
                ),
                flat_indices,
            ).bool()
            & self._flat_rows(batch["human_mask"], flat_indices).bool()
            & source_gt_id.ge(0)
        )
        source_joint_mask = self._flat_rows(
            batch["joint_mask"], flat_indices).bool()
        source_skeleton = self._flat_rows(
            batch["skeleton"], flat_indices).float()
        source_joint_velocity = (
            source_skeleton[..., 3:6]
            if source_skeleton.shape[-1] >= 6 else
            torch.zeros_like(source_skeleton[..., :3])
        )
        _, _, source_sphere_valid, _ = (
            self.model.deployable_human_collision_geometry(
                source_skeleton[..., :3],
                source_joint_velocity,
                source_joint_mask,
            ))
        source_match_mask &= source_sphere_valid.any(-1)
        if not bool(source_match_mask.any(-1).all()):
            raise RuntimeError(
                "counterfactual start selection retained an unmatched row")
        match_error_slots = self._flat_rows(
            batch["human_gt_match_error_m"], flat_indices).float()
        match_error = match_error_slots[source_match_mask]
        unmatched_count = 0

        flat_state = self.model._flatten_state_time(states)
        initial = self.model._select_state_rows(flat_state, flat_indices)
        people = auxiliary["human_mask"].shape[-1]
        values: dict[str, torch.Tensor] = {
            "human_mask": auxiliary["human_mask"].reshape(
                -1, people).index_select(0, flat_indices),
            "goal_position": self._flat_rows(
                batch["goal_position"], flat_indices).float(),
            "ego_state": self._flat_rows(
                batch["ego_state"], flat_indices).float(),
            "previous_action": self._flat_rows(
                batch["action"], flat_indices).float(),
            "human_root": self._flat_rows(
                batch["human_root"], flat_indices).float(),
            "human_quality": self._flat_rows(
                batch["human_observation_quality"], flat_indices).float(),
            "human_joint_mask": source_joint_mask,
        }
        # Counterfactual World supervision must roll the same complete Markov
        # state as the Actor update.  Omitting these fields used to create a
        # second, geometry-blind rollout path and now fails outright for the
        # exact-task-state Actor.
        counterfactual_task_fields = {
            "task_geometry": "task_geometry",
            "task_origin_xyz": "counterfactual_origin_xyz",
            "task_origin_yaw": "counterfactual_origin_yaw",
            "task_flight_bounds_xy": "counterfactual_flight_bounds_world",
            "task_bounds_valid": "counterfactual_bounds_valid",
            "task_maximum_altitude_m": (
                "counterfactual_maximum_altitude_m"),
            "task_maximum_altitude_valid": (
                "counterfactual_maximum_altitude_valid"),
            "task_crash_altitude_m": (
                "counterfactual_crash_altitude_m"),
            "task_crash_min_elapsed_s": (
                "counterfactual_crash_min_elapsed_s"),
            "task_watchdog_min_elapsed_s": (
                "counterfactual_watchdog_min_elapsed_s"),
            "task_watchdog_no_progress_timeout_s": (
                "counterfactual_watchdog_no_progress_timeout_s"),
            "task_watchdog_progress_epsilon_m": (
                "counterfactual_watchdog_progress_epsilon_m"),
            "task_watchdog_stuck_max_horizontal_speed_mps": (
                "counterfactual_watchdog_stuck_max_horizontal_speed_mps"),
            "goal_radius_m": "counterfactual_goal_radius_m",
            "task_static_obstacle_aabbs_xy": (
                "counterfactual_static_obstacle_aabbs_world"),
            "task_static_obstacle_valid": (
                "counterfactual_static_obstacle_valid"),
            "task_static_terminal_valid": (
                "counterfactual_static_terminal_valid"),
            "task_memory": "task_memory",
        }
        for destination, source in counterfactual_task_fields.items():
            if source in batch:
                values[destination] = self._flat_rows(
                    batch[source], flat_indices)
        if bool(getattr(self.model, "actor_task_physical_state_enabled", False)):
            required_task = set(counterfactual_task_fields)
            missing_task = sorted(required_task.difference(values))
            if missing_task:
                raise KeyError(
                    "full-state counterfactual rollout lacks task fields "
                    f"{missing_task}")
        self._require_current_static_terminal_contract(
            values, context="counterfactual rollout")
        self._require_task_memory_action_contract(
            values, context="counterfactual rollout")
        values["human_joints_body"] = source_skeleton[..., :3]
        if source_skeleton.shape[-1] >= 6:
            values["human_joint_velocity_body"] = source_skeleton[..., 3:6]
        else:
            values["human_joint_velocity_body"] = torch.zeros_like(
                values["human_joints_body"])
        values["human_joint_clearance"] = (
            self.model.deployable_human_clearance(
            values["human_joints_body"],
            values["human_joint_velocity_body"],
            values["human_joint_mask"],
            per_human=True,
        ))

        destination_indices = flat_indices + 1
        # Compact-v3's authoritative action is the command applied to RSSM and
        # the environment.  Candidate zero must reproduce its horizontal/yaw
        # coordinates after the current stateful smoother.  Directly reusing
        # this already-smoothed target as a raw override would filter it twice.
        destination_action = self._flat_rows(
            batch["action"], destination_indices).float().clamp(-1.0, 1.0)
        desired_smoothed_policy = self.model.policy_from_applied_action(
            destination_action)
        previous_smoothed_policy = self.model.policy_from_applied_action(
            values["previous_action"])
        if self.model.action_smoother is None:
            base_policy = desired_smoothed_policy
        else:
            base_policy, baseline_reachable = (
                self.model.action_smoother.raw_target_for_smoothed(
                    desired_smoothed_policy, previous_smoothed_policy))
            if not bool(baseline_reachable.all()):
                raise RuntimeError(
                    "counterfactual selector retained a replay action that "
                    "the current stateful smoother cannot reproduce")
        candidate_count = 5
        candidates = base_policy[:, None].expand(
            -1, candidate_count, -1).clone()
        candidates[:, 1, 0] = float(
            self.config.counterfactual_fast_forward_policy)
        candidates[:, 2, 0] = float(
            self.config.counterfactual_mild_forward_policy)
        candidates[:, 3, 1] = float(
            self.config.counterfactual_lateral_policy)
        candidates[:, 4, 1] = -float(
            self.config.counterfactual_lateral_policy)

        repeated_initial = self.model._repeat_state_candidates(
            initial, candidate_count)

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand(
                -1, candidate_count, *value.shape[1:]
            ).reshape(
                value.shape[0] * candidate_count, *value.shape[1:])

        repeated_values = {
            name: repeat(value)
            for name, value in values.items()
            if torch.is_tensor(value)
        }
        imagined = self._imagine(
            repeated_initial,
            repeated_values,
            horizon_steps=horizon,
            policy_action_override=candidates.reshape(
                -1, candidates.shape[-1]),
            policy_action_override_steps=horizon,
            sample=False,
        )

        baseline_smoothed = imagined["smoothed_policy_action"][
            ::candidate_count, 0]
        baseline_action_error = (
            baseline_smoothed - desired_smoothed_policy).abs().amax()
        if (
            not bool(torch.isfinite(baseline_action_error))
            or float(baseline_action_error.detach()) > 2.0e-5
        ):
            raise RuntimeError(
                "counterfactual baseline failed to reproduce the replay "
                "horizontal/yaw action after stateful smoothing; maximum "
                f"error={float(baseline_action_error.detach()):.8f}")

        candidate_rows = batch_index[:, None].expand(
            -1, candidate_count).reshape(-1)
        candidate_starts = time_index[:, None].expand(
            -1, candidate_count).reshape(-1)
        candidate_source_ids = repeat(source_gt_id)
        candidate_source_valid = repeat(source_match_mask).bool()
        candidate_source_sphere_valid = repeat(source_sphere_valid).bool()
        candidate_source_joint_valid = repeat(source_joint_mask).bool()
        aligned_model_mask = candidate_source_valid
        start_count = int(flat_indices.numel())

        # Candidate UAV commands necessarily change Human coordinates in the
        # moving body frame.  The corresponding episode-frame root and joints
        # are nevertheless action-exogenous in this simulator.  Constrain the
        # learned recursive Human path itself, rather than forcing a policy
        # direction or removing the action needed for the frame transform.
        candidate_root_episode = body_points_to_episode(
            imagined["human_root"][:, 1:horizon + 1, ..., :3],
            imagined["ego_state"][:, 1:horizon + 1],
        ).reshape(
            start_count, candidate_count, horizon, people, 3)
        candidate_joint_episode = body_points_to_episode(
            imagined["human_joints_body"][:, 1:horizon + 1],
            imagined["ego_state"][:, 1:horizon + 1],
        ).reshape(
            start_count, candidate_count, horizon, people,
            source_joint_mask.shape[-1], 3)
        candidate_root_valid = (
            imagined["human_mask_rollout"][:, 1:horizon + 1].bool()
            & candidate_source_valid[:, None, :]
        )
        candidate_joint_valid = (
            candidate_root_valid[..., None]
            & candidate_source_joint_valid[:, None]
        )
        candidate_root_valid = candidate_root_valid.reshape(
            start_count, candidate_count, horizon, people)
        candidate_joint_valid = candidate_joint_valid.reshape(
            start_count, candidate_count, horizon, people,
            source_joint_mask.shape[-1])
        (
            human_root_exogeneity,
            human_root_exogeneity_count,
            human_root_exogeneity_mean_m,
            human_root_exogeneity_max_m,
        ) = self._candidate_episode_trajectory_exogeneity(
            candidate_root_episode,
            candidate_root_valid,
            normalization_m=float(self.config.human_hard_clearance_m),
        )
        (
            human_joint_exogeneity,
            human_joint_exogeneity_count,
            human_joint_exogeneity_mean_m,
            human_joint_exogeneity_max_m,
        ) = self._candidate_episode_trajectory_exogeneity(
            candidate_joint_episode,
            candidate_joint_valid,
            normalization_m=float(self.config.human_hard_clearance_m),
        )
        human_exogeneity = human_root_exogeneity + human_joint_exogeneity
        ego = repeat(values["ego_state"])
        relative_steps: list[torch.Tensor] = []
        aligned_valid_steps: list[torch.Tensor] = []
        surface_radii_steps: list[torch.Tensor] = []
        contact_offset_steps: list[torch.Tensor] = []
        geometry_valid_steps: list[torch.Tensor] = []
        truth_steps: list[torch.Tensor] = []
        truth_valid_steps: list[torch.Tensor] = []
        transition_active_steps: list[torch.Tensor] = []
        transition_stop_fraction_steps: list[torch.Tensor] = []
        task_active = torch.ones(
            candidate_rows.shape[0], dtype=torch.bool,
            device=candidate_rows.device,
        )
        for step in range(0, horizon + 1):
            if step > 0:
                ego = analytic_ego_step(
                    ego, imagined["action"][:, step - 1].detach(),
                    dt_s=float(self.config.imagination_dt_s),
                    velocity_response=(
                        self.model.analytic_ego_velocity_response),
                    attitude_coefficients=(
                        self.model.analytic_ego_attitude_coefficients),
                )
            target_time = candidate_starts + step
            target = batch["priv_collision_joints_episode"][
                candidate_rows, target_time].float()
            target_valid = batch["priv_collision_joint_valid"][
                candidate_rows, target_time].bool()
            target_id = batch["priv_human_id"][
                candidate_rows, target_time].long()
            target_human_mask = batch["priv_human_mask"][
                candidate_rows, target_time].bool()
            target_people = min(target.shape[-3], target_id.shape[-1])
            target = target[:, :target_people]
            target_valid = target_valid[:, :target_people]
            target_id = target_id[:, :target_people]
            target_human_mask = target_human_mask[:, :target_people]
            matched_person = (
                target_id[..., None].eq(candidate_source_ids[:, None, :])
                & candidate_source_valid[:, None, :]
                & target_human_mask[..., None]
            )
            if bool(matched_person.sum(1).gt(1).any()):
                raise RuntimeError(
                    "privileged Human IDs are not unique within a frame")
            aligned_target = (
                target[:, :, None]
                * matched_person[..., None, None].to(target)
            ).sum(1)
            aligned_valid = (
                target_valid[:, :, None]
                & matched_person[..., None]
            ).any(1)
            aligned_valid &= candidate_source_sphere_valid
            geometry_valid = batch["privileged_geometry_available"][
                candidate_rows, target_time].bool().reshape(-1)
            surface_radii = batch[
                "counterfactual_collision_surface_radii_m"
            ][candidate_rows, target_time].float()
            if surface_radii.shape[-1] != aligned_target.shape[-2]:
                raise RuntimeError(
                    "privileged joint/radius topology changed after collation")
            contact_offset = batch[
                "counterfactual_collision_contact_offset_m"
            ][candidate_rows, target_time].float().reshape(-1)
            if not bool(
                torch.isfinite(contact_offset).all()
                and contact_offset.ge(0.0).all()
            ):
                raise RuntimeError(
                    "counterfactual contact offsets must be finite and non-negative")
            aligned_valid &= geometry_valid[:, None, None]
            relative = aligned_target - ego[:, None, None, :3]
            relative_steps.append(relative)
            aligned_valid_steps.append(aligned_valid)
            surface_radii_steps.append(surface_radii)
            contact_offset_steps.append(contact_offset)
            geometry_valid_steps.append(geometry_valid)
            if step > 0:
                transition_index = step - 1
                task_terminal = imagined["analytic_terminal"][
                    :, transition_index].bool().any(-1)
                stop_fraction = torch.where(
                    task_terminal,
                    imagined["analytic_terminal_first_fraction"][
                        :, transition_index, 0].float().clamp(0.0, 1.0),
                    torch.ones_like(task_terminal, dtype=relative.dtype),
                )
                transition_active_steps.append(task_active.clone())
                transition_stop_fraction_steps.append(stop_fraction)
                endpoint_valid = (
                    aligned_valid & task_active[:, None, None]
                    & (~task_terminal)[:, None, None]
                )
                signed_gap = (
                    torch.linalg.vector_norm(relative, dim=-1)
                    - surface_radii[:, None, :]
                ).masked_fill(~endpoint_valid, torch.inf)
                truth_steps.append(signed_gap.flatten(1).amin(-1))
                truth_valid_steps.append(
                    endpoint_valid.flatten(1).any(-1))
                task_active &= ~task_terminal
        truth = torch.stack(truth_steps, 1).clamp(-0.5, 4.0)
        truth_valid = torch.stack(truth_valid_steps, 1)
        swept_truth_steps: list[torch.Tensor] = []
        swept_valid_steps: list[torch.Tensor] = []
        swept_slot_truth_steps: list[torch.Tensor] = []
        swept_slot_valid_steps: list[torch.Tensor] = []
        swept_contact_offset_steps: list[torch.Tensor] = []
        for step in range(horizon):
            swept_valid = (
                aligned_valid_steps[step]
                & aligned_valid_steps[step + 1]
                & transition_active_steps[step][:, None, None]
            )
            # Radii and PhysX contact offsets are episode constants. Taking
            # the larger endpoint also remains conservative if a future
            # recorder deliberately changes collision geometry mid-episode.
            swept_radii = torch.maximum(
                surface_radii_steps[step], surface_radii_steps[step + 1])
            stopped_relative = (
                relative_steps[step]
                + transition_stop_fraction_steps[step][:, None, None, None]
                * (relative_steps[step + 1] - relative_steps[step])
            )
            swept_slot_truth = swept_relative_point_signed_gap(
                relative_steps[step], stopped_relative, swept_valid,
                surface_radii_m=swept_radii[:, None, :],
                fallback_gap_m=6.0,
                per_human=True,
            )
            swept_slot_valid = self._counterfactual_swept_slot_valid(
                swept_valid,
                geometry_valid_steps[step],
                geometry_valid_steps[step + 1],
            )
            swept_slot_truth_steps.append(swept_slot_truth)
            swept_slot_valid_steps.append(swept_slot_valid)
            swept_truth_steps.append(
                swept_slot_truth.masked_fill(
                    ~swept_slot_valid, torch.inf).amin(-1).masked_fill(
                        ~swept_slot_valid.any(-1), 6.0))
            swept_valid_steps.append(swept_slot_valid.any(-1))
            swept_contact_offset_steps.append(torch.maximum(
                contact_offset_steps[step], contact_offset_steps[step + 1]))
        swept_truth = torch.stack(swept_truth_steps, 1).clamp(-0.5, 4.0)
        swept_truth_valid = torch.stack(swept_valid_steps, 1)
        swept_slot_truth = torch.stack(
            swept_slot_truth_steps, 1).clamp(-0.5, 4.0)
        swept_slot_valid = torch.stack(swept_slot_valid_steps, 1)
        swept_contact_offset = torch.stack(
            swept_contact_offset_steps, 1)

        learned_clearance = imagined[
            "learned_human_joint_clearance_next"][:, :horizon]
        rollout_mask = imagined[
            "human_mask_rollout"][:, 1:horizon + 1].bool()
        rollout_mask &= aligned_model_mask[:, None, :]
        learned_clearance = learned_clearance.masked_fill(
            ~rollout_mask, torch.inf)
        predicted = learned_clearance.amin(-1)
        predicted = torch.where(
            rollout_mask.any(-1), predicted,
            torch.full_like(predicted, 6.0),
        ).clamp(-0.5, 4.0)

        def asymmetric_clearance_loss(
            prediction: torch.Tensor,
            target: torch.Tensor,
            valid_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            error = prediction - target.detach()
            # These are the task's existing predictive/safe/hard clearance
            # bands. False-safe overestimation receives the extra one-sided
            # term from the first positive error; the symmetric metric loss still
            # prevents a globally conservative distance collapse.
            danger_weight = (
                1.0
                + 2.0 * target.lt(
                    float(self.config.human_predictive_safe_clearance_m)
                ).to(error)
                + 2.0 * target.lt(
                    float(self.config.human_safe_clearance_m)
                ).to(error)
                + 4.0 * target.lt(
                    float(self.config.human_hard_clearance_m)
                ).to(error)
            )
            false_safe_error = F.relu(error)
            per_item = (
                F.smooth_l1_loss(
                    prediction, target.detach(), reduction="none", beta=0.20)
                + 2.0 * false_safe_error.square()
            )
            weight = danger_weight * valid_mask.to(error)
            loss = (
                (per_item * weight).sum()
                / weight.sum().clamp_min(1.0)
            )
            return loss, false_safe_error

        endpoint_clearance_loss, false_safe = asymmetric_clearance_loss(
            predicted, truth, truth_valid)

        # This is the recorder's literal PhysX contact contract: minimum
        # swept joint surface gap <= serialized contactOffset. Endpoint-only
        # gap<=0 misses between-frame contacts and the contact shell.
        collision_target = swept_truth.le(swept_contact_offset)
        slot_collision_target = swept_slot_truth.le(
            swept_contact_offset[..., None])
        slot_positive = swept_slot_valid & slot_collision_target

        # Actor return consumes recursively predicted swept articulated gaps,
        # not endpoint-only clearance. Supervise that exact geometry path in
        # the same surface-gap convention as privileged truth. The production
        # path subtracts contactOffset inside its contact radii, so add the
        # serialized offset back here before comparing metric distances.
        _, _, predicted_swept_contact_gap, predicted_swept_valid = (
            self._imagined_swept_human_contact(imagined))
        predicted_swept = (
            predicted_swept_contact_gap
            + swept_contact_offset[..., None]
        ).clamp(-0.5, 4.0)
        swept_loss_valid = (
            predicted_swept_valid
            & swept_slot_valid
            & aligned_model_mask[:, None, :]
        )
        swept_clearance_loss, swept_false_safe = (
            asymmetric_clearance_loss(
                predicted_swept, swept_slot_truth, swept_loss_valid))

        truth_by_candidate = swept_slot_truth.reshape(
            start_count, candidate_count, horizon, -1)
        valid_by_candidate = swept_loss_valid.reshape(
            start_count, candidate_count, horizon, -1)
        predicted_by_candidate = predicted_swept.reshape(
            start_count, candidate_count, horizon, -1)
        truth_minimum = truth_by_candidate.masked_fill(
            ~valid_by_candidate, torch.inf).flatten(-2).amin(-1)
        predicted_minimum = predicted_by_candidate.masked_fill(
            ~valid_by_candidate, torch.inf).flatten(-2).amin(-1)
        candidate_valid = valid_by_candidate.flatten(-2).any(-1)
        span_separable = (
            candidate_valid.all(-1)
            & (truth_minimum.amax(-1) - truth_minimum.amin(-1) >= 0.05)
        )
        (
            clearance_ranking,
            clearance_ranking_pair_count,
            clearance_ranking_pairwise_accuracy,
            clearance_ranking_shortfall_m,
        ) = self._all_pair_counterfactual_clearance_ranking(
            predicted_minimum,
            truth_minimum,
            candidate_valid,
            minimum_truth_margin_m=0.05,
            required_prediction_margin_m=0.05,
            normalization_m=max(
                float(self.config.human_hard_clearance_m), 1.0e-3),
        )
        (
            clearance_ranking_cvar,
            clearance_ranking_cvar_state_count,
            clearance_ranking_cvar_shortfall_m,
        ) = self._statewise_counterfactual_clearance_cvar(
            predicted_minimum,
            truth_minimum,
            candidate_valid,
            minimum_truth_margin_m=0.05,
            required_prediction_margin_m=0.05,
            normalization_m=max(
                float(self.config.human_hard_clearance_m), 1.0e-3),
            tail_fraction=float(
                self.config.counterfactual_pairwise_cvar_fraction),
        )

        # Privileged counterfactuals supervise metric geometry and its action
        # ordering only. The Event head is factual replay calibration and is
        # audited at the low-frequency checkpoint boundary; evaluating it here
        # every update would be a diagnostic-only second rollout with no loss
        # gradient and no effect on Actor/Critic semantics.
        clearance_ranking_risk_mixture = (
            self._convex_counterfactual_clearance_ranking(
                clearance_ranking,
                clearance_ranking_cvar,
                tail_weight=float(
                    self.config.counterfactual_pairwise_cvar_scale),
            )
        )
        total = (
            endpoint_clearance_loss
            + swept_clearance_loss
            + clearance_ranking_risk_mixture
            + float(self.config.counterfactual_human_exogeneity_scale)
            * human_exogeneity
        )
        valid_float = truth_valid.to(predicted)
        swept_valid_float = swept_loss_valid.to(predicted_swept)
        conservative = imagined[
            "human_joint_clearance"][:, 1:horizon + 1].masked_fill(
                ~rollout_mask, torch.inf).amin(-1).clamp(-0.5, 4.0)
        return total, {
            "start_count": truth.new_tensor(float(start_count)),
            "source_sequence_count": truth.new_tensor(float(
                torch.unique(batch_index).numel())),
            "unmatched_start_count": truth.new_tensor(
                float(unmatched_count)),
            "baseline_action_max_error": baseline_action_error.detach(),
            "source_match_error_m": match_error.mean().detach(),
            "first_danger_step_mean": (
                first_danger_depth.float().mean().detach()),
            "first_danger_time_s_mean": (
                first_danger_depth.float().mean()
                * float(self.config.imagination_dt_s)).detach(),
            "endpoint_clearance_loss": endpoint_clearance_loss.detach(),
            "swept_clearance_loss": swept_clearance_loss.detach(),
            "clearance_loss": (
                endpoint_clearance_loss + swept_clearance_loss).detach(),
            "slot_collision_count": (
                slot_positive.sum().to(truth).detach()),
            "clearance_ranking": clearance_ranking.detach(),
            "clearance_ranking_cvar": clearance_ranking_cvar.detach(),
            "clearance_ranking_risk_mixture": (
                clearance_ranking_risk_mixture.detach()),
            "clearance_ranking_cvar_state_count": (
                clearance_ranking_cvar_state_count.detach()),
            "clearance_ranking_cvar_shortfall_m": (
                clearance_ranking_cvar_shortfall_m.detach()),
            "clearance_ranking_pair_count": (
                clearance_ranking_pair_count.detach()),
            "clearance_ranking_pairwise_accuracy": (
                clearance_ranking_pairwise_accuracy.detach()),
            "clearance_ranking_shortfall_m": (
                clearance_ranking_shortfall_m.detach()),
            "human_episode_exogeneity": human_exogeneity.detach(),
            "human_root_episode_exogeneity": (
                human_root_exogeneity.detach()),
            "human_joint_episode_exogeneity": (
                human_joint_exogeneity.detach()),
            "human_root_episode_exogeneity_candidate_point_count": (
                human_root_exogeneity_count),
            "human_joint_episode_exogeneity_candidate_point_count": (
                human_joint_exogeneity_count),
            "human_root_episode_exogeneity_mean_deviation_m": (
                human_root_exogeneity_mean_m),
            "human_root_episode_exogeneity_max_deviation_m": (
                human_root_exogeneity_max_m),
            "human_joint_episode_exogeneity_mean_deviation_m": (
                human_joint_exogeneity_mean_m),
            "human_joint_episode_exogeneity_max_deviation_m": (
                human_joint_exogeneity_max_m),
            "truth_clearance_m": (
                (truth * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)).detach(),
            "learned_clearance_m": (
                (predicted * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)).detach(),
            "conservative_clearance_m": (
                (conservative * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)).detach(),
            "false_safe_m": (
                (false_safe * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)).detach(),
            "swept_truth_clearance_m": (
                (swept_slot_truth * swept_valid_float).sum()
                / swept_valid_float.sum().clamp_min(1.0)).detach(),
            "swept_learned_clearance_m": (
                (predicted_swept * swept_valid_float).sum()
                / swept_valid_float.sum().clamp_min(1.0)).detach(),
            "swept_false_safe_m": (
                (swept_false_safe * swept_valid_float).sum()
                / swept_valid_float.sum().clamp_min(1.0)).detach(),
            "collision_candidate_ratio": (
                (collision_target & swept_truth_valid).float().sum()
                / swept_truth_valid.float().sum().clamp_min(1.0)).detach(),
            "swept_contact_margin_m": (
                ((swept_truth - swept_contact_offset)
                 * swept_truth_valid.to(swept_truth)).sum()
                / swept_truth_valid.float().sum().clamp_min(1.0)).detach(),
            "truth_action_span_m": (
                truth_minimum.amax(-1) - truth_minimum.amin(-1)
            )[span_separable].mean().detach() if bool(
                span_separable.any()) else zero.detach(),
            "predicted_action_span_m": (
                predicted_minimum.amax(-1) - predicted_minimum.amin(-1)
            )[span_separable].mean().detach() if bool(
                span_separable.any()) else zero.detach(),
        }

    @staticmethod
    def _counterfactual_swept_slot_valid(
        swept_joint_valid: torch.Tensor,
        source_geometry_valid: torch.Tensor,
        destination_geometry_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce joint validity without confusing candidates and people."""
        if swept_joint_valid.ndim != 3:
            raise ValueError(
                "counterfactual swept joint validity must be [C,N,J]")
        candidates = swept_joint_valid.shape[0]
        if source_geometry_valid.shape != (candidates,) or (
            destination_geometry_valid.shape != (candidates,)
        ):
            raise ValueError(
                "counterfactual geometry validity must be [C]")
        return (
            swept_joint_valid.bool().any(-1)
            & source_geometry_valid.bool()[:, None]
            & destination_geometry_valid.bool()[:, None]
        )

    def _flat_valid_starts(
        self, batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        valid = ~batch["is_last"].bool().reshape(
            batch["is_last"].shape[0], batch["is_last"].shape[1], -1).any(-1)
        if "sequence_valid" in batch:
            valid &= batch["sequence_valid"].bool().reshape_as(valid)
        if "action_valid" in batch:
            action_valid = batch["action_valid"].bool().reshape_as(valid)
            physical_first = batch.get("physical_is_first")
            if physical_first is not None:
                action_valid |= physical_first.bool().reshape_as(valid)
            valid &= action_valid
        physical_first = batch.get("physical_is_first")
        burn_in = int(self.config.imagination_window_burn_in)
        if physical_first is not None and burn_in > 0:
            physical = physical_first.bool().reshape_as(valid)
            artificial_window = ~physical.any(dim=1)
            offset = torch.arange(
                valid.shape[1], device=valid.device)[None]
            valid &= ~(
                artificial_window[:, None] & offset.lt(burn_in))
        return valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    @staticmethod
    def _flat_rows(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return value.reshape(-1, *value.shape[2:]).index_select(0, indices)

    def _measure_actor_human_action_response(
        self,
    ) -> tuple[float, float, float]:
        """Measure the actual discrete horizontal step response.

        Adding the software-filter and vehicle-loop time constants is not a
        rise-time measurement once the policy target also passes through a
        nonlinear slew limiter.  Roll the exact deployed command chain from
        rest under a unit horizontal target instead.  The returned time is
        the slower horizontal axis' first crossing of the versioned 80% step
        response, and the second value is the slower axis' response at the
        final imagination transition.
        """
        response = torch.as_tensor(
            getattr(
                self.model, "analytic_ego_velocity_response", (1.0, 1.0)),
            dtype=torch.float64,
        ).reshape(-1)[:2]
        if response.numel() != 2 or not bool(torch.isfinite(response).all()) \
                or bool((response <= 0.0).any()) \
                or bool((response > 1.0).any()):
            raise RuntimeError(
                "Actor diagnostics require valid horizontal velocity response")
        smoother = getattr(self.model, "action_smoother", None)
        action_dim = int(getattr(smoother, "action_dim", 3))
        if action_dim < 2:
            raise RuntimeError("Actor action smoother lacks horizontal axes")
        device = (
            smoother.maximum_step_delta.device
            if smoother is not None else response.device)
        dtype = torch.float64
        raw_target = torch.ones(1, action_dim, device=device, dtype=dtype)
        smoothed = torch.zeros_like(raw_target)
        velocity = torch.zeros(2, device=device, dtype=dtype)
        horizontal_response = response.to(device=device, dtype=dtype)
        target = float(self.ACTOR_HORIZONTAL_STEP_RESPONSE_TARGET)
        actionable_target = float(
            self.ACTOR_ACTIONABLE_HORIZONTAL_RESPONSE_TARGET)
        dt_s = float(self.config.imagination_dt_s)
        horizon_steps = int(self.config.imagination_horizon)
        if (
            horizon_steps <= 0
            or not 0.0 < actionable_target < target < 1.0
        ):
            raise RuntimeError("Actor response measurement contract is invalid")
        crossing_step = torch.zeros(2, device=device, dtype=torch.long)
        actionable_crossing_step = torch.zeros(
            2, device=device, dtype=torch.long)
        horizon_fraction: torch.Tensor | None = None
        maximum_steps = max(horizon_steps, int(math.ceil(30.0 / dt_s)))
        with torch.no_grad():
            for step in range(1, maximum_steps + 1):
                smoothed = (
                    raw_target
                    if smoother is None else smoother(raw_target, smoothed))
                velocity = velocity + horizontal_response * (
                    smoothed[0, :2] - velocity)
                newly_crossed = crossing_step.eq(0) & velocity.ge(target)
                crossing_step = torch.where(
                    newly_crossed,
                    torch.full_like(crossing_step, step),
                    crossing_step,
                )
                newly_actionable = (
                    actionable_crossing_step.eq(0)
                    & velocity.ge(actionable_target))
                actionable_crossing_step = torch.where(
                    newly_actionable,
                    torch.full_like(actionable_crossing_step, step),
                    actionable_crossing_step,
                )
                if step == horizon_steps:
                    horizon_fraction = velocity.clone()
                if bool(crossing_step.gt(0).all()) and horizon_fraction is not None:
                    break
        if (
            horizon_fraction is None
            or not bool(crossing_step.gt(0).all())
            or not bool(actionable_crossing_step.gt(0).all())
        ):
            raise RuntimeError(
                "Actor horizontal command chain did not reach its response target")
        response_time = float(crossing_step.max()) * dt_s
        actionable_response_time = (
            float(actionable_crossing_step.max()) * dt_s)
        horizon_response = float(horizon_fraction.min())
        if (
            not math.isfinite(response_time)
            or not math.isfinite(actionable_response_time)
            or not math.isfinite(horizon_response)
        ):
            raise RuntimeError("Actor actuator response measurement is invalid")
        return response_time, horizon_response, actionable_response_time

    @staticmethod
    def _allocate_imagination_start_counts(
        populations: Sequence[int],
        maximum: int,
        actionable_target: int,
        imminent_target: int,
        preemptive_target: int,
        near_goal_ymax_misaligned_target: int,
        side_boundary_front_human_target: int,
        near_goal_target: int,
        near_boundary_target: int,
    ) -> list[int]:
        """Allocate a non-empty sample to every represented stratum."""
        population = [int(value) for value in populations]
        total = sum(population)
        if total <= int(maximum):
            return population
        nonempty = [index for index, value in enumerate(population) if value]
        if int(maximum) < len(nonempty):
            raise ValueError(
                "max_imagination_starts cannot represent every non-empty "
                "imagination-start stratum")
        selected = [int(value > 0) for value in population]
        remaining = int(maximum) - sum(selected)

        # The final index is ordinary replay. Grow every rare stratum in
        # lockstep up to its requested coverage before assigning the remaining
        # budget to ordinary visitation. This avoids an ordering advantage
        # among collision, compound task, and generic task strata.
        rare_targets = (
            min(population[0], max(1, int(actionable_target)))
            if population[0] else 0,
            min(population[1], max(1, int(imminent_target)))
            if population[1] else 0,
            min(population[2], max(1, int(preemptive_target)))
            if population[2] else 0,
            min(population[3], max(
                1, int(near_goal_ymax_misaligned_target)))
            if population[3] else 0,
            min(population[4], max(
                1, int(side_boundary_front_human_target)))
            if population[4] else 0,
            min(population[5], max(1, int(near_goal_target)))
            if population[5] else 0,
            min(population[6], max(1, int(near_boundary_target)))
            if population[6] else 0,
        )
        while remaining > 0 and any(
            selected[index] < rare_targets[index]
            for index in range(len(rare_targets))
        ):
            for index in range(len(rare_targets)):
                if remaining and selected[index] < rare_targets[index]:
                    selected[index] += 1
                    remaining -= 1
        ordinary_index = len(population) - 1
        if remaining and selected[ordinary_index] < population[ordinary_index]:
            addition = min(
                remaining,
                population[ordinary_index] - selected[ordinary_index],
            )
            selected[ordinary_index] += addition
            remaining -= addition
        # If ordinary rows are scarce, use the full compute budget rather
        # than discarding available rare rows.
        while remaining > 0:
            progressed = False
            for index in range(len(population)):
                if remaining and selected[index] < population[index]:
                    selected[index] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        if sum(selected) != int(maximum):
            raise RuntimeError("imagination-start allocation lost its budget")
        return selected

    def _imagination_start_strata(
        self,
        batch: Mapping[str, torch.Tensor],
        valid_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Partition valid rows into disjoint rare-decision strata."""
        required = (
            "future_collision", "time_to_collision_valid",
            "time_to_collision_s",
        )
        if any(name not in batch for name in required):
            empty = valid_indices.new_empty((0,))
            return (
                empty, empty, empty, empty, empty, empty, empty,
                valid_indices,
            )
        leading = batch["is_last"].shape[:2]

        def flat_scalar(name: str) -> torch.Tensor:
            value = batch[name]
            if value.shape[:2] != leading:
                raise ValueError(
                    f"{name} does not match replay batch/time dimensions")
            flattened = value.reshape(*leading, -1)
            if flattened.shape[-1] != 1:
                raise ValueError(f"{name} must contain one scalar per row")
            return flattened.reshape(-1).index_select(0, valid_indices)

        collision = flat_scalar("future_collision").bool()
        ttc_valid = flat_scalar("time_to_collision_valid").bool()
        ttc = flat_scalar("time_to_collision_s").float()
        finite = torch.isfinite(ttc)
        labelled = collision & ttc_valid & finite & ttc.ge(0.0)
        response_time = float(self.actor_actionable_response_time_s)
        horizon_time = (
            float(self.config.imagination_horizon)
            * float(self.config.imagination_dt_s))
        actionable_mask = (
            labelled & ttc.ge(response_time) & ttc.le(horizon_time))
        imminent_mask = labelled & ttc.lt(response_time)
        preemptive_mask = (
            labelled & ttc.gt(horizon_time)
            & ttc.le(float(self.config.human_ttc_horizon_s)))
        collision_strata_mask = (
            actionable_mask | imminent_mask | preemptive_mask)
        selected_ego: torch.Tensor | None = None
        selected_goal: torch.Tensor | None = None
        if "goal_position" not in batch or "ego_state" not in batch:
            near_goal_base = torch.zeros_like(actionable_mask)
            goal_misaligned = torch.zeros_like(actionable_mask)
        else:
            goal = batch["goal_position"].float()
            ego = batch["ego_state"].float()
            if goal.shape[:2] != leading or ego.shape[:2] != leading:
                raise ValueError(
                    "goal_position/ego_state do not match replay dimensions")
            if goal.shape[-1] != 3 or ego.shape[-1] < 14:
                raise ValueError("near-goal strata require 3-D positions")
            selected_ego = ego.reshape(-1, ego.shape[-1]).index_select(
                0, valid_indices)
            selected_goal = goal.reshape(-1, 3).index_select(
                0, valid_indices)
            goal_delta = selected_goal - selected_ego[..., :3]
            goal_distance = torch.linalg.vector_norm(goal_delta, dim=-1)
            near_goal_base = (
                torch.isfinite(goal_distance)
                & goal_distance.le(float(self.config.near_goal_start_distance_m))
            )
            route_goal_xy = selected_goal[..., :2]
            route_goal_norm = torch.linalg.vector_norm(
                route_goal_xy, dim=-1).clamp_min(1.0e-6)
            cross_track = (
                route_goal_xy[..., 0] * selected_ego[..., 1]
                - route_goal_xy[..., 1] * selected_ego[..., 0]
            ).abs() / route_goal_norm
            sin_yaw = selected_ego[..., 12]
            cos_yaw = selected_ego[..., 13]
            forward_route = (
                cos_yaw * route_goal_xy[..., 0]
                + sin_yaw * route_goal_xy[..., 1])
            lateral_route = (
                -sin_yaw * route_goal_xy[..., 0]
                + cos_yaw * route_goal_xy[..., 1])
            heading_error = torch.atan2(
                lateral_route.abs(), forward_route)
            goal_misaligned = (
                torch.isfinite(cross_track)
                & torch.isfinite(heading_error)
                & (
                    cross_track.gt(float(
                        self.config.route_cross_track_tolerance_m))
                    | heading_error.gt(float(
                        self.NEAR_GOAL_ROUTE_HEADING_ERROR_THRESHOLD_RAD))
                )
            )
        boundary_fields = (
            "ego_state", "counterfactual_origin_xyz",
            "counterfactual_origin_yaw",
            "counterfactual_flight_bounds_world",
            "counterfactual_bounds_valid",
        )
        if any(name not in batch for name in boundary_fields):
            bounds_valid = torch.zeros_like(actionable_mask)
            boundary_clearance = torch.full_like(
                ttc, torch.inf, dtype=torch.float32)
            side_boundary_clearance = boundary_clearance
            ymax_clearance = boundary_clearance
        else:
            ego = batch["ego_state"].float()
            selected_ego = ego.reshape(-1, ego.shape[-1]).index_select(
                0, valid_indices)
            origin = batch["counterfactual_origin_xyz"].float().reshape(
                -1, 3).index_select(0, valid_indices)
            yaw = batch["counterfactual_origin_yaw"].float().reshape(
                -1).index_select(0, valid_indices)
            bounds = batch[
                "counterfactual_flight_bounds_world"
            ].float().reshape(-1, 4).index_select(0, valid_indices)
            bounds_valid = flat_scalar(
                "counterfactual_bounds_valid").bool()
            cosine, sine = yaw.cos(), yaw.sin()
            world_x = (
                origin[..., 0]
                + cosine * selected_ego[..., 0]
                - sine * selected_ego[..., 1])
            world_y = (
                origin[..., 1]
                + sine * selected_ego[..., 0]
                + cosine * selected_ego[..., 1])
            boundary_clearances = torch.stack((
                world_x - bounds[..., 0],
                bounds[..., 1] - world_x,
                world_y - bounds[..., 2],
                bounds[..., 3] - world_y,
            ), -1)
            boundary_clearance = boundary_clearances.amin(-1)
            side_boundary_clearance = boundary_clearances[..., :2].amin(-1)
            ymax_clearance = boundary_clearances[..., 3]

        front_human = torch.zeros_like(actionable_mask)
        if "human_root" in batch and "human_mask" in batch:
            human_root = batch["human_root"].float()
            human_mask = batch["human_mask"].bool()
            if (
                human_root.shape[:2] != leading
                or human_mask.shape[:2] != leading
                or human_root.shape[:-1] != human_mask.shape
                or human_root.shape[-1] < 3
            ):
                raise ValueError(
                    "front-Human strata require aligned root/mask slots")
            people = human_mask.shape[-1]
            selected_root = human_root.reshape(
                -1, people, human_root.shape[-1]).index_select(
                    0, valid_indices)
            selected_human_mask = human_mask.reshape(
                -1, people).index_select(0, valid_indices)
            human_xy = selected_root[..., :2]
            lookahead = float(self.config.human_corridor_lookahead_m)
            front_human = (
                selected_human_mask
                & torch.isfinite(human_xy).all(-1)
                & selected_root[..., 0].gt(0.0)
                & selected_root[..., 0].le(lookahead)
                & torch.linalg.vector_norm(human_xy, dim=-1).le(lookahead)
            ).any(-1)

        # H15 reaches 4.65 m at the maximum audited horizontal speed. Reserve
        # the compound quotas for the preemptive band between the generic
        # 1.5 m boundary zone and that H15 edge. Replay audit showed that the
        # <=1.5 m compound rows were overwhelmingly already-failing states
        # with saturated forward actions and misleading local gradients. They
        # remain in ordinary near-goal/near-boundary replay, but no longer gain
        # a second late-state quota at the expense of recoverable precursors.
        ymax_response_max = (
            float(self.config.task_maximum_horizontal_speed_mps)
            * horizon_time)
        preemptive_boundary_min = float(
            self.config.near_boundary_start_clearance_m)
        near_goal_ymax_misaligned_mask = (
            ~collision_strata_mask
            & near_goal_base
            & goal_misaligned
            & bounds_valid
            & torch.isfinite(ymax_clearance)
            & ymax_clearance.gt(preemptive_boundary_min)
            & ymax_clearance.le(ymax_response_max)
        )
        side_boundary_front_human_mask = (
            ~(collision_strata_mask | near_goal_ymax_misaligned_mask)
            & bounds_valid
            & torch.isfinite(side_boundary_clearance)
            & side_boundary_clearance.gt(preemptive_boundary_min)
            & side_boundary_clearance.le(ymax_response_max)
            & front_human
        )
        compound_mask = (
            collision_strata_mask
            | near_goal_ymax_misaligned_mask
            | side_boundary_front_human_mask)
        near_goal_mask = ~compound_mask & near_goal_base
        near_boundary_mask = (
            ~(compound_mask | near_goal_mask)
            & bounds_valid
            & torch.isfinite(boundary_clearance)
            & boundary_clearance.le(float(
                self.config.near_boundary_start_clearance_m))
        )
        ordinary_mask = ~(
            compound_mask | near_goal_mask | near_boundary_mask)
        return tuple(
            valid_indices.index_select(
                0, mask.nonzero(as_tuple=False).squeeze(-1))
            for mask in (
                actionable_mask,
                imminent_mask,
                preemptive_mask,
                near_goal_ymax_misaligned_mask,
                side_boundary_front_human_mask,
                near_goal_mask,
                near_boundary_mask,
                ordinary_mask,
            )
        )

    def _sample_imagination_starts(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Stratify rare decisions while preserving uniform replay value.

        Future collision labels influence only which posterior rows receive
        limited compute. The returned per-row Horvitz-Thompson weights exactly
        restore the uniform valid-state replay expectation used by standard
        Dreamer, so no privileged label enters policy state, reward, or the
        target distribution.
        """
        indices = self._flat_valid_starts(batch)
        if indices.numel() == 0:
            raise ValueError("replay batch has no valid imagination start")
        maximum = int(self.config.max_imagination_starts)
        stratum_names = (
            "actionable", "imminent", "preemptive",
            "near_goal_ymax_misaligned", "side_boundary_front_human",
            "near_goal", "near_boundary", "ordinary",
        )
        if indices.numel() <= maximum:
            importance = torch.ones(
                indices.numel(), device=indices.device, dtype=torch.float32)
            strata = self._imagination_start_strata(batch, indices)
            counts = [int(value.numel()) for value in strata]
            metrics = {
                "actor_imagination_start_population_count": importance.new_tensor(
                    float(indices.numel())),
            }
            for name, count in zip(stratum_names, counts):
                metrics[
                    f"actor_imagination_{name}_population_count"
                ] = importance.new_tensor(float(count))
                metrics[
                    f"actor_imagination_{name}_selected_count"
                ] = importance.new_tensor(float(count))
            return indices, importance, metrics

        strata = self._imagination_start_strata(batch, indices)
        populations = [int(value.numel()) for value in strata]
        if all(population == 0 for population in populations[:-1]):
            order = torch.randperm(indices.numel(), device=indices.device)
            selected = indices.index_select(0, order[:maximum])
            importance = torch.ones(
                maximum, device=indices.device, dtype=torch.float32)
            metrics = {
                "actor_imagination_start_population_count": (
                    importance.new_tensor(float(indices.numel()))),
            }
            for name in stratum_names[:-1]:
                metrics[
                    f"actor_imagination_{name}_population_count"
                ] = importance.new_zeros(())
                metrics[
                    f"actor_imagination_{name}_selected_count"
                ] = importance.new_zeros(())
            metrics["actor_imagination_ordinary_population_count"] = (
                importance.new_tensor(float(indices.numel())))
            metrics["actor_imagination_ordinary_selected_count"] = (
                importance.new_tensor(float(maximum)))
            return selected, importance, metrics
        selected_counts = self._allocate_imagination_start_counts(
            populations,
            maximum,
            self.config.actionable_collision_start_target,
            self.config.imminent_collision_start_target,
            self.config.preemptive_collision_start_target,
            self.config.near_goal_start_target,
            self.config.near_boundary_start_target,
            self.config.near_goal_start_target,
            self.config.near_boundary_start_target,
        )
        selected_parts = []
        importance_parts = []
        total_population = float(indices.numel())
        for stratum, population, count in zip(
            strata, populations, selected_counts,
        ):
            if count == 0:
                continue
            order = torch.randperm(stratum.numel(), device=stratum.device)
            selected_parts.append(stratum.index_select(0, order[:count]))
            # The final .mean() is over `maximum` sampled starts. This factor
            # makes it the stratified estimator of the N-state replay mean.
            importance_parts.append(torch.full(
                (count,),
                float(maximum * population) / (total_population * count),
                device=indices.device,
                dtype=torch.float32,
            ))
        selected = torch.cat(selected_parts)
        importance = torch.cat(importance_parts)
        order = torch.randperm(selected.numel(), device=selected.device)
        selected = selected.index_select(0, order)
        importance = importance.index_select(0, order)
        if not torch.allclose(
            importance.mean(), importance.new_ones(()),
            rtol=1.0e-5, atol=1.0e-6,
        ):
            raise RuntimeError(
                "stratified imagination-start weights do not preserve mass")
        metrics = {
            "actor_imagination_start_population_count": importance.new_tensor(
                total_population),
        }
        for name, population, count in zip(
            stratum_names, populations, selected_counts,
        ):
            metrics[
                f"actor_imagination_{name}_population_count"
            ] = importance.new_tensor(float(population))
            metrics[
                f"actor_imagination_{name}_selected_count"
            ] = importance.new_tensor(float(count))
        return selected, importance, metrics

    def _sample_imagination_start_indices(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compatibility helper returning only sampled flat row indices."""
        return self._sample_imagination_starts(batch)[0]

    def _imagination_inputs(
        self,
        states: Any,
        auxiliary: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        indices, importance, sampling_metrics = (
            self._sample_imagination_starts(batch))
        initial, values = self._imagination_rows(
            states, auxiliary, batch, indices)
        values["actor_imagination_start_count"] = indices.new_tensor(
            float(indices.numel()), dtype=torch.float32)
        values["actor_imagination_start_importance"] = importance[:, None]
        values.update(sampling_metrics)
        return initial, values

    def _imagination_rows(
        self,
        states: Any,
        auxiliary: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Build the production imagination state for explicit flat rows.

        Audits and world-only counterfactuals must use this same constructor;
        rebuilding an older subset of Actor state can otherwise make a green
        evaluation describe a different policy from training/deployment.
        """
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("imagination row indices must be a 1-D long tensor")
        if indices.numel() == 0:
            raise ValueError("imagination rows cannot be empty")
        flat_state = self.model._flatten_state_time(states)
        state_rows = next(iter(flat_state["ego"].values())).shape[0]
        if bool((indices < 0).any()) or bool((indices >= state_rows).any()):
            raise IndexError("imagination row index is outside posterior state")
        initial = self.model._select_state_rows(flat_state, indices)
        values: dict[str, torch.Tensor] = {
            "human_mask": auxiliary["human_mask"].reshape(
                -1, auxiliary["human_mask"].shape[-1]).index_select(0, indices),
            "goal_position": self._flat_rows(batch["goal_position"], indices),
            "ego_state": self._flat_rows(batch["ego_state"], indices).float(),
            "previous_action": self._flat_rows(batch["action"], indices).float(),
        }
        initial_horizontal_speed = torch.linalg.vector_norm(
            values["ego_state"][..., 3:5], dim=-1)
        if bool((initial_horizontal_speed > (
            float(self.config.task_maximum_horizontal_speed_mps) + 1.0e-5
        )).any()):
            raise ValueError(
                "imagination start exceeds the audited horizontal-speed "
                "contract")
        optional_task_fields = {
            "task_geometry": "task_geometry",
            "task_origin_xyz": "counterfactual_origin_xyz",
            "task_origin_yaw": "counterfactual_origin_yaw",
            "task_flight_bounds_xy": "counterfactual_flight_bounds_world",
            "task_bounds_valid": "counterfactual_bounds_valid",
            "task_maximum_altitude_m": (
                "counterfactual_maximum_altitude_m"),
            "task_maximum_altitude_valid": (
                "counterfactual_maximum_altitude_valid"),
            "task_crash_altitude_m": "counterfactual_crash_altitude_m",
            "task_crash_min_elapsed_s": (
                "counterfactual_crash_min_elapsed_s"),
            "task_memory": "task_memory",
            "task_watchdog_min_elapsed_s": (
                "counterfactual_watchdog_min_elapsed_s"),
            "task_watchdog_no_progress_timeout_s": (
                "counterfactual_watchdog_no_progress_timeout_s"),
            "task_watchdog_progress_epsilon_m": (
                "counterfactual_watchdog_progress_epsilon_m"),
            "task_watchdog_stuck_max_horizontal_speed_mps": (
                "counterfactual_watchdog_stuck_max_horizontal_speed_mps"),
            "goal_radius_m": "counterfactual_goal_radius_m",
            "task_static_obstacle_aabbs_xy": (
                "counterfactual_static_obstacle_aabbs_world"),
            "task_static_obstacle_valid": (
                "counterfactual_static_obstacle_valid"),
            "task_static_terminal_valid": (
                "counterfactual_static_terminal_valid"),
        }
        for destination, source in optional_task_fields.items():
            if source in batch:
                values[destination] = self._flat_rows(
                    batch[source], indices)
        self._require_current_static_terminal_contract(
            values, context="Actor imagination")
        if bool(getattr(self.model, "task_memory_enabled", False)):
            if "task_memory" not in values:
                raise KeyError(
                    "full-state pure Dreamer requires deployable task_memory")
            self._require_task_memory_action_contract(
                values, context="Actor imagination")
            expected_watchdog = {
                "task_crash_altitude_m": (
                    self.config.task_crash_altitude_world_m),
                "task_crash_min_elapsed_s": (
                    self.config.crash_min_elapsed_s),
                "task_watchdog_min_elapsed_s": (
                    self.config.watchdog_min_elapsed_s),
                "task_watchdog_no_progress_timeout_s": (
                    self.config.watchdog_no_progress_timeout_s),
                "task_watchdog_progress_epsilon_m": (
                    self.config.watchdog_progress_epsilon_m),
                "task_watchdog_stuck_max_horizontal_speed_mps": (
                    self.config.watchdog_stuck_max_horizontal_speed_mps),
            }
            for key, expected in expected_watchdog.items():
                actual = values.get(key)
                if actual is None:
                    raise KeyError(f"replay is missing {key}")
                if not torch.allclose(
                    actual.float(), torch.full_like(actual.float(), expected),
                    atol=1.0e-5, rtol=0.0,
                ):
                    raise ValueError(
                        f"replay {key} differs from objective config")
        geometry_sources = (
            "human_root", "human_observation_quality", "joint_mask")
        if all(source in batch for source in geometry_sources):
            values["human_root"] = self._flat_rows(
                batch["human_root"], indices)
            values["human_quality"] = self._flat_rows(
                batch["human_observation_quality"], indices)
            values["human_joint_mask"] = self._flat_rows(
                batch["joint_mask"], indices)
        if "human_root" in values and "skeleton" in batch:
            skeleton = self._flat_rows(batch["skeleton"], indices).float()
            values["human_joints_body"] = skeleton[..., :3]
            if skeleton.shape[-1] >= 6:
                values["human_joint_velocity_body"] = skeleton[..., 3:6]
            values["human_joint_clearance"] = (
                self.model.deployable_human_clearance(
                values["human_joints_body"],
                values["human_joint_velocity_body"],
                values["human_joint_mask"].bool(),
                per_human=True,
            ))
        if self.config.imagined_explicit_human_event:
            required_values = {
                "human_root", "human_quality", "human_joint_mask",
                "human_joints_body",
            }
            missing = sorted(required_values.difference(values))
            if missing:
                raise KeyError(
                    "explicit imagined Human Event lacks replay context "
                    f"{missing}")
        return initial, values

    def _imagine(
        self,
        initial: Any,
        values: Mapping[str, torch.Tensor],
        *,
        first_applied_action: torch.Tensor | None = None,
        first_policy_action: torch.Tensor | None = None,
        horizon_steps: int | None = None,
        sample: bool = True,
        antithetic_policy_pairs: bool = False,
        policy_action_override: torch.Tensor | None = None,
        policy_action_override_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        horizon = (
            self.config.imagination_horizon
            if horizon_steps is None else int(horizon_steps))
        if horizon <= 0:
            raise ValueError("imagination horizon must be positive")
        _, imagined_raw = self.model._imagine_goal_conditioned(
            initial,
            horizon + 1,
            values["human_mask"],
            values["goal_position"],
            first_applied_action=first_applied_action,
            first_policy_action=first_policy_action,
            policy_action_override=policy_action_override,
            policy_action_override_steps=int(policy_action_override_steps),
            initial_ego_state=values["ego_state"],
            initial_previous_applied_action=values["previous_action"],
            initial_human_root=values.get("human_root"),
            initial_human_quality=values.get("human_quality"),
            initial_human_joint_clearance=values.get(
                "human_joint_clearance"),
            initial_human_joint_mask=values.get("human_joint_mask"),
            initial_human_joints_body=values.get("human_joints_body"),
            initial_human_joint_velocity_body=values.get(
                "human_joint_velocity_body"),
            initial_task_geometry=values.get("task_geometry"),
            initial_task_memory=values.get("task_memory"),
            task_progress_epsilon_m=(
                self.config.watchdog_progress_epsilon_m),
            task_stuck_max_horizontal_speed_mps=(
                self.config.watchdog_stuck_max_horizontal_speed_mps),
            task_acceleration_filter_alpha=(
                self.config.acceleration_filter_alpha),
            task_origin_xyz=values.get("task_origin_xyz"),
            task_origin_yaw=values.get("task_origin_yaw"),
            task_flight_bounds_xy=values.get("task_flight_bounds_xy"),
            task_static_obstacle_aabbs_xy=values.get(
                "task_static_obstacle_aabbs_xy"),
            task_static_obstacle_valid=values.get(
                "task_static_obstacle_valid"),
            task_bounds_valid=values.get("task_bounds_valid"),
            task_maximum_altitude_m=values.get(
                "task_maximum_altitude_m"),
            task_maximum_altitude_valid=values.get(
                "task_maximum_altitude_valid"),
            task_static_terminal_valid=values.get(
                "task_static_terminal_valid"),
            authoritative_analytic_ego=(
                self.config.authoritative_analytic_ego),
            sample=bool(sample),
            antithetic_policy_pairs=bool(antithetic_policy_pairs),
            collect_states=False,
        )
        imagined = dict(imagined_raw)
        for name in (
            "actor_imagination_start_count",
            "actor_imagination_sample_count",
            "actor_imagination_start_importance",
            "actor_imagination_start_population_count",
            "actor_imagination_actionable_population_count",
            "actor_imagination_imminent_population_count",
            "actor_imagination_preemptive_population_count",
            "actor_imagination_near_goal_ymax_misaligned_population_count",
            "actor_imagination_side_boundary_front_human_population_count",
            "actor_imagination_near_goal_population_count",
            "actor_imagination_near_boundary_population_count",
            "actor_imagination_ordinary_population_count",
            "actor_imagination_actionable_selected_count",
            "actor_imagination_imminent_selected_count",
            "actor_imagination_preemptive_selected_count",
            "actor_imagination_near_goal_ymax_misaligned_selected_count",
            "actor_imagination_side_boundary_front_human_selected_count",
            "actor_imagination_near_goal_selected_count",
            "actor_imagination_near_boundary_selected_count",
            "actor_imagination_ordinary_selected_count",
        ):
            if name in values:
                imagined[name] = values[name]
        goal_radius = values.get("goal_radius_m")
        if goal_radius is None:
            # Unit/mocked callers predate episode task metadata. Production
            # replay is required to match this fixed radius below.
            goal_radius = imagined["ego_state"].new_full(
                (imagined["ego_state"].shape[0], 1),
                self.config.goal_radius_m,
            )
        elif (
            not torch.isfinite(goal_radius).all()
            or not torch.allclose(
                goal_radius.float(),
                torch.full_like(
                    goal_radius.float(), self.config.goal_radius_m),
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "goal radius varies from the fixed Actor task contract")
        imagined["goal_radius_m"] = goal_radius.float()

        ego = imagined["ego_state"]
        batch_size, horizon = ego.shape[:2]
        start, end = ego[:, :-1], ego[:, 1:]
        goal = imagined["goal_position"][:, None, :].expand(
            -1, horizon - 1, -1)
        radius = goal_radius.float().reshape(batch_size, 1, 1, 1)
        goal_fraction = swept_relative_point_first_contact_fraction(
            (start[..., :3] - goal)[:, :, None, None, :],
            (end[..., :3] - goal)[:, :, None, None, :],
            torch.ones(
                batch_size, horizon - 1, 1, 1,
                dtype=torch.bool, device=ego.device),
            surface_radii_m=radius,
            per_human=False,
        )
        boundary_fraction = ego.new_full(
            (batch_size, horizon - 1), torch.inf)

        exact_bounds = all(key in values for key in (
            "task_origin_xyz", "task_origin_yaw", "task_flight_bounds_xy",
        ))
        if exact_bounds:
            origin = values["task_origin_xyz"].float().reshape(
                batch_size, 1, 3).expand(batch_size, horizon, 3)
            yaw = values["task_origin_yaw"].float().reshape(
                batch_size, 1).expand(batch_size, horizon)
            bounds = values["task_flight_bounds_xy"].float().reshape(
                batch_size, 1, 4).expand(batch_size, horizon, 4)
            signed = episode_to_world_flight_bounds_signed_clearance_torch(
                ego, origin, yaw, bounds)
            bounds_valid = values.get("task_bounds_valid")
            if bounds_valid is not None:
                validity = bounds_valid.bool().reshape(
                    batch_size, 1).expand(batch_size, horizon)
                signed = torch.where(
                    validity, signed, torch.full_like(signed, torch.inf))
            cosine = values["task_origin_yaw"].float().reshape(
                batch_size, 1).cos()
            sine = values["task_origin_yaw"].float().reshape(
                batch_size, 1).sin()
            world_xy = torch.stack((
                values["task_origin_xyz"][:, None, 0]
                + cosine * ego[..., 0] - sine * ego[..., 1],
                values["task_origin_xyz"][:, None, 1]
                + sine * ego[..., 0] + cosine * ego[..., 1],
            ), -1)
            raw_bounds = values["task_flight_bounds_xy"].float()
            allowed = torch.stack((
                raw_bounds[:, 0], raw_bounds[:, 2],
                raw_bounds[:, 1], raw_bounds[:, 3],
            ), -1)[:, None].expand(-1, horizon - 1, -1)
            xy_fraction = swept_segment_exit_aabb_fraction_2d(
                world_xy[:, :-1], world_xy[:, 1:], allowed)
            if bounds_valid is not None:
                row_valid = bounds_valid.bool().reshape(batch_size, 1)
                xy_fraction = torch.where(
                    row_valid, xy_fraction,
                    torch.full_like(xy_fraction, torch.inf))
            boundary_fraction = torch.minimum(
                boundary_fraction, xy_fraction)

            maximum_altitude = values.get("task_maximum_altitude_m")
            altitude_valid = values.get("task_maximum_altitude_valid")
            if maximum_altitude is not None and altitude_valid is not None:
                world_z = origin[..., 2] + ego[..., 2]
                maximum = maximum_altitude.float().reshape(
                    batch_size, 1).expand(batch_size, horizon)
                altitude_fraction = swept_scalar_exit_interval_fraction(
                    world_z[:, :-1], world_z[:, 1:],
                    torch.full_like(maximum[:, :-1], -torch.inf),
                    maximum[:, :-1],
                )
                altitude_row_valid = altitude_valid.bool().reshape(
                    batch_size, 1)
                altitude_fraction = torch.where(
                    altitude_row_valid,
                    altitude_fraction,
                    torch.full_like(altitude_fraction, torch.inf),
                )
                boundary_fraction = torch.minimum(
                    boundary_fraction, altitude_fraction)
                altitude_signed = maximum - world_z
                altitude_signed = torch.where(
                    altitude_row_valid,
                    altitude_signed,
                    torch.full_like(altitude_signed, torch.inf),
                )
                signed = torch.minimum(signed, altitude_signed)
            imagined["flight_bounds_signed_clearance_m"] = signed[..., None]

            if all(key in values for key in (
                "task_static_obstacle_aabbs_xy",
                "task_static_obstacle_valid",
            )):
                obstacles = values[
                    "task_static_obstacle_aabbs_xy"].float()
                obstacle_valid = values[
                    "task_static_obstacle_valid"].bool()
                static_entry_fraction = torch.full_like(
                    boundary_fraction, torch.inf)
                if obstacles.shape[-2] > 0:
                    static_entry_fraction = (
                        swept_segment_aabb_first_contact_fraction_2d(
                            world_xy[:, :-1],
                            world_xy[:, 1:],
                            obstacles[:, None].expand(
                                batch_size,
                                horizon - 1,
                                *obstacles.shape[1:],
                            ),
                            obstacle_valid[:, None].expand(
                                batch_size,
                                horizon - 1,
                                *obstacle_valid.shape[1:],
                            ),
                        )
                    )
                static_terminal_valid = values.get(
                    "task_static_terminal_valid")
                if static_terminal_valid is None:
                    # Old unit callers have no authority to upgrade an
                    # explicitly inexact proximity proxy into a task event.
                    static_terminal_row = torch.zeros(
                        batch_size, 1, dtype=torch.bool, device=ego.device)
                else:
                    static_terminal_row = static_terminal_valid.bool().reshape(
                        batch_size, 1)
                effective_static_entry = torch.where(
                    static_terminal_row,
                    static_entry_fraction,
                    torch.full_like(static_entry_fraction, torch.inf),
                )
                boundary_fraction = torch.minimum(
                    boundary_fraction, effective_static_entry)
                imagined["static_safety_envelope_fraction"] = (
                    effective_static_entry[..., None])
                imagined["static_terminal_valid"] = (
                    static_terminal_row[:, None]
                    .expand(batch_size, horizon, 1))
                static_clearance = episode_to_world_static_clearance_torch(
                    ego,
                    values["task_origin_xyz"].float()[:, None, :],
                    values["task_origin_yaw"].float().reshape(
                        batch_size, 1).expand(batch_size, horizon),
                    obstacles[:, None].expand(
                        batch_size, horizon, *obstacles.shape[1:]),
                    obstacle_valid[:, None].expand(
                        batch_size, horizon, *obstacle_valid.shape[1:]),
                )
                imagined["static_proxy_clearance_m"] = (
                    static_clearance[..., None])
        # All task terminals use a continuous first-occurrence fraction.  The
        # simulator checks goal first, then boundary, crash and stuck at every
        # physics frame; endpoint booleans would incorrectly turn an earlier
        # crash/stuck into a later success within the same 10 Hz transition.
        crash_fraction = torch.full_like(goal_fraction, torch.inf)
        stuck_fraction = torch.full_like(goal_fraction, torch.inf)
        memory = imagined.get("task_memory")
        if memory is not None:
            if (
                memory.shape[:2] != ego.shape[:2]
                or memory.shape[-1] != TASK_MEMORY_DIM
            ):
                raise RuntimeError("imagined task-memory trajectory is invalid")
            elapsed_start = memory[:, :-1, 0]
            elapsed_end = memory[:, 1:, 0]
            minimum_elapsed = values.get("task_watchdog_min_elapsed_s")
            if minimum_elapsed is None:
                minimum_elapsed = elapsed_end.new_full(
                    (batch_size, 1), self.config.watchdog_min_elapsed_s)
            minimum_elapsed = minimum_elapsed.float().reshape(batch_size, 1)
            eligibility_fraction = swept_scalar_enter_lower_bound_fraction(
                elapsed_start, elapsed_end, minimum_elapsed)
            crash_altitude = values.get("task_crash_altitude_m")
            origin_xyz = values.get("task_origin_xyz")
            if crash_altitude is not None and origin_xyz is not None:
                crash_minimum_elapsed = values.get(
                    "task_crash_min_elapsed_s")
                if crash_minimum_elapsed is None:
                    crash_minimum_elapsed = elapsed_end.new_full(
                        (batch_size, 1), self.config.crash_min_elapsed_s)
                crash_eligibility_fraction = (
                    swept_scalar_enter_lower_bound_fraction(
                        elapsed_start,
                        elapsed_end,
                        crash_minimum_elapsed.float().reshape(batch_size, 1),
                    )
                )
                world_z = (
                    origin_xyz.float().reshape(batch_size, 1, 3)[..., 2]
                    + ego[..., 2])
                altitude_threshold = crash_altitude.float().reshape(
                    batch_size, 1)
                altitude_entry = swept_scalar_enter_upper_bound_fraction(
                    world_z[:, :-1], world_z[:, 1:], altitude_threshold)
                altitude_exit = swept_scalar_exit_interval_fraction(
                    world_z[:, :-1], world_z[:, 1:],
                    torch.full_like(altitude_threshold, -torch.inf),
                    altitude_threshold,
                )
                crash_candidate = torch.maximum(
                    crash_eligibility_fraction, altitude_entry)
                crash_valid = (
                    torch.isfinite(crash_candidate)
                    & (crash_candidate <= altitude_exit)
                )
                crash_fraction = crash_candidate.masked_fill(
                    ~crash_valid, torch.inf)
            timeout = values.get("task_watchdog_no_progress_timeout_s")
            if timeout is None:
                timeout = elapsed_end.new_full(
                    (batch_size, 1),
                    self.config.watchdog_no_progress_timeout_s)
            threshold = timeout.float().reshape(batch_size, 1)
            dt = elapsed_end - elapsed_start
            stalled_entry = swept_scalar_enter_lower_bound_fraction(
                memory[:, :-1, 2], memory[:, :-1, 2] + dt, threshold)
            low_motion_entry = swept_scalar_enter_lower_bound_fraction(
                memory[:, :-1, 3], memory[:, :-1, 3] + dt, threshold)
            stuck_candidate = torch.maximum(
                eligibility_fraction,
                torch.maximum(stalled_entry, low_motion_entry),
            )

            # The watchdog updates/reset both timers before testing them.  A
            # newly sufficient goal improvement therefore wins an exact tie;
            # the low-motion reset is strict (> speed threshold), so equality
            # still permits stuck at that instant.
            best_distance = memory[:, :-1, 1]
            progress_radius = (
                best_distance
                - float(self.config.watchdog_progress_epsilon_m)
            ).clamp_min(0.0)
            relative_goal_start = start[..., :3] - goal
            relative_goal_end = end[..., :3] - goal
            progress_possible = progress_radius > 0.0
            progress_reset_fraction = (
                swept_relative_point_first_contact_fraction(
                    relative_goal_start[:, :, None, None, :],
                    relative_goal_end[:, :, None, None, :],
                    progress_possible[:, :, None, None],
                    surface_radii_m=progress_radius[:, :, None, None],
                    per_human=False,
                )
            )
            speed_threshold = values.get(
                "task_watchdog_stuck_max_horizontal_speed_mps")
            if speed_threshold is None:
                speed_threshold = ego.new_full(
                    (batch_size, 1),
                    self.config.watchdog_stuck_max_horizontal_speed_mps,
                )
            motion_reset_fraction = swept_vector_exit_ball_fraction(
                start[..., 3:5], end[..., 3:5],
                speed_threshold.float().reshape(batch_size, 1),
            )
            stuck_valid = (
                torch.isfinite(stuck_candidate)
                & (stuck_candidate <= 1.0)
                & (stuck_candidate < progress_reset_fraction)
                & (stuck_candidate <= motion_reset_fraction)
            )
            stuck_fraction = stuck_candidate.masked_fill(
                ~stuck_valid, torch.inf)

        analytic_fraction = torch.stack((
            goal_fraction, boundary_fraction, crash_fraction, stuck_fraction,
        ), -1)
        imagined["analytic_terminal_fraction"] = analytic_fraction
        imagined["analytic_terminal"] = first_analytic_task_event(
            analytic_fraction).to(ego.dtype)
        first_fraction = analytic_fraction.amin(-1, keepdim=True)
        imagined["analytic_terminal_first_fraction"] = torch.where(
            torch.isfinite(first_fraction),
            first_fraction.clamp(0.0, 1.0),
            torch.ones_like(first_fraction),
        )
        return imagined

    @staticmethod
    def _body_vectors_to_episode_frame(
        vectors: torch.Tensor,
        ego_state: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate body-FLU vectors into the fixed episode frame."""
        if vectors.ndim < 4 or vectors.shape[-1] != 3:
            raise ValueError("body vectors must be [...,time,entities,3]")
        if ego_state.shape[:-1] != vectors.shape[:2] or ego_state.shape[-1] < 14:
            raise ValueError("Ego14 rows must match body-vector batch/time")
        sine = ego_state[..., 12]
        cosine = ego_state[..., 13]
        while sine.ndim < vectors.ndim - 1:
            sine = sine.unsqueeze(-1)
            cosine = cosine.unsqueeze(-1)
        x = cosine * vectors[..., 0] - sine * vectors[..., 1]
        y = sine * vectors[..., 0] + cosine * vectors[..., 1]
        return torch.stack((x, y, vectors[..., 2]), -1)

    def _imagined_swept_human_contact(
        self,
        imagined: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Return exact contact plus per-person swept gaps and validity.

        Articulated endpoints come from the recursively imagined learned Human
        dynamics. The known UAV/joint sphere radii and PhysX contact offset
        then determine contact in a common episode frame. The constant-
        velocity audit branch deliberately does not acquire terminal veto
        power here; it remains a conservative dense-clearance fallback.  Exact
        simulator ordering uses the earliest contact across people, while the
        learned-geometry uncertainty path retains each person's gap so its
        probability can be composed as a differentiable union.
        """
        required = {
            "ego_state", "human_joints_body", "human_joint_velocity_body",
            "human_joint_mask", "human_mask_rollout",
        }
        missing = sorted(required.difference(imagined))
        if missing:
            raise KeyError(
                "swept Human contact lacks imagined fields " + str(missing))
        ego = imagined["ego_state"]
        joints = imagined["human_joints_body"]
        velocity = imagined["human_joint_velocity_body"]
        if joints.shape != velocity.shape or joints.shape[-1] != 3:
            raise ValueError("imagined Human joint kinematics are inconsistent")
        if joints.shape[:2] != ego.shape[:2]:
            raise ValueError("imagined Human/Ego horizons differ")
        joint_mask = imagined["human_joint_mask"].bool()
        if joint_mask.ndim == 3:
            joint_mask = joint_mask[:, None].expand(
                -1, joints.shape[1], -1, -1)
        if joint_mask.shape != joints.shape[:-1]:
            raise ValueError("imagined Human joint mask has invalid shape")
        human_mask = imagined["human_mask_rollout"].bool()
        if human_mask.shape != joints.shape[:-2]:
            raise ValueError("imagined Human rollout mask has invalid shape")
        spheres, _, sphere_mask, joint_radii = (
            self.model.deployable_human_collision_geometry(
                joints, velocity, joint_mask))
        sphere_mask = sphere_mask.bool() & human_mask[..., None]
        relative_episode = self._body_vectors_to_episode_frame(spheres, ego)
        transition_valid = sphere_mask[:, :-1] & sphere_mask[:, 1:]
        contact_radii = (
            joint_radii.to(relative_episode)
            + float(self.config.drone_collision_radius_m)
            + float(self.config.human_collision_contact_offset_m)
        )
        transition_count = relative_episode.shape[1] - 1
        stop_fraction = relative_episode.new_ones(
            relative_episode.shape[0], transition_count)
        active = torch.ones(
            relative_episode.shape[0], transition_count,
            dtype=torch.bool, device=relative_episode.device,
        )
        if "analytic_terminal" in imagined:
            terminal = imagined["analytic_terminal"].bool().any(-1)
            first_fraction = imagined.get(
                "analytic_terminal_first_fraction")
            if terminal.shape != stop_fraction.shape or first_fraction is None:
                raise ValueError(
                    "imagined analytic terminal horizon is inconsistent")
            if first_fraction.shape != (*terminal.shape, 1):
                raise ValueError(
                    "imagined analytic terminal fraction is inconsistent")
            stop_fraction = torch.where(
                terminal,
                first_fraction[..., 0].float().clamp(0.0, 1.0),
                stop_fraction,
            )
            terminal_before = terminal.long().cumsum(1) - terminal.long()
            active = terminal_before.eq(0)
        stopped_relative_episode = (
            relative_episode[:, :-1]
            + stop_fraction[:, :, None, None, None]
            * (relative_episode[:, 1:] - relative_episode[:, :-1])
        )
        transition_valid &= active[:, :, None, None]
        per_human_first_fraction = (
            swept_relative_point_first_contact_fraction(
            relative_episode[:, :-1], stopped_relative_episode,
            transition_valid, surface_radii_m=contact_radii,
            per_human=True,
        ))
        finite_contact_fraction = torch.isfinite(per_human_first_fraction)
        # Never feed ``inf`` into a multiplication and mask it afterwards:
        # torch.where correctly masks the forward value, but its backward can
        # still encounter ``0 * inf`` in MulBackward and poison Actor
        # gradients on ordinary non-contact Human slots.
        contact_fraction_for_scale = torch.where(
            finite_contact_fraction,
            per_human_first_fraction,
            torch.zeros_like(per_human_first_fraction),
        )
        scaled_contact_fraction = (
            contact_fraction_for_scale * stop_fraction[:, :, None])
        per_human_first_fraction = torch.where(
            finite_contact_fraction,
            scaled_contact_fraction,
            torch.full_like(scaled_contact_fraction, torch.inf),
        )
        per_human_swept_gap = swept_relative_point_signed_gap(
            relative_episode[:, :-1], stopped_relative_episode,
            transition_valid, surface_radii_m=contact_radii,
            fallback_gap_m=6.0, per_human=True,
        )
        per_human_valid = transition_valid.any(-1)
        first_fraction = per_human_first_fraction.amin(-1)
        contact = torch.isfinite(first_fraction)
        bounded_fraction = torch.where(
            contact, first_fraction.clamp(0.0, 1.0),
            torch.ones_like(first_fraction))
        return (
            contact.to(ego)[..., None],
            bounded_fraction[..., None],
            per_human_swept_gap,
            per_human_valid,
        )

    def _imagined_objective(
        self, imagined: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        feature = imagined["joint_feat"]
        decision_feature = imagined["decision_feat"]
        current = feature[:, :-1]
        next_feature = feature[:, 1:]
        action = imagined["action"][:, :-1]
        # Scalar/component heads remain factual diagnostics. Actor reward uses
        # only structured quantities carried by the imagined Markov state;
        # otherwise an unconstrained reward decoder can invent action
        # gradients unrelated to the recorder formula.
        scalar_reward = _mode(self.model.reward(next_feature))
        component_values_live = self.model.policy_reward_components(next_feature)
        learned_dense_reward = component_values_live[..., :1] * 0.0
        agl = imagined["ego_state"][:, 1:, 9:10]
        height_excess = F.relu(
            (agl - float(self.config.cruise_height_m)).abs()
            - float(self.config.height_tolerance_m))
        height_term = (
            height_excess / float(self.config.height_scale_m)
        ).square().clamp(max=4.0)
        analytic_height_reward = (
            -float(self.config.height_weight_per_sec)
            * float(self.config.imagination_dt_s)
            * height_term)
        analytic_time_reward = torch.full_like(
            learned_dense_reward,
            -float(self.config.time_cost_per_sec)
            * float(self.config.imagination_dt_s),
        )
        task_memory = imagined.get("task_memory")
        if task_memory is None:
            raise RuntimeError(
                "full-state objective lacks imagined task/reward memory")
        analytic_smoothness, smoothness_state = analytic_smoothness_reward(
            task_memory[:, :-1], task_memory[:, 1:],
            imagined["ego_state"][:, :-1], imagined["ego_state"][:, 1:],
            dt_s=self.config.imagination_dt_s,
            acceleration_weight_per_sec=(
                self.config.acceleration_weight_per_sec),
            acceleration_scale_mps2=self.config.acceleration_scale_mps2,
            jerk_weight_per_sec=self.config.jerk_weight_per_sec,
            jerk_scale_mps3=self.config.jerk_scale_mps3,
            yaw_rate_weight_per_sec=self.config.yaw_rate_weight_per_sec,
            yaw_rate_scale_rps=self.config.yaw_rate_scale_rps,
            term_clip=self.config.smooth_term_clip,
        )
        progress = analytic_progress_reward(
            imagined["ego_state"],
            imagined["goal_position"],
            dt_s=self.config.imagination_dt_s,
            progress_weight_per_m=self.config.progress_weight_per_m,
            max_progress_speed_mps=self.config.max_progress_speed_mps,
        )[:, 1:]
        _, cross_track = cross_track_potential_reward(
            imagined["ego_state"],
            imagined["goal_position"],
            discount=self.config.discount,
            tolerance_m=self.config.route_cross_track_tolerance_m,
            potential_scale=self.config.route_potential_scale,
        )
        route_potential = (
            -float(self.config.route_potential_scale)
            * F.relu(
                cross_track - float(
                    self.config.route_cross_track_tolerance_m)))
        route_deviation_full, _ = analytic_route_deviation_reward(
            imagined["ego_state"][:, 1:],
            imagined["goal_position"][:, None, :],
            tolerance_m=self.config.route_cross_track_tolerance_m,
            weight_per_m_per_sec=(
                self.config.route_deviation_weight_per_m_per_sec),
            dt_s=self.config.imagination_dt_s,
        )
        heading_potential, heading_error = route_heading_potential(
            imagined["ego_state"],
            imagined["goal_position"],
            potential_scale=self.config.route_heading_potential_scale,
        )
        component_values = component_values_live.detach()
        component_metrics = {}
        for index, name in enumerate(REWARD_COMPONENT_KEYS):
            values = component_values[..., index]
            component_metrics[f"reward_component_{name}_mean"] = values.mean()
            component_metrics[
                f"reward_component_{name}_abs_max"] = values.abs().amax()
        learned_continuation = self.model.cont(
            next_feature).mean.clamp(0.0, 1.0)
        explicit_human_context = all(key in imagined for key in (
            "ego_branch_feature", "human_branch_feature", "human_root",
            "human_observation_quality", "human_presence",
            "human_joint_clearance", "human_mask_rollout",
            "human_joints_body", "human_joint_velocity_body",
        ))
        if not explicit_human_context:
            raise RuntimeError(
                "full-state imagined Event context was not retained")
        joint_mask = imagined["human_joint_mask"]
        if joint_mask.ndim == 3:
            joint_mask = joint_mask[:, None].expand(
                -1, current.shape[1], -1, -1)
        else:
            joint_mask = joint_mask[:, :-1]
        event = self.model.transition_event.forward_non_goal(
            current,
            action,
            ego_feature=imagined["ego_branch_feature"][:, :-1],
            human_feature=imagined["human_branch_feature"][:, :-1],
            human_root=imagined["human_root"][:, :-1],
            human_quality=imagined[
                "human_observation_quality"][:, :-1],
            human_mask=imagined["human_mask_rollout"][:, :-1],
            human_joint_clearance=imagined[
                "human_joint_clearance"][:, :-1],
            human_joints_body=imagined["human_joints_body"][:, :-1],
            human_joint_velocity_body=imagined[
                "human_joint_velocity_body"][:, :-1],
            human_joint_mask=joint_mask,
            human_presence=imagined["human_presence"][:, :-1],
            detach_parameters=True,
        )
        probability = event["probability"]
        learned_human_event_probability = event[
            "human_collision_probability"]
        learned_static_event_diagnostic = probability[..., 2:3]
        goal = imagined["goal_position"][:, None, :].expand(
            -1, imagined["ego_state"].shape[1] - 1, -1)
        next_goal_distance = point_goal_distance(
            imagined["ego_state"][:, 1:], goal)
        boundary_signed_clearance_all = imagined.get(
            "flight_bounds_signed_clearance_m")
        analytic_bounds_available = boundary_signed_clearance_all is not None
        # A plane clearance changes at no more than the audited horizontal
        # speed norm. Forward and lateral commands can be simultaneous, so
        # use the full horizontal-speed support over the complete rollout.
        boundary_lookahead_margin_m = (
            float(self.config.task_maximum_horizontal_speed_mps)
            * float(self.config.imagination_dt_s)
            * int(current.shape[1]))
        if boundary_signed_clearance_all is None:
            boundary_signed_clearance_all = torch.full_like(
                imagined["ego_state"][..., :1],
                boundary_lookahead_margin_m,
            )
            boundary_margin_excess = torch.zeros_like(
                boundary_signed_clearance_all)
            boundary_signed_clearance = next_goal_distance.new_full(
                next_goal_distance.shape, torch.inf)
        else:
            boundary_margin_excess = F.relu(
                boundary_lookahead_margin_m
                - boundary_signed_clearance_all.float())
            boundary_signed_clearance = boundary_signed_clearance_all[:, 1:]
        static_clearance_all = imagined.get("static_proxy_clearance_m")
        if static_clearance_all is None:
            static_potential = torch.zeros_like(
                boundary_signed_clearance_all)
            static_margin_excess = torch.zeros_like(
                boundary_signed_clearance_all)
        else:
            exact_margin = static_clearance_all.new_full(
                static_clearance_all.shape,
                float(self.config.task_maximum_horizontal_speed_mps)
                * float(self.config.imagination_dt_s)
                * int(current.shape[1]),
            )
            proxy_margin = static_clearance_all.new_full(
                static_clearance_all.shape,
                float(self.config.static_proxy_clearance_margin_m),
            )
            static_terminal_valid = imagined.get("static_terminal_valid")
            static_margin = (
                proxy_margin
                if static_terminal_valid is None else
                torch.where(
                    static_terminal_valid.bool(), exact_margin, proxy_margin)
            )
            static_margin_excess = F.relu(
                static_margin - static_clearance_all.float())
            static_potential = (
                -float(self.config.static_proxy_potential_scale)
                * static_margin_excess)
        analytic_event_rewards = probability.new_tensor((
            self.model.task_goal_event_reward,
            self.model.task_other_terminal_reward,
            self.model.task_other_terminal_reward,
            self.model.task_stuck_terminal_reward,
        ))
        (
            swept_human_terminal,
            swept_human_first_fraction,
            swept_human_contact_gap_per_person,
            swept_human_contact_valid_per_person,
        ) = self._imagined_swept_human_contact(imagined)
        rollout_depth = torch.arange(
            swept_human_contact_gap_per_person.shape[1],
            device=swept_human_contact_gap_per_person.device,
            dtype=swept_human_contact_gap_per_person.dtype,
        ).view(1, -1, 1)
        collision_prediction_bias = (
            float(self.config.human_collision_prediction_bias_m)
            + float(
                self.config.
                human_collision_prediction_bias_growth_m_per_step)
            * rollout_depth
        )
        uncertain_human_slot_probability = swept_human_contact_probability(
            swept_human_contact_gap_per_person,
            prediction_bias_m=collision_prediction_bias,
            prediction_scale_m=(
                self.config.human_collision_prediction_scale_m),
        )
        # The gap residual is one trajectory-level model error, not a fresh
        # independent draw on every 100 ms transition.  Convert its cumulative
        # per-person CDF into conditional hazards so a persistent near miss is
        # represented once rather than repeatedly charged across H15.
        uncertain_human_probability = (
            trajectory_correlated_human_contact_hazard(
                uncertain_human_slot_probability,
                swept_human_contact_valid_per_person,
            ))
        swept_human_contact_gap = (
            swept_human_contact_gap_per_person.masked_fill(
                ~swept_human_contact_valid_per_person, torch.inf
            ).amin(-1, keepdim=True)
        )
        swept_human_contact_gap = torch.where(
            swept_human_contact_valid_per_person.any(-1, keepdim=True),
            swept_human_contact_gap,
            torch.full_like(swept_human_contact_gap, 6.0),
        )
        event_reward, event_continuation, analytic_event = (
            compose_uncertain_swept_human_analytic_events(
                swept_human_terminal,
                swept_human_first_fraction,
                uncertain_human_probability,
                imagined["analytic_terminal"],
                imagined["analytic_terminal_first_fraction"],
                human_collision_reward=(
                    self.model.task_human_collision_reward),
                analytic_event_rewards=analytic_event_rewards,
            ))
        # The simulator force-writes a terminal row at the physical event
        # frame, often before the next 10 Hz boundary. Both Human contact and
        # analytic task events now carry exact swept fractions, so partial
        # dense reward is evaluated at the event that actually wins.
        terminal_fraction = imagined[
            "analytic_terminal_first_fraction"].to(event_reward)
        analytic_probability = analytic_event["analytic"]
        analytic_terminal_probability = analytic_probability.sum(
            -1, keepdim=True)
        expected_step_fraction = analytic_event[
            "expected_active_fraction"]
        partial_progress = analytic_fractional_progress_reward(
            imagined["ego_state"][:, :-1],
            imagined["ego_state"][:, 1:],
            imagined["goal_position"],
            terminal_fraction,
            dt_s=self.config.imagination_dt_s,
            progress_weight_per_m=self.config.progress_weight_per_m,
            max_progress_speed_mps=self.config.max_progress_speed_mps,
        )
        human_mean_fraction = analytic_event[
            "human_mean_event_fraction"].to(event_reward)
        human_event_probability = analytic_event["human"]
        human_event_fractions = human_mean_fraction.unsqueeze(-2)
        human_event_mass_weights = human_event_probability.unsqueeze(-2)
        quadrature_count = int(human_event_fractions.shape[-2])
        human_event_source_ego = imagined[
            "ego_state"][:, :-1].unsqueeze(-2).expand(
                -1, -1, quadrature_count, -1)
        human_event_destination_ego = imagined[
            "ego_state"][:, 1:].unsqueeze(-2).expand_as(
                human_event_source_ego)
        human_event_goal = imagined["goal_position"][:, None, None, :].expand(
            -1, current.shape[1], quadrature_count, -1)
        human_partial_progress_nodes = analytic_fractional_progress_reward(
            human_event_source_ego,
            human_event_destination_ego,
            human_event_goal,
            human_event_fractions,
            dt_s=self.config.imagination_dt_s,
            progress_weight_per_m=self.config.progress_weight_per_m,
            max_progress_speed_mps=self.config.max_progress_speed_mps,
        )
        human_partial_progress = (
            human_event_mass_weights * human_partial_progress_nodes
        ).sum(-2)
        progress = (
            event_continuation * progress
            + human_partial_progress
            + analytic_terminal_probability * partial_progress
        )
        partial_event_ego = interpolate_ego_state(
            imagined["ego_state"][:, :-1],
            imagined["ego_state"][:, 1:],
            terminal_fraction,
        )
        partial_smoothness, partial_smoothness_state = (
            analytic_fractional_smoothness_reward(
                task_memory[:, :-1],
                imagined["ego_state"][:, :-1],
                partial_event_ego,
                terminal_fraction,
                dt_s=self.config.imagination_dt_s,
                acceleration_filter_alpha=(
                    self.config.acceleration_filter_alpha),
                acceleration_weight_per_sec=(
                    self.config.acceleration_weight_per_sec),
                acceleration_scale_mps2=(
                    self.config.acceleration_scale_mps2),
                jerk_weight_per_sec=self.config.jerk_weight_per_sec,
                jerk_scale_mps3=self.config.jerk_scale_mps3,
                yaw_rate_weight_per_sec=(
                    self.config.yaw_rate_weight_per_sec),
                yaw_rate_scale_rps=self.config.yaw_rate_scale_rps,
                term_clip=self.config.smooth_term_clip,
            ))
        human_event_ego = interpolate_ego_state(
            human_event_source_ego,
            human_event_destination_ego,
            human_event_fractions,
        )
        partial_route_deviation, _ = analytic_route_deviation_reward(
            partial_event_ego,
            imagined["goal_position"][:, None, :],
            tolerance_m=self.config.route_cross_track_tolerance_m,
            weight_per_m_per_sec=(
                self.config.route_deviation_weight_per_m_per_sec),
            dt_s=(
                float(self.config.imagination_dt_s) * terminal_fraction),
        )
        human_route_deviation_nodes, _ = analytic_route_deviation_reward(
            human_event_ego,
            human_event_goal,
            tolerance_m=self.config.route_cross_track_tolerance_m,
            weight_per_m_per_sec=(
                self.config.route_deviation_weight_per_m_per_sec),
            dt_s=(
                float(self.config.imagination_dt_s)
                * human_event_fractions),
        )
        human_partial_route_deviation = (
            human_event_mass_weights * human_route_deviation_nodes
        ).sum(-2)
        route_deviation_reward = (
            event_continuation * route_deviation_full
            + human_partial_route_deviation
            + analytic_terminal_probability * partial_route_deviation
        )
        human_event_task_memory = task_memory[:, :-1].unsqueeze(-2).expand(
            -1, -1, quadrature_count, -1)
        human_partial_smoothness_nodes, human_partial_smoothness_state_nodes = (
            analytic_fractional_smoothness_reward(
                human_event_task_memory,
                human_event_source_ego,
                human_event_ego,
                human_event_fractions,
                dt_s=self.config.imagination_dt_s,
                acceleration_filter_alpha=(
                    self.config.acceleration_filter_alpha),
                acceleration_weight_per_sec=(
                    self.config.acceleration_weight_per_sec),
                acceleration_scale_mps2=(
                    self.config.acceleration_scale_mps2),
                jerk_weight_per_sec=self.config.jerk_weight_per_sec,
                jerk_scale_mps3=self.config.jerk_scale_mps3,
                yaw_rate_weight_per_sec=(
                    self.config.yaw_rate_weight_per_sec),
                yaw_rate_scale_rps=self.config.yaw_rate_scale_rps,
                term_clip=self.config.smooth_term_clip,
            ))
        human_partial_smoothness = (
            human_event_mass_weights * human_partial_smoothness_nodes
        ).sum(-2)
        human_event_conditional_weights = (
            human_event_mass_weights
            / human_event_probability.unsqueeze(-2).clamp_min(1.0e-12)
        )
        human_partial_smoothness_state = {
            name: (human_event_conditional_weights * value).sum(-2)
            for name, value in human_partial_smoothness_state_nodes.items()
        }
        analytic_smoothness = (
            event_continuation * analytic_smoothness
            + human_partial_smoothness
            + analytic_terminal_probability * partial_smoothness)
        partial_agl = (
            imagined["ego_state"][:, :-1, 9:10]
            + terminal_fraction * (
                imagined["ego_state"][:, 1:, 9:10]
                - imagined["ego_state"][:, :-1, 9:10])
        )
        partial_height_excess = F.relu(
            (partial_agl - float(self.config.cruise_height_m)).abs()
            - float(self.config.height_tolerance_m))
        partial_height_reward = (
            -float(self.config.height_weight_per_sec)
            * float(self.config.imagination_dt_s)
            * terminal_fraction
            * (partial_height_excess / float(
                self.config.height_scale_m)).square().clamp(max=4.0)
        )
        human_partial_agl = human_event_ego[..., 9:10]
        human_partial_height_excess = F.relu(
            (human_partial_agl - float(self.config.cruise_height_m)).abs()
            - float(self.config.height_tolerance_m))
        human_partial_height_reward_nodes = (
            -float(self.config.height_weight_per_sec)
            * float(self.config.imagination_dt_s)
            * human_event_fractions
            * (human_partial_height_excess / float(
                self.config.height_scale_m)).square().clamp(max=4.0)
        )
        human_partial_height_reward = (
            human_event_mass_weights * human_partial_height_reward_nodes
        ).sum(-2)
        analytic_height_reward = (
            event_continuation * analytic_height_reward
            + human_partial_height_reward
            + analytic_terminal_probability * partial_height_reward
        )
        analytic_time_reward = analytic_time_reward * expected_step_fraction
        # Reward and survival must come from the same closed competing-risk
        # distribution. Using the independent continuation head here charged
        # the Human terminal reward repeatedly on later imagined steps and
        # produced candidate return spreads larger than the terminal reward
        # itself. The generic continuation head remains trained and logged as
        # a world-model diagnostic, but it cannot double-count these terminals.
        continuation = event_continuation
        # All shaping potentials use the same absorbing-terminal survival as
        # the lambda return. This preserves policy invariance at a terminal
        # instead of charging Phi(terminal) as though another state followed.
        route_reward = terminal_aware_potential_reward(
            route_potential, continuation, discount=self.config.discount)
        heading_potential_reward = terminal_aware_potential_reward(
            heading_potential, continuation, discount=self.config.discount)
        boundary_proximity_reward, boundary_margin_excess = (
            analytic_boundary_proximity_reward(
                boundary_signed_clearance_all,
                continuation,
                lookahead_margin_m=boundary_lookahead_margin_m,
                weight_per_second=(
                    self.config.boundary_proximity_weight_per_sec),
                dt_s=self.config.imagination_dt_s,
            ))
        static_potential_reward = terminal_aware_potential_reward(
            static_potential, continuation, discount=self.config.discount)
        human_risk = analytic_event["human"]
        observed_human_risk = event[
            "observed_human_collision_probability"]
        unobserved_human_risk = event[
            "unobserved_human_collision_probability"]
        static_risk = torch.zeros_like(human_risk)
        goal_probability = analytic_event["goal"]
        boundary_probability = analytic_event["boundary"]
        task_memory_terminal_probability = (
            analytic_event["crash"] + analytic_event["stuck"])
        other_terminal_probability = (
            task_memory_terminal_probability + boundary_probability)

        human_risk_state: dict[str, torch.Tensor] = {}
        if "human_joint_clearance" in imagined:
            clearance = imagined["human_joint_clearance"][:, 1:]
            next_joints = imagined["human_joints_body"][:, 1:]
            next_joint_velocity = imagined[
                "human_joint_velocity_body"][:, 1:]
            rollout_joint_mask = imagined["human_joint_mask"]
            if rollout_joint_mask.ndim == 3:
                destination_rollout_joint_mask = rollout_joint_mask[:, None].expand(
                    -1, next_joints.shape[1], -1, -1)
                source_rollout_joint_mask = destination_rollout_joint_mask
            else:
                destination_rollout_joint_mask = rollout_joint_mask[:, 1:]
                source_rollout_joint_mask = rollout_joint_mask[:, :-1]
            (
                collision_spheres,
                collision_sphere_velocity,
                collision_sphere_mask,
                _,
            ) = (
                self.model.deployable_human_collision_geometry(
                    next_joints, next_joint_velocity,
                    destination_rollout_joint_mask.bool()))
            clearance_valid = (
                imagined["human_mask_rollout"][:, 1:].bool()
                & collision_sphere_mask.any(-1)
            )
            clearance = clearance.masked_fill(~clearance_valid, torch.inf)
            minimum_human_clearance = clearance.amin(-1, keepdim=True)
            minimum_human_clearance = torch.where(
                clearance_valid.any(-1, keepdim=True),
                minimum_human_clearance,
                torch.full_like(minimum_human_clearance, 6.0),
            )
            human_clearance_reward, human_risk_state = analytic_human_risk_reward(
                imagined["human_joint_clearance"][:, 1:],
                collision_spheres[..., 0, :],
                collision_sphere_velocity[..., 0, :],
                clearance_valid,
                pelvis_valid=collision_sphere_mask[..., 0],
                human_presence=imagined["human_presence"][:, 1:],
                ego_velocity_body=ego_velocity_full_body(
                    imagined["ego_state"][:, 1:]),
                hard_clearance_m=self.config.human_hard_clearance_m,
                safe_clearance_m=self.config.human_safe_clearance_m,
                predictive_safe_clearance_m=(
                    self.config.human_predictive_safe_clearance_m),
                predictive_horizon_s=self.config.human_ttc_horizon_s,
                corridor_half_width_m=(
                    self.config.human_corridor_half_width_m),
                corridor_lookahead_m=(
                    self.config.human_corridor_lookahead_m),
                drone_radius_m=self.config.drone_collision_radius_m,
                pelvis_radius_m=self.config.pelvis_collision_radius_m,
                weight_per_second=self.config.human_clearance_weight_per_sec,
                dt_s=self.config.imagination_dt_s,
            )
            # Analytic task events may occur before the end of a 0.1 s policy
            # transition.  The recorder evaluates Human risk at that forced
            # event frame, not at a fictitious complete-step endpoint.  Put
            # source/destination spheres in one episode frame, interpolate to
            # the exact first-event fraction, and mix that partial reward only
            # into the analytic-terminal probability mass.
            source_joints = imagined["human_joints_body"][:, :-1]
            source_joint_velocity = imagined[
                "human_joint_velocity_body"][:, :-1]
            (
                source_collision_spheres,
                source_collision_velocity,
                source_collision_mask,
                collision_radii,
            ) = self.model.deployable_human_collision_geometry(
                source_joints, source_joint_velocity,
                source_rollout_joint_mask.bool())
            event_ego_state = interpolate_ego_state(
                imagined["ego_state"][:, :-1],
                imagined["ego_state"][:, 1:],
                terminal_fraction,
            )
            event_spheres, event_sphere_velocity = interpolate_body_kinematics(
                source_collision_spheres,
                collision_spheres,
                source_collision_velocity,
                collision_sphere_velocity,
                imagined["ego_state"][:, :-1],
                imagined["ego_state"][:, 1:],
                event_ego_state,
                terminal_fraction,
            )
            event_sphere_mask = (
                source_collision_mask & collision_sphere_mask)
            event_valid = (
                imagined["human_mask_rollout"][:, 1:].bool()
                & event_sphere_mask.any(-1))
            event_clearance = analytic_joint_signed_surface_gap(
                event_spheres,
                event_sphere_mask,
                drone_radius_m=self.config.drone_collision_radius_m,
                joint_radii_m=collision_radii,
                fallback_gap_m=6.0,
                per_human=True,
            )
            partial_human_reward, partial_human_risk_state = (
                analytic_human_risk_reward(
                    event_clearance,
                    event_spheres[..., 0, :],
                    event_sphere_velocity[..., 0, :],
                    event_valid,
                    pelvis_valid=event_sphere_mask[..., 0],
                    human_presence=imagined["human_presence"][:, 1:],
                    ego_velocity_body=ego_velocity_full_body(event_ego_state),
                    hard_clearance_m=self.config.human_hard_clearance_m,
                    safe_clearance_m=self.config.human_safe_clearance_m,
                    predictive_safe_clearance_m=(
                        self.config.human_predictive_safe_clearance_m),
                    predictive_horizon_s=self.config.human_ttc_horizon_s,
                    corridor_half_width_m=(
                        self.config.human_corridor_half_width_m),
                    corridor_lookahead_m=(
                        self.config.human_corridor_lookahead_m),
                    drone_radius_m=self.config.drone_collision_radius_m,
                    pelvis_radius_m=self.config.pelvis_collision_radius_m,
                    weight_per_second=(
                        self.config.human_clearance_weight_per_sec),
                    dt_s=self.config.imagination_dt_s,
                ))
            partial_human_reward = (
                terminal_fraction * partial_human_reward)
            source_collision_spheres_quadrature = (
                source_collision_spheres.unsqueeze(2).expand(
                    -1, -1, quadrature_count, -1, -1, -1))
            collision_spheres_quadrature = collision_spheres.unsqueeze(
                2).expand_as(source_collision_spheres_quadrature)
            source_collision_velocity_quadrature = (
                source_collision_velocity.unsqueeze(2).expand_as(
                    source_collision_spheres_quadrature))
            collision_sphere_velocity_quadrature = (
                collision_sphere_velocity.unsqueeze(2).expand_as(
                    source_collision_spheres_quadrature))
            human_event_spheres, human_event_sphere_velocity = (
                interpolate_body_kinematics(
                    source_collision_spheres_quadrature,
                    collision_spheres_quadrature,
                    source_collision_velocity_quadrature,
                    collision_sphere_velocity_quadrature,
                    human_event_source_ego,
                    human_event_destination_ego,
                    human_event_ego,
                    human_event_fractions,
                ))
            human_event_sphere_mask = event_sphere_mask.unsqueeze(2).expand(
                -1, -1, quadrature_count, -1, -1)
            human_event_clearance = analytic_joint_signed_surface_gap(
                human_event_spheres,
                human_event_sphere_mask,
                drone_radius_m=self.config.drone_collision_radius_m,
                joint_radii_m=collision_radii,
                fallback_gap_m=6.0,
                per_human=True,
            )
            human_event_valid = event_valid.unsqueeze(2).expand(
                -1, -1, quadrature_count, -1)
            human_event_presence = imagined[
                "human_presence"][:, 1:].unsqueeze(2).expand(
                    -1, -1, quadrature_count, -1)
            human_event_clearance_reward_nodes, human_event_risk_state_nodes = (
                analytic_human_risk_reward(
                    human_event_clearance,
                    human_event_spheres[..., 0, :],
                    human_event_sphere_velocity[..., 0, :],
                    human_event_valid,
                    pelvis_valid=human_event_sphere_mask[..., 0],
                    human_presence=human_event_presence,
                    ego_velocity_body=ego_velocity_full_body(
                        human_event_ego),
                    hard_clearance_m=self.config.human_hard_clearance_m,
                    safe_clearance_m=self.config.human_safe_clearance_m,
                    predictive_safe_clearance_m=(
                        self.config.human_predictive_safe_clearance_m),
                    predictive_horizon_s=(
                        self.config.human_ttc_horizon_s),
                    corridor_half_width_m=(
                        self.config.human_corridor_half_width_m),
                    corridor_lookahead_m=(
                        self.config.human_corridor_lookahead_m),
                    drone_radius_m=self.config.drone_collision_radius_m,
                    pelvis_radius_m=self.config.pelvis_collision_radius_m,
                    weight_per_second=(
                        self.config.human_clearance_weight_per_sec),
                    dt_s=self.config.imagination_dt_s,
                ))
            human_event_clearance_reward = (
                human_event_mass_weights
                * human_event_fractions
                * human_event_clearance_reward_nodes
            ).sum(-2)
            human_event_risk_state = {
                name: (
                    value.amin(-2)
                    if name == "minimum_ttc_s" else
                    (human_event_conditional_weights * value).sum(-2)
                )
                for name, value in human_event_risk_state_nodes.items()
            }
            human_clearance_reward = (
                event_continuation * human_clearance_reward
                + human_event_clearance_reward
                + analytic_terminal_probability * partial_human_reward
            )
            human_risk_state.update({
                f"partial_{name}": value
                for name, value in partial_human_risk_state.items()
            })
            human_risk_state.update({
                f"human_event_{name}": value
                for name, value in human_event_risk_state.items()
            })
        else:
            minimum_human_clearance = learned_dense_reward.new_full(
                learned_dense_reward.shape, 6.0)
            human_clearance_reward = torch.zeros_like(learned_dense_reward)

        birth_risk = imagined.get("human_birth_probability")
        if birth_risk is None:
            birth_risk = torch.zeros_like(learned_dense_reward)
        else:
            birth_risk = birth_risk[:, :-1]
        reward = (
            learned_dense_reward
            + progress
            + human_clearance_reward
            + event_reward
            + route_reward
            + route_deviation_reward
            + heading_potential_reward
            + boundary_proximity_reward
            + static_potential_reward
            + analytic_height_reward
            + analytic_time_reward
            + analytic_smoothness
        )
        reward_abs_max = reward.detach().abs().amax()
        if not torch.isfinite(reward_abs_max):
            raise FloatingPointError("imagined task reward contains NaN/Inf")
        if float(reward_abs_max) > self.config.imagined_reward_abs_max_limit:
            raise FloatingPointError(
                "imagined task reward exceeded audited task guard: "
                f"{float(reward_abs_max):.4f} > "
                f"{self.config.imagined_reward_abs_max_limit:.4f}")
        effective_reward = reward
        # The explicit per-person geometry is part of the Markov information
        # used by the Actor's safety return.  Bootstrap from the matching
        # decision representation instead of the geometry-blind joint token.
        value = _mode(self.model.value(decision_feature))
        slow_value = _mode(self.model.slow_value(decision_feature))
        # Lambda targets must bootstrap from the delayed target Critic.  Using
        # the online Critic here makes the same network both define and chase
        # its target on every update; the slow network then degenerates into a
        # secondary regularizer instead of stabilizing long-horizon returns.
        # Parameters remain frozen during the Actor pass, while gradients with
        # respect to slow_value's input still preserve the dynamics-gradient
        # path from future value back through imagined actions.
        returns = self.lambda_return(
            effective_reward,
            continuation,
            slow_value,
            discount=self.config.discount,
            lamb=self.config.return_lambda,
        )
        prefix = torch.cat((
            torch.ones_like(continuation[:, :1]),
            torch.cumprod(
                self.config.discount * continuation[:, :-1], dim=1),
        ), dim=1).detach()
        start_importance = imagined.get("actor_imagination_start_importance")
        if start_importance is None:
            start_importance = prefix.new_ones((prefix.shape[0], 1))
        start_importance = start_importance.to(prefix).reshape(
            prefix.shape[0], 1, 1)
        if not bool(torch.isfinite(start_importance).all()) or bool(
            (start_importance <= 0.0).any()
        ):
            raise FloatingPointError(
                "imagination-start importance contains invalid values")
        prefix = prefix * start_importance
        return returns, prefix, value[:, :-1], slow_value[:, :-1], {
            "actor_imagination_start_count": imagined.get(
                "actor_imagination_start_count",
                reward.new_tensor(float(reward.shape[0]))),
            "actor_imagination_sample_count": imagined.get(
                "actor_imagination_sample_count",
                reward.new_ones(())),
            "actor_imagination_start_population_count": imagined.get(
                "actor_imagination_start_population_count",
                reward.new_tensor(float(reward.shape[0]))),
            "actor_imagination_actionable_population_count": imagined.get(
                "actor_imagination_actionable_population_count",
                reward.new_zeros(())),
            "actor_imagination_imminent_population_count": imagined.get(
                "actor_imagination_imminent_population_count",
                reward.new_zeros(())),
            "actor_imagination_preemptive_population_count": imagined.get(
                "actor_imagination_preemptive_population_count",
                reward.new_zeros(())),
            "actor_imagination_near_goal_ymax_misaligned_population_count": (
                imagined.get(
                    "actor_imagination_near_goal_ymax_misaligned_"
                    "population_count",
                    reward.new_zeros(()))),
            "actor_imagination_side_boundary_front_human_population_count": (
                imagined.get(
                    "actor_imagination_side_boundary_front_human_"
                    "population_count",
                    reward.new_zeros(()))),
            "actor_imagination_near_goal_population_count": imagined.get(
                "actor_imagination_near_goal_population_count",
                reward.new_zeros(())),
            "actor_imagination_near_boundary_population_count": imagined.get(
                "actor_imagination_near_boundary_population_count",
                reward.new_zeros(())),
            "actor_imagination_ordinary_population_count": imagined.get(
                "actor_imagination_ordinary_population_count",
                reward.new_tensor(float(reward.shape[0]))),
            "actor_imagination_actionable_selected_count": imagined.get(
                "actor_imagination_actionable_selected_count",
                reward.new_zeros(())),
            "actor_imagination_imminent_selected_count": imagined.get(
                "actor_imagination_imminent_selected_count",
                reward.new_zeros(())),
            "actor_imagination_preemptive_selected_count": imagined.get(
                "actor_imagination_preemptive_selected_count",
                reward.new_zeros(())),
            "actor_imagination_near_goal_ymax_misaligned_selected_count": (
                imagined.get(
                    "actor_imagination_near_goal_ymax_misaligned_"
                    "selected_count",
                    reward.new_zeros(()))),
            "actor_imagination_side_boundary_front_human_selected_count": (
                imagined.get(
                    "actor_imagination_side_boundary_front_human_"
                    "selected_count",
                    reward.new_zeros(()))),
            "actor_imagination_near_goal_selected_count": imagined.get(
                "actor_imagination_near_goal_selected_count",
                reward.new_zeros(())),
            "actor_imagination_near_boundary_selected_count": imagined.get(
                "actor_imagination_near_boundary_selected_count",
                reward.new_zeros(())),
            "actor_imagination_ordinary_selected_count": imagined.get(
                "actor_imagination_ordinary_selected_count",
                reward.new_tensor(float(reward.shape[0]))),
            "actor_imagination_start_importance_min": (
                start_importance.amin().detach()),
            "actor_imagination_start_importance_max": (
                start_importance.amax().detach()),
            "actor_imagination_start_effective_sample_size": (
                start_importance.sum().square()
                / start_importance.square().sum().clamp_min(1.0e-12)
            ).detach(),
            "actor_human_action_response_time_s": reward.new_tensor(
                self.actor_human_action_response_time_s),
            "actor_human_action_response_target_fraction": reward.new_tensor(
                self.ACTOR_HORIZONTAL_STEP_RESPONSE_TARGET),
            "actor_actionable_response_time_s": reward.new_tensor(
                self.actor_actionable_response_time_s),
            "actor_actionable_response_target_fraction": reward.new_tensor(
                self.ACTOR_ACTIONABLE_HORIZONTAL_RESPONSE_TARGET),
            "actor_human_action_horizon_response_fraction": reward.new_tensor(
                self.actor_human_action_horizon_response_fraction),
            "human_collision_prediction_bias_mean_m": (
                collision_prediction_bias.mean().detach()),
            "human_collision_prediction_bias_max_m": (
                collision_prediction_bias.amax().detach()),
            "reward": reward.mean(),
            "scalar_reward_diagnostic": scalar_reward.mean(),
            "reward_abs_max": reward_abs_max,
            "scalar_reward_abs_max": scalar_reward.abs().amax(),
            "effective_reward": effective_reward.mean(),
            "continuation": continuation.mean(),
            "learned_continuation": learned_continuation.mean(),
            "event_closed_continuation": event_continuation.mean(),
            "human_event_risk": human_risk.mean(),
            "learned_human_event_risk_diagnostic": (
                learned_human_event_probability.mean()),
            "observed_human_event_risk": observed_human_risk.mean(),
            "unobserved_human_event_risk": unobserved_human_risk.mean(),
            "human_event_geometry_path": human_risk.new_ones(()),
            "swept_human_contact_gap_m": (
                swept_human_contact_gap.clamp(-1.0, 6.0).mean()),
            "swept_human_contact_probability": (
                human_event_probability.mean()),
            "swept_human_contact_slot_probability": (
                uncertain_human_slot_probability.masked_fill(
                    ~swept_human_contact_valid_per_person, 0.0
                ).sum()
                / swept_human_contact_valid_per_person.float().sum(
                ).clamp_min(1.0)),
            "swept_human_contact_active_people": (
                swept_human_contact_valid_per_person.float().sum(
                    -1, keepdim=True).mean()),
            "exact_swept_human_contact_ratio": (
                swept_human_terminal.mean()),
            "swept_human_contact_fraction": (
                (human_event_probability * human_mean_fraction).sum()
                / human_event_probability.sum().clamp_min(1.0)),
            "human_birth_probability": birth_risk.mean(),
            "static_event_risk": static_risk.mean(),
            "learned_static_event_diagnostic": (
                learned_static_event_diagnostic.mean()),
            "goal_event_probability": goal_probability.mean(),
            "boundary_event_probability": boundary_probability.mean(),
            "task_memory_terminal_probability": (
                task_memory_terminal_probability.mean()),
            "other_terminal_probability": other_terminal_probability.mean(),
            "analytic_event_probability_sum_error": (
                analytic_event["probability_sum"] - 1.0).abs().amax(),
            "analytic_goal_inside_ratio": (
                imagined["analytic_terminal"][..., 0:1].mean()),
            "analytic_boundary_outside_ratio": (
                imagined["analytic_terminal"][..., 1:2].mean()),
            "analytic_crash_ratio": (
                imagined["analytic_terminal"][..., 2:3].mean()),
            "analytic_stuck_ratio": (
                imagined["analytic_terminal"][..., 3:4].mean()),
            "analytic_terminal_step_fraction": terminal_fraction.mean(),
            "human_mean_event_step_fraction": human_mean_fraction.mean(),
            "human_integrated_hazard": analytic_event[
                "integrated_human_hazard"].mean(),
            "expected_dense_step_fraction": expected_step_fraction.mean(),
            "analytic_bounds_available": reward.new_tensor(
                float(analytic_bounds_available)),
            "event_expected_reward": event_reward.mean(),
            "analytic_progress_reward": progress.mean(),
            "learned_dense_reward": learned_dense_reward.mean(),
            "analytic_height_reward": analytic_height_reward.mean(),
            "analytic_time_reward": analytic_time_reward.mean(),
            "analytic_smoothness_reward": analytic_smoothness.mean(),
            "smooth_acceleration_norm_mps2": smoothness_state[
                "acceleration_norm_mps2"].mean(),
            "smooth_jerk_norm_mps3": smoothness_state[
                "jerk_norm_mps3"].mean(),
            "smooth_yaw_rate_rps": smoothness_state[
                "yaw_rate_rps"].mean(),
            "partial_smooth_jerk_norm_mps3": partial_smoothness_state[
                "jerk_norm_mps3"].mean(),
            "human_event_smooth_jerk_norm_mps3": (
                human_partial_smoothness_state["jerk_norm_mps3"].mean()),
            "analytic_human_clearance_reward": human_clearance_reward.mean(),
            "analytic_human_instant_risk": human_risk_state.get(
                "instant_risk", human_clearance_reward.new_zeros(())).mean(),
            "analytic_human_predictive_risk": human_risk_state.get(
                "predictive_risk", human_clearance_reward.new_zeros(())).mean(),
            "analytic_human_corridor_risk": human_risk_state.get(
                "corridor_risk", human_clearance_reward.new_zeros(())).mean(),
            "human_min_clearance_m": minimum_human_clearance.mean(),
            "route_potential_reward": route_reward.mean(),
            "route_deviation_reward": route_deviation_reward.mean(),
            "route_cross_track_m": cross_track[:, 1:].mean(),
            "route_cross_track_max_m": cross_track[:, 1:].amax(),
            "route_heading_potential_reward": (
                heading_potential_reward.mean()),
            "route_heading_error_rad": heading_error[:, 1:].mean(),
            "route_heading_error_max_rad": heading_error[:, 1:].amax(),
            "boundary_proximity_reward": boundary_proximity_reward.mean(),
            "boundary_margin_excess_m": boundary_margin_excess.mean(),
            "static_proxy_potential_reward": static_potential_reward.mean(),
            "static_proxy_margin_excess_m": (
                static_margin_excess[:, 1:].mean()),
            "analytic_crash_probability": analytic_event["crash"].mean(),
            "analytic_stuck_probability": analytic_event["stuck"].mean(),
            "flight_bounds_signed_clearance_m": (
                boundary_signed_clearance.clamp(-100.0, 100.0).mean()),
            "authoritative_analytic_ego": imagined[
                "authoritative_analytic_ego"],
            "learned_ego_task_position_gap_m": torch.linalg.vector_norm(
                imagined["learned_ego_state"][..., :3]
                - imagined["ego_state"][..., :3],
                dim=-1,
            ).mean(),
            "return": returns.mean(),
            "value": value[:, :-1].mean(),
            "slow_value": slow_value[:, :-1].mean(),
            "return_bootstrap_uses_slow_value": slow_value.new_ones(()),
            **component_metrics,
        }

    @torch.no_grad()
    def _assert_actor_output_guard(
        self,
        feature: torch.Tensor,
        *,
        context: str,
    ) -> None:
        """Fail closed when the deployable Actor loses tanh gradient headroom."""
        distribution = self.model.actor(feature)
        pre_tanh_mean = getattr(
            distribution, "pre_tanh_mean",
            getattr(distribution, "_mean", None))
        if pre_tanh_mean is None:
            raise TypeError(
                "pure Dreamer Actor distribution lacks a pre-tanh mean")
        raw_pre_tanh_mean = getattr(
            distribution, "raw_pre_tanh_mean", pre_tanh_mean)
        mean_bound_jacobian = getattr(
            distribution, "mean_bound_jacobian", None)
        raw_std_residual = getattr(
            distribution, "raw_std_residual", None)
        if not torch.isfinite(raw_pre_tanh_mean).all():
            raise FloatingPointError(
                f"Actor raw mean contains NaN/Inf {context}")
        if mean_bound_jacobian is None:
            raise TypeError(
                "pure Dreamer Actor distribution lacks its mean-bound "
                "Jacobian")
        if (
            not torch.isfinite(mean_bound_jacobian).all()
            or bool((mean_bound_jacobian <= 0.0).any())
        ):
            raise FloatingPointError(
                "Actor mean-bound Jacobian is non-finite/non-positive "
                f"{context}")
        if (
            raw_std_residual is not None
            and not torch.isfinite(raw_std_residual).all()
        ):
            raise FloatingPointError(
                f"Actor raw standard deviation contains NaN/Inf {context}")
        pre_tanh_abs_max = pre_tanh_mean.abs().amax()
        if not torch.isfinite(pre_tanh_abs_max):
            raise FloatingPointError(
                f"Actor pre-tanh mean contains NaN/Inf {context}")
        if float(pre_tanh_abs_max) > self.config.actor_pre_tanh_abs_max_limit:
            absolute = pre_tanh_mean.abs()
            flat_index = int(absolute.reshape(-1).argmax())
            action_dim = int(pre_tanh_mean.shape[-1])
            action_axis = flat_index % action_dim
            row_index = flat_index // action_dim
            time_size = (
                int(pre_tanh_mean.shape[-2])
                if pre_tanh_mean.ndim >= 3 else 1)
            start_index = row_index // time_size
            time_index = row_index % time_size
            axis_names = ("forward", "lateral", "yaw")
            axis_name = (
                axis_names[action_axis]
                if action_dim == len(axis_names) else str(action_axis))
            per_axis_max = absolute.reshape(-1, action_dim).amax(0)
            signed_value = pre_tanh_mean.reshape(
                -1, action_dim)[row_index, action_axis]
            actor = self.model.actor
            decision_encoder = getattr(actor, "decision_encoder", None)
            encoded_feature = (
                decision_encoder(feature)
                if callable(decision_encoder) else feature)
            raise FloatingPointError(
                "Actor pre-tanh mean exceeded fail-fast limit "
                f"{context}: {float(pre_tanh_abs_max):.4f} > "
                f"{self.config.actor_pre_tanh_abs_max_limit:.4f}; "
                f"axis={axis_name} signed={float(signed_value):.4f} "
                f"start={start_index} imagination_step={time_index} "
                f"per_axis_max={[float(value) for value in per_axis_max]} "
                f"decision_feature_abs_max={float(feature.abs().amax()):.4f} "
                "encoded_feature_abs_max="
                f"{float(encoded_feature.abs().amax()):.4f}")

    def _repeat_actor_imagination_samples(
        self,
        initial: Any,
        values: Mapping[str, torch.Tensor],
        sample_count: int,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Duplicate adjacent starts for variance-reduced policy rollouts."""
        count = int(sample_count)
        if count <= 1:
            return initial, dict(values)
        batch_size = int(initial["ego"]["deter"].shape[0])
        repeated_initial = self.model._repeat_state_candidates(initial, count)
        repeated_values: dict[str, torch.Tensor] = {}
        for name, value in values.items():
            if not torch.is_tensor(value):
                raise TypeError(f"imagination value {name} is not a tensor")
            if value.ndim == 0:
                repeated_values[name] = value
                continue
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"imagination value {name} has no start batch axis")
            repeated_values[name] = value[:, None].expand(
                batch_size, count, *value.shape[1:]
            ).reshape(batch_size * count, *value.shape[1:])
        repeated_values["actor_imagination_sample_count"] = (
            initial["ego"]["deter"].new_tensor(float(count)))
        return repeated_initial, repeated_values

    def _actor_critic_update(
        self,
        states: Any,
        auxiliary: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        initial, values = self._imagination_inputs(states, auxiliary, batch)
        actor_sample_count = int(self.config.actor_imagination_samples)
        initial, values = self._repeat_actor_imagination_samples(
            initial, values, actor_sample_count)
        frozen = self.world_parameters + self.critic_parameters
        self.actor_optimizer.zero_grad(set_to_none=True)
        with self._temporarily_frozen(frozen):
            # Continuous-action Dreamer uses reparameterized samples and
            # differentiates imagined returns through the frozen World Model.
            # Freezing parameters does not stop gradients with respect to the
            # Actor's actions: action -> smoother -> RSSM/analytic Ego ->
            # reward/risk/value remains one differentiable computation graph.
            imagined = self._imagine(
                initial,
                values,
                antithetic_policy_pairs=actor_sample_count > 1,
            )
            returns, weights, value, slow_value, metrics = (
                self._imagined_objective(imagined))
            return_offset, return_scale, return_ema_bootstrapped = (
                self._normalize_returns(
                    returns,
                    imagined.get("actor_imagination_start_importance"),
                ))
            normalized_return = (returns - return_offset) / return_scale
            # Retain the value-baseline statistic for diagnostics only.  The
            # continuous Actor objective is the dynamics gradient of return,
            # not a score-function log-probability estimator.
            advantage = (returns - value) / return_scale

            actor_guard_feature = imagined["actor_feat"][:, :-1].detach()
            self._assert_actor_output_guard(
                actor_guard_feature, context="before optimizer step")
            distribution = self.model.actor(imagined["actor_feat"][:, :-1])
            if "policy_pre_tanh" not in imagined:
                raise RuntimeError(
                    "pure Dreamer requires retained pre-tanh policy samples")
            log_prob_from_pre_tanh = getattr(
                distribution, "log_prob_from_pre_tanh", None)
            if not callable(log_prob_from_pre_tanh):
                raise TypeError(
                    "pure Dreamer Actor distribution must support stable "
                    "pre-tanh log probability")
            policy_pre_tanh = imagined[
                "policy_pre_tanh"][:, :-1].detach()
            log_probability = log_prob_from_pre_tanh(policy_pre_tanh)
            if log_probability.ndim == 2:
                log_probability = log_probability[..., None]
            entropy = distribution.entropy()
            if entropy.ndim == 2:
                entropy = entropy[..., None]
            actor_dynamics_entropy_loss = (
                weights.detach()
                * -(normalized_return + self.config.actor_entropy * entropy)
            ).mean()
            raw_pre_tanh_mean = getattr(
                distribution, "raw_pre_tanh_mean", None)
            if raw_pre_tanh_mean is None:
                raise TypeError(
                    "pure Dreamer Actor distribution lacks its raw mean")
            # The algebraic map is one-to-one but nearly action-redundant at
            # its extremes.  With an actuator-extreme optimum, Adam therefore
            # keeps increasing raw logits for imperceptible action changes and
            # erases the very pathwise gradient needed on rare failure states.
            # This soft penalty starts outside the calibrated raw support and
            # is deliberately independent of replay labels or risk classes.
            raw_headroom_excess = F.relu(
                raw_pre_tanh_mean.float().abs()
                - self.config.actor_raw_pre_tanh_soft_limit)
            actor_raw_headroom_penalty = raw_headroom_excess.square().mean()
            actor_loss = (
                actor_dynamics_entropy_loss
                + self.config.actor_raw_pre_tanh_headroom_scale
                * actor_raw_headroom_penalty)
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("non-finite Actor loss")
            actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_parameters, self.config.actor_grad_clip,
            error_if_nonfinite=True)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_feature = imagined["decision_feat"][:, :-1].detach()
        critic_distribution = self.model.value(critic_feature)
        critic_support = getattr(critic_distribution, "bins", None)
        critic_return_outside_support = returns.new_zeros(())
        if critic_support is not None:
            support_minimum = critic_support[0].to(returns)
            support_maximum = critic_support[-1].to(returns)
            outside_support = (
                (returns.detach() < support_minimum)
                | (returns.detach() > support_maximum)
            )
            critic_return_outside_support = outside_support.float().mean()
            if bool(outside_support.any()):
                raise FloatingPointError(
                    "Critic lambda return exceeded its registered raw "
                    "two-hot support: observed="
                    f"[{float(returns.detach().amin()):.4f},"
                    f"{float(returns.detach().amax()):.4f}] support="
                    f"[{float(support_minimum):.4f},"
                    f"{float(support_maximum):.4f}]")
        critic_nll = -critic_distribution.log_prob(returns.detach())
        slow_critic_nll = -critic_distribution.log_prob(slow_value.detach())
        if critic_nll.ndim == 2:
            critic_nll = critic_nll[..., None]
            slow_critic_nll = slow_critic_nll[..., None]
        critic_loss = (
            weights.detach() * (critic_nll + slow_critic_nll)
        ).mean()
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("non-finite Critic loss")
        critic_loss.backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_parameters, self.config.critic_grad_clip,
            error_if_nonfinite=True)
        metrics = {
            **metrics,
            "entropy": entropy.mean().detach(),
            "log_probability": log_probability.mean().detach(),
            "return_005": self.model.return_ema.ema_vals[0].detach(),
            "return_095": self.model.return_ema.ema_vals[1].detach(),
            "return_scale": return_scale.detach(),
            "return_ema_bootstrapped": return_ema_bootstrapped.detach(),
            "normalized_return": normalized_return.mean().detach(),
            "actor_dynamics_entropy_loss": (
                actor_dynamics_entropy_loss.detach()),
            "actor_raw_headroom_penalty": (
                actor_raw_headroom_penalty.detach()),
            "actor_raw_headroom_loss": (
                self.config.actor_raw_pre_tanh_headroom_scale
                * actor_raw_headroom_penalty.detach()),
            "actor_raw_headroom_exceed_fraction": (
                raw_headroom_excess.gt(0.0).float().mean().detach()),
            "actor_grad_clip_scale": torch.clamp(
                actor_norm.new_tensor(self.config.actor_grad_clip)
                / actor_norm.detach().clamp_min(1.0e-12),
                max=1.0,
            ),
            "critic_grad_clip_scale": torch.clamp(
                critic_norm.new_tensor(self.config.critic_grad_clip)
                / critic_norm.detach().clamp_min(1.0e-12),
                max=1.0,
            ),
            "dynamics_objective": (
                weights.detach() * normalized_return).mean().detach(),
            "advantage": advantage.mean().detach(),
            "advantage_std": advantage.std(unbiased=False).detach(),
            "action_abs_mean": imagined["policy_action"][:, :-1].abs().mean().detach(),
            "action_smoother_local_scale_mean": imagined[
                "action_smoother_local_scale"][:, :-1].mean().detach(),
            "action_smoother_nonzero_jacobian_fraction": imagined[
                "action_smoother_local_scale"][:, :-1].gt(
                    1.0e-10).float().mean().detach(),
            # Keep Dreamer's unit slow-Critic regularizer unchanged, but expose
            # its two targets separately.  A large value/return lag must not be
            # mistaken for an Actor loss or "fixed" by silently changing the
            # standard lambda-return/Critic objective.
            "critic_return_nll": (
                weights.detach() * critic_nll.detach()).mean(),
            "critic_slow_regularizer_nll": (
                weights.detach() * slow_critic_nll.detach()).mean(),
            "critic_return_mae": (
                weights.detach() * (value - returns.detach()).abs()).mean(),
            "critic_slow_value_mae": (
                weights.detach() * (value - slow_value.detach()).abs()).mean(),
            "critic_return_outside_support_fraction": (
                critic_return_outside_support.detach()),
        }
        action_axis_names = ("forward", "lateral", "yaw")
        raw_policy = imagined["policy_action"][:, :-1]
        applied_policy = imagined["smoothed_policy_action"][:, :-1]
        local_scale = imagined["action_smoother_local_scale"][:, :-1]
        for axis, name in enumerate(action_axis_names):
            metrics.update({
                f"actor_sample_{name}_mean": raw_policy[
                    ..., axis].mean().detach(),
                f"applied_policy_{name}_mean": applied_policy[
                    ..., axis].mean().detach(),
                f"applied_policy_{name}_bound_ratio": applied_policy[
                    ..., axis].abs().gt(0.95).float().mean().detach(),
                f"action_smoother_{name}_local_scale_mean": local_scale[
                    ..., axis].mean().detach(),
            })
        mode = distribution.mode
        if callable(mode):
            mode = mode()
        metrics.update({
            "actor_mode_abs_mean": mode.abs().mean().detach(),
            "actor_mode_saturation_ratio": mode.abs().gt(0.95).float().mean().detach(),
            "actor_learned_std_enabled": mode.new_tensor(float(
                getattr(
                    self.model.actor.config, "std_parameterization", "")
                == "learned_bounded")),
        })
        pre_tanh_mean = getattr(
            distribution, "pre_tanh_mean",
            getattr(distribution, "_mean", None))
        if pre_tanh_mean is not None:
            metrics.update({
                "actor_pre_tanh_abs_mean": pre_tanh_mean.abs().mean().detach(),
                "actor_pre_tanh_abs_max": pre_tanh_mean.abs().amax().detach(),
                "actor_pre_tanh_headroom": (
                    self.config.actor_pre_tanh_abs_max_limit
                    - pre_tanh_mean.abs().amax()).detach(),
            })
        raw_pre_tanh_mean = getattr(
            distribution, "raw_pre_tanh_mean", None)
        if raw_pre_tanh_mean is not None:
            metrics.update({
                "actor_raw_pre_tanh_abs_mean": (
                    raw_pre_tanh_mean.abs().mean().detach()),
                "actor_raw_pre_tanh_abs_max": (
                    raw_pre_tanh_mean.abs().amax().detach()),
            })
        mean_bound_jacobian = getattr(
            distribution, "mean_bound_jacobian", None)
        if mean_bound_jacobian is not None:
            raw_to_mode_jacobian = mean_bound_jacobian * (
                1.0 - mode.square())
            metrics.update({
                "actor_mean_bound_jacobian_mean": (
                    mean_bound_jacobian.mean().detach()),
                "actor_mean_bound_jacobian_min": (
                    mean_bound_jacobian.amin().detach()),
                "actor_mean_bound_zero_jacobian_fraction": (
                    mean_bound_jacobian.eq(0.0).float().mean().detach()),
                "actor_raw_to_mode_jacobian_mean": (
                    raw_to_mode_jacobian.mean().detach()),
                "actor_raw_to_mode_jacobian_min": (
                    raw_to_mode_jacobian.amin().detach()),
            })
        raw_std_residual = getattr(
            distribution, "raw_std_residual", None)
        if raw_std_residual is not None:
            metrics.update({
                "actor_raw_std_residual_abs_mean": (
                    raw_std_residual.abs().mean().detach()),
                "actor_raw_std_residual_abs_max": (
                    raw_std_residual.abs().amax().detach()),
            })
        std = getattr(distribution, "stddev", None)
        if std is not None:
            metrics.update({
                "actor_std_mean": std.mean().detach(),
                "actor_std_min": std.amin().detach(),
                "actor_std_max": std.amax().detach(),
            })
            if std.shape[-1] == 3:
                for axis, name in enumerate(("forward", "lateral", "yaw")):
                    axis_std = std[..., axis]
                    metrics.update({
                        f"actor_std_{name}_mean": axis_std.mean().detach(),
                        f"actor_std_{name}_min": axis_std.amin().detach(),
                        f"actor_std_{name}_max": axis_std.amax().detach(),
                    })
        goal_mean = getattr(distribution, "goal_mean", None)
        if goal_mean is not None:
            metrics["actor_goal_mean_abs"] = goal_mean.abs().mean().detach()
        residual = getattr(distribution, "avoidance_residual", None)
        if residual is not None:
            metrics["actor_avoidance_residual_abs"] = (
                residual.abs().mean().detach())
        gate = getattr(distribution, "conflict_gate", None)
        if gate is not None:
            metrics.update({
                "actor_conflict_gate_mean": gate.mean().detach(),
                "actor_conflict_gate_max": gate.amax().detach(),
            })
        if gate is not None and residual is not None:
            residual_bound = float(getattr(
                getattr(self.model.actor, "config", None),
                "residual_bound", 1.0))
            correction = gate * residual_bound * residual
            metrics["actor_avoidance_correction_abs"] = (
                correction.abs().mean().detach())
            if goal_mean is not None:
                metrics["actor_avoidance_to_goal_ratio"] = (
                    correction.abs().mean()
                    / goal_mean.abs().mean().clamp_min(1.0e-6)
                ).detach()
            if all(key in imagined for key in (
                "human_joint_clearance", "human_presence",
            )):
                clearance = imagined["human_joint_clearance"][:, :-1]
                presence = imagined["human_presence"][:, :-1].gt(0.05)
                clearance = clearance.masked_fill(~presence, torch.inf)
                has_human = presence.any(-1)
                minimum = clearance.amin(-1)
                near = has_human & minimum.le(
                    self.config.human_safe_clearance_m)
                far = has_human & ~near

                def conditional_mean(
                    value: torch.Tensor, mask: torch.Tensor,
                ) -> torch.Tensor:
                    weight = mask.to(value)[..., None]
                    return (
                        (value * weight).sum()
                        / (weight.sum() * value.shape[-1]).clamp_min(1.0)
                    )

                metrics.update({
                    "actor_conflict_gate_near_human_mean": conditional_mean(
                        gate, near).detach(),
                    "actor_conflict_gate_far_human_mean": conditional_mean(
                        gate, far).detach(),
                    "actor_conflict_gate_no_human_mean": conditional_mean(
                        gate, ~has_human).detach(),
                    "actor_near_human_fraction": near.float().mean().detach(),
                })
            actor_config = getattr(self.model.actor, "config", None)
            if bool(getattr(
                actor_config, "human_reflection_equivariant", False
            )):
                # This is a read-only invariant audit, not an auxiliary Actor
                # loss. The only optimized objective remains imagined return.
                with torch.no_grad():
                    feature = imagined["actor_feat"][:, :-1].detach()
                    goal_token, ego_token, joint_token, human_token = (
                        torch.chunk(feature, 4, dim=-1))
                    mirrored_human = (
                        self.model.actor._mirror_physical_human_token(
                            human_token))
                    mirrored_feature = torch.cat((
                        goal_token, ego_token, joint_token, mirrored_human,
                    ), -1)
                    mirrored_distribution = self.model.actor(mirrored_feature)
                    mirrored_correction = (
                        mirrored_distribution.conflict_gate
                        * residual_bound
                        * mirrored_distribution.avoidance_residual)
                    even_error = (
                        mirrored_correction[..., 0]
                        - correction[..., 0]).abs()
                    odd_error = (
                        mirrored_correction[..., 1:]
                        + correction[..., 1:]).abs()
                    metrics.update({
                        "actor_human_reflection_even_error_max": (
                            even_error.amax()),
                        "actor_human_reflection_odd_error_max": (
                            odd_error.amax()),
                        "actor_human_reflection_gate_error_max": (
                            mirrored_distribution.conflict_gate
                            - gate).abs().amax(),
                    })
                    if all(key in imagined for key in (
                        "human_joint_clearance", "human_presence",
                    )):
                        nearest_y = human_token[..., 1]
                        away_correction = (
                            -nearest_y.sign() * correction[..., 1])
                        near_weight = near.to(away_correction)
                        metrics[
                            "actor_near_human_away_correction_mean"
                        ] = (
                            (away_correction * near_weight).sum()
                            / near_weight.sum().clamp_min(1.0)
                        )
                        metrics[
                            "actor_near_human_away_correction_positive_fraction"
                        ] = (
                            (away_correction.gt(0.0) & near).float().sum()
                            / near.float().sum().clamp_min(1.0)
                        )
        return (
            actor_loss.detach(), critic_loss.detach(),
            torch.as_tensor(actor_norm).detach(), torch.as_tensor(critic_norm).detach(),
            metrics, actor_guard_feature,
        )

    @torch.no_grad()
    def _update_slow_value(self) -> None:
        fraction = self.config.slow_value_fraction
        for target, source in zip(
            self.model.slow_value.parameters(),
            self.model.value.parameters(),
            strict=True,
        ):
            target.data.lerp_(source.data, fraction)

    def _update_once(
        self, batch: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        profile_runtime = os.environ.get(
            "PURE_DREAMER_PROFILE_UPDATE_TIMES", "0") == "1"
        profile_previous = time.perf_counter()
        profile_metrics: dict[str, float] = {}

        def profile_mark(name: str) -> None:
            nonlocal profile_previous
            if not profile_runtime:
                return
            device = next(iter(self.model.parameters())).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            profile_metrics[f"runtime/profile_{name}_ms"] = (
                now - profile_previous) * 1000.0
            profile_previous = now

        self.model.transition_event.require_identity_human_probability_calibration()
        self.checkpoint_safe = False
        self._optimizer_step_started = False
        self.model.train(True)
        self.world_optimizer.zero_grad(set_to_none=True)
        parameter_device = next(iter(self.model.parameters())).device
        with torch.autocast(
            device_type=parameter_device.type,
            dtype=self.amp_dtype,
            enabled=self.autocast_enabled,
        ):
            world_loss, world_metrics, states, auxiliary = self._world_loss(
                batch)
        profile_mark("world_forward")
        if not torch.isfinite(world_loss):
            raise FloatingPointError("non-finite world-model loss")
        world_loss.backward()
        profile_mark("world_backward")
        world_metrics["runtime/world_grad_float64_fallback"] = (
            world_loss.detach().new_zeros(()))
        try:
            world_norm, used_float64_fallback = (
                _clip_grad_norm_with_float64_fallback_(
                    self.world_parameters, self.config.world_grad_clip))
            world_metrics["runtime/world_grad_clip_scale"] = torch.clamp(
                world_norm.new_tensor(self.config.world_grad_clip)
                / world_norm.detach().clamp_min(1.0e-12),
                max=1.0,
            )
            if used_float64_fallback:
                world_metrics["runtime/world_grad_float64_fallback"] = (
                    world_loss.detach().new_ones(()))
        except RuntimeError as error:
            # A finite scalar loss can still have a singular derivative.  Do
            # not silently skip or scale such an update: identify the exact
            # World module so the underlying formula can be corrected while
            # the last checkpoint remains transactional and publishable.
            bad_gradients: list[str] = []
            for name, parameter in self.model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                detached = gradient.detach().float()
                finite = torch.isfinite(detached)
                finite_values = detached[finite]
                finite_max = (
                    float(finite_values.abs().amax())
                    if finite_values.numel() else float("nan")
                )
                tensor_norm = torch.linalg.vector_norm(detached)
                if not bool(finite.all()) or not bool(
                    torch.isfinite(tensor_norm)
                ):
                    bad_gradients.append(
                        f"{name}(nan={int(torch.isnan(detached).sum())},"
                        f"inf={int(torch.isinf(detached).sum())},"
                        f"finite_max={finite_max:.6g})"
                    )
            finite_metrics = {
                name: float(value)
                for name, value in world_metrics.items()
                if math.isfinite(float(value))
            }
            largest_metrics = sorted(
                finite_metrics.items(), key=lambda item: abs(item[1]),
                reverse=True,
            )[:12]
            raise FloatingPointError(
                "non-finite world-model gradients; parameters="
                f"{bad_gradients[:24]}; largest_world_metrics="
                f"{largest_metrics}"
            ) from error
        profile_mark("world_gradient_clip")

        # All three gradient sets are built and validated before any optimizer
        # mutates parameters.  Actor/Critic use a detached replay posterior,
        # which is the standard Dreamer boundary and makes the joint call
        # transactional with respect to non-finite losses/gradients.
        states = _tree_detach(states)
        auxiliary = _tree_detach(auxiliary)
        with torch.autocast(
            device_type=parameter_device.type,
            dtype=self.amp_dtype,
            enabled=self.autocast_enabled,
        ):
            (
                actor_loss, critic_loss, actor_norm, critic_norm, imagined,
                actor_guard_feature,
            ) = self._actor_critic_update(states, auxiliary, batch)
        profile_mark("actor_critic")
        pending_metrics = {
            **{name: float(value) for name, value in world_metrics.items()},
            **{f"imag/{name}": float(value.detach()) for name, value in imagined.items()},
            "loss/world": float(world_loss.detach()),
            "loss/actor": float(actor_loss),
            "loss/critic": float(critic_loss),
            "loss/total": float(world_loss.detach() + actor_loss + critic_loss),
            "grad/world": float(world_norm),
            "grad/actor": float(actor_norm),
            "grad/critic": float(critic_norm),
            "opt/update_count": float(self.update_count + 1),
        }
        if self.event_prior is not None:
            for key, value in self.event_prior[
                "categorical_probability"].items():
                pending_metrics[f"prior/event_probability/{key}"] = float(value)
            pending_metrics["prior/explicit_human_probability"] = float(
                self.event_prior["explicit_human_probability"])
            pending_metrics["prior/continuation_probability"] = float(
                self.event_prior["continuation_probability"])
            for key, value in self.event_prior[
                "dense_reward_component_symlog_mean"].items():
                pending_metrics[
                    f"prior/reward_component_symlog/{key}"] = float(value)
        if not all(math.isfinite(value) for value in pending_metrics.values()):
            raise FloatingPointError("pure Dreamer metrics contain NaN/Inf")
        profile_mark("metric_materialization")
        self._optimizer_step_started = True
        self.world_optimizer.step()
        self.actor_optimizer.step()
        # The pre-step audit alone is insufficient: one finite AdamW update can
        # move an otherwise valid policy into tanh saturation.  Re-evaluate the
        # exact deployable Actor on the exact same decision states before this
        # joint update can increment its clock or become checkpoint-publishable.
        self._assert_actor_output_guard(
            actor_guard_feature, context="after optimizer step")
        self.critic_optimizer.step()
        self._update_slow_value()
        profile_mark("optimizer_and_actor_guard")
        for name, parameters in (
            ("World", self.world_parameters),
            ("Actor", self.actor_parameters),
            ("Critic", self.critic_parameters),
        ):
            if not _all_floating_tensors_finite(parameters):
                raise FloatingPointError(
                    f"{name} parameters became non-finite after optimizer step")
        profile_mark("parameter_integrity")
        for name, optimizer in (
            ("World", self.world_optimizer),
            ("Actor", self.actor_optimizer),
            ("Critic", self.critic_optimizer),
        ):
            state_tensors = [
                value
                for state in optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            ]
            if not _all_floating_tensors_finite(state_tensors):
                raise FloatingPointError(
                    f"{name} optimizer state became non-finite")
        profile_mark("optimizer_integrity")
        pending_metrics.update(profile_metrics)
        self.update_count += 1
        self.checkpoint_safe = True
        self._optimizer_step_started = False
        return pending_metrics

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        started = time.perf_counter()
        if not self.event_prior_initialized:
            self._initialize_event_priors_from_batch(batch)
        # Return normalization is the only persistent state updated while
        # losses are still being validated.  Restore it when a failure occurs
        # before optimizer mutation, so a rejected minibatch changes neither
        # parameters nor resumable statistics.
        return_ema_state = {
            key: value.detach().clone()
            for key, value in self.model.return_ema.state_dict().items()
        }
        try:
            metrics = self._update_once(batch)
            metrics["runtime/update_ms"] = (
                time.perf_counter() - started) * 1000.0
            return metrics
        except Exception:
            if not self._optimizer_step_started:
                self.model.return_ema.load_state_dict(return_ema_state)
            self.world_optimizer.zero_grad(set_to_none=True)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            raise

    def state_dict(self) -> dict[str, Any]:
        self.model.transition_event.require_identity_human_probability_calibration()
        return {
            "objective_version": self.OBJECTIVE_VERSION,
            "imagination_start_sampling_contract": (
                self.IMAGINATION_START_SAMPLING_CONTRACT_VERSION),
            "config": asdict(self.config),
            "world_optimizer": self.world_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": int(self.update_count),
            "event_prior": copy.deepcopy(self.event_prior),
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, allow_legacy_v4: bool = False,
    ) -> None:
        objective = state.get("objective_version")
        legacy = bool(
            allow_legacy_v4
            and objective == self.LEGACY_MIGRATION_OBJECTIVE)
        if objective != self.OBJECTIVE_VERSION and not legacy:
            raise RuntimeError(
                f"pure Dreamer objective mismatch: {objective!r}")
        sampling_contract = state.get("imagination_start_sampling_contract")
        if (
            sampling_contract is not None
            and sampling_contract
            != self.IMAGINATION_START_SAMPLING_CONTRACT_VERSION
            and sampling_contract
            not in self.LEGACY_IMAGINATION_START_SAMPLING_CONTRACTS
        ):
            raise RuntimeError(
                "pure Dreamer imagination-start sampling mismatch: "
                f"{sampling_contract!r}")
        saved_config = state.get("config")
        expected_config = asdict(self.config)
        if not isinstance(saved_config, Mapping):
            raise RuntimeError("pure Dreamer checkpoint has no trainer config")
        if legacy:
            def compatible_subset(saved: Any, current: Any) -> bool:
                if isinstance(saved, Mapping) and isinstance(current, Mapping):
                    return all(
                        key in current
                        and compatible_subset(value, current[key])
                        for key, value in saved.items())
                return saved == current

            mismatched = {
                key: (value, expected_config.get(key))
                for key, value in saved_config.items()
                if key in expected_config
                and not compatible_subset(
                    value,
                    ({
                        **expected_config[key],
                        "ego_pred": value.get("ego_pred"),
                    } if key == "world_loss_scales"
                     and isinstance(value, Mapping) else expected_config[key]),
                )
            }
        else:
            mismatched = (
                {} if dict(saved_config) == expected_config else
                {"saved": dict(saved_config), "current": expected_config})
        if mismatched:
            raise RuntimeError(
                f"pure Dreamer trainer config mismatch: {mismatched}")
        def load_optimizer(
            optimizer: torch.optim.Optimizer,
            saved: Mapping[str, Any],
            *,
            allow_appended_parameters: bool,
        ) -> None:
            payload = copy.deepcopy(dict(saved))
            if allow_appended_parameters:
                current = optimizer.state_dict()
                saved_groups = payload.get("param_groups", ())
                current_groups = current.get("param_groups", ())
                if len(saved_groups) != len(current_groups):
                    raise RuntimeError(
                        "legacy optimizer parameter-group count mismatch")
                for saved_group, current_group in zip(
                    saved_groups, current_groups, strict=True,
                ):
                    saved_parameters = list(saved_group["params"])
                    current_parameters = list(current_group["params"])
                    if len(saved_parameters) > len(current_parameters):
                        raise RuntimeError(
                            "legacy optimizer has more parameters than v6")
                    # v6.9 appends only the zero-initialized task-geometry
                    # adapter; every legacy parameter retains its ordinal.
                    saved_group["params"] = (
                        saved_parameters
                        + current_parameters[len(saved_parameters):]
                    )
            optimizer.load_state_dict(payload)

        load_optimizer(
            self.world_optimizer,
            state["world_optimizer"],
            allow_appended_parameters=legacy,
        )
        load_optimizer(
            self.actor_optimizer,
            state["actor_optimizer"],
            allow_appended_parameters=False,
        )
        load_optimizer(
            self.critic_optimizer,
            state["critic_optimizer"],
            allow_appended_parameters=False,
        )
        # Optimizer moments are resumable state; learning rates and decay are
        # part of the current, fingerprinted objective and must not be silently
        # overwritten by an older param group.
        for optimizer, learning_rate, weight_decay in (
            (self.world_optimizer, self.config.world_lr,
             self.config.world_weight_decay),
            (self.actor_optimizer, self.config.actor_lr,
             self.config.actor_weight_decay),
            (self.critic_optimizer, self.config.critic_lr,
             self.config.critic_weight_decay),
        ):
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)
                group["weight_decay"] = float(weight_decay)
        self.update_count = int(state.get("update_count", 0))
        event_prior = state.get("event_prior")
        if event_prior is None and self.update_count > 0 and not legacy:
            raise RuntimeError(
                "v12 checkpoint is missing its empirical Event prior")
        self.event_prior = (
            None if event_prior is None else copy.deepcopy(dict(event_prior)))
        self.checkpoint_safe = True
