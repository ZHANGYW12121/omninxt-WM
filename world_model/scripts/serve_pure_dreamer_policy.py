#!/usr/bin/env python3
"""Serve only the deterministic pure-Dreamer Actor (scheme one).

The observation packing, skeleton history, action adapter and stateful action
smoother are reused from the established factorized runtime.  Candidate
generation, world-model planning, geometry projection, shadow selection and
runtime exploration are hard-disabled here rather than controlled by flags.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import math
from pathlib import Path
import socketserver
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytic_ego_contract import load_analytic_ego_contract  # noqa: E402
from modules.factorized_actor_v62 import (  # noqa: E402
    CompleteDecisionStateEncoder,
)
from modules.task_geometry import (  # noqa: E402
    TASK_SCENE_GEOMETRY_CONTRACT_VERSION,
    fixed_task_scene_geometry_contract,
    fixed_task_scene_geometry_equivalent,
)
from pure_dreamer import PureDreamerTrainer  # noqa: E402
from scripts.serve_factorized_policy import (  # noqa: E402
    PROTOCOL,
    FactorizedPolicyRuntime,
    PolicyHandler,
    ProspectiveValidationExplorer,
    ReloadableFactorizedPolicyRuntime,
)


PURE_DECISION_ENCODED_DIM = 1768
PURE_CRITIC_RETURN_DISTRIBUTION = "linear_twohot"
PURE_CRITIC_RETURN_SUPPORT = (-256.0, 256.0, 255)
PURE_ACTOR_PRE_TANH_MEAN_BOUND = 3.3


class PureActorRuntime(FactorizedPolicyRuntime):
    """Compatibility observation runtime with a non-overridable Actor path."""

    EXPECTED_PURE_OBJECTIVE_VERSION = PureDreamerTrainer.OBJECTIVE_VERSION

    def __init__(
        self, checkpoint_path: Path, device: str, *, collection: bool = False,
        validation_collection: bool = False,
        allow_out_of_support_task_evaluation: bool = False,
    ) -> None:
        if collection and validation_collection:
            raise ValueError(
                "training collection and prospective validation collection "
                "are mutually exclusive")
        self.validation_collection = bool(validation_collection)
        self.collection = bool(collection or validation_collection)
        self.allow_out_of_support_task_evaluation = bool(
            allow_out_of_support_task_evaluation)
        if self.collection and self.allow_out_of_support_task_evaluation:
            raise ValueError(
                "out-of-support task geometry is evaluation-only and cannot "
                "be used for replay collection")
        self.collection_checkpoint_sha256 = ""
        if self.validation_collection:
            digest = hashlib.sha256()
            with checkpoint_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.collection_checkpoint_sha256 = digest.hexdigest()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_objective_version") != (
            PureDreamerTrainer.OBJECTIVE_VERSION
        ):
            raise RuntimeError(
                "serve_pure_dreamer_policy accepts only pure Dreamer checkpoints")
        if checkpoint.get("architecture_version") != (
            PureDreamerTrainer.ARCHITECTURE_VERSION
        ):
            raise RuntimeError(
                "serve_pure_dreamer_policy requires the current complete-state "
                "architecture")
        if checkpoint.get("control_transition_contract") != (
            PureDreamerTrainer.CONTROL_TRANSITION_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the acknowledged physical "
                "control-step contract")
        if checkpoint.get("rssm_initial_state_contract") != (
            PureDreamerTrainer.RSSM_INITIAL_STATE_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the learned categorical "
                "RSSM reset-state contract")
        if checkpoint.get("human_event_dense_reward_contract") != (
            PureDreamerTrainer.HUMAN_EVENT_DENSE_REWARD_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the current continuous-time "
                "Human reward contract")
        if checkpoint.get("human_kinematic_residual_contract") != (
            PureDreamerTrainer.HUMAN_KINEMATIC_RESIDUAL_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the audited Human kinematic "
                "residual contract")
        if checkpoint.get("human_geometry_sanitation_contract") != (
            PureDreamerTrainer.HUMAN_GEOMETRY_SANITATION_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the causal anthropometric "
                "Human geometry contract")
        if checkpoint.get("action_smoother_contract") != (
            PureDreamerTrainer.ACTION_SMOOTHER_CONTRACT_VERSION
        ) or checkpoint.get("action_smoother_parameterization") != (
            PureDreamerTrainer.ACTION_SMOOTHER_PARAMETERIZATION
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the reachable-gradient "
                "action parameterization contract")
        if not math.isclose(
            float(checkpoint.get(
                "human_root_max_velocity_residual_mps", float("nan"))),
            0.70, rel_tol=0.0, abs_tol=1.0e-9,
        ) or not math.isclose(
            float(checkpoint.get(
                "human_joint_max_velocity_residual_mps", float("nan"))),
            1.50, rel_tol=0.0, abs_tol=1.0e-9,
        ) or not math.isclose(
            float(checkpoint.get(
                "human_root_max_speed_mps", float("nan"))),
            2.50, rel_tol=0.0, abs_tol=1.0e-9,
        ) or not math.isclose(
            float(checkpoint.get(
                "human_joint_max_speed_mps", float("nan"))),
            6.00, rel_tol=0.0, abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint Human residual/absolute speed "
                "bounds differ from the audited architecture")
        if checkpoint.get("training_mode") != "pure_dreamer_joint":
            raise RuntimeError("checkpoint was not produced by joint pure Dreamer")
        if checkpoint.get("training_phase") != "joint_training" \
                or checkpoint.get("world_only_updates") != 0:
            raise RuntimeError(
                "checkpoint was produced by a staged/non-joint training phase")
        if checkpoint.get(
            "permutation_invariant_decision_entities_enabled"
        ) is not True:
            raise RuntimeError(
                "pure Dreamer checkpoint lacks masked permutation-invariant "
                "Human/obstacle encoders")
        if checkpoint.get("actor_std_parameterization") != "learned_bounded":
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the learned bounded Actor std")
        if not math.isclose(
            float(checkpoint.get(
                "actor_pre_tanh_mean_bound", float("nan"))),
            PURE_ACTOR_PRE_TANH_MEAN_BOUND,
            rel_tol=0.0, abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the bounded Actor mean")
        if checkpoint.get(
            "actor_pre_tanh_mean_parameterization"
        ) != "algebraic_sqrt":
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the reachable-gradient "
                "algebraic Actor mean")
        actor_capacity = int(checkpoint.get(
            "actor_human_physical_slots", 0))
        if actor_capacity != 23:
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the exact 23-slot local "
                "Human observation")
        if int(checkpoint.get("actor_input_dim", 0)) != 7229 or int(
            checkpoint.get("critic_input_dim", 0)
        ) != 7229:
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the exact shared 7229-D "
                "Actor/Critic decision state")
        if (
            checkpoint.get("decision_entity_encoder_contract")
            != CompleteDecisionStateEncoder.CONTRACT_VERSION
            or int(checkpoint.get("actor_encoded_dim", 0))
            != PURE_DECISION_ENCODED_DIM
            or int(checkpoint.get("critic_encoded_dim", 0))
            != PURE_DECISION_ENCODED_DIM
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the current entity-associated "
                "Actor/Critic encoding contract")
        critic_low, critic_high, critic_bins = PURE_CRITIC_RETURN_SUPPORT
        if (
            checkpoint.get("critic_return_distribution")
            != PURE_CRITIC_RETURN_DISTRIBUTION
            or not math.isclose(float(checkpoint.get(
                "critic_return_support_low", float("nan"))), critic_low,
                rel_tol=0.0, abs_tol=1.0e-9)
            or not math.isclose(float(checkpoint.get(
                "critic_return_support_high", float("nan"))), critic_high,
                rel_tol=0.0, abs_tol=1.0e-9)
            or int(checkpoint.get("critic_return_support_bins", 0))
            != critic_bins
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks the raw expected-return "
                "Critic distribution contract")
        if checkpoint.get("analytic_ego_dynamics_contract") != (
            load_analytic_ego_contract()
        ):
            raise RuntimeError(
                "checkpoint lacks the current train-only analytic Ego "
                "provenance contract")
        trainer_state = checkpoint.get("pure_dreamer_training_state")
        trainer_config = (
            trainer_state.get("config")
            if isinstance(trainer_state, dict) else None)
        if not isinstance(trainer_config, dict):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks its task objective config")
        update_counts: dict[str, int] = {}
        for field in (
            "world_update_count", "actor_update_count",
            "critic_update_count", "event_update_count",
        ):
            value = checkpoint.get(field)
            if (
                not isinstance(value, int) or isinstance(value, bool)
                or value < 0
            ):
                raise RuntimeError(
                    "pure Dreamer checkpoint lacks a valid joint-update "
                    f"counter: {field}")
            update_counts[field] = value
        if len(set(update_counts.values())) != 1:
            raise RuntimeError(
                "pure Dreamer checkpoint was not updated jointly by World, "
                "Actor, Critic and Event")
        actor_update_count = update_counts["actor_update_count"]
        if checkpoint.get("step") != actor_update_count:
            raise RuntimeError(
                "pure Dreamer checkpoint step differs from its joint clock")
        trainer_update_count = trainer_state.get("update_count")
        if trainer_update_count != actor_update_count:
            raise RuntimeError(
                "pure Dreamer trainer state differs from its joint clock")
        replay_task_contract = checkpoint.get("replay_task_contract")
        if actor_update_count > 0:
            if not isinstance(replay_task_contract, dict):
                raise RuntimeError(
                    "trained pure Dreamer checkpoint lacks its replay support "
                    "contract")
            support_fields = {
                "maximum_simultaneous_actor_humans": actor_capacity,
                "maximum_static_proxy_count": 256,
            }
            deployment_support: dict[str, int] = {}
            for field, capacity in support_fields.items():
                value = replay_task_contract.get(field)
                if (
                    not isinstance(value, int) or isinstance(value, bool)
                    or value < 0 or value > capacity
                ):
                    raise RuntimeError(
                        "trained pure Dreamer checkpoint has an invalid "
                        f"{field} replay support limit")
                deployment_support[field] = value
            self.empirical_maximum_supported_actor_humans = (
                deployment_support["maximum_simultaneous_actor_humans"])
            # Growing online replay must be able to encounter a later scene
            # with one more person than its initial prefill.  Capping the
            # collector at the empirical maximum makes that support impossible
            # to acquire and permanently converts every such episode to holds.
            # Collection may expand up to the explicit set-Actor capacity;
            # deterministic evaluation remains fail-closed at the checkpoint's
            # empirical support envelope.
            self.maximum_supported_actor_humans = (
                actor_capacity if self.collection else
                self.empirical_maximum_supported_actor_humans)
            self.maximum_supported_static_proxy_count = deployment_support[
                "maximum_static_proxy_count"]
            scene_contract = replay_task_contract.get(
                "fixed_scene_geometry_contract")
            if not isinstance(scene_contract, dict):
                raise RuntimeError(
                    "trained pure Dreamer checkpoint lacks its fixed-scene "
                    "geometry identity")
            try:
                reconstructed_scene_contract = (
                    fixed_task_scene_geometry_contract(
                        scene_contract["flight_bounds_xy"],
                        scene_contract["static_obstacle_aabbs_world"],
                        maximum_altitude_m=float(
                            scene_contract["maximum_altitude_m"]),
                    ))
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "trained pure Dreamer checkpoint has an invalid fixed-"
                    "scene geometry identity") from error
            if (
                scene_contract.get("version")
                != TASK_SCENE_GEOMETRY_CONTRACT_VERSION
                or scene_contract.get("sha256")
                != reconstructed_scene_contract["sha256"]
            ):
                raise RuntimeError(
                    "trained pure Dreamer checkpoint fixed-scene geometry "
                    "identity is inconsistent")
            self.task_scene_geometry_contract = reconstructed_scene_contract
            try:
                self.replay_fixed_goal_altitude_agl_m = float(
                    replay_task_contract["fixed_goal_altitude_agl_m"])
                self.replay_minimum_goal_altitude_agl_m = float(
                    replay_task_contract["minimum_goal_altitude_agl_m"])
                self.replay_maximum_goal_altitude_agl_m = float(
                    replay_task_contract["maximum_goal_altitude_agl_m"])
                self.replay_static_ground_z_world = float(
                    replay_task_contract["static_ground_z_world"])
                self.replay_sample_rate_hz = float(
                    replay_task_contract["sample_rate_hz"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "trained pure Dreamer checkpoint lacks its fixed task "
                    "support constants") from error
            action_support = replay_task_contract.get(
                "world_dynamics_action_support")
            if not isinstance(action_support, dict) \
                    or action_support.get("formal_ready") is not True:
                raise RuntimeError(
                    "trained pure Dreamer checkpoint lacks formal joint "
                    "action support")
            if int(reconstructed_scene_contract[
                "active_static_obstacle_count"
            ]) != self.maximum_supported_static_proxy_count:
                raise RuntimeError(
                    "trained pure Dreamer scene and static population support "
                    "disagree")
        else:
            # Smooth-random prefill has not fitted a population-count regime.
            # It may use architecture capacity; trained checkpoints switch to
            # empirical replay support even though entity weights are shared.
            self.maximum_supported_actor_humans = actor_capacity
            self.empirical_maximum_supported_actor_humans = 0
            self.maximum_supported_static_proxy_count = 256
            self.task_scene_geometry_contract = None
            self.replay_fixed_goal_altitude_agl_m = float("nan")
            self.replay_minimum_goal_altitude_agl_m = float("nan")
            self.replay_maximum_goal_altitude_agl_m = float("nan")
            self.replay_static_ground_z_world = float("nan")
            self.replay_sample_rate_hz = float("nan")
        self.task_goal_world_xy_support = np.asarray((
            (trainer_config.get("task_goal_world_x_min_m"),
             trainer_config.get("task_goal_world_x_max_m")),
            (trainer_config.get("task_goal_world_y_min_m"),
             trainer_config.get("task_goal_world_y_max_m")),
        ), np.float64)
        self.task_initial_world_xy_support = np.asarray((
            (trainer_config.get("task_initial_world_x_min_m"),
             trainer_config.get("task_initial_world_x_max_m")),
            (trainer_config.get("task_initial_world_y_min_m"),
             trainer_config.get("task_initial_world_y_max_m")),
        ), np.float64)
        if (
            not np.isfinite(self.task_goal_world_xy_support).all()
            or not np.isfinite(self.task_initial_world_xy_support).all()
            or bool((self.task_goal_world_xy_support[:, 0]
                     >= self.task_goal_world_xy_support[:, 1]).any())
            or bool((self.task_initial_world_xy_support[:, 0]
                     >= self.task_initial_world_xy_support[:, 1]).any())
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks fixed traversal regions")
        self.task_maximum_horizontal_speed_mps = float(
            trainer_config.get(
                "task_maximum_horizontal_speed_mps", float("nan")))
        self.task_goal_radius_m = float(
            trainer_config.get("goal_radius_m", float("nan")))
        self.task_crash_altitude_world_m = float(trainer_config.get(
            "task_crash_altitude_world_m", float("nan")))
        self.task_static_ground_z_world = float(trainer_config.get(
            "task_static_ground_z_world", float("nan")))
        action_adapter_contract = checkpoint.get("action_adapter_contract")
        self.task_target_altitude_agl_m = float(
            action_adapter_contract.get(
                "target_altitude_agl_m", float("nan"))
            if isinstance(action_adapter_contract, dict) else float("nan"))
        self.policy_step_duration_s = float(trainer_config.get(
            "imagination_dt_s", float("nan")))
        checkpoint_step_duration_s = float(checkpoint.get(
            "imagination_dt_s", float("nan")))
        if not (
            self.task_maximum_horizontal_speed_mps > 0.0
            and bool(torch.isfinite(torch.tensor(
                self.task_maximum_horizontal_speed_mps)))
            and self.task_goal_radius_m > 0.0
            and bool(torch.isfinite(torch.tensor(self.task_goal_radius_m)))
            and bool(torch.isfinite(torch.tensor(
                self.task_crash_altitude_world_m)))
            and math.isfinite(self.task_static_ground_z_world)
            and math.isfinite(self.task_target_altitude_agl_m)
            and math.isclose(
                self.task_target_altitude_agl_m,
                float(trainer_config.get("cruise_height_m", float("nan"))),
                rel_tol=0.0, abs_tol=1.0e-9)
            and self.policy_step_duration_s > 0.0
            and bool(torch.isfinite(torch.tensor(
                self.policy_step_duration_s)))
            and checkpoint_step_duration_s > 0.0
            and math.isclose(
                checkpoint_step_duration_s,
                self.policy_step_duration_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint lacks finite task constants")
        if actor_update_count > 0 and not (
            math.isclose(
                self.replay_fixed_goal_altitude_agl_m,
                self.task_target_altitude_agl_m,
                rel_tol=0.0, abs_tol=1.0e-9,
            )
            and math.isclose(
                self.replay_minimum_goal_altitude_agl_m,
                self.task_target_altitude_agl_m,
                rel_tol=0.0, abs_tol=1.0e-9,
            )
            and math.isclose(
                self.replay_maximum_goal_altitude_agl_m,
                self.task_target_altitude_agl_m,
                rel_tol=0.0, abs_tol=1.0e-9,
            )
            and math.isclose(
                self.replay_static_ground_z_world,
                self.task_static_ground_z_world,
                rel_tol=0.0, abs_tol=1.0e-9,
            )
            and math.isclose(
                self.replay_sample_rate_hz,
                1.0 / self.policy_step_duration_s,
                rel_tol=0.0, abs_tol=1.0e-9,
            )
        ):
            raise RuntimeError(
                "trained pure Dreamer replay support disagrees with its "
                "fixed task constants")
        if actor_update_count > 0:
            expected_goal_support = {
                "x_range_m": self.task_goal_world_xy_support[0].tolist(),
                "y_range_m": self.task_goal_world_xy_support[1].tolist(),
            }
            expected_initial_support = {
                "x_range_m": self.task_initial_world_xy_support[0].tolist(),
                "y_range_m": self.task_initial_world_xy_support[1].tolist(),
            }
            for field, expected in (
                ("task_goal_world_xy_support", expected_goal_support),
                ("task_initial_world_xy_support", expected_initial_support),
            ):
                support = replay_task_contract.get(field)
                if not isinstance(support, dict) or any(
                    support.get(name) != value
                    for name, value in expected.items()
                ):
                    raise RuntimeError(
                        "trained pure Dreamer replay support disagrees with "
                        f"its fixed traversal region: {field}")
        if actor_update_count > 0:
            assert isinstance(replay_task_contract, dict)
            minimum_interval_s = float(replay_task_contract.get(
                "minimum_nonterminal_control_interval_s", float("nan")))
            maximum_interval_s = float(replay_task_contract.get(
                "maximum_nonterminal_control_interval_s", float("nan")))
            interval_count = replay_task_contract.get(
                "ordinary_control_interval_count")
            if (
                not math.isfinite(minimum_interval_s)
                or not math.isfinite(maximum_interval_s)
                or minimum_interval_s <= 0.0
                or minimum_interval_s > maximum_interval_s
                or minimum_interval_s < 0.85 * self.policy_step_duration_s
                or maximum_interval_s > 1.15 * self.policy_step_duration_s
                or not isinstance(interval_count, int)
                or isinstance(interval_count, bool)
                or interval_count <= 0
            ):
                raise RuntimeError(
                    "trained pure Dreamer checkpoint has invalid empirical "
                    "control-period support")
            self.minimum_supported_control_interval_s = minimum_interval_s
            self.maximum_supported_control_interval_s = maximum_interval_s
        else:
            self.minimum_supported_control_interval_s = (
                0.85 * self.policy_step_duration_s)
            self.maximum_supported_control_interval_s = (
                1.15 * self.policy_step_duration_s)
        super().__init__(
            checkpoint_path,
            device,
            # The historical runtime audit expects imitation-policy metrics.
            # This entry point replaces that audit with the strict objective
            # checks above; it does not relax the action path below.
            allow_unsafe_policy=True,
            world_model_planner=False,
            r2_candidate_planner=False,
            shadow_imagination=False,
            # Standard Dreamer collection samples its learned Actor
            # distribution. Evaluation executes the deterministic mode.
            # Neither path enables the historical macro/candidate planner.
            online_exploration_scale=0.0,
            stochastic_actor=bool(self.collection),
            horizontal_altitude_hold=False,
        )
        if self.validation_collection:
            # The loaded model supplies only the versioned observation/action
            # adapter. Its Actor, RSSM state and values cannot affect the
            # prospective validation command.
            self.prefill_explorer = ProspectiveValidationExplorer()
            self.prefill_active = True
        if self.model.action_smoother is None or getattr(
            self.model.action_smoother, "parameterization", ""
        ) != PureDreamerTrainer.ACTION_SMOOTHER_PARAMETERIZATION:
            raise RuntimeError(
                "pure Dreamer runtime reconstructed the wrong action "
                "parameterization")
        if self.prefill_active and bool((
            np.asarray(self.prefill_explorer.target_limits, np.float32)
            <= 0.25
        ).any()):
            raise RuntimeError(
                "random-prefill target support cannot reach the learner's "
                "required +/-0.25 signed action extent")
        if not self.model.actor_explicit_human_geometry_enabled:
            raise RuntimeError(
                "pure Dreamer checkpoint/runtime disabled the explicit Human "
                "geometry used during Actor training")
        if not self.model.actor_authoritative_ego_token_enabled:
            raise RuntimeError(
                "pure Dreamer checkpoint/runtime disabled the authoritative "
                "Ego token used during Actor training")
        if not bool(self.model.actor.config.unified_policy):
            raise RuntimeError(
                "pure Dreamer checkpoint/runtime disabled the full-state "
                "unified Actor contract")
        if not bool(
            self.model.actor.config.permutation_invariant_entities
        ):
            raise RuntimeError(
                "pure Dreamer runtime disabled the entity-set encoder")
        if not bool(self.model.critic_full_state_enabled):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled the full-state Critic")
        if not bool(self.model.deterministic_evaluation_state_enabled):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled deterministic posterior "
                "state during evaluation")
        if not bool(self.model.direct_task_geometry_enabled):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled direct task geometry")
        if not bool(self.model.actor_task_physical_state_enabled) or int(
            self.model.actor_task_physical_obstacle_slots
        ) != 256:
            raise RuntimeError(
                "pure Dreamer checkpoint disabled complete exact task state")
        if not bool(self.model.task_memory_enabled) or not bool(
            self.model.direct_task_memory_enabled
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled deployable task memory")
        if not bool(
            self.model.transition_event.analytic_task_memory_events
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint learns non-Markov watchdog events")
        if not bool(self.model.transition_event.explicit_joint_kinematics):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled directional joint geometry")
        if not bool(
            self.model.transition_event.event_full_articulated_state
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled full articulated Event state")
        if not bool(
            self.model.transition_event.actor_full_learned_per_slot
        ) or int(
            self.model.transition_event.actor_full_learned_fields_per_slot
        ) != 116:
            raise RuntimeError(
                "pure Dreamer checkpoint pooled away per-person learned "
                "Human state")
        if not bool(self.model.encoder.ego_encoder.metric_scaling):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled metric Ego encoding")
        if not bool(
            self.model.encoder.human_encoder.metric_quality_enabled
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint disabled metric Human quality encoding")
        if not bool(
            self.model.prediction_heads.config.kinematic_joint_velocity_only
        ):
            raise RuntimeError(
                "pure Dreamer checkpoint/runtime disabled kinematic joint "
                "prediction used during Actor training")
        if bool(self.model.conservative_human_clearance_enabled):
            raise RuntimeError(
                "pure Dreamer runtime enabled the legacy learned/CV min return")
        self.model.transition_event.require_identity_human_probability_calibration()
        if self.prefill_active and not self.collection:
            raise RuntimeError(
                "evaluation refuses a random-prefill pure Dreamer checkpoint")
        if self.world_model_planner or self.r2_candidate_planner:
            raise RuntimeError("pure Actor runtime cannot enable a planner")
        # The compatibility wrapper builds historical optimizer/train-step
        # objects even in inference processes. Make accidental online updates
        # impossible and release their references; only ``model.act_step`` is
        # used below.
        for name in (
            "_optimizer", "_actor_optimizer", "_task_critic_optimizer",
            "_safety_critic_optimizer", "_risk_optimizer", "_scheduler",
            "_train_step",
        ):
            if hasattr(self.agent, name):
                setattr(self.agent, name, None)

    def reset(
        self, *, collection_scene_seed: int | None = None,
    ) -> dict[str, object]:
        """Reset both the inherited latent state and execution handshake."""
        response = super().reset(
            collection_scene_seed=collection_scene_seed)
        response["collection_policy_source"] = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.collection_checkpoint_sha256,
            "loaded_checkpoint_step": int(self.checkpoint_step),
            "evaluated_checkpoint_independent": bool(
                self.validation_collection),
        }
        self._last_issued_control_step_index: int | None = None
        self._last_issued_timestamp_s: float | None = None
        return response

    def _validate_control_transition(
        self, request: dict,
    ) -> tuple[int, float]:
        """Require one acknowledged physical control interval per RSSM step.

        A stale/missing observation makes the Isaac controller apply a hold
        without calling the policy.  A lost response can conversely advance
        the server while no returned action is applied.  Either case makes
        the recurrent ``prev_action`` cease to describe the physical MDP.
        Sequence and acknowledgement fields make those failures observable;
        timestamp bounds additionally reject a delayed callback that did not
        execute the intervening nominal 10-Hz steps.
        """
        step = request.get("control_step_index")
        acknowledged = request.get("previous_policy_response_step_index")
        if (
            not isinstance(step, int) or isinstance(step, bool)
            or not isinstance(acknowledged, int)
            or isinstance(acknowledged, bool)
        ):
            raise ValueError(
                "pure Dreamer requires integer control-step handshake fields")
        timestamp_s = float(request.get("timestamp_s", float("nan")))
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")

        previous_step = self._last_issued_control_step_index
        previous_timestamp = self._last_issued_timestamp_s
        if previous_step is None:
            if step != 0 or acknowledged != -1:
                raise RuntimeError(
                    "pure Dreamer first control step must be 0 with no prior "
                    "policy response")
            return step, timestamp_s
        if step != previous_step + 1:
            raise RuntimeError(
                "pure Dreamer control-step discontinuity: a physical hold or "
                "dropped control interval cannot reuse the recurrent state")
        if acknowledged != previous_step:
            raise RuntimeError(
                "pure Dreamer previous policy response was not acknowledged "
                "as physically applied")
        if previous_timestamp is None:
            raise RuntimeError("pure Dreamer timestamp handshake is corrupt")
        elapsed_s = timestamp_s - previous_timestamp
        minimum_s = self.minimum_supported_control_interval_s
        maximum_s = self.maximum_supported_control_interval_s
        if elapsed_s < minimum_s - 1.0e-9 \
                or elapsed_s > maximum_s + 1.0e-9:
            raise RuntimeError(
                "pure Dreamer control timestamp is outside the audited "
                f"single-step interval [{minimum_s:.6f}, {maximum_s:.6f}] s")
        return step, timestamp_s

    def step(self, request: dict) -> dict[str, object]:
        """Reject observations the versioned full-state Actor cannot encode."""
        control_step, timestamp_s = self._validate_control_transition(request)
        raw_ids = np.asarray(
            request.get("human_track_id", ()), np.int64).reshape(-1)
        observed_people = int(raw_ids.size)
        capacity = int(self.model.actor.config.human_physical_slots)
        if observed_people > capacity:
            raise RuntimeError(
                "pure Dreamer Human observation exceeds the complete-state "
                f"Actor capacity: {observed_people} > {capacity}")
        active_people = self.slots.prospective_active_count(
            timestamp_s, raw_ids)
        if active_people > self.maximum_supported_actor_humans:
            raise RuntimeError(
                "pure Dreamer Human state exceeds the checkpoint's trained "
                "replay support after lifecycle retention: "
                f"{active_people} > {self.maximum_supported_actor_humans}")
        static_obstacles = request.get("static_obstacle_aabbs_world", ())
        try:
            static_count = int(
                np.asarray(static_obstacles, np.float32).reshape(-1, 4).shape[0])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "pure Dreamer request has invalid static obstacle AABBs"
            ) from error
        if static_count > self.maximum_supported_static_proxy_count:
            raise RuntimeError(
                "pure Dreamer static geometry exceeds the checkpoint's "
                "trained replay support: "
                f"{static_count} > "
                f"{self.maximum_supported_static_proxy_count}")
        ego_state = request.get("ego_state")
        if ego_state is None:
            raise ValueError("pure Dreamer request lacks ego_state")
        ego = torch.as_tensor(ego_state, dtype=torch.float32).reshape(-1)
        if ego.numel() != 14 or not bool(torch.isfinite(ego).all()):
            raise ValueError("pure Dreamer request has invalid Ego14")
        horizontal_speed = torch.linalg.vector_norm(ego[3:5])
        if float(horizontal_speed) > (
            self.task_maximum_horizontal_speed_mps + 1.0e-5
        ):
            raise RuntimeError(
                "pure Dreamer Ego state exceeds the audited horizontal-speed "
                "contract")
        try:
            goal_radius = float(request["goal_radius_m"])
            crash_altitude = float(request["crash_altitude_m"])
            maximum_altitude = float(request["maximum_altitude_m"])
            boundary_z = float(request["boundary_position_world_z"])
            boundary_xy = np.asarray(
                request["boundary_position_world_xy"], np.float64).reshape(2)
            boundary_yaw = float(request["boundary_yaw_rad"])
            goal_position = np.asarray(
                request["goal_position"], np.float64).reshape(3)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "pure Dreamer request lacks numeric task constants") from error
        task_values = torch.tensor((
            goal_radius, crash_altitude, maximum_altitude, boundary_z,
            *boundary_xy.tolist(), boundary_yaw,
            *goal_position.tolist(),
        ))
        if not bool(torch.isfinite(task_values).all()):
            raise ValueError("pure Dreamer request has non-finite task constants")
        if self.task_scene_geometry_contract is not None:
            try:
                request_scene_contract = fixed_task_scene_geometry_contract(
                    request["boundary_flight_bounds_xy"],
                    static_obstacles,
                    maximum_altitude_m=maximum_altitude,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "pure Dreamer request lacks valid fixed-scene geometry"
                ) from error
            scene_is_training_equivalent = fixed_task_scene_geometry_equivalent(
                request_scene_contract,
                self.task_scene_geometry_contract,
            )
            if (
                not scene_is_training_equivalent
                and self.allow_out_of_support_task_evaluation
            ):
                # This explicit evaluation mode may change the Actor-visible
                # flight rectangle, but it must still be the same physical
                # Warehouse.  Rebuild the request identity with the training
                # bounds so the unchanged static AABB set and altitude remain
                # fail-closed.  Goal/initial spatial support is handled below.
                request_static_scene_contract = (
                    fixed_task_scene_geometry_contract(
                        self.task_scene_geometry_contract["flight_bounds_xy"],
                        static_obstacles,
                        maximum_altitude_m=maximum_altitude,
                    )
                )
                scene_is_training_equivalent = (
                    fixed_task_scene_geometry_equivalent(
                        request_static_scene_contract,
                        self.task_scene_geometry_contract,
                    )
                )
            if not scene_is_training_equivalent:
                raise RuntimeError(
                    "pure Dreamer request scene geometry is outside the "
                    "checkpoint's fixed-layout training support")
        relative_yaw = math.atan2(float(ego[12]), float(ego[13]))
        origin_yaw = boundary_yaw - relative_yaw
        cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
        current_episode_xy_world = np.asarray((
            cosine * float(ego[0]) - sine * float(ego[1]),
            sine * float(ego[0]) + cosine * float(ego[1]),
        ))
        episode_origin_world_xy = boundary_xy - current_episode_xy_world
        goal_delta_world_xy = np.asarray((
            cosine * float(goal_position[0])
            - sine * float(goal_position[1]),
            sine * float(goal_position[0])
            + cosine * float(goal_position[1]),
        ))
        goal_world_xy = episode_origin_world_xy + goal_delta_world_xy
        if not self.allow_out_of_support_task_evaluation and (
            bool((goal_world_xy < self.task_goal_world_xy_support[:, 0]
                  - 1.0e-6).any()) or bool((
                goal_world_xy > self.task_goal_world_xy_support[:, 1] + 1.0e-6
            ).any())
        ):
            raise RuntimeError(
                "pure Dreamer goal is outside the fixed traversal training "
                "region")
        if (
            not self.allow_out_of_support_task_evaluation
            and control_step == 0 and (
                bool((boundary_xy < self.task_initial_world_xy_support[:, 0]
                      - 0.25).any())
                or bool((
                    boundary_xy > self.task_initial_world_xy_support[:, 1]
                    + 0.25
                ).any())
            )
        ):
            raise RuntimeError(
                "pure Dreamer initial position is outside the fixed traversal "
                "training region")
        if abs(goal_radius - self.task_goal_radius_m) > 1.0e-6:
            raise RuntimeError(
                "pure Dreamer goal radius differs from the checkpoint MDP")
        if abs(
            crash_altitude - self.task_crash_altitude_world_m
        ) > 1.0e-6:
            raise RuntimeError(
                "pure Dreamer crash altitude differs from the checkpoint MDP")
        if maximum_altitude <= crash_altitude:
            raise ValueError(
                "pure Dreamer maximum altitude must exceed crash altitude")
        episode_origin_world_z = boundary_z - float(ego[2])
        goal_altitude_agl_m = (
            episode_origin_world_z + float(goal_position[2])
            - self.task_static_ground_z_world)
        if abs(
            goal_altitude_agl_m - self.task_target_altitude_agl_m
        ) > 1.0e-6:
            raise RuntimeError(
                "pure Dreamer is a fixed-altitude horizontal policy; goal "
                "altitude differs from its ActionAdapter target")
        if request.get("static_obstacle_geometry_use") != (
            "analytic_task_terminal_and_actor_proximity_cost"
        ):
            raise RuntimeError(
                "pure Dreamer deployment requires the same analytic static-"
                "terminal MDP used by Actor training")
        response = super().step(request)
        self._last_issued_control_step_index = control_step
        self._last_issued_timestamp_s = timestamp_s
        response["control_step_index"] = control_step
        return response

    def handle(self, request: dict) -> dict[str, object]:
        response = super().handle(request)
        if str(request.get("type", "")) == "status":
            # The launcher probes the exact loaded task before starting Isaac.
            # A trained checkpoint must not be probed with an invented empty
            # scene, because the Actor directly observes the fixed AABB set.
            response["fixed_scene_geometry_contract"] = (
                None if self.task_scene_geometry_contract is None
                else dict(self.task_scene_geometry_contract)
            )
            response["out_of_support_task_evaluation_enabled"] = bool(
                self.allow_out_of_support_task_evaluation)
        return response


class PureActorServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9775)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reload-checkpoint-on-reset", action="store_true",
        help="Adopt an atomically replaced checkpoint only between episodes.",
    )
    collection_group = parser.add_mutually_exclusive_group()
    collection_group.add_argument(
        "--collection", action="store_true",
        help=(
            "Standard online-Dreamer behavior: smooth-random checkpoint "
            "prefill, then samples from the exact Actor distribution used "
            "in imagination. Evaluation remains deterministic."),
    )
    collection_group.add_argument(
        "--validation-collection", action="store_true",
        help=(
            "Collect the prospectively frozen Actor-independent validation "
            "distribution, including real per-axis saturation plateaus."),
    )
    parser.add_argument(
        "--allow-out-of-support-task-evaluation",
        action="store_true",
        help=(
            "Evaluation only: permit different flight/goal/initial spatial "
            "support while still requiring the checkpoint's exact physical "
            "static-obstacle scene and all non-spatial MDP contracts."),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.validation_collection and args.reload_checkpoint_on_reset:
        raise RuntimeError(
            "prospective validation collection forbids checkpoint reload; "
            "the episode-phase schedule must remain one immutable run")
    runtime_factory = partial(
        PureActorRuntime,
        collection=args.collection,
        validation_collection=args.validation_collection,
        allow_out_of_support_task_evaluation=(
            args.allow_out_of_support_task_evaluation),
    )
    runtime = (
        ReloadableFactorizedPolicyRuntime(
            checkpoint,
            args.device,
            runtime_factory=runtime_factory,
        )
        if args.reload_checkpoint_on_reset
        else runtime_factory(checkpoint, args.device)
    )
    with PureActorServer((args.host, args.port), PolicyHandler) as server:
        server.runtime = runtime  # type: ignore[attr-defined]
        print(
            f"PURE_DREAMER_ACTOR_READY protocol={PROTOCOL} host={args.host} "
            f"port={args.port} device={args.device} checkpoint={checkpoint} "
            f"step={runtime.checkpoint_step} smoothing="
            f"{int(runtime.model.action_smoother is not None)} planner=0 "
            f"collection={int(args.collection)} stochastic_actor="
            f"{int(runtime.stochastic_actor)} prefill="
            f"{int(runtime.prefill_active)} "
            f"validation_collection={int(args.validation_collection)} "
            "out_of_support_task_evaluation="
            f"{int(args.allow_out_of_support_task_evaluation)} "
            f"reload_on_reset={int(args.reload_checkpoint_on_reset)}",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
