"""Transition event and human-safety heads.

The current Human event uses an explicit per-person geometry readout and the
literal survival product over visible people plus a separately supervised
unobserved-person competing risk. Metric geometry is responsible for learning
negligible visible-slot hazard for irrelevant distant people; crowd count is
not erased by normalization, and occluded contacts are not mislabeled as a
visible person's risk.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from modules.se2_relative_dynamics import (
    DEPLOYABLE_DRONE_COLLISION_RADIUS_M,
    coco12_collision_spheres,
)


TRANSITION_EVENT_KEYS = (
    "continue",
    "human_collision",
    "static_collision",
    "reached_goal",
    "other_task_terminal",
)
CONTINUE_INDEX = 0
HUMAN_COLLISION_INDEX = 1
NON_GOAL_EVENT_KEYS = (
    "continue",
    "human_collision",
    "static_collision",
    "hard_failure",
    "stuck_timeout",
)
ANALYTIC_TASK_NON_HUMAN_EVENT_KEYS = (
    "continue",
    "static_collision",
)
LEGACY_NON_HUMAN_EVENT_KEYS = (
    "continue",
    "static_collision",
    "hard_failure",
    "stuck_timeout",
)


class TransitionEventHead(nn.Module):
    """Closed competing-risk model with explicit per-person geometry.

    Human collision is composed as a noisy-OR survival product over valid
    Human slots and a named unobserved-person residual hazard. A separate
    global network predicts the conditional outcome given that no
    Human collision occurred.  The resulting five probabilities are closed by
    construction and are consumed directly as probabilities/log-probabilities;
    they are never passed through a second softmax.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        *,
        ego_feat_dim: int | None = None,
        human_feat_dim: int | None = None,
        human_root_dim: int = 10,
        human_quality_dim: int = 7,
        explicit_human_geometry: bool = False,
        geometry_topk_physical_slots: int = 0,
        explicit_joint_kinematics: bool = False,
        explicit_human_presence_physical: bool = False,
        analytic_task_memory_events: bool = False,
        actor_full_state_slots: int = 0,
        actor_full_fields_per_slot: int | None = None,
        actor_full_learned_per_slot: bool = False,
        event_full_articulated_state: bool = False,
    ) -> None:
        super().__init__()
        input_dim = int(feature_dim) + int(action_dim)
        hidden_dim = int(hidden_dim)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim, action_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(TRANSITION_EVENT_KEYS)),
        )
        self.explicit_human_geometry = bool(explicit_human_geometry)
        self.explicit_joint_kinematics = bool(explicit_joint_kinematics)
        self.explicit_human_presence_physical = bool(
            explicit_human_presence_physical)
        self.analytic_task_memory_events = bool(
            analytic_task_memory_events)
        if self.explicit_joint_kinematics and not self.explicit_human_geometry:
            raise ValueError(
                "explicit joint kinematics requires explicit Human geometry")
        if (
            self.explicit_human_presence_physical
            and not self.explicit_joint_kinematics
        ):
            raise ValueError(
                "physical Human presence requires explicit joint kinematics")
        self.non_goal_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(
                (ANALYTIC_TASK_NON_HUMAN_EVENT_KEYS
                 if self.analytic_task_memory_events else
                 LEGACY_NON_HUMAN_EVENT_KEYS)
                if self.explicit_human_geometry
                else NON_GOAL_EVENT_KEYS)),
        )
        self.geometry_dim = int(feature_dim)
        # The Actor and Critic must not have to recover signed, metric Human
        # geometry from a token trained mainly by left/right-symmetric scalar
        # collision labels.  Reserve a small prefix for an exact physical
        # summary and leave the remaining dimensions learned.  This is a
        # state representation, not an avoidance rule: Dreamer's imagined
        # return still decides what action to take from it.
        self.geometry_slot_physical_dim = min(
            12 if self.explicit_joint_kinematics else 8,
            self.geometry_dim,
        )
        self.geometry_topk_physical_slots = int(
            geometry_topk_physical_slots)
        if self.geometry_topk_physical_slots < 0:
            raise ValueError("geometry top-k slot count must be non-negative")
        self.geometry_physical_fields_per_slot = (
            12 if self.explicit_joint_kinematics else 6)
        pooled_physical_dim = (
            self.geometry_physical_fields_per_slot
            * self.geometry_topk_physical_slots)
        if pooled_physical_dim > self.geometry_dim:
            raise ValueError(
                "top-k Human physical geometry exceeds decision feature size")
        # This prefix is copied exactly into the Critic's decision state.
        self.geometry_physical_dim = (
            pooled_physical_dim if pooled_physical_dim > 0
            else self.geometry_slot_physical_dim)
        ego_feat_dim = int(feature_dim if ego_feat_dim is None else ego_feat_dim)
        human_feat_dim = int(
            feature_dim if human_feat_dim is None else human_feat_dim)
        self.human_root_dim = int(human_root_dim)
        self.human_quality_dim = int(human_quality_dim)
        self.actor_full_state_slots = int(actor_full_state_slots)
        if self.actor_full_state_slots < 0:
            raise ValueError("Actor full-state slot count must be non-negative")
        # Per person: Root10, Quality7, lifecycle presence, signed clearance,
        # then COCO12 x [valid, xyz, vxyz].  Keep this formula tied to the
        # configured root/quality widths so the Actor construction cannot
        # silently disagree with the geometry producer.
        expected_actor_fields = (
            self.human_root_dim + self.human_quality_dim + 2 + 12 * 7)
        self.actor_full_fields_per_slot = int(
            expected_actor_fields
            if actor_full_fields_per_slot is None
            else actor_full_fields_per_slot)
        if (
            self.actor_full_state_slots > 0
            and self.actor_full_fields_per_slot != expected_actor_fields
        ):
            raise ValueError(
                "Actor full Human field layout must be root+quality+2+12x7")
        self.actor_full_learned_per_slot = bool(
            actor_full_learned_per_slot)
        self.actor_full_learned_fields_per_slot = (
            self.geometry_dim - self.geometry_slot_physical_dim
            if self.actor_full_learned_per_slot else 0)
        if (
            self.actor_full_learned_per_slot
            and self.actor_full_state_slots <= 0
        ):
            raise ValueError(
                "per-slot learned Actor state requires full-state slots")
        self.actor_full_state_dim = (
            self.actor_full_state_slots * (
                self.actor_full_fields_per_slot
                + self.actor_full_learned_fields_per_slot)
            + (0 if self.actor_full_learned_per_slot else self.geometry_dim))
        self.event_full_articulated_state = bool(
            event_full_articulated_state)
        self.event_per_human_exact_dim = (
            expected_actor_fields
            if self.actor_full_state_slots > 0
            and self.event_full_articulated_state else 0)
        self.geometry_network = (
            nn.Sequential(
                nn.Linear(
                    human_feat_dim + self.human_root_dim
                    + self.human_quality_dim + 1
                    + (6 if self.explicit_joint_kinematics else 0),
                    hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.geometry_dim),
                nn.LayerNorm(self.geometry_dim),
            ) if self.explicit_human_geometry else nn.Identity())
        self.per_human_network = (
            nn.Sequential(
                nn.Linear(
                    int(feature_dim) + ego_feat_dim + self.geometry_dim
                    + self.event_per_human_exact_dim
                    + int(action_dim),
                    hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ) if self.explicit_human_geometry else nn.Identity())
        # A contact partner is not always present in the deployable perception
        # slots.  Model that irreducible POMDP/occlusion risk explicitly rather
        # than assigning its positive label to an unrelated visible person.
        # Its probability is composed with visible per-person survival below;
        # this is a named competing risk, not a policy gate or hard override.
        self.unobserved_human_network = (
            nn.Sequential(
                nn.Linear(
                    int(feature_dim) + ego_feat_dim + self.geometry_dim
                    + int(action_dim),
                    hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ) if self.explicit_human_geometry else nn.Identity())
        # Post-hoc factual calibration is deliberately a tiny, independently
        # fitted state.  Buffers (rather than optimizer-owned Parameters)
        # preserve checkpoint/deployment identity while preventing balanced
        # ranking losses from silently changing probability semantics.
        self.register_buffer(
            "human_calibration_log_temperature", torch.zeros(()))
        self.register_buffer("human_calibration_bias", torch.zeros(()))
        self.register_buffer("human_calibration_count_bias", torch.zeros(()))
        self.register_buffer(
            "human_calibration_shrinkage_weight", torch.ones(()))
        self.register_buffer(
            "human_calibration_prior_probability", torch.zeros(()))
        self.register_buffer(
            "human_calibration_fitted", torch.zeros((), dtype=torch.bool))

    def set_human_probability_calibration(
        self, *, log_temperature: float, bias: float, count_bias: float,
        shrinkage_weight: float = 1.0,
        prior_probability: float = 0.0,
    ) -> None:
        """Install a frozen factual calibration map.

        ``shrinkage_weight=1`` is exactly the legacy three-parameter map.
        Smaller values conservatively blend its probability with a frozen
        calibration-split prevalence selected without validation leakage.
        """
        values = (
            float(log_temperature), float(bias), float(count_bias),
            float(shrinkage_weight), float(prior_probability),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Human Event calibration values must be finite")
        if not math.log(0.25) <= values[0] <= math.log(4.0):
            raise ValueError("Human Event calibration temperature is out of range")
        if not 0.0 <= values[3] <= 1.0:
            raise ValueError("Human Event calibration shrinkage is out of range")
        if not 0.0 <= values[4] <= 1.0:
            raise ValueError("Human Event calibration prior is out of range")
        self.human_calibration_log_temperature.fill_(values[0])
        self.human_calibration_bias.fill_(values[1])
        self.human_calibration_count_bias.fill_(values[2])
        self.human_calibration_shrinkage_weight.fill_(values[3])
        self.human_calibration_prior_probability.fill_(values[4])
        self.human_calibration_fitted.fill_(True)

    def require_identity_human_probability_calibration(self) -> None:
        """Require the raw hazard used while jointly training the Actor.

        A post-hoc probability map changes both imagined terminal reward and
        continuation.  Installing one after Actor/Critic training would thus
        deploy a different MDP objective from the one the policy optimized.
        Current pure Dreamer trains the cause hazards with natural replay NLL
        and validates those raw probabilities on untouched episodes, so its
        training, audit and runtime paths all require this exact identity map.
        """
        expected = (
            (self.human_calibration_log_temperature, 0.0),
            (self.human_calibration_bias, 0.0),
            (self.human_calibration_count_bias, 0.0),
            (self.human_calibration_shrinkage_weight, 1.0),
            (self.human_calibration_prior_probability, 0.0),
        )
        if bool(self.human_calibration_fitted) or any(
            not bool(torch.isfinite(value).all())
            or not bool(torch.equal(value, value.new_tensor(target)))
            for value, target in expected
        ):
            raise RuntimeError(
                "pure Dreamer requires the identity Human Event calibration; "
                "post-hoc hazard maps change the Actor's optimized return")

    @staticmethod
    def _apply_network(
        network: nn.Sequential,
        value: torch.Tensor,
        *,
        detach_parameters: bool,
    ) -> torch.Tensor:
        if not detach_parameters:
            return network(value)
        for layer in network:
            if isinstance(layer, nn.Linear):
                value = F.linear(
                    value,
                    layer.weight.detach(),
                    None if layer.bias is None else layer.bias.detach(),
                )
            else:
                value = layer(value)
        return value

    def forward(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> dict[str, torch.Tensor]:
        if feature.shape[:-1] != action.shape[:-1]:
            raise ValueError("feature and action leading dimensions must match")
        value = torch.cat((feature, action), dim=-1)
        logits = self._apply_network(
            self.network, value, detach_parameters=detach_parameters)
        probability = logits.softmax(dim=-1)
        result = {
            "logits": logits,
            "probability": probability,
            "continue_probability": probability[..., CONTINUE_INDEX:CONTINUE_INDEX + 1],
            "human_collision_probability": probability[
                ..., HUMAN_COLLISION_INDEX:HUMAN_COLLISION_INDEX + 1],
        }
        for index, key in enumerate(TRANSITION_EVENT_KEYS):
            result[f"event_probability/{key}"] = probability[..., index:index + 1]
        return result

    def forward_non_goal(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        ego_feature: torch.Tensor | None = None,
        human_feature: torch.Tensor | None = None,
        human_root: torch.Tensor | None = None,
        human_quality: torch.Tensor | None = None,
        human_mask: torch.Tensor | None = None,
        human_joint_clearance: torch.Tensor | None = None,
        human_joints_body: torch.Tensor | None = None,
        human_joint_velocity_body: torch.Tensor | None = None,
        human_joint_mask: torch.Tensor | None = None,
        human_presence: torch.Tensor | None = None,
        detach_parameters: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return the five-class event distribution used by fresh Dreamer.

        This is a separate calibrated output branch. The legacy five-class
        branch remains available through :meth:`forward` for old checkpoints;
        its reached-goal/other logits are never reinterpreted. Reached-goal in
        the fresh objective is composed analytically from predicted next Ego.
        """
        if feature.shape[:-1] != action.shape[:-1]:
            raise ValueError("feature and action leading dimensions must match")
        value = torch.cat((feature, action), dim=-1)
        global_logits = self._apply_network(
            self.non_goal_network,
            value,
            detach_parameters=detach_parameters,
        )
        if not self.explicit_human_geometry:
            probability = global_logits.softmax(-1)
            log_probability = global_logits.log_softmax(-1)
            return {
                "logits": global_logits,
                "log_probability": log_probability,
                "probability": probability,
                "continue_probability": probability[..., 0:1],
                "human_collision_probability": probability[..., 1:2],
                **{
                    f"event_probability/{key}": probability[
                        ..., index:index + 1]
                    for index, key in enumerate(NON_GOAL_EVENT_KEYS)
                },
            }
        global_log_probability = global_logits.log_softmax(-1)
        global_probability = global_log_probability.exp()

        human = self.forward_human_hazard(
            feature,
            action,
            ego_feature=ego_feature,
            human_feature=human_feature,
            human_root=human_root,
            human_quality=human_quality,
            human_mask=human_mask,
            human_joint_clearance=human_joint_clearance,
            human_joints_body=human_joints_body,
            human_joint_velocity_body=human_joint_velocity_body,
            human_joint_mask=human_joint_mask,
            human_presence=human_presence,
            detach_parameters=detach_parameters,
        )
        human_probability = human["human_collision_probability"].squeeze(-1)
        no_human_probability = human["human_survival_probability"].squeeze(-1)
        log_no_human = human["human_log_survival_probability"].squeeze(-1)

        probability = feature.new_zeros((*feature.shape[:-1], 5))
        probability[..., 0] = no_human_probability * global_probability[..., 0]
        probability[..., 1] = human_probability
        if self.analytic_task_memory_events:
            probability[..., 2] = (
                no_human_probability * global_probability[..., 1]
            )
        else:
            probability[..., 2:] = (
                no_human_probability[..., None]
                * global_probability[..., 1:]
            )
        # Compute class log-probabilities from the chain rule rather than from
        # a second softmax. Clamp only the logarithm, never the probabilities.
        log_probability = feature.new_empty((*feature.shape[:-1], 5))
        log_probability[..., 0] = log_no_human + global_log_probability[..., 0]
        log_probability[..., 1] = torch.log(
            human_probability.clamp_min(1.0e-12))
        if self.analytic_task_memory_events:
            log_probability[..., 2] = (
                log_no_human + global_log_probability[..., 1]
            )
            # Crash and stuck are retained as compatibility output slots but
            # have no learned mass. Their recorder definitions depend on
            # explicit task memory and are composed analytically.
            log_probability[..., 3:] = -torch.inf
        else:
            log_probability[..., 2:] = (
                log_no_human[..., None] + global_log_probability[..., 1:]
            )
        result = {
            # Compatibility alias. These are normalized log-probabilities,
            # not unconstrained logits and must not be softmaxed again.
            "logits": log_probability,
            "log_probability": log_probability,
            "probability": probability,
            "continue_probability": probability[..., 0:1],
            "human_collision_probability": probability[..., 1:2],
            "global_non_human_logits": global_logits,
            "global_non_human_probability": global_probability,
            "slot_hazard": human["slot_hazard"],
            "observed_human_collision_probability": human[
                "observed_human_collision_probability"],
            "unobserved_human_collision_probability": human[
                "unobserved_human_collision_probability"],
            "unobserved_human_hazard_logit": human[
                "unobserved_human_hazard_logit"],
            "slot_hazard_logit": human["slot_hazard_logit"],
            "slot_geometry_token": human["slot_geometry_token"],
            "human_geometry_pool": human["human_geometry_pool"],
            "human_presence": human["human_presence"],
            "human_proximity_weight": human["human_proximity_weight"],
        }
        for index, key in enumerate(NON_GOAL_EVENT_KEYS):
            result[f"event_probability/{key}"] = probability[
                ..., index:index + 1]
        return result

    def forward_human_hazard(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        ego_feature: torch.Tensor,
        human_feature: torch.Tensor,
        human_root: torch.Tensor,
        human_quality: torch.Tensor,
        human_mask: torch.Tensor,
        human_joint_clearance: torch.Tensor | None = None,
        human_joints_body: torch.Tensor | None = None,
        human_joint_velocity_body: torch.Tensor | None = None,
        human_joint_mask: torch.Tensor | None = None,
        human_presence: torch.Tensor | None = None,
        detach_parameters: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return the v20 binary Human transition hazard.

        This is the only learned Human terminal probability consumed by the v20
        Actor/Critic objective.  The legacy global non-Human classifier is not
        evaluated here, so zero-sample static/stuck classes cannot alter Human
        survival or imagined returns.

        The returned probability is a closed noisy-OR over visible per-person
        hazards and one separately supervised unobserved-person hazard.
        Absolute calibration is owned by natural-distribution Human labels;
        geometry supplies visible-slot attribution, while the residual branch
        owns only contacts whose recorded partner is absent from Actor state.
        """
        if not self.explicit_human_geometry:
            raise RuntimeError(
                "v20 Human hazard requires explicit per-Human geometry")
        if feature.shape[:-1] != action.shape[:-1]:
            raise ValueError("feature and action leading dimensions must match")
        context = self._human_context(
            feature,
            action,
            ego_feature=ego_feature,
            human_feature=human_feature,
            human_root=human_root,
            human_quality=human_quality,
            human_mask=human_mask,
            human_joint_clearance=human_joint_clearance,
            human_joints_body=human_joints_body,
            human_joint_velocity_body=human_joint_velocity_body,
            human_joint_mask=human_joint_mask,
            human_presence=human_presence,
            detach_parameters=detach_parameters,
        )
        slot_hazard = context["slot_hazard"]
        slot_mask = context["slot_mask"]
        presence = context["presence"]
        # Collision means contact with *any* person.  The mathematically closed
        # aggregation is therefore a noisy-OR survival product, not the old
        # proximity-normalized geometric mean (which assigned two equally
        # dangerous people the same risk as one).  Far-person hazards must be
        # learned near zero from their explicit metric geometry; population
        # count is not allowed to disappear from the event semantics.
        effective_hazard = (
            presence * slot_hazard
        ).clamp(0.0, 1.0 - 1.0e-6)
        log_survival_slot = torch.where(
            slot_mask,
            torch.log1p(-effective_hazard),
            torch.zeros_like(effective_hazard),
        )
        proximity_weight = context["proximity_weight"]
        observed_log_survival = log_survival_slot.sum(-1, keepdim=True)
        observed_raw_survival = observed_log_survival.exp()
        observed_raw_hazard = (-torch.expm1(
            observed_log_survival)).clamp(0.0, 1.0)
        unobserved_input = torch.cat((
            feature,
            ego_feature,
            context["human_geometry_pool"],
            action,
        ), -1)
        unobserved_logit = self._apply_network(
            self.unobserved_human_network,
            unobserved_input,
            detach_parameters=detach_parameters,
        )
        unobserved_hazard = torch.sigmoid(unobserved_logit).clamp(
            0.0, 1.0 - 1.0e-6)
        log_survival = (
            observed_log_survival + torch.log1p(-unobserved_hazard))
        raw_survival = log_survival.exp()
        raw_hazard = (-torch.expm1(log_survival)).clamp(0.0, 1.0)
        effective_count = torch.where(
            slot_mask, presence, torch.zeros_like(presence)).sum(
                -1, keepdim=True)
        raw_logit = torch.logit(raw_hazard.clamp(1.0e-8, 1.0 - 1.0e-8))
        temperature = self.human_calibration_log_temperature.exp().clamp(
            0.25, 4.0)
        calibrated_logit = (
            raw_logit / temperature
            + self.human_calibration_bias
            + self.human_calibration_count_bias
            * torch.log1p(effective_count)
        )
        calibrated_hazard = torch.sigmoid(calibrated_logit)
        hazard = (
            self.human_calibration_shrinkage_weight * calibrated_hazard
            + (1.0 - self.human_calibration_shrinkage_weight)
            * self.human_calibration_prior_probability
        )
        survival = 1.0 - hazard
        return {
            "human_collision_probability": hazard,
            "human_survival_probability": survival,
            "human_log_survival_probability": torch.log1p(-hazard),
            "raw_human_collision_probability": raw_hazard,
            "raw_human_survival_probability": raw_survival,
            "human_effective_count": effective_count,
            "observed_human_collision_probability": observed_raw_hazard,
            "observed_human_survival_probability": observed_raw_survival,
            "unobserved_human_collision_probability": unobserved_hazard,
            "unobserved_human_hazard_logit": unobserved_logit,
            "slot_hazard": slot_hazard,
            "slot_hazard_logit": context["slot_hazard_logit"],
            "slot_geometry_token": context["slot_geometry_token"],
            "human_geometry_pool": context["human_geometry_pool"],
            "human_presence": presence,
            "human_proximity_weight": proximity_weight,
        }

    def _human_context(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        ego_feature: torch.Tensor | None,
        human_feature: torch.Tensor | None,
        human_root: torch.Tensor | None,
        human_quality: torch.Tensor | None,
        human_mask: torch.Tensor | None,
        human_joint_clearance: torch.Tensor | None,
        human_joints_body: torch.Tensor | None,
        human_joint_velocity_body: torch.Tensor | None,
        human_joint_mask: torch.Tensor | None,
        human_presence: torch.Tensor | None,
        detach_parameters: bool,
    ) -> dict[str, torch.Tensor]:
        supplied = (
            ego_feature, human_feature, human_root, human_quality, human_mask)
        if all(value is None for value in supplied):
            empty_shape = (*feature.shape[:-1], 0)
            return {
                "slot_hazard": feature.new_zeros(empty_shape),
                "slot_hazard_logit": feature.new_zeros(empty_shape),
                "slot_mask": torch.zeros(
                    empty_shape, dtype=torch.bool, device=feature.device),
                "presence": feature.new_zeros(empty_shape),
                "slot_geometry_token": feature.new_zeros(
                    (*empty_shape, self.geometry_dim)),
                "human_geometry_pool": feature.new_zeros(
                    (*feature.shape[:-1], self.geometry_dim)),
                "proximity_weight": feature.new_zeros(empty_shape),
            }
        if any(value is None for value in supplied):
            raise ValueError(
                "per-human Event context requires ego/human features, root, "
                "quality, and mask together")
        assert ego_feature is not None and human_feature is not None
        assert human_root is not None and human_quality is not None
        assert human_mask is not None
        leading = feature.shape[:-1]
        if ego_feature.shape[:-1] != leading:
            raise ValueError("ego_feature must match Event leading dimensions")
        if human_feature.shape[:-2] != leading:
            raise ValueError("human_feature must be [...,N,D]")
        slot_shape = human_feature.shape[:-1]
        if human_root.shape[:-1] != slot_shape or human_quality.shape[:-1] != slot_shape:
            raise ValueError("human root/quality must match Human slots")
        if human_mask.shape != slot_shape:
            raise ValueError("human_mask must match Human slots")
        if human_root.shape[-1] > self.human_root_dim:
            human_root = human_root[..., :self.human_root_dim]
        elif human_root.shape[-1] < self.human_root_dim:
            human_root = F.pad(
                human_root, (0, self.human_root_dim - human_root.shape[-1]))
        if human_quality.shape[-1] > self.human_quality_dim:
            human_quality = human_quality[..., :self.human_quality_dim]
        elif human_quality.shape[-1] < self.human_quality_dim:
            human_quality = F.pad(
                human_quality,
                (0, self.human_quality_dim - human_quality.shape[-1]))
        geometry_context = self.human_geometry_pool(
            human_feature, human_root, human_quality, human_mask,
            human_joint_clearance=human_joint_clearance,
            human_joints_body=human_joints_body,
            human_joint_velocity_body=human_joint_velocity_body,
            human_joint_mask=human_joint_mask,
            human_presence=human_presence,
            detach_parameters=detach_parameters,
        )
        slot_mask = geometry_context["slot_mask"]
        presence = geometry_context["presence"]
        geometry = geometry_context["slot_geometry_token"]
        expanded_feature = feature.unsqueeze(-2).expand(*slot_shape, feature.shape[-1])
        expanded_ego = ego_feature.unsqueeze(-2).expand(*slot_shape, ego_feature.shape[-1])
        expanded_action = action.unsqueeze(-2).expand(*slot_shape, action.shape[-1])
        hazard_input = torch.cat((
            expanded_feature,
            expanded_ego,
            geometry,
            *(() if self.event_per_human_exact_dim == 0 else (
                geometry_context["actor_slot_state"],)),
            expanded_action,
        ), -1)
        hazard_logit = self._apply_network(
            self.per_human_network,
            hazard_input,
            detach_parameters=detach_parameters,
        ).squeeze(-1)
        hazard = torch.sigmoid(hazard_logit).masked_fill(~slot_mask, 0.0)

        return {
            "slot_hazard": hazard,
            "slot_hazard_logit": hazard_logit,
            "slot_mask": slot_mask,
            "presence": presence,
            "slot_geometry_token": geometry,
            "human_geometry_pool": geometry_context["human_geometry_pool"],
            "proximity_weight": geometry_context["proximity_weight"],
        }

    def human_geometry_pool(
        self,
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
        detach_parameters: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode explicit slot geometry and proximity-pool it for the Actor."""
        slot_shape = human_feature.shape[:-1]
        if human_root.shape[:-1] != slot_shape or human_quality.shape[:-1] != slot_shape:
            raise ValueError("human root/quality must match Human slots")
        if human_mask.shape != slot_shape:
            raise ValueError("human_mask must match Human slots")
        nearest_joint_fields = None
        actor_joint_state = None
        actor_slot_state = None
        if self.explicit_joint_kinematics:
            if (
                human_joints_body is None
                or human_joint_velocity_body is None
                or human_joint_mask is None
            ):
                raise ValueError(
                    "explicit joint geometry requires positions, velocities, "
                    "and masks")
            if human_joints_body.shape[:-2] != slot_shape:
                raise ValueError("Human joint positions must match slots")
            if human_joint_velocity_body.shape != human_joints_body.shape:
                raise ValueError("Human joint velocities must match positions")
            if human_joint_mask.shape != human_joints_body.shape[:-1]:
                raise ValueError("Human joint mask must match positions")
            spheres, sphere_velocity, sphere_mask, sphere_radii = (
                coco12_collision_spheres(
                    human_joints_body,
                    human_joint_mask,
                    human_joint_velocity_body,
                ))
            sphere_distance = torch.linalg.vector_norm(spheres, dim=-1)
            sphere_signed_gap = (
                sphere_distance
                - DEPLOYABLE_DRONE_COLLISION_RADIUS_M
                - sphere_radii
            ).masked_fill(~sphere_mask, torch.inf)
            # Clearance is defined by minimum surface gap, not centre distance.
            # Select xyz/velocity with the same criterion so those directional
            # fields describe the articulated sphere that owns the clearance.
            nearest_sphere = sphere_signed_gap.argmin(-1)
            gather_index = nearest_sphere[..., None, None].expand(
                *nearest_sphere.shape, 1, 3)
            nearest_position = spheres.gather(-2, gather_index).squeeze(-2)
            nearest_velocity = sphere_velocity.gather(
                -2, gather_index).squeeze(-2)
            nearest_valid = sphere_mask.any(-1)
            nearest_position = nearest_position.masked_fill(
                ~nearest_valid[..., None], 0.0)
            nearest_velocity = nearest_velocity.masked_fill(
                ~nearest_valid[..., None], 0.0)
            nearest_joint_fields = torch.cat((
                (nearest_position / nearest_position.new_tensor(
                    (6.0, 6.0, 3.0))).clamp(-5.0, 5.0),
                (nearest_velocity / nearest_velocity.new_tensor(
                    (3.0, 3.0, 3.0))).clamp(-5.0, 5.0),
            ), -1)
            # The nearest sphere is sufficient for a scalar clearance head but
            # not for a full-state Actor: a hand/foot that is not currently
            # nearest can determine which side is safe later in the rollout.
            # Preserve every deployable COCO12 joint and its causal velocity,
            # together with an explicit validity bit. Invalid coordinates are
            # zero only because their validity field is also zero.
            joint_valid = human_joint_mask.bool()
            normalized_joint_position = (
                human_joints_body.float()
                / human_joints_body.new_tensor((6.0, 6.0, 3.0))
            ).clamp(-5.0, 5.0).masked_fill(
                ~joint_valid[..., None], 0.0)
            normalized_joint_velocity = (
                human_joint_velocity_body.float()
                / human_joint_velocity_body.new_tensor((3.0, 3.0, 3.0))
            ).clamp(-5.0, 5.0).masked_fill(
                ~joint_valid[..., None], 0.0)
            actor_joint_state = torch.cat((
                joint_valid.to(normalized_joint_position)[..., None],
                normalized_joint_position,
                normalized_joint_velocity,
            ), -1).flatten(-2)
            if human_joint_clearance is None:
                human_joint_clearance = sphere_signed_gap.amin(-1)
                human_joint_clearance = torch.where(
                    nearest_valid,
                    human_joint_clearance,
                    human_joint_clearance.new_full(
                        human_joint_clearance.shape, 6.0),
                )
        if human_joint_clearance is None:
            # Legacy/non-pure callers still receive a physically meaningful
            # geometry scalar. Pure R2-Dreamer supplies the nearest predicted
            # joint gap explicitly, matching the recorder target.
            human_joint_clearance = (
                torch.linalg.vector_norm(
                    human_root[..., :3].float(), dim=-1) - 0.25
            ).clamp(0.0, 6.0)
        if human_joint_clearance.shape != slot_shape:
            raise ValueError("human_joint_clearance must match Human slots")
        if human_root.shape[-1] > self.human_root_dim:
            human_root = human_root[..., :self.human_root_dim]
        elif human_root.shape[-1] < self.human_root_dim:
            human_root = F.pad(
                human_root, (0, self.human_root_dim - human_root.shape[-1]))
        if human_quality.shape[-1] > self.human_quality_dim:
            human_quality = human_quality[..., :self.human_quality_dim]
        elif human_quality.shape[-1] < self.human_quality_dim:
            human_quality = F.pad(
                human_quality,
                (0, self.human_quality_dim - human_quality.shape[-1]))
        slot_mask = human_mask.bool()
        presence = (
            slot_mask.to(human_feature.dtype)
            if human_presence is None else human_presence.to(human_feature.dtype)
        )
        if presence.shape != slot_shape:
            raise ValueError("human_presence must match Human slots")
        presence = presence.clamp(0.0, 1.0) * slot_mask.to(human_feature.dtype)
        # Use fixed physical scales instead of a per-slot input LayerNorm.
        # LayerNorm over [latent, root, quality] can erase the absolute range
        # signal that distinguishes a person at 0.2 m from one at 2.0 m.
        root_scale = human_root.new_ones(self.human_root_dim)
        root_scale[: min(3, self.human_root_dim)] = human_root.new_tensor(
            (6.0, 6.0, 3.0))[: min(3, self.human_root_dim)]
        if self.human_root_dim > 3:
            end = min(6, self.human_root_dim)
            root_scale[3:end] = 3.0
        if self.human_root_dim > 6:
            end = min(9, self.human_root_dim)
            root_scale[6:end] = 3.0
        quality_scale = human_quality.new_ones(self.human_quality_dim)
        # measured ratio, predicted ratio, track age, measurement age,
        # consecutive prediction frames, velocity sigma, identity confidence.
        canonical_quality_scale = human_quality.new_tensor(
            (1.0, 1.0, 10.0, 2.0, 20.0, 3.0, 1.0))
        count = min(self.human_quality_dim, canonical_quality_scale.numel())
        quality_scale[:count] = canonical_quality_scale[:count]
        normalized_root = (human_root.float() / root_scale).clamp(-5.0, 5.0)
        normalized_quality = (
            human_quality.float() / quality_scale).clamp(-5.0, 5.0)
        learned_inputs = [
            human_feature,
            normalized_root,
            normalized_quality,
            # Keep the safety-critical sub-metre range at metre scale.
            human_joint_clearance.float().clamp(-0.5, 4.0)[..., None],
        ]
        if nearest_joint_fields is not None:
            learned_inputs.append(nearest_joint_fields)
        learned_geometry = self._apply_network(
            self.geometry_network,
            torch.cat(learned_inputs, -1),
            detach_parameters=detach_parameters,
        )
        physical_root = normalized_root[..., :6]
        if physical_root.shape[-1] < 6:
            physical_root = F.pad(
                physical_root, (0, 6 - physical_root.shape[-1]))
        if nearest_joint_fields is None:
            physical = torch.cat((
                physical_root,
                human_joint_clearance.float().clamp(-0.5, 4.0)[..., None],
                presence[..., None],
            ), -1)
        else:
            if self.explicit_human_presence_physical:
                # Per person: root xy/vxy, existence probability, signed gap,
                # and nearest articulated sphere xyz/vxyz. Root z is redundant
                # with the direct sphere z; using that coordinate for presence
                # closes the imagined lifecycle state without increasing the
                # fixed 10x12 Actor/Critic geometry width.
                physical = torch.cat((
                    normalized_root[..., (0, 1, 3, 4)],
                    presence[..., None],
                    human_joint_clearance.float().clamp(-0.5, 4.0)[..., None],
                    nearest_joint_fields,
                ), -1)
            else:
                # Historical v7.2 layout retained for checkpoint deployment.
                physical = torch.cat((
                    normalized_root[..., :5],
                    human_joint_clearance.float().clamp(-0.5, 4.0)[..., None],
                    nearest_joint_fields,
                ), -1)
        slot_physical_dim = self.geometry_slot_physical_dim
        geometry = torch.cat((
            physical[..., :slot_physical_dim],
            learned_geometry[..., slot_physical_dim:],
        ), -1)
        distance = torch.linalg.vector_norm(human_root[..., :3].float(), dim=-1)
        proximity_logit = (-distance).masked_fill(~slot_mask, -torch.inf)
        # ``softmax([-inf, ...])`` is NaN.  A following ``where`` makes the
        # forward value look harmless but its backward can still poison the
        # Human encoder on frames with no visible person.  Give only those
        # all-empty rows finite dummy logits, then remove every masked weight.
        has_visible_human = slot_mask.any(-1, keepdim=True)
        proximity_logit = torch.where(
            has_visible_human,
            proximity_logit,
            torch.zeros_like(proximity_logit),
        )
        proximity_weight = (
            torch.softmax(proximity_logit, dim=-1)
            * slot_mask.to(proximity_logit.dtype)
        )
        learned_pool = (
            learned_geometry * proximity_weight[..., None]).sum(-2)
        nearest_index = distance.masked_fill(~slot_mask, torch.inf).argmin(-1)
        nearest_physical = physical.gather(
            -2,
            nearest_index[..., None, None].expand(
                *nearest_index.shape, 1, physical.shape[-1]),
        ).squeeze(-2)
        nearest_physical = torch.where(
            slot_mask.any(-1)[..., None],
            nearest_physical,
            torch.zeros_like(nearest_physical),
        )
        if self.geometry_topk_physical_slots > 0:
            # Preserve signed metric structure for every Human count observed
            # in the audited replay (maximum nine; configured capacity ten).
            # Sorting by distance is permutation-invariant. Gathered values
            # remain differentiable except at the measure-zero rank swap.
            count = min(
                self.geometry_topk_physical_slots, human_feature.shape[-2])
            # Capacity must retain the people whose articulated collision
            # envelopes are nearest, not merely the nearest pelvis/root.  A
            # farther pelvis can own the minimum gap through an extended hand
            # or foot; root-distance truncation would then remove the most
            # safety-critical directional state from Actor and Critic.
            physical_priority = (
                human_joint_clearance.float()
                if self.explicit_joint_kinematics else distance)
            order = physical_priority.masked_fill(~slot_mask, torch.inf).topk(
                count, dim=-1, largest=False).indices
            # The physical layout is versioned above; learned pooling retains
            # additional pose/quality and population context.
            ordered_source = (
                physical if self.explicit_joint_kinematics else
                torch.cat((
                    normalized_root[..., :5],
                    human_joint_clearance.float().clamp(
                        -0.5, 4.0)[..., None],
                ), -1)
            )
            ordered = ordered_source.gather(
                -2, order[..., None].expand(
                    *order.shape,
                    self.geometry_physical_fields_per_slot,
                ))
            ordered_valid = slot_mask.gather(-1, order)
            ordered = ordered * ordered_valid[..., None].to(ordered)
            if count < self.geometry_topk_physical_slots:
                ordered = F.pad(
                    ordered,
                    (0, 0, 0, self.geometry_topk_physical_slots - count))
            pooled_physical = ordered.flatten(-2)
            physical_dim = self.geometry_physical_dim
            geometry_pool = torch.cat((
                pooled_physical,
                learned_pool[..., physical_dim:],
            ), -1)
        else:
            physical_dim = self.geometry_physical_dim
            geometry_pool = torch.cat((
                nearest_physical[..., :physical_dim],
                learned_pool[..., physical_dim:],
            ), -1)
        actor_human_state = geometry_pool
        if self.actor_full_state_slots > 0:
            if actor_joint_state is None:
                raise RuntimeError(
                    "full-state Actor requires articulated joint kinematics")
            actor_slot_state = torch.cat((
                normalized_root,
                normalized_quality,
                presence[..., None],
                human_joint_clearance.float().clamp(-0.5, 4.0)[..., None],
                actor_joint_state,
            ), -1)
            if actor_slot_state.shape[-1] != self.actor_full_fields_per_slot:
                raise RuntimeError(
                    "full-state Actor Human field layout changed unexpectedly")
            actor_count = min(
                self.actor_full_state_slots, human_feature.shape[-2])
            actor_priority = human_joint_clearance.float().masked_fill(
                ~slot_mask, torch.inf)
            actor_order = actor_priority.topk(
                actor_count, dim=-1, largest=False).indices
            actor_ordered = actor_slot_state.gather(
                -2, actor_order[..., None].expand(
                    *actor_order.shape, actor_slot_state.shape[-1]))
            actor_valid = slot_mask.gather(-1, actor_order)
            actor_ordered = (
                actor_ordered * actor_valid[..., None].to(actor_ordered))
            if actor_count < self.actor_full_state_slots:
                actor_ordered = F.pad(
                    actor_ordered,
                    (0, 0, 0, self.actor_full_state_slots - actor_count))
            if self.actor_full_learned_per_slot:
                # Preserve the learned motion/history suffix for each person
                # with the exact same clearance-sorted identity ordering as
                # its physical fields. A single pooled suffix aliases which
                # intent belongs to which articulated body and leaves Event
                # with per-person information unavailable to Actor/Critic.
                learned_suffix = learned_geometry[
                    ..., self.geometry_slot_physical_dim:]
                actor_ordered_learned = learned_suffix.gather(
                    -2, actor_order[..., None].expand(
                        *actor_order.shape, learned_suffix.shape[-1]))
                actor_ordered_learned = (
                    actor_ordered_learned
                    * actor_valid[..., None].to(actor_ordered_learned))
                if actor_count < self.actor_full_state_slots:
                    actor_ordered_learned = F.pad(
                        actor_ordered_learned,
                        (0, 0, 0,
                         self.actor_full_state_slots - actor_count))
                # Keep all exact physical coordinates as one contiguous
                # prefix, followed by the aligned learned suffixes.
                actor_human_state = torch.cat((
                    actor_ordered.flatten(-2),
                    actor_ordered_learned.flatten(-2),
                ), -1)
            else:
                # Historical layout: exact fields plus one lossy population
                # pool, retained only for old checkpoints.
                actor_human_state = torch.cat((
                    actor_ordered.flatten(-2), learned_pool,
                ), -1)
            if actor_human_state.shape[-1] != self.actor_full_state_dim:
                raise RuntimeError("Actor Human state width is inconsistent")
        return {
            "slot_geometry_token": geometry,
            "human_geometry_pool": geometry_pool,
            "actor_human_state": actor_human_state,
            "actor_slot_state": actor_slot_state,
            "slot_mask": slot_mask,
            "presence": presence,
            "proximity_weight": proximity_weight,
        }


class NextHumanClearanceHead(nn.Module):
    """Predict privileged human surface clearance at the next observation."""

    def __init__(
        self, feature_dim: int, action_dim: int, hidden_dim: int = 256, *,
        ego_feat_dim: int | None = None,
        human_feat_dim: int | None = None,
        human_root_dim: int = 10,
        human_quality_dim: int = 7,
        explicit_human_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.explicit_human_geometry = bool(explicit_human_geometry)
        ego_feat_dim = int(feature_dim if ego_feat_dim is None else ego_feat_dim)
        human_feat_dim = int(
            feature_dim if human_feat_dim is None else human_feat_dim)
        self.human_root_dim = int(human_root_dim)
        self.human_quality_dim = int(human_quality_dim)
        input_dim = (
            int(feature_dim) + ego_feat_dim + human_feat_dim
            + self.human_root_dim + self.human_quality_dim + int(action_dim)
            if self.explicit_human_geometry else
            int(feature_dim) + int(action_dim))
        hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.empty_network = (
            nn.Sequential(
                nn.Linear(int(feature_dim) + int(action_dim), hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, 1),
            ) if self.explicit_human_geometry else None)

    def forward(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        *,
        ego_feature: torch.Tensor | None = None,
        human_feature: torch.Tensor | None = None,
        human_root: torch.Tensor | None = None,
        human_quality: torch.Tensor | None = None,
        human_mask: torch.Tensor | None = None,
        human_presence: torch.Tensor | None = None,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        if feature.shape[:-1] != action.shape[:-1]:
            raise ValueError("feature and action leading dimensions must match")
        if not self.explicit_human_geometry:
            value = TransitionEventHead._apply_network(
                self.network, torch.cat((feature, action), -1),
                detach_parameters=detach_parameters)
            return F.softplus(value)
        context = (ego_feature, human_feature, human_root, human_quality, human_mask)
        if all(item is None for item in context):
            assert self.empty_network is not None
            value = TransitionEventHead._apply_network(
                self.empty_network, torch.cat((feature, action), -1),
                detach_parameters=detach_parameters)
            return F.softplus(value)
        if any(item is None for item in context):
            raise ValueError("clearance requires complete per-human context")
        assert ego_feature is not None and human_feature is not None
        assert human_root is not None and human_quality is not None
        assert human_mask is not None
        slot_shape = human_feature.shape[:-1]
        if human_root.shape[:-1] != slot_shape or human_quality.shape[:-1] != slot_shape:
            raise ValueError("clearance Human context shape mismatch")
        mask = human_mask.bool()
        if human_root.shape[-1] > self.human_root_dim:
            human_root = human_root[..., :self.human_root_dim]
        elif human_root.shape[-1] < self.human_root_dim:
            human_root = F.pad(
                human_root, (0, self.human_root_dim - human_root.shape[-1]))
        if human_quality.shape[-1] > self.human_quality_dim:
            human_quality = human_quality[..., :self.human_quality_dim]
        elif human_quality.shape[-1] < self.human_quality_dim:
            human_quality = F.pad(
                human_quality,
                (0, self.human_quality_dim - human_quality.shape[-1]))
        presence = (
            mask.to(feature.dtype)
            if human_presence is None else human_presence.to(feature.dtype)
        ).clamp(0.0, 1.0) * mask.to(feature.dtype)
        expanded = (
            feature.unsqueeze(-2).expand(*slot_shape, feature.shape[-1]),
            ego_feature.unsqueeze(-2).expand(*slot_shape, ego_feature.shape[-1]),
            human_feature,
            human_root.float(),
            human_quality.float(),
            action.unsqueeze(-2).expand(*slot_shape, action.shape[-1]),
        )
        slot_value = TransitionEventHead._apply_network(
            self.network, torch.cat(expanded, -1),
            detach_parameters=detach_parameters)
        slot_gap = F.softplus(slot_value).squeeze(-1)
        # Vanishing tracks should not become the minimum-clearance person.
        effective_gap = slot_gap + (1.0 - presence) * 6.0
        effective_gap = effective_gap.masked_fill(~mask, torch.inf)
        minimum = effective_gap.amin(-1, keepdim=True)
        assert self.empty_network is not None
        fallback = F.softplus(TransitionEventHead._apply_network(
            self.empty_network, torch.cat((feature, action), -1),
            detach_parameters=detach_parameters))
        return torch.where(mask.any(-1, keepdim=True), minimum, fallback)


class HumanSafetyCritic(nn.Module):
    """Bounded estimate of future episode human-collision probability."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.logits(feature).sigmoid()


def survival_human_collision_return(
    human_hazard: torch.Tensor,
    continue_probability: torch.Tensor,
    bootstrap_probability: torch.Tensor,
) -> torch.Tensor:
    """Return exact finite-horizon collision probability for each start.

    Inputs are ``[B,H,1]`` one-step event probabilities. Output has the same
    shape and uses an undiscounted absorbing-event recursion:

    ``Q_t = h_t + p_continue_t * Q_(t+1)``.
    """
    if human_hazard.shape != continue_probability.shape:
        raise ValueError("hazard and continuation tensors must have equal shape")
    if bootstrap_probability.shape != human_hazard[:, -1].shape:
        raise ValueError("bootstrap probability must match one time slice")
    next_value = bootstrap_probability.clamp(0.0, 1.0)
    returns = []
    for index in range(human_hazard.shape[1] - 1, -1, -1):
        next_value = (
            human_hazard[:, index]
            + continue_probability[:, index] * next_value
        ).clamp(0.0, 1.0)
        returns.append(next_value)
    return torch.stack(tuple(reversed(returns)), dim=1)
