"""Goal/avoidance-factorized stochastic Actor for v6.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from distributions import TanhNormal
from modules.se2_relative_dynamics import (
    DEPLOYABLE_DRONE_COLLISION_RADIUS_M,
    ego_velocity_full_body,
)


@dataclass(frozen=True)
class FactorizedActorV62Config:
    token_dim: int = 128
    hidden_dim: int = 256
    policy_action_dim: int = 3
    residual_bound: float = 0.75
    # Match the standard Dreamer Actor's small final-layer initialization.
    # The custom factorized Actor must not begin collection with an arbitrary
    # saturated left/right command merely because it bypasses MLPHead.
    goal_output_scale: float = 0.01
    min_std: float | tuple[float, ...] = 0.05
    max_std: float | tuple[float, ...] = 0.80
    initial_std: tuple[float, ...] | None = None
    # Historical v6 Actors used a hard upper clamp that was annealed onto the
    # minimum standard deviation.  ``learned_bounded`` instead uses a smooth
    # sigmoid between the configured per-axis limits and initializes exactly
    # at ``initial_std``; no schedule can silently turn the stochastic Actor
    # into a fixed near-deterministic policy.
    std_parameterization: str = "annealed_cap"
    # A tanh-squashed action cannot become materially stronger once its Normal
    # location is large, but an unbounded location can keep growing until the
    # pathwise gradient disappears.  A positive value softly bounds the
    # Normal location while retaining one ordinary stochastic distribution.
    pre_tanh_mean_bound: float = 0.0
    # ``tanh`` is retained for historical checkpoints. Current pure Dreamer
    # uses the algebraic square-root bound: it has the same finite asymptote
    # and unit slope at the origin, but its derivative decays polynomially
    # instead of rounding to exactly zero under bfloat16 autocast.
    pre_tanh_mean_parameterization: str = "tanh"
    # A strongly negative bias creates a two-factor cold start: the
    # zero-initialized avoidance residual gives the gate no gradient, while
    # the nearly closed gate suppresses the residual's own gradient.  A
    # neutral gate leaves the initial policy unchanged (the residual is still
    # exactly zero) without starving the Human-conditioned branch.
    initial_gate_bias: float = 0.0
    # Make the residual semantically Human-conditioned instead of allowing a
    # second goal/navigation policy to hide inside the avoidance branch.  The
    # same network is evaluated with and without the Human token; their
    # difference is exactly zero in an empty scene and remains learned solely
    # from Dreamer's imagined return.
    human_conditioned_residual: bool = False
    # The body-frame Human geometry has an exact reflection contract:
    # (y, vy) -> (-y, -vy).  When enabled, the Human correction is even for
    # forward speed and odd for lateral speed/yaw.  This removes an otherwise
    # unconstrained side-independent lateral policy without providing an action
    # target; all correction magnitudes are still learned from Dreamer return.
    human_reflection_equivariant: bool = False
    human_physical_slots: int = 0
    human_physical_fields_per_slot: int = 6
    human_physical_presence_state: bool = False
    # The unified Actor may consume a Human state wider than one RSSM token.
    # ``None`` preserves the historical four-equal-token layout.
    human_geometry_dim: int | None = None
    # Exact task geometry is a fifth, independently injective state block.
    # It cannot be folded into the eight-ray Ego token because the analytic
    # return depends on AABBs that may miss every ray.
    task_geometry_dim: int = 0
    # Pure Dreamer v18 uses one ordinary policy mean over the complete
    # deployable decision state.  This removes the non-identifiable
    # gate-times-residual product and lets the Actor read the same learned
    # joint/Human context that shapes reward and value.  The historical
    # factorized path remains available only for old checkpoint contracts.
    unified_policy: bool = False
    # Current full-state checkpoints treat Humans and obstacles as physical
    # sets. A flat MLP over clearance-sorted ranks gives every rank unrelated
    # parameters and can jump when two entities exchange order.
    permutation_invariant_entities: bool = False
    human_learned_fields_per_slot: int = 0
    task_physical_obstacle_slots: int = 0
    task_physical_base_dim: int = 16
    task_physical_fields_per_obstacle: int = 7
    decision_entity_encoder_contract: str = (
        "clearance_cpa_exact_interaction_context_query_v3")


class CompleteDecisionStateEncoder(nn.Module):
    """Encode the complete state without erasing entity-level conflicts.

    A plain Deep-Sets sum/max is permutation invariant, but it is a poor
    bottleneck for this task: the maximum in different embedding coordinates
    can belong to different people, while the sum is dominated by the crowd.
    The policy then has to reconstruct which signed pose, motion history and
    articulated envelope belong to the same imminent collision partner from
    an aggregate that no longer contains that association.

    Keep the global population summaries, and add continuous soft-nearest
    summaries whose weights are computed from exact physical clearance.  Each
    weighted value is one complete per-entity embedding, so physical fields
    and their aligned learned suffix remain associated.  A context-conditioned
    attention summary lets goal/Ego/boundary context select a different entity
    without introducing an action rule, candidate search or entity rank.
    """

    HUMAN_PRESENCE_FIELD = 17
    HUMAN_CLEARANCE_FIELD = 18
    OBSTACLE_VALID_FIELD = 0
    CONTRACT_VERSION = "clearance_cpa_exact_interaction_context_query_v3"
    CPA_LEARNED_CONTRACT_VERSION = (
        "clearance_cpa_soft_conflict_context_query_v2")
    LEGACY_CONTRACT_VERSION = "clearance_soft_nearest_context_query_v1"
    HUMAN_CLEARANCE_TEMPERATURES_M = (0.10, 0.35, 1.00)
    HUMAN_CPA_RISK_TEMPERATURES = (0.10, 0.35)
    HUMAN_CPA_HORIZON_S = 2.5
    HUMAN_CPA_HARD_CLEARANCE_M = 0.10
    HUMAN_CPA_SAFE_CLEARANCE_M = 0.90
    HUMAN_PELVIS_RADIUS_M = 0.14
    HUMAN_INTERACTION_FIELDS = 10
    EGO14_METRIC_SCALE = (
        32.0, 12.0, 1.0, 3.0, 3.0, 2.0, 8.0,
        8.0, 8.0, 2.0, 1.0, 1.0, 1.0, 1.0,
    )
    # Obstacle center/extent fields are normalized by the task geometry's
    # twelve-metre metric scale.  These temperatures are therefore also in
    # that normalized coordinate system (roughly 0.6 m and 2.4 m).
    OBSTACLE_CLEARANCE_TEMPERATURES = (0.05, 0.20)

    def __init__(
        self,
        *,
        token_dim: int,
        human_slots: int,
        human_physical_fields: int,
        human_learned_fields: int,
        task_base_dim: int,
        task_obstacle_slots: int,
        task_obstacle_fields: int,
        human_embedding_dim: int = 128,
        obstacle_embedding_dim: int = 64,
        contract_version: str = CONTRACT_VERSION,
    ) -> None:
        super().__init__()
        current_contract_version = type(self).CONTRACT_VERSION
        self.token_dim = int(token_dim)
        self.human_slots = int(human_slots)
        self.human_physical_fields = int(human_physical_fields)
        self.human_learned_fields = int(human_learned_fields)
        self.task_base_dim = int(task_base_dim)
        self.task_obstacle_slots = int(task_obstacle_slots)
        self.task_obstacle_fields = int(task_obstacle_fields)
        if contract_version not in (
            self.CONTRACT_VERSION,
            self.CPA_LEARNED_CONTRACT_VERSION,
            self.LEGACY_CONTRACT_VERSION,
        ):
            raise ValueError(
                f"unsupported decision entity encoder {contract_version!r}")
        # Keep archived v8.5 checkpoints loadable for read-only audits. New
        # training uses the class default and records this instance value in
        # the checkpoint fingerprint.
        self.CONTRACT_VERSION = str(contract_version)
        self.cpa_interaction_enabled = (
            contract_version != self.LEGACY_CONTRACT_VERSION)
        self.exact_cpa_passthrough_enabled = (
            contract_version == current_contract_version)
        if self.token_dim <= 0 or self.human_slots <= 0:
            raise ValueError("set decision encoder requires tokens and Humans")
        if self.human_physical_fields <= self.HUMAN_PRESENCE_FIELD:
            raise ValueError("Human layout lacks its lifecycle-presence field")
        if self.human_learned_fields <= 0:
            raise ValueError("set decision encoder requires learned Human state")
        if self.task_base_dim <= 0 or self.task_obstacle_slots <= 0:
            raise ValueError("set decision encoder requires exact task geometry")
        if self.task_obstacle_fields <= self.OBSTACLE_VALID_FIELD:
            raise ValueError("obstacle layout lacks its validity field")

        human_width = (
            self.human_physical_fields + self.human_learned_fields)
        interaction_width = (
            self.HUMAN_INTERACTION_FIELDS
            if self.cpa_interaction_enabled else 0)
        self.human_encoder = nn.Sequential(
            nn.Linear(
                human_width + interaction_width,
                human_embedding_dim,
            ), nn.SiLU(),
            nn.Linear(human_embedding_dim, human_embedding_dim), nn.SiLU(),
        )
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(self.task_obstacle_fields, obstacle_embedding_dim),
            nn.SiLU(),
            nn.Linear(obstacle_embedding_dim, obstacle_embedding_dim),
            nn.SiLU(),
        )
        self.human_embedding_dim = int(human_embedding_dim)
        self.obstacle_embedding_dim = int(obstacle_embedding_dim)
        obstacle_context_dim = 3 * self.token_dim + self.task_base_dim
        self.obstacle_query = nn.Sequential(
            nn.Linear(obstacle_context_dim, self.obstacle_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.obstacle_embedding_dim,
                      self.obstacle_embedding_dim),
        )
        self.obstacle_key = nn.Linear(
            self.obstacle_embedding_dim, self.obstacle_embedding_dim,
            bias=False)
        self.obstacle_base_summary_dim = 2 * self.obstacle_embedding_dim + 1
        self.obstacle_soft_summary_dim = (
            len(self.OBSTACLE_CLEARANCE_TEMPERATURES)
            * self.obstacle_embedding_dim)
        self.obstacle_query_summary_dim = self.obstacle_embedding_dim
        self.obstacle_summary_dim = (
            self.obstacle_base_summary_dim
            + self.obstacle_soft_summary_dim
            + self.obstacle_query_summary_dim)
        human_context_dim = (
            3 * self.token_dim + self.task_base_dim
            + self.obstacle_summary_dim)
        self.human_query = nn.Sequential(
            nn.Linear(human_context_dim, self.human_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.human_embedding_dim, self.human_embedding_dim),
        )
        self.human_key = nn.Linear(
            self.human_embedding_dim, self.human_embedding_dim, bias=False)
        self.human_base_summary_dim = 2 * self.human_embedding_dim + 1
        self.human_soft_summary_dim = (
            len(self.HUMAN_CLEARANCE_TEMPERATURES_M)
            * self.human_embedding_dim)
        self.human_cpa_summary_dim = (
            len(self.HUMAN_CPA_RISK_TEMPERATURES)
            * self.human_embedding_dim
            if self.cpa_interaction_enabled else 0)
        # v2 used exact CPA state only to form a learned per-person embedding,
        # then pooled the embedding.  That makes the policy reconstruct the
        # signed closest-point side and relative motion from a bottleneck even
        # though those quantities are already known exactly.  v3 retains two
        # continuous risk-weighted views of all ten interaction fields plus
        # population risk mass/maximum.  This remains a permutation-invariant
        # observation encoder: it neither selects an action nor encodes an
        # avoidance side rule.
        self.human_exact_cpa_summary_dim = (
            len(self.HUMAN_CPA_RISK_TEMPERATURES)
            * self.HUMAN_INTERACTION_FIELDS + 2
            if self.exact_cpa_passthrough_enabled else 0)
        self.human_query_summary_dim = self.human_embedding_dim
        self.human_summary_dim = (
            self.human_base_summary_dim
            + self.human_soft_summary_dim
            + self.human_cpa_summary_dim
            + self.human_exact_cpa_summary_dim
            + self.human_query_summary_dim)
        self.input_dim = (
            3 * self.token_dim
            + self.human_slots * human_width
            + self.task_base_dim
            + self.task_obstacle_slots * self.task_obstacle_fields
        )
        self.output_dim = (
            3 * self.token_dim + self.task_base_dim
            + self.human_summary_dim
            + self.obstacle_summary_dim
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _masked_set_summary(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = mask.to(encoded.dtype)[..., None]
        # Bound aggregate scale while retaining population count explicitly.
        reference = float(self.human_slots)
        summed = (encoded * weight).sum(-2) / reference ** 0.5
        masked = encoded.masked_fill(~mask[..., None], -torch.inf)
        maximum = masked.max(-2).values
        maximum = torch.where(
            mask.any(-1)[..., None], maximum, torch.zeros_like(maximum))
        fraction = weight.sum(-2) / reference
        return torch.cat((summed, maximum, fraction), -1)

    @staticmethod
    def _masked_weighted_summary(
        encoded: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool complete entity embeddings with a finite empty-set path."""
        if logits.shape != mask.shape or encoded.shape[:-1] != mask.shape:
            raise ValueError("entity attention shapes are inconsistent")
        safe_logits = logits.masked_fill(~mask, -torch.inf)
        has_entity = mask.any(-1, keepdim=True)
        # Softmax over an all--inf row is NaN even if it is masked later.
        safe_logits = torch.where(
            has_entity, safe_logits, torch.zeros_like(safe_logits))
        weight = torch.softmax(safe_logits, -1) * mask.to(logits.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1.0e-12)
        return (encoded * weight[..., None]).sum(-2)

    @classmethod
    def _masked_soft_nearest_summary(
        cls,
        encoded: torch.Tensor,
        priority: torch.Tensor,
        mask: torch.Tensor,
        temperatures: Sequence[float],
    ) -> torch.Tensor:
        if priority.shape != mask.shape:
            raise ValueError("entity priority must match its validity mask")
        summaries = []
        for temperature in temperatures:
            if float(temperature) <= 0.0:
                raise ValueError("entity temperature must be positive")
            summaries.append(cls._masked_weighted_summary(
                encoded, -priority / float(temperature), mask))
        return torch.cat(summaries, -1)

    def _masked_query_summary(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
        *,
        query_network: nn.Module,
        key_network: nn.Module,
    ) -> torch.Tensor:
        query = query_network(context)
        key = key_network(encoded)
        logits = (
            key * query[..., None, :]
        ).sum(-1) / float(key.shape[-1]) ** 0.5
        return self._masked_weighted_summary(encoded, logits, mask)

    def _human_interaction_state(
        self,
        core: torch.Tensor,
        physical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deployable per-person Ego-relative CPA state and risk.

        Root position/velocity are already present in every Human slot, but a
        set encoder otherwise has to reconstruct their interaction with Ego
        velocity before pooling people.  Compute that continuous interaction
        while the fields still belong to one person.  This is state only: it
        neither chooses a side nor constrains the Actor action.

        Production tokens contain the normalized Ego14 physical prefix. Tiny
        unit-test/legacy layouts with a token narrower than Ego14 retain a
        finite zero-Ego-velocity fallback; current checkpoints require the
        versioned production contract above.
        """
        root_scale = physical.new_tensor((6.0, 6.0, 3.0))
        velocity_scale = physical.new_tensor((3.0, 3.0, 3.0))
        relative_position = physical[..., :3].float() * root_scale
        human_velocity = physical[..., 3:6].float() * velocity_scale
        if self.token_dim >= len(self.EGO14_METRIC_SCALE):
            ego_normalized = core[
                ..., self.token_dim:self.token_dim + 14].float()
            ego_state = ego_normalized * core.new_tensor(
                self.EGO14_METRIC_SCALE).float()
            ego_velocity = ego_velocity_full_body(ego_state)
        else:
            ego_velocity = torch.zeros_like(relative_position[..., 0, :])
        relative_velocity = human_velocity - ego_velocity[..., None, :]
        speed_sq = relative_velocity.square().sum(-1)
        raw_tcpa = -(
            relative_position * relative_velocity
        ).sum(-1) / speed_sq.clamp_min(1.0e-8)
        tcpa = raw_tcpa.clamp(0.0, self.HUMAN_CPA_HORIZON_S)
        closest = relative_position + relative_velocity * tcpa[..., None]
        closest_clearance = (
            torch.linalg.vector_norm(closest, dim=-1)
            - float(DEPLOYABLE_DRONE_COLLISION_RADIUS_M)
            - self.HUMAN_PELVIS_RADIUS_M
        )
        root_range = torch.linalg.vector_norm(
            relative_position, dim=-1).clamp_min(1.0e-4)
        closing_speed = -(
            relative_position * relative_velocity
        ).sum(-1) / root_range
        spatial_risk = (
            (self.HUMAN_CPA_SAFE_CLEARANCE_M - closest_clearance)
            / (
                self.HUMAN_CPA_SAFE_CLEARANCE_M
                - self.HUMAN_CPA_HARD_CLEARANCE_M
            )
        ).clamp(0.0, 1.0).square()
        temporal_risk = (
            1.0 - tcpa / self.HUMAN_CPA_HORIZON_S
        ).clamp(0.0, 1.0).square()
        approaching = (
            speed_sq.gt(1.0e-8)
            & raw_tcpa.gt(0.0)
            & raw_tcpa.lt(self.HUMAN_CPA_HORIZON_S)
        )
        predictive_risk = (
            spatial_risk * temporal_risk * approaching.to(spatial_risk))
        interaction = torch.cat((
            (relative_velocity / velocity_scale).clamp(-5.0, 5.0),
            (raw_tcpa / self.HUMAN_CPA_HORIZON_S).clamp(-2.0, 2.0)[
                ..., None],
            (closest / root_scale).clamp(-5.0, 5.0),
            closest_clearance.clamp(-0.5, 4.0)[..., None],
            (closing_speed / 3.0).clamp(-5.0, 5.0)[..., None],
            predictive_risk[..., None],
        ), -1)
        if interaction.shape[-1] != self.HUMAN_INTERACTION_FIELDS:
            raise RuntimeError("Human CPA interaction width changed")
        return interaction.to(physical), predictive_risk.to(physical)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.shape[-1] != self.input_dim:
            raise ValueError(
                f"set decision encoder expects {self.input_dim} fields, got "
                f"{feature.shape[-1]}")
        cursor = 0
        core_width = 3 * self.token_dim
        core = feature[..., cursor:cursor + core_width]
        cursor += core_width
        physical_width = self.human_slots * self.human_physical_fields
        physical = feature[..., cursor:cursor + physical_width].reshape(
            *feature.shape[:-1], self.human_slots,
            self.human_physical_fields)
        cursor += physical_width
        learned_width = self.human_slots * self.human_learned_fields
        learned = feature[..., cursor:cursor + learned_width].reshape(
            *feature.shape[:-1], self.human_slots,
            self.human_learned_fields)
        cursor += learned_width
        task_base = feature[..., cursor:cursor + self.task_base_dim]
        cursor += self.task_base_dim
        obstacles = feature[..., cursor:].reshape(
            *feature.shape[:-1], self.task_obstacle_slots,
            self.task_obstacle_fields)

        obstacle_mask = obstacles[..., self.OBSTACLE_VALID_FIELD] > 0.5
        obstacle_encoded = self.obstacle_encoder(obstacles)
        # Obstacle capacity never changed and remains part of its fixed task
        # state contract.  Use the historical normalization directly here;
        # the Human reference above is the only migratable population scale.
        obstacle_weight = obstacle_mask.to(obstacle_encoded.dtype)[..., None]
        obstacle_summed = (
            obstacle_encoded * obstacle_weight
        ).sum(-2) / float(self.task_obstacle_slots) ** 0.5
        obstacle_masked = obstacle_encoded.masked_fill(
            ~obstacle_mask[..., None], -torch.inf)
        obstacle_maximum = obstacle_masked.max(-2).values
        obstacle_maximum = torch.where(
            obstacle_mask.any(-1)[..., None], obstacle_maximum,
            torch.zeros_like(obstacle_maximum))
        obstacle_fraction = (
            obstacle_weight.sum(-2) / float(self.task_obstacle_slots))
        obstacle_base_summary = torch.cat((
            obstacle_summed, obstacle_maximum, obstacle_fraction,
        ), -1)
        obstacle_center = obstacles[..., 1:3]
        obstacle_half_extent = obstacles[..., 3:5]
        obstacle_priority = torch.linalg.vector_norm(
            torch.relu(obstacle_center.abs() - obstacle_half_extent), dim=-1)
        obstacle_soft_summary = self._masked_soft_nearest_summary(
            obstacle_encoded, obstacle_priority, obstacle_mask,
            self.OBSTACLE_CLEARANCE_TEMPERATURES)
        obstacle_context = torch.cat((core, task_base), -1)
        obstacle_query_summary = self._masked_query_summary(
            obstacle_encoded, obstacle_mask, obstacle_context,
            query_network=self.obstacle_query,
            key_network=self.obstacle_key,
        )
        obstacle_summary = torch.cat((
            obstacle_base_summary,
            obstacle_soft_summary,
            obstacle_query_summary,
        ), -1)

        human_mask = physical[..., self.HUMAN_PRESENCE_FIELD] > 0.0
        if self.cpa_interaction_enabled:
            human_interaction, human_cpa_risk = (
                self._human_interaction_state(core, physical))
            human_input = torch.cat((
                physical, learned, human_interaction,
            ), -1)
        else:
            human_cpa_risk = None
            human_input = torch.cat((physical, learned), -1)
        human_encoded = self.human_encoder(human_input)
        human_base_summary = self._masked_set_summary(
            human_encoded, human_mask)
        if self.human_physical_fields > self.HUMAN_CLEARANCE_FIELD:
            human_priority = physical[..., self.HUMAN_CLEARANCE_FIELD]
        else:
            # Small compatibility/unit-test layouts predate the articulated
            # clearance field.  Root range remains a physical, invariant
            # fallback and is never used by production complete-state models.
            human_priority = torch.linalg.vector_norm(
                physical[..., :3], dim=-1)
        human_soft_summary = self._masked_soft_nearest_summary(
            human_encoded, human_priority, human_mask,
            self.HUMAN_CLEARANCE_TEMPERATURES_M)
        human_cpa_summary = (
            torch.cat([
                self._masked_weighted_summary(
                    human_encoded,
                    human_cpa_risk / float(temperature),
                    human_mask,
                )
                for temperature in self.HUMAN_CPA_RISK_TEMPERATURES
            ], -1)
            if human_cpa_risk is not None else
            human_encoded.new_empty(human_encoded.shape[:-2] + (0,))
        )
        if self.exact_cpa_passthrough_enabled:
            exact_weighted = [
                self._masked_weighted_summary(
                    human_interaction,
                    human_cpa_risk / float(temperature),
                    human_mask,
                )
                for temperature in self.HUMAN_CPA_RISK_TEMPERATURES
            ]
            risk_weight = human_mask.to(human_cpa_risk.dtype)
            risk_mass = (
                human_cpa_risk * risk_weight
            ).sum(-1, keepdim=True) / float(self.human_slots) ** 0.5
            masked_risk = human_cpa_risk.masked_fill(~human_mask, -torch.inf)
            risk_maximum = masked_risk.max(-1, keepdim=True).values
            risk_maximum = torch.where(
                human_mask.any(-1, keepdim=True),
                risk_maximum,
                torch.zeros_like(risk_maximum),
            )
            human_exact_cpa_summary = torch.cat((
                *exact_weighted, risk_mass, risk_maximum,
            ), -1)
        else:
            human_exact_cpa_summary = human_encoded.new_empty(
                human_encoded.shape[:-2] + (0,))
        human_context = torch.cat((core, task_base, obstacle_summary), -1)
        human_query_summary = self._masked_query_summary(
            human_encoded, human_mask, human_context,
            query_network=self.human_query,
            key_network=self.human_key,
        )
        human_summary = torch.cat((
            human_base_summary,
            human_soft_summary,
            human_cpa_summary,
            human_exact_cpa_summary,
            human_query_summary,
        ), -1)
        return torch.cat((
            core, task_base, human_summary, obstacle_summary,
        ), -1)


class PermutationInvariantDecisionHead(nn.Module):
    """Apply a distribution head to a set-encoded complete state."""

    def __init__(self, encoder: CompleteDecisionStateEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.input_dim = int(encoder.input_dim)
        self.encoded_dim = int(encoder.output_dim)

    def forward(self, feature: torch.Tensor):
        return self.head(self.encoder(feature))


class FactorizedActorV62(nn.Module):
    """One policy distribution over the complete deployable decision state.

    Input is ``[private_goal_token, private_ego_token, joint_token,
    human_geometry, exact_task_geometry]``.  v18 uses one full-state mean network.  The historical
    separable goal/residual/gate path is retained only to load and audit older
    checkpoints.
    """

    def __init__(self, config: FactorizedActorV62Config) -> None:
        super().__init__()
        self.config = config
        d = int(config.token_dim)
        h = int(config.hidden_dim)
        a = int(config.policy_action_dim)
        self.human_physical_slots = int(config.human_physical_slots)
        self.human_physical_fields_per_slot = int(
            config.human_physical_fields_per_slot)
        self.human_physical_presence_state = bool(
            config.human_physical_presence_state)
        self.human_geometry_dim = int(
            d if config.human_geometry_dim is None
            else config.human_geometry_dim)
        self.task_geometry_dim = int(config.task_geometry_dim)
        if self.human_physical_slots < 0:
            raise ValueError("human_physical_slots must be non-negative")
        if not 0.0 < float(config.goal_output_scale) <= 1.0:
            raise ValueError("goal_output_scale must lie in (0,1]")
        if config.std_parameterization not in (
            "annealed_cap", "learned_bounded",
        ):
            raise ValueError(
                "Actor std parameterization must be annealed_cap or "
                "learned_bounded")
        if float(config.pre_tanh_mean_bound) < 0.0:
            raise ValueError("Actor pre-tanh mean bound must be non-negative")
        if config.pre_tanh_mean_parameterization not in (
            "tanh", "algebraic_sqrt",
        ):
            raise ValueError(
                "Actor pre-tanh mean parameterization must be tanh or "
                "algebraic_sqrt")
        if self.human_physical_fields_per_slot <= 0:
            raise ValueError(
                "human_physical_fields_per_slot must be positive")
        physical_dim = (
            self.human_physical_slots * self.human_physical_fields_per_slot)
        if self.human_geometry_dim <= 0:
            raise ValueError("Human decision-state dimension must be positive")
        if self.task_geometry_dim < 0:
            raise ValueError("task decision-state dimension must be non-negative")
        if physical_dim > self.human_geometry_dim:
            raise ValueError(
                "Human physical geometry prefix exceeds the Human token")
        if config.unified_policy and (
            config.human_conditioned_residual
            or config.human_reflection_equivariant
        ):
            raise ValueError(
                "unified policy is mutually exclusive with the historical "
                "Human residual/reflection branches")
        if not config.unified_policy and (
            self.human_geometry_dim != d or self.task_geometry_dim != 0
        ):
            raise ValueError(
                "historical factorized Actor requires four equal tokens")
        if config.human_reflection_equivariant:
            if not config.human_conditioned_residual:
                raise ValueError(
                    "Human reflection equivariance requires a Human-conditioned "
                    "residual")
            if a != 3:
                raise ValueError(
                    "Human reflection equivariance requires [forward,lateral,yaw]")
            if self.human_physical_slots <= 0:
                raise ValueError(
                    "Human reflection equivariance requires physical slots")
            if self.human_physical_fields_per_slot < 5:
                raise ValueError(
                    "Human reflection requires x,y,z,vx,vy physical fields")
        self.human_physical_dim = physical_dim
        self.input_dim = (
            3 * d + self.human_geometry_dim + self.task_geometry_dim)
        if config.unified_policy:
            # Every exact physical prefix has already been normalized by a
            # fixed, field-specific scale.  A samplewise LayerNorm across the
            # complete heterogeneous state would make (for example) one
            # person's range change the normalized Goal/Ego coordinates and
            # would identify states that differ by a global affine transform.
            # Preserve direct physical scaling, but do not assign semantics to
            # an entity's transient clearance rank.
            if config.permutation_invariant_entities:
                learned_fields = int(config.human_learned_fields_per_slot)
                expected_human = self.human_physical_slots * (
                    self.human_physical_fields_per_slot + learned_fields)
                expected_task = (
                    int(config.task_physical_base_dim)
                    + int(config.task_physical_obstacle_slots)
                    * int(config.task_physical_fields_per_obstacle)
                )
                if self.human_geometry_dim != expected_human:
                    raise ValueError(
                        "set Actor Human layout disagrees with input width")
                if self.task_geometry_dim != expected_task:
                    raise ValueError(
                        "set Actor task layout disagrees with input width")
                self.decision_encoder = CompleteDecisionStateEncoder(
                    token_dim=d,
                    human_slots=self.human_physical_slots,
                    human_physical_fields=(
                        self.human_physical_fields_per_slot),
                    human_learned_fields=learned_fields,
                    task_base_dim=int(config.task_physical_base_dim),
                    task_obstacle_slots=int(
                        config.task_physical_obstacle_slots),
                    task_obstacle_fields=int(
                        config.task_physical_fields_per_obstacle),
                    contract_version=str(
                        config.decision_entity_encoder_contract),
                )
                if self.decision_encoder.input_dim != self.input_dim:
                    raise ValueError("set Actor external input width changed")
                policy_input_dim = self.decision_encoder.output_dim
            else:
                policy_input_dim = self.input_dim
            self.encoded_dim = int(policy_input_dim)
            self.mean_net = nn.Sequential(
                nn.Linear(policy_input_dim, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(), nn.Linear(h, a),
            )
            self.std_net = nn.Sequential(
                nn.Linear(policy_input_dim, h), nn.SiLU(),
                nn.Linear(h, a),
            )
        else:
            self.goal_net = nn.Sequential(
                nn.LayerNorm(2 * d), nn.Linear(2 * d, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(), nn.Linear(h, a),
            )
            self.avoid_net = nn.Sequential(
                nn.LayerNorm(3 * d), nn.Linear(3 * d, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(), nn.Linear(h, a),
            )
            self.gate_net = nn.Sequential(
                nn.LayerNorm(3 * d), nn.Linear(3 * d, h), nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.std_net = nn.Sequential(
                nn.LayerNorm(4 * d), nn.Linear(4 * d, h), nn.SiLU(),
                nn.Linear(h, a),
            )
        std_min = self._std_vector(config.min_std, a, "min_std")
        std_max = self._std_vector(config.max_std, a, "max_std")
        std_initial = (
            std_min + (std_max - std_min) * torch.sigmoid(torch.tensor(2.0))
            if config.initial_std is None
            else self._std_vector(config.initial_std, a, "initial_std")
        )
        if not torch.all((std_min > 0.0) & (std_min <= std_initial)):
            raise ValueError("Actor std requires 0 < min_std <= initial_std")
        if not torch.all(std_initial <= std_max):
            raise ValueError("Actor std requires initial_std <= max_std")
        if config.std_parameterization == "learned_bounded" and not torch.all(
            (std_min < std_initial) & (std_initial < std_max)
        ):
            raise ValueError(
                "learned bounded Actor std requires "
                "min_std < initial_std < max_std")
        self.register_buffer("std_min", std_min)
        self.register_buffer("std_max", std_max)
        self.register_buffer("std_initial", std_initial)
        self.register_buffer("std_schedule_progress", torch.zeros(()))
        self._reset_parameters()

    @staticmethod
    def _std_vector(
        value: float | Sequence[float], size: int, name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(size).clone()
        if tensor.numel() != size:
            raise ValueError(f"{name} must be scalar or have {size} entries")
        return tensor

    @torch.no_grad()
    def set_std_schedule_progress(self, progress: float) -> None:
        if self.config.std_parameterization == "learned_bounded":
            return
        self.std_schedule_progress.fill_(min(1.0, max(0.0, float(progress))))

    def _policy_std(
        self, raw_std_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a valid per-axis std and its active schedule progress."""
        if self.config.std_parameterization == "learned_bounded":
            # This is the standard smooth bounded-scale parameterization, with
            # an axis-specific bias derived from the requested initial scale.
            # Unlike clamp, its std head has a non-zero gradient everywhere
            # inside the finite configured interval.
            initial_fraction = (
                (self.std_initial - self.std_min)
                / (self.std_max - self.std_min)
            ).to(raw_std_residual)
            initial_logit = (
                torch.log(initial_fraction)
                - torch.log1p(-initial_fraction)
            )
            fraction = torch.sigmoid(raw_std_residual + initial_logit)
            std = self.std_min.to(raw_std_residual) + (
                self.std_max.to(raw_std_residual)
                - self.std_min.to(raw_std_residual)
            ) * fraction
            return std, raw_std_residual.new_zeros(())

        progress = self.std_schedule_progress.to(raw_std_residual.dtype)
        effective_max = torch.exp(
            torch.log(self.std_initial)
            + progress * (
                torch.log(self.std_min) - torch.log(self.std_initial))
        )
        log_std = (
            torch.log(self.std_initial) + raw_std_residual
        ).clamp(
            min=torch.log(self.std_min),
            max=torch.log(effective_max),
        )
        return log_std.exp(), progress

    def _policy_mean(
        self, raw_pre_tanh_mean: torch.Tensor,
    ) -> torch.Tensor:
        bound = float(self.config.pre_tanh_mean_bound)
        if bound <= 0.0:
            return raw_pre_tanh_mean
        if self.config.pre_tanh_mean_parameterization == "tanh":
            # Historical contract. Under bfloat16 this can round to an exact
            # +/-bound and therefore has a zero derivative for large logits.
            return bound * torch.tanh(raw_pre_tanh_mean / bound)
        # Compute the reachable-gradient contract outside autocast precision.
        # f(x)=x/sqrt(1+(x/b)^2) stays in (-b,b), has f'(0)=1, and has the
        # strictly-positive polynomial derivative (1+(x/b)^2)^(-3/2) for every
        # finite x. It therefore preserves ordinary pathwise Dreamer gradients
        # without a straight-through estimator or an auxiliary Actor target.
        raw_float = raw_pre_tanh_mean.float()
        scaled = raw_float / bound
        finite_bounded = raw_float * torch.rsqrt(1.0 + scaled.square())
        # Keep distribution construction finite so the trainer's raw-head
        # guard can report the actual non-finite source tensor transactionally.
        fallback = torch.nan_to_num(
            raw_float, nan=0.0, posinf=bound, neginf=-bound)
        return torch.where(torch.isfinite(raw_float), finite_bounded, fallback)

    def _policy_mean_bound_jacobian(
        self,
        raw_pre_tanh_mean: torch.Tensor,
        pre_tanh_mean: torch.Tensor,
    ) -> torch.Tensor:
        bound = float(self.config.pre_tanh_mean_bound)
        if bound <= 0.0:
            return torch.ones_like(pre_tanh_mean, dtype=torch.float32)
        if self.config.pre_tanh_mean_parameterization == "tanh":
            return (
                1.0 - (pre_tanh_mean / bound).square()
            ).float()
        scaled = raw_pre_tanh_mean.float() / bound
        finite_jacobian = (1.0 + scaled.square()).pow(-1.5)
        return torch.where(
            torch.isfinite(scaled),
            finite_jacobian,
            torch.zeros_like(finite_jacobian),
        )

    @torch.no_grad()
    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.config.unified_policy:
            self.mean_net[-1].weight.mul_(
                float(self.config.goal_output_scale))
        else:
            nn.init.zeros_(self.avoid_net[-1].weight)
            nn.init.zeros_(self.avoid_net[-1].bias)
            self.goal_net[-1].weight.mul_(
                float(self.config.goal_output_scale))
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.constant_(self.gate_net[-1].bias,
                              self.config.initial_gate_bias)
        nn.init.zeros_(self.std_net[-1].weight)

    def _physical_human_token(self, human: torch.Tensor) -> torch.Tensor:
        """Drop unconstrained learned suffixes from the safety correction.

        The Event head and Critic continue to consume the complete learned
        geometry token.  The Actor correction uses only the exact, metric,
        distance-sorted prefix so its reflection operator is physically known.
        """
        if self.human_physical_dim <= 0:
            return human
        return torch.cat((
            human[..., :self.human_physical_dim],
            torch.zeros_like(human[..., self.human_physical_dim:]),
        ), -1)

    def _mirror_physical_human_token(
        self, human: torch.Tensor,
    ) -> torch.Tensor:
        """Reflect the explicit body-FLU Human prefix across the forward axis."""
        physical = human[..., :self.human_physical_dim].reshape(
            *human.shape[:-1], self.human_physical_slots,
            self.human_physical_fields_per_slot).clone()
        lateral_indices = (
            (1, 3, 7, 10)
            if self.human_physical_presence_state else
            (1, 4, 7, 10)
            if self.human_physical_fields_per_slot >= 12 else
            (1, 4)
        )
        for index in lateral_indices:
            physical[..., index] = -physical[..., index]
        return torch.cat((
            physical.flatten(-2),
            torch.zeros_like(human[..., self.human_physical_dim:]),
        ), -1)

    def forward(self, feature: torch.Tensor) -> TanhNormal:
        if feature.shape[-1] != self.input_dim:
            raise ValueError(
                f"v6.2 Actor expects {self.input_dim} features, got "
                f"{feature.shape[-1]}")
        if self.config.unified_policy:
            policy_feature = (
                self.decision_encoder(feature)
                if self.config.permutation_invariant_entities else feature)
            raw_pre_tanh_mean = self.mean_net(policy_feature)
            pre_tanh_mean = self._policy_mean(raw_pre_tanh_mean)
            mean_bound_jacobian = self._policy_mean_bound_jacobian(
                raw_pre_tanh_mean, pre_tanh_mean)
            raw_std_residual = self.std_net(policy_feature)
            std, progress = self._policy_std(raw_std_residual)
            distribution = TanhNormal(pre_tanh_mean, std)
            distribution.raw_pre_tanh_mean = raw_pre_tanh_mean
            distribution.pre_tanh_mean = pre_tanh_mean
            distribution.mean_bound_jacobian = mean_bound_jacobian
            distribution.raw_std_residual = raw_std_residual
            distribution.std_schedule_progress = progress
            return distribution

        goal, ego, joint, human = torch.chunk(feature, 4, dim=-1)
        goal_input = torch.cat((goal, ego), dim=-1)
        if self.config.human_reflection_equivariant:
            # Only exact physical fields participate in the Human correction.
            # For f(h), reflection equivariance is obtained by the standard
            # even/odd projections 0.5*(f(h)+/-f(Mh)).  The no-Human subtraction
            # keeps the even forward component exactly zero in an empty scene.
            zero_joint = torch.zeros_like(joint)
            physical_human = self._physical_human_token(human)
            mirrored_human = self._mirror_physical_human_token(human)
            avoidance_input = torch.cat((
                ego, zero_joint, physical_human), dim=-1)
            mirrored_avoidance_input = torch.cat((
                ego, zero_joint, mirrored_human), dim=-1)
            no_human_input = torch.cat((
                ego, zero_joint, torch.zeros_like(human)), dim=-1)
            raw = self.avoid_net(avoidance_input)
            mirrored_raw = self.avoid_net(mirrored_avoidance_input)
            no_human_raw = self.avoid_net(no_human_input)
            even = 0.5 * (raw + mirrored_raw) - no_human_raw
            odd = 0.5 * (raw - mirrored_raw)
            residual_logit = torch.stack((
                even[..., 0], odd[..., 1], odd[..., 2]), dim=-1)
            # A directional gate could itself reintroduce a one-sided bias.
            # Project it onto the reflection-even subspace instead.
            gate_logit = 0.5 * (
                self.gate_net(avoidance_input)
                + self.gate_net(mirrored_avoidance_input))
        elif self.config.human_conditioned_residual:
            # Keep the historical layer shapes/checkpoint layout, but do not
            # let the goal-conditioned joint token become an independent
            # one-sided navigation branch. Ego retains task/boundary context,
            # so the Human response can still choose a feasible side.
            zero_joint = torch.zeros_like(joint)
            avoidance_input = torch.cat((ego, zero_joint, human), dim=-1)
            no_human_input = torch.cat((
                ego, zero_joint, torch.zeros_like(human)), dim=-1)
            residual_logit = (
                self.avoid_net(avoidance_input)
                - self.avoid_net(no_human_input))
        else:
            avoidance_input = torch.cat((ego, joint, human), dim=-1)
            residual_logit = self.avoid_net(avoidance_input)
        if not self.config.human_reflection_equivariant:
            gate_logit = self.gate_net(avoidance_input)
        goal_mean = self.goal_net(goal_input)
        residual = torch.tanh(residual_logit)
        gate = torch.sigmoid(gate_logit)
        raw_pre_tanh_mean = (
            goal_mean
            + gate * float(self.config.residual_bound) * residual
        )
        pre_tanh_mean = self._policy_mean(raw_pre_tanh_mean)
        mean_bound_jacobian = self._policy_mean_bound_jacobian(
            raw_pre_tanh_mean, pre_tanh_mean)
        if self.config.human_reflection_equivariant:
            # Exploration scale is reflection-invariant as well.  Joint/Human
            # learned suffixes are excluded here for the same reason as above;
            # the goal and Ego context still determine the supported scale.
            std_input = torch.cat((
                goal, ego, torch.zeros_like(joint), physical_human), dim=-1)
            mirrored_std_input = torch.cat((
                goal, ego, torch.zeros_like(joint), mirrored_human), dim=-1)
            raw_std_residual = 0.5 * (
                self.std_net(std_input) + self.std_net(mirrored_std_input))
        else:
            raw_std_residual = self.std_net(feature)
        std, progress = self._policy_std(raw_std_residual)
        distribution = TanhNormal(pre_tanh_mean, std)
        distribution.goal_mean = goal_mean
        distribution.avoidance_residual = residual
        distribution.conflict_gate = gate
        distribution.conflict_gate_logit = gate_logit
        distribution.raw_pre_tanh_mean = raw_pre_tanh_mean
        distribution.pre_tanh_mean = pre_tanh_mean
        distribution.mean_bound_jacobian = mean_bound_jacobian
        distribution.raw_std_residual = raw_std_residual
        distribution.std_schedule_progress = progress
        return distribution
