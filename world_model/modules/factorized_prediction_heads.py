"""Ego/Human prediction heads for the two-branch world model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from modules.skeleton_topology import (
    COCO12_BODY_JOINT_COUNT, hip_joint_indices,
    project_coco12_geometry_torch,
)
from modules.se2_relative_dynamics import analytic_relative_human_step


@dataclass(frozen=True)
class FactorizedPredictionConfig:
    hidden_dim: int = 256
    ego_dim: int = 14
    num_joints: int = COCO12_BODY_JOINT_COUNT
    smooth_l1_beta: float = 0.05
    yaw_sin_index: int = 12
    yaw_cos_index: int = 13
    yaw_unit_weight: float = 0.1
    # Compatibility defaults to the historical two-residual decoder.  The
    # v18 Human-repair config enables velocity-only integration so position
    # and velocity can no longer contradict one another.
    kinematic_velocity_only: bool = False
    # Compact-v3 measured Human velocities show that 99% of per-axis
    # corrections to the analytic constant-velocity transition lie within
    # roughly 0.35 m/s.  Keep the learned term a bounded correction instead of
    # allowing it to dominate recursively imagined motion.
    max_velocity_residual_mps: float = 0.35
    # Optional absolute vector-speed bounds.  They are left disabled for
    # legacy configs; the production pure-Dreamer contract supplies values
    # derived from simulator limits and train-only replay quantiles.
    max_root_speed_mps: float | None = None
    # v18 also advances every observed joint from its measured body-frame
    # velocity.  The learned output is a bounded velocity correction, not a
    # freshly decoded absolute skeleton at every recursive step.
    kinematic_joint_velocity_only: bool = False
    max_joint_velocity_residual_mps: float = 0.35
    max_joint_speed_mps: float | None = None


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype, device=value.device)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


class EgoPredictionHead(nn.Module):
    """Decode current Ego14 and predict its next-step delta."""

    def __init__(self, feat_dim: int, config: FactorizedPredictionConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(feat_dim, config.hidden_dim)
        self.current = nn.Linear(config.hidden_dim, config.ego_dim)
        self.delta = nn.Linear(config.hidden_dim, config.ego_dim)

    def _normalize_yaw(self, state: torch.Tensor) -> torch.Tensor:
        sin_i, cos_i = self.config.yaw_sin_index, self.config.yaw_cos_index
        yaw = state[..., [sin_i, cos_i]]
        norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
        normalized = yaw / norm.clamp_min(1e-6)
        fallback = torch.zeros_like(normalized)
        fallback[..., 1] = 1.0
        yaw = torch.where(norm > 1e-6, normalized, fallback)
        state = state.clone()
        state[..., sin_i], state[..., cos_i] = yaw[..., 0], yaw[..., 1]
        return state

    def decode_current(self, feat: torch.Tensor) -> torch.Tensor:
        return self._normalize_yaw(self.current(self.trunk(feat)))

    def forward(self, feat: torch.Tensor, current_state: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(feat)
        reconstruction = self._normalize_yaw(self.current(hidden))
        delta = self.delta(hidden)
        predicted = self._normalize_yaw(current_state.float() + delta)
        return {"current_state": reconstruction, "delta": delta, "next_state": predicted}

    def loss(self, pred, target, sequence_valid=None):
        row_valid = (
            torch.ones_like(target[..., :1], dtype=torch.bool)
            if sequence_valid is None else sequence_valid.bool())
        transition_valid = row_valid[:, :-1] & row_valid[:, 1:]
        reconstruction_raw = F.smooth_l1_loss(
            pred["current_state"], target.float(),
            beta=self.config.smooth_l1_beta, reduction="none",
        )
        reconstruction = _masked_mean(reconstruction_raw, row_valid)
        next_state_raw = F.smooth_l1_loss(
            pred["next_state"][:, :-1], target[:, 1:].float(),
            beta=self.config.smooth_l1_beta, reduction="none",
        )
        next_state = _masked_mean(next_state_raw, transition_valid)
        raw_next = target[:, :-1].float() + pred["delta"][:, :-1]
        sin_i, cos_i = self.config.yaw_sin_index, self.config.yaw_cos_index
        yaw_unit = (
            raw_next[..., sin_i].square() + raw_next[..., cos_i].square() - 1.0
        ).square()
        yaw_unit = _masked_mean(
            yaw_unit, transition_valid.reshape(*yaw_unit.shape))
        return {"ego_recon": reconstruction, "ego_pred": next_state, "yaw_unit": yaw_unit}


class Human3DPredictionHead(nn.Module):
    _NOMINAL_TRANSITION_DT_S = 0.1

    def __init__(self, feat_dim: int, config: FactorizedPredictionConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(feat_dim, config.hidden_dim)
        self.root_delta = nn.Linear(config.hidden_dim, 3)
        self.velocity_delta = nn.Linear(config.hidden_dim, 3)
        if config.kinematic_velocity_only:
            # Keep the tensor in state_dict so v17 checkpoints load exactly,
            # but remove it from the new optimizer contract and forward graph.
            self.root_delta.requires_grad_(False)
        self.joints = nn.Linear(config.hidden_dim, config.num_joints * 3)
        self.survival = nn.Linear(config.hidden_dim, 1)
        # Birth is diagnostic until a new slot state can be initialized from
        # an observation/occupancy token.  It is deliberately not used to
        # invent an unobserved 3-D person during Actor imagination.
        self.birth = nn.Linear(config.hidden_dim, 1)
        if float(config.max_velocity_residual_mps) <= 0.0:
            raise ValueError("max_velocity_residual_mps must be positive")
        if float(config.max_joint_velocity_residual_mps) <= 0.0:
            raise ValueError(
                "max_joint_velocity_residual_mps must be positive")
        for name, value in (
            ("max_root_speed_mps", config.max_root_speed_mps),
            ("max_joint_speed_mps", config.max_joint_speed_mps),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        # The structured baseline is already a valid constant-velocity model.
        # A fresh model must start exactly at that baseline and learn only the
        # correction supported by replay, rather than injecting a random drift
        # that compounds over a 15-step Dreamer rollout.
        nn.init.zeros_(self.velocity_delta.weight)
        nn.init.zeros_(self.velocity_delta.bias)
        if config.kinematic_joint_velocity_only:
            nn.init.zeros_(self.joints.weight)
            nn.init.zeros_(self.joints.bias)

    def forward(self, feat, current_root, *, current_ego=None, next_ego=None,
                current_joints=None, current_joint_velocity=None,
                current_joint_mask=None,
                lifecycle_feat=None,
                dt_s: float | torch.Tensor = 0.1):
        """Predict one physical transition from destination latent state.

        ``feat`` is the posterior/prior state at the destination row.  This
        makes the transition selected by the source action participate in the
        h1 prediction instead of delaying learned Human dynamics by one tick.
        Slot survival and birth are source-state events, so their logits are
        decoded from ``lifecycle_feat`` when it is supplied.
        """
        hidden = self.trunk(feat)
        dt = torch.as_tensor(
            dt_s, dtype=hidden.dtype, device=hidden.device)
        if not torch.isfinite(dt).all() or bool((dt <= 0.0).any()):
            raise ValueError("Human prediction dt_s must be positive")
        while dt.ndim < hidden.ndim:
            dt = dt.unsqueeze(-1)
        try:
            residual_exposure = torch.broadcast_to(
                dt, hidden.shape[:-1] + (1,)
            ) / float(self._NOMINAL_TRANSITION_DT_S)
        except RuntimeError as error:
            raise ValueError(
                "Human prediction dt_s cannot broadcast to slots") from error
        # The learned output is a bounded change in velocity over one nominal
        # 100 ms transition.  Compact-v3 terminal rows can be force-written as
        # early as 16 ms; applying a full velocity jump on those rows teaches
        # an unphysical, duration-dependent Human model that is then rolled at
        # 100 ms by the Actor.  Interpret the correction as a bounded
        # acceleration integrated over the actual exposure instead.
        velocity_residual = (
            float(self.config.max_velocity_residual_mps)
            * torch.tanh(self.velocity_delta(hidden))
            * residual_exposure
        )
        root_position = current_root[..., :3].float()
        if current_root.shape[-1] >= 6:
            root_velocity = current_root[..., 3:6].float()
        else:
            root_velocity = torch.zeros_like(root_position)
        if self.config.kinematic_velocity_only:
            position_residual = torch.zeros_like(root_position)
        else:
            position_residual = self.root_delta(hidden)
        if current_ego is not None or next_ego is not None:
            if current_ego is None or next_ego is None:
                raise ValueError("current_ego and next_ego must be supplied together")
            root, velocity = analytic_relative_human_step(
                root_position, root_velocity, current_ego, next_ego,
                dt_s=dt_s,
                position_residual_next_body=(
                    None if self.config.kinematic_velocity_only
                    else position_residual
                ),
                velocity_residual_current_body=velocity_residual,
                maximum_velocity_mps=self.config.max_root_speed_mps,
            )
        else:
            velocity = root_velocity + velocity_residual
            if self.config.max_root_speed_mps is not None:
                speed = torch.linalg.vector_norm(
                    velocity, dim=-1, keepdim=True)
                velocity = velocity * torch.clamp(
                    velocity.new_tensor(float(self.config.max_root_speed_mps))
                    / speed.clamp_min(1.0e-8),
                    max=1.0,
                )
            root = (
                root_position + velocity * dt
                if self.config.kinematic_velocity_only
                else root_position + position_residual
            )
        joint_output = self.joints(hidden).reshape(
            *hidden.shape[:-1], self.config.num_joints, 3)
        if self.config.kinematic_joint_velocity_only:
            if current_joints is None:
                raise ValueError(
                    "kinematic joint prediction requires current_joints")
            if current_joints.shape != joint_output.shape:
                raise ValueError(
                    "current_joints must match Human slots and joint count")
            if current_joint_velocity is None:
                current_joint_velocity = torch.zeros_like(current_joints)
            if current_joint_velocity.shape != current_joints.shape:
                raise ValueError(
                    "current_joint_velocity must match current_joints")
            if current_joint_mask is None:
                current_joint_mask = torch.ones_like(
                    current_joints[..., 0], dtype=torch.bool)
            if current_joint_mask.shape != current_joints.shape[:-1]:
                raise ValueError(
                    "current_joint_mask must match current_joints")
            joint_velocity_residual = (
                float(self.config.max_joint_velocity_residual_mps)
                * torch.tanh(joint_output)
                * residual_exposure[..., None]
            )
            if current_ego is not None or next_ego is not None:
                if current_ego is None or next_ego is None:
                    raise ValueError(
                        "current_ego and next_ego must be supplied together")
                leading = current_joints.shape[:-3]
                flat_shape = (*leading, -1, 3)
                joints, joint_velocity = analytic_relative_human_step(
                    current_joints.float().reshape(flat_shape),
                    current_joint_velocity.float().reshape(flat_shape),
                    current_ego,
                    next_ego,
                    dt_s=dt_s,
                    velocity_residual_current_body=(
                        joint_velocity_residual.reshape(flat_shape)),
                    maximum_velocity_mps=self.config.max_joint_speed_mps,
                )
                joints = joints.reshape_as(current_joints)
                joint_velocity = joint_velocity.reshape_as(
                    current_joint_velocity)
            else:
                joint_velocity = (
                    current_joint_velocity.float()
                    + joint_velocity_residual)
                if self.config.max_joint_speed_mps is not None:
                    speed = torch.linalg.vector_norm(
                        joint_velocity, dim=-1, keepdim=True)
                    joint_velocity = joint_velocity * torch.clamp(
                        joint_velocity.new_tensor(
                            float(self.config.max_joint_speed_mps))
                        / speed.clamp_min(1.0e-8),
                        max=1.0,
                    )
                joints = (
                    current_joints.float()
                    + joint_velocity * dt[..., None])
            unprojected_joints = joints
            joints = project_coco12_geometry_torch(
                joints,
                current_joint_mask,
                root,
            )
            # Keep the recursive velocity state consistent with the physical
            # position actually emitted by the projection.  This correction
            # is expressed in the destination body frame and remains inside
            # the same absolute articulated-speed ball.
            joint_velocity = joint_velocity + (
                (joints - unprojected_joints) / dt[..., None])
            if self.config.max_joint_speed_mps is not None:
                speed = torch.linalg.vector_norm(
                    joint_velocity, dim=-1, keepdim=True)
                joint_velocity = joint_velocity * torch.clamp(
                    joint_velocity.new_tensor(
                        float(self.config.max_joint_speed_mps))
                    / speed.clamp_min(1.0e-8),
                    max=1.0,
                )
            relative = joints - root[..., None, :]
        else:
            relative = joint_output
            joints = root[..., None, :] + relative
            joint_velocity = torch.zeros_like(joints)
            joint_velocity_residual = torch.zeros_like(joints)
        lifecycle_hidden = (
            hidden if lifecycle_feat is None else self.trunk(lifecycle_feat))
        survival_logit = self.survival(lifecycle_hidden).squeeze(-1)
        # Empty Human slots are hard-zeroed by the RSSM, so the current birth
        # head can identify only a global empty-slot base rate; it has no
        # observation token from which to infer a new person's geometry or
        # direction.  Keep that useful calibration diagnostic trainable, but
        # do not let its necessarily unconditioned BCE update the shared
        # motion/survival trunk used by the 15-step Human rollout.
        birth_logit = self.birth(lifecycle_hidden.detach()).squeeze(-1)
        return {
            "root_delta": position_residual,
            "root_residual_next_body": position_residual,
            "velocity_residual_current_body": velocity_residual,
            "root_velocity": velocity,
            "root": root, "joint_relative": relative,
            "joints": joints,
            "joint_velocity": joint_velocity,
            "joint_velocity_residual_current_body": joint_velocity_residual,
            "survival_logit": survival_logit,
            # Compatibility alias for older diagnostics/checkpoints.  Its
            # semantics are now explicitly source-slot survival.
            "presence_logit": survival_logit,
            "birth_logit": birth_logit,
        }

    @staticmethod
    def _contract(batch, human_mask):
        """Return source-aligned identity and lifecycle supervision masks."""
        if "human_motion_valid" in batch:
            motion = batch["human_motion_valid"][:, :-1].bool()
        elif "human_ids" in batch:
            motion = (
                human_mask[:, :-1].bool() & human_mask[:, 1:].bool()
                & (batch["human_ids"][:, :-1] == batch["human_ids"][:, 1:])
            )
            if "human_is_first" in batch:
                motion &= ~batch["human_is_first"][:, 1:].bool()
        else:
            motion = human_mask[:, :-1].bool() & human_mask[:, 1:].bool()

        if "human_survival_valid" in batch:
            survival_valid = batch["human_survival_valid"][:, :-1].bool()
            survival_target = batch["human_survival_target"][:, :-1].bool()
        else:
            survival_valid = human_mask[:, :-1].bool()
            survival_target = motion
        return motion, survival_valid, survival_target

    def loss(self, pred, skeleton, root_target, human_mask, joint_mask, batch):
        xyz = skeleton[..., :3].float()
        motion, survival_valid, survival_target = self._contract(
            batch, human_mask)
        sequence_valid = batch.get("sequence_valid")
        if sequence_valid is not None:
            row_valid = sequence_valid.bool().reshape(
                *sequence_valid.shape[:2], -1).all(-1)
            transition_row_valid = row_valid[:, :-1] & row_valid[:, 1:]
            motion &= transition_row_valid[..., None]
            survival_valid &= transition_row_valid[..., None]
        measured_root = batch.get("measured_root_target", root_target)
        measured_root_valid = batch.get(
            "measured_root_target_valid", human_mask).bool()
        measured_velocity_valid = batch.get(
            "measured_velocity_target_valid", measured_root_valid).bool()
        measured_joints = batch.get("measured_joint_target", xyz).float()
        measured_joint_valid = batch.get(
            "measured_joint_target_valid", joint_mask).bool()
        root_valid = motion & measured_root_valid[:, 1:]
        root_raw = F.smooth_l1_loss(
            pred["root"][:, :-1], measured_root[:, 1:, ..., :3].float(), reduction="none",
            beta=self.config.smooth_l1_beta,
        )
        root_loss = _masked_mean(root_raw, root_valid)
        if measured_root.shape[-1] >= 6:
            velocity_valid = motion & measured_velocity_valid[:, 1:]
            velocity_raw = F.smooth_l1_loss(
                pred["root_velocity"][:, :-1],
                measured_root[:, 1:, ..., 3:6].float(), reduction="none",
                beta=self.config.smooth_l1_beta,
            )
            velocity_loss = _masked_mean(velocity_raw, velocity_valid)
        else:
            velocity_loss = root_loss.detach() * 0.0
        # This kinematic decoder integrates a joint only when its source state
        # exists. A newly measured destination joint has no causal initial
        # position or velocity, so supervising it is mathematically
        # unreachable and pushes the bounded residual toward saturation.
        joint_valid = (
            joint_mask[:, :-1].bool()
            & measured_joint_valid[:, 1:]
            & motion[..., None]
        )
        joint_error = torch.linalg.vector_norm(
            pred["joints"][:, :-1] - measured_joints[:, 1:], dim=-1,
        )
        mpjpe = _masked_mean(joint_error, joint_valid)
        survival_raw = F.binary_cross_entropy_with_logits(
            pred["survival_logit"][:, :-1], survival_target.float(), reduction="none",
        )
        survival = _masked_mean(survival_raw, survival_valid)
        birth_target = batch.get("human_birth_target")
        if birth_target is None:
            birth = survival.detach() * 0.0
        else:
            birth_raw = F.binary_cross_entropy_with_logits(
                pred["birth_logit"][:, :-1],
                birth_target[:, 1:].float(), reduction="none")
            # A birth is defined for a previously empty slot. Existing slots
            # are handled exclusively by survival, preventing contradictory
            # positive lifecycle labels.
            birth_valid = batch.get("human_birth_valid")
            if birth_valid is None:
                birth_valid = ~human_mask[:, :-1].bool()
            else:
                birth_valid = birth_valid[:, 1:].bool()
            if sequence_valid is not None:
                birth_valid &= transition_row_valid[..., None]
            birth = _masked_mean(birth_raw, birth_valid)
        losses = {
            "human_root": root_loss,
            "human_mpjpe": mpjpe,
            # Keep the historical key so older objectives/checkpoints remain
            # loadable; its definition is now the correctly masked survival
            # BCE, not next-row occupancy BCE.
            "human_presence": survival,
        }
        if birth_target is not None:
            losses["human_birth"] = birth
        if measured_root.shape[-1] >= 6:
            losses["human_velocity"] = velocity_loss
        return losses


class FactorizedPredictionHeads(nn.Module):
    def __init__(self, ego_feat_dim, human_feat_dim,
                 config: FactorizedPredictionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FactorizedPredictionConfig()
        self.ego = EgoPredictionHead(ego_feat_dim, self.config)
        self.human = Human3DPredictionHead(human_feat_dim, self.config)

    def decode_ego_state(self, ego_feat: torch.Tensor) -> torch.Tensor:
        return self.ego.decode_current(ego_feat)

    def forward_loss(self, branch_feats, batch):
        skeleton = batch["skeleton"]
        if "human_root" in batch:
            root = batch["human_root"].float()
        else:
            hips = hip_joint_indices(skeleton.shape[-2])
            if hips is None:
                raise ValueError("human_root is required for an unsupported joint topology")
            left_hip, right_hip = hips
            root_position = 0.5 * (
                skeleton[..., left_hip, :3] + skeleton[..., right_hip, :3])
            root = torch.cat((
                root_position, torch.zeros_like(root_position)), dim=-1)
        next_ego = torch.cat((
            batch["ego_state"][:, 1:], batch["ego_state"][:, -1:]), dim=1)
        if "dt_s" in batch:
            recorded_dt = batch["dt_s"].float()
            nominal_dt = recorded_dt.new_full(
                recorded_dt[:, -1:].shape, 0.1)
            next_dt = torch.cat((recorded_dt[:, 1:], nominal_dt), dim=1)
            # Artificial/padded rows are excluded by the loss masks. Keep the
            # forward finite without reinterpreting a valid zero-duration row.
            next_dt = torch.where(
                torch.isfinite(next_dt) & next_dt.gt(0.0),
                next_dt,
                torch.full_like(next_dt, 0.1),
            )
        else:
            next_dt = 0.1
        # Motion for t -> t+1 is decoded from posterior state t+1.  Its prior
        # counterpart is exactly what factual/Actor imagination obtains after
        # img_step(action[t+1]), so one-step and recursive training have the
        # same temporal contract.  The duplicated tail is excluded by loss.
        human_destination_feat = torch.cat((
            branch_feats["human"][:, 1:],
            branch_feats["human"][:, -1:],
        ), dim=1)
        pred = {
            "ego": self.ego(branch_feats["ego"], batch["ego_state"]),
            "human": self.human(
                human_destination_feat, root,
                current_ego=batch["ego_state"], next_ego=next_ego,
                current_joints=skeleton[..., :3],
                current_joint_velocity=skeleton[..., 3:6],
                current_joint_mask=batch["joint_mask"],
                lifecycle_feat=branch_feats["human"],
                dt_s=next_dt,
            ),
        }
        losses = {}
        losses.update(self.ego.loss(
            pred["ego"], batch["ego_state"], batch.get("sequence_valid")))
        losses.update(self.human.loss(
            pred["human"], skeleton, root, batch["human_mask"], batch["joint_mask"],
            batch,
        ))
        return pred, losses
