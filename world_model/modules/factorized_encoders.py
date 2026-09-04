"""Observation encoders for the two-branch Ego--Human world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from modules.causal_pose_encoder import CausalMultiPersonSTGCNEncoder
from modules.skeleton_topology import COCO12_BODY_JOINT_COUNT, hip_joint_indices
from modules.sparse_ego_human_attention import (
    SparseEgoHumanAttention,
    SparseEgoHumanAttentionConfig,
)


@dataclass(frozen=True)
class FactorizedEncoderConfig:
    model_dim: int = 128
    ego_state_dim: int = 14
    ego_hidden_dim: int = 128
    ego_state_mean: tuple[float, ...] | None = None
    ego_state_std: tuple[float, ...] | None = None
    human_root_dim: int = 10
    human_feat_dim: int = 7
    human_hidden_dim: int = 128
    human_quality_dim: int = 7
    # Fresh online/offline runs use the same fixed physical scaling.  The old
    # Ego path standardized and then applied a per-row LayerNorm, making it
    # invariant to part of the metric position/velocity magnitude and giving
    # live identity-normalized runs a different representation from offline.
    ego_metric_scaling: bool = False
    # Compatibility defaults to the historical per-sample LayerNorm. v18 turns
    # this on so absolute metric root distances and velocities are identifiable.
    human_root_metric_scaling: bool = False
    human_quality_metric_scaling: bool = False
    use_stgcn: bool = True
    pose_history: int = 8
    observation_heads: int = 4
    observation_ff_mult: int = 4
    dropout: float = 0.0


class Ego14Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 mean: tuple[float, ...] | None = None,
                 std: tuple[float, ...] | None = None,
                 *, metric_scaling: bool = False) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        mean_tensor = torch.zeros(in_dim) if mean is None else torch.tensor(mean, dtype=torch.float32)
        std_tensor = torch.ones(in_dim) if std is None else torch.tensor(std, dtype=torch.float32)
        if mean_tensor.numel() != in_dim or std_tensor.numel() != in_dim or (std_tensor <= 0).any():
            raise ValueError("ego training mean/std must contain one valid value per input dimension")
        self.register_buffer("input_mean", mean_tensor)
        self.register_buffer("input_std", std_tensor)
        self.metric_scaling = bool(metric_scaling)
        if self.metric_scaling:
            if self.in_dim != 14:
                raise ValueError("metric Ego scaling requires Ego14")
            # episode xyz; body/world velocity; body/world acceleration; AGL;
            # roll, pitch, sin(yaw), cos(yaw).  Scales cover the audited task
            # contract while clipping only corrupt/extreme recorder spikes.
            self.register_buffer("metric_scale", torch.tensor((
                32.0, 12.0, 1.0,
                3.0, 3.0, 2.0,
                8.0, 8.0, 8.0,
                2.0, 1.0, 1.0, 1.0, 1.0,
            )))
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim),
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.in_dim:
            raise ValueError(f"ego_state must end in {self.in_dim}, got {tuple(state.shape)}")
        normalized = (
            (state.float() / self.metric_scale).clamp(-5.0, 5.0)
            if self.metric_scaling else
            (state.float() - self.input_mean) / self.input_std.clamp_min(1e-6)
        )
        return self.net(normalized)


def root_and_root_relative_joints(
    skeleton: torch.Tensor,
    human_mask: torch.Tensor,
    joint_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split absolute Ego-frame skeletons into root and root-relative inputs.

    The root feature is ``[position(3), velocity(3), extent(3), confidence]``.
    Joint features remain seven dimensional, but position and velocity are
    expressed relative to the person's root.  All axes remain Ego-frame axes.
    """

    if skeleton.ndim != 5 or skeleton.shape[-1] != 7:
        raise ValueError("skeleton must be [B,T,N,J,7]")
    b, t, n, j, _ = skeleton.shape
    if human_mask.shape != (b, t, n):
        raise ValueError("human_mask must be [B,T,N]")
    if joint_mask is None:
        joint_mask = human_mask[..., None].expand(b, t, n, j)
    if joint_mask.shape != (b, t, n, j):
        raise ValueError("joint_mask must be [B,T,N,J]")
    valid = joint_mask.bool() & human_mask.bool()[..., None]
    xyz, velocity, confidence = skeleton[..., :3].float(), skeleton[..., 3:6].float(), skeleton[..., 6].float()
    weight = valid.to(xyz.dtype)[..., None]
    denom = weight.sum(dim=-2).clamp_min(1.0)
    mean_pos = (xyz * weight).sum(dim=-2) / denom
    mean_vel = (velocity * weight).sum(dim=-2) / denom
    hips = hip_joint_indices(j)
    if hips is not None:
        left_hip, right_hip = hips
        hips_valid = valid[..., left_hip] & valid[..., right_hip]
        hip_pos = 0.5 * (xyz[..., left_hip, :] + xyz[..., right_hip, :])
        hip_vel = 0.5 * (velocity[..., left_hip, :] + velocity[..., right_hip, :])
        root_pos = torch.where(hips_valid[..., None], hip_pos, mean_pos)
        root_vel = torch.where(hips_valid[..., None], hip_vel, mean_vel)
    else:
        root_pos, root_vel = mean_pos, mean_vel

    positive_inf = torch.full_like(xyz, torch.inf)
    low = torch.where(valid[..., None], xyz, positive_inf).amin(dim=-2)
    high = torch.where(valid[..., None], xyz, -positive_inf).amax(dim=-2)
    extent = torch.where(valid.any(-1)[..., None], high - low, torch.zeros_like(low))
    root_conf = (
        (confidence * valid.to(confidence.dtype)).sum(-1)
        / valid.sum(-1).clamp_min(1).to(confidence.dtype)
    )
    root = torch.cat((root_pos, root_vel, extent, root_conf[..., None]), dim=-1)
    root = root.masked_fill(~human_mask.bool()[..., None], 0.0)

    relative = skeleton.float().clone()
    relative[..., :3] = xyz - root_pos[..., None, :]
    relative[..., 3:6] = velocity - root_vel[..., None, :]
    relative = relative.masked_fill(~valid[..., None], 0.0)
    return root, relative, valid


class CausalPerPersonGRUEncoder(nn.Module):
    """Causal fallback pose encoder with shared weights for all Human slots."""

    def __init__(self, feat_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.joint_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, skeleton: torch.Tensor, human_mask: torch.Tensor,
                joint_mask: torch.Tensor | None = None,
                human_is_first: torch.Tensor | None = None) -> torch.Tensor:
        del human_is_first
        b, t, n, j, f = skeleton.shape
        if f != self.feat_dim or human_mask.shape != (b, t, n):
            raise ValueError("skeleton feature dimension or human_mask shape mismatch")
        if joint_mask is None:
            joint_mask = human_mask[..., None].expand(b, t, n, j)
        valid = joint_mask.bool() & human_mask.bool()[..., None]
        joint = self.joint_mlp(skeleton.float()).masked_fill(~valid[..., None], 0.0)
        person = joint.sum(dim=3) / valid.sum(dim=3, keepdim=True).clamp_min(1).to(joint.dtype)
        person = person.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        token, _ = self.gru(person)
        token = self.out(token).reshape(b, n, t, -1).permute(0, 2, 1, 3)
        return token.masked_fill(~human_mask.bool()[..., None], 0.0)


class CausalSTGCNHumanEncoder(nn.Module):
    """Build reset-safe past-only windows and encode each person independently."""

    def __init__(self, feat_dim: int, hidden_dim: int, out_dim: int,
                 num_joints: int = COCO12_BODY_JOINT_COUNT, history: int = 8) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_joints = int(num_joints)
        self.history = int(history)
        self.encoder = CausalMultiPersonSTGCNEncoder(
            num_joints=self.num_joints, in_channels=self.feat_dim,
            hidden_channels=hidden_dim, out_dim=out_dim,
        )

    def forward(self, skeleton: torch.Tensor, human_mask: torch.Tensor,
                joint_mask: torch.Tensor | None = None,
                human_is_first: torch.Tensor | None = None) -> torch.Tensor:
        if skeleton.ndim != 5:
            raise ValueError("skeleton must be [B,T,N,J,F]")
        b, t, n, j, f = skeleton.shape
        if j != self.num_joints or f != self.feat_dim or human_mask.shape != (b, t, n):
            raise ValueError("ST-GCN skeleton/mask shape mismatch")
        if joint_mask is None:
            joint_mask = human_mask[..., None].expand(b, t, n, j)
        valid_joint = joint_mask.bool() & human_mask.bool()[..., None]
        clean = skeleton.float().masked_fill(~valid_joint[..., None], 0.0)
        if human_is_first is None:
            human_is_first = torch.zeros_like(human_mask, dtype=torch.bool)
            human_is_first[:, 0] = human_mask[:, 0]
        windows = clean.new_zeros((b, t, n, self.history, j, f))
        window_mask = torch.zeros((b, t, n, self.history), dtype=torch.bool, device=clean.device)
        for now in range(t):
            segment_valid = human_mask[:, now].bool().clone()
            for offset in range(self.history):
                source = now - offset
                if source < 0:
                    break
                if offset > 0:
                    segment_valid &= ~human_is_first[:, source + 1].bool()
                valid = segment_valid & human_mask[:, source].bool()
                dst = self.history - 1 - offset
                windows[:, now, :, dst] = clean[:, source]
                window_mask[:, now, :, dst] = valid
                windows[:, now, :, dst].masked_fill_(~valid[..., None, None], 0.0)
        result = self.encoder(
            windows, pose_window_mask=window_mask, pose_token_mask=human_mask.bool(),
        )
        return result["pose_tokens"]


class HumanRootPoseEncoder(nn.Module):
    def __init__(self, config: FactorizedEncoderConfig) -> None:
        super().__init__()
        d = int(config.model_dim)
        self.metric_root_enabled = bool(config.human_root_metric_scaling)
        self.metric_quality_enabled = bool(
            config.human_quality_metric_scaling)
        self.human_quality_dim = int(config.human_quality_dim)
        if self.metric_root_enabled and int(config.human_root_dim) != 10:
            raise ValueError(
                "metric Human root encoding requires the compact-v3 "
                "[position,velocity,extent,confidence] 10-D contract")
        # A LayerNorm directly over these ten heterogeneous physical fields is
        # invariant to a per-person affine transform.  It can therefore map
        # physically different distances and speeds to the same normalized
        # input, which made the per-slot Human latent unable to retain absolute
        # root x/y.  Fixed, documented metric scales preserve both magnitude
        # and sign; LayerNorm is safe only after the learned projection.
        if self.metric_root_enabled:
            self.register_buffer(
                "root_metric_scale",
                torch.tensor(
                    (6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0),
                    dtype=torch.float32,
                ),
            )
            self.root = nn.Sequential(
                nn.Linear(config.human_root_dim, config.human_hidden_dim),
                nn.SiLU(), nn.Linear(config.human_hidden_dim, d),
                nn.LayerNorm(d),
            )
        else:
            self.root = nn.Sequential(
                nn.LayerNorm(config.human_root_dim),
                nn.Linear(config.human_root_dim, config.human_hidden_dim),
                nn.SiLU(), nn.Linear(config.human_hidden_dim, d),
                nn.LayerNorm(d),
            )
        pose_cls = CausalSTGCNHumanEncoder if config.use_stgcn else CausalPerPersonGRUEncoder
        pose_kwargs = {"history": config.pose_history} if config.use_stgcn else {}
        self.pose = pose_cls(config.human_feat_dim, config.human_hidden_dim, d, **pose_kwargs)
        if self.metric_quality_enabled:
            if self.human_quality_dim != 7:
                raise ValueError(
                    "metric Human quality encoding requires the compact-v3 "
                    "seven-field contract")
            self.register_buffer("quality_metric_scale", torch.tensor(
                (1.0, 1.0, 10.0, 2.0, 20.0, 3.0, 1.0),
                dtype=torch.float32,
            ))
            self.quality = nn.Sequential(
                nn.Linear(config.human_quality_dim, config.human_hidden_dim),
                nn.SiLU(), nn.Linear(config.human_hidden_dim, d),
                nn.LayerNorm(d),
            )
            quality_output_index = 2
        else:
            self.quality = nn.Sequential(
                nn.LayerNorm(config.human_quality_dim),
                nn.Linear(config.human_quality_dim, config.human_hidden_dim),
                nn.SiLU(), nn.Linear(config.human_hidden_dim, d),
                nn.LayerNorm(d),
            )
            quality_output_index = 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.SiLU(), nn.LayerNorm(d),
        )
        nn.init.zeros_(self.quality[quality_output_index].weight)
        nn.init.zeros_(self.quality[quality_output_index].bias)

    def forward(self, root: torch.Tensor, joints: torch.Tensor,
                human_mask: torch.Tensor, joint_mask: torch.Tensor,
                human_is_first: torch.Tensor | None,
                observation_quality: torch.Tensor | None = None) -> torch.Tensor:
        metric_root = root.float()
        if self.metric_root_enabled:
            metric_root = (metric_root / self.root_metric_scale).clamp(-5.0, 5.0)
        root_token = self.root(metric_root)
        pose_token = self.pose(joints, human_mask, joint_mask, human_is_first)
        if observation_quality is None:
            observation_quality = root.new_zeros(
                (*root.shape[:-1], self.human_quality_dim))
        quality_input = observation_quality.float()
        if self.metric_quality_enabled:
            quality_input = (
                quality_input / self.quality_metric_scale).clamp(-5.0, 5.0)
        quality_token = self.quality(quality_input)
        # Residual injection preserves the previously trained root/pose fusion
        # at migration time; the zero-initialized quality projection learns
        # only when measurement reliability is predictive.
        token = self.fusion(torch.cat((root_token, pose_token), dim=-1)) + quality_token
        return token.masked_fill(~human_mask.bool()[..., None], 0.0)


class FactorizedObservationEncoder(nn.Module):
    """Normal Ego/Human encoders followed by sparse asymmetric attention."""

    def __init__(self, config: FactorizedEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or FactorizedEncoderConfig()
        d = int(self.config.model_dim)
        self.ego_encoder = Ego14Encoder(
            self.config.ego_state_dim, self.config.ego_hidden_dim, d,
            self.config.ego_state_mean, self.config.ego_state_std,
            metric_scaling=self.config.ego_metric_scaling,
        )
        self.human_encoder = HumanRootPoseEncoder(self.config)
        self.observation_attention = SparseEgoHumanAttention(
            SparseEgoHumanAttentionConfig(
                model_dim=d, num_heads=self.config.observation_heads,
                ff_mult=self.config.observation_ff_mult,
                dropout=self.config.dropout,
            )
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        human_mask = batch["human_mask"].bool()
        joint_mask = batch.get("joint_mask")
        if "human_root" in batch and "human_joints" in batch:
            root = batch["human_root"].float()
            joints = batch["human_joints"].float()
            if joint_mask is None:
                joint_mask = human_mask[..., None].expand(*joints.shape[:-1])
        else:
            root, joints, joint_mask = root_and_root_relative_joints(
                batch["skeleton"], human_mask, joint_mask,
            )
        if root.shape[:-1] != human_mask.shape or root.shape[-1] != self.config.human_root_dim:
            raise ValueError("human_root must be [B,T,N,10]")
        if joints.shape[:-2] != human_mask.shape or joints.shape[-1] != self.config.human_feat_dim:
            raise ValueError("human_joints must be [B,T,N,J,7]")

        ego = self.ego_encoder(batch["ego_state"])[..., None, :]
        human = self.human_encoder(
            root, joints, human_mask, joint_mask, batch.get("human_is_first"),
            batch.get("human_observation_quality"),
        )
        b, t, n, d = human.shape
        attended = self.observation_attention(
            ego.reshape(b * t, 1, d), human.reshape(b * t, n, d),
            human_mask.reshape(b * t, n),
        )
        return {
            "ego_obs_token": ego,
            "human_obs_tokens": human,
            "ego_embed": attended["ego"].reshape(b, t, d),
            "human_tokens": attended["human"].reshape(b, t, n, d),
            "human_token_mask": human_mask,
            "human_root": root,
            "human_joints": joints,
            "observation_attention_weights": attended["attention_weights"].reshape(
                b, t, *attended["attention_weights"].shape[1:]
            ),
            "observation_token_mask": attended["token_mask"].reshape(
                b, t, *attended["token_mask"].shape[1:]
            ),
        }
