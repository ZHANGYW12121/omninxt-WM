"""Three-stream observation encoder for the UAV crowd world model.

The encoder keeps the three information streams structured:

1. human stream: causal pose tokens enhanced by BEV cross-attention
2. environment stream: PointPillars BEV tokens/map from LiDAR
3. ego stream: drone state MLP embedding

It does not collapse everything into a single embedding internally.  Downstream
RSSM adapters can decide how to consume ``human_embed``, ``env_embed`` and
``ego_embed`` while prediction/reconstruction heads can still use token/map
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from modules.causal_pose_encoder import CausalMultiPersonSTGCNEncoder
from modules.pointpillars_bev import EgoPointPillarsBEVEncoder, EgoPointPillarsConfig
from modules.structured_posterior import DEFAULT_STRUCTURED_EMBED_KEYS, flatten_structured_embed


@dataclass(frozen=True)
class ThreeStreamEncoderConfig:
    """Default dimensions for the structured world-model encoder."""

    model_dim: int = 256
    ego_state_dim: int = 17
    ego_hidden_dim: int = 256
    pose_hidden_channels: int = 64
    pose_token_dim: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.0
    pointpillars: EgoPointPillarsConfig = EgoPointPillarsConfig()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    mask_f = mask.to(dtype=x.dtype, device=x.device)
    denom = mask_f.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=dim) / denom


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or dim * 4)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with residual MLP."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = ResidualMLP(dim, dropout=dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Enhance query tokens with context tokens.

        Args:
            query_tokens: ``[N, Q, D]``.
            context_tokens: ``[N, K, D]``.
            query_mask: optional ``[N, Q]`` true for valid queries.
            context_mask: optional ``[N, K]`` true for valid keys/values.
        """

        if query_tokens.numel() == 0:
            return query_tokens

        if context_tokens.shape[1] == 0:
            out = query_tokens
        else:
            key_padding_mask = None
            if context_mask is not None:
                # MultiheadAttention uses True for ignored keys.
                key_padding_mask = ~context_mask.to(dtype=torch.bool, device=context_tokens.device)
                # If all keys are masked for a sample, unmask them to avoid NaNs.
                all_masked = key_padding_mask.all(dim=1)
                if all_masked.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_masked] = False

            attn_out, _ = self.attn(
                self.q_norm(query_tokens),
                self.kv_norm(context_tokens),
                self.kv_norm(context_tokens),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            out = query_tokens + attn_out
        out = self.ffn(out)
        if query_mask is not None:
            out = out * query_mask.to(dtype=out.dtype, device=out.device)[..., None]
        return out


class EgoStateEncoder(nn.Module):
    """MLP encoder for UAV state vectors."""

    def __init__(self, in_dim: int = 17, hidden_dim: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, ego_state: torch.Tensor) -> torch.Tensor:
        if ego_state.shape[-1] <= 0:
            raise ValueError("ego_state must have a non-empty last dimension.")
        return self.net(ego_state.float())


class ThreeStreamWorldModelEncoder(nn.Module):
    """Structured human/environment/ego encoder for the world model."""

    def __init__(self, config: ThreeStreamEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ThreeStreamEncoderConfig()
        dim = int(self.config.model_dim)

        self.pose_encoder = CausalMultiPersonSTGCNEncoder(
            hidden_channels=int(self.config.pose_hidden_channels),
            out_dim=int(self.config.pose_token_dim),
            dropout=float(self.config.dropout),
        )
        self.pointpillars = EgoPointPillarsBEVEncoder(self.config.pointpillars)
        self.ego_encoder = EgoStateEncoder(
            in_dim=int(self.config.ego_state_dim),
            hidden_dim=int(self.config.ego_hidden_dim),
            out_dim=dim,
        )

        self.pose_proj = nn.Sequential(
            nn.LayerNorm(int(self.config.pose_token_dim)),
            nn.Linear(int(self.config.pose_token_dim), dim),
        )
        self.bev_proj = nn.Sequential(
            nn.LayerNorm(int(self.pointpillars.out_dim)),
            nn.Linear(int(self.pointpillars.out_dim), dim),
        )
        self.human_bev_cross_attn = CrossAttentionBlock(
            dim=dim,
            num_heads=int(self.config.num_attention_heads),
            dropout=float(self.config.dropout),
        )
        self.human_pool_norm = nn.LayerNorm(dim)
        self.env_pool_norm = nn.LayerNorm(dim)
        self.rssm_embed_size = {key: dim for key in DEFAULT_STRUCTURED_EMBED_KEYS}
        self.out_dim = sum(self.rssm_embed_size.values())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode one collated offline batch.

        Required batch keys:

        - ``pose_windows``, ``pose_window_mask``, ``pose_token_mask``
        - ``lidar``, ``lidar_mask``
        - ``ego_state``
        """

        pose_out = self.pose_encoder(
            batch["pose_windows"],
            pose_window_mask=batch.get("pose_window_mask"),
            pose_token_mask=batch.get("pose_token_mask"),
            image_size_hw=batch.get("pose_image_size_hw"),
        )
        raw_pose_tokens = pose_out["pose_tokens"]
        human_mask = pose_out["pose_token_mask"]
        human_query = self.pose_proj(raw_pose_tokens)

        bev_out = self.pointpillars(batch["lidar"], point_mask=batch.get("lidar_mask"))
        raw_bev_tokens = bev_out["bev_tokens"]
        env_tokens = self.bev_proj(raw_bev_tokens)
        env_mask = bev_out["bev_token_mask"]

        if human_query.dim() != 4 or env_tokens.dim() != 4:
            raise ValueError(
                "ThreeStreamWorldModelEncoder expects time-major batch tensors: "
                f"human={tuple(human_query.shape)}, env={tuple(env_tokens.shape)}"
            )
        b, t, m, d = human_query.shape
        _, _, n, _ = env_tokens.shape

        flat_human = human_query.reshape(b * t, m, d)
        flat_env = env_tokens.reshape(b * t, n, d)
        flat_human_mask = human_mask.reshape(b * t, m)
        flat_env_mask = env_mask.reshape(b * t, n)
        human_tokens = self.human_bev_cross_attn(
            flat_human,
            flat_env,
            query_mask=flat_human_mask,
            context_mask=flat_env_mask,
        ).reshape(b, t, m, d)

        human_embed = self.human_pool_norm(_masked_mean(human_tokens, human_mask, dim=2))
        env_embed = self.env_pool_norm(_masked_mean(env_tokens, env_mask, dim=2))
        ego_embed = self.ego_encoder(batch["ego_state"])

        return {
            # Human stream.
            "human_tokens": human_tokens,
            "human_embed": human_embed,
            "human_token_mask": human_mask,
            "raw_pose_tokens": raw_pose_tokens,
            # Environment stream.
            "env_tokens": env_tokens,
            "env_embed": env_embed,
            "env_token_mask": env_mask,
            "bev_map": bev_out["bev_feature"],
            "pillar_bev": bev_out["pillar_bev"],
            "pillar_occupancy": bev_out["pillar_occupancy"],
            "bev_hw": bev_out["bev_hw"],
            "bev_range": bev_out["bev_range"],
            # Ego stream.
            "ego_embed": ego_embed,
        }

    def flatten_for_aux(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten human/env/ego embeds for auxiliary embedding losses."""

        return flatten_structured_embed(encoded, keys=DEFAULT_STRUCTURED_EMBED_KEYS)


def encode_three_stream_batch(
    encoder: ThreeStreamWorldModelEncoder,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convenience wrapper for batches returned by ``crowd_collate``."""

    return encoder(batch)
