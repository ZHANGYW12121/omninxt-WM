"""Slot-preserving Ego--Environment--Human interaction layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class InteractionConfig:
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    ff_mult: int = 4
    dropout: float = 0.0
    human_position_dim: int = 3


class MaskedHumanAttentionPool(nn.Module):
    """Pool variable-count human slots without reading padded slots."""

    def __init__(self, model_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.empty_human = nn.Parameter(torch.zeros(model_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.empty_human, std=0.02)

    def forward(self, human_tokens: torch.Tensor, human_mask: torch.Tensor) -> torch.Tensor:
        if human_tokens.ndim != 3:
            raise ValueError(f"human_tokens must be [B,N,D], got {tuple(human_tokens.shape)}")
        if human_mask.shape != human_tokens.shape[:2]:
            raise ValueError(
                f"human_mask must be {tuple(human_tokens.shape[:2])}, got {tuple(human_mask.shape)}"
            )
        mask = human_mask.to(device=human_tokens.device, dtype=torch.bool)
        batch, people, dim = human_tokens.shape
        if people == 0:
            return self.empty_human.to(human_tokens).expand(batch, dim)

        all_empty = ~mask.any(dim=1)
        # MHA cannot accept a row with every key masked. Temporarily expose one
        # zero key, then replace that row by the learned empty-human feature.
        safe_tokens = human_tokens.masked_fill(~mask[..., None], 0.0)
        safe_mask = mask.clone()
        if all_empty.any():
            safe_mask[all_empty, 0] = True
        query = self.query.to(human_tokens).expand(batch, -1, -1)
        pooled, _ = self.attn(
            query,
            self.norm(safe_tokens),
            self.norm(safe_tokens),
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        pooled = pooled[:, 0]
        return torch.where(all_empty[:, None], self.empty_human.to(pooled), pooled)


class SlotPreservingInteraction(nn.Module):
    """Self-attend over one ego, one environment, and N human tokens.

    Token count and ordering are preserved. Human slot indices receive no
    positional embedding; only modality type and physical relative position are
    added. ``human_mask=True`` means a real person, whereas PyTorch's internal
    key-padding mask uses the opposite convention.
    """

    def __init__(self, config: InteractionConfig | None = None) -> None:
        super().__init__()
        self.config = config or InteractionConfig()
        dim = int(self.config.model_dim)
        self.modality_embedding = nn.Parameter(torch.empty(3, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.human_position = nn.Sequential(
            nn.LayerNorm(int(self.config.human_position_dim)),
            nn.Linear(int(self.config.human_position_dim), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(self.config.num_heads),
            dim_feedforward=dim * int(self.config.ff_mult),
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(self.config.num_layers),
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        ego_token: torch.Tensor,
        env_token: torch.Tensor,
        human_tokens: torch.Tensor,
        human_mask: torch.Tensor,
        human_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ego_token.ndim == 2:
            ego_token = ego_token[:, None]
        if env_token.ndim == 2:
            env_token = env_token[:, None]
        if ego_token.ndim != 3 or ego_token.shape[1] != 1:
            raise ValueError(f"ego_token must be [B,1,D], got {tuple(ego_token.shape)}")
        if env_token.ndim != 3 or env_token.shape[1] != 1:
            raise ValueError(f"env_token must be [B,1,D], got {tuple(env_token.shape)}")
        if human_tokens.ndim != 3:
            raise ValueError(f"human_tokens must be [B,N,D], got {tuple(human_tokens.shape)}")
        if human_mask.shape != human_tokens.shape[:2]:
            raise ValueError(f"human_mask shape mismatch: {tuple(human_mask.shape)}")
        if ego_token.shape[0] != human_tokens.shape[0] or env_token.shape[0] != human_tokens.shape[0]:
            raise ValueError("ego/env/human batch sizes must match")
        if ego_token.shape[-1] != self.config.model_dim or env_token.shape[-1] != self.config.model_dim:
            raise ValueError("ego/env token dimension does not match interaction model_dim")
        if human_tokens.shape[-1] != self.config.model_dim:
            raise ValueError("human token dimension does not match interaction model_dim")

        mask = human_mask.to(device=human_tokens.device, dtype=torch.bool)
        ego = ego_token + self.modality_embedding[0].to(ego_token)[None, None]
        env = env_token + self.modality_embedding[1].to(env_token)[None, None]
        humans = human_tokens + self.modality_embedding[2].to(human_tokens)[None, None]
        if human_position is not None:
            expected = (*human_tokens.shape[:2], int(self.config.human_position_dim))
            if tuple(human_position.shape) != expected:
                raise ValueError(f"human_position must be {expected}, got {tuple(human_position.shape)}")
            humans = humans + self.human_position(human_position.to(dtype=humans.dtype))
        humans = humans.masked_fill(~mask[..., None], 0.0)

        tokens = torch.cat((ego, env, humans), dim=1)
        key_padding_mask = torch.cat(
            (
                torch.zeros(mask.shape[0], 2, dtype=torch.bool, device=mask.device),
                ~mask,
            ),
            dim=1,
        )
        interacted = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        # Transformer masks keys/values, not query rows. Explicitly clear padded
        # human query outputs so they cannot leak into later transitions/losses.
        human_context = interacted[:, 2:].masked_fill(~mask[..., None], 0.0)
        return {
            "ego_context": interacted[:, :1],
            "env_context": interacted[:, 1:2],
            "human_context": human_context,
            "human_mask": mask,
        }
