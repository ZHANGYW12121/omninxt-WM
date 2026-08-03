"""Sparse asymmetric attention for factorized Ego--Human tokens.

The visibility pattern is deliberately directional:

* the Ego query can read Ego and every valid Human slot;
* Human ``n`` can read only Ego and Human ``n``;
* Human slots never read one another.

The module is used with independent parameters for observation fusion and for
causal latent coupling.  It contains no explicit/learned relation branch: Q,
K, and V all come from the normally encoded token content.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SparseEgoHumanAttentionConfig:
    model_dim: int = 128
    num_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    modality_embeddings: bool = True


class SparseEgoHumanAttention(nn.Module):
    """Content-only self-attention with an Ego--Human structural mask."""

    def __init__(self, config: SparseEgoHumanAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SparseEgoHumanAttentionConfig()
        d = int(self.config.model_dim)
        if d % int(self.config.num_heads):
            raise ValueError("model_dim must be divisible by num_heads")
        self.modality_embedding = (
            nn.Parameter(torch.empty(2, d)) if self.config.modality_embeddings else None
        )
        if self.modality_embedding is not None:
            nn.init.normal_(self.modality_embedding, std=0.02)
        self.norm1 = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(
            d, int(self.config.num_heads), dropout=float(self.config.dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * int(self.config.ff_mult)), nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(d * int(self.config.ff_mult), d),
            nn.Dropout(float(self.config.dropout)),
        )

    @staticmethod
    def structural_mask(num_people: int, device: torch.device) -> torch.Tensor:
        """Return boolean ``[1+N,1+N]`` mask where True means blocked."""

        slots = 1 + int(num_people)
        blocked = torch.ones((slots, slots), dtype=torch.bool, device=device)
        blocked[0] = False  # Ego query reads all valid slots.
        index = torch.arange(1, slots, device=device)
        blocked[index, 0] = False  # Every Human query reads Ego.
        blocked[index, index] = False  # Every Human query reads itself.
        return blocked

    def forward(
        self,
        ego_token: torch.Tensor,
        human_tokens: torch.Tensor,
        human_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if ego_token.ndim == 2:
            ego_token = ego_token[:, None]
        if ego_token.ndim != 3 or ego_token.shape[1] != 1:
            raise ValueError(f"ego_token must be [B,1,D], got {tuple(ego_token.shape)}")
        if human_tokens.ndim != 3:
            raise ValueError("human_tokens must be [B,N,D]")
        if human_mask.shape != human_tokens.shape[:2]:
            raise ValueError("human_mask must match human_tokens [B,N]")
        if ego_token.shape[0] != human_tokens.shape[0] or ego_token.shape[-1] != human_tokens.shape[-1]:
            raise ValueError("Ego/Human token batch and feature dimensions must match")

        batch, people = human_mask.shape
        tokens = torch.cat((ego_token, human_tokens), dim=1)
        if self.modality_embedding is not None:
            tokens = tokens.clone()
            tokens[:, :1] = tokens[:, :1] + self.modality_embedding[0].to(tokens)
            tokens[:, 1:] = tokens[:, 1:] + self.modality_embedding[1].to(tokens)
        valid = torch.cat((
            torch.ones((batch, 1), dtype=torch.bool, device=tokens.device),
            human_mask.bool(),
        ), dim=1)
        normed = self.norm1(tokens)
        message, weights = self.attention(
            normed, normed, normed,
            attn_mask=self.structural_mask(people, tokens.device),
            key_padding_mask=~valid,
            need_weights=True,
            average_attn_weights=False,
        )
        output = tokens + message
        output = output + self.ffn(self.norm2(output))
        output = output.masked_fill(~valid[..., None], 0.0)
        weights = weights.masked_fill(~valid[:, None, :, None], 0.0)
        weights = weights.masked_fill(~valid[:, None, None, :], 0.0)
        return {
            "ego": output[:, :1],
            "human": output[:, 1:],
            "tokens": output,
            "token_mask": valid,
            "attention_weights": weights,
        }
