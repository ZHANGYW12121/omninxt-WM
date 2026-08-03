"""Posterior-only relation-guided observation attention.

Relations decide *where* to attend (Q/K), while modality-private content and
relations together decide *what* is transmitted (V).  Slot order is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RelationAttentionConfig:
    private_dim: int = 128
    explicit_dim: int = 10
    learned_relation_dim: int = 64
    relation_dim: int = 128
    model_dim: int = 128
    num_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0


class _RelationAdapter(nn.Module):
    def __init__(self, config: RelationAttentionConfig) -> None:
        super().__init__()
        self.explicit_norm = nn.LayerNorm(config.explicit_dim)
        self.learned = nn.Sequential(
            nn.LayerNorm(config.private_dim),
            nn.Linear(config.private_dim, config.learned_relation_dim),
            nn.SiLU(),
        )
        self.relation = nn.Sequential(
            nn.Linear(config.explicit_dim + config.learned_relation_dim, config.relation_dim),
            nn.LayerNorm(config.relation_dim),
        )
        self.content = nn.Sequential(
            nn.Linear(config.private_dim + config.relation_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )

    def forward(self, private: torch.Tensor, explicit: torch.Tensor):
        learned = self.learned(private)
        relation = self.relation(torch.cat((self.explicit_norm(explicit), learned), dim=-1))
        content = self.content(torch.cat((private, relation), dim=-1))
        return relation, content, learned


class RelationGuidedObservationAttention(nn.Module):
    """Relation-QK/content-V attention with attention and FFN residuals."""

    def __init__(self, config: RelationAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or RelationAttentionConfig()
        c = self.config
        if c.model_dim % c.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.ego_adapter = _RelationAdapter(c)
        self.env_adapter = _RelationAdapter(c)
        self.human_adapter = _RelationAdapter(c)
        self.q_proj = nn.Linear(c.relation_dim, c.model_dim)
        self.k_proj = nn.Linear(c.relation_dim, c.model_dim)
        self.v_proj = nn.Linear(c.model_dim, c.model_dim)
        self.out_proj = nn.Linear(c.model_dim, c.model_dim)
        self.attn_norm = nn.LayerNorm(c.model_dim)
        self.ffn_norm = nn.LayerNorm(c.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(c.model_dim, c.model_dim * c.ff_mult), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.model_dim * c.ff_mult, c.model_dim),
        )
        self.dropout = nn.Dropout(c.dropout)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        b, m, _ = value.shape
        return value.reshape(b, m, self.config.num_heads, -1).transpose(1, 2)

    def forward(
        self,
        ego_private: torch.Tensor,
        ego_explicit: torch.Tensor,
        env_private: torch.Tensor,
        env_explicit: torch.Tensor,
        human_private: torch.Tensor,
        human_explicit: torch.Tensor,
        env_mask: torch.Tensor,
        human_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if ego_private.ndim != 3 or ego_private.shape[1] != 1:
            raise ValueError("ego_private must be [B,1,D]")
        if env_mask.shape != env_private.shape[:2] or human_mask.shape != human_private.shape[:2]:
            raise ValueError("slot masks must match private token shapes")
        er, ec, el = self.ego_adapter(ego_private, ego_explicit)
        vr, vc, vl = self.env_adapter(env_private, env_explicit)
        hr, hc, hl = self.human_adapter(human_private, human_explicit)
        relation = torch.cat((er, vr, hr), dim=1)
        content = torch.cat((ec, vc, hc), dim=1)
        mask = torch.cat((
            torch.ones(ego_private.shape[0], 1, dtype=torch.bool, device=ego_private.device),
            env_mask.bool(), human_mask.bool(),
        ), dim=1)

        q = self._heads(self.q_proj(relation))
        k = self._heads(self.k_proj(relation))
        v = self._heads(self.v_proj(self.attn_norm(content)))
        scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        message = torch.matmul(weights, v).transpose(1, 2).reshape_as(content)
        # Content is the residual anchor. Relations route messages but never
        # replace a slot's own observation content.
        output = content + self.dropout(self.out_proj(message))
        output = output + self.dropout(self.ffn(self.ffn_norm(output)))
        output = output.masked_fill(~mask[..., None], 0.0)
        weights = weights.masked_fill(~mask[:, None, :, None], 0.0)
        n_env = env_private.shape[1]
        split = 1 + n_env
        return {
            "ego_obs_embed": output[:, :1],
            "env_obs_embed": output[:, 1:split],
            "human_obs_embed": output[:, split:],
            "attention_weights": weights,
            "relation_tokens": relation.masked_fill(~mask[..., None], 0.0),
            "content_tokens": content.masked_fill(~mask[..., None], 0.0),
            "query_tokens": q,
            "key_tokens": k,
            "value_tokens": v,
            "learned_relations": {"ego": el, "env": vl, "human": hl},
            "token_mask": mask,
        }
