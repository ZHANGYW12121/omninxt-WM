"""Goal-conditioned Action-token readout over Ego/Human RSSM latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentPolicyAttentionConfig:
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    ff_mult: int = 4
    dropout: float = 0.0
    # Goal8 contains metric deltas/distance together with unit directions.
    # A samplewise LayerNorm over those heterogeneous fields can make two
    # different remaining distances unnecessarily hard to distinguish.
    goal_metric_scaling: bool = False


class _LatentAttentionBlock(nn.Module):
    def __init__(self, config: LatentPolicyAttentionConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * config.ff_mult), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(d * config.ff_mult, d), nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        normed = self.norm1(tokens)
        message, weights = self.attn(
            normed, normed, normed, key_padding_mask=~mask,
            need_weights=True, average_attn_weights=False,
        )
        tokens = tokens + message
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.masked_fill(~mask[..., None], 0.0), weights


class ActionTokenLatentAttention(nn.Module):
    """Aggregate ``[Action, Goal, Ego, Humans...]`` and return Action only.

    This is the second attention layer.  Unlike observation attention it is
    intentionally dense: the Action query needs the target, vehicle state and
    every valid person when choosing a control action.
    """

    def __init__(self, ego_dim: int, human_dim: int, goal_dim: int,
                 config: LatentPolicyAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentPolicyAttentionConfig()
        d = self.config.model_dim
        self.goal_dim = int(goal_dim)
        self.goal_metric_scaling = bool(self.config.goal_metric_scaling)
        if self.goal_metric_scaling:
            if self.goal_dim != 8:
                raise ValueError("fixed metric Goal scaling requires Goal8")
            if d < self.goal_dim:
                raise ValueError(
                    "metric Goal token must fit the eight physical fields")
            # [dx_body,dy_body,dz,distance,unit_xyz,heading/pi].  These scales
            # cover the current warehouse routes without mixing physical
            # magnitudes with dimensionless direction fields.
            self.register_buffer("goal_metric_scale", torch.tensor((
                40.0, 40.0, 3.0, 50.0, 1.0, 1.0, 1.0, 1.0,
            )))
            self.goal_projector = nn.Linear(self.goal_dim, d)
        else:
            self.goal_projector = nn.Sequential(
                nn.LayerNorm(self.goal_dim), nn.Linear(self.goal_dim, d))
        self.ego_projector = nn.Sequential(nn.LayerNorm(ego_dim), nn.Linear(ego_dim, d))
        self.human_projector = nn.Sequential(nn.LayerNorm(human_dim), nn.Linear(human_dim, d))
        self.action_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.action_token, std=0.02)
        self.modality_embedding = nn.Parameter(torch.empty(4, d))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.layers = nn.ModuleList(
            [_LatentAttentionBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(d)

    def forward(self, goal: torch.Tensor, ego_feat: torch.Tensor,
                human_feat: torch.Tensor, human_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if goal.ndim == 2:
            goal = goal[:, None]
        if ego_feat.ndim == 2:
            ego_feat = ego_feat[:, None]
        if goal.ndim != 3 or goal.shape[1] != 1 or goal.shape[-1] != self.goal_dim:
            raise ValueError(f"goal must be [B,{self.goal_dim}] or [B,1,{self.goal_dim}]")
        if ego_feat.ndim != 3 or ego_feat.shape[1] != 1:
            raise ValueError("ego_feat must be [B,D] or [B,1,D]")
        if human_feat.ndim != 3 or human_mask.shape != human_feat.shape[:2]:
            raise ValueError("human_feat/mask must be [B,N,D]/[B,N]")
        b = ego_feat.shape[0]
        action = self.action_token.to(ego_feat).expand(b, -1, -1)
        goal_input = goal.float()
        if self.goal_metric_scaling:
            goal_input = (
                goal_input / self.goal_metric_scale.to(goal_input)
            ).clamp(-5.0, 5.0)
        goal_token = self.goal_projector(goal_input)
        ego_token = self.ego_projector(ego_feat)
        human_tokens = self.human_projector(human_feat)
        action = action + self.modality_embedding[0].to(action)
        goal_token = goal_token + self.modality_embedding[1].to(goal_token)
        ego_token = ego_token + self.modality_embedding[2].to(ego_token)
        human_tokens = human_tokens + self.modality_embedding[3].to(human_tokens)
        # Preserve private Goal/Ego readouts before dense Human attention.  The
        # goal branch of the factorized Actor consumes these two tensors, while
        # interaction-aware tokens remain available to avoidance and all world
        # heads.  Cloning is unnecessary: the attention blocks are functional
        # and do not mutate their inputs in-place.
        private_goal_token = self.final_norm(goal_token)
        if self.goal_metric_scaling:
            # The Actor/Critic private Goal token must retain exact remaining
            # distance and signed body-frame deltas. A learned projection
            # followed by LayerNorm alone can still attenuate those magnitudes.
            # Reserve the prefix exactly, mirroring the explicit Human/task
            # geometry contract; the remaining dimensions stay learned.
            private_goal_token = torch.cat((
                goal_input,
                private_goal_token[..., self.goal_dim:],
            ), -1)
        private_ego_token = self.final_norm(ego_token)
        tokens = torch.cat((action, goal_token, ego_token, human_tokens), dim=1)
        mask = torch.cat((
            torch.ones(b, 3, dtype=torch.bool, device=tokens.device), human_mask.bool(),
        ), dim=1)
        attention_weights = []
        output = tokens
        for layer in self.layers:
            output, weights = layer(output, mask)
            attention_weights.append(weights)
        output = self.final_norm(output).masked_fill(~mask[..., None], 0.0)
        return {
            "joint_feat": output[:, 0],
            "goal_token": output[:, 1],
            "ego_token": output[:, 2],
            "private_goal_token": private_goal_token[:, 0],
            "private_ego_token": private_ego_token[:, 0],
            "human_tokens": output[:, 3:],
            "latent_tokens": output,
            "latent_mask": mask,
            "attention_weights": attention_weights[-1],
            "all_attention_weights": attention_weights,
        }
