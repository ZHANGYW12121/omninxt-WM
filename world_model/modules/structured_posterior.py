"""Structured posterior adapter for the shared RSSM.

The crowd world model keeps human, environment and ego observations as separate
streams.  A vanilla RSSM posterior usually concatenates ``[deter, embed]`` and
predicts the stochastic-state logits from that single vector.  This adapter
keeps the posterior input structured a little longer: each stream gets its own
projection and a deterministic-state-conditioned gate before all evidence is
mixed into one posterior context.

The RSSM itself remains shared.  There is still one deterministic state, one
stochastic state and one prior.  Only the observation-conditioned posterior head
is changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from tools import weight_init_


DEFAULT_STRUCTURED_EMBED_KEYS = ("human_embed", "env_embed", "ego_embed")


def structured_embed_size(embed_size: Mapping[str, int]) -> int:
    """Return the flattened size of structured RSSM evidence streams."""

    return int(sum(int(v) for v in embed_size.values()))


def flatten_structured_embed(
    embed: Mapping[str, torch.Tensor],
    keys: Sequence[str] = DEFAULT_STRUCTURED_EMBED_KEYS,
) -> torch.Tensor:
    """Flatten structured encoder output for auxiliary objectives.

    This is intentionally not used by the RSSM posterior.  It is only a
    compatibility helper for losses such as R2-Dreamer / InfoNCE that expect a
    single observation embedding tensor.
    """

    tensors = []
    missing = []
    for key in keys:
        value = embed.get(key)
        if value is None:
            missing.append(key)
        else:
            tensors.append(value)
    if missing:
        raise KeyError(f"Structured embed is missing required keys: {missing}")
    return torch.cat(tensors, dim=-1)


class ResidualBlock(nn.Module):
    """Small residual MLP block used inside the posterior adapter."""

    def __init__(self, dim: int, *, act: type[nn.Module], dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32),
            nn.Linear(dim, dim * 4, bias=True),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StructuredPosteriorAdapter(nn.Module):
    """Convert human/env/ego evidence into an RSSM posterior context.

    Args:
        deter_dim: deterministic RSSM state dimension.
        stream_dims: mapping from stream name to embedding dimension.  For the
            current crowd model this is usually
            ``{"human_embed": D, "env_embed": D, "ego_embed": D}``.
        hidden_dim: posterior context dimension returned by this module.
        layers: number of residual mixing blocks after gated stream fusion.
        act: torch activation class name.
        dropout: dropout inside residual mixing blocks.

    Shape:
        ``deter`` is ``[..., deter_dim]`` and every stream tensor is
        ``[..., stream_dim]`` with the same leading dimensions.  The return value
        is ``[..., hidden_dim]``.
    """

    def __init__(
        self,
        *,
        deter_dim: int,
        stream_dims: Mapping[str, int],
        hidden_dim: int,
        layers: int = 1,
        act: str = "SiLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.deter_dim = int(deter_dim)
        self.hidden_dim = int(hidden_dim)
        self.stream_dims = {str(k): int(v) for k, v in stream_dims.items()}
        self.stream_keys = tuple(self.stream_dims.keys())
        if not self.stream_keys:
            raise ValueError("StructuredPosteriorAdapter requires at least one stream.")
        if any(dim <= 0 for dim in self.stream_dims.values()):
            raise ValueError(f"Invalid stream dimensions: {self.stream_dims}")

        act_cls = getattr(nn, act)
        self.deter_proj = nn.Sequential(
            nn.RMSNorm(self.deter_dim, eps=1e-04, dtype=torch.float32),
            nn.Linear(self.deter_dim, self.hidden_dim, bias=True),
            act_cls(),
        )
        self.stream_proj = nn.ModuleDict()
        self.stream_gate = nn.ModuleDict()
        for key, dim in self.stream_dims.items():
            self.stream_proj[key] = nn.Sequential(
                nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32),
                nn.Linear(dim, self.hidden_dim, bias=True),
                act_cls(),
            )
            self.stream_gate[key] = nn.Sequential(
                nn.RMSNorm(self.deter_dim + dim, eps=1e-04, dtype=torch.float32),
                nn.Linear(self.deter_dim + dim, self.hidden_dim, bias=True),
                nn.Sigmoid(),
            )

        blocks = []
        for _ in range(int(layers)):
            blocks.append(ResidualBlock(self.hidden_dim, act=act_cls, dropout=dropout))
        blocks.append(nn.RMSNorm(self.hidden_dim, eps=1e-04, dtype=torch.float32))
        self.mixer = nn.Sequential(*blocks)
        self.apply(weight_init_)

    @property
    def out_dim(self) -> int:
        return self.hidden_dim

    def forward(self, deter: torch.Tensor, embed: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if deter.shape[-1] != self.deter_dim:
            raise ValueError(f"Expected deter last dim {self.deter_dim}, got {tuple(deter.shape)}")

        context = self.deter_proj(deter)
        missing = []
        for key in self.stream_keys:
            value = embed.get(key)
            if value is None:
                missing.append(key)
                continue
            if value.shape[:-1] != deter.shape[:-1]:
                raise ValueError(
                    f"Structured stream {key!r} leading shape {tuple(value.shape[:-1])} "
                    f"does not match deter {tuple(deter.shape[:-1])}."
                )
            expected = self.stream_dims[key]
            if value.shape[-1] != expected:
                raise ValueError(f"Structured stream {key!r} expected dim {expected}, got {tuple(value.shape)}")

            evidence = self.stream_proj[key](value)
            gate = self.stream_gate[key](torch.cat([deter, value], dim=-1))
            context = context + gate * evidence

        if missing:
            raise KeyError(f"Structured RSSM posterior input is missing streams: {missing}")
        return self.mixer(context)
