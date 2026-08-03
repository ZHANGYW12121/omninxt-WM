"""Trainable causal multi-person skeleton encoder.

This module intentionally does **not** save features to disk.  It consumes
causal skeleton windows prepared by the dataset and produces pose tokens during
the world-model forward pass, so its parameters are trained jointly with the
world model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from modules.stgcn_lite import STGCNLiteEncoder


class CausalMultiPersonSTGCNEncoder(nn.Module):
    """Encode per-person causal skeleton windows into pose tokens.

    Expected input:

    ``pose_windows``: ``[B, T, M, W, V, C]``

    - ``B``: batch size
    - ``T``: sequence length
    - ``M``: max people per frame
    - ``W``: causal history window ending at current frame
    - ``V``: joints, COCO-17 by default
    - ``C``: ``x, y, confidence`` or ``x, y``

    The module flattens ``B*T*M`` person-windows, runs a trainable ST-GCN over
    the causal window, and keeps only the final-window output as the token for
    the current frame/person.  Because every window is constructed as
    ``[t-W+1, ..., t]``, no future frame can enter the token.
    """

    def __init__(
        self,
        *,
        num_joints: int = 17,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_dim: int = 256,
        image_size: tuple[int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_joints = int(num_joints)
        self.in_channels = int(in_channels)
        self.image_size = image_size
        self.encoder = STGCNLiteEncoder(
            num_joints=num_joints,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_dim=out_dim,
            image_size=image_size,
            dropout=dropout,
        )

    def forward(
        self,
        pose_windows: torch.Tensor,
        *,
        pose_window_mask: torch.Tensor | None = None,
        pose_token_mask: torch.Tensor | None = None,
        image_size_hw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return pose tokens and masks.

        Args:
            pose_windows: skeleton windows shaped ``[B,T,M,W,17,2/3]``.
            pose_window_mask: optional valid-frame mask ``[B,T,M,W]``.
            pose_token_mask: optional current-person mask ``[B,T,M]``.
            image_size_hw: optional per-frame image size ``[B,T,2]`` as
                ``height,width``.  Used only when ``self.image_size`` is None.

        Returns:
            A dict with:

            - ``pose_tokens``: ``[B,T,M,out_dim]``
            - ``pose_token_mask``: ``[B,T,M]``
        """

        if pose_windows.dim() != 6:
            raise ValueError(f"Expected pose_windows [B,T,M,W,V,C], got {tuple(pose_windows.shape)}")

        b, t, m, w, v, c = pose_windows.shape
        if v != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {v}")
        if c == 2:
            conf = torch.ones_like(pose_windows[..., :1])
            pose_windows = torch.cat([pose_windows, conf], dim=-1)
            c = 3
        if c != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {c}")

        x = pose_windows.float()
        if pose_window_mask is not None:
            if pose_window_mask.shape != (b, t, m, w):
                raise ValueError(
                    "pose_window_mask must have shape "
                    f"{(b, t, m, w)}, got {tuple(pose_window_mask.shape)}"
                )
            x = x * pose_window_mask.to(dtype=x.dtype, device=x.device)[..., None, None]

        # If the ST-GCN was not constructed with a fixed image size, allow the
        # dataset to provide per-frame image sizes.  We normalize before
        # flattening because image sizes are per current frame, not per flattened
        # person-window.
        if self.image_size is None and image_size_hw is not None:
            if image_size_hw.shape[:2] != (b, t) or image_size_hw.shape[-1] != 2:
                raise ValueError(
                    "image_size_hw must have shape [B,T,2], "
                    f"got {tuple(image_size_hw.shape)}"
                )
            size = image_size_hw.to(dtype=x.dtype, device=x.device).clamp_min(1.0)
            height = size[..., 0].view(b, t, 1, 1, 1)
            width = size[..., 1].view(b, t, 1, 1, 1)
            x = x.clone()
            x[..., 0] = x[..., 0] / width
            x[..., 1] = x[..., 1] / height

        flat = x.reshape(b * t * m, w, v, c)
        encoded = self.encoder(flat)
        # [B*T*M, W, D] -> keep only current timestep token.
        tokens = encoded[:, -1].reshape(b, t, m, self.out_dim)

        if pose_token_mask is None:
            if pose_window_mask is not None:
                pose_token_mask = pose_window_mask[..., -1]
            else:
                # Treat non-zero confidence in the current frame as a person.
                pose_token_mask = pose_windows[..., -1, :, 2].amax(dim=-1) > 0
        else:
            if pose_token_mask.shape != (b, t, m):
                raise ValueError(
                    "pose_token_mask must have shape "
                    f"{(b, t, m)}, got {tuple(pose_token_mask.shape)}"
                )

        mask = pose_token_mask.to(device=tokens.device, dtype=torch.bool)
        tokens = tokens * mask[..., None].to(tokens.dtype)
        return {
            "pose_tokens": tokens,
            "pose_token_mask": mask,
        }


def encode_pose_batch(
    encoder: CausalMultiPersonSTGCNEncoder,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Convenience wrapper for batches returned by ``crowd_collate``."""

    return encoder(
        batch["pose_windows"],
        pose_window_mask=batch.get("pose_window_mask"),
        pose_token_mask=batch.get("pose_token_mask"),
        image_size_hw=batch.get("pose_image_size_hw"),
    )
