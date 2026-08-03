"""Stable row-major spatial region slots for ego-centric BEV maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FixedSpatialRegionPooling(nn.Module):
    def __init__(self, grid: tuple[int, int] = (4, 4)) -> None:
        super().__init__()
        self.grid = tuple(int(x) for x in grid)
        if min(self.grid) <= 0:
            raise ValueError("region grid dimensions must be positive")

    @property
    def num_regions(self) -> int:
        return self.grid[0] * self.grid[1]

    def forward(self, feature_map: torch.Tensor, occupancy: torch.Tensor | None = None):
        has_time = feature_map.ndim == 5
        if has_time:
            b, t, c, h, w = feature_map.shape
            flat = feature_map.reshape(b * t, c, h, w)
        elif feature_map.ndim == 4:
            b, c, h, w = feature_map.shape
            t = None
            flat = feature_map
        else:
            raise ValueError("feature_map must be [B,C,H,W] or [B,T,C,H,W]")
        pooled = F.adaptive_avg_pool2d(flat, self.grid).flatten(2).transpose(1, 2)
        if occupancy is None:
            mask = torch.ones(pooled.shape[:2], dtype=torch.bool, device=pooled.device)
            probability = torch.ones_like(mask, dtype=pooled.dtype)
        else:
            occ = occupancy.reshape(flat.shape[0], occupancy.shape[-3], *occupancy.shape[-2:]).float()
            probability = F.adaptive_avg_pool2d(occ, self.grid).mean(1).flatten(1)
            mask = F.adaptive_max_pool2d(occ, self.grid).amax(1).flatten(1) > 0
        if has_time:
            pooled = pooled.reshape(b, t, self.num_regions, c)
            mask = mask.reshape(b, t, self.num_regions)
            probability = probability.reshape(b, t, self.num_regions)
        return {"tokens": pooled, "mask": mask, "occupancy": probability}

    def geometry(self, bev_range: torch.Tensor, *, batch_shape: tuple[int, ...],
                 dtype: torch.dtype, device: torch.device):
        values = bev_range.to(device=device, dtype=dtype).reshape(-1)
        x0, y0, z0, x1, y1, z1 = values[:6]
        rows, cols = self.grid
        xs = x0 + (torch.arange(cols, device=device, dtype=dtype) + 0.5) * ((x1 - x0) / cols)
        ys = y0 + (torch.arange(rows, device=device, dtype=dtype) + 0.5) * ((y1 - y0) / rows)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(self.num_regions, 3)
        extent = torch.stack(((x1 - x0) / cols, (y1 - y0) / rows, z1 - z0)).expand(
            self.num_regions, 3
        )
        view = (1,) * len(batch_shape) + centers.shape
        return centers.reshape(view).expand(*batch_shape, *centers.shape), \
            extent.reshape(view).expand(*batch_shape, *extent.shape)
