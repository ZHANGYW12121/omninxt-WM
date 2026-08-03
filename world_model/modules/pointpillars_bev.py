"""Trainable ego-centric PointPillars BEV encoder.

The old ``simple_pointpillars.py`` file is a useful standalone demo, but it can
infer a different point-cloud range for every frame.  That is not appropriate
for a world model because the same BEV cell must mean the same ego-centric
location at every timestep.

This module uses a fixed range in the UAV/LiDAR frame and produces trainable BEV
feature maps/tokens during the world-model forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


# Ego/LiDAR-frame default chosen for the current CityTower + Warehouse task.
#
# Coordinate convention follows the recorded .npy point clouds:
#   x: lateral left/right around the UAV
#   y: forward/backward around the UAV
#   z: vertical around the UAV
#
# CityTower people are roughly within a narrower lateral corridor, Warehouse
# people can spread to about +/-7.5m.  +/-16m lateral and 40m forward give both
# scenes enough margin while keeping the BEV grid small enough for training.
DEFAULT_EGO_BEV_RANGE = (-16.0, -8.0, -2.0, 16.0, 40.0, 8.0)


@dataclass(frozen=True)
class EgoPointPillarsConfig:
    """Configuration for fixed-range ego-centric PointPillars."""

    voxel_size: tuple[float, float] = (0.5, 0.5)
    point_cloud_range: tuple[float, float, float, float, float, float] = DEFAULT_EGO_BEV_RANGE
    max_points_per_pillar: int = 32
    # Default grid is 96x64 = 6144 cells, so 8000 avoids truncation for this range.
    max_pillars: int = 8000
    point_feature_dim: int = 3
    pfn_channels: int = 64
    backbone_channels: tuple[int, ...] = (64, 128)
    backbone_strides: tuple[int, ...] = (2, 2)

    @property
    def grid_size(self) -> tuple[int, int]:
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        voxel_x, voxel_y = self.voxel_size
        width = int(round((x_max - x_min) / voxel_x))
        height = int(round((y_max - y_min) / voxel_y))
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid BEV grid size from range={self.point_cloud_range}, voxel={self.voxel_size}")
        return height, width

    @property
    def feature_grid_size(self) -> tuple[int, int]:
        height, width = self.grid_size
        for stride in self.backbone_strides:
            height = math.ceil(height / int(stride))
            width = math.ceil(width / int(stride))
        return height, width

    @property
    def out_channels(self) -> int:
        return int(self.backbone_channels[-1]) if self.backbone_channels else int(self.pfn_channels)


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(int(max_groups), int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


class PillarFeatureNet(nn.Module):
    """Simplified trainable PFN layer for per-pillar point features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        # LayerNorm is stable for variable pillar/point counts and tiny batches.
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode pillars.

        Args:
            features: ``[num_pillars, max_points, in_channels]``.
            mask: ``[num_pillars, max_points]`` where true marks real points.

        Returns:
            ``[num_pillars, out_channels]``.
        """

        if features.numel() == 0:
            return features.new_zeros((0, self.linear.out_features))
        x = self.act(self.norm(self.linear(features)))
        x = x.masked_fill(~mask[..., None], -1.0e9)
        x = x.max(dim=1).values
        # If a malformed empty pillar sneaks in, keep it finite.
        has_point = mask.any(dim=1)
        x = torch.where(has_point[:, None], x, torch.zeros_like(x))
        return x


class BEVBackbone(nn.Module):
    """Small trainable CNN over the pseudo-image BEV map."""

    def __init__(self, in_channels: int, channels: Sequence[int], strides: Sequence[int]) -> None:
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("backbone_channels and backbone_strides must have the same length.")
        layers: list[nn.Module] = []
        cur = int(in_channels)
        for out_ch, stride in zip(channels, strides):
            out_ch = int(out_ch)
            layers.append(nn.Conv2d(cur, out_ch, kernel_size=3, stride=int(stride), padding=1, bias=False))
            layers.append(nn.GroupNorm(_group_count(out_ch), out_ch))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.GroupNorm(_group_count(out_ch), out_ch))
            layers.append(nn.SiLU(inplace=True))
            cur = out_ch
        self.net = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pillarize_ego_points(
    points: torch.Tensor,
    config: EgoPointPillarsConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one ego-frame point cloud into dense per-pillar tensors."""

    if points.ndim != 2 or points.shape[-1] < 3:
        raise ValueError(f"Expected points [N, >=3], got {tuple(points.shape)}")

    device = points.device
    dtype = points.dtype
    x_min, y_min, z_min, x_max, y_max, z_max = config.point_cloud_range
    voxel_x, voxel_y = config.voxel_size
    height, width = config.grid_size

    point_dim = int(config.point_feature_dim)
    if points.shape[-1] < point_dim:
        raise ValueError(f"Input point_dim={points.shape[-1]} is smaller than config.point_feature_dim={point_dim}.")
    points = points[:, :point_dim]
    finite = torch.isfinite(points[:, :3]).all(dim=-1)
    points = points[finite]
    if points.numel() == 0:
        feature_dim = point_dim + 5
        return (
            points.new_zeros((0, config.max_points_per_pillar, feature_dim)),
            torch.zeros((0, config.max_points_per_pillar), dtype=torch.bool, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )

    xyz = points[:, :3]
    in_range = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] < x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] < y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] < z_max)
    )
    points = points[in_range]
    if points.numel() == 0:
        feature_dim = point_dim + 5
        return (
            points.new_zeros((0, config.max_points_per_pillar, feature_dim)),
            torch.zeros((0, config.max_points_per_pillar), dtype=torch.bool, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )

    x_idx = torch.floor((points[:, 0] - x_min) / voxel_x).long().clamp(0, width - 1)
    y_idx = torch.floor((points[:, 1] - y_min) / voxel_y).long().clamp(0, height - 1)
    point_coords = torch.stack((y_idx, x_idx), dim=1)
    unique_coords, inverse = torch.unique(point_coords, dim=0, return_inverse=True)

    num_pillars = min(int(unique_coords.shape[0]), int(config.max_pillars))
    keep_pillar = inverse < num_pillars
    points = points[keep_pillar]
    inverse = inverse[keep_pillar]
    coords = unique_coords[:num_pillars]
    if num_pillars == 0 or points.numel() == 0:
        feature_dim = point_dim + 5
        return (
            points.new_zeros((0, config.max_points_per_pillar, feature_dim)),
            torch.zeros((0, config.max_points_per_pillar), dtype=torch.bool, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )

    counts = torch.bincount(inverse, minlength=num_pillars)
    order = torch.argsort(inverse, stable=True)
    sorted_inverse = inverse[order]
    sorted_points = points[order]
    start_offsets = torch.cumsum(counts, dim=0) - counts
    point_offsets = torch.arange(sorted_points.shape[0], device=device) - start_offsets[sorted_inverse]
    keep_point = point_offsets < int(config.max_points_per_pillar)
    sorted_points = sorted_points[keep_point]
    sorted_inverse = sorted_inverse[keep_point]
    point_offsets = point_offsets[keep_point]

    max_points = int(config.max_points_per_pillar)
    pillars = points.new_zeros((num_pillars, max_points, point_dim))
    mask = torch.zeros((num_pillars, max_points), dtype=torch.bool, device=device)
    pillars[sorted_inverse, point_offsets] = sorted_points
    mask[sorted_inverse, point_offsets] = True

    real_counts = mask.sum(dim=1).clamp(min=1).to(dtype)
    xyz_sum = (pillars[:, :, :3] * mask[..., None]).sum(dim=1)
    xyz_mean = xyz_sum / real_counts[:, None]
    cluster_offset = pillars[:, :, :3] - xyz_mean[:, None, :]

    x_center = x_min + (coords[:, 1].to(dtype) + 0.5) * voxel_x
    y_center = y_min + (coords[:, 0].to(dtype) + 0.5) * voxel_y
    center_offset = torch.stack(
        (
            pillars[:, :, 0] - x_center[:, None],
            pillars[:, :, 1] - y_center[:, None],
        ),
        dim=-1,
    )

    features = torch.cat((pillars, cluster_offset, center_offset), dim=-1)
    features = features * mask[..., None]
    return features, mask, coords


def scatter_pillars_to_bev(
    pillar_features: torch.Tensor,
    coords: torch.Tensor,
    config: EgoPointPillarsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter pillar features into a BEV pseudo-image and occupancy map."""

    height, width = config.grid_size
    channels = int(config.pfn_channels)
    bev = pillar_features.new_zeros((channels, height, width))
    occupancy = torch.zeros((1, height, width), dtype=torch.bool, device=pillar_features.device)
    if pillar_features.numel() > 0:
        bev[:, coords[:, 0], coords[:, 1]] = pillar_features.t()
        occupancy[:, coords[:, 0], coords[:, 1]] = True
    return bev, occupancy


class EgoPointPillarsBEVEncoder(nn.Module):
    """Fixed-range trainable PointPillars encoder for world-model observations."""

    def __init__(self, config: EgoPointPillarsConfig | None = None) -> None:
        super().__init__()
        self.config = config or EgoPointPillarsConfig()
        if self.config.point_feature_dim < 3:
            raise ValueError("point_feature_dim must be at least 3 for xyz.")
        pfn_in_channels = int(self.config.point_feature_dim) + 3 + 2
        self.pfn = PillarFeatureNet(pfn_in_channels, int(self.config.pfn_channels))
        self.backbone = BEVBackbone(
            int(self.config.pfn_channels),
            self.config.backbone_channels,
            self.config.backbone_strides,
        )

    @property
    def out_dim(self) -> int:
        return self.config.out_channels

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.config.grid_size

    @property
    def feature_grid_size(self) -> tuple[int, int]:
        return self.config.feature_grid_size

    def _encode_single(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask, coords = pillarize_ego_points(points, self.config)
        if features.shape[0] == 0:
            pillar_features = features.new_zeros((0, int(self.config.pfn_channels)))
        else:
            pillar_features = self.pfn(features, mask)
        return scatter_pillars_to_bev(pillar_features, coords, self.config)

    def forward(
        self,
        points: torch.Tensor,
        *,
        point_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode padded point clouds.

        Args:
            points: ``[B,T,N,C]`` or ``[B,N,C]`` ego-frame point clouds.
            point_mask: optional valid-point mask ``[B,T,N]`` or ``[B,N]``.

        Returns:
            Dict containing:

            - ``pillar_bev``: raw PFN BEV map, ``[B,T,C0,H,W]`` or ``[B,C0,H,W]``
            - ``pillar_occupancy``: raw occupancy, ``[B,T,1,H,W]`` or ``[B,1,H,W]``
            - ``bev_feature``: CNN BEV map, ``[B,T,C,Hf,Wf]`` or ``[B,C,Hf,Wf]``
            - ``bev_tokens``: flattened BEV tokens, ``[B,T,Hf*Wf,C]`` or ``[B,Hf*Wf,C]``
            - ``bev_token_mask``: pooled occupancy mask, ``[B,T,Hf*Wf]`` or ``[B,Hf*Wf]``
            - ``bev_hw``: tensor ``[Hf, Wf]``
            - ``bev_range``: tensor ``[x_min, y_min, z_min, x_max, y_max, z_max]``
        """

        if points.dim() == 3:
            has_time = False
            b, n, c = points.shape
            flat_points = points
            flat_mask = point_mask
        elif points.dim() == 4:
            has_time = True
            b, t, n, c = points.shape
            flat_points = points.reshape(b * t, n, c)
            flat_mask = None if point_mask is None else point_mask.reshape(b * t, n)
        else:
            raise ValueError(f"Expected points [B,N,C] or [B,T,N,C], got {tuple(points.shape)}")

        if flat_mask is not None and flat_mask.shape[:2] != flat_points.shape[:2]:
            raise ValueError(
                f"point_mask shape {tuple(flat_mask.shape)} does not match points {tuple(flat_points.shape)}"
            )

        bev_maps: list[torch.Tensor] = []
        occupancy_maps: list[torch.Tensor] = []
        for i in range(flat_points.shape[0]):
            sample = flat_points[i]
            if flat_mask is not None:
                sample = sample[flat_mask[i].to(dtype=torch.bool, device=sample.device)]
            bev, occupancy = self._encode_single(sample)
            bev_maps.append(bev)
            occupancy_maps.append(occupancy)

        pillar_bev = torch.stack(bev_maps, dim=0)
        pillar_occupancy = torch.stack(occupancy_maps, dim=0)
        bev_feature = self.backbone(pillar_bev)

        token_mask_2d = pillar_occupancy.float()
        for stride in self.config.backbone_strides:
            token_mask_2d = F.max_pool2d(token_mask_2d, kernel_size=int(stride), stride=int(stride), ceil_mode=True)
        token_mask_2d = token_mask_2d > 0

        tokens = bev_feature.flatten(2).transpose(1, 2).contiguous()
        token_mask = token_mask_2d.flatten(2).squeeze(1).contiguous()

        if has_time:
            out_b = b
            out_t = t
            c0 = int(self.config.pfn_channels)
            hf, wf = bev_feature.shape[-2:]
            pillar_bev = pillar_bev.reshape(out_b, out_t, c0, *self.config.grid_size)
            pillar_occupancy = pillar_occupancy.reshape(out_b, out_t, 1, *self.config.grid_size)
            bev_feature = bev_feature.reshape(out_b, out_t, self.out_dim, hf, wf)
            tokens = tokens.reshape(out_b, out_t, hf * wf, self.out_dim)
            token_mask = token_mask.reshape(out_b, out_t, hf * wf)

        device = points.device
        return {
            "pillar_bev": pillar_bev,
            "pillar_occupancy": pillar_occupancy,
            "bev_feature": bev_feature,
            "bev_tokens": tokens,
            "bev_token_mask": token_mask,
            "bev_hw": torch.tensor(self.feature_grid_size, dtype=torch.long, device=device),
            "bev_range": torch.tensor(self.config.point_cloud_range, dtype=points.dtype, device=device),
        }


def encode_lidar_batch(
    encoder: EgoPointPillarsBEVEncoder,
    batch: dict[str, torch.Tensor],
    *,
    lidar_key: str = "lidar",
    mask_key: str = "lidar_mask",
) -> dict[str, torch.Tensor]:
    """Convenience wrapper for batches returned by ``crowd_collate``."""

    return encoder(batch[lidar_key], point_mask=batch.get(mask_key))
