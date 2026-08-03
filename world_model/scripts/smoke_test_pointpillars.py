#!/usr/bin/env python3
"""Smoke-test the trainable ego-centric PointPillars BEV encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from datasets.isaac_crowd import IsaacCrowdSequenceDataset, crowd_collate
from modules import EgoPointPillarsBEVEncoder, EgoPointPillarsConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--index_path", type=Path, default=None)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_lidar_points", type=int, default=8192)
    parser.add_argument(
        "--range",
        type=float,
        nargs=6,
        default=(-16.0, -8.0, -2.0, 16.0, 40.0, 8.0),
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--voxel_size", type=float, nargs=2, default=(0.5, 0.5), metavar=("VX", "VY"))
    parser.add_argument("--max_points_per_pillar", type=int, default=32)
    parser.add_argument("--max_pillars", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = IsaacCrowdSequenceDataset(
        data_root=args.data_root if args.index_path is None else None,
        index_path=args.index_path,
        sequence_length=args.sequence_length,
        stride=args.stride,
        load_lidar=True,
        max_lidar_points=args.max_lidar_points,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=crowd_collate, num_workers=0)
    batch = next(iter(loader))

    config = EgoPointPillarsConfig(
        voxel_size=tuple(args.voxel_size),
        point_cloud_range=tuple(args.range),
        max_points_per_pillar=args.max_points_per_pillar,
        max_pillars=args.max_pillars,
    )
    encoder = EgoPointPillarsBEVEncoder(config)
    out = encoder(batch["lidar"], point_mask=batch["lidar_mask"])

    print("Input:")
    print(f"  lidar:      {tuple(batch['lidar'].shape)} {batch['lidar'].dtype}")
    print(f"  lidar_mask: {tuple(batch['lidar_mask'].shape)} {batch['lidar_mask'].dtype}")
    print("Config:")
    print(f"  range:      {config.point_cloud_range}")
    print(f"  voxel_size: {config.voxel_size}")
    print(f"  grid:       {config.grid_size}")
    print(f"  feat_grid:  {config.feature_grid_size}")
    print("Output:")
    for key, value in out.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
