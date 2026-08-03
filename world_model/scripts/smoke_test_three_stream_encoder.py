#!/usr/bin/env python3
"""Smoke-test the structured human/environment/ego world-model encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from datasets.isaac_crowd import IsaacCrowdSequenceDataset, crowd_collate
from modules import EgoPointPillarsConfig, ThreeStreamEncoderConfig, ThreeStreamWorldModelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--index_path", type=Path, default=None)
    parser.add_argument("--pose_cache_root", type=Path, required=True)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_lidar_points", type=int, default=4096)
    parser.add_argument("--pose_window_size", type=int, default=4)
    parser.add_argument("--max_pose_people", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument(
        "--bev_range",
        type=float,
        nargs=6,
        default=(-16.0, -8.0, -2.0, 16.0, 40.0, 8.0),
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--voxel_size", type=float, nargs=2, default=(0.5, 0.5), metavar=("VX", "VY"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = IsaacCrowdSequenceDataset(
        data_root=args.data_root if args.index_path is None else None,
        index_path=args.index_path,
        pose_cache_root=args.pose_cache_root,
        sequence_length=args.sequence_length,
        stride=args.stride,
        load_lidar=True,
        max_lidar_points=args.max_lidar_points,
        load_pose_windows=True,
        pose_window_size=args.pose_window_size,
        max_pose_people=args.max_pose_people,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=crowd_collate, num_workers=0)
    batch = next(iter(loader))

    config = ThreeStreamEncoderConfig(
        model_dim=args.model_dim,
        pose_token_dim=args.model_dim,
        pointpillars=EgoPointPillarsConfig(
            voxel_size=tuple(args.voxel_size),
            point_cloud_range=tuple(args.bev_range),
        ),
    )
    encoder = ThreeStreamWorldModelEncoder(config)
    out = encoder(batch)

    print("Input batch:")
    for key in ("pose_windows", "pose_window_mask", "pose_token_mask", "lidar", "lidar_mask", "ego_state"):
        value = batch[key]
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")

    print("Three-stream encoder output:")
    for key, value in out.items():
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")


if __name__ == "__main__":
    main()
