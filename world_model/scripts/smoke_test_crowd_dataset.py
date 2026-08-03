#!/usr/bin/env python3
"""Quickly check the Isaac crowd Dataset adapter without modifying raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from datasets.isaac_crowd import IsaacCrowdSequenceDataset, build_episode_index, crowd_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--index_path", type=Path, default=None)
    parser.add_argument("--pose_cache_root", type=Path, default=None)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--min_frames", type=int, default=30)
    parser.add_argument("--load_images", action="store_true")
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=90)
    parser.add_argument("--load_lidar", action="store_true")
    parser.add_argument("--max_lidar_points", type=int, default=4096)
    parser.add_argument("--load_pose_windows", action="store_true")
    parser.add_argument("--pose_window_size", type=int, default=8)
    parser.add_argument("--max_pose_people", type=int, default=16)
    parser.add_argument("--pose_min_score", type=float, default=0.05)
    parser.add_argument(
        "--run_pose_encoder",
        action="store_true",
        help="Run CausalMultiPersonSTGCNEncoder on one batch when pose windows are loaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.index_path is None:
        episodes = build_episode_index(args.data_root, min_frames=args.min_frames)
        print(f"Accepted episodes from raw scan: {len(episodes)}")
        ds = IsaacCrowdSequenceDataset(
            episodes=episodes,
            sequence_length=args.sequence_length,
            stride=args.stride,
            pose_cache_root=args.pose_cache_root,
            load_images=args.load_images,
            image_size=(args.image_width, args.image_height) if args.load_images else None,
            load_lidar=args.load_lidar,
            max_lidar_points=args.max_lidar_points if args.load_lidar else None,
            load_pose_windows=args.load_pose_windows,
            pose_window_size=args.pose_window_size,
            max_pose_people=args.max_pose_people,
            pose_min_score=args.pose_min_score,
        )
    else:
        ds = IsaacCrowdSequenceDataset(
            index_path=args.index_path,
            sequence_length=args.sequence_length,
            stride=args.stride,
            pose_cache_root=args.pose_cache_root,
            load_images=args.load_images,
            image_size=(args.image_width, args.image_height) if args.load_images else None,
            load_lidar=args.load_lidar,
            max_lidar_points=args.max_lidar_points if args.load_lidar else None,
            load_pose_windows=args.load_pose_windows,
            pose_window_size=args.pose_window_size,
            max_pose_people=args.max_pose_people,
            pose_min_score=args.pose_min_score,
        )
        print(f"Loaded index: {args.index_path}")

    print(f"Dataset chunks: {len(ds)}")
    sample = ds[0]
    print("One sample:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape} dtype={getattr(value, 'dtype', None)}")
        elif isinstance(value, list):
            preview = value[0] if value else None
            print(f"  {key}: list len={len(value)} first={preview}")
        else:
            print(f"  {key}: {value}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=crowd_collate, num_workers=0)
    batch = next(iter(loader))
    print("One batch:")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)} dtype={getattr(value, 'dtype', None)}")
        elif isinstance(value, list):
            print(f"  {key}: list len={len(value)}")
        else:
            print(f"  {key}: {value}")

    if args.run_pose_encoder:
        if not args.load_pose_windows:
            raise ValueError("--run_pose_encoder requires --load_pose_windows.")
        from modules import CausalMultiPersonSTGCNEncoder

        encoder = CausalMultiPersonSTGCNEncoder(out_dim=128)
        encoded = encoder(
            batch["pose_windows"],
            pose_window_mask=batch["pose_window_mask"],
            pose_token_mask=batch["pose_token_mask"],
            image_size_hw=batch.get("pose_image_size_hw"),
        )
        print("Pose encoder output:")
        for key, value in encoded.items():
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")


if __name__ == "__main__":
    main()
