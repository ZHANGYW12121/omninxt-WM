#!/usr/bin/env python3
"""Smoke-test three-stream encoder + structured RSSM posterior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rssm
from datasets.isaac_crowd import IsaacCrowdSequenceDataset, crowd_collate
from modules import EgoPointPillarsConfig, ThreeStreamEncoderConfig, ThreeStreamWorldModelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--index_path", type=Path, default=None)
    parser.add_argument("--pose_cache_root", type=Path, default=None)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_lidar_points", type=int, default=1024)
    parser.add_argument("--pose_window_size", type=int, default=4)
    parser.add_argument("--max_pose_people", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--rssm_stoch", type=int, default=16)
    parser.add_argument("--rssm_discrete", type=int, default=16)
    parser.add_argument("--rssm_deter", type=int, default=128)
    parser.add_argument("--rssm_hidden", type=int, default=128)
    parser.add_argument("--rssm_blocks", type=int, default=4)
    parser.add_argument("--structured_layers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--bev_range",
        type=float,
        nargs=6,
        default=(-16.0, -8.0, -2.0, 16.0, 40.0, 8.0),
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--voxel_size", type=float, nargs=2, default=(0.5, 0.5), metavar=("VX", "VY"))
    return parser.parse_args()


def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def make_synthetic_batch(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    b = int(args.batch_size)
    t = int(args.sequence_length)
    m = int(args.max_pose_people)
    w = int(args.pose_window_size)
    n = int(args.max_lidar_points)
    x_min, y_min, z_min, x_max, y_max, z_max = args.bev_range

    pose_windows = torch.zeros(b, t, m, w, 17, 3, device=device)
    pose_windows[..., 0] = torch.rand(b, t, m, w, 17, device=device) * 640.0
    pose_windows[..., 1] = torch.rand(b, t, m, w, 17, device=device) * 480.0
    pose_windows[..., 2] = torch.rand(b, t, m, w, 17, device=device)
    pose_window_mask = torch.rand(b, t, m, w, device=device) > 0.1
    pose_token_mask = pose_window_mask[..., -1]

    lidar = torch.empty(b, t, n, 3, device=device)
    lidar[..., 0].uniform_(float(x_min), float(x_max))
    lidar[..., 1].uniform_(float(y_min), float(y_max))
    lidar[..., 2].uniform_(float(z_min), float(z_max))
    lidar_mask = torch.ones(b, t, n, dtype=torch.bool, device=device)

    is_first = torch.zeros(b, t, 1, dtype=torch.bool, device=device)
    is_first[:, 0] = True
    return {
        "pose_windows": pose_windows,
        "pose_window_mask": pose_window_mask,
        "pose_token_mask": pose_token_mask,
        "pose_image_size_hw": torch.tensor([480.0, 640.0], device=device).view(1, 1, 2).expand(b, t, 2),
        "lidar": lidar,
        "lidar_mask": lidar_mask,
        "ego_state": torch.randn(b, t, 17, device=device),
        "action": torch.randn(b, t, 4, device=device).clamp(-1.0, 1.0),
        "is_first": is_first,
    }


def load_real_batch(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.data_root is None or args.pose_cache_root is None:
        raise ValueError("Real-data mode requires --data_root and --pose_cache_root.")
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
    return to_device(next(iter(loader)), device)


def make_rssm_config(args: argparse.Namespace, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        stoch=args.rssm_stoch,
        deter=args.rssm_deter,
        hidden=args.rssm_hidden,
        discrete=args.rssm_discrete,
        img_layers=2,
        obs_layers=1,
        dyn_layers=1,
        blocks=args.rssm_blocks,
        act="SiLU",
        norm=True,
        unimix_ratio=0.01,
        initial="learned",
        device=str(device),
        structured_posterior=SimpleNamespace(
            hidden=args.rssm_hidden,
            layers=args.structured_layers,
            dropout=0.0,
        ),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    batch = (
        load_real_batch(args, device)
        if args.data_root is not None or args.pose_cache_root is not None
        else make_synthetic_batch(args, device)
    )

    encoder_config = ThreeStreamEncoderConfig(
        model_dim=args.model_dim,
        pose_token_dim=args.model_dim,
        pointpillars=EgoPointPillarsConfig(
            voxel_size=tuple(args.voxel_size),
            point_cloud_range=tuple(args.bev_range),
        ),
    )
    encoder = ThreeStreamWorldModelEncoder(encoder_config).to(device)
    model = rssm.RSSM(make_rssm_config(args, device), encoder.rssm_embed_size, act_dim=4).to(device)

    encoded = encoder(batch)
    b = batch["action"].shape[0]
    initial = model.initial(b)
    post_stoch, post_deter, post_logit = model.observe(
        encoded,
        batch["action"],
        initial,
        batch["is_first"],
    )
    prior_stoch, prior_logit = model.prior(post_deter)
    dyn_loss, rep_loss = model.kl_loss(post_logit, prior_logit, free=1.0)
    feat = model.get_feat(post_stoch, post_deter)

    print("Structured encoder output:")
    for key in ("human_embed", "env_embed", "ego_embed", "human_tokens", "env_tokens", "bev_map"):
        value = encoded[key]
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")
    print("Structured RSSM posterior output:")
    print(f"  post_stoch: {tuple(post_stoch.shape)} {post_stoch.dtype}")
    print(f"  post_deter: {tuple(post_deter.shape)} {post_deter.dtype}")
    print(f"  post_logit: {tuple(post_logit.shape)} {post_logit.dtype}")
    print(f"  prior_stoch: {tuple(prior_stoch.shape)} {prior_stoch.dtype}")
    print(f"  prior_logit: {tuple(prior_logit.shape)} {prior_logit.dtype}")
    print(f"  feat: {tuple(feat.shape)} {feat.dtype}")
    print(f"  dyn_loss: {tuple(dyn_loss.shape)} mean={dyn_loss.mean().item():.4f}")
    print(f"  rep_loss: {tuple(rep_loss.shape)} mean={rep_loss.mean().item():.4f}")


if __name__ == "__main__":
    main()
