#!/usr/bin/env python3
"""Smoke-test structured RSSM features with the three prediction heads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rssm
from modules import (
    CrowdWorldModelPredictionHeads,
    EgoPointPillarsConfig,
    PredictionHeadConfig,
    ThreeStreamEncoderConfig,
    ThreeStreamWorldModelEncoder,
)
from smoke_test_structured_rssm import load_real_batch, make_rssm_config, make_synthetic_batch


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
    parser.add_argument("--pred_hidden", type=int, default=128)
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
    initial = model.initial(batch["action"].shape[0])
    post_stoch, post_deter, post_logit = model.observe(
        encoded,
        batch["action"],
        initial,
        batch["is_first"],
    )
    feat = model.get_feat(post_stoch, post_deter)

    pred_config = PredictionHeadConfig(
        hidden_dim=args.pred_hidden,
        max_pose_people=args.max_pose_people,
        bev_hw=tuple(int(x) for x in encoder.pointpillars.feature_grid_size),
    )
    heads = CrowdWorldModelPredictionHeads(model.feat_size, pred_config).to(device)
    pred, losses, metrics = heads.forward_loss(feat, batch, encoded)

    print("Prediction output:")
    print(f"  human.xy: {tuple(pred['human']['xy'].shape)} {pred['human']['xy'].dtype}")
    print(f"  human.keypoint_conf_logits: {tuple(pred['human']['keypoint_conf_logits'].shape)}")
    print(f"  human.person_logits: {tuple(pred['human']['person_logits'].shape)}")
    print(f"  env.occupancy_logits: {tuple(pred['env']['occupancy_logits'].shape)}")
    print(f"  ego.state: {tuple(pred['ego']['state'].shape)}")
    print("Prediction losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.detach().item():.6f}")
    print("Prediction metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value.detach().item():.6f}")


if __name__ == "__main__":
    main()
