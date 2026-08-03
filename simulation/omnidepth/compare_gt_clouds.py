#!/usr/bin/env python3
"""Compare OmniDepth GT-disparity and direct Isaac GT point clouds."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def read_ascii_pcd(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "data ascii") + 1
    return np.loadtxt(lines[start:], dtype=np.float32).reshape(-1, 3)


def distances(source, target):
    return cKDTree(target).query(source, k=1, workers=-1)[0]


def summary(values):
    return {"mean_m": float(np.mean(values)), "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p50_m": float(np.percentile(values, 50)), "p90_m": float(np.percentile(values, 90)),
            "p95_m": float(np.percentile(values, 95)), "p99_m": float(np.percentile(values, 99))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "shared/output")
    ap.add_argument("--max-radius", type=float, default=30.0)
    args = ap.parse_args()
    omni_path = args.root / "isaac_gt_disparity" / f"{args.frame}_gt_disparity.pcd"
    direct_path = args.root / "isaac_direct_gt" / f"{args.frame}_direct_gt.pcd"
    omni, direct = read_ascii_pcd(omni_path), read_ascii_pcd(direct_path)
    omni = omni[np.isfinite(omni).all(1) & (np.linalg.norm(omni, axis=1) < args.max_radius)]
    direct = direct[np.isfinite(direct).all(1) & (np.linalg.norm(direct, axis=1) < args.max_radius)]
    od, do = distances(omni, direct), distances(direct, omni)
    report = {"frame": args.frame, "frame_id": "imu", "omni_points": len(omni),
              "direct_gt_points": len(direct), "omni_to_direct": summary(od),
              "direct_to_omni": summary(do),
              "symmetric_chamfer_mean_m": float(0.5 * (od.mean() + do.mean()))}
    out_dir = args.root / "comparison" / args.frame
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#111")
    views = [(omni, "OmniDepth + GT disparity", "cyan"),
             (direct, "Isaac direct GT", "orange")]
    for ax, (cloud, title, color) in zip(axes[:2], views):
        idx = rng.choice(len(cloud), min(len(cloud), 80000), replace=False)
        ax.scatter(cloud[idx, 0], cloud[idx, 1], s=.25, c=color, alpha=.65)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_facecolor("#111")
    oi = rng.choice(len(omni), min(len(omni), 50000), replace=False)
    di = rng.choice(len(direct), min(len(direct), 50000), replace=False)
    axes[2].scatter(direct[di, 0], direct[di, 1], s=.2, c="orange", alpha=.35, label="direct GT")
    axes[2].scatter(omni[oi, 0], omni[oi, 1], s=.2, c="cyan", alpha=.35, label="OmniDepth")
    axes[2].set_title("Top-view overlay"); axes[2].legend(); axes[2].set_aspect("equal")
    axes[2].set_facecolor("#111")
    for ax in axes:
        ax.tick_params(colors="white"); ax.title.set_color("white"); ax.set_xlabel("IMU x [m]")
        ax.set_ylabel("IMU y [m]"); ax.grid(alpha=.12)
    fig.suptitle(f"{args.frame} | Chamfer={report['symmetric_chamfer_mean_m']:.3f} m", color="white")
    fig.tight_layout(); fig.savefig(out_dir / "comparison.png", dpi=160, facecolor=fig.get_facecolor())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
