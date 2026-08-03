#!/usr/bin/env python3
"""Build a four-fisheye Isaac GT cloud directly from distance-to-camera range."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from process_isaac_gt_disparity import fit_mei_to_ftheta, save_densemap, write_cloud


ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame")
    ap.add_argument("--input-root", type=Path, default=ROOT / "shared/input")
    ap.add_argument("--output-root", type=Path, default=ROOT / "shared/output/isaac_direct_gt")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "D2SLAM/config/quadcam_drone_nxt_isaac")
    ap.add_argument("--fisheye-fov-deg", type=float, default=220.0)
    ap.add_argument("--pixel-step", type=int, default=4)
    ap.add_argument("--min-range", type=float, default=0.10)
    ap.add_argument("--max-range", type=float, default=30.0)
    args = ap.parse_args()

    frame = args.input_root / args.frame
    fis = yaml.safe_load((args.config / "fisheye_cams.yaml").read_text())
    sectors, stats = [], {"frame": args.frame, "source": "Isaac direct distance_to_camera",
                          "frame_id": "imu", "cameras": {}}

    for cam in range(4):
        spec = fis[f"cam{cam}"]
        image = cv2.imread(str(frame / f"cam{cam}.png"), cv2.IMREAD_COLOR)
        rng = np.load(frame / f"cam{cam}_range.npy").astype(np.float32)
        if image is None or rng.shape != image.shape[:2]:
            raise ValueError(f"cam{cam}: RGB/range shape mismatch")

        _, _, _, cx, cy = map(float, spec["intrinsics"])
        coeff = fit_mei_to_ftheta(spec, args.fisheye_fov_deg)
        yy, xx = np.mgrid[0:rng.shape[0], 0:rng.shape[1]]
        dx, dy = xx.astype(np.float64) - cx, yy.astype(np.float64) - cy
        radius = np.hypot(dx, dy)
        theta = sum(coeff[i] * radius ** i for i in range(5))
        phi = np.arctan2(dy, dx)
        ray = np.stack((np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1)
        valid = (np.isfinite(rng) & (rng >= args.min_range) &
                 (rng <= args.max_range) & (theta <= np.deg2rad(args.fisheye_fov_deg / 2)))
        sampled = np.zeros_like(valid)
        sampled[::args.pixel_step, ::args.pixel_step] = True
        valid &= sampled

        point_cam = ray[valid] * rng[valid, None]
        cam_to_imu = np.linalg.inv(np.asarray(spec["T_cam_imu"], np.float64))
        point_imu = point_cam @ cam_to_imu[:3, :3].T + cam_to_imu[:3, 3]
        sectors.append(point_imu.astype(np.float32))
        stats["cameras"][f"cam{cam}"] = {
            "points": int(len(point_imu)),
            "range_percentiles_m": np.percentile(rng[valid], [1, 5, 50, 95, 99]).tolist(),
        }

    cloud = np.concatenate(sectors, axis=0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = args.output_root / f"{args.frame}_direct_gt"
    write_cloud(base.with_suffix(".pcd"), cloud)
    save_densemap(cloud, base.with_name(base.name + "_densemap.png"), "Isaac direct GT")
    stats["points"] = int(len(cloud))
    stats["output_pcd"] = str(base.with_suffix(".pcd"))
    base.with_name(base.name + "_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
