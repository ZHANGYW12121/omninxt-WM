#!/usr/bin/env python3
"""Rebuild the Omni-Depth cloud with Isaac GT disparity instead of HITNet.

This is an A/B diagnostic path.  It deliberately leaves the original ROS/TRT
pipeline untouched, but preserves its downstream geometry:

  Isaac distance_to_camera -> virtual-left axial Z -> d=fB/Z (via Q)
  -> cv2.reprojectImageTo3D -> rectified-to-camera -> camera-to-IMU
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "shared/input"
DEFAULT_OUTPUT = ROOT / "shared/output/isaac_gt_disparity"
DEFAULT_DEBUG = ROOT / "shared/debug/isaac_gt_disparity"
DEFAULT_CFG = ROOT / "D2SLAM/config/quadcam_drone_nxt_isaac"
PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))
SIZE = (320, 240)


def fit_mei_to_ftheta(spec, fov_deg):
    xi, fx, fy, cx, cy = map(float, spec["intrinsics"])
    distortion = spec.get("distortion", spec.get("distortion_coeffs"))
    k1, k2, p1, p2 = map(float, distortion)
    radii, thetas = [], []
    for theta in np.linspace(0.0, math.radians(fov_deg * 0.5), 96):
        st, ct = math.sin(theta), math.cos(theta)
        for phi in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            den = ct + xi
            if abs(den) < 1e-9:
                continue
            x, y = st * math.cos(phi) / den, st * math.sin(phi) / den
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            radii.append(math.hypot(fx * xd, fy * yd))
            thetas.append(theta)
    return np.polyfit(np.asarray(radii), np.asarray(thetas), 4)[::-1]


def write_cloud(path, points):
    points = np.asarray(points, np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n")
        f.write(f"TYPE F F F\nCOUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\n")
        f.write(f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        np.savetxt(f, points, fmt="%.8g")
    ply = path.with_suffix(".ply")
    with ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.8g")


def save_densemap(points, path, title):
    xyz = np.asarray(points, np.float32).reshape(-1, 3)
    canvas = np.full((900, 1600, 3), 18, np.uint8)
    radius = np.linalg.norm(xyz, axis=1)
    keep = np.isfinite(xyz).all(axis=1) & (radius > 0.15) & (radius < 30.0)
    cloud, radius = xyz[keep], radius[keep]
    if len(cloud):
        lo, hi = np.percentile(radius, [2, 98])
        norm = np.clip((radius - lo) / max(hi - lo, 1e-6), 0, 1)
        hsv = np.stack(((1.0 - norm) * 120, np.full(len(cloud), 240),
                        np.full(len(cloud), 255)), axis=1).astype(np.uint8)
        colors = cv2.cvtColor(hsv[:, None, :], cv2.COLOR_HSV2BGR)[:, 0]
        yaw, pitch = np.deg2rad(-35.0), np.deg2rad(24.0)
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        view = cloud @ (rz @ rx).T
        scale = 700.0 / max(np.ptp(view[:, 0]), np.ptp(view[:, 1]), 1e-3)
        u = np.clip((view[:, 0] * scale + 800).astype(int), 0, 1599)
        v = np.clip((450 - view[:, 1] * scale).astype(int), 0, 899)
        order = np.argsort(view[:, 2])[::-1]
        canvas[v[order], u[order]] = colors[order]
        cv2.circle(canvas, (800, 450), 8, (255, 255, 255), -1)
    cv2.putText(canvas, f"{title} | {len(xyz)} points", (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 245, 245), 2)
    cv2.imwrite(str(path), canvas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame", help="frame directory name, e.g. frame_001265")
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--debug-root", type=Path, default=DEFAULT_DEBUG)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--fisheye-fov-deg", type=float, default=220.0,
                    help="FOV used by Isaac's Mei-to-F-theta fit")
    ap.add_argument("--pixel-step", type=int, default=2)
    ap.add_argument("--min-z", type=float, default=0.1)
    ap.add_argument("--max-z", type=float, default=10.0)
    args = ap.parse_args()

    frame = args.input_root / args.frame
    if not frame.is_dir():
        raise FileNotFoundError(frame)
    for cam in range(4):
        for suffix in (".png", "_range.npy"):
            p = frame / f"cam{cam}{suffix}"
            if not p.exists():
                raise FileNotFoundError(p)

    fis = yaml.safe_load((args.config / "fisheye_cams.yaml").read_text())
    coeffs = {i: fit_mei_to_ftheta(fis[f"cam{i}"], args.fisheye_fov_deg)
              for i in range(4)}
    out_dir, dbg_dir = args.output_root, args.debug_root / args.frame
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    # Virtual left view is +45 degrees, matching FisheyeUndist half id 1.
    a = math.pi / 4.0
    r_virtual_to_fish = np.array([[math.cos(a), 0, math.sin(a)],
                                  [0, 1, 0],
                                  [-math.sin(a), 0, math.cos(a)]], np.float64)
    u, v = np.meshgrid(np.arange(SIZE[0], dtype=np.float32),
                       np.arange(SIZE[1], dtype=np.float32))
    sectors, report = [], {"frame": args.frame, "source": "Isaac distance_to_camera",
                           "disparity": "GT derived from rectified axial Z and Q",
                           "pixel_step": args.pixel_step, "sectors": {}}

    for sid, (left, right) in enumerate(PAIRS):
        st = yaml.safe_load((args.config / f"stereo_calib_{left}_{right}_240_320.yaml").read_text())
        intr = list(map(float, st["cam0"]["intrinsics"]))
        k = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], np.float64)
        d0 = np.asarray(st["cam0"]["distortion_coeffs"], np.float64)
        d1 = np.asarray(st["cam1"]["distortion_coeffs"], np.float64)
        baseline = np.asarray(st["cam1"]["T_cn_cnm1"], np.float64)
        r1, _, p1, _, q, _, _ = cv2.stereoRectify(
            k, d0, k, d1, SIZE, baseline[:3, :3], baseline[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        # Rectified pixel ray -> virtual left -> raw fisheye optical ray.
        rect_ray = np.stack(((u - p1[0, 2]) / p1[0, 0],
                             (v - p1[1, 2]) / p1[1, 1], np.ones_like(u)), axis=-1)
        fish_ray_mei = rect_ray @ r1 @ r_virtual_to_fish.T
        fish_ray_mei /= np.linalg.norm(fish_ray_mei, axis=2, keepdims=True)

        c = fis[f"cam{left}"]
        xi, fx, fy, cx, cy = map(float, c["intrinsics"])
        k1, k2, t1, t2 = map(float, c["distortion_coeffs"])
        den = fish_ray_mei[..., 2] + xi
        x, y = fish_ray_mei[..., 0] / den, fish_ray_mei[..., 1] / den
        rr = x * x + y * y
        radial = 1.0 + k1 * rr + k2 * rr * rr
        xd = x * radial + 2.0 * t1 * x * y + t2 * (rr + 2.0 * x * x)
        yd = y * radial + t1 * (rr + 2.0 * y * y) + 2.0 * t2 * x * y
        mx, my = (fx * xd + cx).astype(np.float32), (fy * yd + cy).astype(np.float32)

        rng = np.load(frame / f"cam{left}_range.npy").astype(np.float32)
        image = cv2.imread(str(frame / f"cam{left}.png"), cv2.IMREAD_COLOR)
        sampled_range = cv2.remap(rng, mx, my, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        rect_image = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Record the fitted F-theta/Mei ray mismatch, but keep the GT point on the
        # ideal virtual-pinhole ray.  The virtual images and Q are defined by that
        # model; injecting the small fitted-render ray offset at the same output
        # pixel would intentionally break the pinhole/Q closure.
        dx, dy = mx - cx, my - cy
        rad = np.sqrt(dx * dx + dy * dy)
        theta = sum(coeffs[left][i] * rad ** i for i in range(5))
        phi = np.arctan2(dy, dx)
        fish_ray_gt = np.stack((np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1)
        safe_range = np.where(np.isfinite(sampled_range), sampled_range, 0.0)
        point_fish_gt = fish_ray_mei * safe_range[..., None]
        point_rect_gt = point_fish_gt @ r_virtual_to_fish @ r1.T
        axial_z = point_rect_gt[..., 2]

        # Algebraic inverse of OpenCV Q: Z = Q[2,3]/(Q[3,2]*d + Q[3,3]).
        disparity = np.full(axial_z.shape, np.nan, dtype=np.float64)
        nonzero_z = np.isfinite(axial_z) & (np.abs(axial_z) > 1e-9)
        disparity[nonzero_z] = ((q[2, 3] / axial_z[nonzero_z] - q[3, 3]) /
                                q[3, 2])
        valid = (np.isfinite(disparity) & np.isfinite(sampled_range) &
                 (sampled_range > 0.03) & (sampled_range < 30.0) &
                 (disparity > 0.0) & (axial_z > args.min_z) & (axial_z < args.max_z) &
                 (mx >= 0) & (mx < image.shape[1] - 1) &
                 (my >= 0) & (my < image.shape[0] - 1))
        disparity_safe = disparity.astype(np.float32)
        disparity_safe[~valid] = np.nan

        # Keep the same Q reconstruction and rectified->left->IMU transform as Omni-Depth.
        point_rect_q = cv2.reprojectImageTo3D(disparity_safe, q)
        point_fish_q = point_rect_q @ r1 @ r_virtual_to_fish.T
        cam_to_imu = np.linalg.inv(np.asarray(c["T_cam_imu"], np.float64))
        point_imu = point_fish_q @ cam_to_imu[:3, :3].T + cam_to_imu[:3, 3]
        sampled = np.zeros_like(valid)
        sampled[::args.pixel_step, ::args.pixel_step] = True
        cloud = point_imu[valid & sampled]
        sectors.append(cloud)

        prefix = dbg_dir / f"stereo{sid}_gt"
        cv2.imwrite(str(prefix) + "_disparity.tiff", disparity_safe)
        show = np.nan_to_num(disparity_safe, nan=0.0)
        show = cv2.applyColorMap(np.clip(show * 255.0 / 32.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        show[~valid] = 0
        cv2.imwrite(str(prefix) + "_disparity.png", show)
        cv2.imwrite(str(prefix) + "_left_rect.png", rect_image)
        sector_path = dbg_dir / f"stereo{sid}_gt.pcd"
        write_cloud(sector_path, cloud)
        closure = np.linalg.norm(point_rect_q[valid] - point_rect_gt[valid], axis=1)
        ray_dot = np.clip(np.sum(fish_ray_mei * fish_ray_gt, axis=2), -1.0, 1.0)
        ray_error_deg = np.degrees(np.arccos(ray_dot))[valid]
        report["sectors"][f"stereo{sid}"] = {
            "left_camera": left, "right_camera": right, "points": int(len(cloud)),
            "valid_disparity_pixels": int(valid.sum()),
            "disparity_px_percentiles": np.percentile(disparity[valid], [1, 5, 50, 95, 99]).tolist(),
            "q_closure_rmse_m": float(np.sqrt(np.mean(closure * closure))),
            "q_closure_max_m": float(np.max(closure)),
            "ftheta_mei_ray_error_deg_rms": float(np.sqrt(np.mean(ray_error_deg * ray_error_deg))),
            "ftheta_mei_ray_error_deg_max": float(np.max(ray_error_deg)),
        }

    merged = np.concatenate(sectors, axis=0)
    base = out_dir / f"{args.frame}_gt_disparity"
    write_cloud(base.with_suffix(".pcd"), merged)
    save_densemap(merged, base.with_name(base.name + "_densemap.png"), "Omni-Depth Isaac GT disparity")
    report["points"] = int(len(merged))
    report["output_pcd"] = str(base.with_suffix(".pcd"))
    (base.with_name(base.name + "_stats.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
