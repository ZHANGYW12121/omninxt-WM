#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import re
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Undistort an OmniNxt Mei+radtan fisheye image and run RTMPose."
    )
    parser.add_argument(
        "--input",
        default="omninxt_cam3_left_cam_preview.jpg",
        help="Raw distorted OmniNxt fisheye image.",
    )
    parser.add_argument(
        "--config",
        default="omninxt_b033501_camera_config.json",
        help="OmniNxt camera calibration JSON.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera id in config, for example cam3. If omitted, infer from input name.",
    )
    parser.add_argument(
        "--out-dir",
        default="omninxt_undistort_pose_outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--projection",
        default="cylindrical",
        choices=("cylindrical", "pinhole"),
        help="Undistorted output model. cylindrical follows the OmniNxt/Omni-VINS path.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output image width.")
    parser.add_argument("--height", type=int, default=720, help="Output image height.")
    parser.add_argument(
        "--cyl-h-fov-deg",
        type=float,
        default=190.0,
        help="Horizontal FoV for cylindrical undistortion.",
    )
    parser.add_argument(
        "--pinhole-h-fov-deg",
        type=float,
        default=120.0,
        help="Horizontal FoV for pinhole undistortion.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="RTMPose device.",
    )
    parser.add_argument(
        "--pose-mode",
        default="balanced",
        choices=("performance", "lightweight", "balanced"),
        help="RTMPose Body model mode.",
    )
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--skip-pose", action="store_true")
    return parser.parse_args()


def infer_camera_id(image_path):
    match = re.search(r"(cam\d+)", Path(image_path).name)
    if match:
        return match.group(1)
    return "cam0"


def load_camera_spec(config_path, camera_id):
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)

    rig = config["camera_rig"]
    cameras = rig["cameras"]
    if camera_id not in cameras:
        raise KeyError(f"{camera_id!r} is not present in {config_path}")
    return rig, cameras[camera_id]


def mei_radtan_project(ray_x, ray_y, ray_z, spec):
    xi, fx, fy, cx, cy = [float(v) for v in spec["intrinsics"]]
    k1, k2, p1, p2 = [float(v) for v in spec["distortion"]]

    d = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    denom = ray_z + xi * d

    x = ray_x / denom
    y = ray_y / denom
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2

    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    map_x = fx * x_dist + cx
    map_y = fy * y_dist + cy
    return map_x, map_y


def build_cylindrical_map(width, height, src_width, src_height, spec, h_fov_deg):
    # OmniNxt/Omni-VINS cylindrical preprocessing:
    # phi = atan2(X, Z), v = f * Y / sqrt(X^2 + Z^2)
    h_fov_rad = np.deg2rad(float(h_fov_deg))
    f_phi = float(width) / h_fov_rad
    u0 = 0.5 * (width - 1)
    v0 = 0.5 * (height - 1)

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    phi = (uu - u0) / f_phi
    y_over_rho = (vv - v0) / f_phi

    ray_x = np.sin(phi)
    ray_y = y_over_rho
    ray_z = np.cos(phi)
    return finish_remap(mei_radtan_project(ray_x, ray_y, ray_z, spec), src_width, src_height)


def build_pinhole_map(width, height, src_width, src_height, spec, h_fov_deg):
    h_fov_rad = np.deg2rad(float(h_fov_deg))
    focal = 0.5 * float(width) / np.tan(0.5 * h_fov_rad)
    u0 = 0.5 * (width - 1)
    v0 = 0.5 * (height - 1)

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    ray_x = (uu - u0) / focal
    ray_y = (vv - v0) / focal
    ray_z = np.ones_like(ray_x)
    return finish_remap(mei_radtan_project(ray_x, ray_y, ray_z, spec), src_width, src_height)


def finish_remap(maps, src_width, src_height):
    map_x, map_y = maps
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= src_width - 1)
        & (map_y >= 0.0)
        & (map_y <= src_height - 1)
    )
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid


def undistort_image(image, spec, args):
    src_height, src_width = image.shape[:2]
    width = int(args.width)
    height = int(args.height)

    if args.projection == "cylindrical":
        map_x, map_y, valid = build_cylindrical_map(
            width, height, src_width, src_height, spec, args.cyl_h_fov_deg
        )
    else:
        map_x, map_y, valid = build_pinhole_map(
            width, height, src_width, src_height, spec, args.pinhole_h_fov_deg
        )

    undistorted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return undistorted, valid


def safe_draw_skeleton(img, keypoints, scores, kpt_thr=0.3, openpose_skeleton=False):
    from rtmlib import draw_skeleton

    sig = inspect.signature(draw_skeleton)
    kwargs = {"kpt_thr": kpt_thr}
    if "openpose_skeleton" in sig.parameters:
        kwargs["openpose_skeleton"] = openpose_skeleton
    elif "to_openpose" in sig.parameters:
        kwargs["to_openpose"] = openpose_skeleton
    return draw_skeleton(img, keypoints, scores, **kwargs)


def draw_joint_ids(img, keypoints, scores, kpt_thr=0.3):
    vis = img.copy()
    if keypoints.ndim != 3:
        return vis

    for person_id in range(keypoints.shape[0]):
        for joint_id in range(keypoints.shape[1]):
            if scores[person_id, joint_id] < kpt_thr:
                continue
            x, y = keypoints[person_id, joint_id, :2]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            x_i, y_i = int(round(float(x))), int(round(float(y)))
            cv2.circle(vis, (x_i, y_i), 4, (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(joint_id),
                (x_i + 5, y_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return vis


def select_main_person(keypoints, scores):
    if keypoints is None or scores is None:
        return np.zeros((17, 3), dtype=np.float32)

    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    if keypoints.ndim != 3 or scores.ndim != 2 or keypoints.shape[0] == 0:
        return np.zeros((17, 3), dtype=np.float32)

    best_id = int(np.argmax(scores.mean(axis=1)))
    xy = keypoints[best_id]
    conf = scores[best_id, :, None]
    skeleton_xyc = np.concatenate([xy, conf], axis=-1).astype(np.float32)
    skeleton_xyc[~np.isfinite(skeleton_xyc)] = 0.0
    return skeleton_xyc


def run_rtmpose(image_path, out_dir, device, mode, kpt_thr):
    from rtmlib import Body

    pose_dir = out_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image for RTMPose: {image_path}")

    pose_model = Body(
        to_openpose=False,
        mode=mode,
        backend="onnxruntime",
        device=device,
    )

    keypoints, scores = pose_model(img)
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    skeleton_xyc = select_main_person(keypoints, scores)

    stem = image_path.stem
    np.save(pose_dir / f"{stem}_keypoints.npy", keypoints)
    np.save(pose_dir / f"{stem}_scores.npy", scores)
    np.save(pose_dir / f"{stem}_skeleton_xyc.npy", skeleton_xyc)

    vis = img.copy()
    if keypoints.ndim == 3 and keypoints.shape[0] > 0:
        try:
            vis = safe_draw_skeleton(vis, keypoints, scores, kpt_thr=kpt_thr)
        except Exception as exc:
            print(f"[Warning] draw_skeleton failed: {exc}")
        vis = draw_joint_ids(vis, keypoints, scores, kpt_thr=kpt_thr)
    vis_path = pose_dir / f"{stem}_rtmpose.jpg"
    cv2.imwrite(str(vis_path), vis)

    summary = {
        "person_count": int(keypoints.shape[0]) if keypoints.ndim == 3 else 0,
        "main_person_mean_score": float(skeleton_xyc[:, 2].mean()),
        "keypoints_path": str(pose_dir / f"{stem}_keypoints.npy"),
        "scores_path": str(pose_dir / f"{stem}_scores.npy"),
        "skeleton_xyc_path": str(pose_dir / f"{stem}_skeleton_xyc.npy"),
        "visualization_path": str(vis_path),
    }
    return summary


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_id = args.camera or infer_camera_id(input_path)
    rig, spec = load_camera_spec(args.config, camera_id)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    undistorted, valid = undistort_image(image, spec, args)
    output_image = out_dir / f"{input_path.stem}_{args.projection}_undistorted.jpg"
    cv2.imwrite(str(output_image), undistorted)

    metadata = {
        "input": str(input_path),
        "output_image": str(output_image),
        "config": str(args.config),
        "camera": camera_id,
        "camera_alias": spec.get("alias", camera_id),
        "projection": args.projection,
        "output_width": int(args.width),
        "output_height": int(args.height),
        "valid_pixel_ratio": float(valid.mean()),
        "projection_params": {
            "cyl_h_fov_deg": float(args.cyl_h_fov_deg),
            "pinhole_h_fov_deg": float(args.pinhole_h_fov_deg),
        },
        "model": {
            "projection_model": rig.get("projection_model"),
            "distortion_model": rig.get("distortion_model"),
            "intrinsics": spec["intrinsics"],
            "distortion": spec["distortion"],
        },
    }

    if not args.skip_pose:
        metadata["rtmpose"] = run_rtmpose(
            output_image,
            out_dir=out_dir,
            device=args.device,
            mode=args.pose_mode,
            kpt_thr=args.kpt_thr,
        )

    metadata_path = out_dir / f"{input_path.stem}_{args.projection}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved undistorted image: {output_image}")
    print(f"Saved metadata: {metadata_path}")
    if "rtmpose" in metadata:
        print(f"Saved RTMPose visualization: {metadata['rtmpose']['visualization_path']}")
        print(f"Main-person mean score: {metadata['rtmpose']['main_person_mean_score']:.4f}")


if __name__ == "__main__":
    main()
