#!/usr/bin/env python3
"""Portable reference for OmniNxt sparse/dense stereo geometry.

This file intentionally has no ROS, TensorRT or D2SLAM dependency.  It can be
imported by a simulator or used as a CLI to compare 17 left/right keypoints.
"""

import argparse
import json
import sys

import cv2
import numpy as np
import yaml


COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def load_pair(geometry_path, pair_name):
    with open(geometry_path, encoding="utf-8") as stream:
        geometry = yaml.safe_load(stream)
    matches = [pair for pair in geometry["stereo_pairs"]
               if pair["name"] == pair_name]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate pair: " + pair_name)
    return matches[0]


def triangulate_points(pair, left_uv, right_uv, min_disparity=.75,
                       min_depth=.25, max_depth=8.0,
                       max_reprojection_error=3.0):
    """Triangulate corresponding pixels into calibrated ``base_link``."""
    left_uv = np.asarray(left_uv, dtype=np.float64).reshape(-1, 2)
    right_uv = np.asarray(right_uv, dtype=np.float64).reshape(-1, 2)
    if left_uv.shape != right_uv.shape:
        raise ValueError("left/right point arrays must have equal shape")
    p1 = np.asarray(pair["P1_rectified_left"], dtype=np.float64)
    p2 = np.asarray(pair["P2_rectified_right"], dtype=np.float64)
    t_body_rect = np.asarray(
        pair.get("T_base_link_rectleft", pair["T_imu_rectleft"]),
        dtype=np.float64)
    output = []
    for left, right in zip(left_uv, right_uv):
        disparity = float(left[0] - right[0])
        result = {"valid": False, "disparity_px": disparity,
                  "xyz_rectleft_m": None, "xyz_base_link_m": None,
                  "xyz_imu_m": None,
                  "reprojection_error_px": None}
        if not np.all(np.isfinite(np.r_[left, right])) or \
                disparity < min_disparity:
            output.append(result)
            continue
        homogeneous = cv2.triangulatePoints(
            p1, p2, left.reshape(2, 1), right.reshape(2, 1))[:, 0]
        if abs(float(homogeneous[3])) < 1e-12:
            output.append(result)
            continue
        xyz = homogeneous[:3] / homogeneous[3]
        if not np.all(np.isfinite(xyz)) or \
                not (min_depth <= xyz[2] <= max_depth):
            output.append(result)
            continue

        def project(projection):
            pixel = projection.dot(np.r_[xyz, 1.0])
            return pixel[:2] / pixel[2]

        error = max(float(np.linalg.norm(project(p1) - left)),
                    float(np.linalg.norm(project(p2) - right)))
        if error > max_reprojection_error:
            output.append(result)
            continue
        xyz_body = t_body_rect.dot(np.r_[xyz, 1.0])[:3]
        result.update({
            "valid": True,
            "xyz_rectleft_m": xyz.tolist(),
            "xyz_base_link_m": xyz_body.tolist(),
            # Backward-compatible alias for older consumers and Kalibr names.
            "xyz_imu_m": xyz_body.tolist(),
            "reprojection_error_px": error,
        })
        output.append(result)
    return output


def disparity_to_depth_and_body(pair, disparity):
    """Match the C++ cv::reprojectImageTo3D(Q) dense-depth convention."""
    disparity = np.asarray(disparity, dtype=np.float32)
    if disparity.shape != (240, 320):
        raise ValueError("disparity must have shape [240,320]")
    q = np.asarray(pair["Q_disparity_to_rectleft_xyz"], dtype=np.float64)
    xyz_rect = cv2.reprojectImageTo3D(disparity, q, handleMissingValues=False,
                                     ddepth=cv2.CV_32F)
    transform = np.asarray(
        pair.get("T_base_link_rectleft", pair["T_imu_rectleft"]),
        dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    xyz_body = xyz_rect.dot(rotation.T) + translation
    # Runtime /stereo_i/depth is rectified optical Z, not Euclidean range.
    return xyz_rect[..., 2], xyz_rect, xyz_body


def disparity_to_depth_and_imu(pair, disparity):
    """Backward-compatible alias; the returned XYZ frame is ``base_link``."""
    return disparity_to_depth_and_body(pair, disparity)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--left-json")
    parser.add_argument("--right-json")
    parser.add_argument("--disparity-npy")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = arguments()
    pair = load_pair(args.geometry, args.pair)
    result = {"pair": args.pair}
    if args.left_json or args.right_json:
        if not (args.left_json and args.right_json):
            raise ValueError("both --left-json and --right-json are required")
        with open(args.left_json, encoding="utf-8") as stream:
            left = json.load(stream)
        with open(args.right_json, encoding="utf-8") as stream:
            right = json.load(stream)
        points = triangulate_points(pair, left, right)
        result["joints"] = [dict(id=index, name=COCO17[index], **value)
                            for index, value in enumerate(points)]
    if args.disparity_npy:
        disparity = np.load(args.disparity_npy)
        depth, _, xyz_body = disparity_to_depth_and_body(pair, disparity)
        valid = np.isfinite(depth) & (depth > 0)
        result["dense_summary"] = {
            "valid_pixels": int(np.count_nonzero(valid)),
            "median_rectified_z_m": (
                float(np.median(depth[valid])) if np.any(valid) else None),
            "median_xyz_base_link_m": (
                np.median(xyz_body[valid], axis=0).tolist()
                if np.any(valid) else None),
            "median_xyz_imu_m": (
                np.median(xyz_body[valid], axis=0).tolist()
                if np.any(valid) else None),
        }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        sys.stdout.write(encoded + "\n")


if __name__ == "__main__":
    main()
