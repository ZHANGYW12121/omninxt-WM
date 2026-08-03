#!/usr/bin/env python3
"""Validate the four calibrated OmniNxt virtual stereo pairs."""

import argparse
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import yaml

try:
    import rosbag
except ImportError:
    rosbag = None


PAIRS = (
    ("0_1", "/cam_0_1/compressed", "/cam_1_0/compressed"),
    ("1_2", "/cam_1_1/compressed", "/cam_2_0/compressed"),
    ("2_3", "/cam_2_1/compressed", "/cam_3_0/compressed"),
    ("3_0", "/cam_3_1/compressed", "/cam_0_0/compressed"),
)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def matrix_from_camera(camera):
    fx, fy, cx, cy = camera["intrinsics"]
    return np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))


def decode(message):
    return cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def extract_reprojection_std(path):
    text = open(path, encoding="utf-8").read()
    found = re.findall(r"reprojection error:.*?\+- \[([^\]]+)\]", text)
    if len(found) != 2:
        raise RuntimeError(f"Could not parse reprojection errors from {path}")
    return [[float(value) for value in item.replace(",", " ").split()] for item in found]


def build_rectification(config):
    cam0 = config["cam0"]
    cam1 = config["cam1"]
    size = tuple(cam0["resolution"])
    if size != tuple(cam1["resolution"]) or size != (320, 240):
        raise RuntimeError(f"Unexpected stereo resolution: {size}")
    k0 = matrix_from_camera(cam0)
    k1 = matrix_from_camera(cam1)
    d0 = np.asarray(cam0["distortion_coeffs"], dtype=np.float64)
    d1 = np.asarray(cam1["distortion_coeffs"], dtype=np.float64)
    transform = np.asarray(cam1["T_cn_cnm1"], dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    r0, r1, p0, p1, _, _, _ = cv2.stereoRectify(
        k0,
        d0,
        k1,
        d1,
        size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map0 = cv2.initUndistortRectifyMap(k0, d0, r0, p0, size, cv2.CV_32FC1)
    map1 = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_32FC1)
    return rotation, translation, map0, map1


def orb_vertical_residual(left, right):
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=8)
    key0, desc0 = detector.detectAndCompute(left, None)
    key1, desc1 = detector.detectAndCompute(right, None)
    if desc0 is None or desc1 is None:
        return [], []
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc0, desc1, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    residuals = [abs(key0[item.queryIdx].pt[1] - key1[item.trainIdx].pt[1]) for item in good]
    return residuals, [(key0[item.queryIdx].pt, key1[item.trainIdx].pt) for item in good]


def save_evidence(path, left, right, matches):
    canvas = cv2.cvtColor(np.hstack((left, right)), cv2.COLOR_GRAY2BGR)
    for row in range(20, canvas.shape[0], 20):
        cv2.line(canvas, (0, row), (canvas.shape[1] - 1, row), (0, 170, 0), 1)
    for point0, point1 in matches[:30]:
        p0 = tuple(int(round(value)) for value in point0)
        p1 = (int(round(point1[0])) + left.shape[1], int(round(point1[1])))
        cv2.line(canvas, p0, p1, (0, 190, 255), 1)
    cv2.imwrite(path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = os.path.realpath(args.directory)
    os.makedirs(args.output, exist_ok=True)

    all_topics = {topic for _, left, right in PAIRS for topic in (left, right)}
    counts = defaultdict(int)
    stamps = defaultdict(list)
    samples = defaultdict(dict)
    if rosbag is None:
        raise RuntimeError("Run this validator in the ROS calibration container")

    with rosbag.Bag(args.bag) as bag:
        for topic, message, _ in bag.read_messages(topics=all_topics):
            counts[topic] += 1
            stamp = message.header.stamp.to_nsec()
            stamps[topic].append(stamp)
            # Store evenly spread candidates while keeping memory bounded.
            if counts[topic] % 97 == 1:
                samples[topic][stamp] = decode(message)

    report = []
    baselines = []
    failures = []
    for pair, topic0, topic1 in PAIRS:
        yaml_path = os.path.join(directory, f"stereo_calib_{pair}_240_320.yaml")
        results_path = os.path.join(directory, f"stereo_calib_{pair}_240_320-results.txt")
        config = load_yaml(yaml_path)
        rotation, translation, map0, map1 = build_rectification(config)
        baseline = float(np.linalg.norm(translation))
        baselines.append(baseline)
        orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        determinant = float(np.linalg.det(rotation))
        rotation_angle = math.degrees(
            math.acos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        )
        reprojection = extract_reprojection_std(results_path)

        if counts[topic0] != counts[topic1] or stamps[topic0] != stamps[topic1]:
            failures.append(f"{pair}: stereo timestamps are not exactly synchronized")
        common = sorted(set(samples[topic0]) & set(samples[topic1]))
        residuals = []
        best = None
        for stamp in common:
            left = cv2.remap(samples[topic0][stamp], map0[0], map0[1], cv2.INTER_LINEAR)
            right = cv2.remap(samples[topic1][stamp], map1[0], map1[1], cv2.INTER_LINEAR)
            current, matches = orb_vertical_residual(left, right)
            residuals.extend(current)
            if best is None or len(current) > best[0]:
                best = (len(current), left, right, matches)
        if not residuals:
            failures.append(f"{pair}: no matched features for epipolar validation")
            median = p95 = float("nan")
        else:
            median = float(np.median(residuals))
            p95 = float(np.percentile(residuals, 95))
        if best is not None:
            save_evidence(
                os.path.join(args.output, f"rectified_{pair}.png"),
                best[1],
                best[2],
                best[3],
            )

        if not 0.12 <= baseline <= 0.18:
            failures.append(f"{pair}: baseline {baseline:.6f} m is implausible")
        if orthogonality > 1e-6 or abs(determinant - 1.0) > 1e-6:
            failures.append(f"{pair}: invalid rotation matrix")
        if max(max(axis) for axis in reprojection) > 0.35:
            failures.append(f"{pair}: reprojection standard deviation is too high")
        report.append(
            f"{pair}: messages={counts[topic0]}, baseline={baseline * 1000:.3f} mm, "
            f"rotation={rotation_angle:.4f} deg, reproj_std={reprojection}, "
            f"ORB_repeated_grid_diagnostic_median={median:.3f} px, p95={p95:.3f} px"
        )

    spread = (max(baselines) - min(baselines)) * 1000.0
    if spread > 2.0:
        failures.append(f"baseline spread {spread:.3f} mm is too high")
    report.append(f"baseline_spread={spread:.3f} mm")
    report.append(
        "note=ORB residual is non-authoritative because repeated AprilGrid cells "
        "create ambiguous descriptor matches; Kalibr ID-aware corner residual is the criterion."
    )
    report.append("status=" + ("FAIL" if failures else "PASS"))
    if failures:
        report.extend(f"failure={item}" for item in failures)
    report_path = os.path.join(args.output, "virtual_stereo_validation.txt")
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    print("\n".join(report))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
