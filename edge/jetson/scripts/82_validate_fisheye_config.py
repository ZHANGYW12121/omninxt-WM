#!/usr/bin/env python3
"""Validate the four-camera Kalibr results and their common-IMU geometry."""

import argparse
import json
import math
import os
import re

import numpy as np
import yaml


CAMERAS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")
ADJACENT = (
    ("CAM_A", "CAM_B"),
    ("CAM_B", "CAM_C"),
    ("CAM_C", "CAM_D"),
    ("CAM_D", "CAM_A"),
)


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def last_metric(text, label):
    pattern = rf"{re.escape(label)}:\s+mean ([0-9.eE+-]+)"
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from result file")
    return float(matches[-1])


def rotation_angle_degrees(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    args = parser.parse_args()
    prepared = os.path.realpath(args.prepared_dir)

    combined_path = os.path.join(prepared, "fisheye_cams.yaml")
    combined = load_yaml(combined_path)
    transforms = {}
    cameras = {}
    failures = []

    for index, camera in enumerate(CAMERAS):
        entry = combined[f"cam{index}"]
        transform = np.asarray(entry["T_cam_imu"], dtype=float)
        rotation = transform[:3, :3]
        determinant = float(np.linalg.det(rotation))
        orthogonality_error = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3))
        )
        transforms[camera] = transform

        results_path = os.path.join(prepared, f"{camera}-results-imucam.txt")
        with open(results_path, encoding="utf-8") as stream:
            results_text = stream.read()

        camera_result = {
            "timeshift_cam_imu_s": float(entry["timeshift_cam_imu"]),
            "rotation_determinant": determinant,
            "rotation_orthogonality_error": orthogonality_error,
            "reprojection_mean_px": last_metric(
                results_text, "Reprojection error (cam0) [px]"
            ),
            "gyroscope_mean_rad_s": last_metric(
                results_text, "Gyroscope error (imu0) [rad/s]"
            ),
            "accelerometer_mean_m_s2": last_metric(
                results_text, "Accelerometer error (imu0) [m/s^2]"
            ),
        }
        cameras[camera] = camera_result

        if abs(determinant - 1.0) > 1e-6:
            failures.append(f"{camera}: rotation determinant is {determinant}")
        if orthogonality_error > 1e-6:
            failures.append(
                f"{camera}: rotation orthogonality error is "
                f"{orthogonality_error}"
            )
        if camera_result["reprojection_mean_px"] > 0.8:
            failures.append(
                f"{camera}: reprojection mean exceeds 0.8 px"
            )

    adjacent = {}
    closure = np.eye(4)
    for source, target in ADJACENT:
        relative = transforms[target] @ np.linalg.inv(transforms[source])
        rotation_degrees = rotation_angle_degrees(relative[:3, :3])
        baseline_m = float(np.linalg.norm(relative[:3, 3]))
        name = f"{source}-{target}"
        adjacent[name] = {
            "rotation_degrees": rotation_degrees,
            "baseline_m": baseline_m,
            "T_target_source": relative.tolist(),
        }
        closure = relative @ closure
        if not 75.0 <= rotation_degrees <= 105.0:
            failures.append(
                f"{name}: adjacent rotation is {rotation_degrees:.3f} deg"
            )
        if not 0.08 <= baseline_m <= 0.25:
            failures.append(f"{name}: baseline is {baseline_m:.4f} m")

    shifts = [item["timeshift_cam_imu_s"] for item in cameras.values()]
    shift_spread = max(shifts) - min(shifts)
    closure_rotation = rotation_angle_degrees(closure[:3, :3])
    closure_translation = float(np.linalg.norm(closure[:3, 3]))
    if shift_spread > 0.005:
        failures.append(
            f"camera-IMU time-shift spread is {shift_spread:.6f} s"
        )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "source": combined_path,
        "cameras": cameras,
        "adjacent_pairs": adjacent,
        "timeshift_spread_s": shift_spread,
        "common_reference_closure": {
            "rotation_degrees": closure_rotation,
            "translation_m": closure_translation,
            "note": (
                "This closure is algebraic because all four transforms share "
                "the same IMU reference; virtual-stereo validation remains "
                "an independent downstream check."
            ),
        },
        "failures": failures,
    }

    json_path = os.path.join(prepared, "fisheye_calibration_validation.json")
    text_path = os.path.join(prepared, "fisheye_calibration_validation.txt")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    with open(text_path, "w", encoding="utf-8") as stream:
        stream.write(f"status: {report['status']}\n")
        for camera, item in cameras.items():
            stream.write(
                f"{camera}: reprojection={item['reprojection_mean_px']:.6f} px, "
                f"gyro={item['gyroscope_mean_rad_s']:.8f} rad/s, "
                f"accel={item['accelerometer_mean_m_s2']:.8f} m/s^2, "
                f"timeshift={item['timeshift_cam_imu_s']:.9f} s\n"
            )
        for name, item in adjacent.items():
            stream.write(
                f"{name}: rotation={item['rotation_degrees']:.6f} deg, "
                f"baseline={item['baseline_m']:.6f} m\n"
            )
        stream.write(f"timeshift_spread={shift_spread:.9f} s\n")
        stream.write(
            f"common_reference_closure_rotation={closure_rotation:.9f} deg\n"
        )
        stream.write(
            f"common_reference_closure_translation={closure_translation:.12f} m\n"
        )
        for failure in failures:
            stream.write(f"failure: {failure}\n")

    print(text_path)
    print(json_path)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
