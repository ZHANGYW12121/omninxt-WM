#!/usr/bin/env python3
"""Export the exact runtime virtual-stereo geometry for simulation.

Every homogeneous matrix is named T_target_source and maps a homogeneous
point expressed in ``source`` into ``target``.
"""

import argparse
import math
import os

import cv2
import numpy as np
import yaml


PAIRS = (
    ("A_B_RIGHT", 0, 1, "right", "stereo_calib_0_1_240_320.yaml"),
    ("B_C_REAR", 1, 2, "rear", "stereo_calib_1_2_240_320.yaml"),
    ("C_D_LEFT", 2, 3, "left", "stereo_calib_2_3_240_320.yaml"),
    ("D_A_FRONT", 3, 0, "front", "stereo_calib_3_0_240_320.yaml"),
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def matrix4(rotation, translation=None):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def k_and_d(camera):
    fx, fy, cx, cy = camera["intrinsics"]
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(camera["distortion_coeffs"], dtype=np.float64)
    return intrinsic, distortion


def clean(value):
    if isinstance(value, np.ndarray):
        return [[float(number) for number in row] for row in value.tolist()]
    return value


def main():
    args = arguments()
    with open(os.path.join(args.config_dir, "fisheye_cams.yaml"),
              encoding="utf-8") as stream:
        raw_cameras = yaml.safe_load(stream)

    angle = math.pi / 4.0
    r_cam_virtual_left = np.array(
        [[math.cos(angle), 0.0, math.sin(angle)],
         [0.0, 1.0, 0.0],
         [-math.sin(angle), 0.0, math.cos(angle)]], dtype=np.float64)
    r_cam_virtual_right = np.array(
        [[math.cos(-angle), 0.0, math.sin(-angle)],
         [0.0, 1.0, 0.0],
         [-math.sin(-angle), 0.0, math.cos(-angle)]], dtype=np.float64)

    exported = {
        "schema": "omninxt_sim_geometry_v1",
        "matrix_convention": (
            "T_target_source maps homogeneous source coordinates to target"
        ),
        "body_frame": {
            "name": "base_link", "convention": "ROS_FLU",
            "x": "front", "y": "left", "z": "up",
            "origin": "fixed Kalibr body/IMU calibration origin",
            "runtime_flight_controller_required": False,
        },
        "legacy_name_note": (
            "T_*_imu keys are retained for Kalibr compatibility; imu denotes "
            "the fixed base_link calibration origin, not a runtime IMU input"
        ),
        "image_convention": {"x": "right", "y": "down", "z": "forward"},
        "runtime": {
            "raw_resolution": [1280, 720],
            "rectified_resolution": [320, 240],
            "configured_fov_deg": 190.0,
            "virtual_pinhole_fov_deg": 100.0,
            "virtual_view_yaw_deg": {"left_member": 45.0,
                                     "right_member": -45.0},
            "stereo_rectify_flags": "CALIB_ZERO_DISPARITY",
            "stereo_rectify_alpha": -1.0,
        },
        "physical_mapping": {
            "CAM_A": "FRONT_RIGHT", "CAM_B": "REAR_RIGHT",
            "CAM_C": "REAR_LEFT", "CAM_D": "FRONT_LEFT",
            "clockwise_order_from_above": ["CAM_A", "CAM_B", "CAM_C", "CAM_D"],
        },
        "raw_cameras": {},
        "stereo_pairs": [],
    }

    for camera_id in range(4):
        camera = raw_cameras["cam{}".format(camera_id)]
        t_camera_imu = np.asarray(camera["T_cam_imu"], dtype=np.float64)
        t_imu_camera = np.linalg.inv(t_camera_imu)
        exported["raw_cameras"]["CAM_{}".format(chr(65 + camera_id))] = {
            "model": "Mei/omni+radtan",
            "intrinsics_xi_fx_fy_cx_cy": [float(v) for v in camera["intrinsics"]],
            "distortion_k1_k2_p1_p2": [
                float(v) for v in camera["distortion_coeffs"]],
            "resolution": camera["resolution"],
            "T_camera_base_link": clean(t_camera_imu),
            "T_base_link_camera": clean(t_imu_camera),
            "T_camera_imu": clean(t_camera_imu),
            "T_imu_camera": clean(t_imu_camera),
            "timeshift_cam_imu_s": float(camera["timeshift_cam_imu"]),
        }

    for pair_id, (name, left_id, right_id, side, filename) in enumerate(PAIRS):
        with open(os.path.join(args.config_dir, filename),
                  encoding="utf-8") as stream:
            stereo = yaml.safe_load(stream)
        k0, d0 = k_and_d(stereo["cam0"])
        k1, d1 = k_and_d(stereo["cam1"])
        t_rightvirtual_leftvirtual = np.asarray(
            stereo["cam1"]["T_cn_cnm1"], dtype=np.float64)
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            k0, d0, k1, d1, (320, 240),
            t_rightvirtual_leftvirtual[:3, :3],
            t_rightvirtual_leftvirtual[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
        )

        t_imu_leftcamera = np.linalg.inv(np.asarray(
            raw_cameras["cam{}".format(left_id)]["T_cam_imu"],
            dtype=np.float64))
        t_imu_leftvirtual = t_imu_leftcamera.dot(
            matrix4(r_cam_virtual_left))
        t_imu_rectleft = t_imu_leftvirtual.dot(matrix4(r1.T))

        # X_right_rect = R2 * (R * R1^T * X_left_rect + T)
        t_rectright_rectleft = matrix4(
            r2.dot(t_rightvirtual_leftvirtual[:3, :3]).dot(r1.T),
            r2.dot(t_rightvirtual_leftvirtual[:3, 3]),
        )
        t_imu_rectright = t_imu_rectleft.dot(
            np.linalg.inv(t_rectright_rectleft))
        optical_axis = t_imu_rectleft[:3, :3].dot(
            np.array([0.0, 0.0, 1.0]))
        azimuth_deg = math.degrees(math.atan2(
            optical_axis[1], optical_axis[0]))

        exported["stereo_pairs"].append({
            "id": pair_id,
            "name": name,
            "physical_side": side,
            "left_raw_camera": "CAM_{}".format(chr(65 + left_id)),
            "right_raw_camera": "CAM_{}".format(chr(65 + right_id)),
            "left_virtual_view_index": 1,
            "right_virtual_view_index": 0,
            "source_calibration": filename,
            "K_left_virtual": clean(k0),
            "D_left_virtual_k1_k2_p1_p2": [float(v) for v in d0],
            "K_right_virtual": clean(k1),
            "D_right_virtual_k1_k2_p1_p2": [float(v) for v in d1],
            "T_rightvirtual_leftvirtual": clean(t_rightvirtual_leftvirtual),
            "baseline_m": float(np.linalg.norm(
                t_rightvirtual_leftvirtual[:3, 3])),
            "R1_leftvirtual_to_rectleft": clean(r1),
            "R2_rightvirtual_to_rectright": clean(r2),
            "P1_rectified_left": clean(p1),
            "P2_rectified_right": clean(p2),
            "Q_disparity_to_rectleft_xyz": clean(q),
            "valid_roi_left_xywh": [int(v) for v in roi1],
            "valid_roi_right_xywh": [int(v) for v in roi2],
            "T_imu_leftvirtual": clean(t_imu_leftvirtual),
            "T_imu_rectleft": clean(t_imu_rectleft),
            "T_imu_rectright": clean(t_imu_rectright),
            "T_base_link_leftvirtual": clean(t_imu_leftvirtual),
            "T_base_link_rectleft": clean(t_imu_rectleft),
            "T_base_link_rectright": clean(t_imu_rectright),
            "rectified_optical_axis_in_imu": [float(v) for v in optical_axis],
            "rectified_optical_axis_azimuth_deg": float(azimuth_deg),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        yaml.safe_dump(exported, stream, sort_keys=False,
                       allow_unicode=True, width=120)


if __name__ == "__main__":
    main()
