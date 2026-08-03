#!/usr/bin/env python3
"""Generate Isaac camera profiles from the authoritative sync geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


CAMERA_KEYS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")


def inverse_rigid(matrix):
    rotation = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    translation = [float(matrix[row][3]) for row in range(3)]
    rotation_t = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(rotation_t[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [rotation_t[row] + [inverse_translation[row]] for row in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]
    ]


def raw_profile(geometry):
    cameras = {}
    for index, physical_name in enumerate(CAMERA_KEYS):
        source = geometry["raw_cameras"][physical_name]
        cameras[f"cam{index}"] = {
            "alias": f"{physical_name}_{geometry['physical_mapping'][physical_name]}",
            "physical_name": physical_name,
            "sensor_name": f"sync_raw_{physical_name.lower()}",
            "kind": "raw_mei",
            "projection_model": "mei_exact_via_equidistant_bridge",
            "intrinsics": source["intrinsics_xi_fx_fy_cx_cy"],
            "distortion": source["distortion_k1_k2_p1_p2"],
            "T_cam_imu": source["T_camera_base_link"],
            "target_resolution": source["resolution"],
            "render_resolution": [1280, 1280],
            "render_fov_deg": 250.0,
        }
    return {
        "name": "omninxt_sync_20260802_raw_mei",
        "mode": "raw_mei",
        "fps": 20.0,
        "processing_fps": 10.0,
        "camera_order": list(cameras),
        "active_camera": "cam0",
        "projection_model": "mei_unified",
        "distortion_model": "radtan",
        "native_isaac_bridge": "equidistant_ftheta_then_exact_mei_remap",
        "parent_frame": "base_link",
        "body_frame": "ROS_FLU",
        "optical_frame": "OpenCV",
        "rotate_180": False,
        "cameras": cameras,
    }


def pinhole_spec(name, alias, kind, resolution, intrinsic, transform, **extra):
    fx, fy, cx, cy = (float(value) for value in intrinsic)
    result = {
        "alias": alias,
        "sensor_name": name,
        "kind": kind,
        "projection_model": "pinhole",
        "intrinsics": [fx, fy, cx, cy],
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "T_cam_imu": inverse_rigid(transform),
        "target_resolution": [int(value) for value in resolution],
        "render_resolution": [int(value) for value in resolution],
    }
    result.update(extra)
    return result


def validation_profile(geometry):
    cameras = {}
    anchor_width, anchor_height = 416, 320
    anchor_focal = anchor_width / (2.0 * math.tan(math.radians(60.0)))
    for index, physical_name in enumerate(CAMERA_KEYS):
        raw = geometry["raw_cameras"][physical_name]
        cameras[f"anchor{index}"] = pinhole_spec(
            f"sync_anchor_{physical_name.lower()}",
            f"{physical_name}_anchor_120deg",
            "anchor",
            [anchor_width, anchor_height],
            [anchor_focal, anchor_focal, anchor_width * 0.5, anchor_height * 0.5],
            raw["T_base_link_camera"],
            physical_name=physical_name,
            anchor_index=index,
        )
    for pair in geometry["stereo_pairs"]:
        pair_id = int(pair["id"])
        p1 = pair["P1_rectified_left"]
        p2 = pair["P2_rectified_right"]
        left_k = [p1[0][0], p1[1][1], p1[0][2], p1[1][2]]
        right_k = [p2[0][0], p2[1][1], p2[0][2], p2[1][2]]
        cameras[f"rect{pair_id}_left"] = pinhole_spec(
            f"sync_rect{pair_id}_left",
            f"{pair['name']}_left",
            "rectified_left",
            [320, 240],
            left_k,
            pair["T_base_link_rectleft"],
            pair_id=pair_id,
            pair_name=pair["name"],
            side="left",
        )
        cameras[f"rect{pair_id}_right"] = pinhole_spec(
            f"sync_rect{pair_id}_right",
            f"{pair['name']}_right",
            "rectified_right",
            [320, 240],
            right_k,
            pair["T_base_link_rectright"],
            pair_id=pair_id,
            pair_name=pair["name"],
            side="right",
        )
    return {
        "name": "omninxt_sync_20260802_rectified_validation",
        "mode": "rectified_validation",
        "fps": 10.0,
        "processing_fps": 10.0,
        "camera_order": list(cameras),
        "active_camera": "rect3_left",
        "projection_model": "pinhole_direct",
        "distortion_model": "none",
        "parent_frame": "base_link",
        "body_frame": "ROS_FLU",
        "optical_frame": "OpenCV",
        "rotate_180": False,
        "cameras": cameras,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.geometry, encoding="utf-8") as stream:
        geometry = yaml.safe_load(stream)
    output = {
        "schema": "omninxt_isaac_camera_profiles_v1",
        "source_geometry": str(Path(args.geometry).resolve()),
        "matrix_convention": geometry["matrix_convention"],
        "default_profile": "raw_mei",
        "render": {
            "pixel_size_um": 3.0,
            "f_stop": 0.0,
            "focus_distance_m": 3.0,
            "clipping_range_m": [0.03, 100.0],
        },
        "profiles": {
            "raw_mei": raw_profile(geometry),
            "rectified_validation": validation_profile(geometry),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
