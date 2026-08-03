#!/usr/bin/env python3
"""Combine per-camera intrinsic and camera-IMU results for OmniNxt."""

import argparse
import os

import yaml


CAMERAS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    args = parser.parse_args()
    prepared = os.path.realpath(args.prepared_dir)
    output = {}

    for index, camera in enumerate(CAMERAS):
        intrinsics_path = os.path.join(
            prepared, camera, "log1-camchain.yaml"
        )
        imu_path = os.path.join(
            prepared, f"{camera}-camchain-imucam.yaml"
        )
        intrinsic = load_yaml(intrinsics_path)["cam0"]
        imu_result = load_yaml(imu_path)["cam0"]
        intrinsic["T_cam_imu"] = imu_result["T_cam_imu"]
        if "timeshift_cam_imu" in imu_result:
            intrinsic["timeshift_cam_imu"] = imu_result["timeshift_cam_imu"]
        intrinsic["rostopic"] = f"/{camera}"
        output[f"cam{index}"] = intrinsic

    output_path = os.path.join(prepared, "fisheye_cams.yaml")
    with open(output_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, default_flow_style=False, sort_keys=False)
    print(output_path)


if __name__ == "__main__":
    main()

