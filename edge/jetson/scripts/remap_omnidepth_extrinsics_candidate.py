#!/usr/bin/env python3
"""Create a diagnostic PCD using the documented OmniNxt/Kalibr pose direction."""

import argparse
import math
import os

import numpy as np
import yaml


COLORS_TO_CAMERA = {
    0xFFFF0000: 0,
    0xFF00FF00: 1,
    0xFF0000FF: 2,
    0xFFFFFF00: 3,
}


def read_pcd(path):
    points = []
    data = False
    with open(path, encoding="ascii") as stream:
        for line in stream:
            if data:
                values = line.split()
                if len(values) >= 4:
                    xyz = [float(value) for value in values[:3]]
                    if all(math.isfinite(value) for value in xyz):
                        points.append((*xyz, int(values[3])))
            elif line.strip().lower() == "data ascii":
                data = True
    return points


def write_pcd(path, points):
    with open(path, "w", encoding="ascii") as stream:
        stream.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F U\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA ascii\n"
        )
        for x, y, z, color in points:
            stream.write(f"{x:.9g} {y:.9g} {z:.9g} {color}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pcd")
    parser.add_argument("fisheye_yaml")
    parser.add_argument("output_pcd")
    args = parser.parse_args()
    with open(args.fisheye_yaml, encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream)

    angle = math.pi / 4.0
    virtual = np.eye(4)
    virtual[:3, :3] = np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    transforms = {}
    for camera_id in range(4):
        imu_to_camera = np.asarray(
            cameras[f"cam{camera_id}"]["T_cam_imu"], dtype=float
        )
        transforms[camera_id] = (
            imu_to_camera @ virtual,
            np.linalg.inv(imu_to_camera) @ virtual,
        )

    result = []
    for x, y, z, color in read_pcd(args.input_pcd):
        camera_id = COLORS_TO_CAMERA[color]
        current, corrected = transforms[camera_id]
        point = np.array([x, y, z])
        virtual_point = current[:3, :3].T @ (point - current[:3, 3])
        corrected_point = (
            corrected[:3, :3] @ virtual_point + corrected[:3, 3]
        )
        result.append((*corrected_point.tolist(), color))
    output = os.path.realpath(args.output_pcd)
    write_pcd(output, result)
    print(output)


if __name__ == "__main__":
    main()
