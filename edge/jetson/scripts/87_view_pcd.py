#!/usr/bin/env python3
"""Interactive XYZ PCD viewer using the host's existing Matplotlib."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers projection="3d"


def load_ascii_pcd(path):
    data_line = None
    with open(path, encoding="ascii") as stream:
        for line_number, line in enumerate(stream):
            if line.strip().lower() == "data ascii":
                data_line = line_number + 1
                break
    if data_line is None:
        raise ValueError("Only ASCII PCD files are supported")
    points = np.loadtxt(path, skiprows=data_line, usecols=(0, 1, 2))
    return np.atleast_2d(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcd")
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--max-range", type=float, default=10.0)
    parser.add_argument("--save", help="Save a PNG instead of opening a window")
    args = parser.parse_args()

    path = os.path.realpath(args.pcd)
    points = load_ascii_pcd(path)
    finite = np.isfinite(points).all(axis=1)
    ranges = np.linalg.norm(points, axis=1)
    points = points[finite & (ranges <= args.max_range)]
    ranges = np.linalg.norm(points, axis=1)
    if not len(points):
        raise SystemExit("No finite points remain after range filtering")
    if len(points) > args.max_points:
        indices = np.linspace(0, len(points) - 1, args.max_points, dtype=int)
        points = points[indices]
        ranges = ranges[indices]

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    cloud = axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=ranges,
        cmap="viridis",
        s=0.8,
        linewidths=0,
    )
    figure.colorbar(cloud, ax=axis, pad=0.08, label="Range (m)")
    axis.set_xlabel("X / IMU frame (m)")
    axis.set_ylabel("Y / IMU frame (m)")
    axis.set_zlabel("Z / IMU frame (m)")
    axis.set_title(f"{os.path.basename(path)} — {len(points)} displayed points")

    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    half = max(high - low) / 2.0
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.view_init(elev=24, azim=-55)
    figure.tight_layout()

    if args.save:
        figure.savefig(args.save, dpi=180)
        print(os.path.realpath(args.save))
    else:
        print("Drag to rotate; use the mouse wheel to zoom; close the window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
