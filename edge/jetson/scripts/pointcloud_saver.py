#!/usr/bin/env python3
"""Save one finite, non-empty PointCloud2 message as an ASCII PCD."""

import argparse
import math
import os
import struct
import sys
import threading

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--topic", default="/depth_estimation/pointcloud")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--skip-messages", type=int, default=0,
        help="Ignore initial messages while a lazy debug publisher warms up",
    )
    args = parser.parse_args()

    rospy.init_node("omninxt_pointcloud_saver", anonymous=True)
    selected = {}
    received = 0
    ready = threading.Event()

    def callback(message):
        nonlocal received
        received += 1
        if received > args.skip_messages and not ready.is_set():
            selected["cloud"] = message
            ready.set()

    subscriber = rospy.Subscriber(
        args.topic, PointCloud2, callback, queue_size=2,
        buff_size=32 * 1024 * 1024,
    )
    if not ready.wait(args.timeout):
        subscriber.unregister()
        rospy.logerr("pointcloud timeout on %s", args.topic)
        return 2
    subscriber.unregister()
    cloud = selected["cloud"]

    fields = {field.name for field in cloud.fields}
    if not {"x", "y", "z"}.issubset(fields):
        rospy.logerr("PointCloud2 lacks x/y/z fields: %s", sorted(fields))
        return 3
    has_rgb = "rgb" in fields
    field_names = ("x", "y", "z", "rgb") if has_rgb else ("x", "y", "z")
    points = []
    for values in point_cloud2.read_points(
            cloud, field_names=field_names, skip_nans=False):
        x, y, z = (float(values[0]), float(values[1]), float(values[2]))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        if has_rgb:
            packed = struct.unpack("<I", struct.pack("<f", float(values[3])))[0]
            points.append((x, y, z, packed))
        else:
            points.append((x, y, z))
    if not points:
        rospy.logerr("PointCloud2 contains no finite XYZ points")
        return 4

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="ascii") as handle:
        field_header = (
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F U\n"
            "COUNT 1 1 1 1\n"
            if has_rgb else
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
        )
        handle.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            + field_header +
            "WIDTH {}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS {}\n"
            "DATA ascii\n".format(len(points), len(points))
        )
        for point in points:
            if has_rgb:
                x, y, z, rgb = point
                handle.write(
                    "{:.9g} {:.9g} {:.9g} {}\n".format(x, y, z, rgb)
                )
            else:
                x, y, z = point
                handle.write("{:.9g} {:.9g} {:.9g}\n".format(x, y, z))
    rospy.loginfo("saved %d finite points to %s", len(points), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
