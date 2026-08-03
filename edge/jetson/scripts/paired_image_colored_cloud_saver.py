#!/usr/bin/env python3
"""Save a timestamp-matched assembled image and sector-colored point cloud."""

import argparse
import json
import math
import os
import struct
import threading
from collections import deque

import cv2
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2


def image_to_bgr(message):
    packed = np.frombuffer(message.data, dtype=np.uint8)
    rows = packed.reshape(message.height, message.step)
    image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
    if message.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif message.encoding != "bgr8":
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    return image.copy()


def colored_points(message):
    result = []
    for x, y, z, rgb_float in point_cloud2.read_points(
            message, field_names=("x", "y", "z", "rgb"), skip_nans=False):
        x, y, z = float(x), float(y), float(z)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        packed = struct.unpack("<I", struct.pack("<f", float(rgb_float)))[0]
        result.append((x, y, z, packed))
    return result


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
        for x, y, z, rgb in points:
            stream.write(f"{x:.9g} {y:.9g} {z:.9g} {rgb}\n")


def save_images(output, image):
    if image.shape != (720, 5120, 3):
        raise ValueError(f"Expected 5120x720 bgr8 image, got {image.shape}")
    raw = os.path.join(output, "assembled_CAM_A_B_C_D.png")
    if not cv2.imwrite(raw, image):
        raise RuntimeError(f"Failed to save {raw}")
    names = (
        "CAM_A_FRONT_RIGHT", "CAM_B_REAR_RIGHT",
        "CAM_C_REAR_LEFT", "CAM_D_FRONT_LEFT",
    )
    previews = []
    for index, name in enumerate(names):
        camera = image[:, index * 1280:(index + 1) * 1280]
        cv2.imwrite(os.path.join(output, f"{name}.png"), camera)
        preview = cv2.resize(camera, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(preview, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(
            preview, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        previews.append(preview)
    labeled = np.hstack(previews)
    preview_path = os.path.join(output, "assembled_labeled_preview.jpg")
    if not cv2.imwrite(preview_path, labeled, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Failed to save {preview_path}")
    return raw, preview_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-delta-ms", type=float, default=2.0)
    args = parser.parse_args()
    output = os.path.abspath(args.output_directory)
    os.makedirs(output, exist_ok=True)

    images = deque(maxlen=12)
    lock = threading.Lock()
    ready = threading.Event()
    selected = {}
    cloud_count = 0

    def image_callback(message):
        with lock:
            images.append((message.header.stamp.to_nsec(), message))

    def cloud_callback(message):
        nonlocal cloud_count
        cloud_count += 1
        if ready.is_set() or cloud_count <= 2:
            return
        stamp = message.header.stamp.to_nsec()
        with lock:
            if len(images) < 4:
                return
            image_stamp, image = min(images, key=lambda item: abs(item[0] - stamp))
            delta_ms = abs(image_stamp - stamp) / 1_000_000.0
            if delta_ms > args.max_delta_ms:
                return
            selected.update(
                image=image, cloud=message, image_stamp=image_stamp,
                cloud_stamp=stamp, delta_ms=delta_ms,
            )
            ready.set()

    rospy.init_node("omninxt_html_capture", anonymous=True)
    image_sub = rospy.Subscriber(
        "/oak_ffc_4p/assemble_image", Image, image_callback,
        queue_size=12, buff_size=48 * 1024 * 1024,
    )
    cloud_sub = rospy.Subscriber(
        "/depth_estimation/pointcloud_sectors", PointCloud2, cloud_callback,
        queue_size=3, buff_size=32 * 1024 * 1024,
    )
    if not ready.wait(args.timeout):
        raise RuntimeError("Timed out waiting for an exact image/cloud pair")
    image_sub.unregister()
    cloud_sub.unregister()

    image = image_to_bgr(selected["image"])
    points = colored_points(selected["cloud"])
    if not points:
        raise RuntimeError("Colored cloud contains no finite points")
    raw_image, preview = save_images(output, image)
    pcd = os.path.join(output, "pointcloud_sectors.pcd")
    write_pcd(pcd, points)
    metadata = {
        "image_stamp_ns": selected["image_stamp"],
        "pointcloud_stamp_ns": selected["cloud_stamp"],
        "absolute_stamp_delta_ms": selected["delta_ms"],
        "finite_xyzrgb_points": len(points),
        "image": raw_image,
        "preview": preview,
        "pcd": pcd,
    }
    with open(os.path.join(output, "metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
