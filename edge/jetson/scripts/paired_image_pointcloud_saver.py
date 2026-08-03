#!/usr/bin/env python3
"""Save the closest timestamped assembled image and OmniDepth point cloud."""

import argparse
import json
import math
import os
import threading
from collections import deque

import cv2
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2


def image_to_bgr(message):
    if message.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    packed = np.frombuffer(message.data, dtype=np.uint8)
    rows = packed.reshape(message.height, message.step)
    image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
    if message.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def finite_xyz(message):
    return [
        (float(x), float(y), float(z))
        for x, y, z in point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
    ]


def write_pcd(path, points):
    with open(path, "w", encoding="ascii") as stream:
        stream.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA ascii\n"
        )
        for x, y, z in points:
            stream.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


def save_images(directory, assembled):
    if assembled.shape != (720, 5120, 3):
        raise ValueError(f"Expected 5120x720 bgr8 image, got {assembled.shape}")
    raw_path = os.path.join(directory, "assembled_CAM_A_B_C_D.png")
    if not cv2.imwrite(raw_path, assembled):
        raise RuntimeError(f"Failed to save {raw_path}")

    names = ("CAM_A_FRONT_RIGHT", "CAM_B_REAR_RIGHT",
             "CAM_C_REAR_LEFT", "CAM_D_FRONT_LEFT")
    previews = []
    for index, name in enumerate(names):
        image = assembled[:, index * 1280 : (index + 1) * 1280]
        path = os.path.join(directory, f"{name}.png")
        if not cv2.imwrite(path, image):
            raise RuntimeError(f"Failed to save {path}")
        preview = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(preview, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(
            preview, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2, cv2.LINE_AA
        )
        previews.append(preview)
    labeled_path = os.path.join(directory, "assembled_labeled_preview.png")
    if not cv2.imwrite(labeled_path, np.hstack(previews)):
        raise RuntimeError(f"Failed to save {labeled_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-delta-ms", type=float, default=60.0)
    args = parser.parse_args()
    output = os.path.abspath(args.output_directory)
    os.makedirs(output, exist_ok=False)

    lock = threading.Lock()
    finished = threading.Event()
    images = deque(maxlen=8)
    selected = {}

    def image_callback(message):
        with lock:
            images.append((message.header.stamp.to_nsec(), message))

    def cloud_callback(message):
        if finished.is_set():
            return
        cloud_stamp = message.header.stamp.to_nsec()
        with lock:
            # Ignore clouds already queued before this subscriber accumulated
            # enough source images. The depth node now preserves the exact
            # inference-input timestamp, so its matching image should remain
            # in this short history.
            if len(images) < 4:
                return
            image_stamp, image_message = min(
                images, key=lambda item: abs(item[0] - cloud_stamp)
            )
            delta_ms = abs(image_stamp - cloud_stamp) / 1_000_000.0
            if delta_ms > args.max_delta_ms:
                return
            selected.update(
                image=image_message,
                cloud=message,
                image_stamp=image_stamp,
                cloud_stamp=cloud_stamp,
                delta_ms=delta_ms,
            )
            finished.set()

    rospy.init_node("omninxt_paired_capture", anonymous=True)
    image_sub = rospy.Subscriber(
        "/oak_ffc_4p/assemble_image", Image, image_callback,
        queue_size=8, buff_size=48 * 1024 * 1024
    )
    cloud_sub = rospy.Subscriber(
        "/depth_estimation/pointcloud", PointCloud2, cloud_callback,
        queue_size=2, buff_size=16 * 1024 * 1024
    )
    if not finished.wait(args.timeout):
        raise RuntimeError("Timed out waiting for a timestamp-matched image and point cloud")
    image_sub.unregister()
    cloud_sub.unregister()

    assembled = image_to_bgr(selected["image"])
    points = finite_xyz(selected["cloud"])
    if not points:
        raise RuntimeError("Matched point cloud contains no finite XYZ points")
    save_images(output, assembled)
    pcd_path = os.path.join(output, "pointcloud.pcd")
    write_pcd(pcd_path, points)

    metadata = {
        "image_topic": "/oak_ffc_4p/assemble_image",
        "pointcloud_topic": "/depth_estimation/pointcloud",
        "image_seq": selected["image"].header.seq,
        "pointcloud_seq": selected["cloud"].header.seq,
        "image_stamp_ns": selected["image_stamp"],
        "pointcloud_stamp_ns": selected["cloud_stamp"],
        "absolute_stamp_delta_ms": selected["delta_ms"],
        "image_encoding": selected["image"].encoding,
        "image_width": selected["image"].width,
        "image_height": selected["image"].height,
        "pointcloud_frame_id": selected["cloud"].header.frame_id,
        "finite_xyz_points": len(points),
        "pipeline_note": (
            "The depth node carries the exact inference-input image header "
            "through TensorRT and uses it for the output cloud."
        ),
    }
    with open(os.path.join(output, "metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
