#!/usr/bin/env python3
"""Capture one exact-stamp set of four rectified-left and depth images."""

import argparse
import json
import os
import threading
from collections import defaultdict

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


PAIRS = (
    ("A_B_RIGHT", "CAM_A", "CAM_B", "right"),
    ("B_C_REAR", "CAM_B", "CAM_C", "rear"),
    ("C_D_LEFT", "CAM_C", "CAM_D", "left"),
    ("D_A_FRONT", "CAM_D", "CAM_A", "front"),
)
KINDS = ("left", "depth")


def decode_image(message):
    if message.encoding == "mono8":
        dtype = np.dtype(np.uint8)
    elif message.encoding == "32FC1":
        dtype = np.dtype(np.float32)
    else:
        raise ValueError("Unsupported encoding: " + message.encoding)
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    row_values = message.step // dtype.itemsize
    array = np.frombuffer(message.data, dtype=dtype).reshape(
        message.height, row_values
    )[:, : message.width]
    return array.astype(dtype.newbyteorder("="), copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    required = {(index, kind) for index in range(4) for kind in KINDS}
    buckets = defaultdict(dict)
    lock = threading.Lock()
    done = threading.Event()
    selected = {}
    selected_stamp = None

    def callback(message, key):
        nonlocal selected, selected_stamp
        image = decode_image(message)
        stamp = message.header.stamp.to_nsec()
        with lock:
            buckets[stamp][key] = image
            if required.issubset(buckets[stamp]) and not done.is_set():
                selected = dict(buckets[stamp])
                selected_stamp = stamp
                done.set()
            for old_stamp in sorted(buckets)[:-40]:
                buckets.pop(old_stamp, None)

    rospy.init_node("capture_pose_depth_frame", anonymous=True, disable_signals=True)
    subscribers = []
    for index in range(4):
        for kind in KINDS:
            subscribers.append(
                rospy.Subscriber(
                    "/depth_estimation/stereo_{}/{}".format(index, kind),
                    Image,
                    callback,
                    callback_args=(index, kind),
                    queue_size=1,
                    buff_size=8 * 1024 * 1024,
                )
            )

    if not done.wait(args.timeout):
        with lock:
            recent = {
                str(stamp): sorted("{}:{}".format(*key) for key in value)
                for stamp, value in sorted(buckets.items())[-10:]
            }
        raise RuntimeError(
            "Timed out waiting for 8 exact-stamp topics: "
            + json.dumps(recent, sort_keys=True)
        )
    for subscriber in subscribers:
        subscriber.unregister()

    metadata = {
        "stamp_ns": selected_stamp,
        "frame": "imu",
        "exact_header_match": True,
        "pairs": [],
    }
    for index, (name, left_camera, right_camera, physical_side) in enumerate(PAIRS):
        pair_dir = os.path.join(output_dir, name)
        os.makedirs(pair_dir, exist_ok=True)
        left = selected[(index, "left")]
        depth = selected[(index, "depth")]
        if left.shape != (240, 320) or depth.shape != left.shape:
            raise RuntimeError("Unexpected image shape for " + name)
        cv2.imwrite(os.path.join(pair_dir, "left.png"), left)
        np.save(os.path.join(pair_dir, "depth_m.npy"), depth.astype(np.float32))
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
        depth_mm[valid] = np.clip(
            np.rint(depth[valid] * 1000.0), 1, 65535
        ).astype(np.uint16)
        cv2.imwrite(os.path.join(pair_dir, "depth_mm.png"), depth_mm)
        metadata["pairs"].append(
            {
                "index": index,
                "name": name,
                "left_camera": left_camera,
                "right_camera": right_camera,
                "physical_side": physical_side,
                "finite_positive_depth_fraction": round(float(np.mean(valid)), 8),
            }
        )

    with open(os.path.join(output_dir, "capture_metadata.json"), "w") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
