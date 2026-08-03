#!/usr/bin/env python3
"""Capture a robust flat-field or dark-field median from one OAK camera."""

import argparse
import json
import os
import threading
import time

import cv2
import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, choices=("CAM_A", "CAM_B", "CAM_C", "CAM_D"))
    parser.add_argument("--kind", required=True, choices=("flat", "dark"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def percentiles(image):
    return {
        f"p{value:02d}": round(float(np.percentile(image, value)), 3)
        for value in (1, 5, 10, 50, 90, 95, 99)
    }


def main():
    args = parse_args()
    if args.frames < 30 or args.frames > 500:
        raise ValueError("--frames must be between 30 and 500")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")

    bridge = cv_bridge.CvBridge()
    lock = threading.Lock()
    done = threading.Event()
    frames = []
    received = 0
    first_stamp_ns = None
    last_stamp_ns = None

    def receive(message):
        nonlocal received, first_stamp_ns, last_stamp_ns
        image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with lock:
            received += 1
            if received <= args.warmup:
                return
            if len(frames) >= args.frames:
                return
            stamp_ns = message.header.stamp.to_nsec()
            if first_stamp_ns is None:
                first_stamp_ns = stamp_ns
            last_stamp_ns = stamp_ns
            frames.append(gray.copy())
            if len(frames) == args.frames:
                done.set()

    rospy.init_node(
        f"capture_photometric_{args.camera.lower()}_{args.kind}",
        anonymous=True,
        disable_signals=True,
    )
    subscriber = rospy.Subscriber(
        f"/oak_ffc_4p/{args.camera}",
        Image,
        receive,
        queue_size=1,
        buff_size=8 * 1024 * 1024,
    )

    timeout = max(30.0, (args.frames + args.warmup) / 10.0)
    if not done.wait(timeout):
        subscriber.unregister()
        raise RuntimeError(
            f"Timed out after {timeout:.1f}s; captured {len(frames)}/{args.frames} frames"
        )
    subscriber.unregister()

    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    sample = stack[len(stack) // 2]
    temporal_std = np.std(stack.astype(np.float32), axis=0)
    height, width = median.shape
    center = median[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]

    os.makedirs(args.output_dir, exist_ok=True)
    median_path = os.path.join(args.output_dir, f"{args.kind}_median.png")
    sample_path = os.path.join(args.output_dir, f"{args.kind}_sample.png")
    std_path = os.path.join(args.output_dir, f"{args.kind}_temporal_std_x16.png")
    metadata_path = os.path.join(args.output_dir, f"{args.kind}_metadata.json")
    cv2.imwrite(median_path, median)
    cv2.imwrite(sample_path, sample)
    cv2.imwrite(
        std_path,
        np.clip(np.rint(temporal_std * 16.0), 0, 65535).astype(np.uint16),
    )

    duration = (
        (last_stamp_ns - first_stamp_ns) / 1e9
        if first_stamp_ns is not None and last_stamp_ns is not None
        else 0.0
    )
    metadata = {
        "camera": args.camera,
        "kind": args.kind,
        "frames": len(frames),
        "warmup_frames": args.warmup,
        "resolution": [width, height],
        "duration_s": round(duration, 6),
        "effective_hz": round((len(frames) - 1) / duration, 6) if duration > 0 else None,
        "full_percentiles": percentiles(median),
        "center_percentiles": percentiles(center),
        "full_saturated_fraction": round(float(np.mean(median >= 250)), 8),
        "center_saturated_fraction": round(float(np.mean(center >= 250)), 8),
        "full_dark_fraction": round(float(np.mean(median <= 5)), 8),
        "center_dark_fraction": round(float(np.mean(center <= 5)), 8),
        "temporal_std_mean": round(float(np.mean(temporal_std)), 6),
        "temporal_std_p95": round(float(np.percentile(temporal_std, 95)), 6),
        "created_unix_s": time.time(),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.kind == "flat":
        center_p50 = metadata["center_percentiles"]["p50"]
        center_sat = metadata["center_saturated_fraction"]
        if center_p50 < 80:
            print("WARNING: flat field is too dark; increase illumination or exposure.")
        if center_p50 > 230 or center_sat > 0.01:
            print("WARNING: flat field is too bright/saturated; reduce exposure.")
    else:
        if metadata["center_percentiles"]["p99"] > 20:
            print("WARNING: dark frame is too bright; make the lens cover fully opaque.")


if __name__ == "__main__":
    main()
