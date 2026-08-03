#!/usr/bin/env python3
"""Capture exact-header-matched OmniDepth images, depths, and point cloud."""

import argparse
import html
import importlib.util
import json
import os
import threading
from collections import defaultdict

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_helper(module_name, filename):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(SCRIPT_DIR, filename)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


depth_tools = load_helper("omninxt_depth_capture", "96_capture_stereo_depth_views.py")
cloud_tools = load_helper(
    "omninxt_cloud_capture", "paired_image_colored_cloud_saver.py"
)

PAIRS = depth_tools.PAIRS
KINDS = depth_tools.KINDS
DEBUG_KEYS = tuple(
    (index, kind) for index in range(len(PAIRS)) for kind in KINDS
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--max-disparity", type=float, default=96.0)
    parser.add_argument("--max-cloud-delta-ns", type=int, default=1000)
    return parser.parse_args()


def write_index(path):
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniDepth 同帧深度与点云</title>
<style>
body{margin:0;background:#0b1018;color:#e5edf7;font-family:system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:18px}
.notice{background:#172131;border:1px solid #344155;border-radius:9px;padding:12px}
a{color:#7dd3fc} iframe{width:100%;height:70vh;border:1px solid #344155;border-radius:9px}
img{display:block;width:100%;height:auto;border-radius:9px}
</style></head><body><main>
<h1>OmniDepth 同帧深度与点云</h1>
<div class="notice">
以下四组左右图、视差、深度和拼接图具有完全相同的ROS纳秒时间戳；
点云来自同一输入帧，允许PCL头转换产生最多1微秒的时间戳量化差。
<a href="metadata.json">查看同步元数据</a> ·
<a href="depth_viewer.html">单独打开深度页面</a> ·
<a href="pointcloud_viewer.html">单独打开点云页面</a>
</div>
<h2>交互式点云</h2>
<iframe src="pointcloud_viewer.html"></iframe>
<h2>四组深度总览</h2>
<a href="depth_viewer.html"><img src="depth_overview.png" alt="四组深度总览"></a>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(document)


def main():
    args = parse_args()
    if not 0 < args.min_depth < args.max_depth:
        raise ValueError("Expected 0 < min-depth < max-depth")
    if args.max_disparity <= 0:
        raise ValueError("max-disparity must be positive")

    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    lock = threading.Lock()
    done = threading.Event()
    output_messages = defaultdict(dict)
    assembled_messages = {}
    cloud_messages = {}
    selected = {}
    selected_stamp = None
    selected_cloud_stamp = None
    required_output_keys = set(DEBUG_KEYS)

    def prune_locked():
        stamps = sorted(
            set(output_messages) | set(assembled_messages) | set(cloud_messages)
        )
        for old_stamp in stamps[:-80]:
            output_messages.pop(old_stamp, None)
            assembled_messages.pop(old_stamp, None)
            cloud_messages.pop(old_stamp, None)

    def select_if_complete_locked(stamp):
        nonlocal selected, selected_stamp, selected_cloud_stamp
        if done.is_set() or stamp not in assembled_messages:
            return
        bucket = output_messages.get(stamp, {})
        if not required_output_keys.issubset(bucket):
            return
        if not cloud_messages:
            return
        cloud_stamp = min(
            cloud_messages, key=lambda candidate: abs(candidate - stamp)
        )
        if abs(cloud_stamp - stamp) > args.max_cloud_delta_ns:
            return
        selected = dict(bucket)
        selected["cloud"] = cloud_messages[cloud_stamp]
        selected["assembled"] = assembled_messages[stamp]
        selected_stamp = stamp
        selected_cloud_stamp = cloud_stamp
        done.set()

    def debug_callback(message, key):
        image = depth_tools.image_message_to_array(message)
        stamp = message.header.stamp.to_nsec()
        with lock:
            output_messages[stamp][key] = image
            select_if_complete_locked(stamp)
            prune_locked()

    def cloud_callback(message):
        stamp = message.header.stamp.to_nsec()
        with lock:
            cloud_messages[stamp] = message
            for image_stamp in sorted(output_messages)[-8:]:
                select_if_complete_locked(image_stamp)
            prune_locked()

    def assembled_callback(message):
        stamp = message.header.stamp.to_nsec()
        with lock:
            assembled_messages[stamp] = message
            select_if_complete_locked(stamp)
            prune_locked()

    rospy.init_node(
        "capture_depth_cloud_bundle", anonymous=True, disable_signals=True
    )
    subscribers = []
    for index in range(len(PAIRS)):
        for kind in KINDS:
            subscribers.append(
                rospy.Subscriber(
                    f"/depth_estimation/stereo_{index}/{kind}",
                    Image,
                    debug_callback,
                    callback_args=(index, kind),
                    queue_size=1,
                    buff_size=8 * 1024 * 1024,
                )
            )
    subscribers.append(
        rospy.Subscriber(
            "/depth_estimation/pointcloud_sectors",
            PointCloud2,
            cloud_callback,
            queue_size=2,
            buff_size=32 * 1024 * 1024,
        )
    )
    subscribers.append(
        rospy.Subscriber(
            "/oak_ffc_4p/assemble_image",
            Image,
            assembled_callback,
            queue_size=20,
            buff_size=48 * 1024 * 1024,
        )
    )

    if not done.wait(args.timeout):
        with lock:
            availability = {
                str(stamp): {
                    "assembled": stamp in assembled_messages,
                    "output_keys": sorted(
                        f"{key[0]}:{key[1]}"
                        if isinstance(key, tuple)
                        else str(key)
                        for key in output_messages.get(stamp, {})
                    ),
                }
                for stamp in sorted(
                    set(output_messages) | set(assembled_messages)
                )[-12:]
            }
            cloud_stamps = sorted(cloud_messages)[-12:]
        raise RuntimeError(
            "Timed out waiting for one exact-header-matched bundle: "
            + json.dumps(availability, sort_keys=True)
            + "; recent cloud stamps: "
            + json.dumps(cloud_stamps)
        )

    for subscriber in subscribers:
        subscriber.unregister()

    assembled = cloud_tools.image_to_bgr(selected["assembled"])
    raw_image, preview = cloud_tools.save_images(output_dir, assembled)
    points = cloud_tools.colored_points(selected["cloud"])
    if not points:
        raise RuntimeError("Colored cloud contains no finite points")
    pcd_path = os.path.join(output_dir, "pointcloud_sectors.pcd")
    cloud_tools.write_pcd(pcd_path, points)

    depth_metadata = {
        "stamp_ns": selected_stamp,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "max_disparity_px": args.max_disparity,
        "depth_colormap": (
            "TURBO; near=warm, far=cool, black=invalid/out-of-range"
        ),
        "disparity_colormap": "TURBO; small=cool, large=warm",
        "pairs": [],
    }
    overview_rows = []
    for index, (name, left_camera, right_camera, physical_side) in enumerate(
        PAIRS
    ):
        pair_dir = os.path.join(output_dir, name)
        os.makedirs(pair_dir, exist_ok=True)
        left = selected[(index, "left")]
        right = selected[(index, "right")]
        disparity = selected[(index, "disparity")]
        depth = selected[(index, "depth")]
        if left.shape != (240, 320) or right.shape != left.shape:
            raise RuntimeError(f"Unexpected rectified image size for {name}")
        if disparity.shape != left.shape or depth.shape != left.shape:
            raise RuntimeError(f"Unexpected disparity/depth size for {name}")

        disparity_color = depth_tools.colorize_disparity(
            disparity, args.max_disparity
        )
        depth_color, display_valid = depth_tools.colorize_depth(
            depth, args.min_depth, args.max_depth
        )
        disparity_valid = np.isfinite(disparity) & (disparity > 0)
        disparity_u16 = np.zeros(disparity.shape, dtype=np.uint16)
        disparity_u16[disparity_valid] = np.clip(
            np.rint(disparity[disparity_valid] * 256.0), 1, 65535
        ).astype(np.uint16)
        depth_positive = (
            np.isfinite(depth) & (depth > 0) & (depth < 65.535)
        )
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[depth_positive] = np.clip(
            np.rint(depth[depth_positive] * 1000.0), 1, 65535
        ).astype(np.uint16)

        cv2.imwrite(os.path.join(pair_dir, "left.png"), left)
        cv2.imwrite(os.path.join(pair_dir, "right.png"), right)
        cv2.imwrite(
            os.path.join(pair_dir, "stereo_pair.png"),
            np.hstack((left, right)),
        )
        cv2.imwrite(
            os.path.join(pair_dir, "disparity_x256.png"), disparity_u16
        )
        cv2.imwrite(
            os.path.join(pair_dir, "disparity_color.png"), disparity_color
        )
        cv2.imwrite(os.path.join(pair_dir, "depth_mm.png"), depth_mm)
        cv2.imwrite(os.path.join(pair_dir, "depth_color.png"), depth_color)

        depth_metadata["pairs"].append(
            {
                "name": name,
                "left_camera": left_camera,
                "right_camera": right_camera,
                "physical_side": physical_side,
                "disparity_px": depth_tools.percentile_stats(
                    disparity[disparity_valid]
                ),
                "depth_m_positive": depth_tools.percentile_stats(
                    depth[depth_positive]
                ),
                "display_depth_valid_fraction": round(
                    float(np.mean(display_valid)), 8
                ),
                "finite_positive_disparity_fraction": round(
                    float(np.mean(disparity_valid)), 8
                ),
            }
        )
        overview_rows.append(
            np.hstack(
                (
                    depth_tools.label(
                        cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                        f"{name} LEFT",
                    ),
                    depth_tools.label(
                        cv2.cvtColor(right, cv2.COLOR_GRAY2BGR),
                        f"{name} RIGHT",
                    ),
                    depth_tools.label(
                        disparity_color, "RAW HITNET DISPARITY"
                    ),
                    depth_tools.label(
                        depth_color,
                        f"DEPTH {args.min_depth:.1f}-{args.max_depth:.1f} m",
                    ),
                )
            )
        )

    cv2.imwrite(
        os.path.join(output_dir, "depth_overview.png"),
        np.vstack(overview_rows),
    )
    depth_tools.write_html(
        os.path.join(output_dir, "depth_viewer.html"), depth_metadata
    )

    metadata = {
        "stamp_ns": selected_stamp,
        "image_debug_exact_header_match": True,
        "cloud_stamp_ns": selected_cloud_stamp,
        "cloud_stamp_delta_ns": abs(selected_cloud_stamp - selected_stamp),
        "max_cloud_stamp_delta_ns": args.max_cloud_delta_ns,
        "cloud_stamp_precision": (
            "PCL PointCloud2 header is quantized to microseconds"
        ),
        "matched_topic_count": len(DEBUG_KEYS) + 2,
        "topics": {
            "assembled_image": "/oak_ffc_4p/assemble_image",
            "pointcloud": "/depth_estimation/pointcloud_sectors",
            "debug_images": [
                f"/depth_estimation/stereo_{index}/{kind}"
                for index, kind in DEBUG_KEYS
            ],
        },
        "finite_xyzrgb_points": len(points),
        "image": raw_image,
        "preview": preview,
        "pcd": pcd_path,
        "depth": depth_metadata,
    }
    with open(
        os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    write_index(os.path.join(output_dir, "capture_index.html"))
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
