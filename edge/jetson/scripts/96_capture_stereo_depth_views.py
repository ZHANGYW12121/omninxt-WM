#!/usr/bin/env python3
"""Capture one synchronized four-sector HITNet diagnostic frame."""

import argparse
import html
import json
import os
import threading
from collections import defaultdict

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


PAIRS = (
    ("A_B_RIGHT", "CAM_A", "CAM_B", "机体右侧"),
    ("B_C_REAR", "CAM_B", "CAM_C", "机体后侧"),
    ("C_D_LEFT", "CAM_C", "CAM_D", "机体左侧"),
    ("D_A_FRONT", "CAM_D", "CAM_A", "机体前侧"),
)
KINDS = ("left", "right", "disparity", "depth")


def image_message_to_array(message):
    """Decode the two encodings published by the OmniDepth debug topics."""
    if message.encoding == "mono8":
        dtype = np.dtype(np.uint8)
        channels = 1
    elif message.encoding == "32FC1":
        dtype = np.dtype(np.float32)
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    row_values = message.step // dtype.itemsize
    expected_row_values = message.width * channels
    if row_values < expected_row_values:
        raise ValueError(
            f"Invalid image step {message.step} for {message.encoding} "
            f"{message.width}x{message.height}"
        )
    array = np.frombuffer(message.data, dtype=dtype)
    array = array.reshape(message.height, row_values)
    array = array[:, :expected_row_values]
    return array.reshape(message.height, message.width).astype(
        dtype.newbyteorder("="), copy=True
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--max-disparity", type=float, default=96.0)
    return parser.parse_args()


def percentile_stats(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": round(float(np.min(values)), 6),
        "p01": round(float(np.percentile(values, 1)), 6),
        "p10": round(float(np.percentile(values, 10)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p90": round(float(np.percentile(values, 90)), 6),
        "p99": round(float(np.percentile(values, 99)), 6),
        "max": round(float(np.max(values)), 6),
    }


def colorize_disparity(disparity, max_disparity):
    valid = np.isfinite(disparity) & (disparity > 0)
    scaled = np.zeros(disparity.shape, dtype=np.uint8)
    scaled[valid] = np.clip(
        disparity[valid] / max_disparity * 255.0, 0, 255
    ).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def colorize_depth(depth, min_depth, max_depth):
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    # Near is warm and far is cool, with one fixed scale for all four pairs.
    scaled[valid] = np.clip(
        (max_depth - depth[valid]) / (max_depth - min_depth) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, valid


def label(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 31), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def write_html(path, metadata):
    cards = []
    for index, pair in enumerate(metadata["pairs"]):
        prefix = pair["name"]
        stats = html.escape(json.dumps(pair, ensure_ascii=False, indent=2))
        cards.append(
            f"""
            <section class="pair">
              <h2>#{index} {html.escape(prefix)} — {html.escape(pair["physical_side"])}</h2>
              <div class="grid">
                <figure><img src="{prefix}/left.png"><figcaption>虚拟左图</figcaption></figure>
                <figure><img src="{prefix}/right.png"><figcaption>虚拟右图</figcaption></figure>
                <figure><img src="{prefix}/disparity_color.png"><figcaption>HITNet原始视差（0–{metadata["max_disparity_px"]} px）</figcaption></figure>
                <figure><img src="{prefix}/depth_color.png"><figcaption>米制深度（{metadata["min_depth_m"]}–{metadata["max_depth_m"]} m）</figcaption></figure>
              </div>
              <details><summary>数值统计</summary><pre>{stats}</pre></details>
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>OmniDepth 四组虚拟双目深度诊断</title>
<style>
body{{margin:0;background:#111827;color:#e5e7eb;font-family:system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:20px}}
.notice{{background:#1f2937;border:1px solid #374151;padding:14px;border-radius:8px}}
.pair{{margin:22px 0;padding:16px;background:#182234;border-radius:10px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}
figure{{margin:0}} img{{width:100%;image-rendering:auto;background:#000}}
figcaption{{padding:7px;text-align:center;background:#0f172a}}
pre{{white-space:pre-wrap}} a{{color:#7dd3fc}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>OmniDepth 四组虚拟双目深度诊断</h1>
<div class="notice">
所有深度图使用同一色标：近处为红/黄，远处为蓝，黑色表示非有限或超出显示范围。
这里显示的是未做有效性过滤的HITNet原始结果。原始毫米深度和×256视差PNG可在各组目录下载。
</div>
<p><a href="depth_overview.png">打开四组总览PNG</a></p>
{''.join(cards)}
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


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
    by_stamp = defaultdict(dict)
    selected = {}
    selected_stamp = None

    def receive(message, key):
        nonlocal selected, selected_stamp
        stamp = message.header.stamp.to_nsec()
        image = image_message_to_array(message)
        with lock:
            by_stamp[stamp][key] = image.copy()
            if len(by_stamp[stamp]) == len(PAIRS) * len(KINDS):
                selected = by_stamp[stamp]
                selected_stamp = stamp
                done.set()
            if len(by_stamp) > 8:
                for old_stamp in sorted(by_stamp)[:-4]:
                    del by_stamp[old_stamp]

    rospy.init_node("capture_stereo_depth_views", anonymous=True, disable_signals=True)
    subscribers = []
    for index in range(len(PAIRS)):
        for kind in KINDS:
            subscribers.append(
                rospy.Subscriber(
                    f"/depth_estimation/stereo_{index}/{kind}",
                    Image,
                    receive,
                    callback_args=(index, kind),
                    queue_size=1,
                    buff_size=8 * 1024 * 1024,
                )
            )
    if not done.wait(args.timeout):
        available = {
            str(stamp): sorted(f"{key[0]}:{key[1]}" for key in values)
            for stamp, values in by_stamp.items()
        }
        raise RuntimeError(
            f"Timed out waiting for one synchronized diagnostic frame: {available}"
        )
    for subscriber in subscribers:
        subscriber.unregister()

    metadata = {
        "stamp_ns": selected_stamp,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "max_disparity_px": args.max_disparity,
        "depth_colormap": "TURBO; near=warm, far=cool, black=invalid/out-of-range",
        "disparity_colormap": "TURBO; small=cool, large=warm",
        "pairs": [],
    }
    overview_rows = []
    for index, (name, left_camera, right_camera, physical_side) in enumerate(PAIRS):
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

        disparity_color = colorize_disparity(disparity, args.max_disparity)
        depth_color, display_valid = colorize_depth(
            depth, args.min_depth, args.max_depth
        )
        disparity_u16 = np.zeros(disparity.shape, dtype=np.uint16)
        disparity_finite = np.isfinite(disparity) & (disparity > 0)
        disparity_u16[disparity_finite] = np.clip(
            np.rint(disparity[disparity_finite] * 256.0), 1, 65535
        ).astype(np.uint16)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_positive = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
        depth_mm[depth_positive] = np.clip(
            np.rint(depth[depth_positive] * 1000.0), 1, 65535
        ).astype(np.uint16)

        cv2.imwrite(os.path.join(pair_dir, "left.png"), left)
        cv2.imwrite(os.path.join(pair_dir, "right.png"), right)
        cv2.imwrite(os.path.join(pair_dir, "stereo_pair.png"), np.hstack((left, right)))
        cv2.imwrite(os.path.join(pair_dir, "disparity_x256.png"), disparity_u16)
        cv2.imwrite(os.path.join(pair_dir, "disparity_color.png"), disparity_color)
        cv2.imwrite(os.path.join(pair_dir, "depth_mm.png"), depth_mm)
        cv2.imwrite(os.path.join(pair_dir, "depth_color.png"), depth_color)

        pair_metadata = {
            "name": name,
            "left_camera": left_camera,
            "right_camera": right_camera,
            "physical_side": physical_side,
            "disparity_px": percentile_stats(disparity[disparity_finite]),
            "depth_m_positive": percentile_stats(depth[depth_positive]),
            "display_depth_valid_fraction": round(float(np.mean(display_valid)), 8),
            "finite_positive_disparity_fraction": round(
                float(np.mean(disparity_finite)), 8
            ),
        }
        metadata["pairs"].append(pair_metadata)
        overview_rows.append(
            np.hstack(
                (
                    label(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), f"{name} LEFT"),
                    label(cv2.cvtColor(right, cv2.COLOR_GRAY2BGR), f"{name} RIGHT"),
                    label(disparity_color, "RAW HITNET DISPARITY"),
                    label(depth_color, f"DEPTH {args.min_depth:.1f}-{args.max_depth:.1f} m"),
                )
            )
        )

    cv2.imwrite(os.path.join(output_dir, "depth_overview.png"), np.vstack(overview_rows))
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    write_html(os.path.join(output_dir, "depth_viewer.html"), metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
