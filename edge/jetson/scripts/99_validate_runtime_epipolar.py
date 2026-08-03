#!/usr/bin/env python3
"""Validate runtime virtual-stereo epipolar alignment with AprilGrid IDs."""

import argparse
import html
import json
import os

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

import aslam_cv as acv
from kalibr_camera_calibration.CameraCalibrator import TargetDetector
from kalibr_common.ConfigReader import AslamCamera, CalibrationTargetParameters


PAIRS = (
    ("AB_RIGHT", 0, "CAM_A", "CAM_B", "机体右侧"),
    ("BC_REAR", 1, "CAM_B", "CAM_C", "机体后侧"),
    ("CD_LEFT", 2, "CAM_C", "CAM_D", "机体左侧"),
    ("DA_FRONT", 3, "CAM_D", "CAM_A", "机体前侧"),
)


def build_detector(target):
    camera = AslamCamera(
        "pinhole",
        [133.0, 133.0, 160.0, 120.0],
        "radtan",
        [0.0, 0.0, 0.0, 0.0],
        [320, 240],
    )
    return TargetDetector(target, camera.geometry)


def decode(message, bridge):
    return bridge.imgmsg_to_cv2(message, desired_encoding="mono8")


def detect(detector, message, bridge):
    gray = decode(message, bridge)
    success, observation = detector.detector.findTargetNoTransformation(
        acv.Time(message.header.stamp.to_sec()), gray
    )
    corners = np.asarray(
        observation.getCornersImageFrame(), dtype=np.float64
    ).reshape(-1, 2)
    indices = np.asarray(observation.getCornersIdx()).reshape(-1)
    if len(indices) != len(corners):
        raise RuntimeError("AprilGrid corner ID count does not match corners")
    return gray, bool(success), {
        int(index): corners[position]
        for position, index in enumerate(indices)
    }


def percentile(values, q):
    return round(float(np.percentile(values, q)), 6) if values else None


def summarize(values):
    absolute = [abs(value) for value in values]
    return {
        "count": len(values),
        "signed_mean_px": (
            round(float(np.mean(values)), 6) if values else None
        ),
        "signed_median_px": percentile(values, 50),
        "absolute_median_px": percentile(absolute, 50),
        "absolute_p90_px": percentile(absolute, 90),
        "absolute_p95_px": percentile(absolute, 95),
        "absolute_max_px": round(float(max(absolute)), 6) if absolute else None,
    }


def classify(metrics, both_frames):
    if both_frames < 15 or metrics["count"] < 200:
        return "INSUFFICIENT_DATA"
    if (
        metrics["absolute_median_px"] < 0.5
        and metrics["absolute_p90_px"] < 1.0
    ):
        return "PASS"
    if (
        metrics["absolute_median_px"] < 0.8
        and metrics["absolute_p90_px"] < 1.5
    ):
        return "BORDERLINE"
    return "FAIL"


def draw_evidence(path, left, right, matches, title):
    left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack((left_bgr, right_bgr))
    for row in range(20, 240, 20):
        cv2.line(canvas, (0, row), (639, row), (0, 110, 0), 1)
    for corner_id, point_left, point_right, dy in matches:
        p0 = tuple(int(round(value)) for value in point_left)
        p1 = (
            int(round(point_right[0])) + 320,
            int(round(point_right[1])),
        )
        color = (0, 255, 0) if abs(dy) < 1.0 else (0, 0, 255)
        cv2.circle(canvas, p0, 2, color, -1)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.line(canvas, p0, p1, color, 1)
    cv2.rectangle(canvas, (0, 0), (639, 27), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(path, canvas)


def analyze_pair(bag_path, left_topic, right_topic, target, samples, output):
    stamps = {left_topic: [], right_topic: []}
    with rosbag.Bag(bag_path) as bag:
        for topic, message, _ in bag.read_messages(topics=list(stamps)):
            stamps[topic].append(message.header.stamp.to_nsec())
    common = sorted(set(stamps[left_topic]) & set(stamps[right_topic]))
    if not common:
        raise RuntimeError(f"No exact synchronized timestamps in {bag_path}")
    sample_count = min(samples, len(common))
    positions = np.linspace(0, len(common) - 1, sample_count)
    selected = {common[int(round(position))] for position in positions}

    bridge = CvBridge()
    detectors = {
        "left": build_detector(target),
        "right": build_detector(target),
    }
    pending = {}
    all_dy = []
    regions = {f"{row},{col}": [] for row in range(3) for col in range(3)}
    detected_left = 0
    detected_right = 0
    both_detected = 0
    best = None
    processed = set()
    topic_side = {left_topic: "left", right_topic: "right"}

    with rosbag.Bag(bag_path) as bag:
        for topic, message, _ in bag.read_messages(topics=list(topic_side)):
            stamp = message.header.stamp.to_nsec()
            if stamp not in selected or stamp in processed:
                continue
            side = topic_side[topic]
            gray, success, corners = detect(
                detectors[side], message, bridge
            )
            frame = pending.setdefault(stamp, {})
            frame[side] = (gray, success, corners)
            if len(frame) != 2:
                continue
            left, left_ok, left_corners = frame["left"]
            right, right_ok, right_corners = frame["right"]
            detected_left += int(left_ok)
            detected_right += int(right_ok)
            matches = []
            if left_ok and right_ok:
                common_ids = sorted(set(left_corners) & set(right_corners))
                for corner_id in common_ids:
                    point_left = left_corners[corner_id]
                    point_right = right_corners[corner_id]
                    dy = float(point_left[1] - point_right[1])
                    matches.append(
                        (corner_id, point_left, point_right, dy)
                    )
                    all_dy.append(dy)
                    x = float(
                        (point_left[0] + point_right[0]) / (2.0 * 320.0)
                    )
                    y = float(
                        (point_left[1] + point_right[1]) / (2.0 * 240.0)
                    )
                    col = min(2, max(0, int(x * 3)))
                    row = min(2, max(0, int(y * 3)))
                    regions[f"{row},{col}"].append(dy)
                if matches:
                    both_detected += 1
            if best is None or len(matches) > len(best[2]):
                best = (left, right, matches, stamp)
            del pending[stamp]
            processed.add(stamp)

    metrics = summarize(all_dy)
    status = classify(metrics, both_detected)
    evidence = os.path.join(output, os.path.basename(bag_path) + ".png")
    stem = os.path.splitext(os.path.basename(bag_path))[0]
    reference_left = os.path.join(output, stem + "_reference_left.png")
    reference_right = os.path.join(output, stem + "_reference_right.png")
    reference_corners = os.path.join(
        output, stem + "_reference_corners.json"
    )
    if best is not None:
        draw_evidence(
            evidence,
            best[0],
            best[1],
            best[2],
            (
                f"{os.path.basename(bag_path)} | common corners "
                f"{len(best[2])} | green |dy|<1 px"
            ),
        )
        cv2.imwrite(reference_left, best[0])
        cv2.imwrite(reference_right, best[1])
        with open(reference_corners, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "stamp_ns": best[3],
                    "matches": [
                        {
                            "corner_id": int(corner_id),
                            "left": [float(value) for value in point_left],
                            "right": [float(value) for value in point_right],
                            "reference_disparity_px": float(
                                point_left[0] - point_right[0]
                            ),
                            "vertical_error_px": float(dy),
                        }
                        for corner_id, point_left, point_right, dy in best[2]
                    ],
                },
                stream,
                indent=2,
            )
            stream.write("\n")
    return {
        "bag": bag_path,
        "topics": [left_topic, right_topic],
        "messages": {
            "left": len(stamps[left_topic]),
            "right": len(stamps[right_topic]),
            "exact_common_timestamps": len(common),
        },
        "sampled_synchronized_frames": sample_count,
        "left_detected_frames": detected_left,
        "right_detected_frames": detected_right,
        "both_detected_with_common_ids": both_detected,
        "simultaneous_detection_rate": round(
            both_detected / sample_count, 6
        ),
        "vertical_error": metrics,
        "regions_3x3": {
            region: summarize(values)
            for region, values in regions.items()
            if values
        },
        "status": status,
        "evidence": os.path.basename(evidence),
        "reference_left": os.path.basename(reference_left),
        "reference_right": os.path.basename(reference_right),
        "reference_corners": os.path.basename(reference_corners),
    }


def write_html(path, results):
    rows = []
    for result in results:
        metrics = result["vertical_error"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['stage'])}</td>"
            f"<td>{html.escape(result['physical_side'])}</td>"
            f"<td>{result['both_detected_with_common_ids']}/"
            f"{result['sampled_synchronized_frames']}</td>"
            f"<td>{metrics['count']}</td>"
            f"<td>{metrics['absolute_median_px']}</td>"
            f"<td>{metrics['absolute_p90_px']}</td>"
            f"<td>{html.escape(result['status'])}</td>"
            "</tr>"
        )
    cards = "".join(
        f"<h2>{html.escape(item['stage'])}</h2>"
        f"<img src='{html.escape(item['evidence'])}'>"
        f"<pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre>"
        for item in results
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>OmniNxt运行态极线验证</title>
<style>
body{{background:#101827;color:#e5e7eb;font-family:system-ui;margin:20px}}
table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #475569}}
img{{max-width:100%;image-rendering:auto}}pre{{white-space:pre-wrap;background:#172033;padding:12px}}
</style></head><body><h1>运行态AprilGrid极线验证</h1>
<table><tr><th>双目</th><th>方向</th><th>共同检测帧</th><th>共同角点</th>
<th>|dy|中位数</th><th>|dy| P90</th><th>状态</th></tr>
{''.join(rows)}</table>{cards}</body></html>"""
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    run_dir = os.path.realpath(args.run_dir)
    output = os.path.join(run_dir, "analysis")
    os.makedirs(output, exist_ok=True)
    target = CalibrationTargetParameters(args.target)
    results = []
    for stage, stereo_id, left_camera, right_camera, physical_side in PAIRS:
        result = analyze_pair(
            os.path.join(run_dir, "raw", stage + ".bag"),
            f"/depth_estimation/stereo_{stereo_id}/left",
            f"/depth_estimation/stereo_{stereo_id}/right",
            target,
            args.samples,
            output,
        )
        result.update(
            {
                "stage": stage,
                "stereo_id": stereo_id,
                "left_camera": left_camera,
                "right_camera": right_camera,
                "physical_side": physical_side,
            }
        )
        results.append(result)
        print(
            f"{stage}: {result['status']}, "
            f"both={result['both_detected_with_common_ids']}/"
            f"{result['sampled_synchronized_frames']}, "
            f"corners={result['vertical_error']['count']}, "
            f"median={result['vertical_error']['absolute_median_px']} px, "
            f"p90={result['vertical_error']['absolute_p90_px']} px",
            flush=True,
        )
    report = {
        "criterion": {
            "PASS": ">=15 frames, >=200 corners, median <0.5 px and P90 <1.0 px",
            "BORDERLINE": ">=15 frames, >=200 corners, median <0.8 px and P90 <1.5 px",
        },
        "results": results,
        "overall_status": (
            "PASS"
            if all(item["status"] == "PASS" for item in results)
            else "REVIEW"
        ),
    }
    report_path = os.path.join(output, "epipolar_validation.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_html(os.path.join(output, "epipolar_validation.html"), results)
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
