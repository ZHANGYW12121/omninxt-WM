#!/usr/bin/env python3
"""Sample a mono calibration bag with the same AprilGrid detector as Kalibr."""

import argparse
import json

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

import aslam_cv as acv
from kalibr_camera_calibration.CameraCalibrator import TargetDetector
from kalibr_common.ConfigReader import AslamCamera, CalibrationTargetParameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--topic",
        help="Image topic override (default: /oak_ffc_4p/<camera>)",
    )
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    topic = args.topic or f"/oak_ffc_4p/{args.camera}"
    bridge = CvBridge()
    target = CalibrationTargetParameters(args.target)
    target_params = target.getTargetParams()
    expected = int(target_params["tagRows"]) * int(target_params["tagCols"]) * 4

    with rosbag.Bag(args.bag, "r") as bag:
        count = bag.get_message_count(topic_filters=[topic])
        if count == 0:
            raise RuntimeError(f"No messages on {topic}")
        sample_indices = set(
            int(round(value))
            for value in np.linspace(0, count - 1, min(args.samples, count))
        )
        records = []
        detector = None
        for index, (_, message, _) in enumerate(bag.read_messages(topics=[topic])):
            if index not in sample_indices:
                continue
            color = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            if detector is None:
                camera = AslamCamera(
                    "pinhole",
                    [500.0, 500.0, width / 2.0, height / 2.0],
                    "radtan",
                    [0.0, 0.0, 0.0, 0.0],
                    [width, height],
                )
                detector = TargetDetector(target, camera.geometry)
            success, observation = detector.detector.findTargetNoTransformation(
                acv.Time(message.header.stamp.to_sec()), gray
            )
            corners = np.asarray(observation.getCornersImageFrame()).reshape(-1, 2)
            record = {
                "frame_index": index,
                "success": bool(success),
                "corners": int(len(corners)),
                "gray_mean": round(float(gray.mean()), 2),
                "white_clip_percent": round(
                    float(np.count_nonzero(gray >= 250)) * 100.0 / gray.size, 2
                ),
                "laplacian_variance": round(
                    float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2
                ),
            }
            if len(corners):
                x_min, y_min = corners.min(axis=0)
                x_max, y_max = corners.max(axis=0)
                record.update(
                    {
                        "centroid_x": round(float(corners[:, 0].mean() / width), 3),
                        "centroid_y": round(float(corners[:, 1].mean() / height), 3),
                        "bbox_area": round(
                            float((x_max - x_min) * (y_max - y_min) / (width * height)),
                            4,
                        ),
                    }
                )
            records.append(record)

    detected = [record for record in records if record["success"]]
    centroids = [
        (record["centroid_x"], record["centroid_y"])
        for record in detected
        if "centroid_x" in record
    ]
    occupied_cells = sorted(
        {
            f"{min(int(x * 3), 2)},{min(int(y * 3), 2)}"
            for x, y in centroids
        }
    )
    result = {
        "bag": args.bag,
        "camera": args.camera,
        "topic_frames": count,
        "sampled_frames": len(records),
        "successful_detections": len(detected),
        "detection_rate": round(len(detected) / len(records), 3),
        "mean_detected_corners": (
            round(float(np.mean([record["corners"] for record in detected])), 1)
            if detected
            else 0.0
        ),
        "centroid_3x3_cells": occupied_cells,
        "centroid_x_range": (
            [min(x for x, _ in centroids), max(x for x, _ in centroids)]
            if centroids
            else []
        ),
        "centroid_y_range": (
            [min(y for _, y in centroids), max(y for _, y in centroids)]
            if centroids
            else []
        ),
        "board_bbox_area_range": (
            [
                min(record["bbox_area"] for record in detected),
                max(record["bbox_area"] for record in detected),
            ]
            if detected
            else []
        ),
        "gray_mean_range": [
            min(record["gray_mean"] for record in records),
            max(record["gray_mean"] for record in records),
        ],
        "white_clip_percent_max": max(
            record["white_clip_percent"] for record in records
        ),
        "laplacian_variance_range": [
            min(record["laplacian_variance"] for record in records),
            max(record["laplacian_variance"] for record in records),
        ],
        "samples": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
