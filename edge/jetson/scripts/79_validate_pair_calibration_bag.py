#!/usr/bin/env python3
"""Validate simultaneous AprilGrid observations in one camera-pair bag."""

import argparse
import json

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

import aslam_cv as acv
from kalibr_camera_calibration.CameraCalibrator import TargetDetector
from kalibr_common.ConfigReader import AslamCamera, CalibrationTargetParameters


def build_detector(target, width, height):
    camera = AslamCamera(
        "pinhole",
        [500.0, 500.0, width / 2.0, height / 2.0],
        "radtan",
        [0.0, 0.0, 0.0, 0.0],
        [width, height],
    )
    return TargetDetector(target, camera.geometry)


def detect(detector, message, bridge):
    color = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    success, observation = detector.detector.findTargetNoTransformation(
        acv.Time(message.header.stamp.to_sec()), gray
    )
    corners = np.asarray(observation.getCornersImageFrame()).reshape(-1, 2)
    result = {
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
        result.update(
            {
                "centroid_x": round(float(corners[:, 0].mean() / width), 3),
                "centroid_y": round(float(corners[:, 1].mean() / height), 3),
                "bbox_area": round(
                    float((x_max - x_min) * (y_max - y_min) / (width * height)),
                    4,
                ),
            }
        )
    return result


def summarize_camera(records, camera):
    camera_records = [record[camera] for record in records]
    detected = [record for record in camera_records if record["success"]]
    centroids = [
        (record["centroid_x"], record["centroid_y"])
        for record in detected
        if "centroid_x" in record
    ]
    return {
        "successful_detections": len(detected),
        "mean_detected_corners": (
            round(float(np.mean([record["corners"] for record in detected])), 1)
            if detected
            else 0.0
        ),
        "centroid_3x3_cells": sorted(
            {
                f"{min(int(x * 3), 2)},{min(int(y * 3), 2)}"
                for x, y in centroids
            }
        ),
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
        "white_clip_percent_max": max(
            record["white_clip_percent"] for record in camera_records
        ),
        "laplacian_variance_range": [
            min(record["laplacian_variance"] for record in camera_records),
            max(record["laplacian_variance"] for record in camera_records),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--samples", type=int, default=45)
    args = parser.parse_args()

    topics = {
        args.left: f"/oak_ffc_4p/{args.left}",
        args.right: f"/oak_ffc_4p/{args.right}",
    }
    stamps_by_camera = {}
    with rosbag.Bag(args.bag, "r") as bag:
        for camera, topic in topics.items():
            stamps_by_camera[camera] = [
                message.header.stamp.to_nsec()
                for _, message, _ in bag.read_messages(topics=[topic])
            ]

    common_stamps = sorted(
        set(stamps_by_camera[args.left]) & set(stamps_by_camera[args.right])
    )
    sample_stamps = {
        common_stamps[int(round(index))]
        for index in np.linspace(
            0, len(common_stamps) - 1, min(args.samples, len(common_stamps))
        )
    }

    bridge = CvBridge()
    target = CalibrationTargetParameters(args.target)
    detectors = {}
    pending = {}
    records = []
    with rosbag.Bag(args.bag, "r") as bag:
        topic_to_camera = {topic: camera for camera, topic in topics.items()}
        for topic, message, _ in bag.read_messages(topics=list(topic_to_camera)):
            stamp = message.header.stamp.to_nsec()
            if stamp not in sample_stamps:
                continue
            camera = topic_to_camera[topic]
            if camera not in detectors:
                detectors[camera] = build_detector(
                    target, message.width, message.height
                )
            detection = detect(detectors[camera], message, bridge)
            frame = pending.setdefault(stamp, {"stamp_ns": stamp})
            frame[camera] = detection
            if args.left in frame and args.right in frame:
                records.append(frame)
                del pending[stamp]

    records.sort(key=lambda record: record["stamp_ns"])
    both_success = sum(
        record[args.left]["success"] and record[args.right]["success"]
        for record in records
    )
    result = {
        "bag": args.bag,
        "pair": [args.left, args.right],
        "topic_frame_counts": {
            camera: len(stamps) for camera, stamps in stamps_by_camera.items()
        },
        "exact_common_timestamp_sets": len(common_stamps),
        "sampled_synchronized_sets": len(records),
        "both_detected_sets": both_success,
        "simultaneous_detection_rate": (
            round(both_success / len(records), 3) if records else 0.0
        ),
        "cameras": {
            args.left: summarize_camera(records, args.left),
            args.right: summarize_camera(records, args.right),
        },
        "samples": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
