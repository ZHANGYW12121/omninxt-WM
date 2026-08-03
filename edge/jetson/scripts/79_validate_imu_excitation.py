#!/usr/bin/env python3
"""Summarize IMU timing and motion excitation in a calibration bag."""

import argparse
import json

import numpy as np
import rosbag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/mavros/imu/data_raw")
    args = parser.parse_args()

    stamps = []
    gyro = []
    accel = []
    with rosbag.Bag(args.bag, "r") as bag:
        all_topics = sorted(bag.get_type_and_topic_info().topics)
        for _, message, _ in bag.read_messages(topics=[args.topic]):
            stamps.append(message.header.stamp.to_sec())
            gyro.append(
                [
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ]
            )
            accel.append(
                [
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ]
            )

    stamps = np.asarray(stamps)
    gyro = np.asarray(gyro)
    accel = np.asarray(accel)
    if len(stamps) < 2:
        raise RuntimeError("Not enough IMU samples")
    deltas = np.diff(stamps)
    positive = deltas[deltas > 0]
    gyro_norm = np.linalg.norm(gyro, axis=1)

    result = {
        "bag": args.bag,
        "recorded_topics": all_topics,
        "samples": int(len(stamps)),
        "header_duration_seconds": round(float(stamps[-1] - stamps[0]), 3),
        "header_rate_hz": round(
            float((len(stamps) - 1) / (stamps[-1] - stamps[0])), 3
        ),
        "non_monotonic_deltas": int(np.count_nonzero(deltas <= 0)),
        "delta_ms": {
            "median": round(float(np.median(positive) * 1000.0), 3),
            "p95": round(float(np.percentile(positive, 95) * 1000.0), 3),
            "max": round(float(positive.max() * 1000.0), 3),
        },
        "gyro_axis_min_rad_s": np.round(gyro.min(axis=0), 3).tolist(),
        "gyro_axis_max_rad_s": np.round(gyro.max(axis=0), 3).tolist(),
        "gyro_axis_std_rad_s": np.round(gyro.std(axis=0), 3).tolist(),
        "gyro_norm_rad_s": {
            "p95": round(float(np.percentile(gyro_norm, 95)), 3),
            "max": round(float(gyro_norm.max()), 3),
        },
        "accel_axis_min_m_s2": np.round(accel.min(axis=0), 3).tolist(),
        "accel_axis_max_m_s2": np.round(accel.max(axis=0), 3).tolist(),
        "accel_axis_std_m_s2": np.round(accel.std(axis=0), 3).tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
