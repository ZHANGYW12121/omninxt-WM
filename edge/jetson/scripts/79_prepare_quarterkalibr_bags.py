#!/usr/bin/env python3
"""Prepare explicitly staged bags for the local quarterKalibr workflow."""

import argparse
import json
import os
import shutil

import cv2
import rosbag
from cv_bridge import CvBridge


STAGES = {
    "CAM_A": ("CAM_A",),
    "CAM_B": ("CAM_B",),
    "CAM_C": ("CAM_C",),
    "CAM_D": ("CAM_D",),
    "CAM_D-CAM_A": ("CAM_D", "CAM_A"),
    "CAM_A-CAM_B": ("CAM_A", "CAM_B"),
    "CAM_B-CAM_C": ("CAM_B", "CAM_C"),
    "CAM_C-CAM_D": ("CAM_C", "CAM_D"),
}
PAIR_STAGES = tuple(list(STAGES)[4:])
CAMERAS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")
SOURCE_PREFIX = "/oak_ffc_4p/"
IMU_TOPIC = "/mavros/imu/data_raw"
TIME_TOPIC = "/mavros/time_reference"


def stamp_key(message):
    return message.header.stamp.to_nsec()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.realpath(args.run_dir)
    raw_dir = os.path.join(run_dir, "raw")
    prepared_dir = os.path.join(run_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)

    for filename in ("april_6x6.yaml", "imu_BMI088_PROVISIONAL.yaml"):
        source = os.path.join(run_dir, filename)
        if not os.path.isfile(source):
            raise RuntimeError(f"Missing run input: {source}")
    shutil.copy2(
        os.path.join(run_dir, "april_6x6.yaml"),
        os.path.join(prepared_dir, "april_6x6.yaml"),
    )
    shutil.copy2(
        os.path.join(run_dir, "imu_BMI088_PROVISIONAL.yaml"),
        os.path.join(prepared_dir, "imu.yaml"),
    )

    bridge = CvBridge()
    summary = {"stages": {}}
    stereo_path = os.path.join(prepared_dir, "stereo_depth_calibration.bag")
    with rosbag.Bag(stereo_path, "w", compression=rosbag.Compression.LZ4) as stereo:
        for stage, selected_cameras in STAGES.items():
            source_path = os.path.join(raw_dir, f"{stage}.bag")
            if not os.path.isfile(source_path):
                raise RuntimeError(f"Missing formal stage bag: {source_path}")
            output_path = os.path.join(prepared_dir, f"{stage}.bag")
            counts = {camera: 0 for camera in selected_cameras}
            counts["imu"] = 0
            assembled_count = 0
            frame_sets = {}

            with rosbag.Bag(source_path, "r") as source, rosbag.Bag(
                output_path, "w", compression=rosbag.Compression.LZ4
            ) as output:
                for topic, message, recorded_time in source.read_messages():
                    if topic == IMU_TOPIC:
                        output.write(IMU_TOPIC, message, recorded_time)
                        counts["imu"] += 1
                        continue
                    if topic == TIME_TOPIC:
                        output.write(TIME_TOPIC, message, recorded_time)
                        continue
                    if not topic.startswith(SOURCE_PREFIX):
                        continue
                    camera = topic[len(SOURCE_PREFIX) :]
                    if camera not in CAMERAS:
                        continue

                    if camera in selected_cameras:
                        output.write(f"/{camera}", message, recorded_time)
                        counts[camera] += 1

                    if stage in PAIR_STAGES:
                        key = stamp_key(message)
                        frame_set = frame_sets.setdefault(key, {})
                        frame_set[camera] = (message, recorded_time)
                        if len(frame_set) == len(CAMERAS):
                            images = [
                                bridge.imgmsg_to_cv2(
                                    frame_set[name][0], desired_encoding="bgr8"
                                )
                                for name in CAMERAS
                            ]
                            assembled = bridge.cv2_to_imgmsg(
                                cv2.hconcat(images), encoding="bgr8"
                            )
                            assembled.header = frame_set["CAM_A"][0].header
                            stereo.write(
                                "/oak_ffc_4p/assemble_image",
                                assembled,
                                frame_set["CAM_A"][1],
                            )
                            assembled_count += 1
                            del frame_sets[key]

            if counts["imu"] == 0 or any(counts[camera] == 0 for camera in selected_cameras):
                raise RuntimeError(f"Empty required topic in {stage}: {counts}")
            summary["stages"][stage] = {
                "selected_cameras": list(selected_cameras),
                "counts": counts,
                "assembled_frames_added": assembled_count,
                "prepared_bag": output_path,
            }

    summary["stereo_depth_calibration_bag"] = stereo_path
    summary_path = os.path.join(prepared_dir, "preparation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

