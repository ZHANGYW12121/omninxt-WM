#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


STAGE_CAMERAS = {
    "CAM_A": ("CAM_A",),
    "CAM_B": ("CAM_B",),
    "CAM_C": ("CAM_C",),
    "CAM_D": ("CAM_D",),
    "CAM_D-CAM_A": ("CAM_D", "CAM_A"),
    "CAM_A-CAM_B": ("CAM_A", "CAM_B"),
    "CAM_B-CAM_C": ("CAM_B", "CAM_C"),
    "CAM_C-CAM_D": ("CAM_C", "CAM_D"),
}

ALL_CAMERA_TOPICS = [
    "/oak_ffc_4p/CAM_A",
    "/oak_ffc_4p/CAM_B",
    "/oak_ffc_4p/CAM_C",
    "/oak_ffc_4p/CAM_D",
]

IMU_TOPICS = [
    "/mavros/imu/data_raw",
    "/mavros/time_reference",
]


class StageRecorder:
    def __init__(self, stage, output_base, duration):
        self.stage = stage
        self.output_base = output_base
        self.duration = duration
        self.cameras = STAGE_CAMERAS[stage]
        self.bridge = CvBridge()
        self.frames = {}
        self.lock = threading.Lock()
        self.recorder = None
        self.recording = False
        self.stop_requested = False
        self.started_at = None

        for camera in self.cameras:
            rospy.Subscriber(
                f"/oak_ffc_4p/{camera}",
                Image,
                self.image_callback,
                callback_args=camera,
                queue_size=1,
                buff_size=2**24,
            )

    def image_callback(self, message, camera):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:
            rospy.logwarn_throttle(2.0, f"Image conversion failed: {error}")
            return
        with self.lock:
            self.frames[camera] = frame

    @staticmethod
    def fit(frame, width=800, height=600):
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(
            frame,
            (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - resized.shape[1]) // 2
        y = (height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return canvas

    def compose_preview(self):
        with self.lock:
            available = {key: value.copy() for key, value in self.frames.items()}

        panels = []
        for camera in self.cameras:
            if camera in available:
                panel = self.fit(
                    available[camera],
                    width=800 if len(self.cameras) == 1 else 640,
                    height=600 if len(self.cameras) == 1 else 480,
                )
            else:
                panel = np.zeros(
                    (600 if len(self.cameras) == 1 else 480,
                     800 if len(self.cameras) == 1 else 640, 3),
                    dtype=np.uint8,
                )
                cv2.putText(
                    panel,
                    "Waiting for image...",
                    (40, panel.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                panel,
                camera,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)

        preview = panels[0] if len(panels) == 1 else cv2.hconcat(panels)
        if self.recording:
            elapsed = time.monotonic() - self.started_at
            remaining = max(0.0, self.duration - elapsed)
            status = (
                f"RECORDING {elapsed:5.1f}s | remaining {remaining:5.1f}s"
                " | Ctrl+C to finish early"
            )
            color = (0, 0, 255)
        else:
            status = "PREVIEW | Press SPACE to start | Q/ESC to abort"
            color = (0, 255, 255)
        cv2.rectangle(
            preview,
            (0, preview.shape[0] - 54),
            (preview.shape[1], preview.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            preview,
            status,
            (18, preview.shape[0] - 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        return preview

    def all_preview_images_ready(self):
        with self.lock:
            return all(camera in self.frames for camera in self.cameras)

    def start_recording(self):
        if self.recording:
            return
        if not self.all_preview_images_ready():
            print("Images are not ready; wait for the preview, then press SPACE.", flush=True)
            return
        os.makedirs(os.path.dirname(self.output_base), exist_ok=True)
        if len(self.cameras) == 1:
            camera_topics = [f"/oak_ffc_4p/{self.cameras[0]}"]
        else:
            # Pair stages retain all four cameras because the current preparation
            # workflow also builds the synchronized 5120-pixel assembled bag.
            camera_topics = ALL_CAMERA_TOPICS
        record_topics = [*camera_topics, *IMU_TOPICS]
        command = ["rosbag", "record", "--lz4", "-O", self.output_base, *record_topics]
        self.recorder = subprocess.Popen(command, preexec_fn=os.setsid)
        self.recording = True
        self.started_at = time.monotonic()
        print("", flush=True)
        print(f"Recording started: {self.output_base}.bag", flush=True)
        print("Topics: " + ", ".join(record_topics), flush=True)
        print("Return to this terminal and press Ctrl+C when the stage is complete.", flush=True)

    def stop_recording(self):
        if self.recorder is None:
            return
        if self.recorder.poll() is None:
            print("\nStopping rosbag and writing the index...", flush=True)
            os.killpg(os.getpgid(self.recorder.pid), signal.SIGINT)
            try:
                self.recorder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print(
                    "rosbag did not finish indexing within 120 seconds; terminating.",
                    file=sys.stderr,
                    flush=True,
                )
                os.killpg(os.getpgid(self.recorder.pid), signal.SIGTERM)
                self.recorder.wait(timeout=5)
        self.recorder = None

    def request_stop(self, _signum=None, _frame=None):
        self.stop_requested = True

    def run(self):
        window = f"OmniNxt formal calibration: {self.stage}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960 if len(self.cameras) == 1 else 1280, 720)

        while not self.stop_requested and not rospy.is_shutdown():
            cv2.imshow(window, self.compose_preview())
            key = cv2.waitKey(20) & 0xFF
            if key == ord(" ") and not self.recording:
                self.start_recording()
            elif key in (ord("q"), 27):
                if self.recording:
                    print("Stop requested from the preview window.", flush=True)
                self.stop_requested = True
            if (
                self.recording
                and time.monotonic() - self.started_at >= self.duration
            ):
                print(f"\nReached {self.duration:.0f} seconds; finalizing the bag.", flush=True)
                self.stop_requested = True

        self.stop_recording()
        cv2.destroyAllWindows()

        bag_path = self.output_base + ".bag"
        if self.recording and os.path.isfile(bag_path) and os.path.getsize(bag_path) > 0:
            print(f"Bag finalized: {bag_path}", flush=True)
            return 0
        if not self.recording:
            print("Recording was cancelled before SPACE was pressed.", flush=True)
            return 3
        print(f"Recording did not produce a valid bag: {bag_path}", file=sys.stderr)
        return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGE_CAMERAS)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--duration", type=float, default=75.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("omninxt_formal_stage_recorder", anonymous=True, disable_signals=True)
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    recorder = StageRecorder(args.stage, args.output_base, args.duration)
    signal.signal(signal.SIGINT, recorder.request_stop)
    signal.signal(signal.SIGTERM, recorder.request_stop)
    return recorder.run()


if __name__ == "__main__":
    sys.exit(main())
