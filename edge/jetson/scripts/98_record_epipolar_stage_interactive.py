#!/usr/bin/env python3
"""Preview and record one runtime-rectified virtual stereo pair."""

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


class EpipolarRecorder:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.frames = {}
        self.lock = threading.Lock()
        self.recorder = None
        self.recording = False
        self.stop_requested = False
        self.started_at = None
        rospy.Subscriber(
            args.left_topic,
            Image,
            self.image_callback,
            callback_args="LEFT",
            queue_size=1,
            buff_size=2**22,
        )
        rospy.Subscriber(
            args.right_topic,
            Image,
            self.image_callback,
            callback_args="RIGHT",
            queue_size=1,
            buff_size=2**22,
        )

    def image_callback(self, message, side):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="mono8"
            )
        except Exception as error:
            rospy.logwarn_throttle(2.0, f"Image conversion failed: {error}")
            return
        with self.lock:
            self.frames[side] = frame.copy()

    @staticmethod
    def panel(gray, side):
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
        for row in range(40, 240, 40):
            y = row * 2
            cv2.line(image, (0, y), (639, y), (0, 190, 0), 1)
            cv2.putText(
                image,
                f"y={row}",
                (5, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            side,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    def compose(self):
        with self.lock:
            frames = {key: value.copy() for key, value in self.frames.items()}
        panels = []
        for side in ("LEFT", "RIGHT"):
            if side in frames:
                panels.append(self.panel(frames[side], side))
            else:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    waiting,
                    f"Waiting for {side}...",
                    (100, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(waiting)
        preview = cv2.hconcat(panels)
        if self.recording:
            elapsed = time.monotonic() - self.started_at
            remaining = max(0.0, self.args.duration - elapsed)
            text = (
                f"RECORDING {elapsed:4.1f}s | remaining {remaining:4.1f}s"
                " | Ctrl+C to finish early"
            )
            color = (0, 0, 255)
        else:
            text = "PREVIEW | SPACE=start | Q/ESC=cancel"
            color = (0, 255, 255)
        cv2.rectangle(preview, (0, 430), (1280, 480), (0, 0, 0), -1)
        cv2.putText(
            preview,
            text,
            (18, 464),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        return preview

    def images_ready(self):
        with self.lock:
            return "LEFT" in self.frames and "RIGHT" in self.frames

    def start_recording(self):
        if self.recording:
            return
        if not self.images_ready():
            print("Both images are not ready yet.", flush=True)
            return
        os.makedirs(os.path.dirname(self.args.output_base), exist_ok=True)
        command = [
            "rosbag",
            "record",
            "--lz4",
            "-O",
            self.args.output_base,
            self.args.left_topic,
            self.args.right_topic,
        ]
        self.recorder = subprocess.Popen(command, preexec_fn=os.setsid)
        self.recording = True
        self.started_at = time.monotonic()
        print(f"Recording started: {self.args.output_base}.bag", flush=True)

    def stop_recording(self):
        if self.recorder is None:
            return
        if self.recorder.poll() is None:
            print("Finalizing rosbag...", flush=True)
            os.killpg(os.getpgid(self.recorder.pid), signal.SIGINT)
            try:
                self.recorder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.recorder.pid), signal.SIGTERM)
                self.recorder.wait(timeout=5)
        self.recorder = None

    def request_stop(self, *_args):
        self.stop_requested = True

    def run(self):
        window = f"OmniNxt epipolar validation: {self.args.stage}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 540)
        while not self.stop_requested and not rospy.is_shutdown():
            cv2.imshow(window, self.compose())
            key = cv2.waitKey(20) & 0xFF
            if key == ord(" ") and not self.recording:
                self.start_recording()
            elif key in (ord("q"), 27):
                self.stop_requested = True
            if (
                self.recording
                and time.monotonic() - self.started_at >= self.args.duration
            ):
                print(
                    f"Reached {self.args.duration:.0f}s; finalizing.",
                    flush=True,
                )
                self.stop_requested = True
        self.stop_recording()
        cv2.destroyAllWindows()
        bag_path = self.args.output_base + ".bag"
        if self.recording and os.path.isfile(bag_path):
            os.chown(bag_path, 1000, 1000)
            print(f"Bag finalized: {bag_path}", flush=True)
            return 0
        return 3 if not self.recording else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--left-topic", required=True)
    parser.add_argument("--right-topic", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--duration", type=float, default=25.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    rospy.init_node(
        "omninxt_epipolar_recorder", anonymous=True, disable_signals=True
    )
    recorder = EpipolarRecorder(args)
    signal.signal(signal.SIGINT, recorder.request_stop)
    signal.signal(signal.SIGTERM, recorder.request_stop)
    return recorder.run()


if __name__ == "__main__":
    sys.exit(main())
