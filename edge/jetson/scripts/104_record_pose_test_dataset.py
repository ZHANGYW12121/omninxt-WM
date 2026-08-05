#!/usr/bin/env python3
"""Interactive 30 s raw OAK recording for offline pose regression tests."""

import argparse
import json
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


TOPIC = "/oak_ffc_4p/assemble_image"
CAMERA_NAMES = (
    "CAM_A / FRONT_RIGHT",
    "CAM_B / REAR_RIGHT",
    "CAM_C / REAR_LEFT",
    "CAM_D / FRONT_LEFT",
)


class PoseTestRecorder:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_stamp_ns = None
        self.last_video_stamp_ns = None
        self.received_frames = 0
        self.written_frames = 0
        self.stop_requested = False
        self.countdown_started = None
        self.recording_started = None
        self.recording_start_ros_ns = None
        self.recording_end_ros_ns = None
        self.recorder = None
        self.video = None
        rospy.Subscriber(
            TOPIC, Image, self.image_callback,
            queue_size=1, buff_size=2 ** 26)

    def image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="bgr8")
        except Exception as error:
            rospy.logwarn_throttle(2.0, "Image conversion failed: %s", error)
            return
        with self.lock:
            self.frame = frame.copy()
            self.frame_stamp_ns = message.header.stamp.to_nsec()
            self.received_frames += 1

    @staticmethod
    def _empty_panel(text):
        panel = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(panel, text, (55, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 190, 255), 2, cv2.LINE_AA)
        return panel

    def compose(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            stamp_ns = self.frame_stamp_ns
        panels = []
        if frame is not None and frame.ndim == 3 and frame.shape[1] % 4 == 0:
            camera_width = frame.shape[1] // 4
            for camera_id, name in enumerate(CAMERA_NAMES):
                camera = frame[:, camera_id * camera_width:
                               (camera_id + 1) * camera_width]
                panel = cv2.resize(
                    camera, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.rectangle(panel, (0, 0), (640, 35), (0, 0, 0), -1)
                cv2.putText(panel, name, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .62,
                            (100, 255, 140), 2, cv2.LINE_AA)
                panels.append(panel)
        else:
            panels = [self._empty_panel("Waiting for assembled image...")
                      for _ in range(4)]
        preview = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))

        now = time.monotonic()
        if self.recording_started is not None:
            elapsed = now - self.recording_started
            remaining = max(0.0, self.args.duration - elapsed)
            text = "RECORDING {:04.1f}s | remaining {:04.1f}s".format(
                elapsed, remaining)
            color = (30, 40, 255)
        elif self.countdown_started is not None:
            remaining = max(0.0, self.args.countdown -
                            (now - self.countdown_started))
            text = "STARTING IN {:.1f}s | get into position".format(remaining)
            color = (0, 180, 255)
        else:
            text = "PREVIEW | focus this window and press SPACE | Q/ESC cancels"
            color = (0, 255, 255)
        cv2.rectangle(preview, (0, preview.shape[0] - 50),
                      (preview.shape[1], preview.shape[0]), (0, 0, 0), -1)
        cv2.putText(preview, text, (18, preview.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)
        return preview, stamp_ns

    def image_ready(self):
        with self.lock:
            return self.frame is not None

    def begin_countdown(self):
        if self.countdown_started is not None or \
                self.recording_started is not None:
            return
        if not self.image_ready():
            print("Image is not ready; wait and press SPACE again.", flush=True)
            return
        self.countdown_started = time.monotonic()
        print("Countdown started: recording begins in {:.0f} seconds.".format(
            self.args.countdown), flush=True)

    def start_recording(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        bag_base = os.path.join(self.args.output_dir, "input")
        command = [
            "rosbag", "record", "--lz4", "-O", bag_base, TOPIC,
        ]
        self.recorder = subprocess.Popen(command, preexec_fn=os.setsid)
        self.video = cv2.VideoWriter(
            os.path.join(self.args.output_dir, "preview_4cam.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280, 720))
        if not self.video.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 writer")
        self.recording_started = time.monotonic()
        with self.lock:
            self.recording_start_ros_ns = self.frame_stamp_ns
        print("Recording started: {}.bag".format(bag_base), flush=True)

    def write_video_frame(self, preview, stamp_ns):
        if self.video is None or stamp_ns is None or \
                stamp_ns == self.last_video_stamp_ns:
            return
        self.video.write(preview)
        self.last_video_stamp_ns = stamp_ns
        self.written_frames += 1
        self.recording_end_ros_ns = stamp_ns

    def stop_recording(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.recorder is not None and self.recorder.poll() is None:
            print("Finalizing rosbag index...", flush=True)
            os.killpg(os.getpgid(self.recorder.pid), signal.SIGINT)
            try:
                self.recorder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.recorder.pid), signal.SIGTERM)
                self.recorder.wait(timeout=5)
        self.recorder = None

    def save_metadata(self):
        metadata = {
            "label": self.args.label,
            "topic": TOPIC,
            "requested_countdown_s": self.args.countdown,
            "requested_duration_s": self.args.duration,
            "recording_start_ros_ns": self.recording_start_ros_ns,
            "recording_end_ros_ns": self.recording_end_ros_ns,
            "received_frames_process_lifetime": self.received_frames,
            "preview_video_frames": self.written_frames,
            "camera_order": list(CAMERA_NAMES),
            "raw_layout": "CAM_A|CAM_B|CAM_C|CAM_D, bgr8 5120x720",
        }
        with open(os.path.join(self.args.output_dir, "metadata.json"),
                  "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)

    def request_stop(self, *_args):
        self.stop_requested = True

    def run(self):
        window = "OmniNxt pose test recorder: {}".format(self.args.label)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 760)
        completed = False
        try:
            while not self.stop_requested and not rospy.is_shutdown():
                preview, stamp_ns = self.compose()
                cv2.imshow(window, preview)
                key = cv2.waitKey(15) & 0xFF
                if key == ord(" "):
                    self.begin_countdown()
                elif key in (ord("q"), 27):
                    self.stop_requested = True
                now = time.monotonic()
                if self.countdown_started is not None and \
                        self.recording_started is None and \
                        now - self.countdown_started >= self.args.countdown:
                    self.start_recording()
                if self.recording_started is not None:
                    self.write_video_frame(preview, stamp_ns)
                    if now - self.recording_started >= self.args.duration:
                        completed = True
                        break
        finally:
            self.stop_recording()
            cv2.destroyAllWindows()
        if self.recording_started is not None:
            self.save_metadata()
        bag_path = os.path.join(self.args.output_dir, "input.bag")
        if completed and os.path.isfile(bag_path) and os.path.getsize(bag_path):
            print("Recording complete: {}".format(self.args.output_dir),
                  flush=True)
            return 0
        if self.recording_started is None:
            print("Cancelled before recording.", flush=True)
            return 3
        print("Recording stopped early or bag is missing.", file=sys.stderr)
        return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--countdown", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.countdown < 0 or args.duration <= 0:
        raise ValueError("countdown must be >=0 and duration must be >0")
    rospy.init_node("omninxt_pose_test_recorder", anonymous=True,
                    disable_signals=True)
    recorder = PoseTestRecorder(args)
    signal.signal(signal.SIGINT, recorder.request_stop)
    signal.signal(signal.SIGTERM, recorder.request_stop)
    return recorder.run()


if __name__ == "__main__":
    sys.exit(main())
