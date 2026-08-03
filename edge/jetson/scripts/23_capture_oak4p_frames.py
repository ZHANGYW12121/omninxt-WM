#!/usr/bin/env python3
import json
import os
import threading

import cv2
import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image


OUTPUT_DIR = "/root/oak_ffc_ws/runtime/oak4p_frames"
CAMERAS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")
bridge = cv_bridge.CvBridge()
frames = {}
metadata = {}
lock = threading.Lock()
done = threading.Event()


def receive(message, camera):
    image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
    with lock:
        if camera in frames:
            return
        frames[camera] = image
        metadata[camera] = {
            "stamp_ns": message.header.stamp.to_nsec(),
            "encoding": message.encoding,
            "width": message.width,
            "height": message.height,
            "channel_mean_bgr": [round(float(v), 3) for v in image.mean(axis=(0, 1))],
            "channel_std_bgr": [round(float(v), 3) for v in image.std(axis=(0, 1))],
        }
        if len(frames) == len(CAMERAS):
            done.set()


rospy.init_node("capture_oak4p_frames", anonymous=True)
subscribers = [
    rospy.Subscriber(
        f"/oak_ffc_4p/{camera}",
        Image,
        receive,
        callback_args=camera,
        queue_size=1,
        buff_size=4 * 1024 * 1024,
    )
    for camera in CAMERAS
]

if not done.wait(10):
    raise RuntimeError(f"Timed out; received {sorted(frames)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
panels = []
for camera in CAMERAS:
    image = frames[camera]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{camera}.png"), image)
    panel = cv2.resize(image, (640, 360))
    cv2.putText(
        panel,
        camera,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    panels.append(panel)

montage = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
cv2.imwrite(os.path.join(OUTPUT_DIR, "CAM_A_to_D_montage.png"), montage)
print(json.dumps(metadata, indent=2, sort_keys=True))
