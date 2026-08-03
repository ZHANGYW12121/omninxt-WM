#!/usr/bin/env python3
import json
import os

import cv2
import cv_bridge
import rospy
from sensor_msgs.msg import Image


output_path = "/root/oak_ffc_ws/runtime/oak4p_frames/assemble_CAM_A_B_C_D.png"
rospy.init_node("capture_oak4p_assembled", anonymous=True)
message = rospy.wait_for_message(
    "/oak_ffc_4p/assemble_image", Image, timeout=10
)
image = cv_bridge.CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
if not cv2.imwrite(output_path, image):
    raise RuntimeError(f"Failed to write {output_path}")
print(
    json.dumps(
        {
            "topic": "/oak_ffc_4p/assemble_image",
            "stamp_ns": message.header.stamp.to_nsec(),
            "encoding": message.encoding,
            "width": message.width,
            "height": message.height,
            "output": output_path,
        },
        indent=2,
        sort_keys=True,
    )
)
