#!/usr/bin/env python3
"""Publish a recorded 4-camera montage without changing its pixel order."""

import argparse
import os
import sys

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="5120x720 BGR montage")
    parser.add_argument("--topic", default="/oak_ffc_4p/assemble_image")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error("image does not exist: {}".format(args.image))
    if args.rate <= 0 or args.count <= 0:
        parser.error("--rate and --count must be positive")
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        parser.error("OpenCV could not decode: {}".format(args.image))
    if image.shape != (720, 5120, 3) or image.dtype.name != "uint8":
        parser.error(
            "expected uint8 BGR 5120x720x3, got {} {}".format(
                image.shape, image.dtype
            )
        )

    rospy.init_node("omninxt_offline_feeder", anonymous=True)
    publisher = rospy.Publisher(args.topic, Image, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(args.rate)
    for sequence in range(args.count):
        if rospy.is_shutdown():
            break
        message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        message.header.seq = sequence
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "oak_ffc_4p"
        publisher.publish(message)
        rate.sleep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
