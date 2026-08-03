#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--topic", default="/oak_ffc_4p/assemble_image")
    p.add_argument("--publish-rate", type=float, default=10.0)
    p.add_argument("--repeat-count", type=int, default=50)
    p.add_argument("--frame-id", default="imu")
    p.add_argument("--connection-timeout", type=float, default=600.0)
    args = p.parse_args(rospy.myargv()[1:])
    frame_dir = Path(args.input_dir)
    metadata = json.loads((frame_dir / "metadata.json").read_text())
    order = metadata.get("camera_order", ["cam0", "cam1", "cam2", "cam3"])
    images = [cv2.imread(str(frame_dir / f"{name}.png"), cv2.IMREAD_COLOR) for name in order]
    if any(image is None for image in images):
        raise RuntimeError(f"missing/unreadable camera PNG in {frame_dir}")
    if len({image.shape for image in images}) != 1:
        raise RuntimeError("camera image dimensions differ")
    quad = cv2.hconcat(images)
    rospy.init_node("omninxt_offline_feeder", anonymous=True)
    publisher = rospy.Publisher(args.topic, Image, queue_size=1)
    deadline = rospy.Time.now() + rospy.Duration(args.connection_timeout)
    while publisher.get_num_connections() == 0 and not rospy.is_shutdown() and rospy.Time.now() < deadline:
        rospy.sleep(0.1)
    if publisher.get_num_connections() == 0:
        raise RuntimeError(f"no subscriber on {args.topic}")
    bridge, rate = CvBridge(), rospy.Rate(args.publish_rate)
    for _ in range(args.repeat_count):
        if rospy.is_shutdown(): break
        msg = bridge.cv2_to_imgmsg(quad, encoding="bgr8")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = args.frame_id
        publisher.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
