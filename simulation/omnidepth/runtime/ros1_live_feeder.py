#!/usr/bin/env python3
"""Bridge Isaac's atomic latest-frame tmpfs bundle into ROS 1 topics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from isaac_gt_disparity import IsaacGtDisparity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-root", default="/dev/shm/omninxt_sync/live")
    parser.add_argument("--poll-hz", type=float, default=200.0)
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--raw-topic", default="/oak_ffc_4p/assemble_image")
    parser.add_argument(
        "--gt-disparity-topic", default="/omninxt_gt/disparity_mosaic"
    )
    parser.add_argument(
        "--config-dir",
        default=("/root/swarm_ws/src/D2SLAM/config/"
                 "quadcam_drone_nxt_sync_20260802"),
    )
    return parser.parse_args(rospy.myargv()[1:])


def stable_latest(root):
    latest = root / "LATEST"
    if not latest.is_file():
        return None
    name = latest.read_text(encoding="utf-8").strip()
    if not name or "/" in name or name.startswith("."):
        return None
    frame = root / name
    return frame if (frame / "metadata.json").is_file() else None


def image_message(bridge, array, encoding, stamp, frame_id):
    message = bridge.cv2_to_imgmsg(np.ascontiguousarray(array), encoding=encoding)
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    return message


def main():
    args = parse_args()
    root = Path(args.live_root)
    rospy.init_node("omninxt_isaac_live_feeder", anonymous=False)
    bridge = CvBridge()
    raw_pub = rospy.Publisher(args.raw_topic, Image, queue_size=1)
    gt_disparity_pub = rospy.Publisher(
        args.gt_disparity_topic, Image, queue_size=1
    )
    gt_converter = IsaacGtDisparity(args.config_dir)
    anchor_pub = rospy.Publisher(
        "/depth_estimation/pose_anchor_mosaic", Image, queue_size=1
    )
    stereo_pub = rospy.Publisher(
        "/depth_estimation/pose_stereo_mosaic", Image, queue_size=1
    )
    # Do not use rospy.Rate here.  The depth launch enables /use_sim_time and
    # Isaac's file bridge is the clock source only through image stamps, so a
    # rospy.Rate would wait forever when no /clock topic is present.
    poll_period = 1.0 / max(1.0, args.poll_hz)
    last_name = None
    frames = 0
    started = time.monotonic()
    rospy.loginfo("Waiting for Isaac live frames in %s", root)
    while not rospy.is_shutdown():
        try:
            frame = stable_latest(root)
            if frame is None or frame.name == last_name:
                time.sleep(poll_period)
                continue
            metadata = json.loads(
                (frame / "metadata.json").read_text(encoding="utf-8")
            )
            stamp = rospy.Time.from_sec(float(metadata["sim_time_sec"]))
            mode = metadata["sensor_mode"]
            if mode == "raw_mei":
                quad = np.load(frame / metadata["payload"], allow_pickle=False)
                if quad.shape != (720, 5120, 3) or quad.dtype != np.uint8:
                    raise ValueError("invalid raw quad shape/dtype: {} {}".format(
                        quad.shape, quad.dtype))
                if not metadata.get("gt_range_enabled", False):
                    raise ValueError("live frame has no Isaac GT range payload")
                quad_range = np.load(
                    frame / metadata["gt_range_payload"], allow_pickle=False
                )
                if quad_range.shape != (720, 5120) or quad_range.dtype != np.float32:
                    raise ValueError("invalid GT range shape/dtype: {} {}".format(
                        quad_range.shape, quad_range.dtype))
                disparity, disparity_stats = gt_converter.convert(quad_range)
                if disparity.shape != (960, 320) or disparity.dtype != np.float32:
                    raise ValueError("invalid GT disparity mosaic: {} {}".format(
                        disparity.shape, disparity.dtype))
                # Both messages carry the exact same Isaac simulation stamp.
                # The C++ node consumes them with an ExactTime synchronizer.
                raw_pub.publish(image_message(
                    bridge, quad, "rgb8", stamp, args.frame_id
                ))
                gt_disparity_pub.publish(image_message(
                    bridge, disparity, "32FC1", stamp, args.frame_id
                ))
            elif mode == "rectified_validation":
                anchors = np.load(
                    frame / metadata["anchor_payload"], allow_pickle=False
                )
                stereo = np.load(
                    frame / metadata["stereo_payload"], allow_pickle=False
                )
                if anchors.shape != (640, 832) or stereo.shape != (960, 640):
                    raise ValueError("invalid validation mosaics: {} {}".format(
                        anchors.shape, stereo.shape))
                anchor_pub.publish(image_message(
                    bridge, anchors, "mono8", stamp, args.frame_id
                ))
                stereo_pub.publish(image_message(
                    bridge, stereo, "mono8", stamp, args.frame_id
                ))
            else:
                raise ValueError("unsupported sensor mode " + str(mode))
            last_name = frame.name
            frames += 1
            elapsed = time.monotonic() - started
            if frames % 50 == 0:
                rospy.loginfo(
                    "Forwarded %d frames (%.2f Hz), mode=%s, sim=%.3f",
                    frames, frames / max(elapsed, 1e-6), mode,
                    float(metadata["sim_time_sec"]),
                )
                if mode == "raw_mei":
                    rospy.loginfo("Isaac GT disparity: %s", disparity_stats)
        except FileNotFoundError:
            # The writer removes old rolling slots after publishing a newer one.
            pass
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Isaac live frame rejected: %s", exc)
        time.sleep(poll_period)


if __name__ == "__main__":
    main()
