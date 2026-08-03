#!/usr/bin/env python3
"""Continuously publish the newest exact-stamp pose/depth inputs via an NPZ file."""

import argparse
import json
import os
import threading
import time
from collections import defaultdict

import numpy as np
import rospy
from sensor_msgs.msg import Image


KINDS = ("left", "depth")


def decode(message):
    if message.encoding == "mono8":
        dtype = np.dtype(np.uint8)
    elif message.encoding == "32FC1":
        dtype = np.dtype(np.float32)
    else:
        raise ValueError("Unsupported encoding " + message.encoding)
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    values = message.step // dtype.itemsize
    array = np.frombuffer(message.data, dtype=dtype).reshape(message.height, values)
    return array[:, :message.width].astype(dtype.newbyteorder("="), copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--max-hz", type=float, default=10.0)
    args = parser.parse_args()
    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "producer.pid"), "w") as stream:
        stream.write(str(os.getpid()))

    required = {(i, kind) for i in range(4) for kind in KINDS}
    buckets = defaultdict(dict)
    ready = threading.Event()
    lock = threading.Lock()
    latest = None
    latest_stamp = None

    def callback(message, key):
        nonlocal latest, latest_stamp
        stamp = message.header.stamp.to_nsec()
        image = decode(message)
        with lock:
            buckets[stamp][key] = image
            if required.issubset(buckets[stamp]):
                latest = dict(buckets[stamp])
                latest_stamp = stamp
                ready.set()
            for old in sorted(buckets)[:-30]:
                buckets.pop(old, None)

    rospy.init_node("stream_pose_depth_frames", anonymous=True)
    subscribers = []
    for index in range(4):
        for kind in KINDS:
            subscribers.append(rospy.Subscriber(
                "/depth_estimation/stereo_{}/{}".format(index, kind), Image,
                callback, callback_args=(index, kind), queue_size=1,
                buff_size=8 * 1024 * 1024))

    period = 1.0 / max(0.1, args.max_hz)
    last_written_stamp = None
    frame_count = 0
    started = time.monotonic()
    while not rospy.is_shutdown():
        if not ready.wait(1.0):
            continue
        loop_start = time.monotonic()
        with lock:
            stamp = latest_stamp
            frame = None if latest is None else dict(latest)
            ready.clear()
        if frame is None or stamp == last_written_stamp:
            continue
        payload = {"stamp_ns": np.asarray(stamp, dtype=np.int64)}
        for index in range(4):
            payload["left{}".format(index)] = frame[(index, "left")]
            payload["depth{}".format(index)] = frame[(index, "depth")]
        temporary = os.path.join(output_dir, "latest.tmp.npz")
        final = os.path.join(output_dir, "latest.npz")
        np.savez(temporary, **payload)
        os.replace(temporary, final)
        last_written_stamp = stamp
        frame_count += 1
        status = {"stamp_ns": stamp, "frames": frame_count,
                  "producer_hz": round(frame_count / max(1e-6, time.monotonic() - started), 3)}
        status_tmp = os.path.join(output_dir, "producer_status.tmp")
        with open(status_tmp, "w") as stream:
            json.dump(status, stream)
        os.replace(status_tmp, os.path.join(output_dir, "producer_status.json"))
        remaining = period - (time.monotonic() - loop_start)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    main()
