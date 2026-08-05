#!/usr/bin/env python3
"""Capture pose status JSON and write a compact offline-regression summary."""

import argparse
import json
import os
import time
from collections import Counter

import rospy
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-people", type=int, required=True)
    parser.add_argument("--idle-timeout", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "status.jsonl")
    samples = []
    last_message = None
    last_stamp_ns = None

    rospy.init_node("omninxt_pose_regression_logger", anonymous=True,
                    disable_signals=True)

    stream = open(jsonl_path, "w", buffering=1)

    def callback(message):
        nonlocal last_message, last_stamp_ns
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        stamp_ns = payload.get("status", {}).get("stamp_ns")
        if stamp_ns is None or stamp_ns == last_stamp_ns:
            return
        last_stamp_ns = stamp_ns
        samples.append(payload)
        stream.write(json.dumps(payload, ensure_ascii=False,
                                separators=(",", ":")) + "\n")
        last_message = time.monotonic()

    rospy.Subscriber("/omninxt_pose/status", String, callback,
                     queue_size=20, tcp_nodelay=True)

    started = time.monotonic()
    while not rospy.is_shutdown():
        time.sleep(.1)
        if last_message is not None and \
                time.monotonic() - last_message >= args.idle_timeout:
            break
        if last_message is None and time.monotonic() - started > 90.0:
            break
    stream.close()

    statuses = [sample.get("status", {}) for sample in samples]
    people_counts = [int(status.get("people", 0)) for status in statuses]
    raw_counts = [int(status.get("raw_sector_people", 0))
                  for status in statuses]
    identities = set()
    for sample in samples:
        for person in sample.get("people", []):
            if person.get("person_id") is not None:
                identities.add(int(person["person_id"]))
    tracker_states = [status.get("identity_tracker", {})
                      for status in statuses]
    processing_hz = [float(status.get("processing_hz", 0.0))
                     for status in statuses if status.get("processing_hz")]
    summary = {
        "samples": len(samples),
        "expected_people": args.expected_people,
        "people_histogram": dict(sorted(Counter(people_counts).items())),
        "raw_people_histogram": dict(sorted(Counter(raw_counts).items())),
        "frames_above_expected": sum(
            value > args.expected_people for value in people_counts),
        "frames_below_expected": sum(
            value < args.expected_people for value in people_counts),
        "maximum_people": max(people_counts, default=0),
        "maximum_raw_people": max(raw_counts, default=0),
        "published_person_ids": sorted(identities),
        "maximum_next_person_id": max(
            (int(state.get("next_person_id", 0)) for state in tracker_states),
            default=0),
        "maximum_active_tracks": max(
            (int(state.get("active_tracks", 0)) for state in tracker_states),
            default=0),
        "pose_duplicate_rois_removed": sum(
            int(status.get("pose_duplicate_rois_removed", 0))
            for status in statuses),
        "new_measurements_suppressed": sum(
            int(status.get("identity_tracker", {}).get(
                "new_measurements_suppressed", 0)) for status in statuses),
        "processing_hz_median": (float(sorted(processing_hz)[
            len(processing_hz) // 2]) if processing_hz else 0.0),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
