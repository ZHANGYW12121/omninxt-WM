#!/usr/bin/env python3
"""Measure local pose cadence and non-blocking TCP sender health."""

import argparse
import json
import statistics
import time
from collections import Counter

import rospy
from std_msgs.msg import String


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=45.0)
    args = parser.parse_args()

    samples = []
    receive_times = []
    rospy.init_node("pose_transport_monitor", anonymous=True,
                    disable_signals=True)

    def callback(message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        samples.append(payload.get("status", {}))
        receive_times.append(time.monotonic())

    rospy.Subscriber("/omninxt_pose/status", String, callback,
                     queue_size=100, tcp_nodelay=True)
    started = time.monotonic()
    while time.monotonic() - started < args.duration and \
            not rospy.is_shutdown():
        time.sleep(.05)

    wall_gaps = [(second - first) * 1000.0
                 for first, second in zip(receive_times, receive_times[1:])]
    stamps = [int(sample.get("stamp_ns", 0)) for sample in samples]
    stamp_gaps = [(second - first) / 1e6
                  for first, second in zip(stamps, stamps[1:])
                  if second > first]
    total_ms = [float(sample.get("total_ms", 0.0)) for sample in samples]
    processing_hz = [float(sample.get("processing_hz", 0.0))
                     for sample in samples if sample.get("processing_hz")]
    backends = [sample.get("backend_stream", {}) for sample in samples]
    sent = [int(value.get("sent", 0)) for value in backends
            if value.get("enabled")]
    dropped = [int(value.get("dropped", 0)) for value in backends
               if value.get("enabled")]
    errors = Counter(str(value.get("last_error")) for value in backends
                     if value.get("last_error"))
    connected = [bool(value.get("connected")) for value in backends
                 if value.get("enabled")]
    result = {
        "duration_s": round(time.monotonic() - started, 3),
        "status_samples": len(samples),
        "observed_status_hz": round(
            (len(receive_times) - 1) /
            max(1e-9, receive_times[-1] - receive_times[0]), 3)
            if len(receive_times) > 1 else 0.0,
        "wall_gap_ms": {
            "median": round(percentile(wall_gaps, .5), 3),
            "p95": round(percentile(wall_gaps, .95), 3),
            "maximum": round(max(wall_gaps, default=0.0), 3),
            "over_200ms": sum(value > 200.0 for value in wall_gaps),
            "over_500ms": sum(value > 500.0 for value in wall_gaps),
        },
        "camera_stamp_gap_ms": {
            "median": round(percentile(stamp_gaps, .5), 3),
            "p95": round(percentile(stamp_gaps, .95), 3),
            "maximum": round(max(stamp_gaps, default=0.0), 3),
            "over_200ms": sum(value > 200.0 for value in stamp_gaps),
        },
        "processing_hz": {
            "median": round(percentile(processing_hz, .5), 3),
            "minimum": round(min(processing_hz, default=0.0), 3),
        },
        "frame_compute_ms": {
            "median": round(percentile(total_ms, .5), 3),
            "p95": round(percentile(total_ms, .95), 3),
            "maximum": round(max(total_ms, default=0.0), 3),
            "over_100ms": sum(value > 100.0 for value in total_ms),
            "over_200ms": sum(value > 200.0 for value in total_ms),
        },
        "backend": {
            "targets": sorted({"{}:{}".format(
                value.get("host"), value.get("port")) for value in backends
                if value.get("enabled")}),
            "connected_samples": sum(connected),
            "disconnected_samples": len(connected) - sum(connected),
            "sent_delta": (sent[-1] - sent[0]) if len(sent) > 1 else 0,
            "dropped_delta": ((dropped[-1] - dropped[0])
                              if len(dropped) > 1 else 0),
            "last_errors": dict(errors),
        },
        "people_histogram": dict(sorted(Counter(
            int(sample.get("people", 0)) for sample in samples).items())),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
