#!/usr/bin/env python3
"""Create evenly time-sampled mono bags for memory-safe TartanCalib runs."""

import argparse
import json
import os

import rosbag


CAMERAS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be positive")

    prepared_dir = os.path.realpath(args.prepared_dir)
    output_dir = os.path.join(prepared_dir, "intrinsics_5hz")
    os.makedirs(output_dir, exist_ok=True)
    min_delta_ns = int(round(1.0e9 / args.rate))
    summary = {"requested_rate_hz": args.rate, "cameras": {}}

    for camera in CAMERAS:
        topic = f"/{camera}"
        source_path = os.path.join(prepared_dir, f"{camera}.bag")
        output_path = os.path.join(output_dir, f"{camera}.bag")
        if not os.path.isfile(source_path):
            raise RuntimeError(f"Missing source bag: {source_path}")

        input_count = 0
        output_count = 0
        first_stamp_ns = None
        last_written_ns = None
        last_stamp_ns = None
        with rosbag.Bag(source_path, "r") as source, rosbag.Bag(
            output_path, "w", compression=rosbag.Compression.LZ4
        ) as output:
            for _, message, recorded_time in source.read_messages(topics=[topic]):
                input_count += 1
                stamp_ns = message.header.stamp.to_nsec()
                if first_stamp_ns is None:
                    first_stamp_ns = stamp_ns
                last_stamp_ns = stamp_ns
                if (
                    last_written_ns is None
                    or stamp_ns - last_written_ns >= min_delta_ns
                ):
                    output.write(topic, message, recorded_time)
                    output_count += 1
                    last_written_ns = stamp_ns

        if output_count < 100:
            raise RuntimeError(
                f"Too few frames retained for {camera}: {output_count}"
            )
        duration = (
            (last_stamp_ns - first_stamp_ns) / 1.0e9
            if first_stamp_ns is not None and last_stamp_ns is not None
            else 0.0
        )
        summary["cameras"][camera] = {
            "source": source_path,
            "output": output_path,
            "input_frames": input_count,
            "output_frames": output_count,
            "duration_seconds": round(duration, 6),
            "effective_rate_hz": (
                round((output_count - 1) / duration, 6) if duration > 0 else 0.0
            ),
        }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
