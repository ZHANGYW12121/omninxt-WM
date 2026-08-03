#!/usr/bin/env python3
"""Request a temporary PX4 HIGHRES_IMU MAVLink stream rate."""

import argparse
import time

from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--rate", type=float, default=200.0)
    args = parser.parse_args()

    link = mavutil.mavlink_connection(
        args.device, baud=115200, source_system=245, autoreconnect=False
    )
    deadline = time.monotonic() + 8.0
    heartbeat = None
    while time.monotonic() < deadline and heartbeat is None:
        link.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )
        heartbeat = link.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
    if heartbeat is None:
        raise RuntimeError("No PX4 heartbeat received")

    interval_us = int(round(1_000_000.0 / args.rate))
    link.mav.command_long_send(
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )
    acknowledgement = link.recv_match(
        type="COMMAND_ACK", blocking=True, timeout=3.0
    )
    if (
        acknowledgement is None
        or acknowledgement.command
        != mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
        or acknowledgement.result != mavutil.mavlink.MAV_RESULT_ACCEPTED
    ):
        raise RuntimeError(
            f"PX4 rejected HIGHRES_IMU interval request: {acknowledgement}"
        )
    print(
        f"PX4 accepted temporary HIGHRES_IMU request: "
        f"{args.rate:.1f} Hz ({interval_us} us)"
    )


if __name__ == "__main__":
    main()

