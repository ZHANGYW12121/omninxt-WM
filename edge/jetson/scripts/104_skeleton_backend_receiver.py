#!/usr/bin/env python3
"""Reference backend receiver for OmniNxt fixed-schema skeleton frames."""

import argparse
import json
import socket

from skeleton_stream import StgcnWindow, validate_packet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9765)
    parser.add_argument("--window", type=int, default=30,
                        help="ST-GCN temporal window in frames")
    parser.add_argument("--max-people", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Exit after N frames; 0 runs forever")
    return parser.parse_args()


def main():
    args = parse_args()
    assembler = StgcnWindow(args.window, args.max_people)
    received = 0
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print("Skeleton backend listening on {}:{}".format(
        args.host, args.port), flush=True)
    while args.max_frames <= 0 or received < args.max_frames:
        connection, address = server.accept()
        print("Nano connected from {}:{}".format(*address), flush=True)
        with connection, connection.makefile("rb") as stream:
            for line in stream:
                try:
                    packet = validate_packet(json.loads(line))
                except (ValueError, json.JSONDecodeError) as error:
                    print("Rejected packet: {}".format(error), flush=True)
                    continue
                received += 1
                tensor = assembler.push(packet)
                if received % max(1, args.print_every) == 0:
                    valid = sum(
                        int(row[5]) for person in packet["people"]
                        for row in person["joints"])
                    message = "seq={} people={} valid_joints={}".format(
                        packet["sequence"], len(packet["people"]), valid)
                    if tensor is not None:
                        message += " stgcn_tensor={}".format(tensor.shape)
                    print(message, flush=True)
                if tensor is not None:
                    # Integrate the backend model here.  Tensor layout is:
                    # [N,C,T,V,M], C=[x,y,z,confidence,valid], V=17.
                    # prediction = stgcn_model(tensor)
                    pass
                if args.max_frames > 0 and received >= args.max_frames:
                    break
    server.close()


if __name__ == "__main__":
    main()
