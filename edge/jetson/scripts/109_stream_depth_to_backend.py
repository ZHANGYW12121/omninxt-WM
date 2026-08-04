#!/usr/bin/env python3
"""Non-blocking Nano-to-backend transport for pose images and four depths.

One TCP connection multiplexes two independently timestamped packet kinds:

* ``images``: anchor and rectified-stereo mosaics at the camera processing rate;
* ``depth``: all four exact-timestamp metric HITNet depth maps.

Keeping the rates independent lets a backend run 2-D detection at about 10 Hz
while consuming dense depth at its native slower rate.  Socket I/O and lossless
compression never run in ROS callbacks.  At most the newest pending packet of
each kind is retained, so a slow receiver cannot stall OmniDepth.
"""

import argparse
import json
import math
import socket
import struct
import threading
import time
import uuid
import zlib
from collections import defaultdict

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


MAGIC = b"OPB1"
PREFIX = struct.Struct("!4sII")
IMAGE_SCHEMA = "omninxt.pose_images.v1"
DEPTH_SCHEMA = "omninxt.depth4.v1"
PAIR_NAMES = ("AB_RIGHT", "BC_REAR", "CD_LEFT", "DA_FRONT")
ANCHOR_NAMES = (
    "CAM_A_FRONT_RIGHT", "CAM_B_REAR_RIGHT",
    "CAM_C_REAR_LEFT", "CAM_D_FRONT_LEFT",
)
DEFAULT_ANCHOR_TOPIC = "/depth_estimation/pose_anchor_mosaic"
DEFAULT_STEREO_TOPIC = "/depth_estimation/pose_stereo_mosaic"
DEFAULT_DEPTH_TOPICS = tuple(
    "/depth_estimation/stereo_{}/depth".format(index) for index in range(4)
)


def decode_mono8(message):
    if message.encoding not in ("mono8", "8UC1"):
        raise ValueError("Unsupported image encoding: " + message.encoding)
    if message.step < message.width:
        raise ValueError("Invalid mono8 row step: {}".format(message.step))
    image = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.step
    )[:, :message.width]
    return np.ascontiguousarray(image)


def decode_depth(message):
    """Decode supported ROS depth encodings into native float32 metres."""
    if message.encoding == "32FC1":
        dtype = np.dtype(np.float32)
        scale = 1.0
    elif message.encoding in ("16UC1", "mono16"):
        dtype = np.dtype(np.uint16)
        scale = 0.001
    else:
        raise ValueError("Unsupported depth encoding: " + message.encoding)
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    row_values = message.step // dtype.itemsize
    if row_values < message.width:
        raise ValueError("Invalid depth row step: {}".format(message.step))
    image = np.frombuffer(message.data, dtype=dtype).reshape(
        message.height, row_values
    )[:, :message.width]
    image = image.astype(np.float32, copy=True)
    if scale != 1.0:
        image *= scale
    return image


def wire_depths(depths, wire_format):
    stack = np.stack(depths, axis=0)
    if wire_format == "float32":
        return (
            np.ascontiguousarray(stack.astype("<f4", copy=False)),
            1.0,
            "IEEE754 values preserved; validate finite and positive values",
        )
    if wire_format == "uint16_mm":
        valid = np.isfinite(stack) & (stack > 0.0)
        array = np.zeros(stack.shape, dtype="<u2")
        array[valid] = np.rint(
            np.clip(stack[valid], 0.001, 65.535) * 1000.0
        ).astype(np.uint16)
        return array, 0.001, "0 means invalid; positive values are millimetres"
    raise ValueError("Unsupported wire format: " + wire_format)


def encode_packet(item, session_id, sequence, wire_format,
                  compression, compression_level):
    """Encode one image or depth packet using offset-described binary blocks."""
    block_specs = []
    header_extra = {}
    if item["kind"] == "images":
        block_specs = [
            {
                "name": "anchor_mosaic",
                "array": np.ascontiguousarray(item["anchor"], dtype=np.uint8),
                "topic": item["topics"][0],
                "source_encoding": item["source_encodings"][0],
                "frame_id": item["source_frame_ids"][0],
                "units": "intensity_u8",
                "layout": "CAM_A|CAM_B over CAM_C|CAM_D; each 416x320",
            },
            {
                "name": "stereo_mosaic",
                "array": np.ascontiguousarray(item["stereo"], dtype=np.uint8),
                "topic": item["topics"][1],
                "source_encoding": item["source_encodings"][1],
                "frame_id": item["source_frame_ids"][1],
                "units": "intensity_u8",
                "layout": "AB,BC,CD,DA rows; each row left|right, each view 320x240",
            },
        ]
        schema = IMAGE_SCHEMA
        header_extra = {
            "anchor_names": list(ANCHOR_NAMES),
            "pair_names": list(PAIR_NAMES),
        }
    elif item["kind"] == "depth":
        depths = item["depths"]
        shape = depths[0].shape
        if any(depth.shape != shape for depth in depths):
            raise ValueError("All four depth images must have the same shape")
        depth_array, scale_m, invalid_semantics = wire_depths(
            depths, wire_format
        )
        block_specs = [{
            "name": "depths",
            "array": depth_array,
            "topics": list(item["topics"]),
            "source_encodings": list(item["source_encodings"]),
            "frame_ids": list(item["source_frame_ids"]),
            "units": "metre",
            "scale_m": scale_m,
            "invalid_semantics": invalid_semantics,
            "layout": "pair-major AB,BC,CD,DA; C contiguous",
        }]
        schema = DEPTH_SCHEMA
        header_extra = {"pair_names": list(PAIR_NAMES)}
    else:
        raise ValueError("Unsupported packet kind: " + str(item["kind"]))

    raw_parts = []
    blocks = []
    offset = 0
    for spec in block_specs:
        array = spec.pop("array")
        raw = array.tobytes(order="C")
        metadata = dict(spec)
        metadata.update({
            "dtype": array.dtype.str,
            "shape": [int(value) for value in array.shape],
            "offset": offset,
            "nbytes": len(raw),
        })
        blocks.append(metadata)
        raw_parts.append(raw)
        offset += len(raw)
    raw_payload = b"".join(raw_parts)
    if compression == "zlib":
        payload = zlib.compress(raw_payload, compression_level)
    elif compression == "none":
        payload = raw_payload
    else:
        raise ValueError("Unsupported compression: " + compression)

    header = {
        "schema": schema,
        "kind": item["kind"],
        "session_id": session_id,
        "sequence": int(sequence),
        "source_sequence": int(item["source_sequence"]),
        "capture_timestamp_ns": int(item["stamp_ns"]),
        "send_timestamp_ns": int(time.time_ns()),
        "blocks": blocks,
        "compression": compression,
        "compression_level": int(compression_level) if compression == "zlib" else 0,
        "raw_bytes": len(raw_payload),
        "payload_bytes": len(payload),
        "raw_crc32": "{:08x}".format(zlib.crc32(raw_payload) & 0xFFFFFFFF),
    }
    header.update(header_extra)
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return (
        PREFIX.pack(MAGIC, len(header_bytes), len(payload))
        + header_bytes + payload,
        header,
    )


class LatestKindTcpSender:
    """Retain at most one pending image packet and one pending depth packet."""

    def __init__(self, host, port, wire_format="float32",
                 compression="zlib", compression_level=1,
                 connect_timeout=0.5, send_timeout=0.75):
        self.host = str(host)
        self.port = int(port)
        self.wire_format = wire_format
        self.compression = compression
        self.compression_level = int(compression_level)
        self.connect_timeout = float(connect_timeout)
        self.send_timeout = float(send_timeout)
        self.session_id = str(uuid.uuid4())
        self.condition = threading.Condition()
        self.pending = {}
        self.stop_event = threading.Event()
        self.connection = None
        self.started = time.monotonic()
        self.complete = {"images": 0, "depth": 0}
        self.dropped = {"images": 0, "depth": 0}
        self.encoded = {"images": 0, "depth": 0}
        self.sent = {"images": 0, "depth": 0}
        self.sent_bytes = 0
        self.reconnects = 0
        self.connected = False
        self.last_error = None
        self.last_header = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, item):
        kind = item["kind"]
        with self.condition:
            self.complete[kind] += 1
            if kind in self.pending:
                self.dropped[kind] += 1
            self.pending[kind] = item
            self.condition.notify()

    def _take(self):
        with self.condition:
            while not self.pending and not self.stop_event.is_set():
                self.condition.wait(timeout=0.25)
            if not self.pending:
                return None
            kind = min(
                self.pending,
                key=lambda key: self.pending[key]["stamp_ns"],
            )
            return self.pending.pop(kind)

    def _close_connection(self):
        connection = self.connection
        self.connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        with self.condition:
            self.connected = False

    def _connect(self):
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout
        )
        connection.settimeout(self.send_timeout)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection = connection
        with self.condition:
            self.reconnects += 1
            self.connected = True
            self.last_error = None

    def _run(self):
        sequence = 0
        while not self.stop_event.is_set():
            item = self._take()
            if item is None:
                continue
            sequence += 1
            try:
                packet, header = encode_packet(
                    item, self.session_id, sequence, self.wire_format,
                    self.compression, self.compression_level,
                )
                with self.condition:
                    self.encoded[item["kind"]] += 1
                    self.last_header[item["kind"]] = header
            except (ValueError, MemoryError) as error:
                with self.condition:
                    self.last_error = "encode: {}".format(error)
                continue
            if self.connection is None:
                try:
                    self._connect()
                except OSError as error:
                    with self.condition:
                        self.connected = False
                        self.last_error = "connect: {}".format(error)
                    time.sleep(0.15)
                    continue
            try:
                self.connection.sendall(packet)
                with self.condition:
                    self.sent[item["kind"]] += 1
                    self.sent_bytes += len(packet)
                    self.last_error = None
            except OSError as error:
                with self.condition:
                    self.connected = False
                    self.last_error = "send: {}".format(error)
                self._close_connection()

    def status(self):
        with self.condition:
            elapsed = max(1e-6, time.monotonic() - self.started)
            last = {
                kind: {
                    "capture_timestamp_ns": header.get("capture_timestamp_ns"),
                    "raw_bytes": header.get("raw_bytes"),
                    "payload_bytes": header.get("payload_bytes"),
                }
                for kind, header in self.last_header.items()
            }
            return {
                "schemas": [IMAGE_SCHEMA, DEPTH_SCHEMA],
                "enabled": True,
                "host": self.host,
                "port": self.port,
                "session_id": self.session_id,
                "connected": self.connected,
                "complete": dict(self.complete),
                "dropped_pending": dict(self.dropped),
                "encoded": dict(self.encoded),
                "sent": dict(self.sent),
                "image_sent_hz": round(self.sent["images"] / elapsed, 3),
                "depth_sent_hz": round(self.sent["depth"] / elapsed, 3),
                "sent_mbps": round(self.sent_bytes * 8.0 / elapsed / 1e6, 3),
                "reconnects": self.reconnects,
                "last": last,
                "last_error": self.last_error,
                "wire_format": self.wire_format,
                "compression": self.compression,
            }

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self._close_connection()
        self.thread.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9766)
    parser.add_argument("--wire-format", choices=("float32", "uint16_mm"),
                        default="float32")
    parser.add_argument("--compression", choices=("zlib", "none"),
                        default="zlib")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--image-max-hz", type=float, default=0.0)
    parser.add_argument("--depth-max-hz", type=float, default=0.0)
    parser.add_argument("--connect-timeout", type=float, default=0.5)
    parser.add_argument("--send-timeout", type=float, default=0.75)
    parser.add_argument("--anchor-topic", default=DEFAULT_ANCHOR_TOPIC)
    parser.add_argument("--stereo-topic", default=DEFAULT_STEREO_TOPIC)
    parser.add_argument("--depth-topics", nargs=4,
                        default=DEFAULT_DEPTH_TOPICS,
                        metavar=("DEPTH0", "DEPTH1", "DEPTH2", "DEPTH3"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if not 0 <= args.compression_level <= 9:
        raise SystemExit("--compression-level must be in [0, 9]")
    for name, value in (("image-max-hz", args.image_max_hz),
                        ("depth-max-hz", args.depth_max_hz)):
        if not math.isfinite(value) or value < 0:
            raise SystemExit("--{} must be finite and non-negative".format(name))

    rospy.init_node("omninxt_perception_tcp_sender", anonymous=False)
    status_publisher = rospy.Publisher(
        "/omninxt_perception_stream/status", String, queue_size=1
    )
    sender = LatestKindTcpSender(
        args.host, args.port, args.wire_format, args.compression,
        args.compression_level, args.connect_timeout, args.send_timeout,
    )

    image_buckets = defaultdict(dict)
    depth_buckets = defaultdict(dict)
    image_lock = threading.Lock()
    depth_lock = threading.Lock()
    source_sequence = {"images": 0, "depth": 0}
    last_submit = {"images": 0.0, "depth": 0.0}
    decode_errors = {"images": 0, "depth": 0}

    def rate_allows(kind, max_hz):
        now = time.monotonic()
        period = 0.0 if max_hz <= 0 else 1.0 / max_hz
        if period > 0.0 and now - last_submit[kind] < period:
            return False
        last_submit[kind] = now
        return True

    def image_callback(message, kind):
        try:
            image = decode_mono8(message)
        except (ValueError, TypeError) as error:
            decode_errors["images"] += 1
            rospy.logwarn_throttle(2.0, "Image decode failed: %s", error)
            return
        expected = (640, 832) if kind == "anchor" else (960, 640)
        if image.shape != expected:
            decode_errors["images"] += 1
            rospy.logwarn_throttle(
                2.0, "Unexpected %s mosaic shape %s (expected %s)",
                kind, image.shape, expected,
            )
            return
        stamp_ns = message.header.stamp.to_nsec()
        ready = None
        with image_lock:
            image_buckets[stamp_ns][kind] = {
                "image": image,
                "encoding": message.encoding,
                "frame_id": message.header.frame_id,
            }
            bucket = image_buckets[stamp_ns]
            if "anchor" in bucket and "stereo" in bucket:
                if rate_allows("images", args.image_max_hz):
                    source_sequence["images"] += 1
                    ready = {
                        "kind": "images",
                        "stamp_ns": stamp_ns,
                        "source_sequence": source_sequence["images"],
                        "anchor": bucket["anchor"]["image"],
                        "stereo": bucket["stereo"]["image"],
                        "topics": (args.anchor_topic, args.stereo_topic),
                        "source_encodings": (
                            bucket["anchor"]["encoding"],
                            bucket["stereo"]["encoding"],
                        ),
                        "source_frame_ids": (
                            bucket["anchor"]["frame_id"],
                            bucket["stereo"]["frame_id"],
                        ),
                    }
                image_buckets.pop(stamp_ns, None)
            for old_stamp in sorted(image_buckets)[:-16]:
                image_buckets.pop(old_stamp, None)
        if ready is not None:
            sender.submit(ready)

    def depth_callback(message, index):
        try:
            depth = decode_depth(message)
        except (ValueError, TypeError) as error:
            decode_errors["depth"] += 1
            rospy.logwarn_throttle(2.0, "Depth decode failed: %s", error)
            return
        stamp_ns = message.header.stamp.to_nsec()
        ready = None
        with depth_lock:
            depth_buckets[stamp_ns][index] = {
                "depth": depth,
                "encoding": message.encoding,
                "frame_id": message.header.frame_id,
            }
            bucket = depth_buckets[stamp_ns]
            if all(index_value in bucket for index_value in range(4)):
                if rate_allows("depth", args.depth_max_hz):
                    source_sequence["depth"] += 1
                    ready = {
                        "kind": "depth",
                        "stamp_ns": stamp_ns,
                        "source_sequence": source_sequence["depth"],
                        "depths": [bucket[i]["depth"] for i in range(4)],
                        "topics": tuple(args.depth_topics),
                        "source_encodings": tuple(
                            bucket[i]["encoding"] for i in range(4)
                        ),
                        "source_frame_ids": tuple(
                            bucket[i]["frame_id"] for i in range(4)
                        ),
                    }
                depth_buckets.pop(stamp_ns, None)
            for old_stamp in sorted(depth_buckets)[:-12]:
                depth_buckets.pop(old_stamp, None)
        if ready is not None:
            sender.submit(ready)

    subscribers = [
        rospy.Subscriber(args.anchor_topic, Image, image_callback,
                         callback_args="anchor", queue_size=1,
                         buff_size=2 * 1024 * 1024, tcp_nodelay=True),
        rospy.Subscriber(args.stereo_topic, Image, image_callback,
                         callback_args="stereo", queue_size=1,
                         buff_size=2 * 1024 * 1024, tcp_nodelay=True),
    ]
    subscribers.extend(
        rospy.Subscriber(topic, Image, depth_callback, callback_args=index,
                         queue_size=1, buff_size=2 * 1024 * 1024,
                         tcp_nodelay=True)
        for index, topic in enumerate(args.depth_topics)
    )

    def publish_status(_event):
        status = sender.status()
        status["decode_errors"] = dict(decode_errors)
        status["image_topics"] = [args.anchor_topic, args.stereo_topic]
        status["depth_topics"] = list(args.depth_topics)
        status_publisher.publish(String(data=json.dumps(
            status, ensure_ascii=False, separators=(",", ":")
        )))
        rospy.loginfo_throttle(
            2.0,
            "perception connected=%s image=%.2fHz depth=%.2fHz "
            "drop=%s net=%.2fMbps error=%s",
            status["connected"], status["image_sent_hz"],
            status["depth_sent_hz"], status["dropped_pending"],
            status["sent_mbps"], status["last_error"],
        )

    timer = rospy.Timer(rospy.Duration(0.5), publish_status)
    rospy.loginfo(
        "Perception sender target=%s:%d image=%s,%s depth=%s",
        args.host, args.port, args.anchor_topic, args.stereo_topic,
        ",".join(args.depth_topics),
    )
    try:
        rospy.spin()
    finally:
        timer.shutdown()
        for subscriber in subscribers:
            subscriber.unregister()
        sender.close()


if __name__ == "__main__":
    main()
