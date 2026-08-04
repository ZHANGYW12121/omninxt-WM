from __future__ import annotations

import json
import socket
import struct
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path

import numpy as np

from backend.perception_receiver.server import (
    PerceptionReceiverApplication,
    PerceptionReceiverServer,
)
from interfaces.perception_stream.protocol import (
    DEPTH_SCHEMA,
    IMAGE_SCHEMA,
    MAGIC,
    PREFIX,
    ProtocolError,
    decode_packet,
)


PAIR_NAMES = ["AB_RIGHT", "BC_REAR", "CD_LEFT", "DA_FRONT"]


def nano_packet(
    kind: str,
    sequence: int,
    timestamp_ns: int,
    *,
    session_id: str = "test-session",
    source_sequence: int = 1,
    depth_dtype: str = "float32",
    compression: str = "zlib",
) -> bytes:
    if kind == "images":
        specs = [
            ("anchor_mosaic", np.arange(48, dtype=np.uint8).reshape(6, 8), {
                "topic": "/anchor", "source_encoding": "mono8",
                "frame_id": "anchor", "units": "intensity_u8",
            }),
            ("stereo_mosaic", np.arange(60, dtype=np.uint8).reshape(10, 6), {
                "topic": "/stereo", "source_encoding": "mono8",
                "frame_id": "stereo", "units": "intensity_u8",
            }),
        ]
        schema = IMAGE_SCHEMA
        extra = {"anchor_names": ["A", "B", "C", "D"], "pair_names": PAIR_NAMES}
    else:
        source = np.linspace(0.5, 5.0, 4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
        if depth_dtype == "uint16_mm":
            source = np.rint(source * 1000.0).astype("<u2")
            scale_m = 0.001
        else:
            source = source.astype("<f4")
            source[0, 0, 0] = np.nan
            scale_m = 1.0
        specs = [("depths", source, {
            "topics": ["/d0", "/d1", "/d2", "/d3"],
            "source_encodings": ["32FC1"] * 4,
            "frame_ids": ["d0", "d1", "d2", "d3"],
            "units": "metre", "scale_m": scale_m,
            "invalid_semantics": "preserved", "layout": "pair-major",
        })]
        schema = DEPTH_SCHEMA
        extra = {"pair_names": PAIR_NAMES}

    blocks = []
    raw_parts = []
    offset = 0
    for name, array, metadata in specs:
        raw = array.tobytes(order="C")
        block = dict(metadata)
        block.update({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": len(raw),
        })
        blocks.append(block)
        raw_parts.append(raw)
        offset += len(raw)
    raw_payload = b"".join(raw_parts)
    payload = zlib.compress(raw_payload, 1) if compression == "zlib" else raw_payload
    header = {
        "schema": schema,
        "kind": kind,
        "session_id": session_id,
        "sequence": sequence,
        "source_sequence": source_sequence,
        "capture_timestamp_ns": timestamp_ns,
        "send_timestamp_ns": timestamp_ns + 1_000_000,
        "blocks": blocks,
        "compression": compression,
        "compression_level": 1 if compression == "zlib" else 0,
        "raw_bytes": len(raw_payload),
        "payload_bytes": len(payload),
        "raw_crc32": "{:08x}".format(zlib.crc32(raw_payload) & 0xFFFFFFFF),
    }
    header.update(extra)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return PREFIX.pack(MAGIC, len(header_bytes), len(payload)) + header_bytes + payload


def decode_wire(packet: bytes):
    magic, header_length, payload_length = PREFIX.unpack(packet[:PREFIX.size])
    self_prefix = PREFIX.pack(magic, header_length, payload_length)
    header_start = PREFIX.size
    payload_start = header_start + header_length
    return decode_packet(
        self_prefix,
        packet[header_start:payload_start],
        packet[payload_start:payload_start + payload_length],
    )


class PerceptionProtocolTest(unittest.TestCase):
    def test_decodes_exact_nano_image_and_float_depth_layout(self):
        images = decode_wire(nano_packet("images", 1, 1_000_000_000))
        depth = decode_wire(nano_packet("depth", 2, 1_050_000_000))
        self.assertEqual(images.kind, "images")
        self.assertEqual(images.arrays["anchor_mosaic"].shape, (6, 8))
        self.assertEqual(images.arrays["stereo_mosaic"].shape, (10, 6))
        self.assertEqual(depth.kind, "depth")
        self.assertEqual(depth.arrays["depths"].shape, (4, 3, 5))
        self.assertTrue(np.isnan(depth.arrays["depths"][0, 0, 0]))

    def test_decodes_uncompressed_uint16_depth(self):
        depth = decode_wire(nano_packet(
            "depth", 1, 1_000_000_000,
            depth_dtype="uint16_mm", compression="none",
        ))
        self.assertEqual(depth.arrays["depths"].dtype, np.dtype("<u2"))
        self.assertEqual(depth.header["blocks"][0]["scale_m"], 0.001)

    def test_rejects_corrupt_crc_and_oversized_prefix(self):
        wire = bytearray(nano_packet(
            "images", 1, 1_000_000_000, compression="none"
        ))
        wire[-1] ^= 0x01
        with self.assertRaisesRegex(ProtocolError, "CRC32"):
            decode_wire(bytes(wire))
        bad_prefix = struct.pack("!4sII", MAGIC, 64 * 1024 + 1, 10)
        with self.assertRaisesRegex(ProtocolError, "header length"):
            decode_packet(bad_prefix, b"", b"")


class PerceptionReceiverTest(unittest.TestCase):
    def _start(self, runtime: Path):
        application = PerceptionReceiverApplication(
            runtime_dir=runtime, match_tolerance_ms=120, print_every=0
        )
        server = PerceptionReceiverServer(
            ("127.0.0.1", 0), application,
            max_header_bytes=64 * 1024,
            max_payload_bytes=16 * 1024 * 1024,
            max_raw_bytes=64 * 1024 * 1024,
            max_blocks=8,
            max_elements=32 * 1024 * 1024,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return application, server, thread

    def test_fragmented_multiplex_stream_writes_three_atomic_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            application, server, thread = self._start(runtime)
            try:
                images = nano_packet("images", 1, 1_000_000_000, source_sequence=1)
                depth = nano_packet("depth", 2, 1_050_000_000, source_sequence=1)
                with socket.create_connection(server.server_address, timeout=2.0) as sock:
                    stream = images + depth
                    for offset in range(0, len(stream), 7):
                        sock.sendall(stream[offset:offset + 7])
                deadline = time.monotonic() + 2.0
                while application.frames_accepted < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(application.frames_accepted, 2)
                self.assertEqual(application.matches, 1)
                self.assertEqual(application.sequence_gaps, 0)
                for name in (
                    "latest_images.npz", "latest_depth.npz",
                    "latest_matched_observation.npz", "status.json",
                ):
                    self.assertTrue((runtime / name).is_file(), name)
                with np.load(runtime / "latest_depth.npz", allow_pickle=False) as data:
                    self.assertEqual(data["depth_m"].shape, (4, 3, 5))
                    self.assertFalse(data["depth_valid"][0, 0, 0])
                    self.assertEqual(float(data["depth_m"][0, 0, 0]), 0.0)
                with np.load(
                    runtime / "latest_matched_observation.npz", allow_pickle=False
                ) as data:
                    self.assertEqual(int(data["depth_time_delta_ns"]), 50_000_000)
                    self.assertFalse(bool(data["exact_timestamp_match"]))
                    self.assertEqual(data["anchor_mosaic"].shape, (6, 8))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

    def test_shutdown_does_not_wait_for_connected_nano(self):
        with tempfile.TemporaryDirectory() as directory:
            _, server, thread = self._start(Path(directory))
            connection = socket.create_connection(server.server_address, timeout=2.0)
            try:
                deadline = time.monotonic() + 1.0
                while threading.active_count() < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                started = time.monotonic()
                server.shutdown()
                server.server_close()
                thread.join(timeout=1.0)
                self.assertLess(time.monotonic() - started, 0.8)
                self.assertFalse(thread.is_alive())
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
