"""Receive the Nano ``OPB1`` image/depth stream on Alienware."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import socketserver
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from interfaces.perception_stream.protocol import (
    DEFAULT_MAX_BLOCKS,
    DEFAULT_MAX_ELEMENTS,
    DEFAULT_MAX_HEADER_BYTES,
    DEFAULT_MAX_PAYLOAD_BYTES,
    DEFAULT_MAX_RAW_BYTES,
    PerceptionPacket,
    ProtocolError,
    read_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = REPO_ROOT / ".local" / "run" / "perception_receiver"


class PerceptionFrameStore:
    """Thread-safe in-process handoff for later pose/depth processing."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._images: PerceptionPacket | None = None
        self._depth: PerceptionPacket | None = None
        self._match: tuple[PerceptionPacket, PerceptionPacket, int] | None = None

    def publish(self, packet: PerceptionPacket) -> None:
        with self._condition:
            if packet.kind == "images":
                self._images = packet
            else:
                self._depth = packet
            self._condition.notify_all()

    def publish_match(
        self, images: PerceptionPacket, depth: PerceptionPacket, delta_ns: int
    ) -> None:
        with self._condition:
            self._match = (images, depth, int(delta_ns))
            self._condition.notify_all()

    def latest_images(self) -> PerceptionPacket | None:
        with self._condition:
            return self._images

    def latest_depth(self) -> PerceptionPacket | None:
        with self._condition:
            return self._depth

    def latest_match(
        self,
    ) -> tuple[PerceptionPacket, PerceptionPacket, int] | None:
        with self._condition:
            return self._match


class AtomicPerceptionWriter:
    """Atomically publish latest images, depths, and a timestamp match."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.images_path = self.directory / "latest_images.npz"
        self.depth_path = self.directory / "latest_depth.npz"
        self.match_path = self.directory / "latest_matched_observation.npz"
        self.status_path = self.directory / "status.json"
        self._lock = threading.Lock()

    @staticmethod
    def _header_json(packet: PerceptionPacket) -> np.ndarray:
        return np.asarray(json.dumps(
            packet.header, ensure_ascii=False, separators=(",", ":")
        ))

    @staticmethod
    def _atomic_npz(path: Path, **arrays: Any) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def depth_arrays(packet: PerceptionPacket) -> tuple[np.ndarray, np.ndarray]:
        source = packet.arrays["depths"]
        scale_m = float(packet.header["blocks"][0]["scale_m"])
        depth_m = source.astype(np.float32) * np.float32(scale_m)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_m[~valid] = 0.0
        return depth_m, valid

    def initialize(self, status: dict[str, Any]) -> None:
        with self._lock:
            for path in (self.images_path, self.depth_path, self.match_path):
                path.unlink(missing_ok=True)
            self._atomic_json(self.status_path, status)

    def write_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._atomic_json(self.status_path, status)

    def write_packet(self, packet: PerceptionPacket) -> None:
        with self._lock:
            common = {
                "schema": np.asarray(packet.header["schema"]),
                "kind": np.asarray(packet.kind),
                "session_id": np.asarray(packet.session_id),
                "sequence": np.asarray(packet.sequence, dtype=np.int64),
                "source_sequence": np.asarray(
                    packet.header["source_sequence"], dtype=np.int64
                ),
                "capture_timestamp_ns": np.asarray(
                    packet.capture_timestamp_ns, dtype=np.int64
                ),
                "send_timestamp_ns": np.asarray(
                    packet.header["send_timestamp_ns"], dtype=np.int64
                ),
                "header_json": self._header_json(packet),
            }
            if packet.kind == "images":
                self._atomic_npz(
                    self.images_path,
                    **common,
                    anchor_mosaic=packet.arrays["anchor_mosaic"],
                    stereo_mosaic=packet.arrays["stereo_mosaic"],
                )
            else:
                depth_m, depth_valid = self.depth_arrays(packet)
                self._atomic_npz(
                    self.depth_path,
                    **common,
                    depths_wire=packet.arrays["depths"],
                    depth_m=depth_m,
                    depth_valid=depth_valid,
                    pair_names=np.asarray(packet.header["pair_names"]),
                )

    def write_match(
        self, images: PerceptionPacket, depth: PerceptionPacket, delta_ns: int
    ) -> None:
        depth_m, depth_valid = self.depth_arrays(depth)
        with self._lock:
            self._atomic_npz(
                self.match_path,
                schema=np.asarray("omninxt.perception_match.v1"),
                session_id=np.asarray(images.session_id),
                image_sequence=np.asarray(images.sequence, dtype=np.int64),
                depth_sequence=np.asarray(depth.sequence, dtype=np.int64),
                image_timestamp_ns=np.asarray(
                    images.capture_timestamp_ns, dtype=np.int64
                ),
                depth_timestamp_ns=np.asarray(
                    depth.capture_timestamp_ns, dtype=np.int64
                ),
                depth_time_delta_ns=np.asarray(delta_ns, dtype=np.int64),
                exact_timestamp_match=np.asarray(delta_ns == 0, dtype=np.bool_),
                anchor_mosaic=images.arrays["anchor_mosaic"],
                stereo_mosaic=images.arrays["stereo_mosaic"],
                depth_m=depth_m,
                depth_valid=depth_valid,
                pair_names=np.asarray(depth.header["pair_names"]),
                image_header_json=self._header_json(images),
                depth_header_json=self._header_json(depth),
            )


class PerceptionReceiverApplication:
    """Sequence-check, cache, associate, and atomically publish packets."""

    def __init__(
        self,
        *,
        runtime_dir: str | Path | None = None,
        match_tolerance_ms: float = 120.0,
        cache_frames: int = 32,
        print_every: int = 10,
    ) -> None:
        if match_tolerance_ms < 0.0:
            raise ValueError("match_tolerance_ms must be non-negative")
        if cache_frames < 1:
            raise ValueError("cache_frames must be positive")
        self.store = PerceptionFrameStore()
        self.writer = AtomicPerceptionWriter(runtime_dir or DEFAULT_RUNTIME_DIR)
        self.match_tolerance_ns = int(match_tolerance_ms * 1_000_000.0)
        self.cache_frames = int(cache_frames)
        self.print_every = max(0, int(print_every))
        self._lock = threading.Lock()
        self._caches: dict[str, OrderedDict[int, PerceptionPacket]] = {
            "images": OrderedDict(), "depth": OrderedDict()
        }
        self._last_match_key: tuple[int, int] | None = None
        self.started_wall_ns = time.time_ns()
        self.connections = 0
        self.frames_received = 0
        self.frames_accepted = 0
        self.frames_rejected = 0
        self.frames_by_kind = {"images": 0, "depth": 0}
        self.sequence_gaps = 0
        self.matches = 0
        self.exact_matches = 0
        self.current_session_id: str | None = None
        self.last_source: str | None = None
        self.last_error: str | None = None
        self.last_receive_wall_ns: int | None = None
        self.last_sequence: int | None = None
        self.last_capture_timestamp_ns = {"images": None, "depth": None}
        self.last_source_sequence = {"images": None, "depth": None}
        self.last_match_delta_ns: int | None = None
        self.writer.initialize(self.status())

    def connected(self, address: tuple[str, int]) -> None:
        with self._lock:
            self.connections += 1
            self.last_source = f"{address[0]}:{address[1]}"
            self.last_error = None
        print(f"[PERCEPTION_RX] Nano connected from {address[0]}:{address[1]}", flush=True)
        self.writer.write_status(self.status())

    def received(self) -> None:
        with self._lock:
            self.frames_received += 1

    def rejected(self, error: Exception) -> None:
        with self._lock:
            self.frames_rejected += 1
            self.last_error = str(error)
        print(f"[PERCEPTION_RX] rejected packet: {error}", flush=True)
        self.writer.write_status(self.status())

    def _reset_session_locked(self, session_id: str) -> None:
        for cache in self._caches.values():
            cache.clear()
        self._last_match_key = None
        self.current_session_id = session_id
        self.last_sequence = None
        self.last_capture_timestamp_ns = {"images": None, "depth": None}
        self.last_source_sequence = {"images": None, "depth": None}

    def _nearest_match_locked(
        self, packet: PerceptionPacket
    ) -> tuple[PerceptionPacket, PerceptionPacket, int] | None:
        other_kind = "depth" if packet.kind == "images" else "images"
        other_cache = self._caches[other_kind]
        if not other_cache:
            return None
        nearest = min(
            other_cache.values(),
            key=lambda candidate: abs(
                candidate.capture_timestamp_ns - packet.capture_timestamp_ns
            ),
        )
        delta_ns = nearest.capture_timestamp_ns - packet.capture_timestamp_ns
        if abs(delta_ns) > self.match_tolerance_ns:
            return None
        images, depth = (
            (packet, nearest) if packet.kind == "images" else (nearest, packet)
        )
        key = (images.sequence, depth.sequence)
        if key == self._last_match_key:
            return None
        self._last_match_key = key
        # Public delta is depth time minus image time.
        return images, depth, depth.capture_timestamp_ns - images.capture_timestamp_ns

    def accept(
        self,
        packet: PerceptionPacket,
        address: tuple[str, int],
        *,
        connection_first: bool = False,
    ) -> None:
        with self._lock:
            if packet.session_id != self.current_session_id:
                self._reset_session_locked(packet.session_id)
            elif connection_first and self.last_sequence is not None:
                # Same sender session may reconnect and continue its sequence.
                pass
            if self.last_sequence is not None and packet.sequence <= self.last_sequence:
                raise ProtocolError(
                    f"non-increasing sequence {packet.sequence} after {self.last_sequence}"
                )
            kind = packet.kind
            source_sequence = int(packet.header["source_sequence"])
            previous_source = self.last_source_sequence[kind]
            previous_stamp = self.last_capture_timestamp_ns[kind]
            if previous_source is not None and source_sequence <= previous_source:
                raise ProtocolError(
                    f"non-increasing {kind} source_sequence {source_sequence}"
                )
            if previous_stamp is not None and packet.capture_timestamp_ns <= previous_stamp:
                raise ProtocolError(f"non-increasing {kind} capture timestamp")
            gap = 0 if self.last_sequence is None else max(
                0, packet.sequence - self.last_sequence - 1
            )
            cache = self._caches[kind]
            cache[packet.capture_timestamp_ns] = packet
            while len(cache) > self.cache_frames:
                cache.popitem(last=False)
            match = self._nearest_match_locked(packet)
            self.frames_accepted += 1
            self.frames_by_kind[kind] += 1
            self.sequence_gaps += gap
            self.last_source = f"{address[0]}:{address[1]}"
            self.last_receive_wall_ns = time.time_ns()
            self.last_sequence = packet.sequence
            self.last_source_sequence[kind] = source_sequence
            self.last_capture_timestamp_ns[kind] = packet.capture_timestamp_ns
            self.last_error = None
            if match is not None:
                self.matches += 1
                self.exact_matches += int(match[2] == 0)
                self.last_match_delta_ns = match[2]
            accepted = self.frames_accepted
        self.store.publish(packet)
        self.writer.write_packet(packet)
        if match is not None:
            self.store.publish_match(*match)
            self.writer.write_match(*match)
        status = self.status()
        self.writer.write_status(status)
        if self.print_every and accepted % self.print_every == 0:
            shapes = {name: list(array.shape) for name, array in packet.arrays.items()}
            print(
                "[PERCEPTION_RX] seq={} kind={} shapes={} gaps={} match_dt_ms={}".format(
                    packet.sequence,
                    packet.kind,
                    shapes,
                    status["sequence_gaps"],
                    None if status["last_match_delta_ns"] is None else round(
                        status["last_match_delta_ns"] / 1e6, 3
                    ),
                ),
                flush=True,
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "omninxt.perception_receiver.status.v1",
                "pid": os.getpid(),
                "started_wall_ns": self.started_wall_ns,
                "connections": self.connections,
                "frames_received": self.frames_received,
                "frames_accepted": self.frames_accepted,
                "frames_rejected": self.frames_rejected,
                "frames_by_kind": dict(self.frames_by_kind),
                "sequence_gaps": self.sequence_gaps,
                "matches": self.matches,
                "exact_matches": self.exact_matches,
                "current_session_id": self.current_session_id,
                "last_source": self.last_source,
                "last_error": self.last_error,
                "last_receive_wall_ns": self.last_receive_wall_ns,
                "last_sequence": self.last_sequence,
                "last_capture_timestamp_ns": dict(self.last_capture_timestamp_ns),
                "last_match_delta_ns": self.last_match_delta_ns,
                "match_tolerance_ns": self.match_tolerance_ns,
                "latest_images": str(self.writer.images_path),
                "latest_depth": str(self.writer.depth_path),
                "latest_match": str(self.writer.match_path),
            }


class _PerceptionRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        application: PerceptionReceiverApplication = self.server.application  # type: ignore[attr-defined]
        limits: dict[str, int] = self.server.protocol_limits  # type: ignore[attr-defined]
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        application.connected(self.client_address)
        connection_first = True
        while True:
            try:
                packet = read_packet(self.rfile, **limits)
            except EOFError:
                return
            except (ProtocolError, ValueError, TypeError, MemoryError) as error:
                application.rejected(error)
                # A length/framing error cannot be resynchronized safely.
                return
            application.received()
            try:
                application.accept(
                    packet, self.client_address, connection_first=connection_first
                )
                connection_first = False
            except (ProtocolError, ValueError, TypeError) as error:
                application.rejected(error)


class PerceptionReceiverServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 4
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        address: tuple[str, int],
        application: PerceptionReceiverApplication,
        **protocol_limits: int,
    ) -> None:
        self.application = application
        self.protocol_limits = protocol_limits
        super().__init__(address, _PerceptionRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive Nano OPB1 pose mosaics and four dense depth maps"
    )
    parser.add_argument(
        "--host", default=os.getenv("PERCEPTION_RECEIVER_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("PERCEPTION_RECEIVER_PORT", "9766"))
    )
    parser.add_argument(
        "--runtime-dir",
        default=os.getenv("PERCEPTION_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)),
    )
    parser.add_argument(
        "--match-tolerance-ms", type=float,
        default=float(os.getenv("PERCEPTION_MATCH_TOLERANCE_MS", "120")),
    )
    parser.add_argument("--cache-frames", type=int, default=32)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--max-header-bytes", type=int, default=DEFAULT_MAX_HEADER_BYTES)
    parser.add_argument("--max-payload-bytes", type=int, default=DEFAULT_MAX_PAYLOAD_BYTES)
    parser.add_argument("--max-raw-bytes", type=int, default=DEFAULT_MAX_RAW_BYTES)
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--max-elements", type=int, default=DEFAULT_MAX_ELEMENTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1,65535]")
    application = PerceptionReceiverApplication(
        runtime_dir=args.runtime_dir,
        match_tolerance_ms=args.match_tolerance_ms,
        cache_frames=args.cache_frames,
        print_every=args.print_every,
    )
    server = PerceptionReceiverServer(
        (args.host, args.port),
        application,
        max_header_bytes=args.max_header_bytes,
        max_payload_bytes=args.max_payload_bytes,
        max_raw_bytes=args.max_raw_bytes,
        max_blocks=args.max_blocks,
        max_elements=args.max_elements,
    )

    def stop(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    host, port = server.server_address
    print(
        f"[PERCEPTION_RX] listening on {host}:{port}; "
        f"runtime snapshots: {application.writer.directory}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        print("[PERCEPTION_RX] stopped", flush=True)


if __name__ == "__main__":
    main()
