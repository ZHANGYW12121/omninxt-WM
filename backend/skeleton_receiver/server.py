"""Alienware TCP service for Nano ``omninxt.skeleton3d.v1`` frames."""

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
from pathlib import Path
from typing import Any

import numpy as np

from interfaces.skeleton3d.protocol import (
    DEFAULT_MAX_PACKET_BYTES,
    ProtocolError,
    decode_packet_line,
)

from .adapter import NanoHumanObservationAdapter, WorldModelHumanFrame


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = REPO_ROOT / ".local" / "run" / "skeleton_receiver"


class SkeletonFrameStore:
    """Thread-safe latest-frame handoff for an in-process model runner."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: WorldModelHumanFrame | None = None

    def publish(self, frame: WorldModelHumanFrame) -> None:
        with self._condition:
            self._latest = frame
            self._condition.notify_all()

    def latest(self) -> WorldModelHumanFrame | None:
        with self._condition:
            return self._latest

    def wait_for_sequence(
        self, after_sequence: int | None = None, timeout: float | None = None,
    ) -> WorldModelHumanFrame | None:
        """Wait until a frame newer than ``after_sequence`` is available."""

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._latest is None or (
                after_sequence is not None
                and self._latest.sequence <= int(after_sequence)
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._latest


class AtomicSnapshotWriter:
    """Publish the latest model-ready arrays without exposing partial files."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frame_path = self.directory / "latest_human_observation.npz"
        self.status_path = self.directory / "status.json"

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

    def write(self, frame: WorldModelHumanFrame, status: dict[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".latest_human_observation.", suffix=".npz", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(
                    stream,
                    schema=np.asarray(frame.schema),
                    sequence=np.asarray(frame.sequence, dtype=np.int64),
                    timestamp_ns=np.asarray(frame.timestamp_ns, dtype=np.int64),
                    skeleton=frame.skeleton,
                    skeleton_raw=frame.skeleton_raw,
                    human_root=frame.human_root,
                    human_joints=frame.human_joints,
                    human_mask=frame.human_mask,
                    joint_mask=frame.joint_mask,
                    joint_inferred_mask=frame.joint_inferred_mask,
                    human_ids=frame.human_ids,
                    human_is_first=frame.human_is_first,
                    joint_uncertainty_m=frame.joint_uncertainty_m,
                    joint_prediction_age_ms=frame.joint_prediction_age_ms,
                    truncated_people=np.asarray(frame.truncated_people, dtype=np.int32),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.frame_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        self._atomic_json(self.status_path, status)

    def initialize(self, status: dict[str, Any]) -> None:
        """Start a new receiver session without exposing a stale model frame."""

        self.frame_path.unlink(missing_ok=True)
        self._atomic_json(self.status_path, status)

    def write_status(self, status: dict[str, Any]) -> None:
        self._atomic_json(self.status_path, status)


class SkeletonReceiverApplication:
    """Validate, sequence-check, adapt, and publish incoming frames."""

    def __init__(
        self, *, max_people: int = 20, runtime_dir: str | Path | None = None,
        print_every: int = 10,
    ) -> None:
        self.adapter = NanoHumanObservationAdapter(max_people=max_people)
        self.store = SkeletonFrameStore()
        self.writer = AtomicSnapshotWriter(runtime_dir or DEFAULT_RUNTIME_DIR)
        self.print_every = max(0, int(print_every))
        self._lock = threading.Lock()
        self.started_wall_ns = time.time_ns()
        self.connections = 0
        self.frames_received = 0
        self.frames_accepted = 0
        self.frames_rejected = 0
        self.sequence_gaps = 0
        self.last_source: str | None = None
        self.last_error: str | None = None
        self.last_receive_wall_ns: int | None = None
        self.last_packet_timestamp_ns: int | None = None
        self.last_sequence: int | None = None
        self.writer.initialize(self.status())

    def connected(self, address: tuple[str, int]) -> None:
        with self._lock:
            self.connections += 1
            self.last_source = f"{address[0]}:{address[1]}"
            self.last_error = None
        print(f"[SKELETON_RX] Nano connected from {address[0]}:{address[1]}", flush=True)
        self.writer.write_status(self.status())

    def received(self) -> None:
        with self._lock:
            self.frames_received += 1

    def rejected(self, error: Exception) -> None:
        with self._lock:
            self.frames_rejected += 1
            self.last_error = str(error)
        print(f"[SKELETON_RX] rejected packet: {error}", flush=True)
        self.writer.write_status(self.status())

    def accept(
        self, packet: dict[str, Any], address: tuple[str, int], sequence_gap: int,
        *, connection_first: bool = False,
    ) -> WorldModelHumanFrame:
        timestamp_ns = int(packet["timestamp_ns"])
        sequence = int(packet["sequence"])
        with self._lock:
            source_restarted = (
                connection_first
                and self.last_sequence is not None
                and sequence <= self.last_sequence
            )
            if (
                not source_restarted
                and self.last_packet_timestamp_ns is not None
                and timestamp_ns <= self.last_packet_timestamp_ns
            ):
                raise ProtocolError(
                    "stale packet timestamp {} after {}".format(
                        timestamp_ns, self.last_packet_timestamp_ns
                    )
                )
        if source_restarted:
            self.adapter.reset()
        frame = self.adapter.adapt(packet)
        now_ns = time.time_ns()
        with self._lock:
            self.frames_accepted += 1
            self.sequence_gaps += max(0, int(sequence_gap))
            self.last_source = f"{address[0]}:{address[1]}"
            self.last_receive_wall_ns = now_ns
            self.last_packet_timestamp_ns = frame.timestamp_ns
            self.last_sequence = frame.sequence
            self.last_error = None
            accepted = self.frames_accepted
        self.store.publish(frame)
        status = self.status()
        self.writer.write(frame, status)
        if self.print_every and accepted % self.print_every == 0:
            print(
                "[SKELETON_RX] seq={} people={} valid_joints={} gaps={} output={}".format(
                    frame.sequence,
                    int(frame.human_mask.sum()),
                    int(frame.joint_mask.sum()),
                    status["sequence_gaps"],
                    self.writer.frame_path,
                ),
                flush=True,
            )
        return frame

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "omninxt.skeleton_receiver.status.v1",
                "pid": os.getpid(),
                "started_wall_ns": self.started_wall_ns,
                "connections": self.connections,
                "frames_received": self.frames_received,
                "frames_accepted": self.frames_accepted,
                "frames_rejected": self.frames_rejected,
                "sequence_gaps": self.sequence_gaps,
                "last_source": self.last_source,
                "last_error": self.last_error,
                "last_receive_wall_ns": self.last_receive_wall_ns,
                "last_packet_timestamp_ns": self.last_packet_timestamp_ns,
                "last_sequence": self.last_sequence,
                "snapshot": str(self.writer.frame_path),
            }


class _SkeletonRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        application: SkeletonReceiverApplication = self.server.application  # type: ignore[attr-defined]
        maximum: int = self.server.max_packet_bytes  # type: ignore[attr-defined]
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        application.connected(self.client_address)
        last_sequence: int | None = None
        while True:
            line = self.rfile.readline(maximum + 1)
            if not line:
                return
            application.received()
            if len(line) > maximum:
                application.rejected(ProtocolError("packet exceeds maximum line size"))
                return
            try:
                packet = decode_packet_line(line, max_packet_bytes=maximum)
                sequence = int(packet["sequence"])
                if last_sequence is not None and sequence <= last_sequence:
                    raise ProtocolError(
                        f"non-increasing sequence {sequence} after {last_sequence}"
                    )
                gap = 0 if last_sequence is None else max(0, sequence - last_sequence - 1)
                application.accept(
                    packet, self.client_address, gap,
                    connection_first=last_sequence is None,
                )
                last_sequence = sequence
            except (ProtocolError, ValueError, TypeError) as error:
                application.rejected(error)


class SkeletonReceiverServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Single-active-producer TCP server; Nano reconnects automatically."""

    allow_reuse_address = True
    request_queue_size = 4
    daemon_threads = True
    block_on_close = False

    def __init__(
        self, address: tuple[str, int], application: SkeletonReceiverApplication,
        *, max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
    ) -> None:
        self.application = application
        self.max_packet_bytes = int(max_packet_bytes)
        super().__init__(address, _SkeletonRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive Nano 3-D skeletons and publish model-ready tensors"
    )
    parser.add_argument("--host", default=os.getenv("SKELETON_RECEIVER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("SKELETON_RECEIVER_PORT", "9765"))
    )
    parser.add_argument(
        "--max-people", type=int,
        default=int(os.getenv("SKELETON_MAX_PEOPLE", "20")),
    )
    parser.add_argument(
        "--runtime-dir",
        default=os.getenv("SKELETON_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)),
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--max-packet-bytes", type=int, default=DEFAULT_MAX_PACKET_BYTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    application = SkeletonReceiverApplication(
        max_people=args.max_people,
        runtime_dir=args.runtime_dir,
        print_every=args.print_every,
    )
    server = SkeletonReceiverServer(
        (args.host, args.port), application, max_packet_bytes=args.max_packet_bytes
    )

    def stop(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    host, port = server.server_address
    print(
        f"[SKELETON_RX] listening on {host}:{port}; "
        f"model snapshot: {application.writer.frame_path}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        print("[SKELETON_RX] stopped", flush=True)


if __name__ == "__main__":
    main()
