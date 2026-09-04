"""Threaded newline-delimited JSON receiver for 3D skeleton packets."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque


class SkeletonPacketReceiver:
    def __init__(self, host="127.0.0.1", port=9765, buffer_size=256,
                 max_line_bytes=4 * 1024 * 1024):
        self.host = str(host)
        self.port = int(port)
        self.max_line_bytes = int(max_line_bytes)
        self._packets = deque(maxlen=max(1, int(buffer_size)))
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._server = None
        self._thread = None
        self._last_error = None
        self._received = 0
        self._rejected = 0
        self._connections = 0
        self._session_id = None
        self._session_resets = 0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, name="SkeletonPacketReceiver", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            raise RuntimeError("timed out starting skeleton receiver")
        if not self._thread.is_alive():
            raise RuntimeError(
                "failed to start skeleton receiver: {}".format(self._last_error))

    def stop(self):
        self._stop.set()
        server = self._server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def packets_after(self, arrival_index):
        with self._condition:
            return [dict(item) for item in self._packets
                    if int(item["arrival_index"]) > int(arrival_index)]

    def latest(self):
        with self._condition:
            return None if not self._packets else dict(self._packets[-1])

    def status(self):
        with self._condition:
            return {
                "host": self.host,
                "port": self.port,
                "received": self._received,
                "rejected": self._rejected,
                "connections": self._connections,
                "session_id": self._session_id,
                "session_resets": self._session_resets,
                "buffered": len(self._packets),
                "last_error": self._last_error,
                "running": bool(self._thread and self._thread.is_alive()),
            }

    @staticmethod
    def _validate(packet):
        if packet.get("schema") != "omninxt.skeleton3d.v1":
            raise ValueError("unsupported skeleton schema")
        if not isinstance(packet.get("sequence"), int):
            raise ValueError("missing integer sequence")
        if not isinstance(packet.get("timestamp_ns"), int):
            raise ValueError("missing integer timestamp_ns")
        if packet.get("frame_id") != "base_link":
            raise ValueError("skeleton frame must be base_link")
        if len(packet.get("joint_names", ())) != 17:
            raise ValueError("skeleton must contain COCO17 joint names")
        return packet

    def _run(self):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(0.5)
            server.bind((self.host, self.port))
            self.port = int(server.getsockname()[1])
            server.listen(2)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with self._condition:
                    self._connections += 1
                connection.settimeout(0.5)
                with connection:
                    buffer = bytearray()
                    while not self._stop.is_set():
                        try:
                            chunk = connection.recv(64 * 1024)
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        if len(buffer) > self.max_line_bytes and b"\n" not in buffer:
                            with self._condition:
                                self._rejected += 1
                                self._last_error = "skeleton packet exceeds size limit"
                            break
                        while b"\n" in buffer:
                            line, _, remainder = buffer.partition(b"\n")
                            buffer = bytearray(remainder)
                            if len(line) > self.max_line_bytes:
                                with self._condition:
                                    self._rejected += 1
                                    self._last_error = "skeleton packet exceeds size limit"
                                continue
                            try:
                                packet = self._validate(json.loads(line))
                            except (ValueError, json.JSONDecodeError) as error:
                                with self._condition:
                                    self._rejected += 1
                                    self._last_error = str(error)
                                continue
                            with self._condition:
                                session_id = packet.get("session_id")
                                if (session_id is not None and
                                        self._session_id is not None and
                                        str(session_id) != self._session_id):
                                    # Never let a newly started recorder or a
                                    # temporal policy window consume packets
                                    # from the previous hot-reset episode.
                                    self._packets.clear()
                                    self._session_resets += 1
                                if session_id is not None:
                                    self._session_id = str(session_id)
                                self._received += 1
                                item = {
                                    "packet": packet,
                                    "received_wall_time_ns": time.time_ns(),
                                    "arrival_index": self._received,
                                }
                                self._packets.append(item)
                                self._last_error = None
                                self._condition.notify_all()
        except OSError as error:
            with self._condition:
                self._last_error = str(error)
            self._ready.set()
        finally:
            self._server = None
            self._ready.set()
