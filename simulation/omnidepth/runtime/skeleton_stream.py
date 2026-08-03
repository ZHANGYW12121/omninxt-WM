#!/usr/bin/env python3
"""Fixed-schema 3D skeleton transport and ST-GCN tensor assembly.

The real-time perception thread must never wait for a remote backend.  The TCP
sender therefore runs in a daemon thread and retains only the newest frame
while disconnected or back-pressured.
"""

import json
import math
import queue
import socket
import threading
import time
from collections import deque

import numpy as np


SCHEMA = "omninxt.skeleton3d.v1"
JOINT_FIELDS = (
    "x_m", "y_m", "z_m", "pose_score", "confidence",
    "coordinate_valid", "measured", "predicted",
    "measurement_sigma_m", "measurement_age_ms", "source_code",
)
SOURCE_CODES = {
    "invalid": 0,
    "stereo_geometry": 1,
    "hitnet": 2,
    "temporal_prediction": 3,
    "other": 4,
    "isaac_gt_depth": 5,
}
STGCN_FEATURES = ("x_m", "y_m", "z_m", "confidence", "coordinate_valid")


def _finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _source_code(source, predicted):
    if predicted:
        return SOURCE_CODES["temporal_prediction"]
    source = str(source or "invalid")
    if source == "anchor_epipolar" or source.startswith("stereo_"):
        return SOURCE_CODES["stereo_geometry"]
    if source.startswith("hitnet_"):
        return SOURCE_CODES["hitnet"]
    if source.startswith("isaac_gt_depth"):
        return SOURCE_CODES["isaac_gt_depth"]
    if source.startswith("invalid"):
        return SOURCE_CODES["invalid"]
    return SOURCE_CODES["other"]


def _joint_row(joint):
    xyz = joint.get("xyz_base_link_m", joint.get("xyz_imu_m"))
    valid = xyz is not None and len(xyz) == 3 and all(
        math.isfinite(float(value)) for value in xyz)
    predicted = bool(joint.get("predicted", False)) and valid
    measured = valid and not predicted
    if valid:
        coordinates = [_finite(value) for value in xyz]
    else:
        # A fixed numeric tensor is required by ST-GCN.  Zero is never treated
        # as a measurement because coordinate_valid is carried separately.
        coordinates = [0.0, 0.0, 0.0]
    if predicted:
        pose_score = _finite(joint.get(
            "last_measurement_score", joint.get("score", 0.0)))
        sigma = _finite(joint.get(
            "last_measurement_sigma_m", joint.get("measurement_sigma_m", 0.0)))
        age_ms = max(0.0, _finite(joint.get("measurement_age_ms", 0.0)))
        source = joint.get("measurement_source", joint.get("source"))
        source_weight = 0.45 * math.exp(-age_ms / 350.0)
    else:
        pose_score = _finite(joint.get("score", 0.0))
        sigma = _finite(joint.get("measurement_sigma_m", 0.0))
        age_ms = max(0.0, _finite(joint.get("measurement_age_ms", 0.0)))
        source = joint.get("source")
        code = _source_code(source, False)
        source_weight = 1.0 if code in (
            SOURCE_CODES["stereo_geometry"], SOURCE_CODES["isaac_gt_depth"]
        ) else \
            0.55 if code == SOURCE_CODES["hitnet"] else 0.35
    if not valid:
        confidence = 0.0
    else:
        confidence = float(np.clip(
            pose_score * source_weight / (1.0 + max(0.0, sigma)), 0.0, 1.0))
    return [
        round(coordinates[0], 6), round(coordinates[1], 6),
        round(coordinates[2], 6), round(pose_score, 6),
        round(confidence, 6), int(valid), int(measured), int(predicted),
        round(max(0.0, sigma), 6), round(age_ms, 3),
        _source_code(source, predicted),
    ]


def build_skeleton_packet(people, stamp_ns, sequence, joint_names,
                          frame_id="base_link"):
    """Create one self-describing packet with exactly 17 joints per person."""
    encoded_people = []
    joint_count = len(joint_names)
    for person in sorted(people, key=lambda value: int(value["person_id"])):
        by_id = {int(joint["id"]): joint for joint in person.get("joints", [])
                 if "id" in joint}
        joints = []
        for joint_id in range(joint_count):
            joint = by_id.get(joint_id, {
                "id": joint_id, "name": joint_names[joint_id],
                "score": 0.0, "source": "invalid", "xyz_imu_m": None,
            })
            joints.append(_joint_row(joint))
        encoded_people.append({
            "person_id": int(person["person_id"]),
            "joints": joints,
        })
    return {
        "schema": SCHEMA,
        "sequence": int(sequence),
        "timestamp_ns": int(stamp_ns),
        "frame_id": frame_id,
        "coordinate_convention": "+X forward, +Y left, +Z up",
        "units": "metre",
        "joint_names": list(joint_names),
        "joint_fields": list(JOINT_FIELDS),
        "source_codes": dict(SOURCE_CODES),
        "people": encoded_people,
    }


def validate_packet(packet, expected_joints=17):
    if packet.get("schema") != SCHEMA:
        raise ValueError("Unsupported skeleton schema: {}".format(
            packet.get("schema")))
    if len(packet.get("joint_names", [])) != expected_joints:
        raise ValueError("Expected {} joint names".format(expected_joints))
    if tuple(packet.get("joint_fields", [])) != JOINT_FIELDS:
        raise ValueError("Unexpected joint field layout")
    for person in packet.get("people", []):
        joints = person.get("joints", [])
        if len(joints) != expected_joints:
            raise ValueError("Person {} has {} joints, expected {}".format(
                person.get("person_id"), len(joints), expected_joints))
        if any(len(row) != len(JOINT_FIELDS) for row in joints):
            raise ValueError("Unexpected joint row width")
    return packet


class SkeletonTcpSender:
    """Non-blocking latest-frame TCP client with automatic reconnection."""

    def __init__(self, host, port, connect_timeout=0.35, send_timeout=0.20):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self.send_timeout = float(send_timeout)
        self.frames = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.sent = 0
        self.dropped = 0
        self.connected = False
        self.last_error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, packet):
        try:
            self.frames.get_nowait()
            with self.lock:
                self.dropped += 1
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(packet)
        except queue.Full:
            with self.lock:
                self.dropped += 1

    def status(self):
        with self.lock:
            return {
                "enabled": True, "host": self.host, "port": self.port,
                "connected": self.connected, "sent": self.sent,
                "dropped": self.dropped, "last_error": self.last_error,
            }

    def _set_connection(self, connected, error=None):
        with self.lock:
            self.connected = bool(connected)
            self.last_error = error

    def _run(self):
        connection = None
        while True:
            try:
                packet = self.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            if connection is None:
                try:
                    connection = socket.create_connection(
                        (self.host, self.port), timeout=self.connect_timeout)
                    connection.settimeout(self.send_timeout)
                    connection.setsockopt(socket.IPPROTO_TCP,
                                          socket.TCP_NODELAY, 1)
                    self._set_connection(True, None)
                except OSError as error:
                    self._set_connection(False, str(error))
                    time.sleep(0.15)
                    continue
            payload = (json.dumps(packet, ensure_ascii=False,
                                  separators=(",", ":")) + "\n").encode("utf-8")
            try:
                connection.sendall(payload)
                with self.lock:
                    self.sent += 1
            except OSError as error:
                try:
                    connection.close()
                except OSError:
                    pass
                connection = None
                self._set_connection(False, str(error))


class StgcnWindow:
    """Convert packets into [N,C,T,V,M] tensors with stable person slots."""

    def __init__(self, window=30, max_people=4):
        self.window = int(window)
        self.max_people = int(max_people)
        self.frames = deque(maxlen=self.window)
        self.slot_ids = [None] * self.max_people
        self.last_seen = {}

    def _slot_for(self, person_id, sequence):
        if person_id in self.slot_ids:
            slot = self.slot_ids.index(person_id)
        elif None in self.slot_ids:
            slot = self.slot_ids.index(None)
            self.slot_ids[slot] = person_id
        else:
            slot = min(range(self.max_people),
                       key=lambda index: self.last_seen.get(
                           self.slot_ids[index], -1))
            self.last_seen.pop(self.slot_ids[slot], None)
            self.slot_ids[slot] = person_id
        self.last_seen[person_id] = sequence
        return slot

    def push(self, packet):
        validate_packet(packet)
        sequence = int(packet["sequence"])
        frame = np.zeros((len(STGCN_FEATURES), 17, self.max_people),
                         dtype=np.float32)
        field = {name: index for index, name in enumerate(JOINT_FIELDS)}
        for person in packet.get("people", []):
            slot = self._slot_for(int(person["person_id"]), sequence)
            rows = np.asarray(person["joints"], dtype=np.float32)
            for channel, name in enumerate(STGCN_FEATURES):
                frame[channel, :, slot] = rows[:, field[name]]
        self.frames.append(frame)
        if len(self.frames) < self.window:
            return None
        # Each frame is [C,V,M]; stack time at axis 1 and add batch N.
        return np.stack(tuple(self.frames), axis=1)[None, ...]
