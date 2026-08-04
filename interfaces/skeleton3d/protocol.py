"""Canonical validator for ``omninxt.skeleton3d.v1`` TCP packets.

The Nano sends one compact UTF-8 JSON object per line.  Keeping validation in
``interfaces`` lets receivers depend on the protocol rather than importing
Jetson runtime implementation code.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


SCHEMA = "omninxt.skeleton3d.v1"
FRAME_ID = "base_link"
COORDINATE_CONVENTION = "+X forward, +Y left, +Z up"
UNITS = "metre"
COCO17_JOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
JOINT_FIELDS = (
    "x_m", "y_m", "z_m", "pose_score", "confidence",
    "coordinate_valid", "measured", "predicted",
    "measurement_sigma_m", "measurement_age_ms", "source_code",
)
FLAG_FIELD_INDICES = (5, 6, 7)
SOURCE_CODE_INDEX = 10
DEFAULT_MAX_PACKET_BYTES = 2 * 1024 * 1024


class ProtocolError(ValueError):
    """Raised when a network frame violates the versioned contract."""


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def validate_packet(packet: Any, *, max_people: int | None = None) -> dict[str, Any]:
    """Validate a decoded packet and return it unchanged.

    The function intentionally accepts additional top-level fields so a
    backwards-compatible producer can add diagnostics without changing the
    tensor layout.  Contract-defining fields remain strict.
    """

    if not isinstance(packet, dict):
        raise ProtocolError("packet must be a JSON object")
    if packet.get("schema") != SCHEMA:
        raise ProtocolError(f"unsupported schema: {packet.get('schema')!r}")
    _require_int(packet.get("sequence"), "sequence")
    _require_int(packet.get("timestamp_ns"), "timestamp_ns", minimum=1)
    if packet.get("frame_id") != FRAME_ID:
        raise ProtocolError(f"frame_id must be {FRAME_ID!r}")
    if packet.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise ProtocolError("unexpected coordinate convention")
    if packet.get("units") != UNITS:
        raise ProtocolError(f"units must be {UNITS!r}")
    if tuple(packet.get("joint_names", ())) != COCO17_JOINT_NAMES:
        raise ProtocolError("joint_names must use canonical COCO-17 order")
    if tuple(packet.get("joint_fields", ())) != JOINT_FIELDS:
        raise ProtocolError("unexpected joint field layout")

    source_codes = packet.get("source_codes")
    if not isinstance(source_codes, Mapping) or not source_codes:
        raise ProtocolError("source_codes must be a non-empty object")
    allowed_source_codes = set()
    for name, value in source_codes.items():
        if not isinstance(name, str):
            raise ProtocolError("source code names must be strings")
        allowed_source_codes.add(_require_int(value, f"source_codes.{name}"))

    people = packet.get("people")
    if not isinstance(people, list):
        raise ProtocolError("people must be a list")
    if max_people is not None and len(people) > int(max_people):
        raise ProtocolError(f"packet contains more than {int(max_people)} people")
    seen_ids: set[int] = set()
    for person_index, person in enumerate(people):
        if not isinstance(person, dict):
            raise ProtocolError(f"people[{person_index}] must be an object")
        person_id = _require_int(
            person.get("person_id"), f"people[{person_index}].person_id", minimum=0
        )
        if person_id in seen_ids:
            raise ProtocolError(f"duplicate person_id {person_id}")
        seen_ids.add(person_id)
        joints = person.get("joints")
        if not isinstance(joints, list) or len(joints) != len(COCO17_JOINT_NAMES):
            raise ProtocolError(f"person {person_id} must contain 17 joints")
        for joint_index, row in enumerate(joints):
            label = f"person {person_id} joint {joint_index}"
            if not isinstance(row, list) or len(row) != len(JOINT_FIELDS):
                raise ProtocolError(f"{label} has an invalid row width")
            for field_index, value in enumerate(row):
                _require_finite_number(value, f"{label}.{JOINT_FIELDS[field_index]}")
            for field_index in FLAG_FIELD_INDICES:
                if row[field_index] not in (0, 1):
                    raise ProtocolError(
                        f"{label}.{JOINT_FIELDS[field_index]} must be 0 or 1"
                    )
            source_code = row[SOURCE_CODE_INDEX]
            if isinstance(source_code, bool) or int(source_code) != source_code:
                raise ProtocolError(f"{label}.source_code must be an integer")
            if int(source_code) not in allowed_source_codes:
                raise ProtocolError(f"{label}.source_code is not declared")
    return packet


def decode_packet_line(
    line: bytes, *, max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
    max_people: int | None = None,
) -> dict[str, Any]:
    """Decode and validate one newline-delimited JSON frame."""

    if not line:
        raise ProtocolError("empty packet")
    if len(line) > int(max_packet_bytes):
        raise ProtocolError("packet exceeds maximum line size")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError("packet is not valid UTF-8") from error
    try:
        packet = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON: {error.msg}") from error
    return validate_packet(packet, max_people=max_people)
