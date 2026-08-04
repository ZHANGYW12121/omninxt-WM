"""Validator/decoder for the Nano ``OPB1`` perception stream.

The formal producer is ``edge/jetson/scripts/109_stream_depth_to_backend.py``
on ``main``.  This module intentionally depends only on the documented wire
contract and not on Jetson/ROS runtime code.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import zlib
from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np


MAGIC = b"OPB1"
PREFIX = struct.Struct("!4sII")
PREFIX_SIZE = PREFIX.size
IMAGE_SCHEMA = "omninxt.pose_images.v1"
DEPTH_SCHEMA = "omninxt.depth4.v1"
PAIR_NAMES = ("AB_RIGHT", "BC_REAR", "CD_LEFT", "DA_FRONT")
IMAGE_BLOCK_NAMES = ("anchor_mosaic", "stereo_mosaic")
DEFAULT_MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_RAW_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_BLOCKS = 8
DEFAULT_MAX_ELEMENTS = 32 * 1024 * 1024


class ProtocolError(ValueError):
    """Raised when a frame violates the formal perception contract."""


@dataclass(frozen=True)
class PerceptionPacket:
    header: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def kind(self) -> str:
        return str(self.header["kind"])

    @property
    def sequence(self) -> int:
        return int(self.header["sequence"])

    @property
    def session_id(self) -> str:
        return str(self.header["session_id"])

    @property
    def capture_timestamp_ns(self) -> int:
        return int(self.header["capture_timestamp_ns"])


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def _text(value: Any, name: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ProtocolError(f"{name} must be a non-empty string <= {maximum} chars")
    return value


def _bounded_decompress(payload: bytes, expected: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, expected + 1)
        if len(raw) > expected or decoder.unconsumed_tail:
            raise ProtocolError("zlib output exceeds declared raw_bytes")
        raw += decoder.flush(max(1, expected - len(raw) + 1))
    except zlib.error as error:
        raise ProtocolError("invalid zlib payload") from error
    if not decoder.eof or decoder.unused_data or len(raw) != expected:
        raise ProtocolError("zlib payload is truncated or has trailing data")
    return raw


def _validate_header(
    header: Any,
    *,
    payload_length: int,
    max_raw_bytes: int,
    max_blocks: int,
    max_elements: int,
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    schema = header.get("schema")
    kind = header.get("kind")
    expected_schema = {"images": IMAGE_SCHEMA, "depth": DEPTH_SCHEMA}.get(kind)
    if expected_schema is None or schema != expected_schema:
        raise ProtocolError(f"unsupported schema/kind pair: {schema!r}/{kind!r}")
    _text(header.get("session_id"), "session_id", 128)
    _integer(header.get("sequence"), "sequence", 1)
    _integer(header.get("source_sequence"), "source_sequence", 1)
    _integer(header.get("capture_timestamp_ns"), "capture_timestamp_ns", 1)
    _integer(header.get("send_timestamp_ns"), "send_timestamp_ns", 1)
    compression = header.get("compression")
    if compression not in ("zlib", "none"):
        raise ProtocolError(f"unsupported compression: {compression!r}")
    raw_bytes = _integer(header.get("raw_bytes"), "raw_bytes", 1)
    if raw_bytes > int(max_raw_bytes):
        raise ProtocolError("raw_bytes exceeds configured bound")
    if _integer(header.get("payload_bytes"), "payload_bytes", 1) != payload_length:
        raise ProtocolError("payload_bytes does not match prefix")
    checksum = header.get("raw_crc32")
    if not isinstance(checksum, str) or len(checksum) != 8:
        raise ProtocolError("raw_crc32 must be eight hexadecimal digits")
    try:
        int(checksum, 16)
    except ValueError as error:
        raise ProtocolError("raw_crc32 must be hexadecimal") from error

    blocks = header.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= int(max_blocks):
        raise ProtocolError("blocks must be a bounded non-empty list")
    names: set[str] = set()
    occupied: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        label = f"blocks[{index}]"
        if not isinstance(block, dict):
            raise ProtocolError(f"{label} must be an object")
        name = _text(block.get("name"), f"{label}.name", 128)
        if name in names:
            raise ProtocolError(f"duplicate block name: {name}")
        names.add(name)
        try:
            dtype = np.dtype(_text(block.get("dtype"), f"{label}.dtype", 16))
        except TypeError as error:
            raise ProtocolError(f"{label}.dtype is unsupported") from error
        if dtype.hasobject or dtype.fields or dtype.subdtype:
            raise ProtocolError(f"{label}.dtype must be a plain numeric dtype")
        shape = block.get("shape")
        if not isinstance(shape, list) or not shape or len(shape) > 4:
            raise ProtocolError(f"{label}.shape must have one to four dimensions")
        elements = 1
        for axis, dimension in enumerate(shape):
            elements *= _integer(dimension, f"{label}.shape[{axis}]", 1)
            if elements > int(max_elements):
                raise ProtocolError(f"{label} exceeds maximum element count")
        offset = _integer(block.get("offset"), f"{label}.offset")
        nbytes = _integer(block.get("nbytes"), f"{label}.nbytes", 1)
        if nbytes != elements * dtype.itemsize:
            raise ProtocolError(f"{label}.nbytes does not match dtype and shape")
        end = offset + nbytes
        if end > raw_bytes:
            raise ProtocolError(f"{label} extends beyond raw payload")
        occupied.append((offset, end))

    occupied.sort()
    cursor = 0
    for start, end in occupied:
        if start != cursor:
            raise ProtocolError("block layout must be contiguous and non-overlapping")
        cursor = end
    if cursor != raw_bytes:
        raise ProtocolError("blocks do not describe the complete raw payload")

    if kind == "images":
        if tuple(block["name"] for block in blocks) != IMAGE_BLOCK_NAMES:
            raise ProtocolError("image packet must contain anchor then stereo mosaic")
        for block in blocks:
            if np.dtype(block["dtype"]) != np.dtype("uint8"):
                raise ProtocolError("image mosaics must use uint8")
            if len(block["shape"]) != 2:
                raise ProtocolError("image mosaics must be two-dimensional")
    else:
        if len(blocks) != 1 or blocks[0]["name"] != "depths":
            raise ProtocolError("depth packet must contain one depths block")
        block = blocks[0]
        dtype = np.dtype(block["dtype"])
        if dtype not in (np.dtype("<f4"), np.dtype("<u2")):
            raise ProtocolError("depths dtype must be little-endian float32 or uint16")
        if len(block["shape"]) != 3 or block["shape"][0] != 4:
            raise ProtocolError("depths must have shape [4,H,W]")
        if tuple(header.get("pair_names", ())) != PAIR_NAMES:
            raise ProtocolError("depth pair_names must use AB,BC,CD,DA order")
        expected_scale = 1.0 if dtype == np.dtype("<f4") else 0.001
        if abs(_finite(block.get("scale_m"), "depths.scale_m") - expected_scale) > 1e-9:
            raise ProtocolError("depths.scale_m is inconsistent with dtype")
    return blocks, raw_bytes


def decode_packet(
    prefix: bytes,
    header_bytes: bytes,
    payload: bytes,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_raw_bytes: int = DEFAULT_MAX_RAW_BYTES,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> PerceptionPacket:
    if len(prefix) != PREFIX_SIZE:
        raise ProtocolError("invalid prefix length")
    magic, header_length, payload_length = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("invalid perception packet magic")
    if not 1 <= header_length <= int(max_header_bytes):
        raise ProtocolError("header length exceeds configured bound")
    if not 1 <= payload_length <= int(max_payload_bytes):
        raise ProtocolError("payload length exceeds configured bound")
    if len(header_bytes) != header_length or len(payload) != payload_length:
        raise ProtocolError("frame lengths do not match prefix")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("header is not valid UTF-8 JSON") from error
    blocks, raw_bytes = _validate_header(
        header,
        payload_length=payload_length,
        max_raw_bytes=max_raw_bytes,
        max_blocks=max_blocks,
        max_elements=max_elements,
    )
    raw = (
        _bounded_decompress(payload, raw_bytes)
        if header["compression"] == "zlib"
        else payload
    )
    if len(raw) != raw_bytes:
        raise ProtocolError("raw payload size does not match raw_bytes")
    expected_crc = int(header["raw_crc32"], 16)
    if (zlib.crc32(raw) & 0xFFFFFFFF) != expected_crc:
        raise ProtocolError("raw payload CRC32 mismatch")
    arrays: dict[str, np.ndarray] = {}
    for block in blocks:
        start = int(block["offset"])
        end = start + int(block["nbytes"])
        arrays[block["name"]] = np.frombuffer(
            raw[start:end], dtype=np.dtype(block["dtype"])
        ).reshape(tuple(block["shape"])).copy()
    return PerceptionPacket(header=header, arrays=arrays)


def _read_exact(stream: BinaryIO | socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(length)
    while remaining:
        if hasattr(stream, "recv"):
            chunk = stream.recv(remaining)  # type: ignore[attr-defined]
        else:
            chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("connection closed during perception packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_packet(
    stream: BinaryIO | socket.socket,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_raw_bytes: int = DEFAULT_MAX_RAW_BYTES,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> PerceptionPacket:
    """Read exactly one packet even when TCP fragments prefix or payload."""

    prefix = _read_exact(stream, PREFIX_SIZE)
    magic, header_length, payload_length = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("invalid perception packet magic")
    if not 1 <= header_length <= int(max_header_bytes):
        raise ProtocolError("header length exceeds configured bound")
    if not 1 <= payload_length <= int(max_payload_bytes):
        raise ProtocolError("payload length exceeds configured bound")
    header = _read_exact(stream, header_length)
    payload = _read_exact(stream, payload_length)
    return decode_packet(
        prefix,
        header,
        payload,
        max_header_bytes=max_header_bytes,
        max_payload_bytes=max_payload_bytes,
        max_raw_bytes=max_raw_bytes,
        max_blocks=max_blocks,
        max_elements=max_elements,
    )
