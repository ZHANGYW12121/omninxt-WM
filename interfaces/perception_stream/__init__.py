"""Nano pose-image and dense-depth TCP transport contract."""

from .protocol import (  # noqa: F401
    DEPTH_SCHEMA,
    IMAGE_SCHEMA,
    MAGIC,
    PREFIX_SIZE,
    PerceptionPacket,
    ProtocolError,
    decode_packet,
    read_packet,
)
