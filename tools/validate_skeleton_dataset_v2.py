#!/usr/bin/env python3
"""Validate OmniNxt skeleton/state dataset v2 without Isaac Sim."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_CODE = REPO_ROOT / "simulation" / "isaacsim" / "database"
sys.path.insert(0, str(DATABASE_CODE))

from dataset_v2_schema import SCHEMA, validate_chunk  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    episode = args.episode.resolve()
    metadata_path = episode / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit("missing {}".format(metadata_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise SystemExit("unexpected schema {}".format(metadata.get("schema")))
    forbidden = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.pcd", "*.ply"):
        forbidden.extend(episode.rglob(pattern))
    if forbidden:
        raise SystemExit("forbidden image/point-cloud payload: {}".format(forbidden[0]))
    chunks = sorted((episode / "chunks").glob("chunk_*.npz"))
    if not chunks:
        raise SystemExit("episode has no NPZ chunks")
    expected_frame = 0
    source_counts = {}
    total = 0
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        validate_chunk(arrays)
        indices = np.asarray(arrays["frame_index"], dtype=np.int64)
        if int(indices[0]) != expected_frame:
            raise SystemExit(
                "{} starts at frame {}, expected {}".format(
                    chunk.name, int(indices[0]), expected_frame))
        expected_frame = int(indices[-1]) + 1
        total += len(indices)
        codes, counts = np.unique(
            arrays["human_source_code"][arrays["human_joint_valid"]],
            return_counts=True)
        for code, count in zip(codes, counts):
            source_counts[int(code)] = source_counts.get(int(code), 0) + int(count)
    print("schema={}".format(SCHEMA))
    print("episode={}".format(episode))
    print("chunks={} frames={}".format(len(chunks), total))
    print("valid_joint_source_counts={}".format(source_counts))
    if any(code == 2 for code in source_counts):
        raise SystemExit("HITNet-sourced joints found in simulation dataset observation")
    print("validation=PASS")


if __name__ == "__main__":
    main()
