#!/usr/bin/env python3
"""Validate compact OmniNxt skeleton/state dataset v3 without Isaac Sim."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_CODE = REPO_ROOT / "simulation" / "isaacsim" / "database"
sys.path.insert(0, str(DATABASE_CODE))

from dataset_v3_schema import (  # noqa: E402
    ACTION_KEYS, BODY_JOINT_INDICES, BODY_JOINT_NAMES, SCHEMA,
    TERMINATION_CODES, outcome_for_reason, transition_flags, validate_chunk,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--require-isaac-gt", action="store_true",
        help="reject valid joints sourced from geometry/HITNet instead of GT or GT-based temporal prediction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    episode = args.episode.resolve()
    metadata_path = episode / "metadata.json"
    summary_path = episode / "summary.json"
    if not metadata_path.is_file():
        raise SystemExit(f"missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise SystemExit(f"unexpected schema {metadata.get('schema')}")
    required_metadata = {
        "ego_reference_origin_xyz", "ego_reference_origin_yaw",
        "goal_position_episode_local", "action_normalization_limits",
        "actual_people_count", "stored_people_count", "coordinate_convention",
        "source_joint_topology", "joint_topology",
        "recorded_source_joint_indices", "joint_names",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise SystemExit(f"missing metadata fields: {sorted(missing_metadata)}")
    if int(metadata["stored_people_count"]) > int(metadata["actual_people_count"]):
        raise SystemExit("stored_people_count exceeds actual_people_count")
    if metadata["source_joint_topology"] != "COCO17":
        raise SystemExit("source_joint_topology must be COCO17")
    if metadata["joint_topology"] != "COCO12_BODY":
        raise SystemExit("joint_topology must be COCO12_BODY")
    if tuple(metadata["recorded_source_joint_indices"]) != BODY_JOINT_INDICES:
        raise SystemExit("recorded source joint indices must be COCO17 5..16")
    if tuple(metadata["joint_names"]) != BODY_JOINT_NAMES:
        raise SystemExit("stored joint names do not match COCO12_BODY")
    limits = metadata["action_normalization_limits"]
    if any(float(limits.get(key, 0.0)) <= 0.0 for key in ACTION_KEYS):
        raise SystemExit("action normalization limits must be positive")
    forbidden = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.pcd", "*.ply"):
        forbidden.extend(episode.rglob(pattern))
    if forbidden:
        raise SystemExit(f"forbidden image/point-cloud payload: {forbidden[0]}")
    chunks = sorted((episode / "chunks").glob("chunk_*.npz"))
    if not chunks:
        raise SystemExit("episode has no NPZ chunks")
    expected_frame = 0
    total = 0
    fresh = 0
    all_success = []
    all_codes = []
    all_last = []
    all_terminal = []
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        validate_chunk(arrays)
        indices = np.asarray(arrays["frame_index"], dtype=np.int64)
        if int(indices[0]) != expected_frame:
            raise SystemExit(
                f"{chunk.name} starts at frame {int(indices[0])}, expected {expected_frame}")
        if np.asarray(arrays["human_xyz"]).shape[1] != int(
                metadata["stored_people_count"]):
            raise SystemExit(f"{chunk.name}: people dimension differs from metadata")
        expected_frame = int(indices[-1]) + 1
        total += len(indices)
        fresh += int(np.count_nonzero(arrays["skeleton_fresh"]))
        all_success.extend(np.asarray(arrays["success"], np.bool_).tolist())
        all_codes.extend(np.asarray(arrays["termination_code"], np.uint8).tolist())
        all_last.extend(np.asarray(arrays["is_last"], np.bool_).tolist())
        all_terminal.extend(np.asarray(arrays["is_terminal"], np.bool_).tolist())
    source_counts = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_counts = {
            int(key): int(value) for key, value in
            summary.get("skeleton_valid_joint_source_counts", {}).items()
        }
        reason = str(summary.get("termination_reason", ""))
        if reason not in TERMINATION_CODES:
            raise SystemExit(f"unknown summary termination_reason {reason!r}")
        expected = transition_flags(reason)
        if summary.get("outcome_class") != outcome_for_reason(reason):
            raise SystemExit("summary outcome_class disagrees with termination_reason")
        if bool(summary.get("success")) != bool(expected["success"]):
            raise SystemExit("summary success disagrees with termination_reason")
        if not all_last or sum(bool(value) for value in all_last) != 1 or not all_last[-1]:
            raise SystemExit("episode must have exactly one final is_last row")
        final_actual = (
            bool(all_success[-1]), int(all_codes[-1]),
            bool(all_last[-1]), bool(all_terminal[-1]),
        )
        final_expected = (
            bool(expected["success"]), int(expected["termination_code"]),
            bool(expected["is_last"]), bool(expected["is_terminal"]),
        )
        if final_actual != final_expected:
            raise SystemExit(
                f"final transition flags {final_actual} disagree with summary {final_expected}")
        if any(all_success[:-1]) or any(all_last[:-1]) or any(all_terminal[:-1]):
            raise SystemExit("success/is_last/is_terminal may only be set on the final row")
        if any(int(code) != TERMINATION_CODES["recording"] for code in all_codes[:-1]):
            raise SystemExit("non-final rows must use termination_code=recording")
    print(f"schema={SCHEMA}")
    print(f"episode={episode}")
    print(f"chunks={len(chunks)} frames={total} skeleton_fresh={fresh}/{total}")
    print(f"valid_joint_source_counts={source_counts}")
    if args.require_isaac_gt:
        if not summary_path.is_file():
            raise SystemExit("strict GT validation requires summary.json")
        if not source_counts:
            raise SystemExit("strict GT validation found no valid skeleton joints")
        # 3 is temporal prediction; in strict GT mode its last measurement is
        # still based on source 5 (Isaac exact-timestamp depth). Code 4 means
        # an unknown/other measured source and must remain rejected.
        unexpected = sorted(code for code in source_counts if code not in (3, 5))
        if unexpected:
            raise SystemExit(f"non-GT skeleton source codes found: {unexpected}")
    print("validation=PASS")


if __name__ == "__main__":
    main()
