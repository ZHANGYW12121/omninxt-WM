#!/usr/bin/env python3
"""Verify the frozen step-12000 motion-tracker golden trace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "simulation" / "omnidepth" / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from motion_skeleton_tracker import MotionSkeletonTracker  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "world_model" / "deployment" / "step12000_tracker"


def _person(
    joint_names: list[str],
    template: np.ndarray,
    translation: list[float],
    measurement: dict[str, Any],
    offsets: dict[str, list[float]],
) -> dict[str, Any]:
    translated = template + np.asarray(translation, np.float64)
    joints = []
    for joint_id, name in enumerate(joint_names):
        xyz = translated[joint_id].copy()
        if str(joint_id) in offsets:
            xyz += np.asarray(offsets[str(joint_id)], np.float64)
        joints.append({
            "id": joint_id,
            "name": name,
            "xyz_imu_m": xyz.tolist(),
            "score": float(measurement["score"]),
            "source": str(measurement["source"]),
            "measurement_sigma_m": float(measurement["measurement_sigma_m"]),
        })
    return {"source_pairs": ["GOLDEN"], "joints": joints}


def _round_vector(value: Any) -> list[float]:
    return np.asarray(value, np.float64).round(6).tolist()


def _canonical_person(person: dict[str, Any]) -> dict[str, Any]:
    joint = person["joints"][9]
    return {
        "person_id": int(person["person_id"]),
        "track_uid": str(person["track_uid"]),
        "track_state": str(person["track_state"]),
        "range_gate_center_m": _round_vector(person["range_gate_center_m"]),
        "root_velocity_base_link_mps": _round_vector(
            person["root_velocity_base_link_mps"]),
        "velocity_valid": bool(person["velocity_valid"]),
        "velocity_sigma_mps": person["velocity_sigma_mps"],
        "identity_confidence": float(person["identity_confidence"]),
        "track_age_frames": int(person["track_age_frames"]),
        "consecutive_prediction_frames": int(
            person["consecutive_prediction_frames"]),
        "joint_9_xyz_m": (
            None if joint["xyz_imu_m"] is None
            else _round_vector(joint["xyz_imu_m"])
        ),
        "joint_9_predicted": bool(joint["predicted"]),
        "joint_9_rejected_measurements": int(
            joint.get("kalman_rejected_measurements", 0)),
    }


def run_trace(payload: dict[str, Any]) -> dict[str, Any]:
    config = dict(payload["tracker_config"])
    joint_names = list(payload["joint_names"])
    template = np.asarray(payload["skeleton_template_xyz_m"], np.float64)
    if template.shape != (17, 3):
        raise ValueError("golden skeleton template must be [17,3]")
    tracker = MotionSkeletonTracker(joint_names, **config)
    tracker.session_id = str(payload["initial_session_id"])
    outputs = []
    for frame_index, frame in enumerate(payload["frames"]):
        reset_session = frame.get("reset_session")
        if reset_session is not None:
            tracker.reset_session()
            tracker.session_id = str(reset_session)
        frame_offsets = frame.get("joint_offsets_m", {})
        people = [
            _person(
                joint_names,
                template,
                translation,
                payload["measurement"],
                frame_offsets.get(str(person_index), {}),
            )
            for person_index, translation in enumerate(frame["translations_m"])
        ]
        tracked = tracker.update(people, int(frame["timestamp_ns"]))
        status = tracker.status(int(frame["timestamp_ns"]))
        outputs.append({
            "frame_index": frame_index,
            "timestamp_ns": int(frame["timestamp_ns"]),
            "people": [_canonical_person(person) for person in tracked],
            "active_tracks": int(status["active_tracks"]),
            "next_person_id": int(status["next_person_id"]),
            "kalman_measurements_rejected": int(
                status["kalman_measurements_rejected"]),
        })
    return {
        "schema": "omninxt.step12000.tracker-golden-output.v1",
        "input_schema": str(payload["schema"]),
        "frames": outputs,
    }


def _compare(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"{path}: dictionary keys differ")
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: list length differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            expected, float(actual), rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise AssertionError(f"{path}: expected {expected}, got {actual}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_ROOT / "golden_input.json")
    parser.add_argument(
        "--expected", type=Path, default=DEFAULT_ROOT / "golden_expected.json")
    parser.add_argument(
        "--print-current", action="store_true",
        help="Print the current canonical output without comparing it.")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    actual = run_trace(payload)
    if args.print_current:
        print(json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=False))
        return 0
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    _compare(expected, actual)
    print(f"PASS: {len(actual['frames'])} golden frames match {args.expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
