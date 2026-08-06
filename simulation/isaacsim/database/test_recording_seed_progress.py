#!/usr/bin/env python3
"""Regression tests for inclusive seed ranges and interruption resume."""

import json
import tempfile
import unittest
from pathlib import Path

from recording_seed_progress import (
    mark_seed_completed,
    resolve_seed_progress,
    seed_progress_path,
    write_seed_progress,
)


class RecordingSeedProgressTest(unittest.TestCase):
    def test_inclusive_range_advances_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            state = resolve_seed_progress(directory, "run_a", 10, 12)
            self.assertEqual(state["current_seed"], 10)
            state = mark_seed_completed(state, 10, "reached_goal")
            self.assertEqual(state["current_seed"], 11)
            state = mark_seed_completed(state, 11, "static_collision")
            state = mark_seed_completed(state, 12, "time_limit")
            self.assertEqual(state["current_seed"], 13)
            self.assertEqual(state["status"], "complete")

    def test_atomic_state_resumes_current_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = seed_progress_path(directory, "resume", 20, 22)
            state = resolve_seed_progress(directory, "resume", 20, 22)
            state = mark_seed_completed(state, 20, "human_collision")
            write_seed_progress(path, state)
            resumed = resolve_seed_progress(directory, "resume", 20, 22)
            self.assertEqual(resumed["current_seed"], 21)
            self.assertEqual([item["seed"] for item in resumed["completed"]], [20])

    def test_summary_recovers_after_progress_write_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episodes" / "episode_seed_30"
            episode.mkdir(parents=True)
            metadata = {
                "schema": "omninxt.crowd_skeleton_state.v3",
                "episode": {
                    "recording_run_id": "recover",
                    "seed_start": 30,
                    "seed_end": 32,
                    "crowd_seed": 30,
                },
            }
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            (episode / "summary.json").write_text(json.dumps({
                "termination_reason": "reached_goal",
            }), encoding="utf-8")
            state = resolve_seed_progress(root, "recover", 30, 32)
            self.assertEqual(state["current_seed"], 31)

    def test_shutdown_and_partial_episode_retry_same_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, reason in (
                ("shutdown", "shutdown"),
                ("manual", "manual_stop"),
                ("controller", "controller_error"),
                ("partial", None),
            ):
                episode = root / "episodes" / name
                episode.mkdir(parents=True)
                (episode / "metadata.json").write_text(json.dumps({
                    "schema": "omninxt.crowd_skeleton_state.v3",
                    "episode": {
                        "recording_run_id": "retry",
                        "seed_start": 40,
                        "seed_end": 42,
                        "crowd_seed": 40,
                    },
                }), encoding="utf-8")
                if reason is not None:
                    (episode / "summary.json").write_text(json.dumps({
                        "termination_reason": reason,
                    }), encoding="utf-8")
            state = resolve_seed_progress(root, "retry", 40, 42)
            self.assertEqual(state["current_seed"], 40)
            self.assertEqual(state["completed"], [])

    def test_manual_stop_and_controller_error_do_not_advance(self):
        state = resolve_seed_progress("/tmp/nonexistent_seed_progress_test",
                                      "invalid_retry", 50, 52, resume=False)
        for reason in ("manual_stop", "controller_error", "record_size_limit"):
            retried = mark_seed_completed(state, 50, reason)
            self.assertEqual(retried["current_seed"], 50)
            self.assertEqual(retried["completed"], [])


if __name__ == "__main__":
    unittest.main()
