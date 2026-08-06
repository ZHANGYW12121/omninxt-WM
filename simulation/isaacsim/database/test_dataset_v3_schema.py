#!/usr/bin/env python3
"""Regression tests for the simulator-independent compact dataset v3 contract."""

import json
import socket
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np

from dataset_v3_schema import (
    BODY_JOINT_INDICES,
    JOINT_FIELDS,
    build_ego_state,
    ego_reference,
    normalized_applied_action,
    transition_flags,
    skeleton_to_arrays,
    stack_samples,
    write_npz_atomic,
)
from skeleton_packet_receiver import SkeletonPacketReceiver

if "carb" not in sys.modules:
    sys.modules["carb"] = types.SimpleNamespace(log_warn=lambda message: None)
try:
    from skeleton_dataset_recorder import SkeletonStateDatasetRecorder
except ModuleNotFoundError as error:
    if error.name != "scipy":
        raise
    SkeletonStateDatasetRecorder = None


def sample_packet(sequence=7, timestamp_ns=2_000_000_000):
    rows = []
    for joint in range(17):
        rows.append([
            1.0 + joint * 0.01, 0.2, 0.3, 0.9, 0.85,
            1, 1, 0, 0.02, 0.0, 5,
        ])
    return {
        "schema": "omninxt.skeleton3d.v1",
        "sequence": sequence,
        "timestamp_ns": timestamp_ns,
        "frame_id": "base_link",
        "joint_names": ["joint_{}".format(index) for index in range(17)],
        "joint_fields": list(JOINT_FIELDS),
        "people": [{"person_id": 1, "joints": rows}],
    }


class DatasetV3SchemaTest(unittest.TestCase):
    def test_skeleton_packet_keeps_only_model_input_fields(self):
        first = skeleton_to_arrays(sample_packet(), 4)
        self.assertEqual(int(first["human_track_id"][0]), 1)
        self.assertTrue(np.all(first["human_joint_valid"][0]))
        np.testing.assert_allclose(
            first["human_xyz"][0, :, 0],
            [1.0 + index * 0.01 for index in BODY_JOINT_INDICES])
        self.assertEqual(
            set(first), {"human_xyz", "human_confidence", "human_joint_valid",
                         "human_track_id"})

    def test_applied_action_is_normalized_without_requested_copy(self):
        action = normalized_applied_action({
            "applied_body_flu": dict(zip(
                ("vx_body_mps", "vy_body_mps", "vz_world_mps", "yaw_rate_rps"),
                (1.0, -1.0, 0.5, 0.25))),
            "normalization_limits": dict(zip(
                ("vx_body_mps", "vy_body_mps", "vz_world_mps", "yaw_rate_rps"),
                (2.0, 2.0, 1.0, 0.5))),
        })
        np.testing.assert_allclose(action["action"], [0.5, -0.5, 0.5, 0.5])
        self.assertTrue(action["action_valid"])

    def test_terminal_and_truncation_flags_are_distinct(self):
        success = transition_flags("reached_goal")
        self.assertTrue(success["success"])
        self.assertTrue(success["is_last"])
        self.assertTrue(success["is_terminal"])
        collision = transition_flags("static_collision")
        self.assertFalse(collision["success"])
        self.assertTrue(collision["is_terminal"])
        shutdown = transition_flags("shutdown")
        self.assertTrue(shutdown["is_last"])
        self.assertFalse(shutdown["is_terminal"])
        time_limit = transition_flags("time_limit")
        self.assertTrue(time_limit["is_last"])
        self.assertFalse(time_limit["is_terminal"])

    def test_ego_state_uses_episode_and_body_frames(self):
        initial = {
            "position": [10.0, 20.0, 1.0],
            "velocity": [0.0, 0.0, 0.0],
            "acceleration": [0.0, 0.0, 0.0],
            "roll_pitch_yaw_rad": [0.0, 0.0, np.pi / 2],
            "altitude_agl": 1.0,
        }
        state = dict(initial)
        state.update({"position": [10.0, 21.0, 1.2],
                      "velocity": [0.0, 2.0, 0.1]})
        output = build_ego_state(state, ego_reference(initial))
        self.assertEqual(output.shape, (14,))
        np.testing.assert_allclose(output[:3], [1.0, 0.0, 0.2], atol=1e-5)
        np.testing.assert_allclose(output[3:6], [2.0, 0.0, 0.1], atol=1e-5)

    def test_atomic_chunk_round_trip_without_pickle(self):
        skeleton = skeleton_to_arrays(sample_packet(), 4)
        sample = {
            "frame_index": np.asarray(0, np.int64),
            "simulation_time_s": np.asarray(2.0, np.float64),
            "skeleton_timestamp_ns": np.asarray(2_000_000_000, np.int64),
            "skeleton_fresh": np.asarray(True, np.bool_),
            "ego_state": np.zeros(14, np.float32),
            "ego_altitude_valid": np.asarray(True, np.bool_),
            "ego_state_source": np.asarray(1, np.uint8),
            "action": np.zeros(4, np.float32),
            "action_valid": np.asarray(False, np.bool_),
            "reward": np.asarray(0.0, np.float32),
            "reward_components": np.zeros(6, np.float32),
            "is_first": np.asarray(True, np.bool_),
            "success": np.asarray(False, np.bool_),
            "termination_code": np.asarray(0, np.uint8),
            "is_last": np.asarray(False, np.bool_),
            "is_terminal": np.asarray(False, np.bool_),
            "priv_human_id": np.full(4, -1, np.int32),
            "priv_human_mask": np.zeros(4, np.bool_),
            "priv_human_position_world": np.zeros((4, 3), np.float32),
            "priv_human_velocity_world": np.zeros((4, 3), np.float32),
            "priv_min_human_clearance_m": np.asarray(0.0, np.float32),
            "priv_min_human_clearance_valid": np.asarray(False, np.bool_),
            "priv_min_human_ttc_s": np.asarray(0.0, np.float32),
            "priv_min_human_ttc_valid": np.asarray(False, np.bool_),
            "priv_collision": np.asarray(False, np.bool_),
            "priv_goal_distance_m": np.asarray(10.0, np.float32),
        }
        sample.update(skeleton)
        arrays = stack_samples([sample])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000000.npz"
            write_npz_atomic(path, arrays)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertEqual(loaded["human_xyz"].shape, (1, 4, 12, 3))
                self.assertEqual(set(loaded.files), set(arrays))

    def test_tcp_receiver_buffers_valid_packet(self):
        receiver = SkeletonPacketReceiver(host="127.0.0.1", port=0)
        try:
            receiver.start()
        except RuntimeError as error:
            if "Operation not permitted" in str(error):
                self.skipTest("sandbox forbids binding a local TCP test socket")
            raise
        deadline = time.time() + 2.0
        while receiver.port == 0 and receiver.status()["running"] and time.time() < deadline:
            time.sleep(0.01)
        self.assertNotEqual(receiver.port, 0)
        with socket.create_connection(("127.0.0.1", receiver.port), timeout=1.0) as client:
            client.sendall((json.dumps(sample_packet()) + "\n").encode("utf-8"))
        deadline = time.time() + 2.0
        while receiver.latest() is None and time.time() < deadline:
            time.sleep(0.01)
        latest = receiver.latest()
        receiver.stop()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["packet"]["sequence"], 7)

    def test_recorder_writes_contiguous_skeleton_only_episode(self):
        if SkeletonStateDatasetRecorder is None:
            self.skipTest("system Python lacks scipy; Isaac Python provides it")
        class State:
            position = np.asarray([0.0, 0.0, 1.0])
            linear_velocity = np.zeros(3)
            linear_acceleration = np.zeros(3)
            attitude = np.asarray([0.0, 0.0, 0.0, 1.0])

        class Tracker:
            joint_names = ["joint_{}".format(index) for index in range(10)]
            marker_positions = {}

            @staticmethod
            def get_joint_distances(position):
                return []

        class Receiver:
            def __init__(self):
                self.items = []

            def add(self, packet):
                self.items.append({
                    "packet": packet,
                    "arrival_index": len(self.items) + 1,
                    "received_wall_time_ns": time.time_ns(),
                })

            def packets_after(self, index):
                return [item for item in self.items
                        if item["arrival_index"] > index]

            def latest(self):
                return None if not self.items else self.items[-1]

            def status(self):
                return {"received": len(self.items)}

        clock = [10.0]
        receiver = Receiver()
        receiver.add(sample_packet(sequence=1, timestamp_ns=10_000_000_000))
        action = {
            "normalized": [0.5, 0.0, 0.0, 0.0],
            "normalization_limits": {
                "vx_body_mps": 2.0, "vy_body_mps": 2.0,
                "vz_world_mps": 1.0, "yaw_rate_rps": 1.0,
            },
            "requested": {
                "vx_body_mps": 1.0, "vy_body_mps": 0.0,
                "vz_world_mps": 0.0, "yaw_rate_rps": 0.0,
            },
            "applied_body_flu": {
                "vx_body_mps": 1.0, "vy_body_mps": 0.0,
                "vz_world_mps": 0.0, "yaw_rate_rps": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            recorder = SkeletonStateDatasetRecorder(
                drone=types.SimpleNamespace(state=State()),
                skeleton_tracker=Tracker(),
                skeleton_receiver=receiver,
                target_point=[0.0, 10.0, 1.0],
                goal_region={"x_range": [-1.0, 1.0], "y_min": 10.0},
                dataset_root=directory,
                sample_rate_hz=10.0,
                max_queue_size=8,
                drop_when_writer_busy=False,
                time_source=lambda: clock[0],
                time_source_name="simulation_time",
                action_provider=lambda: action,
                state_provider=lambda: State(),
                altitude_agl_provider=lambda position: 1.0,
                control_rate_hz=10.0,
                chunk_frames=2,
                people_count_provider=lambda: 1,
            )
            recorder.start()
            self.assertTrue(recorder.update())
            clock[0] = 10.1
            receiver.add(sample_packet(
                sequence=2, timestamp_ns=10_100_000_000))
            self.assertTrue(recorder.update())
            clock[0] = 10.2
            # Watchdog/event paths explicitly flush their last transition
            # before calling stop(); a truncation must not be written twice.
            recorder.set_event_status(termination_reason="manual_stop")
            self.assertTrue(recorder.update(force=True))
            recorder.stop(reason="manual_stop")
            episodes = list((Path(directory) / "episodes").iterdir())
            self.assertEqual(len(episodes), 1)
            chunks = sorted((episodes[0] / "chunks").glob("*.npz"))
            self.assertEqual(len(chunks), 2)
            with np.load(chunks[0], allow_pickle=False) as first:
                np.testing.assert_array_equal(first["frame_index"], [0, 1])
                self.assertFalse(bool(first["action_valid"][0]))
                self.assertTrue(bool(first["action_valid"][1]))
                self.assertAlmostEqual(float(first["action"][1, 0]), 0.5)
                self.assertEqual(first["human_xyz"].shape, (2, 1, 12, 3))
            with np.load(chunks[1], allow_pickle=False) as final:
                self.assertEqual(int(final["frame_index"][0]), 2)
                self.assertTrue(bool(final["is_last"][0]))
                self.assertFalse(bool(final["is_terminal"][0]))
                self.assertFalse(bool(final["skeleton_fresh"][0]))


if __name__ == "__main__":
    unittest.main()
