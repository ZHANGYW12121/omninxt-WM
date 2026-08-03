#!/usr/bin/env python3
"""Regression tests for the simulator-independent dataset v2 contract."""

import json
import socket
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np

from dataset_v2_schema import (
    JOINT_FIELDS,
    build_ego_state,
    ego_reference,
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


class DatasetV2SchemaTest(unittest.TestCase):
    def test_skeleton_packet_preserves_gt_source_and_track_reset(self):
        last_seen = {}
        first = skeleton_to_arrays(sample_packet(), 4, last_seen, 0)
        second = skeleton_to_arrays(sample_packet(sequence=8), 4, last_seen, 1)
        self.assertEqual(int(first["human_track_id"][0]), 1)
        self.assertTrue(bool(first["human_is_first"][0]))
        self.assertFalse(bool(second["human_is_first"][0]))
        self.assertTrue(np.all(first["human_source_code"][0] == 5))
        self.assertTrue(np.all(first["human_joint_valid"][0]))

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
        skeleton = skeleton_to_arrays(sample_packet(), 4, {}, 0)
        sample = {
            "frame_index": np.asarray(0, np.int64),
            "simulation_time_s": np.asarray(2.0, np.float64),
            "episode_time_s": np.asarray(0.0, np.float64),
            "wall_timestamp_ns": np.asarray(1, np.int64),
            "dt_s": np.asarray(0.0, np.float32),
            "skeleton_timestamp_ns": np.asarray(2_000_000_000, np.int64),
            "skeleton_sequence": np.asarray(7, np.int64),
            "skeleton_time_offset_ms": np.asarray(0.0, np.float32),
            "ego_state": np.zeros(14, np.float32),
            "ego_quaternion_xyzw": np.asarray([0, 0, 0, 1], np.float32),
            "prev_action_applied": np.zeros(4, np.float32),
            "reward": np.asarray(0.0, np.float32),
            "reward_components": np.zeros(6, np.float32),
            "is_first": np.asarray(True, np.bool_),
            "is_terminal": np.asarray(False, np.bool_),
            "discount": np.asarray(1.0, np.float32),
            "priv_human_position_world": np.zeros((4, 3), np.float32),
            "priv_collision_joints_world": np.zeros((4, 10, 3), np.float32),
        }
        sample.update(skeleton)
        arrays = stack_samples([sample])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000000.npz"
            write_npz_atomic(path, arrays)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertEqual(loaded["human_xyz"].shape, (1, 4, 17, 3))
                self.assertEqual(int(loaded["human_source_code"][0, 0, 0]), 5)

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
            )
            recorder.start()
            self.assertTrue(recorder.update())
            clock[0] = 10.1
            receiver.add(sample_packet(
                sequence=2, timestamp_ns=10_100_000_000))
            self.assertTrue(recorder.update())
            clock[0] = 10.2
            recorder.stop(reason="manual_stop")
            episodes = list((Path(directory) / "episodes").iterdir())
            self.assertEqual(len(episodes), 1)
            chunks = sorted((episodes[0] / "chunks").glob("*.npz"))
            self.assertEqual(len(chunks), 2)
            with np.load(chunks[0], allow_pickle=False) as first:
                np.testing.assert_array_equal(first["frame_index"], [0, 1])
                self.assertFalse(bool(first["prev_action_applied_valid"][0]))
                self.assertTrue(bool(first["prev_action_applied_valid"][1]))
                self.assertTrue(np.all(first["human_source_code"][:, 0] == 5))
            with np.load(chunks[1], allow_pickle=False) as final:
                self.assertEqual(int(final["frame_index"][0]), 2)
                self.assertTrue(bool(final["is_terminal"][0]))
                self.assertFalse(bool(final["skeleton_fresh"][0]))


if __name__ == "__main__":
    unittest.main()
