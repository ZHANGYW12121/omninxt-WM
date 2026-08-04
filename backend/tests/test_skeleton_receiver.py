from __future__ import annotations

import json
import copy
import importlib.util
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from backend.skeleton_receiver.adapter import NanoHumanObservationAdapter
from backend.skeleton_receiver.server import (
    SkeletonReceiverApplication,
    SkeletonReceiverServer,
)
from interfaces.skeleton3d.protocol import (
    COCO17_JOINT_NAMES,
    COORDINATE_CONVENTION,
    FRAME_ID,
    JOINT_FIELDS,
    SCHEMA,
    UNITS,
    ProtocolError,
    validate_packet,
)


def packet(sequence: int, timestamp_ns: int, *, offset_x: float = 0.0) -> dict:
    joints = []
    for joint_id in range(17):
        joints.append([
            2.0 + offset_x + 0.01 * joint_id,
            0.1 + 0.02 * joint_id,
            0.5 + 0.03 * joint_id,
            0.9,
            0.8,
            1,
            1,
            0,
            0.01,
            0.0,
            1,
        ])
    return {
        "schema": SCHEMA,
        "sequence": sequence,
        "timestamp_ns": timestamp_ns,
        "frame_id": FRAME_ID,
        "coordinate_convention": COORDINATE_CONVENTION,
        "units": UNITS,
        "joint_names": list(COCO17_JOINT_NAMES),
        "joint_fields": list(JOINT_FIELDS),
        "source_codes": {
            "invalid": 0,
            "stereo_geometry": 1,
            "hitnet": 2,
            "temporal_prediction": 3,
            "other": 4,
        },
        "people": [{"person_id": 7, "joints": joints}],
    }


def upright_packet(sequence: int, timestamp_ns: int) -> dict:
    value = packet(sequence, timestamp_ns)
    xyz = np.asarray([
        [3.0, 0.00, 0.82],  # nose
        [3.0, 0.04, 0.86], [3.0, -0.04, 0.86],
        [3.0, 0.09, 0.82], [3.0, -0.09, 0.82],
        [3.0, 0.20, 0.52], [3.0, -0.20, 0.52],
        [3.0, 0.34, 0.27], [3.0, -0.34, 0.27],
        [3.0, 0.40, 0.02], [3.0, -0.40, 0.02],
        [3.0, 0.15, 0.00], [3.0, -0.15, 0.00],
        [3.0, 0.15, -0.45], [3.0, -0.15, -0.45],
        [3.0, 0.15, -0.90], [3.0, -0.15, -0.90],
    ], dtype=np.float64)
    for joint, point in zip(value["people"][0]["joints"], xyz):
        joint[:3] = point.tolist()
        joint[8] = 0.02
    return value


class ProtocolAndAdapterTest(unittest.TestCase):
    def test_contract_accepts_nano_tracker_id_zero(self):
        value = packet(1, 1_000_000_000)
        value["people"][0]["person_id"] = 0
        self.assertEqual(validate_packet(value)["people"][0]["person_id"], 0)

    def test_packet_built_by_actual_nano_sender_is_accepted(self):
        sender_path = (
            Path(__file__).resolve().parents[2]
            / "edge" / "jetson" / "scripts" / "skeleton_stream.py"
        )
        spec = importlib.util.spec_from_file_location("nano_skeleton_stream", sender_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        people = [{
            "person_id": 3,
            "joints": [{
                "id": joint_id,
                "name": name,
                "score": 0.9,
                "source": "stereo_geometry",
                "xyz_base_link_m": [1.0, 0.1 * joint_id, 0.5],
            } for joint_id, name in enumerate(COCO17_JOINT_NAMES)],
        }]
        built = module.build_skeleton_packet(
            people, 3_000_000_000, 8, COCO17_JOINT_NAMES
        )
        self.assertIs(validate_packet(built), built)

    def test_contract_rejects_wrong_frame(self):
        value = packet(1, 1_000_000_000)
        value["frame_id"] = "world"
        with self.assertRaisesRegex(ProtocolError, "frame_id"):
            validate_packet(value)

    def test_adapter_produces_current_world_model_schema(self):
        adapter = NanoHumanObservationAdapter(max_people=3, velocity_ema=1.0)
        first = adapter.adapt(validate_packet(packet(10, 1_000_000_000)))
        second = adapter.adapt(validate_packet(packet(
            11, 1_100_000_000, offset_x=0.1
        )))
        self.assertEqual(first.skeleton.shape, (3, 17, 7))
        self.assertEqual(first.skeleton_raw.shape, (3, 17, 7))
        self.assertEqual(first.human_root.shape, (3, 10))
        self.assertEqual(first.human_joints.shape, (3, 17, 7))
        self.assertEqual(first.joint_uncertainty_m.shape, (3, 17))
        self.assertEqual(first.joint_prediction_age_ms.shape, (3, 17))
        self.assertEqual(first.joint_inferred_mask.shape, (3, 17))
        self.assertEqual(first.human_ids.tolist(), [7, -1, -1])
        self.assertTrue(first.human_is_first[0])
        self.assertFalse(second.human_is_first[0])
        self.assertFalse(first.joint_mask[0, :5].any())
        self.assertTrue(first.joint_mask[0, 5:].all())
        np.testing.assert_array_equal(first.skeleton[0, :5], 0.0)
        self.assertTrue(np.any(first.skeleton_raw[0, :5, :3] != 0.0))
        self.assertTrue(np.isfinite(second.skeleton).all())
        self.assertGreater(float(second.skeleton[0, :, 3].mean()), 0.0)
        np.testing.assert_allclose(second.skeleton_raw[0, :, 0],
                                   first.skeleton_raw[0, :, 0] + 0.1,
                                   atol=1e-5)
        np.testing.assert_allclose(second.human_joints[0, :, :3].mean(0), 0.0,
                                   atol=0.2)
        batch = second.as_model_batch()
        self.assertEqual(batch["human_root"].shape, (1, 1, 3, 10))
        self.assertEqual(batch["human_joints"].shape, (1, 1, 3, 17, 7))

    def test_isolated_depth_outlier_does_not_pull_wrist(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        first = adapter.adapt(validate_packet(packet(1, 1_000_000_000)))
        second = adapter.adapt(validate_packet(packet(2, 1_100_000_000)))
        corrupted = copy.deepcopy(packet(3, 1_200_000_000))
        corrupted["people"][0]["joints"][9][0] += 2.0

        third = adapter.adapt(validate_packet(corrupted))

        raw_jump = np.linalg.norm(
            third.skeleton_raw[0, 9, :3] - second.skeleton_raw[0, 9, :3]
        )
        refined_jump = np.linalg.norm(
            third.skeleton[0, 9, :3] - second.skeleton[0, 9, :3]
        )
        self.assertGreater(raw_jump, 1.9)
        self.assertLess(refined_jump, 0.15)

    def test_joint_without_depth_remains_as_low_confidence_inference(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        adapter.adapt(validate_packet(packet(1, 1_000_000_000)))
        missing = copy.deepcopy(packet(2, 1_100_000_000))
        row = missing["people"][0]["joints"][9]
        row[5:8] = [0, 0, 0]
        row[4] = 0.0
        short = adapter.adapt(validate_packet(missing))
        self.assertTrue(short.joint_mask[0, 9])
        self.assertTrue(short.joint_inferred_mask[0, 9])
        self.assertAlmostEqual(float(short.joint_prediction_age_ms[0, 9]), 100.0)

        missing_late = copy.deepcopy(missing)
        missing_late["sequence"] = 3
        missing_late["timestamp_ns"] = 1_400_000_000
        late = adapter.adapt(validate_packet(missing_late))
        self.assertTrue(late.joint_mask[0, 9])
        self.assertTrue(late.joint_inferred_mask[0, 9])
        self.assertGreater(float(np.linalg.norm(late.skeleton[0, 9, :3])), 0.1)
        self.assertGreater(float(late.joint_uncertainty_m[0, 9]), 0.6)

    def test_never_measured_depth_is_completed_from_body_kinematics(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        partial = upright_packet(1, 1_000_000_000)
        missing_ids = (0, 4, 8, 10, 14, 16)
        for joint_id in missing_ids:
            row = partial["people"][0]["joints"][joint_id]
            row[:3] = [0.0, 0.0, 0.0]
            row[4] = 0.0
            row[5:8] = [0, 0, 0]
            row[8:10] = [0.0, 0.0]
            row[10] = 0

        completed = adapter.adapt(validate_packet(partial))

        body_missing_ids = tuple(joint_id for joint_id in missing_ids if joint_id >= 5)
        self.assertEqual(int(completed.joint_mask[0].sum()), 12)
        self.assertFalse(completed.joint_mask[0, :5].any())
        self.assertEqual(
            int(completed.joint_inferred_mask[0].sum()), len(body_missing_ids)
        )
        for joint_id in body_missing_ids:
            self.assertTrue(completed.joint_inferred_mask[0, joint_id])
            np.testing.assert_array_equal(completed.skeleton_raw[0, joint_id], 0.0)
            self.assertTrue(np.isfinite(completed.skeleton[0, joint_id]).all())
            self.assertGreater(
                float(np.linalg.norm(completed.skeleton[0, joint_id, :3])), 0.1
            )
        self.assertLess(
            float(completed.skeleton[0, 10, 6]),
            float(completed.skeleton[0, 9, 6]),
        )

    def test_whole_person_short_dropout_keeps_stable_slot(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        first = adapter.adapt(validate_packet(upright_packet(1, 1_000_000_000)))
        empty = upright_packet(2, 1_100_000_000)
        empty["people"] = []

        predicted = adapter.adapt(validate_packet(empty))

        self.assertTrue(predicted.human_mask[0])
        self.assertEqual(predicted.human_ids[0], first.human_ids[0])
        self.assertFalse(predicted.human_is_first[0])
        self.assertEqual(int(predicted.joint_mask[0].sum()), 12)
        self.assertEqual(int(predicted.joint_inferred_mask[0].sum()), 12)

        expired = upright_packet(3, 1_400_000_000)
        expired["people"] = []
        missing = adapter.adapt(validate_packet(expired))
        self.assertFalse(missing.human_mask[0])

    def test_upright_walking_prior_rejects_inverted_torso_and_legs(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        adapter.adapt(validate_packet(upright_packet(1, 1_000_000_000)))
        adapter.adapt(validate_packet(upright_packet(2, 1_100_000_000)))
        corrupted = upright_packet(3, 1_200_000_000)
        for joint_id in (5, 6):
            corrupted["people"][0]["joints"][joint_id][2] = -0.8
        for joint_id in (13, 14):
            corrupted["people"][0]["joints"][joint_id][2] = 0.7

        refined = adapter.adapt(validate_packet(corrupted)).skeleton[0, :, :3]

        shoulder_z = float(0.5 * (refined[5, 2] + refined[6, 2]))
        hip_z = float(0.5 * (refined[11, 2] + refined[12, 2]))
        self.assertGreater(shoulder_z, hip_z + 0.25)
        self.assertLess(refined[13, 2], refined[11, 2])
        self.assertLess(refined[14, 2], refined[12, 2])

    def test_persistent_common_motion_reacquires_instead_of_locking(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        initial = adapter.adapt(validate_packet(upright_packet(1, 1_000_000_000)))
        adapter.adapt(validate_packet(upright_packet(2, 1_100_000_000)))
        latest = None
        for sequence in range(3, 6):
            moved = upright_packet(sequence, 900_000_000 + sequence * 100_000_000)
            for row in moved["people"][0]["joints"]:
                row[0] += 1.0
            latest = adapter.adapt(validate_packet(moved))
        self.assertIsNotNone(latest)
        initial_root_x = float(initial.human_root[0, 0])
        latest_root_x = float(latest.human_root[0, 0])
        self.assertGreater(latest_root_x, initial_root_x + 0.20)

    def test_large_timestamp_gap_resets_recurrent_slot(self):
        adapter = NanoHumanObservationAdapter(max_people=1)
        adapter.adapt(validate_packet(packet(1, 1_000_000_000)))
        after_gap = adapter.adapt(validate_packet(packet(2, 2_000_000_000)))
        self.assertTrue(after_gap.human_is_first[0])
        self.assertTrue(np.all(after_gap.skeleton[0, :, 3:6] == 0.0))


class TcpReceiverTest(unittest.TestCase):
    def test_server_shutdown_does_not_wait_for_connected_sender(self):
        with tempfile.TemporaryDirectory() as directory:
            application = SkeletonReceiverApplication(
                max_people=1, runtime_dir=directory, print_every=0
            )
            server = SkeletonReceiverServer(("127.0.0.1", 0), application)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = socket.create_connection(server.server_address, timeout=2.0)
            try:
                deadline = time.monotonic() + 1.0
                while application.status()["connections"] < 1:
                    if time.monotonic() >= deadline:
                        self.fail("receiver did not register connected sender")
                    time.sleep(0.01)
                started = time.monotonic()
                server.shutdown()
                self.assertLess(time.monotonic() - started, 1.0)
            finally:
                connection.close()
                server.server_close()
                thread.join(timeout=2.0)

    def test_fragmented_ndjson_rejection_gap_and_atomic_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            application = SkeletonReceiverApplication(
                max_people=4, runtime_dir=directory, print_every=0
            )
            server = SkeletonReceiverServer(("127.0.0.1", 0), application)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            try:
                with socket.create_connection((host, port), timeout=2.0) as connection:
                    connection.sendall(b'{"schema":"wrong"}\n')
                    first = (json.dumps(packet(20, 2_000_000_000)) + "\n").encode()
                    connection.sendall(first[:31])
                    connection.sendall(first[31:])
                    third = (json.dumps(packet(
                        22, 2_100_000_000, offset_x=0.1
                    )) + "\n").encode()
                    connection.sendall(third)

                deadline = time.monotonic() + 3.0
                while application.status()["frames_accepted"] < 2:
                    if time.monotonic() >= deadline:
                        self.fail("receiver did not accept both valid frames")
                    time.sleep(0.02)
                status = application.status()
                self.assertEqual(status["frames_received"], 3)
                self.assertEqual(status["frames_rejected"], 1)
                self.assertEqual(status["sequence_gaps"], 1)
                output = Path(directory) / "latest_human_observation.npz"
                status_path = Path(directory) / "status.json"
                self.assertTrue(output.is_file())
                self.assertTrue(status_path.is_file())
                with np.load(output, allow_pickle=False) as data:
                    self.assertEqual(int(data["sequence"]), 22)
                    self.assertEqual(data["skeleton"].shape, (4, 17, 7))
                    self.assertEqual(data["skeleton_raw"].shape, (4, 17, 7))
                    self.assertEqual(data["joint_uncertainty_m"].shape, (4, 17))
                    self.assertEqual(data["joint_prediction_age_ms"].shape, (4, 17))
                    self.assertEqual(data["joint_inferred_mask"].shape, (4, 17))
                    self.assertEqual(data["human_ids"].tolist(), [7, -1, -1, -1])

                # A producer process may reconnect with sequence numbering
                # restarted. A newer capture timestamp starts a fresh Human
                # recurrent segment instead of being rejected as out-of-order.
                with socket.create_connection((host, port), timeout=2.0) as connection:
                    restarted = packet(0, 1_000_000_000, offset_x=0.2)
                    connection.sendall((json.dumps(restarted) + "\n").encode())
                deadline = time.monotonic() + 3.0
                while application.status()["frames_accepted"] < 3:
                    if time.monotonic() >= deadline:
                        self.fail("receiver did not accept restarted producer")
                    time.sleep(0.02)
                latest = application.store.latest()
                self.assertIsNotNone(latest)
                self.assertTrue(latest.human_is_first[0])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
