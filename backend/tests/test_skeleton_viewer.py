from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from backend.skeleton_viewer.server import SnapshotReader, ViewerServer


class SkeletonViewerTest(unittest.TestCase):
    def test_waiting_without_receiver_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = SnapshotReader(directory).payload()
        self.assertEqual(payload["state"], "waiting")
        self.assertEqual(payload["people"], [])
        self.assertTrue(payload["stale"])

    def test_snapshot_preserves_xy_ground_z_up_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skeleton = np.zeros((2, 17, 7), dtype=np.float32)
            skeleton[0, 11, :3] = [2.0, 0.5, -0.2]
            skeleton[0, 12, :3] = [2.0, -0.5, -0.2]
            skeleton_raw = skeleton.copy()
            skeleton_raw[0, 11, 0] = 9.0
            human_mask = np.asarray([True, False])
            joint_mask = np.zeros((2, 17), dtype=np.bool_)
            joint_mask[0, 11:13] = True
            joint_inferred_mask = np.zeros((2, 17), dtype=np.bool_)
            joint_inferred_mask[0, 12] = True
            np.savez(
                root / "latest_human_observation.npz",
                sequence=np.asarray(9), timestamp_ns=np.asarray(123),
                skeleton=skeleton, skeleton_raw=skeleton_raw,
                human_mask=human_mask,
                joint_mask=joint_mask, human_ids=np.asarray([0, -1]),
                joint_inferred_mask=joint_inferred_mask,
            )
            (root / "status.json").write_text(json.dumps({
                "last_receive_wall_ns": time.time_ns(),
            }), encoding="utf-8")

            payload = SnapshotReader(root).payload()

        self.assertEqual(payload["state"], "ready")
        self.assertEqual(payload["frame_id"], "base_link")
        self.assertEqual(payload["coordinate_convention"],
                         "+X forward, +Y left, +Z up")
        self.assertEqual(payload["people"][0]["id"], 0)
        self.assertEqual(payload["people"][0]["joints"][11], [2.0, 0.5, -0.2])
        self.assertEqual(payload["pose_source"], "causal_kinematic_refinement")
        self.assertEqual(payload["motion_prior"], "upright_walking")
        self.assertEqual(payload["valid_joints"], 2)
        self.assertEqual(payload["inferred_joints"], 1)
        self.assertEqual(payload["people"][0]["inferred"][11:13], [False, True])
        self.assertFalse(payload["stale"])

    def test_http_health_and_frame_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ViewerServer(("127.0.0.1", 0), SnapshotReader(directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/healthz", timeout=2) as response:
                    self.assertEqual(json.load(response), {"ok": True})
                with urlopen(base + "/api/frame", timeout=2) as response:
                    self.assertEqual(json.load(response)["state"], "waiting")
                with urlopen(base + "/", timeout=2) as response:
                    self.assertIn("XY 为地面平面", response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
