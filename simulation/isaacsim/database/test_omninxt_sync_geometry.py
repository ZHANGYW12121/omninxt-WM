#!/usr/bin/env python3
"""CPU-only acceptance tests for the OmniNxt 20260802 Isaac profiles."""

import json
import unittest
from pathlib import Path

import numpy as np
import yaml

from omni_depth_exporter import IsaacQuadcamExporter
from omninxt_projection import MeiRadtanRemapper


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "omninxt_sync_20260802_camera_config.json"
GEOMETRY = ROOT.parent.parent / "omnidepth" / "geometry" / "sim_geometry.yaml"


class SyncGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.geometry = yaml.safe_load(GEOMETRY.read_text(encoding="utf-8"))

    def test_profile_layout(self):
        raw = self.config["profiles"]["raw_mei"]
        validation = self.config["profiles"]["rectified_validation"]
        self.assertEqual(raw["camera_order"], ["cam0", "cam1", "cam2", "cam3"])
        self.assertEqual(len(raw["cameras"]), 4)
        self.assertEqual(len(validation["cameras"]), 12)
        self.assertEqual(validation["active_camera"], "rect3_left")

    def test_raw_transforms_equal_authoritative_geometry(self):
        raw = self.config["profiles"]["raw_mei"]
        for index, name in enumerate(("CAM_A", "CAM_B", "CAM_C", "CAM_D")):
            actual = np.asarray(raw["cameras"][f"cam{index}"]["T_cam_imu"])
            expected = np.asarray(
                self.geometry["raw_cameras"][name]["T_camera_base_link"]
            )
            np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_rectified_intrinsics_and_baselines(self):
        validation = self.config["profiles"]["rectified_validation"]["cameras"]
        for pair in self.geometry["stereo_pairs"]:
            pair_id = pair["id"]
            left = validation[f"rect{pair_id}_left"]
            right = validation[f"rect{pair_id}_right"]
            p1 = np.asarray(pair["P1_rectified_left"])
            p2 = np.asarray(pair["P2_rectified_right"])
            np.testing.assert_allclose(
                left["intrinsics"], [p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]]
            )
            np.testing.assert_allclose(
                right["intrinsics"], [p2[0, 0], p2[1, 1], p2[0, 2], p2[1, 2]]
            )
            left_body = np.linalg.inv(np.asarray(left["T_cam_imu"]))
            right_body = np.linalg.inv(np.asarray(right["T_cam_imu"]))
            baseline = np.linalg.norm(left_body[:3, 3] - right_body[:3, 3])
            self.assertAlmostEqual(baseline, pair["baseline_m"], places=10)

    def test_mei_inverse_and_bridge_coverage(self):
        raw = self.config["profiles"]["raw_mei"]["cameras"]
        expected_minimum = {"cam0": 0.999, "cam1": 0.999, "cam2": 0.999,
                            "cam3": 0.65}
        for name, spec in raw.items():
            remapper = MeiRadtanRemapper(
                spec["intrinsics"], spec["distortion"],
                spec["target_resolution"], spec["render_resolution"],
                spec["render_fov_deg"],
            )
            self.assertGreaterEqual(
                remapper.diagnostics.valid_fraction, expected_minimum[name]
            )
            self.assertLess(remapper.diagnostics.max_forward_error_px, 0.02)
            source = np.zeros((1280, 1280, 3), dtype=np.uint8)
            output = remapper.remap(source)
            self.assertEqual(output.shape, (720, 1280, 3))

    def test_range_is_converted_to_optical_z(self):
        spec = self.config["profiles"]["rectified_validation"]["cameras"][
            "rect0_left"
        ]
        distance = np.full((240, 320), 2.0, dtype=np.float32)
        depth = IsaacQuadcamExporter._pinhole_range_to_z(distance, spec)
        fx, fy, cx, cy = spec["intrinsics"]
        self.assertAlmostEqual(depth[int(round(cy)), int(round(cx))], 2.0, places=3)
        self.assertLess(depth[0, 0], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
