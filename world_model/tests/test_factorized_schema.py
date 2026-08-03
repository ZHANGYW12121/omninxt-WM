import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.factorized_schema import (
    FactorizedIsaacAdapter, HumanDetections, assign_stable_human_slots, detections_from_3d_pose_cache,
    ego14_from_frame, ego14_reference_from_frame, validate_factorized_batch,
)


def det(ids, risks, *, order=None, joints=3):
    ids = np.asarray(ids, np.int64)
    risks = np.asarray(risks, np.float32)
    skeleton = np.zeros((len(ids), joints, 7), np.float32)
    for i, track_id in enumerate(ids):
        skeleton[i, :, 0] = track_id
    mask = np.ones((len(ids), joints), np.bool_)
    if order is not None:
        ids, risks, skeleton, mask = ids[order], risks[order], skeleton[order], mask[order]
    return HumanDetections(ids, skeleton, mask, risks)


class FactorizedSchemaTest(unittest.TestCase):
    def test_ego14_semantics(self):
        first = {"drone_state": {"position": [1, 2, 3], "yaw": np.pi / 2}}
        frame = {"drone_state": {"position": [1, 3, 5], "velocity": [0, 4, 6],
                 "acceleration": [0, 7, 9], "altitude_agl": 2,
                 "roll_pitch_yaw_rad": [.1, .2, np.pi / 2]}}
        state = ego14_from_frame(frame, ego14_reference_from_frame(first))
        self.assertEqual(state.shape, (14,))
        np.testing.assert_allclose(state[:12], [1,0,2,4,0,6,7,0,9,2,.1,.2], atol=1e-6)
        np.testing.assert_allclose(state[12:], [0, 1], atol=1e-6)

    def test_altitude_is_not_silently_world_z(self):
        frame = {"drone_state": {"position": [0, 0, 8]}}
        ref = ego14_reference_from_frame({"drone_state": {"position": [0, 0, 6]}})
        self.assertEqual(float(ego14_from_frame(frame, ref)[9]), 2.0)
        with self.assertRaisesRegex(ValueError, "Missing altitude AGL"):
            ego14_from_frame(frame, ref, strict_altitude=True)

    def test_detector_reordering_does_not_reorder_slots(self):
        frames = [det([11, 22], [.8, .9]), det([11, 22], [.8, .9], order=[1, 0])]
        out = assign_stable_human_slots(frames, max_people=3, joint_count=3, feat_dim=7)
        for track_id in (11, 22):
            self.assertEqual(np.flatnonzero(out.human_ids[0] == track_id).item(),
                             np.flatnonzero(out.human_ids[1] == track_id).item())
        self.assertFalse(out.human_is_first[1].any())

    def test_absence_and_slot_reassignment_reset(self):
        frames = [det([1], [1]), det([], []), det([2], [1]), det([1], [1])]
        out = assign_stable_human_slots(frames, max_people=1, joint_count=3, feat_dim=7)
        self.assertTrue(out.human_is_first[0, 0])
        self.assertFalse(out.human_mask[1, 0])
        self.assertTrue(out.human_is_first[2, 0])
        self.assertTrue(out.human_is_first[3, 0])

    def test_overflow_is_deterministic_risk_selection(self):
        out = assign_stable_human_slots([det([3, 1, 2], [.2, .9, .6])],
                                        max_people=2, joint_count=3, feat_dim=7)
        self.assertEqual(set(out.human_ids[0, out.human_mask[0]].tolist()), {1, 2})
        self.assertEqual(out.truncated_people.tolist(), [1])

    def test_strict_3d_cache_and_batch_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.npz"
            np.savez(path, keypoints_xyz=np.ones((1, 3, 3), np.float32),
                     track_ids=np.array([7]), confidence=np.ones((1, 3), np.float32))
            detection = detections_from_3d_pose_cache(path)
            out = assign_stable_human_slots([detection], max_people=2, joint_count=3, feat_dim=7)
            ego = ego14_from_frame({"drone_state": {}})[None]
            validate_factorized_batch({"ego_state": ego, "skeleton": out.skeleton,
                "human_mask": out.human_mask, "joint_mask": out.joint_mask,
                "human_ids": out.human_ids, "human_is_first": out.human_is_first,
                "goal": np.zeros((1, 8), np.float32),
                "goal_position": np.zeros((1, 3), np.float32)})
            np.savez(path, keypoints_xyc=np.ones((1, 3, 3), np.float32), track_ids=[7])
            with self.assertRaisesRegex(ValueError, "cannot be used as a 3D"):
                detections_from_3d_pose_cache(path)

    def test_dataset_adapter_outputs_training_schema(self):
        class Episode:
            record_name = "record_a"
            record_dir = ""

        class Base:
            episodes = [Episode()]
            def __len__(self): return 1
            def __getitem__(self, index):
                return {"record_name": "record_a", "frame_ids": np.array([0, 1]),
                        "ego_state": np.zeros((2, 17), np.float32),
                        "lidar": np.zeros((2, 4, 3), np.float32),
                        "lidar_mask": np.ones((2, 4), np.bool_)}

        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "raw" / "record_a"
            (record / "frames").mkdir(parents=True)
            Base.episodes[0].record_dir = str(record)
            pose_dir = Path(tmp) / "pose" / "record_a"
            pose_dir.mkdir(parents=True)
            for frame_id in (0, 1):
                frame = {"drone_state": {"position": [frame_id, 0, 1], "yaw": 0},
                         "goal_point": [5, 0, 1]}
                (record / "frames" / f"frame_{frame_id:06d}.json").write_text(json.dumps(frame))
                np.savez(pose_dir / f"frame_{frame_id:06d}_pose.npz",
                         keypoints_xyz=np.ones((1, 3, 3), np.float32), track_ids=[5])
            item = FactorizedIsaacAdapter(Base(), Path(tmp) / "pose", max_people=2,
                                          num_joints=3)[0]
            self.assertEqual(item["ego_state"].shape, (2, 14))
            self.assertEqual(item["goal"].shape, (2, 8))
            self.assertEqual(item["goal_position"].shape, (2, 3))
            self.assertEqual(item["human_root"].shape, (2, 2, 10))
            self.assertNotIn("point_cloud", item)
            self.assertTrue(item["human_is_first"][0].any())
            self.assertFalse(item["human_is_first"][1].any())


if __name__ == "__main__":
    unittest.main()
