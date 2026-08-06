import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.compact_skeleton_v3 import (
    CompactSkeletonV3Dataset,
    derive_relative_joint_velocity,
    discover_compact_episodes,
)


class CompactSkeletonV3Test(unittest.TestCase):
    def test_velocity_does_not_treat_uav_translation_as_human_motion(self):
        xyz = np.zeros((2, 1, 12, 3), np.float32)
        xyz[0, 0, :, 0] = 2.0
        xyz[1, 0, :, 0] = 1.9
        valid = np.ones((2, 1, 12), np.bool_)
        ids = np.zeros((2, 1), np.int64)
        ego = np.zeros((2, 14), np.float32)
        ego[:, 13] = 1.0
        ego[1, 0] = 0.1
        velocity = derive_relative_joint_velocity(
            xyz, valid, ids, ego, np.asarray([1.0, 1.1]))
        np.testing.assert_allclose(velocity[0], 0.0)
        np.testing.assert_allclose(velocity[1], 0.0, atol=1e-5)

    def test_direct_chunk_to_factorized_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp) / "episodes" / "episode_a"
            chunks = episode / "chunks"
            chunks.mkdir(parents=True)
            metadata = {
                "schema": "omninxt.crowd_skeleton_state.v3",
                "stored_people_count": 2,
                "joint_topology": "COCO12_BODY",
                "recorded_joint_count": 12,
                "goal_position_episode_local": [5.0, 0.0, 0.0],
            }
            (episode / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            (episode / "summary.json").write_text(json.dumps({
                "termination_reason": "manual_stop",
                "outcome_class": "truncated",
            }), encoding="utf-8")
            t, n, j = 3, 2, 12
            ego = np.zeros((t, 14), np.float32)
            ego[:, 13] = 1.0
            xyz = np.zeros((t, n, j, 3), np.float32)
            xyz[:, 0, :, 0] = 2.0
            valid = np.zeros((t, n, j), np.bool_)
            valid[:, 0] = True
            confidence = valid.astype(np.float32)
            track = np.full((t, n), -1, np.int32)
            track[:, 0] = 0
            np.savez(chunks / "chunk_000000.npz",
                frame_index=np.arange(t, dtype=np.int64),
                simulation_time_s=np.asarray([1.0, 1.1, 1.2]),
                skeleton_timestamp_ns=np.arange(t, dtype=np.int64),
                skeleton_fresh=np.ones(t, np.bool_),
                ego_state=ego,
                ego_altitude_valid=np.ones(t, np.bool_),
                ego_state_source=np.ones(t, np.uint8),
                human_xyz=xyz,
                human_confidence=confidence,
                human_joint_valid=valid,
                human_track_id=track,
                action=np.zeros((t, 4), np.float32),
                action_valid=np.ones(t, np.bool_),
                reward=np.zeros(t, np.float32),
                reward_components=np.zeros((t, 6), np.float32),
                is_first=np.asarray([True, False, False]),
                success=np.asarray([False, False, False]),
                termination_code=np.asarray([0, 0, 9], np.uint8),
                is_last=np.asarray([False, False, True]),
                is_terminal=np.asarray([False, False, False]))
            episodes = discover_compact_episodes(tmp)
            self.assertEqual([(item.name, item.length) for item in episodes],
                             [("episode_a", 3)])
            dataset = CompactSkeletonV3Dataset(
                tmp, sequence_length=3, max_people=1)
            item = dataset[0]
            self.assertEqual(item["skeleton"].shape, (3, 1, 12, 7))
            self.assertEqual(item["human_root"].shape, (3, 1, 10))
            self.assertEqual(item["goal"].shape, (3, 8))
            self.assertTrue(item["human_mask"].all())
            self.assertEqual(item["human_ids"].tolist(), [[1], [1], [1]])
            self.assertTrue(item["is_first"][0, 0])
            self.assertFalse(item["is_terminal"][-1, 0])
            self.assertTrue(item["is_last"][-1, 0])
            self.assertEqual(int(item["termination_code"][-1, 0]), 9)

if __name__ == "__main__":
    unittest.main()
