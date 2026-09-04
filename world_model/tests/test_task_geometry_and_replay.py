from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.compact_skeleton_v3 import CompactSkeletonV3Dataset  # noqa: E402
from modules.task_geometry import (  # noqa: E402
    TASK_GEOMETRY_MAX_RANGE_M,
    WAREHOUSE_V2_KLT134_SCENE_SHA256,
    WAREHOUSE_V2_KLT134_STATIC_AABB_WORLD,
    episode_to_world_flight_bounds_signed_clearance_torch,
    fixed_task_scene_geometry_contract,
    fixed_task_scene_geometry_equivalent,
    flight_bounds_signed_clearance_torch,
    migrate_warehouse_v2_klt134_static_geometry,
    task_geometry_clearance_torch,
    task_geometry_proximity_numpy,
    task_geometry_proximity_torch,
    task_physical_state_dim,
    task_physical_state_numpy,
    task_physical_state_torch,
)


def _write_empty_episode(root: Path, name: str, length: int) -> None:
    episode = root / "episodes" / name
    chunks = episode / "chunks"
    chunks.mkdir(parents=True)
    (episode / "metadata.json").write_text(json.dumps({
        "schema": "omninxt.crowd_skeleton_state.v3",
        "stored_people_count": 1,
        "joint_topology": "COCO12_BODY",
        "recorded_joint_count": 12,
        "goal_position_episode_local": [5.0, 0.0, 0.0],
    }), encoding="utf-8")
    (episode / "summary.json").write_text(json.dumps({
        "termination_reason": "manual_stop",
        "outcome_class": "truncated",
    }), encoding="utf-8")
    ego = np.zeros((length, 14), np.float32)
    ego[:, 13] = 1.0
    np.savez(
        chunks / "chunk_000000.npz",
        frame_index=np.arange(length, dtype=np.int64),
        simulation_time_s=np.arange(length, dtype=np.float64) * 0.1,
        skeleton_fresh=np.ones(length, np.bool_),
        human_xyz=np.zeros((length, 1, 12, 3), np.float32),
        human_confidence=np.zeros((length, 1, 12), np.float32),
        human_joint_valid=np.zeros((length, 1, 12), np.bool_),
        human_track_id=np.full((length, 1), -1, np.int32),
        human_root_velocity=np.zeros((length, 1, 3), np.float32),
        human_velocity_valid=np.zeros((length, 1), np.bool_),
        human_velocity_sigma_mps=np.zeros((length, 1), np.float32),
        human_measurement_age_s=np.zeros((length, 1), np.float32),
        human_track_age_frames=np.zeros((length, 1), np.int32),
        human_prediction_run_frames=np.zeros((length, 1), np.int32),
        human_identity_confidence=np.zeros((length, 1), np.float32),
        ego_state=ego,
        action=np.zeros((length, 4), np.float32),
        action_valid=np.ones(length, np.bool_),
        reward=np.zeros(length, np.float32),
        reward_components=np.zeros((length, 6), np.float32),
        is_first=np.arange(length) == 0,
        success=np.zeros(length, np.bool_),
        termination_code=np.concatenate((
            np.zeros(max(0, length - 1), np.uint8),
            np.asarray([9], np.uint8),
        )),
        is_last=np.arange(length) == length - 1,
        is_terminal=np.zeros(length, np.bool_),
    )


class TaskGeometryAndReplayTest(unittest.TestCase):

    def test_warehouse_klt134_scene_migration_is_hash_gated_and_idempotent(self):
        legacy = np.asarray([
            [-10.630129, -12.50013, -7.029992, -8.899993],
            [-10.20728, -4.098207, -8.432724, -3.149821],
            [-10.207275, 11.849821, -8.432719, 12.798207],
            [-10.119994, -4.063446, -8.52, -3.336557],
            [-10.119994, -0.063446, -8.52, 0.663443],
            [-10.119994, 3.936554, -8.52, 4.663443],
            [-10.119994, 7.936554, -8.52, 8.663443],
            [-10.119994, 11.936554, -8.52, 12.663443],
            [6.119392, -12.50013, 9.71953, -8.899992],
        ], np.float64)
        migrated, applied = migrate_warehouse_v2_klt134_static_geometry(
            [-8.1, 7.2, -10.7, 29.5], legacy,
            maximum_altitude_m=3.0)
        self.assertTrue(applied)
        self.assertEqual(migrated.shape, (10, 4))
        self.assertTrue(any(np.allclose(
            row, WAREHOUSE_V2_KLT134_STATIC_AABB_WORLD,
            rtol=0.0, atol=5.0e-6) for row in migrated))
        scene = fixed_task_scene_geometry_contract(
            [-8.1, 7.2, -10.7, 29.5], migrated,
            maximum_altitude_m=3.0)
        self.assertEqual(scene["sha256"], WAREHOUSE_V2_KLT134_SCENE_SHA256)

        repeated, applied = migrate_warehouse_v2_klt134_static_geometry(
            [-8.1, 7.2, -10.7, 29.5], migrated,
            maximum_altitude_m=3.0)
        self.assertFalse(applied)
        np.testing.assert_allclose(repeated, migrated, atol=0.0)
        with self.assertRaisesRegex(ValueError, "unknown fixed scene"):
            migrate_warehouse_v2_klt134_static_geometry(
                [-8.1, 7.2, -10.7, 29.5], legacy[:-1],
                maximum_altitude_m=3.0)

    def test_fixed_scene_sort_is_stable_across_float_transport_ties(self):
        # These fixtures share one six-decimal x extent, but their raw float64
        # x values differ by a few ulps.  Canonical order must be determined
        # after quantization so a float32 replay and float64 request remain the
        # same scene rather than permuting the rows by transport noise.
        exact = np.asarray([
            [-10.119994265987586, -0.06344554557064808,
             -8.520000434562919, 0.6634427951133073],
            [-10.119994265987582, 3.936554365022385,
             -8.520000434562915, 4.66344270570634],
            [-10.119994265987586, 7.936554275615422,
             -8.520000434562919, 8.663442616299376],
            [-10.119994265987579, 11.936554186208454,
             -8.520000434562911, 12.663442526892409],
            [-10.119994265987582, -4.063445456163682,
             -8.520000434562915, -3.3365571154797267],
        ], np.float64)
        replay = fixed_task_scene_geometry_contract(
            [-8.1, 7.2, -10.7, 29.5], exact.astype(np.float32),
            maximum_altitude_m=3.0,
        )
        request = fixed_task_scene_geometry_contract(
            [-8.1, 7.2, -10.7, 29.5], exact,
            maximum_altitude_m=3.0,
        )
        self.assertTrue(fixed_task_scene_geometry_equivalent(replay, request))
        self.assertEqual(
            np.argsort(np.asarray(replay["static_obstacle_aabbs_world"])[:, 1],
                       kind="stable").tolist(),
            list(range(exact.shape[0])),
        )
        self.assertEqual(
            np.argsort(np.asarray(request["static_obstacle_aabbs_world"])[:, 1],
                       kind="stable").tolist(),
            list(range(exact.shape[0])),
        )

    def test_signed_bounds_match_simulator_strict_watchdog_contract(self):
        bounds = torch.tensor([-8.1, 7.2, -10.7, 29.5])
        position = torch.tensor([
            [-8.1, 0.0], [7.2, 29.5], [-8.2, 0.0], [0.0, 29.6],
        ])
        signed = flight_bounds_signed_clearance_torch(position, bounds)
        torch.testing.assert_close(
            signed, torch.tensor([0.0, 0.0, -0.1, -0.1]),
            atol=1.0e-6, rtol=0.0)
        self.assertEqual(signed.lt(0.0).tolist(), [False, False, True, True])

    def test_episode_local_bounds_transform_uses_recorded_world_origin(self):
        ego = torch.zeros(1, 2, 14)
        ego[..., 13] = 1.0
        ego[0, 1, 0] = 2.1
        origin = torch.tensor([[[10.0, 20.0, 1.0]]]).expand(1, 2, 3)
        yaw = torch.full((1, 2), torch.pi / 2.0)
        bounds = torch.tensor([[[8.0, 12.0, 18.0, 22.0]]]).expand(1, 2, 4)
        signed = episode_to_world_flight_bounds_signed_clearance_torch(
            ego, origin, yaw, bounds)
        torch.testing.assert_close(
            signed, torch.tensor([[2.0, -0.1]]),
            atol=1.0e-5, rtol=0.0)

    def test_empty_static_obstacle_list_matches_torch(self):
        position = np.asarray([0.0, 0.0], np.float32)
        yaw = np.asarray(0.0, np.float32)
        bounds = np.asarray([-4.0, 14.0, -4.0, 4.0], np.float32)
        obstacles = np.empty((0, 4), np.float32)

        numpy_value = task_geometry_proximity_numpy(
            position, yaw, bounds, obstacles)
        torch_value = task_geometry_proximity_torch(
            torch.from_numpy(position), torch.from_numpy(yaw),
            torch.from_numpy(bounds), torch.from_numpy(obstacles),
        ).numpy()

        self.assertEqual(numpy_value.shape, (8,))
        self.assertTrue(np.isfinite(numpy_value).all())
        np.testing.assert_allclose(numpy_value, torch_value, atol=1.0e-6)

    def test_numpy_and_torch_geometry_are_identical(self):
        position = np.asarray(((0.0, 0.0), (-1.5, 1.5)), np.float32)
        yaw = np.asarray((0.0, 0.37), np.float32)
        bounds = np.asarray(((-3.0, 4.0, -2.0, 5.0),) * 2, np.float32)
        obstacles = np.asarray((
            ((1.0, -0.3, 1.5, 0.3), (-2.0, 1.0, -1.0, 2.0)),
            ((1.0, -0.3, 1.5, 0.3), (-2.0, 1.0, -1.0, 2.0)),
        ), np.float32)
        valid = np.asarray(((True, False), (True, True)), np.bool_)
        numpy_value = task_geometry_proximity_numpy(
            position, yaw, bounds, obstacles, valid)
        torch_value = task_geometry_proximity_torch(
            torch.from_numpy(position), torch.from_numpy(yaw),
            torch.from_numpy(bounds), torch.from_numpy(obstacles),
            torch.from_numpy(valid),
        ).numpy()
        np.testing.assert_allclose(numpy_value, torch_value, atol=1.0e-6)
        # Facing +x from the origin hits the obstacle at 1 m, not the bound.
        self.assertAlmostEqual(
            float(numpy_value[0, 0]),
            1.0 - 1.0 / TASK_GEOMETRY_MAX_RANGE_M,
            places=6,
        )
        clearance = task_geometry_clearance_torch(
            torch.from_numpy(position), torch.from_numpy(bounds),
            torch.from_numpy(obstacles), torch.from_numpy(valid),
        )
        torch.testing.assert_close(
            clearance,
            torch.tensor((1.0, 0.0)),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_complete_task_state_matches_numpy_and_reads_every_aabb_field(self):
        position = np.asarray(((0.2, -0.4, 1.1), (1.0, 1.5, 1.4)), np.float32)
        yaw = np.asarray((0.31, -0.47), np.float32)
        bounds = np.asarray(((-3.0, 4.0, -2.0, 5.0),) * 2, np.float32)
        obstacles = np.asarray((
            ((1.0, -0.3, 1.5, 0.3), (-2.0, 1.0, -1.0, 2.0)),
            ((1.0, -0.3, 1.5, 0.3), (-2.0, 1.0, -1.0, 2.0)),
        ), np.float32)
        valid = np.asarray(((True, False), (True, True)), np.bool_)
        numpy_value = task_physical_state_numpy(
            position, yaw, bounds, obstacles, valid,
            maximum_altitude_m=np.asarray((3.0, 3.0), np.float32),
            maximum_altitude_valid=np.asarray((True, True)),
            maximum_obstacles=3,
        )
        torch_value = task_physical_state_torch(
            torch.from_numpy(position), torch.from_numpy(yaw),
            torch.from_numpy(bounds), torch.from_numpy(obstacles),
            torch.from_numpy(valid),
            maximum_altitude_m=torch.tensor((3.0, 3.0)),
            maximum_altitude_valid=torch.ones(2, dtype=torch.bool),
            maximum_obstacles=3,
        )
        self.assertEqual(numpy_value.shape, (2, task_physical_state_dim(3)))
        np.testing.assert_allclose(
            numpy_value, torch_value.detach().numpy(), atol=1.0e-6)

        differentiable_obstacles = torch.tensor(
            ((1.0, -0.3, 1.5, 0.3), (-2.0, 1.0, -1.0, 2.0)),
            requires_grad=True,
        )
        jacobian = torch.autograd.functional.jacobian(
            lambda value: task_physical_state_torch(
                torch.tensor((0.2, -0.4, 1.1)), torch.tensor(0.31),
                torch.tensor((-3.0, 4.0, -2.0, 5.0)), value,
                maximum_altitude_m=torch.tensor(3.0),
                maximum_altitude_valid=True,
                maximum_obstacles=2,
            ),
            differentiable_obstacles,
        )
        per_aabb_coordinate = jacobian.flatten(0, -3).square().sum(0).sqrt()
        self.assertEqual(per_aabb_coordinate.shape, (2, 4))
        self.assertTrue(per_aabb_coordinate.gt(0.0).all())

    def test_exact_task_state_retains_aabb_that_misses_all_eight_rays(self):
        position_xy = np.zeros(2, np.float32)
        position_xyz = np.zeros(3, np.float32)
        bounds = np.asarray((-8.0, 8.0, -8.0, 8.0), np.float32)
        first = np.asarray(((2.9, 1.1, 3.1, 1.3),), np.float32)
        second = np.asarray(((3.9, 1.1, 4.1, 1.3),), np.float32)
        np.testing.assert_allclose(
            task_geometry_proximity_numpy(position_xy, 0.0, bounds, first),
            task_geometry_proximity_numpy(position_xy, 0.0, bounds, second),
            atol=0.0,
        )
        first_state = task_physical_state_numpy(
            position_xyz, 0.0, bounds, first, maximum_obstacles=1)
        second_state = task_physical_state_numpy(
            position_xyz, 0.0, bounds, second, maximum_obstacles=1)
        self.assertFalse(np.array_equal(first_state, second_state))

    def test_transition_unique_windows_include_short_episodes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_empty_episode(root, "episode_short", 3)
            _write_empty_episode(root, "episode_long", 8)
            dataset = CompactSkeletonV3Dataset(
                root,
                sequence_length=4,
                max_people=1,
                transition_unique_windows=True,
            )
            self.assertEqual(len(dataset), 4)
            self.assertTrue(any(start < 0 for _, start in dataset.windows))

            transitions: Counter[tuple[str, int, int]] = Counter()
            real_rows: dict[str, set[int]] = {
                "episode_short": set(), "episode_long": set()}
            for index in range(len(dataset)):
                item = dataset[index]
                valid = item["sequence_valid"][:, 0]
                frames = item["frame_ids"]
                real_rows[item["record_name"]].update(
                    int(value) for value in frames[valid])
                for source in range(len(frames) - 1):
                    if valid[source] and valid[source + 1]:
                        transitions[(
                            item["record_name"],
                            int(frames[source]),
                            int(frames[source + 1]),
                        )] += 1
                first_real = int(np.flatnonzero(valid)[0])
                self.assertTrue(item["is_first"][first_real, 0])
                self.assertEqual(
                    bool(item["physical_is_first"][first_real, 0]),
                    int(frames[first_real]) == 0,
                )

            expected = {
                (name, source, source + 1)
                for name, length in (("episode_short", 3), ("episode_long", 8))
                for source in range(length - 1)
            }
            self.assertEqual(set(transitions), expected)
            self.assertTrue(all(count == 1 for count in transitions.values()))
            self.assertEqual(real_rows["episode_short"], set(range(3)))
            self.assertEqual(real_rows["episode_long"], set(range(8)))


if __name__ == "__main__":
    unittest.main()
