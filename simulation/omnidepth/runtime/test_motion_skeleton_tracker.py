#!/usr/bin/env python3
import unittest
from collections import deque

import numpy as np

from motion_skeleton_tracker import (
    MotionSkeletonTracker,
    _hungarian_assignment,
    _optimal_assignment,
    _robust_history_velocity,
)
from skeleton_stream import StgcnWindow, build_skeleton_packet


JOINTS = tuple("joint_{}".format(index) for index in range(17))
LOCAL_BODY = np.asarray([
    [0.00, 0.00, 0.82],
    [0.00, 0.04, 0.86], [0.00, -0.04, 0.86],
    [0.00, 0.09, 0.82], [0.00, -0.09, 0.82],
    [0.00, 0.20, 0.52], [0.00, -0.20, 0.52],
    [0.00, 0.34, 0.27], [0.00, -0.34, 0.27],
    [0.00, 0.40, 0.02], [0.00, -0.40, 0.02],
    [0.00, 0.15, 0.00], [0.00, -0.15, 0.00],
    [0.00, 0.15, -0.45], [0.00, -0.15, -0.45],
    [0.00, 0.15, -0.90], [0.00, -0.15, -0.90],
], dtype=np.float64)


def person(x, y=0.0, z=1.0):
    joints = []
    for joint_id, name in enumerate(JOINTS):
        xyz = (np.array([x, y, z]) + LOCAL_BODY[joint_id]).tolist()
        joints.append({
            "id": joint_id, "name": name, "xyz_imu_m": xyz,
            "score": 0.9, "source": "isaac_gt_depth",
            "measurement_sigma_m": 0.01,
        })
    return {"source_pairs": ["AB_RIGHT"], "joints": joints}


def centers_by_id(output):
    result = {}
    for value in output:
        points = [value["joints"][index]["xyz_imu_m"]
                  for index in (5, 6, 11, 12)]
        result[value["person_id"]] = np.median(
            np.asarray(points, dtype=np.float64), axis=0)
    return result


class AssignmentTest(unittest.TestCase):
    def test_dense_assignment_maximizes_cardinality_before_cost(self):
        costs = np.full((13, 13), np.inf, dtype=np.float64)
        costs[0, 0], costs[0, 1] = 1.0, 2.0
        costs[1, 0] = 1.5
        for index in range(2, 13):
            costs[index, index] = 0.1
        pairs = _optimal_assignment(costs)
        self.assertEqual(len(pairs), 13)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 0), pairs)

    def test_hungarian_handles_rectangular_forbidden_matrix(self):
        costs = np.asarray([
            [0.2, np.inf, 1.0],
            [0.1, 0.3, np.inf],
            [np.inf, 0.2, 0.4],
            [np.inf, np.inf, np.inf],
        ])
        pairs = _hungarian_assignment(costs)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len({row for row, _ in pairs}), 3)
        self.assertEqual(len({col for _, col in pairs}), 3)
        self.assertTrue(all(np.isfinite(costs[row, col]) for row, col in pairs))

    def test_hungarian_minimizes_total_cost(self):
        costs = np.asarray([
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ])
        pairs = _hungarian_assignment(costs)
        self.assertEqual(set(pairs), {(0, 1), (1, 0), (2, 2)})
        self.assertEqual(sum(costs[row, col] for row, col in pairs), 5.0)


class MotionSkeletonTrackerTest(unittest.TestCase):
    @staticmethod
    def ego_pose(x=0.0, y=0.0, z=0.0, yaw_rad=0.0):
        return {
            "position_world_m": np.asarray([x, y, z], np.float64),
            "attitude_xyzw": np.asarray([
                0.0, 0.0, np.sin(yaw_rad * 0.5),
                np.cos(yaw_rad * 0.5),
            ], np.float64),
        }

    def test_causal_history_velocity_rejects_one_position_outlier(self):
        expected = np.asarray([0.8, -0.2, 0.0], np.float64)
        history = deque(maxlen=15)
        for index in range(9):
            stamp = index * 0.1
            position = expected * stamp
            if index == 4:
                position = position + np.asarray([1.8, -1.2, 0.5])
            history.append((stamp, position))
        velocity, sigma, valid = _robust_history_velocity(history, 2.0)
        self.assertTrue(valid)
        np.testing.assert_allclose(velocity, expected, atol=0.05)
        self.assertLess(sigma, 0.05)

    def test_packet_exposes_recorded_velocity_quality_contract(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        output = []
        for index in range(5):
            output = tracker.update(
                [person(0.1 * index)], index * 100_000_000)
        tracked = output[0]
        self.assertTrue(tracked["velocity_valid"])
        self.assertEqual(len(tracked["root_velocity_base_link_mps"]), 3)
        self.assertIsInstance(tracked["velocity_sigma_mps"], float)
        self.assertEqual(tracked["consecutive_prediction_frames"], 0)

    def test_reset_session_discards_tracks_and_changes_identity_namespace(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        tracker.update([person(0.0)], 40_000_000_000)
        old_session = tracker.session_id
        self.assertEqual(tracker.status()["active_tracks"], 1)

        tracker.reset_session()

        self.assertNotEqual(tracker.session_id, old_session)
        self.assertEqual(tracker.status()["active_tracks"], 0)
        output = tracker.update([person(1.0)], 100_000_000)
        self.assertEqual(output[0]["person_id"], 1)
        self.assertTrue(output[0]["track_uid"].startswith(
            tracker.session_id + ":"))

    def test_two_people_keep_directional_ids_through_crossing(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        sequence = [
            (-1.0, 1.0), (-0.72, 0.72), (-0.42, 0.42), (-0.14, 0.14),
            (0.14, -0.14), (0.42, -0.42), (0.72, -0.72),
        ]
        output = None
        for index, (first, second) in enumerate(sequence):
            measurements = sorted([person(first), person(second)],
                                  key=lambda value: value["joints"][5][
                                      "xyz_imu_m"][0])
            output = tracker.update(measurements, index * 100_000_000)
        centers = centers_by_id(output)
        self.assertGreater(centers[1][0], 0.45)
        self.assertLess(centers[2][0], -0.45)

    def test_sixty_people_use_hungarian_and_keep_ids(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        first = [person(index * 0.30 - 8.85, (index % 3) * 0.8)
                 for index in range(60)]
        initial = tracker.update(first, 0)
        initial_centers = centers_by_id(initial)
        second = [person(index * 0.30 - 8.83, (index % 3) * 0.8)
                  for index in reversed(range(60))]
        updated = tracker.update(second, 100_000_000)
        updated_centers = centers_by_id(updated)
        self.assertEqual(len(updated), 60)
        self.assertEqual(tracker.status()["assignment_method"], "hungarian")
        for track_id, center in initial_centers.items():
            self.assertLess(abs(updated_centers[track_id][0] - center[0]), 0.1)

    def test_occlusion_prediction_and_session_metadata(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, prediction_timeout=1.2,
            enable_ray_recovery=False)
        tracker.update([person(0.0)], 0)
        tracker.update([person(0.2)], 100_000_000)
        predicted = tracker.update([], 300_000_000)
        self.assertEqual(predicted[0]["person_id"], 1)
        self.assertEqual(predicted[0]["track_state"], "predicted")
        self.assertTrue(predicted[0]["track_uid"].startswith(
            tracker.session_id + ":"))

    def test_translation_is_removed_before_occlusion_prediction(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, prediction_timeout=1.2,
            enable_ray_recovery=False)
        tracker.update(
            [person(5.0)], 0, ego_pose=self.ego_pose())
        predicted = tracker.update(
            [], 100_000_000, ego_pose=self.ego_pose(x=1.0))
        center = centers_by_id(predicted)[1]
        self.assertAlmostEqual(center[0], 4.0, places=5)
        self.assertAlmostEqual(center[1], 0.0, places=5)
        self.assertEqual(tracker.status()["ego_motion_updates"], 1)

    def test_yaw_is_removed_before_association(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, base_gate=0.2,
            enable_ray_recovery=False)
        tracker.update(
            [person(5.0, 0.0)], 0, ego_pose=self.ego_pose())
        output = tracker.update(
            [person(0.0, -5.0)], 100_000_000,
            ego_pose=self.ego_pose(yaw_rad=np.pi / 2.0))
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["person_id"], 1)
        center = centers_by_id(output)[1]
        np.testing.assert_allclose(center[:2], [0.0, -5.0], atol=0.05)

    def test_session_reset_discards_previous_ego_pose(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        tracker.update([person(5.0)], 0, ego_pose=self.ego_pose())
        tracker.reset_session()
        output = tracker.update(
            [person(5.0)], 100_000_000, ego_pose=self.ego_pose(x=20.0))
        self.assertEqual(len(output), 1)
        self.assertAlmostEqual(centers_by_id(output)[1][0], 5.0, places=5)
        self.assertEqual(tracker.status()["ego_motion_updates"], 0)

    def test_individual_missing_joint_is_predicted_then_expires(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, prediction_timeout=1.2,
            enable_ray_recovery=False)
        tracker.update([person(0.0)], 0)
        partial = person(0.02)
        partial["joints"][5]["xyz_imu_m"] = None
        output = tracker.update([partial], 100_000_000)
        self.assertTrue(output[0]["joints"][5]["predicted"])
        self.assertEqual(output[0]["joints"][5]["source"],
                         "motion_prediction")
        output = tracker.update([partial], 1_300_000_000)
        self.assertIsNone(output[0]["joints"][5]["xyz_imu_m"])
        self.assertFalse(output[0]["joints"][5]["predicted"])

    def test_kalman_rejects_isolated_joint_depth_jump(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        baseline = tracker.update([person(2.0)], 0)[0]
        baseline_joint = np.asarray(
            baseline["joints"][9]["xyz_imu_m"], dtype=np.float64)
        corrupted = person(2.0)
        corrupted["joints"][9]["xyz_imu_m"][0] += 2.0
        output = tracker.update([corrupted], 100_000_000)[0]
        filtered_joint = np.asarray(
            output["joints"][9]["xyz_imu_m"], dtype=np.float64)
        self.assertLess(np.linalg.norm(filtered_joint - baseline_joint), 0.10)
        self.assertGreaterEqual(
            tracker.status()["kalman_measurements_rejected"], 1)

    def test_coherent_full_body_jump_relocalizes_instead_of_becoming_sparse(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        tracker.update([person(2.0)], 0)
        output = tracker.update([person(2.72)], 100_000_000)
        self.assertEqual(len(output), 1)
        self.assertEqual(sum(
            joint["xyz_imu_m"] is not None
            for joint in output[0]["joints"]), 17)
        self.assertAlmostEqual(
            centers_by_id(output)[1][0], 2.72, places=2)
        self.assertEqual(
            tracker.status()["coherent_relocalizations"], 1)

    def test_side_view_partial_body_can_coherently_relocalize(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        tracker.update([person(2.0)], 0)
        partial = person(2.60)
        visible_side = {5, 7, 9, 11, 13, 15}
        for joint in partial["joints"]:
            if joint["id"] not in visible_side:
                joint["xyz_imu_m"] = None
        output = tracker.update([partial], 100_000_000)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["person_id"], 1)
        self.assertAlmostEqual(
            output[0]["joints"][5]["xyz_imu_m"][0], 2.60, places=2)
        self.assertEqual(
            tracker.status()["coherent_relocalizations"], 1)

    def test_turning_body_temporally_corrects_symmetric_label_flip(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        tracker.update([person(2.0)], 0)
        flipped = person(2.02)
        for left_id, right_id in ((5, 6), (7, 8), (9, 10),
                                  (11, 12), (13, 14), (15, 16)):
            left_xyz = flipped["joints"][left_id]["xyz_imu_m"]
            right_xyz = flipped["joints"][right_id]["xyz_imu_m"]
            flipped["joints"][left_id]["xyz_imu_m"] = right_xyz
            flipped["joints"][right_id]["xyz_imu_m"] = left_xyz
        output = tracker.update([flipped], 100_000_000)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["person_id"], 1)
        self.assertGreaterEqual(
            tracker.status()["symmetric_joint_pair_swaps"], 6)

    def test_two_point_measurement_never_creates_articulated_track(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        sparse = person(2.0)
        for joint in sparse["joints"]:
            if joint["id"] not in (5, 6):
                joint["xyz_imu_m"] = None
        self.assertEqual(tracker.update([sparse], 0), [])
        self.assertEqual(tracker.status()["active_tracks"], 0)
        self.assertEqual(
            tracker.status()["sparse_measurements_rejected"], 1)

    def test_sparse_measurements_do_not_refresh_a_complete_track_forever(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, prediction_timeout=1.2,
            enable_ray_recovery=False)
        tracker.update([person(2.0)], 0)
        sparse = person(2.1)
        for joint in sparse["joints"]:
            if joint["id"] not in (5, 6):
                joint["xyz_imu_m"] = None
        predicted = tracker.update([sparse], 500_000_000)
        self.assertEqual(len(predicted), 1)
        self.assertGreaterEqual(sum(
            joint["xyz_imu_m"] is not None
            for joint in predicted[0]["joints"]), 6)
        self.assertEqual(
            tracker.update([sparse], 1_300_000_000), [])

    def test_learned_bones_stabilize_noisy_forearm(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        lengths = []
        for index in range(24):
            measured = person(2.0 + 0.02 * index)
            measured["joints"][9]["xyz_imu_m"][1] += (
                0.11 if index % 2 else -0.11)
            output = tracker.update([measured], index * 100_000_000)[0]
            elbow = np.asarray(output["joints"][7]["xyz_imu_m"])
            wrist = np.asarray(output["joints"][9]["xyz_imu_m"])
            if index >= 10:
                lengths.append(float(np.linalg.norm(wrist - elbow)))
        self.assertLess(max(lengths) - min(lengths), 0.035)
        self.assertGreaterEqual(
            tracker.status()["tracks"][0]["bone_constraints"], 8)

    def test_detection_only_obstacle_requires_persistent_confirmation(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        fallback = person(2.0)
        fallback["detection_only_obstacle"] = True
        for index in range(9):
            output = tracker.update([fallback], index * 100_000_000)
            self.assertEqual(output, [])
        output = tracker.update([fallback], 900_000_000)
        self.assertEqual(len(output), 1)
        self.assertEqual(
            tracker.status()["tracks"][0]["required_hits"], 10)

    def test_implausible_full_body_is_not_promoted_to_track(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        impossible = person(2.0)
        root = np.asarray([2.0, 0.0, 1.0])
        for joint in impossible["joints"]:
            joint["xyz_imu_m"] = (
                root + 4.0 * LOCAL_BODY[joint["id"]]).tolist()
        self.assertEqual(tracker.update([impossible], 0), [])
        self.assertEqual(tracker.status()["active_tracks"], 0)
        self.assertEqual(
            tracker.status()["implausible_people_rejected"], 1)

    def test_gt_mode_does_not_suppress_aligned_new_person(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=2, enable_ray_recovery=False)
        tracker.update([person(1.0, 0.0, 0.7)], 0)
        tracker.update([person(1.0, 0.0, 0.7)], 100_000_000)
        aligned = [person(1.0, 0.0, 0.7), person(3.0, 0.0, 2.1)]
        output = tracker.update(aligned, 200_000_000)
        # A later articulated person is provisional only for the configured
        # confirmation interval. It must not be discarded as a same-ray depth
        # fragment in exact Isaac GT mode.
        self.assertEqual(len(output), 1)
        self.assertEqual(tracker.status()["active_tracks"], 2)
        self.assertEqual(tracker.status()["new_measurements_suppressed"], 0)
        output = tracker.update(aligned, 300_000_000)
        self.assertEqual(len(output), 2)


class SkeletonStreamIdentityTest(unittest.TestCase):
    def test_packet_metadata_and_session_reset(self):
        tracked = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False).update(
                [person(0.0)], 0)
        packet_a = build_skeleton_packet(
            tracked, 0, 0, JOINTS, session_id="session-a")
        self.assertEqual(packet_a["session_id"], "session-a")
        self.assertEqual(packet_a["people"][0]["track_state"], "observed")
        window = StgcnWindow(window=2, max_people=2)
        self.assertIsNone(window.push(packet_a))
        packet_b = dict(packet_a)
        packet_b["sequence"] = 1
        self.assertIsNotNone(window.push(packet_b))
        packet_c = dict(packet_b)
        packet_c["session_id"] = "session-b"
        packet_c["sequence"] = 0
        self.assertIsNone(window.push(packet_c))
        self.assertEqual(len(window.frames), 1)


if __name__ == "__main__":
    unittest.main()
