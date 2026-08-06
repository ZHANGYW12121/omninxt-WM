#!/usr/bin/env python3
import unittest

import numpy as np

from motion_skeleton_tracker import (
    MotionSkeletonTracker,
    _hungarian_assignment,
    _optimal_assignment,
)
from skeleton_stream import StgcnWindow, build_skeleton_packet


JOINTS = tuple("joint_{}".format(index) for index in range(17))


def person(x, y=0.0, z=1.0):
    joints = []
    for joint_id, name in enumerate(JOINTS):
        offset = np.array([0.0, (joint_id % 3 - 1) * 0.02,
                           (joint_id // 3) * 0.015])
        xyz = (np.array([x, y, z]) + offset).tolist()
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

    def test_gt_mode_does_not_suppress_aligned_new_person(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, enable_ray_recovery=False)
        tracker.update([person(1.0, 0.0, 0.7)], 0)
        aligned = [person(1.0, 0.0, 0.7), person(3.0, 0.0, 2.1)]
        output = tracker.update(aligned, 100_000_000)
        # A later person is provisional for seven consecutive observations,
        # but it must exist internally rather than being discarded as a
        # same-ray depth fragment in exact Isaac GT mode.
        self.assertEqual(len(output), 1)
        self.assertEqual(tracker.status()["active_tracks"], 2)
        self.assertEqual(tracker.status()["new_measurements_suppressed"], 0)
        for index in range(2, 8):
            output = tracker.update(aligned, index * 100_000_000)
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
