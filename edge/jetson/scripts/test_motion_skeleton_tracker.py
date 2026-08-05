#!/usr/bin/env python3
import unittest

import numpy as np

from motion_skeleton_tracker import MotionSkeletonTracker


JOINTS = tuple("joint_{}".format(index) for index in range(17))


def person(x, y=0.0, z=1.0):
    joints = []
    for joint_id, name in enumerate(JOINTS):
        offset = np.array([0.0, (joint_id % 3 - 1) * .02,
                           (joint_id // 3) * .015])
        xyz = (np.array([x, y, z]) + offset).tolist()
        joints.append({
            "id": joint_id, "name": name, "xyz_imu_m": xyz,
            "score": .9, "source": "anchor_epipolar",
            "measurement_sigma_m": .03,
        })
    return {"source_pairs": ["AB_RIGHT"],
            "source_anchors": ["ANCHOR_CAM_A"], "joints": joints}


def centers_by_id(output):
    result = {}
    for value in output:
        points = [value["joints"][index]["xyz_imu_m"]
                  for index in (5, 6, 11, 12)]
        result[value["person_id"]] = np.median(
            np.asarray(points, dtype=np.float64), axis=0)
    return result


class MotionSkeletonTrackerTest(unittest.TestCase):
    def test_stable_people_receive_two_ids(self):
        tracker = MotionSkeletonTracker(JOINTS, confirmation_hits=1)
        output = tracker.update([person(-1.0), person(1.0)], 0)
        self.assertEqual([value["person_id"] for value in output], [0, 1])

    def test_two_people_keep_directional_ids_through_crossing(self):
        tracker = MotionSkeletonTracker(JOINTS, confirmation_hits=1)
        sequence = [
            (-1.0, 1.0), (-.72, .72), (-.42, .42), (-.14, .14),
            (.14, -.14), (.42, -.42), (.72, -.72),
        ]
        output = None
        for index, (first, second) in enumerate(sequence):
            # Sorting by image/space order at the crossing intentionally swaps
            # the measurement list order. IDs must follow motion, not ordering.
            measurements = sorted([person(first), person(second)],
                                  key=lambda value: value["joints"][5][
                                      "xyz_imu_m"][0])
            output = tracker.update(measurements, index * 100_000_000)
        centers = centers_by_id(output)
        self.assertGreater(centers[0][0], .45)
        self.assertLess(centers[1][0], -.45)

    def test_occluded_person_slows_without_reversing_and_recovers_id(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1, prediction_timeout=1.2)
        tracker.update([person(0.0)], 0)
        tracker.update([person(.20)], 100_000_000)
        observed = tracker.update([person(.40)], 200_000_000)
        last_x = centers_by_id(observed)[0][0]
        increments = []
        for index in range(3, 9):
            predicted = tracker.update([], index * 100_000_000)
            self.assertEqual(predicted[0]["track_state"], "predicted")
            x = centers_by_id(predicted)[0][0]
            increments.append(x - last_x)
            self.assertGreaterEqual(x + 1e-8, last_x)
            last_x = x
        self.assertLess(increments[-1], increments[0])
        recovered = tracker.update([person(last_x + .08)], 900_000_000)
        self.assertEqual(recovered[0]["person_id"], 0)
        self.assertEqual(recovered[0]["track_state"], "observed")

    def test_prediction_eventually_expires(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1,
            prediction_timeout=.5, deletion_timeout=.8)
        tracker.update([person(0.0)], 0)
        self.assertTrue(tracker.update([], 400_000_000))
        self.assertFalse(tracker.update([], 600_000_000))
        tracker.update([], 900_000_000)
        self.assertEqual(tracker.status(900_000_000)["active_tracks"], 0)

    def test_radial_depth_jump_keeps_one_identity(self):
        tracker = MotionSkeletonTracker(JOINTS, confirmation_hits=1)
        tracker.update([person(1.0, .15, .7)], 0)
        output = tracker.update([person(3.0, .45, 2.1)], 100_000_000)
        self.assertEqual([value["person_id"] for value in output], [0])
        self.assertEqual(tracker.status(100_000_000)["next_person_id"], 1)

    def test_intermittent_false_measurement_never_gets_public_id(self):
        tracker = MotionSkeletonTracker(JOINTS, confirmation_hits=3)
        for index in range(3):
            tracker.update([person(0.0)], index * 100_000_000)
        for index in range(3, 12):
            values = [person(.02 * index)]
            if index in (3, 5, 8):
                values.append(person(-2.5, 1.5))
            output = tracker.update(values, index * 100_000_000)
            self.assertEqual([value["person_id"] for value in output], [0])

    def test_real_later_person_requires_seven_consecutive_observations(self):
        tracker = MotionSkeletonTracker(JOINTS, confirmation_hits=3)
        for index in range(3):
            tracker.update([person(0.0)], index * 100_000_000)
        for index in range(3, 9):
            output = tracker.update(
                [person(0.0), person(-2.5, 1.5)], index * 100_000_000)
            self.assertEqual(len(output), 1)
        output = tracker.update(
            [person(0.0), person(-2.5, 1.5)], 900_000_000)
        self.assertEqual([value["person_id"] for value in output], [0, 1])

    def test_identity_survives_two_second_camera_boundary_gap(self):
        tracker = MotionSkeletonTracker(
            JOINTS, confirmation_hits=1,
            prediction_timeout=1.2, deletion_timeout=4.0)
        tracker.update([person(-1.0, 0.0, .7)], 0)
        self.assertFalse(tracker.update([], 1_300_000_000))
        output = tracker.update([person(.3, 1.1, .6)], 2_100_000_000)
        self.assertEqual([value["person_id"] for value in output], [0])
        self.assertEqual(tracker.status(2_100_000_000)["next_person_id"], 1)


if __name__ == "__main__":
    unittest.main()
