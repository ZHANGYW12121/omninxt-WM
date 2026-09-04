#!/usr/bin/env python3

import unittest

import numpy as np

from person_detection_policy import (
    articulated_person_is_valid,
    person_box_is_usable,
)


def standing_pose():
    points = np.zeros((17, 2), dtype=np.float64)
    points[5] = (80, 70)
    points[6] = (120, 70)
    points[7] = (70, 105)
    points[8] = (130, 105)
    points[9] = (60, 135)
    points[10] = (140, 135)
    points[11] = (87, 135)
    points[12] = (113, 135)
    points[13] = (87, 180)
    points[14] = (113, 180)
    points[15] = (87, 225)
    points[16] = (113, 225)
    scores = np.full(17, .90, dtype=np.float64)
    return points, scores


class PersonDetectionPolicyTest(unittest.TestCase):

    def test_accepts_narrow_side_view_box_but_rejects_tiny_texture(self):
        self.assertTrue(person_box_is_usable((20, 10, 20.1, 26)))
        self.assertFalse(person_box_is_usable((20, 10, 20, 26)))
        self.assertFalse(person_box_is_usable((20, 10, 50, 25)))

    def test_accepts_one_visible_side_of_articulated_body(self):
        points, scores = standing_pose()
        scores[5:] = .05
        visible_side = (5, 7, 9, 11, 13, 15)
        scores[list(visible_side)] = .85
        self.assertTrue(articulated_person_is_valid(
            points, scores, .25, (45, 45, 150, 235)))

    def test_accepts_low_confidence_side_body_but_not_two_points(self):
        points, scores = standing_pose()
        scores[5:] = .05
        visible_side = (5, 7, 9, 11, 13, 15)
        scores[list(visible_side)] = .21
        self.assertTrue(articulated_person_is_valid(
            points, scores, .20, (45, 45, 150, 235)))
        scores[7] = scores[9] = scores[13] = scores[15] = .05
        self.assertFalse(articulated_person_is_valid(
            points, scores, .20, (45, 45, 150, 235)))

    def test_accepts_articulated_body_inside_detector_box(self):
        points, scores = standing_pose()
        self.assertTrue(articulated_person_is_valid(
            points, scores, .25, (45, 45, 150, 235)))

    def test_rejects_face_only_detector_response(self):
        points, scores = standing_pose()
        scores[5:] = .05
        self.assertFalse(articulated_person_is_valid(
            points, scores, .25, (45, 45, 150, 235)))

    def test_rejects_spatially_collapsed_pose(self):
        points, scores = standing_pose()
        points[5:] = (100, 100)
        self.assertFalse(articulated_person_is_valid(
            points, scores, .25, (45, 45, 150, 235)))

    def test_rejects_inverted_torso(self):
        points, scores = standing_pose()
        points[[5, 6], 1] = 155
        points[[11, 12], 1] = 100
        self.assertFalse(articulated_person_is_valid(
            points, scores, .25, (45, 45, 150, 235)))


if __name__ == "__main__":
    unittest.main()
