#!/usr/bin/env python3

import math
import unittest
from collections import Counter

from warehouse_crowd_v2.crowd_templates import (
    build_crowd_scene,
    sample_warehouse_goal,
)


POLYGON = [(8.7, -3.8), (-9.6, -3.8), (-9.6, 25.8), (8.7, 25.8)]
COMMON = dict(
    group_spacing="close",
    direction="random_heading",
    drone_distance="near",
    speed="fast",
    seed=1,
    walk_polygon=POLYGON,
    drone_x_range=(-3.0, 4.0),
    drone_y_range=(-10.0, -5.0),
    route_x_min=-7.0,
    template_version="warehouse_v2",
)


class WarehouseCrowdReproductionTest(unittest.TestCase):
    def test_seed_one_reference_and_sparse_shape(self):
        scene = build_crowd_scene(
            num_people=18, valid_people_counts=tuple(range(17, 24)),
            crowd_layout="sparse", dense_profile="transverse40", **COMMON
        )
        self.assertAlmostEqual(scene.drone_spawn[0], -2.0594502912)
        self.assertAlmostEqual(scene.drone_spawn[1], -5.7628313153)
        self.assertEqual(len(scene.person_specs), 18)
        groups = {person.group_id for person in scene.person_specs}
        self.assertEqual(len(groups), 8)
        leaders = {person.group_id: person for person in scene.person_specs}
        longitudinal = sum(
            abs(person.waypoints[-1][1] - person.waypoints[0][1])
            > abs(person.waypoints[-1][0] - person.waypoints[0][0])
            for person in leaders.values()
        )
        self.assertEqual(longitudinal, 2)

    def test_dense_transverse40_contract(self):
        scene = build_crowd_scene(
            num_people=40, valid_people_counts=tuple(range(17, 24)),
            crowd_layout="dense", dense_profile="transverse40", **COMMON
        )
        groups = {}
        for person in scene.person_specs:
            groups.setdefault(person.group_id, []).append(person)
            self.assertEqual(person.direction, "x_flow")
            self.assertEqual(len(person.waypoints), 8)
        self.assertEqual(len(scene.person_specs), 40)
        self.assertEqual(len(groups), 20)
        self.assertEqual(Counter(map(len, groups.values())), {1: 4, 2: 12, 3: 4})
        sides = Counter(
            "left" if members[0].route_direction > 0 else "right"
            for members in groups.values()
        )
        self.assertEqual(sides, {"left": 10, "right": 10})
        self.assertTrue(all(not person.traffic_gates for person in scene.person_specs))

        # The dense map is preplanned as parallel, non-intersecting formation
        # envelopes.  Even if neighbouring groups reach the same route phase,
        # their individual member routes retain full personal space.
        ordered_groups = list(groups.values())
        for group_index, members in enumerate(ordered_groups):
            for first_index, first in enumerate(members):
                for second in members[first_index + 1:]:
                    self.assertGreaterEqual(
                        math.dist(first.init_pos[:2], second.init_pos[:2]), 0.30
                    )
            for other_members in ordered_groups[group_index + 1:]:
                for first in members:
                    for second in other_members:
                        clearance = min(
                            math.dist(a[:2], b[:2])
                            for a, b in zip(first.waypoints, second.waypoints)
                        )
                        self.assertGreaterEqual(clearance, 1.0)

    def test_seed_is_reproducible_and_goal_is_independent(self):
        kwargs = dict(
            num_people=40, valid_people_counts=tuple(range(17, 24)),
            crowd_layout="dense", dense_profile="transverse40", **COMMON
        )
        first = build_crowd_scene(**kwargs)
        second = build_crowd_scene(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            sample_warehouse_goal(1),
            [-0.678401565341606, 28.686360039539785, 1.0],
        )

    def test_formal_routes_stay_outside_reserved_end_zones(self):
        for layout, count in (("sparse", 18), ("dense", 40)):
            scene = build_crowd_scene(
                num_people=count, valid_people_counts=tuple(range(17, 24)),
                crowd_layout=layout, dense_profile="transverse40", **COMMON
            )
            for person in scene.person_specs:
                for x, y, _ in person.waypoints:
                    self.assertGreaterEqual(x, -7.0 - 1e-6)
                    self.assertLessEqual(x, 8.7 + 1e-6)
                    self.assertGreaterEqual(y, -3.8 - 1e-6)
                    self.assertLessEqual(y, 25.8 + 1e-6)


if __name__ == "__main__":
    unittest.main()
