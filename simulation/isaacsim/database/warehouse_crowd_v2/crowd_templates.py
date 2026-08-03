#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import math
import random


VALID_PEOPLE_COUNTS = tuple(range(5, 16))
VALID_GROUP_SPACINGS = ("close", "far")
VALID_DIRECTIONS = (
    "same_direction",
    "opposite_direction",
    "left_to_right",
    "right_to_left",
    "random",
)
VALID_DRONE_DISTANCES = ("near", "far")
VALID_SPEEDS = ("slow", "fast")
VALID_CROWD_LAYOUTS = ("sparse", "dense")
VALID_DENSE_PROFILES = ("transverse40", "legacy")
WAREHOUSE_V2_DIRECTIONS = ("x_flow", "y_flow")
WAREHOUSE_V2_DIRECTION_MODES = WAREHOUSE_V2_DIRECTIONS + (
    "mixed",
    "random_heading",
)
# Singles, pairs and triples coexist.  Members of a pair/triple own parallel
# offset routes and are treated as one social group by the controller.
WAREHOUSE_V2_MULTI_PERSON_GROUPS = True

GROUP_SPACING_RANGES = {
    "close": (0.8, 1.0),
    "far": (1.8, 2.0),
}
DRONE_DISTANCE_RANGES = {
    "near": (0.0, 0.0),
    "far": (0.0, 0.0),
}
ACTIVITY_POLYGON = [
    (7.5, -5.0),
    (-7.5, -5.0),
    (-7.5, 29.0),
    (7.5, 29.0),
]
DRONE_X_RANGE = (-7.5, 7.5)
DRONE_Y_RANGE = (-10.0, -5.0)
WALK_X_RANGE = (-7.5, 7.5)
HORIZONTAL_START_X_RANGE = (-7.5, 7.5)
HORIZONTAL_START_Y_RANGE = (-5.0, 29.0)
WALK_EDGE_MARGIN = 0.8
TURN_EDGE_MARGIN = 0.2
DIRECTION_START_BOUNDS = {
    "same_direction": (-7.5, 7.5, -5.0, 29.0),
    "opposite_direction": (-7.5, 7.5, -5.0, 29.0),
    "left_to_right": (
        HORIZONTAL_START_X_RANGE[0],
        HORIZONTAL_START_X_RANGE[1],
        HORIZONTAL_START_Y_RANGE[0],
        HORIZONTAL_START_Y_RANGE[1],
    ),
    "right_to_left": (
        HORIZONTAL_START_X_RANGE[0],
        HORIZONTAL_START_X_RANGE[1],
        HORIZONTAL_START_Y_RANGE[0],
        HORIZONTAL_START_Y_RANGE[1],
    ),
}
RANDOM_DIRECTION_CHOICES = (
    "same_direction",
    "opposite_direction",
    "left_to_right",
    "right_to_left",
)
HORIZONTAL_DIRECTIONS = ("left_to_right", "right_to_left")
LONGITUDINAL_DIRECTIONS = ("same_direction", "opposite_direction")
FLOW_GROUP_IDS = {
    "same_direction": -1,
    "opposite_direction": -2,
    "left_to_right": -3,
    "right_to_left": -3,
}
MIN_INITIAL_PERSON_DISTANCE = 0.75
WALK_SPEED_RANGES = {
    "slow": (0.65, 0.95),
    "fast": (1.15, 1.55),
}
GROUP_SPEED_JITTER = 0.06
DEFAULT_CHARACTER_NAMES = (
    "original_male_adult_construction_05",
    "original_female_adult_business_02",
)


@dataclass(frozen=True)
class CrowdPersonSpec:
    name: str
    character_name: str
    init_pos: list
    init_yaw: float
    waypoints: list
    speed: float
    group_id: int
    controller_kind: str
    direction: str
    spacing: float
    change_interval: float
    motion_polygon: list = None
    loop: bool = False
    motion_profile: str = "legacy"
    group_size: int = 1
    member_index: int = 0
    formation_lateral: float = 0.0
    formation_longitudinal: float = 0.0
    start_waypoint_index: int = 0
    route_direction: int = 1
    speed_seed: int = 0
    # Warehouse V2 assigns every social group one spatially separated lane.
    # Orthogonal lanes necessarily cross, so their crossings are protected by
    # one deterministic two-phase schedule shared by all controllers.
    traffic_axis: str = ""
    traffic_gates: tuple = ()
    traffic_cycle_sec: float = 18.0
    traffic_green_start_sec: float = 0.0
    traffic_green_duration_sec: float = 7.5
    traffic_clearance_sec: float = 1.5
    planned_walk_sec: float = 0.0
    planned_pause_sec: float = 0.0
    planned_pause_phase_sec: float = 0.0


@dataclass(frozen=True)
class CrowdSceneConfig:
    key: str
    num_people: int
    group_spacing: str
    direction: str
    drone_distance: str
    speed: str
    seed: int
    drone_spawn: list
    person_specs: list
    layout: str = "sparse"
    dense_profile: str = ""


def make_template_key(
    num_people,
    group_spacing,
    direction,
    drone_distance,
    speed,
):
    return (
        f"n{num_people}_spacing_{group_spacing}_dir_{direction}_"
        f"dist_{drone_distance}_speed_{speed}"
    )


def all_template_keys(valid_people_counts=None):
    people_counts = tuple(valid_people_counts or VALID_PEOPLE_COUNTS)
    keys = []
    for num_people in people_counts:
        for group_spacing in VALID_GROUP_SPACINGS:
            for direction in VALID_DIRECTIONS:
                for drone_distance in VALID_DRONE_DISTANCES:
                    for speed in VALID_SPEEDS:
                        keys.append(
                            make_template_key(
                                num_people,
                                group_spacing,
                                direction,
                                drone_distance,
                                speed,
                            )
                        )
    return keys


def build_crowd_scene_from_key(
    key,
    seed=1,
    walk_polygon=None,
    valid_people_counts=None,
    drone_x_range=None,
    drone_y_range=None,
    direction_start_bounds=None,
    route_x_min=None,
    crowd_layout="sparse",
    dense_profile="transverse40",
):
    parts = key.split("_")
    try:
        num_people = int(parts[0].lstrip("n"))
        group_spacing = parts[2]
        dir_start = parts.index("dir") + 1
        dist_index = parts.index("dist")
        direction = "_".join(parts[dir_start:dist_index])
        drone_distance = parts[dist_index + 1]
        speed = parts[dist_index + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid crowd template key: {key}") from exc

    return build_crowd_scene(
        num_people=num_people,
        group_spacing=group_spacing,
        direction=direction,
        drone_distance=drone_distance,
        speed=speed,
        seed=seed,
        walk_polygon=walk_polygon,
        valid_people_counts=valid_people_counts,
        drone_x_range=drone_x_range,
        drone_y_range=drone_y_range,
        direction_start_bounds=direction_start_bounds,
        route_x_min=route_x_min,
        crowd_layout=crowd_layout,
        dense_profile=dense_profile,
    )


def build_crowd_scene(
    num_people=10,
    group_spacing="close",
    direction="same_direction",
    drone_distance="near",
    speed="slow",
    seed=1,
    walk_polygon=None,
    character_names=None,
    valid_people_counts=None,
    drone_x_range=None,
    drone_y_range=None,
    direction_start_bounds=None,
    template_version="legacy",
    route_x_min=None,
    crowd_layout="sparse",
    dense_profile="transverse40",
):
    if template_version == "warehouse_v2":
        return _build_warehouse_v2_scene(
            num_people=num_people,
            group_spacing=group_spacing,
            direction=direction,
            drone_distance=drone_distance,
            speed=speed,
            seed=seed,
            walk_polygon=walk_polygon,
            character_names=character_names,
            valid_people_counts=valid_people_counts,
            drone_x_range=drone_x_range,
            drone_y_range=drone_y_range,
            route_x_min=route_x_min,
            crowd_layout=crowd_layout,
            dense_profile=dense_profile,
        )
    if template_version != "legacy":
        raise ValueError(f"Unknown crowd template_version: {template_version!r}")
    valid_people_counts = tuple(valid_people_counts or VALID_PEOPLE_COUNTS)
    drone_x_range = tuple(drone_x_range or DRONE_X_RANGE)
    drone_y_range = tuple(drone_y_range or DRONE_Y_RANGE)
    direction_start_bounds = _normalize_direction_start_bounds(direction_start_bounds)

    _validate_choice("num_people", num_people, valid_people_counts)
    _validate_choice("group_spacing", group_spacing, VALID_GROUP_SPACINGS)
    _validate_choice("direction", direction, VALID_DIRECTIONS)
    _validate_choice("drone_distance", drone_distance, VALID_DRONE_DISTANCES)
    _validate_choice("speed", speed, VALID_SPEEDS)

    rng = random.Random(seed)
    polygon = walk_polygon or ACTIVITY_POLYGON
    characters = tuple(character_names or DEFAULT_CHARACTER_NAMES)
    drone_spawn = _drone_spawn_for_distance(drone_distance, rng, drone_x_range, drone_y_range)
    group_sizes = _make_group_sizes(num_people, rng)
    group_directions = [
        _group_motion_direction(direction, rng)
        for _ in group_sizes
    ]
    longitudinal_y_slots = {
        longitudinal_direction: _uniform_longitudinal_y_slots(
            sum(
                group_size
                for group_size, group_direction in zip(group_sizes, group_directions)
                if group_direction == longitudinal_direction
            ),
            direction_start_bounds[longitudinal_direction],
            rng,
        )
        for longitudinal_direction in LONGITUDINAL_DIRECTIONS
    }
    longitudinal_slot_indices = {
        longitudinal_direction: 0
        for longitudinal_direction in LONGITUDINAL_DIRECTIONS
    }
    horizontal_person_count = sum(
        group_size
        for group_size, group_direction in zip(group_sizes, group_directions)
        if group_direction in HORIZONTAL_DIRECTIONS
    )
    horizontal_y_min = max(
        direction_start_bounds[direction][2]
        for direction in HORIZONTAL_DIRECTIONS
    )
    horizontal_y_max = min(
        direction_start_bounds[direction][3]
        for direction in HORIZONTAL_DIRECTIONS
    )
    horizontal_y_slots = _uniform_longitudinal_y_slots(
        horizontal_person_count,
        (0.0, 0.0, horizontal_y_min, horizontal_y_max),
        rng,
    )
    horizontal_slot_index = 0
    spacing_min, spacing_max = GROUP_SPACING_RANGES[group_spacing]
    speed_min, speed_max = WALK_SPEED_RANGES[speed]

    specs = []
    person_index = 1
    for group_id, (group_size, motion_direction) in enumerate(
        zip(group_sizes, group_directions)
    ):
        spacing_value = rng.uniform(spacing_min, spacing_max)
        group_speed = rng.uniform(speed_min, speed_max)
        motion_vec = _motion_vector(motion_direction, None, None, rng)
        offsets = _formation_offsets(group_size, spacing_value, motion_vec)
        group_start = _group_start_position(
            group_id,
            len(group_sizes),
            motion_direction,
            drone_distance,
            drone_spawn,
            offsets,
            rng,
            polygon,
            direction_start_bounds,
        )
        start_bounds = direction_start_bounds[motion_direction]
        group_start = _resolve_group_start_spacing(
            group_start,
            offsets,
            start_bounds,
            polygon,
            specs,
            rng,
        )

        for offset in offsets:
            person_direction = _person_motion_direction(motion_direction, rng)
            start_x = _clamp(group_start[0] + offset[0], start_bounds[0], start_bounds[1])
            if motion_direction in LONGITUDINAL_DIRECTIONS:
                slot_index = longitudinal_slot_indices[motion_direction]
                start_y = longitudinal_y_slots[motion_direction][slot_index]
                longitudinal_slot_indices[motion_direction] += 1
            elif motion_direction in HORIZONTAL_DIRECTIONS:
                start_y = horizontal_y_slots[horizontal_slot_index]
                horizontal_slot_index += 1
            else:
                start_y = _clamp(group_start[1] + offset[1], start_bounds[2], start_bounds[3])
            start_xy = _nearest_bounded_polygon_xy(start_x, start_y, start_bounds, polygon, rng)
            if motion_direction in LONGITUDINAL_DIRECTIONS + HORIZONTAL_DIRECTIONS:
                # Preserve the stratified y slot and move only along x if this
                # per-person placement is too close to an existing person.
                start_xy = _resolve_fixed_y_start_spacing(
                    start_xy,
                    start_bounds,
                    polygon,
                    specs,
                    rng,
                )

            waypoints, loop = _waypoints_for_direction(
                person_direction,
                start_xy,
                polygon,
                rng,
                direction_start_bounds,
            )
            yaw = _yaw_from_motion(_motion_vector(person_direction, None, None, rng))

            specs.append(
                CrowdPersonSpec(
                    name=f"person{person_index}",
                    character_name=characters[(person_index - 1) % len(characters)],
                    init_pos=[start_xy[0], start_xy[1], 0.0],
                    init_yaw=yaw,
                    waypoints=waypoints,
                    speed=_member_speed(group_speed, speed_min, speed_max, rng),
                    # Walkers with parallel, non-crossing lanes form one
                    # coordinated flow. Soft avoidance is skipped inside a
                    # flow, while emergency separation below the hard personal
                    # distance remains active in the controller.
                    group_id=FLOW_GROUP_IDS.get(person_direction, group_id),
                    controller_kind="waypoint",
                    direction=person_direction,
                    spacing=spacing_value,
                    change_interval=rng.uniform(6.0, 9.0),
                    motion_polygon=None,
                    loop=loop,
                )
            )
            person_index += 1

    return CrowdSceneConfig(
        key=make_template_key(
            num_people,
            group_spacing,
            direction,
            drone_distance,
            speed,
        ),
        num_people=num_people,
        group_spacing=group_spacing,
        direction=direction,
        drone_distance=drone_distance,
        speed=speed,
        seed=seed,
        drone_spawn=drone_spawn,
        person_specs=specs,
    )


def _build_warehouse_v2_scene(
    num_people=12,
    group_spacing="close",
    direction="mixed",
    drone_distance="near",
    speed="slow",
    seed=None,
    walk_polygon=None,
    character_names=None,
    valid_people_counts=None,
    drone_x_range=None,
    drone_y_range=None,
    route_x_min=None,
    crowd_layout="sparse",
    dense_profile="transverse40",
):
    """Build natural, formation-aware, ping-pong Warehouse pedestrian routes.

    Geometry collision checks remain in Pegasus, after the USD obstacle AABBs
    are available.  This function creates smooth center lines and preserves a
    distinct route for every member of a 1--3 person formation.
    """
    _validate_choice("crowd_layout", crowd_layout, VALID_CROWD_LAYOUTS)
    _validate_choice("dense_profile", dense_profile, VALID_DENSE_PROFILES)
    if crowd_layout == "dense" and dense_profile == "transverse40":
        if int(num_people) != 40:
            raise ValueError(
                "dense + transverse40 requires exactly 40 people, "
                f"got {num_people}"
            )
        return _build_warehouse_v2_dense_transverse_scene(
            num_people=num_people,
            group_spacing=group_spacing,
            direction=direction,
            drone_distance=drone_distance,
            speed=speed,
            seed=seed,
            walk_polygon=walk_polygon,
            character_names=character_names,
            drone_x_range=drone_x_range,
            drone_y_range=drone_y_range,
            route_x_min=route_x_min,
        )

    valid_people_counts = tuple(valid_people_counts or tuple(range(17, 24)))
    _validate_choice("num_people", num_people, valid_people_counts)
    _validate_choice("group_spacing", group_spacing, VALID_GROUP_SPACINGS)
    _validate_choice("direction", direction, WAREHOUSE_V2_DIRECTION_MODES)
    _validate_choice("drone_distance", drone_distance, VALID_DRONE_DISTANCES)
    _validate_choice("speed", speed, VALID_SPEEDS)

    rng = random.Random(seed)
    polygon = list(walk_polygon or ACTIVITY_POLYGON)
    if route_x_min is not None:
        # This is the Warehouse V2 crowd-planning boundary, not the global map
        # or obstacle-scan boundary.  Run the unchanged lane, spawn, crossing,
        # and stop-schedule planner inside the clipped pedestrian region.
        polygon = [
            (max(float(x), float(route_x_min)), float(y))
            for x, y in polygon
        ]
    characters = tuple(character_names or DEFAULT_CHARACTER_NAMES)
    drone_spawn = _drone_spawn_for_distance(drone_distance, rng, drone_x_range, drone_y_range)
    group_sizes = _make_natural_group_sizes(num_people, rng)
    motion_types = _warehouse_v2_motion_types(direction, group_sizes, rng)
    # Natural shoulder-to-shoulder spacing. Group members use a much smaller
    # hard collision radius than unrelated pedestrians, so this distance does
    # not trigger mutual avoidance or left/right oscillation.
    spacing_range = (0.82, 0.96) if group_spacing == "close" else (1.12, 1.32)
    speed_min, speed_max = (
        (0.70, 0.98) if speed == "slow" else (0.95, 1.35)
    )

    group_plans = []
    for group_id, (group_size, motion_type) in enumerate(zip(group_sizes, motion_types)):
        spacing = rng.uniform(*spacing_range)
        group_plans.append({
            "group_id": group_id,
            "group_size": group_size,
            "motion_type": motion_type,
            "spacing": spacing,
            "half_width": 0.5 * (group_size - 1) * spacing,
            "base_speed": rng.uniform(speed_min, speed_max),
            "speed_seed": rng.randrange(1, 2**31 - 1),
            "speed_change_interval": rng.uniform(4.0, 7.0),
            "local_offsets": _natural_formation_offsets(group_size, spacing, rng),
            # Some groups naturally stop to look around. The schedule is fixed
            # at scene generation time, shared by all members, and therefore is
            # not mistaken for an avoidance/deadlock event.
            "planned_walk_sec": rng.uniform(8.0, 15.0),
            "planned_pause_sec": (
                rng.uniform(1.5, 4.5) if rng.random() < 0.55 else 0.0
            ),
        })
        group_plans[-1]["planned_pause_phase_sec"] = rng.uniform(
            0.0, group_plans[-1]["planned_walk_sec"] * 0.35
        )

    # Allocate complete, non-overlapping parallel lanes before any person is
    # instantiated. This replaces the former random center-line search. The
    # wider Y dimension is used to separate x-flow formations; the X dimension
    # separates y-flow formations.
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    if x_min >= x_max:
        raise ValueError(
            "route_x_min must remain left of the Warehouse V2 walk area, "
            f"got {x_min:.2f} >= {x_max:.2f}"
        )
    if direction == "random_heading":
        _assign_random_heading_routes(group_plans, polygon, rng)
        _assign_random_heading_traffic_gates(group_plans)
    else:
        x_flow_plans = [
            plan for plan in group_plans if plan["motion_type"] == "x_flow"
        ]
        y_flow_plans = [
            plan for plan in group_plans if plan["motion_type"] == "y_flow"
        ]
        _assign_separated_lane_centers(
            x_flow_plans, y_min, y_max, boundary_margin=0.75, spread=True
        )
        # Use the original lane allocation unchanged inside the planning polygon.
        _assign_separated_lane_centers(
            y_flow_plans,
            x_min,
            x_max,
            boundary_margin=1.55,
            spread=True,
        )

        # Each orthogonal crossing gets a precomputed protected radius.
        for plan in x_flow_plans:
            plan["traffic_gates"] = tuple(
                (
                    float(other["lane_center"]),
                    float(other["half_width"] + 1.30),
                    int(other["group_id"]),
                    float(other["lane_center"]),
                    float(plan["lane_center"]),
                    float(plan["half_width"] + 1.30),
                )
                for other in y_flow_plans
            )
            plan["green_start"] = 0.0
        for plan in y_flow_plans:
            plan["traffic_gates"] = tuple(
                (
                    float(other["lane_center"]),
                    float(other["half_width"] + 1.30),
                    int(other["group_id"]),
                    float(plan["lane_center"]),
                    float(other["lane_center"]),
                    float(plan["half_width"] + 1.30),
                )
                for other in x_flow_plans
            )
            plan["green_start"] = 9.0

    specs = []
    person_index = 1
    for plan in group_plans:
        group_id = plan["group_id"]
        group_size = plan["group_size"]
        motion_type = plan["motion_type"]
        spacing = plan["spacing"]
        base_speed = plan["base_speed"]
        group_speed_seed = plan["speed_seed"]
        group_speed_change_interval = plan["speed_change_interval"]
        local_offsets = plan["local_offsets"]
        if "center_route" in plan:
            center_route = [list(point) for point in plan["center_route"]]
            forward = tuple(plan["forward"])
        else:
            center_route, forward = _warehouse_v2_planned_lane_route(
                motion_type,
                plan["lane_center"],
                polygon,
                group_size,
                spacing,
                longitudinal_lower=plan.get("route_lower"),
            )
        start_index = _scheduled_start_index(
            center_route,
            motion_type,
            plan["traffic_gates"],
            local_offsets,
            forward,
            specs,
            rng,
        )
        if start_index <= 0:
            route_direction = 1
        elif start_index >= len(center_route) - 1:
            route_direction = -1
        else:
            route_direction = rng.choice((-1, 1))

        for member_index, (longitudinal, lateral) in enumerate(local_offsets):
            member_route = [
                [
                    point[0] + forward[0] * longitudinal - forward[1] * lateral,
                    point[1] + forward[1] * longitudinal + forward[0] * lateral,
                    0.0,
                ]
                for point in center_route
            ]
            init_pos = list(member_route[start_index])
            initial_forward = forward if route_direction > 0 else (-forward[0], -forward[1])
            specs.append(CrowdPersonSpec(
                name=f"person{person_index}",
                character_name=characters[(person_index - 1) % len(characters)],
                init_pos=init_pos,
                init_yaw=_yaw_from_motion(initial_forward),
                waypoints=member_route,
                # A social group shares one walking cadence. Independent
                # jitter made its members continually overtake one another.
                speed=base_speed,
                group_id=group_id,
                controller_kind="natural_waypoint",
                direction=motion_type,
                spacing=spacing,
                change_interval=group_speed_change_interval,
                motion_polygon=polygon,
                loop=True,
                motion_profile="warehouse_v2",
                group_size=group_size,
                member_index=member_index,
                formation_lateral=lateral,
                formation_longitudinal=longitudinal,
                start_waypoint_index=start_index,
                route_direction=route_direction,
                speed_seed=group_speed_seed,
                traffic_axis=(
                    "random_heading"
                    if motion_type == "random_heading"
                    else motion_type
                ),
                traffic_gates=plan["traffic_gates"],
                traffic_cycle_sec=18.0,
                traffic_green_start_sec=plan["green_start"],
                traffic_green_duration_sec=7.5,
                traffic_clearance_sec=1.5,
                planned_walk_sec=plan["planned_walk_sec"],
                planned_pause_sec=plan["planned_pause_sec"],
                planned_pause_phase_sec=plan["planned_pause_phase_sec"],
            ))
            person_index += 1

    return CrowdSceneConfig(
        key=(f"warehouse_v2_n{num_people}_{direction}_seed_{seed}"),
        num_people=num_people,
        group_spacing=group_spacing,
        direction=direction,
        drone_distance=drone_distance,
        speed=speed,
        seed=seed,
        drone_spawn=drone_spawn,
        person_specs=specs,
        layout=crowd_layout,
        dense_profile=(dense_profile if crowd_layout == "dense" else ""),
    )


def _build_warehouse_v2_dense_transverse_scene(
    num_people,
    group_spacing,
    direction,
    drone_distance,
    speed,
    seed,
    walk_polygon,
    character_names,
    drone_x_range,
    drone_y_range,
    route_x_min,
):
    """Build the formal 40-person, all-transverse Warehouse layout.

    Twenty ordered center lines span the useful room length.  Their order is
    preserved at both ends, so the small seeded shear cannot create route
    intersections.  Ten formations start on either side and ping-pong forever.
    """
    _validate_choice("group_spacing", group_spacing, VALID_GROUP_SPACINGS)
    _validate_choice("drone_distance", drone_distance, VALID_DRONE_DISTANCES)
    _validate_choice("speed", speed, VALID_SPEEDS)
    rng = random.Random(seed)
    polygon = list(walk_polygon or ACTIVITY_POLYGON)
    if route_x_min is not None:
        polygon = [
            (max(float(x), float(route_x_min)), float(y))
            for x, y in polygon
        ]
    characters = tuple(character_names or DEFAULT_CHARACTER_NAMES)
    drone_spawn = _drone_spawn_for_distance(
        drone_distance, rng, drone_x_range, drone_y_range
    )
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    if x_min >= x_max:
        raise ValueError("dense transverse route_x_min leaves no walkable width")

    # Formal composition: 12 pairs, 4 singles, 4 triples = 40 people.
    group_sizes = [2] * 12 + [1] * 4 + [3] * 4
    rng.shuffle(group_sizes)
    sides = ["left"] * 10 + ["right"] * 10
    rng.shuffle(sides)

    route_margin_x = 0.94
    route_margin_y = 0.94
    # Build the formations before assigning lanes: equal centre-line spacing
    # is not sufficient because a triple is wider than a single.  Pack each
    # complete formation footprint with at least the normal 1.0 m personal
    # space between neighbouring groups.  This keeps the preplanned dense map
    # collision-free without relying on run-time side stepping.
    formation_offsets = [
        _dense_formation_offsets(group_size, rng)
        for group_size in group_sizes
    ]
    formation_half_widths = [
        max((abs(lateral) for _longitudinal, lateral in offsets), default=0.0)
        for offsets in formation_offsets
    ]
    lane_centers = _pack_dense_lane_centers(
        formation_half_widths,
        y_min + route_margin_y,
        y_max - route_margin_y,
        minimum_group_clearance=1.05,
    )
    # One common small shear keeps all center lines parallel/non-intersecting.
    shear = rng.uniform(-0.10, 0.10)
    route_dx = max(1e-6, (x_max - route_margin_x) - (x_min + route_margin_x))
    route_dy = shear * route_dx
    # Keep both route ends inside the same 0.94 m boundary margin.
    max_abs_dy = max(0.0, route_margin_y - 0.10)
    route_dy = _clamp(route_dy, -max_abs_dy, max_abs_dy)

    plans = []
    side_rank = {"left": 0, "right": 0}
    speed_min, speed_max = (
        (0.70, 0.98) if speed == "slow" else (0.95, 1.35)
    )
    for group_id, (group_size, side, lane_y, local_offsets) in enumerate(
        zip(group_sizes, sides, lane_centers, formation_offsets)
    ):
        rank = side_rank[side]
        side_rank[side] += 1
        route = [
            [
                x_min + route_margin_x
                + ((x_max - route_margin_x) - (x_min + route_margin_x))
                * index / 7.0,
                lane_y + route_dy * (index / 7.0 - 0.5),
            ]
            for index in range(8)
        ]
        # Three deterministic start phases prevent ten formations on the same
        # side from being born in one vertical stack, while retaining a clear
        # left/right initial population.
        phase = rank % 3
        start_index = phase if side == "left" else 7 - phase
        route_direction = 1 if side == "left" else -1
        plans.append({
            "group_id": group_id,
            "group_size": group_size,
            "route": route,
            "start_index": start_index,
            "route_direction": route_direction,
            "local_offsets": local_offsets,
            "base_speed": rng.uniform(speed_min, speed_max),
            "speed_seed": rng.randrange(1, 2**31 - 1),
            "speed_change_interval": rng.uniform(3.0, 8.0),
            "planned_walk_sec": rng.uniform(8.0, 15.0),
            "planned_pause_sec": (
                rng.uniform(1.5, 4.5) if rng.random() < 0.55 else 0.0
            ),
        })
        plans[-1]["planned_pause_phase_sec"] = rng.uniform(
            0.0, plans[-1]["planned_walk_sec"] * 0.35
        )

    specs = []
    person_index = 1
    forward = _normalize((route_dx, route_dy))
    for plan in plans:
        for member_index, (longitudinal, lateral) in enumerate(
            plan["local_offsets"]
        ):
            member_route = [
                [
                    point[0] + forward[0] * longitudinal
                    - forward[1] * lateral,
                    point[1] + forward[1] * longitudinal
                    + forward[0] * lateral,
                    0.0,
                ]
                for point in plan["route"]
            ]
            route_direction = plan["route_direction"]
            initial_forward = (
                forward if route_direction > 0 else (-forward[0], -forward[1])
            )
            specs.append(CrowdPersonSpec(
                name=f"person{person_index}",
                character_name=characters[(person_index - 1) % len(characters)],
                init_pos=list(member_route[plan["start_index"]]),
                init_yaw=_yaw_from_motion(initial_forward),
                waypoints=member_route,
                speed=plan["base_speed"],
                group_id=plan["group_id"],
                controller_kind="natural_waypoint",
                direction="x_flow",
                spacing=0.60,
                change_interval=plan["speed_change_interval"],
                motion_polygon=polygon,
                loop=True,
                motion_profile="warehouse_v2",
                group_size=plan["group_size"],
                member_index=member_index,
                formation_lateral=lateral,
                formation_longitudinal=longitudinal,
                start_waypoint_index=plan["start_index"],
                route_direction=route_direction,
                speed_seed=plan["speed_seed"],
                # Dense groups follow their own preplanned member slots.  A
                # dedicated axis tag prevents leader animation error from
                # being propagated to all followers at runtime.
                traffic_axis="dense_x_flow",
                traffic_gates=(),
                planned_walk_sec=plan["planned_walk_sec"],
                planned_pause_sec=plan["planned_pause_sec"],
                planned_pause_phase_sec=plan["planned_pause_phase_sec"],
            ))
            person_index += 1

    return CrowdSceneConfig(
        key=f"warehouse_v2_dense_transverse40_seed_{seed}",
        num_people=int(num_people),
        group_spacing=group_spacing,
        direction="x_flow",
        drone_distance=drone_distance,
        speed=speed,
        seed=seed,
        drone_spawn=drone_spawn,
        person_specs=specs,
        layout="dense",
        dense_profile="transverse40",
    )


def _dense_formation_offsets(group_size, rng):
    """Compact side-by-side/soft-V formations for the 40-person map."""
    jitter = lambda: rng.uniform(-0.005, 0.005)
    if group_size == 1:
        return [(0.0, 0.0)]
    if group_size == 2:
        return [
            (-0.40 + jitter(), -0.10 + jitter()),
            (0.40 + jitter(), 0.10 + jitter()),
        ]
    if group_size == 3:
        return [
            (0.40 + jitter(), -0.25 + jitter()),
            (-0.30 + jitter(), jitter()),
            (0.40 + jitter(), 0.25 + jitter()),
        ]
    raise ValueError(f"dense transverse formation size must be 1..3, got {group_size}")


def _pack_dense_lane_centers(
    formation_half_widths,
    minimum_center,
    maximum_center,
    minimum_group_clearance=1.05,
):
    """Pack ordered formation envelopes while preserving social clearance."""
    widths = [max(0.0, float(value)) for value in formation_half_widths]
    if not widths:
        return []
    available = float(maximum_center) - float(minimum_center)
    required = (
        2.0 * sum(widths)
        + max(0, len(widths) - 1) * float(minimum_group_clearance)
    )
    if required > available + 1e-9:
        raise ValueError(
            "dense transverse formations do not fit the walkable Y range: "
            f"required={required:.2f}m, available={available:.2f}m"
        )
    slack_per_gap = (
        (available - required) / max(1, len(widths) - 1)
    )
    centers = [float(minimum_center) + widths[0]]
    for previous, current in zip(widths, widths[1:]):
        centers.append(
            centers[-1]
            + previous
            + float(minimum_group_clearance)
            + slack_per_gap
            + current
        )
    return centers


def sample_warehouse_goal(seed, x_range=(-3.0, 4.0), y_range=(27.0, 29.0)):
    """Use an independent stream so crowd refactors never move the goal."""
    rng = random.Random(int(seed) ^ 0x57484F555345)
    return [
        rng.uniform(float(x_range[0]), float(x_range[1])),
        rng.uniform(float(y_range[0]), float(y_range[1])),
        1.0,
    ]


def _make_natural_group_sizes(num_people, rng):
    """Create 8--9 reproducible groups with singles, pairs, and triples.

    A 22-person scene previously became as many as 14 independent traffic
    agents. Their protected crossing intervals necessarily overlapped in the
    warehouse width, making any pairwise scheduler prone to corridor knots.
    Keep two true singles while packing the rest into social pairs/triples.
    """
    if not WAREHOUSE_V2_MULTI_PERSON_GROUPS:
        return [1] * int(num_people)
    num_people = int(num_people)
    group_count = 8 if num_people <= 19 else 9
    group_count = max(math.ceil(num_people / 3.0), min(group_count, num_people))
    result = [1] * group_count
    remaining = num_people - group_count
    # Indices 0 and 1 remain singles. Randomize how the remaining population
    # becomes pairs/triples, then shuffle group order for seed variability.
    expandable = list(range(2, group_count))
    while remaining > 0:
        candidates = [index for index in expandable if result[index] < 3]
        if not candidates:
            raise ValueError("Unable to partition Warehouse crowd into groups")
        index = rng.choice(candidates)
        result[index] += 1
        remaining -= 1
    rng.shuffle(result)
    return result


def _warehouse_v2_motion_types(direction, group_sizes, rng):
    count = len(group_sizes)
    if direction == "random_heading":
        return ["random_heading"] * count
    if direction != "mixed":
        return [direction] * count
    # Put two of the largest formations into the long left-side Y corridor.
    # The remaining groups get spatially disjoint X lanes to its right.
    y_count = min(2, count)
    tie_breakers = [rng.random() for _ in group_sizes]
    y_indices = set(
        sorted(
            range(count),
            key=lambda index: (-group_sizes[index], tie_breakers[index]),
        )[:y_count]
    )
    result = [
        "y_flow" if index in y_indices else "x_flow"
        for index in range(count)
    ]
    return result


def _assign_random_heading_routes(plans, polygon, rng):
    """Assign deterministic sloped routes with no same-family intersections."""
    if not plans:
        return
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    # Keep only a small longitudinal family, as in the previous X/Y planner.
    # Within each family, independently pack both endpoint sides while
    # preserving group order. Straight segments with the same ordering at both
    # sides cannot intersect, but their slopes remain seed-dependent.
    longitudinal_count = 2 if len(plans) >= 6 else 1
    tie_breakers = {int(plan["group_id"]): rng.random() for plan in plans}
    longitudinal_ids = {
        int(plan["group_id"])
        for plan in sorted(
            plans,
            key=lambda plan: (
                -int(plan["group_size"]),
                tie_breakers[int(plan["group_id"])],
            ),
        )[:longitudinal_count]
    }
    longitudinal = [
        plan for plan in plans if int(plan["group_id"]) in longitudinal_ids
    ]
    transverse = [
        plan for plan in plans if int(plan["group_id"]) not in longitudinal_ids
    ]
    rng.shuffle(longitudinal)
    rng.shuffle(transverse)

    _assign_random_packed_centers(
        transverse, y_min, y_max, "route_start_lateral", rng
    )
    _assign_random_packed_centers(
        transverse, y_min, y_max, "route_end_lateral", rng
    )
    _assign_random_packed_centers(
        longitudinal, x_min, x_max, "route_start_lateral", rng
    )
    _assign_random_packed_centers(
        longitudinal, x_min, x_max, "route_end_lateral", rng
    )
    # Keep random non-axis headings, but limit the shear between the two
    # independently packed sides. Convex blending preserves every packed
    # inter-lane gap while keeping cross-family angles close enough to
    # orthogonal that protected gate intervals do not expand into neighbours.
    for plan in plans:
        start_lateral = float(plan["route_start_lateral"])
        end_lateral = float(plan["route_end_lateral"])
        midpoint = 0.5 * (start_lateral + end_lateral)
        half_delta = 0.30 * (end_lateral - start_lateral)
        plan["route_start_lateral"] = midpoint - half_delta
        plan["route_end_lateral"] = midpoint + half_delta

    transverse_half_width = max(
        (float(plan["half_width"]) for plan in transverse), default=0.0
    )
    longitudinal_half_width = max(
        (float(plan["half_width"]) for plan in longitudinal), default=0.0
    )
    transverse_x_margin = 1.10 + transverse_half_width
    longitudinal_y_margin = 1.10 + longitudinal_half_width

    for plan in plans:
        if plan in transverse:
            start = (
                x_min + transverse_x_margin,
                float(plan["route_start_lateral"]),
            )
            end = (
                x_max - transverse_x_margin,
                float(plan["route_end_lateral"]),
            )
            plan["route_family"] = "transverse"
        else:
            start = (
                float(plan["route_start_lateral"]),
                y_min + longitudinal_y_margin,
            )
            end = (
                float(plan["route_end_lateral"]),
                y_max - longitudinal_y_margin,
            )
            plan["route_family"] = "longitudinal"
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length < 1e-6:
            raise ValueError("Warehouse V2 random route is too short")
        forward = (
            (end[0] - start[0]) / length,
            (end[1] - start[1]) / length,
        )
        count = 8
        plan["center_route"] = [
            [
                start[0] + (end[0] - start[0]) * index / float(count - 1),
                start[1] + (end[1] - start[1]) * index / float(count - 1),
            ]
            for index in range(count)
        ]
        plan["forward"] = forward
        plan["heading_degrees"] = math.degrees(
            math.atan2(forward[1], forward[0])
        )
        plan["green_start"] = 9.0 if int(plan["group_id"]) % 2 else 0.0


def _assign_random_packed_centers(plans, lower, upper, key, rng):
    """Pack one ordered endpoint side with randomized reproducible free space."""
    if not plans:
        return
    boundary_margin = 1.10
    # The gap is measured between the outer member center-lines of adjacent
    # formations. 2.90 m also keeps their cross-traffic protected intervals
    # disjoint for the near-orthogonal Warehouse route families.
    minimum_gap = 2.90
    usable_lower = float(lower) + boundary_margin
    usable_upper = float(upper) - boundary_margin
    widths = [2.0 * float(plan["half_width"]) for plan in plans]
    required = sum(widths) + minimum_gap * max(0, len(plans) - 1)
    available = usable_upper - usable_lower
    if required > available + 1e-9:
        raise ValueError(
            "Warehouse V2 planning area cannot pack random route family: "
            f"required={required:.2f}, available={available:.2f}"
        )
    slack = max(0.0, available - required)
    weights = [rng.expovariate(1.0) for _ in range(len(plans) + 1)]
    weight_sum = sum(weights)
    extras = [slack * value / weight_sum for value in weights]
    cursor = usable_lower + extras[0]
    for index, (plan, width) in enumerate(zip(plans, widths)):
        plan[key] = cursor + 0.5 * width
        cursor += width
        if index + 1 < len(plans):
            cursor += minimum_gap + extras[index + 1]


def _assign_random_heading_traffic_gates(plans):
    """Precompute every arbitrary-angle route crossing and protected interval."""
    gates_by_group = {int(plan["group_id"]): [] for plan in plans}
    for first_index, first in enumerate(plans):
        first_route = first["center_route"]
        first_start, first_end = first_route[0], first_route[-1]
        first_forward = first["forward"]
        for second in plans[first_index + 1:]:
            second_route = second["center_route"]
            second_start, second_end = second_route[0], second_route[-1]
            crossing = _segment_intersection_2d(
                first_start,
                first_end,
                second_start,
                second_end,
            )
            if crossing is None:
                continue

            crossing_sine = abs(
                first_forward[0] * second["forward"][1]
                - first_forward[1] * second["forward"][0]
            )
            # Shallow crossings require a longer protected interval because the
            # other formation occupies more distance along this route.
            denominator = max(0.30, crossing_sine)
            first_radius = min(
                4.5,
                (float(second["half_width"]) + 1.30) / denominator,
            )
            second_radius = min(
                4.5,
                (float(first["half_width"]) + 1.30) / denominator,
            )
            cross_x, cross_y = crossing
            first_coordinate = (
                cross_x * first_forward[0] + cross_y * first_forward[1]
            )
            second_coordinate = (
                cross_x * second["forward"][0]
                + cross_y * second["forward"][1]
            )
            first_group = int(first["group_id"])
            second_group = int(second["group_id"])
            gates_by_group[first_group].append(
                (
                    first_coordinate,
                    first_radius,
                    second_group,
                    cross_x,
                    cross_y,
                    second_radius,
                )
            )
            gates_by_group[second_group].append(
                (
                    second_coordinate,
                    second_radius,
                    first_group,
                    cross_x,
                    cross_y,
                    first_radius,
                )
            )

    for plan in plans:
        gates = gates_by_group[int(plan["group_id"])]
        plan["traffic_gates"] = tuple(
            sorted(gates, key=lambda gate: (float(gate[0]), int(gate[2])))
        )


def _line_rectangle_chord(center, direction, bounds):
    """Return the ordered chord of an axis-aligned rectangle through a line."""
    x_min, x_max, y_min, y_max = bounds
    t_min = -float("inf")
    t_max = float("inf")
    for coordinate, component, lower, upper in (
        (center[0], direction[0], x_min, x_max),
        (center[1], direction[1], y_min, y_max),
    ):
        if abs(component) < 1e-9:
            if coordinate < lower or coordinate > upper:
                return None
            continue
        first = (lower - coordinate) / component
        second = (upper - coordinate) / component
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
    if t_max - t_min < 1e-6:
        return None
    return (
        (
            center[0] + direction[0] * t_min,
            center[1] + direction[1] * t_min,
        ),
        (
            center[0] + direction[0] * t_max,
            center[1] + direction[1] * t_max,
        ),
    )


def _segment_intersection_2d(first_start, first_end, second_start, second_end):
    """Return an interior segment crossing, excluding endpoint-only contacts."""
    rx = float(first_end[0]) - float(first_start[0])
    ry = float(first_end[1]) - float(first_start[1])
    sx = float(second_end[0]) - float(second_start[0])
    sy = float(second_end[1]) - float(second_start[1])
    denominator = rx * sy - ry * sx
    if abs(denominator) < 1e-8:
        return None
    qx = float(second_start[0]) - float(first_start[0])
    qy = float(second_start[1]) - float(first_start[1])
    first_t = (qx * sy - qy * sx) / denominator
    second_t = (qx * ry - qy * rx) / denominator
    # Endpoint-near crossings are still real conflicts for formations. Exclude
    # only numerical endpoint contact, not the former outer three percent.
    if not (1e-5 < first_t < 1.0 - 1e-5 and 1e-5 < second_t < 1.0 - 1e-5):
        return None
    return (
        float(first_start[0]) + first_t * rx,
        float(first_start[1]) + first_t * ry,
    )


def _warehouse_v2_center_route(motion_type, polygon, group_size, spacing, rng):
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    formation_half_width = (group_size - 1) * spacing * 0.5
    margin = 0.65 + formation_half_width
    count = rng.randint(6, 8)

    if motion_type == "x_flow":
        xs = _linspace(x_min + margin, x_max - margin, count)
        base_y = rng.uniform(y_min + margin, y_max - margin)
        noise = _smooth_route_noise(count, min(0.60, (y_max - y_min) * 0.04), rng)
        return [[x, _clamp(base_y + delta, y_min + margin, y_max - margin)] for x, delta in zip(xs, noise)], (1.0, 0.0)

    if motion_type == "y_flow":
        ys = _linspace(y_min + margin, y_max - margin, count)
        base_x = rng.uniform(x_min + margin, x_max - margin)
        noise = _smooth_route_noise(count, min(0.60, (x_max - x_min) * 0.05), rng)
        return [[_clamp(base_x + delta, x_min + margin, x_max - margin), y] for y, delta in zip(ys, noise)], (0.0, 1.0)

    raise ValueError(f"Unsupported Warehouse V2 motion type: {motion_type!r}")


def _assign_separated_lane_centers(
    plans,
    lower,
    upper,
    boundary_margin=0.75,
    spread=True,
):
    """Pack formation-width lanes across one map dimension without overlap."""
    if not plans:
        return
    usable_lower = float(lower) + float(boundary_margin)
    usable_upper = float(upper) - float(boundary_margin)
    available = max(0.1, usable_upper - usable_lower)
    widths = [2.0 * float(plan["half_width"]) for plan in plans]
    occupied_width = sum(widths)
    if len(plans) == 1:
        if spread:
            plans[0]["lane_center"] = 0.5 * (usable_lower + usable_upper)
        else:
            plans[0]["lane_center"] = usable_lower + 0.5 * widths[0]
        return

    # At least 1.15 m remains between the outer members of adjacent groups.
    minimum_gap = 1.15
    gap = (
        max(minimum_gap, (available - occupied_width) / float(len(plans) - 1))
        if spread
        else minimum_gap
    )
    required = occupied_width + gap * (len(plans) - 1)
    if required > available:
        # This should not occur for 17--23 people in the warehouse dimensions,
        # but retain deterministic packing rather than reverting to randomness.
        gap = max(0.70, (available - occupied_width) / float(len(plans) - 1))

    cursor = usable_lower
    for plan, width in zip(plans, widths):
        plan["lane_center"] = cursor + 0.5 * width
        cursor += width + gap


def _warehouse_v2_planned_lane_route(
    motion_type,
    lane_center,
    polygon,
    group_size,
    spacing,
    longitudinal_lower=None,
):
    """Return one long, straight lane whose full formation stays in bounds."""
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    formation_half_width = 0.5 * (group_size - 1) * spacing
    lateral_margin = 0.65 + formation_half_width
    count = 8
    if motion_type == "x_flow":
        y = _clamp(float(lane_center), y_min + lateral_margin, y_max - lateral_margin)
        route_lower = (
            x_min + 1.55
            if longitudinal_lower is None
            else max(x_min + 1.55, float(longitudinal_lower))
        )
        xs = _linspace(route_lower, x_max - 0.80, count)
        return [[x, y] for x in xs], (1.0, 0.0)
    if motion_type == "y_flow":
        x = _clamp(float(lane_center), x_min + lateral_margin, x_max - lateral_margin)
        ys = _linspace(y_min + 0.80, y_max - 0.80, count)
        return [[x, y] for y in ys], (0.0, 1.0)
    raise ValueError(f"Unsupported Warehouse V2 motion type: {motion_type!r}")


def _scheduled_start_index(
    route,
    motion_type,
    gates,
    local_offsets,
    forward,
    existing_specs,
    rng,
):
    """Choose a phase-friendly, collision-free point on the long lane."""
    candidates = (
        list(range(len(route)))
        if motion_type == "random_heading"
        else list(range(1, max(2, len(route) - 1)))
    )
    scored = []
    for index in candidates:
        coordinate = (
            float(route[index][0]) * float(forward[0])
            + float(route[index][1]) * float(forward[1])
        )
        start = route[index]
        member_starts = [
            (
                start[0] + forward[0] * longitudinal - forward[1] * lateral,
                start[1] + forward[1] * longitudinal + forward[0] * lateral,
            )
            for longitudinal, lateral in local_offsets
        ]
        if existing_specs:
            person_clearance = min(
                math.hypot(
                    member[0] - float(spec.init_pos[0]),
                    member[1] - float(spec.init_pos[1]),
                )
                for member in member_starts
                for spec in existing_specs
            )
        else:
            person_clearance = float("inf")
        gate_clearance = min(
            (
                abs(coordinate - gate[0]) - gate[1]
                for gate in gates
            ),
            default=4.0,
        )
        safe = person_clearance >= 1.15 and gate_clearance >= 0.35
        score = (
            (100.0 if safe else 0.0)
            + min(person_clearance, 6.0)
            + 0.25 * gate_clearance
            + rng.uniform(0.0, 0.05)
        )
        scored.append((score, index))
    return max(scored)[1]


def _natural_formation_offsets(group_size, spacing, rng):
    laterals = [(index - 0.5 * (group_size - 1)) * spacing for index in range(group_size)]
    longitudinal = [rng.uniform(-0.18, 0.18) for _ in range(group_size)]
    longitudinal_mean = sum(longitudinal) / float(group_size)
    return [
        (value - longitudinal_mean, lateral)
        for value, lateral in zip(longitudinal, laterals)
    ]


def _smooth_route_noise(count, amplitude, rng):
    values = []
    state = rng.uniform(-amplitude * 0.35, amplitude * 0.35)
    for _ in range(count):
        state = _clamp(0.65 * state + rng.uniform(-0.35, 0.35) * amplitude, -amplitude, amplitude)
        values.append(state)
    return values


def _linspace(lower, upper, count):
    if count <= 1:
        return [0.5 * (lower + upper)]
    step = (upper - lower) / float(count - 1)
    return [lower + index * step for index in range(count)]


def _validate_choice(name, value, valid_values):
    if value not in valid_values:
        raise ValueError(f"{name} must be one of {valid_values}, got {value!r}")


def _normalize_direction_start_bounds(direction_start_bounds):
    if direction_start_bounds is None:
        direction_start_bounds = DIRECTION_START_BOUNDS

    bounds = {key: tuple(value) for key, value in direction_start_bounds.items()}
    for direction in RANDOM_DIRECTION_CHOICES:
        if direction not in bounds:
            raise ValueError(f"Missing start bounds for direction {direction!r}")
    return bounds


def _group_motion_direction(direction, rng):
    if direction == "random":
        return rng.choice(RANDOM_DIRECTION_CHOICES)
    return direction


def _person_motion_direction(group_direction, rng):
    if group_direction in HORIZONTAL_DIRECTIONS:
        return rng.choice(HORIZONTAL_DIRECTIONS)
    return group_direction


def _drone_spawn_for_distance(drone_distance, rng, drone_x_range=None, drone_y_range=None):
    x_range = tuple(drone_x_range or DRONE_X_RANGE)
    y_range = tuple(drone_y_range or DRONE_Y_RANGE)
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    return [x, y, 1.0]


def _make_group_sizes(num_people, rng):
    group_sizes = []
    remaining = num_people
    while remaining > 0:
        size = rng.randint(1, min(3, remaining))
        group_sizes.append(size)
        remaining -= size
    return group_sizes


def _member_speed(group_speed, speed_min, speed_max, rng):
    return _clamp(
        group_speed + rng.uniform(-GROUP_SPEED_JITTER, GROUP_SPEED_JITTER),
        speed_min,
        speed_max,
    )


def _uniform_longitudinal_y_slots(person_count, bounds, rng):
    """Return shuffled, evenly spaced y slots for one longitudinal flow."""
    if person_count <= 0:
        return []

    y_min, y_max = float(bounds[2]), float(bounds[3])
    edge_margin = min(0.5, max(0.0, (y_max - y_min) * 0.04))
    usable_min = y_min + edge_margin
    usable_max = y_max - edge_margin
    if person_count == 1:
        return [rng.uniform(usable_min, usable_max)]

    spacing = (usable_max - usable_min) / (person_count - 1)
    jitter = min(0.12, spacing * 0.10)
    slots = []
    for index in range(person_count):
        base = usable_min + spacing * index
        if index == 0:
            delta = rng.uniform(0.0, jitter)
        elif index == person_count - 1:
            delta = rng.uniform(-jitter, 0.0)
        else:
            delta = rng.uniform(-jitter, jitter)
        slots.append(base + delta)

    rng.shuffle(slots)
    return slots


def _group_start_position(
    group_id,
    group_count,
    direction,
    drone_distance,
    drone_spawn,
    offsets,
    rng,
    polygon,
    direction_start_bounds=None,
):
    direction_start_bounds = _normalize_direction_start_bounds(direction_start_bounds)
    bounds = direction_start_bounds[direction]
    x_min, x_max, y_min, y_max = bounds
    safe_x_min, safe_x_max = _safe_center_limits(x_min, x_max, [offset[0] for offset in offsets])
    safe_y_min, safe_y_max = _safe_center_limits(y_min, y_max, [offset[1] for offset in offsets])
    if direction in HORIZONTAL_DIRECTIONS:
        x = _spread_value(group_id, group_count, safe_x_min, safe_x_max, rng)
    else:
        x = rng.uniform(safe_x_min, safe_x_max)
    y = _spread_value(group_id, group_count, safe_y_min, safe_y_max, rng)

    return _nearest_bounded_polygon_xy(x, y, bounds, polygon, rng)


def _safe_center_limits(lower, upper, offsets):
    if not offsets:
        return lower, upper

    safe_lower = lower - min(offsets)
    safe_upper = upper - max(offsets)
    if safe_lower > safe_upper:
        return lower, upper
    return safe_lower, safe_upper


def _spread_value(group_id, group_count, lower, upper, rng):
    margin = min(1.0, max(0.0, (upper - lower) * 0.12))
    usable_min = lower + margin
    usable_max = upper - margin
    if group_count <= 1:
        return rng.uniform(usable_min, usable_max)

    spacing = (usable_max - usable_min) / (group_count - 1)
    base = usable_min + spacing * group_id
    jitter = min(1.2, spacing * 0.25)
    return _clamp(base + rng.uniform(-jitter, jitter), lower, upper)


def _group_end_position(direction, group_start, drone_spawn, rng, polygon):
    if direction == "random":
        direction = rng.choice(RANDOM_DIRECTION_CHOICES)

    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    x, y = group_start

    if direction == "same_direction":
        return _nearest_polygon_xy(x + rng.uniform(-1.0, 1.0), y_max - 1.0, polygon, rng)
    if direction == "opposite_direction":
        return _nearest_polygon_xy(x + rng.uniform(-1.0, 1.0), y_min + 1.0, polygon, rng)
    if direction == "left_to_right":
        return _nearest_polygon_xy(x_max - 1.0, y + rng.uniform(-1.2, 1.2), polygon, rng)
    if direction == "right_to_left":
        return _nearest_polygon_xy(x_min + 1.0, y + rng.uniform(-1.2, 1.2), polygon, rng)

    return _random_polygon_xy(polygon, rng)


def _waypoints_for_direction(direction, start_xy, polygon, rng, direction_start_bounds=None):
    direction_start_bounds = _normalize_direction_start_bounds(direction_start_bounds)
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    x, y = start_xy

    if direction == "same_direction":
        target = _nearest_polygon_xy(
            x,
            y_max - WALK_EDGE_MARGIN,
            polygon,
            rng,
        )
        return [[target[0], target[1], 0.0]], False

    if direction == "opposite_direction":
        target = _nearest_polygon_xy(
            x,
            y_min + WALK_EDGE_MARGIN,
            polygon,
            rng,
        )
        return [[target[0], target[1], 0.0]], False

    if direction in ("left_to_right", "right_to_left"):
        bounds = direction_start_bounds[direction]
        lane_y = _clamp(y, bounds[2], bounds[3])
        left = _nearest_polygon_xy(
            bounds[0] + TURN_EDGE_MARGIN,
            lane_y,
            polygon,
            rng,
        )
        right = _nearest_polygon_xy(
            bounds[1] - TURN_EDGE_MARGIN,
            lane_y,
            polygon,
            rng,
        )
        if direction == "left_to_right":
            return [[right[0], right[1], 0.0], [left[0], left[1], 0.0]], True
        return [[left[0], left[1], 0.0], [right[0], right[1], 0.0]], True

    target = _random_polygon_xy(polygon, rng)
    return [[target[0], target[1], 0.0]], False


def _motion_vector(direction, group_start, group_end, rng):
    if direction == "same_direction":
        return (0.0, 1.0)
    if direction == "opposite_direction":
        return (0.0, -1.0)
    if direction == "left_to_right":
        return (1.0, 0.0)
    if direction == "right_to_left":
        return (-1.0, 0.0)

    if group_start is None or group_end is None:
        angle = rng.uniform(-math.pi, math.pi)
        return (math.cos(angle), math.sin(angle))

    dx = group_end[0] - group_start[0]
    dy = group_end[1] - group_start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        angle = rng.uniform(-math.pi, math.pi)
        return (math.cos(angle), math.sin(angle))
    return (dx / length, dy / length)


def _formation_offsets(group_size, spacing, motion_vec):
    local_offsets = _local_formation_offsets(group_size, spacing)
    forward_x, forward_y = _normalize(motion_vec)
    lateral_x, lateral_y = -forward_y, forward_x
    return [
        (
            forward_x * local_x + lateral_x * local_y,
            forward_y * local_x + lateral_y * local_y,
        )
        for local_x, local_y in local_offsets
    ]


def _local_formation_offsets(group_size, spacing):
    if group_size <= 1:
        return [(0.0, 0.0)]
    if group_size == 2:
        return [(0.0, -spacing * 0.5), (0.0, spacing * 0.5)]
    if group_size == 3:
        return [
            (0.0, -spacing),
            (0.0, 0.0),
            (0.0, spacing),
        ]
    angle_step = 2.0 * math.pi / group_size
    radius = spacing / max(1.0, 2.0 * math.sin(math.pi / group_size))
    return [
        (radius * math.cos(i * angle_step), radius * math.sin(i * angle_step))
        for i in range(group_size)
    ]


def _yaw_from_motion(motion_vec):
    return math.atan2(motion_vec[1], motion_vec[0])


def _normalize(vec):
    x, y = vec
    length = math.hypot(x, y)
    if length < 1e-6:
        return (1.0, 0.0)
    return (x / length, y / length)


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _polygon_bounds(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def _nearest_polygon_xy(x, y, polygon, rng):
    if _point_in_polygon(x, y, polygon):
        return (x, y)

    for radius in (0.5, 1.0, 2.0, 4.0, 8.0):
        for i in range(24):
            angle = 2.0 * math.pi * i / 24.0
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            if _point_in_polygon(px, py, polygon):
                return (px, py)

    return _random_polygon_xy(polygon, rng)


def _nearest_bounded_polygon_xy(x, y, bounds, polygon, rng):
    x_min, x_max, y_min, y_max = bounds
    x = _clamp(x, x_min, x_max)
    y = _clamp(y, y_min, y_max)
    if _point_in_polygon(x, y, polygon):
        return (x, y)

    for radius in (0.3, 0.6, 1.0, 2.0, 4.0):
        for i in range(32):
            angle = 2.0 * math.pi * i / 32.0
            px = _clamp(x + radius * math.cos(angle), x_min, x_max)
            py = _clamp(y + radius * math.sin(angle), y_min, y_max)
            if _point_in_polygon(px, py, polygon):
                return (px, py)

    return _random_bounded_polygon_xy(bounds, polygon, rng)


def _resolve_group_start_spacing(group_start, offsets, bounds, polygon, existing_specs, rng):
    if _is_group_start_valid(group_start, offsets, bounds, polygon, existing_specs):
        return group_start

    x, y = group_start
    safe_x_min, safe_x_max = _safe_center_limits(bounds[0], bounds[1], [offset[0] for offset in offsets])
    safe_y_min, safe_y_max = _safe_center_limits(bounds[2], bounds[3], [offset[1] for offset in offsets])
    for radius in (0.35, 0.6, 0.9, 1.2, 1.6, 2.2):
        for i in range(24):
            angle = 2.0 * math.pi * i / 24.0
            candidate = (
                _clamp(x + radius * math.cos(angle), safe_x_min, safe_x_max),
                _clamp(y + radius * math.sin(angle), safe_y_min, safe_y_max),
            )
            if _is_group_start_valid(candidate, offsets, bounds, polygon, existing_specs):
                return candidate

    for _ in range(120):
        candidate = (rng.uniform(safe_x_min, safe_x_max), rng.uniform(safe_y_min, safe_y_max))
        if _is_group_start_valid(candidate, offsets, bounds, polygon, existing_specs):
            return candidate

    return group_start


def _resolve_fixed_y_start_spacing(start_xy, bounds, polygon, existing_specs, rng):
    """Resolve an initial overlap without changing a longitudinal y slot."""
    if _is_far_enough_from_existing(start_xy, existing_specs):
        return start_xy

    x, y = start_xy
    x_min, x_max = bounds[0], bounds[1]
    directions = (-1.0, 1.0) if rng.random() < 0.5 else (1.0, -1.0)
    for distance in (0.4, 0.8, 1.2, 1.8, 2.6, 3.6, 5.0, 7.0):
        for sign in directions:
            candidate = (_clamp(x + sign * distance, x_min, x_max), y)
            if (
                _point_in_polygon(candidate[0], candidate[1], polygon)
                and _is_far_enough_from_existing(candidate, existing_specs)
            ):
                return candidate

    for _ in range(120):
        candidate = (rng.uniform(x_min, x_max), y)
        if (
            _point_in_polygon(candidate[0], candidate[1], polygon)
            and _is_far_enough_from_existing(candidate, existing_specs)
        ):
            return candidate

    return start_xy


def _is_group_start_valid(group_start, offsets, bounds, polygon, existing_specs):
    x_min, x_max, y_min, y_max = bounds
    for offset in offsets:
        x = group_start[0] + offset[0]
        y = group_start[1] + offset[1]
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False
        if not _point_in_polygon(x, y, polygon):
            return False
        if not _is_far_enough_from_existing((x, y), existing_specs):
            return False
    return True


def _is_far_enough_from_existing(point, existing_specs):
    for spec in existing_specs:
        dx = float(point[0]) - float(spec.init_pos[0])
        dy = float(point[1]) - float(spec.init_pos[1])
        if math.hypot(dx, dy) < MIN_INITIAL_PERSON_DISTANCE:
            return False
    return True


def _nearest_front_polygon_xy(x, y, min_x, polygon, rng):
    if x > min_x and _point_in_polygon(x, y, polygon):
        return (x, y)

    x = max(x, min_x + 0.05)
    if _point_in_polygon(x, y, polygon):
        return (x, y)

    for radius in (0.5, 1.0, 2.0, 4.0, 8.0):
        for i in range(32):
            angle = 2.0 * math.pi * i / 32.0
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            if px > min_x and _point_in_polygon(px, py, polygon):
                return (px, py)

    return _random_front_polygon_xy(min_x, polygon, rng)


def _random_polygon_xy(polygon, rng):
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    for _ in range(1000):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if _point_in_polygon(x, y, polygon):
            return (x, y)
    return ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)


def _random_bounded_polygon_xy(bounds, polygon, rng):
    x_min, x_max, y_min, y_max = bounds
    for _ in range(1000):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if _point_in_polygon(x, y, polygon):
            return (x, y)
    return _random_polygon_xy(polygon, rng)


def _random_front_polygon_xy(min_x, polygon, rng):
    x_min, x_max, y_min, y_max = _polygon_bounds(polygon)
    x_min = max(x_min, min_x)
    if x_min >= x_max:
        return _random_polygon_xy(polygon, rng)

    for _ in range(1000):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if x > min_x and _point_in_polygon(x, y, polygon):
            return (x, y)
    return _random_polygon_xy(polygon, rng)


def _point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) == (y2 > y):
            continue
        x_intersect = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1
        if x < x_intersect:
            inside = not inside
    return inside
