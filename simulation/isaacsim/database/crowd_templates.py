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
):
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
