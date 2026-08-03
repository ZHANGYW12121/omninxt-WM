#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math


def point_in_polygon_2d(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        intersect = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1
        )
        if intersect:
            inside = not inside
    return inside


def sample_point_in_rectangle(xmin, xmax, ymin, ymax):
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    return [x, y, 0.0]


def sample_point_in_polygon(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    while True:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if point_in_polygon_2d(x, y, polygon):
            return [x, y, 0.0]


def inflate_aabb_2d(aabb, margin):
    xmin, ymin, xmax, ymax = aabb
    return (xmin - margin, ymin - margin, xmax + margin, ymax + margin)


def point_in_aabb_2d(x, y, aabb):
    xmin, ymin, xmax, ymax = aabb
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_in_any_aabb_2d(x, y, aabbs):
    return any(point_in_aabb_2d(x, y, aabb) for aabb in aabbs)


def point_is_walkable(x, y, polygon, obstacle_aabbs):
    return point_in_polygon_2d(x, y, polygon) and not point_in_any_aabb_2d(
        x, y, obstacle_aabbs
    )


def point_is_front_walkable(x, y, polygon, obstacle_aabbs, min_x, max_x=None):
    return (
        x > min_x
        and (max_x is None or x <= max_x)
        and point_is_walkable(x, y, polygon, obstacle_aabbs)
    )


def segment_intersects_aabb_2d(p0, p1, aabb):
    x0, y0 = p0
    x1, y1 = p1
    xmin, ymin, xmax, ymax = aabb

    dx = x1 - x0
    dy = y1 - y0
    t_min = 0.0
    t_max = 1.0

    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) < 1e-9:
            if q < 0.0:
                return False
            continue

        t = q / p
        if p < 0.0:
            if t > t_max:
                return False
            t_min = max(t_min, t)
        else:
            if t < t_min:
                return False
            t_max = min(t_max, t)

    return True


def segment_is_clear_2d(p0, p1, obstacle_aabbs):
    for aabb in obstacle_aabbs:
        if segment_intersects_aabb_2d(p0, p1, aabb):
            return False
    return True


def sample_point_in_polygon_avoiding_obstacles(polygon, obstacle_aabbs, max_attempts=500):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    for _ in range(max_attempts):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if point_is_walkable(x, y, polygon, obstacle_aabbs):
            return [x, y, 0.0]

    return sample_point_in_polygon(polygon)


def nearest_walkable_or_sample(preferred, polygon, obstacle_aabbs):
    x, y = float(preferred[0]), float(preferred[1])
    if point_is_walkable(x, y, polygon, obstacle_aabbs):
        return [x, y, 0.0]

    for radius in (1.0, 2.0, 4.0, 8.0):
        for i in range(24):
            theta = 2.0 * math.pi * i / 24.0
            px = x + radius * math.cos(theta)
            py = y + radius * math.sin(theta)
            if point_is_walkable(px, py, polygon, obstacle_aabbs):
                return [px, py, 0.0]

    return sample_point_in_polygon_avoiding_obstacles(polygon, obstacle_aabbs)


def sample_point_in_front_avoiding_obstacles(
    polygon,
    obstacle_aabbs,
    min_x,
    max_x=None,
    max_attempts=2000,
):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    xmin = max(min(xs), min_x)
    xmax = max(xs) if max_x is None else min(max(xs), max_x)
    ymin, ymax = min(ys), max(ys)

    if xmin >= xmax:
        return sample_point_in_polygon_avoiding_obstacles(polygon, obstacle_aabbs)

    for _ in range(max_attempts):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if point_is_front_walkable(x, y, polygon, obstacle_aabbs, min_x, max_x):
            return [x, y, 0.0]

    if not obstacle_aabbs:
        for _ in range(max_attempts):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            if point_in_polygon_2d(x, y, polygon) and x > min_x:
                return [x, y, 0.0]

    return sample_point_in_polygon_avoiding_obstacles(polygon, obstacle_aabbs)


def nearest_front_walkable_or_sample(preferred, polygon, obstacle_aabbs, min_x, max_x=None):
    x, y = float(preferred[0]), float(preferred[1])
    if point_is_front_walkable(x, y, polygon, obstacle_aabbs, min_x, max_x):
        return [x, y, 0.0]

    for radius in (0.5, 1.0, 2.0, 4.0, 8.0):
        for i in range(32):
            theta = 2.0 * math.pi * i / 32.0
            px = x + radius * math.cos(theta)
            py = y + radius * math.sin(theta)
            if point_is_front_walkable(px, py, polygon, obstacle_aabbs, min_x, max_x):
                return [px, py, 0.0]

    return sample_point_in_front_avoiding_obstacles(
        polygon,
        obstacle_aabbs,
        min_x,
        max_x=max_x,
    )
