#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carb
from pxr import Usd, UsdGeom

from geometry_utils import inflate_aabb_2d


def _debug_log(*args, **kwargs):
    pass


class ObstacleScanResult:
    def __init__(self, aabbs, keyword_hits, accepted_paths, rejected_paths):
        self.aabbs = aabbs
        self.keyword_hits = keyword_hits
        self.accepted_paths = accepted_paths
        self.rejected_paths = rejected_paths


def collect_obstacle_aabbs(
    stage,
    keywords,
    walk_polygon=None,
    margin=0.7,
    min_size=0.05,
    log_matches=True,
    max_logged_matches=80,
):
    polygon_aabb = None
    if walk_polygon is not None:
        xs = [p[0] for p in walk_polygon]
        ys = [p[1] for p in walk_polygon]
        polygon_aabb = (min(xs), min(ys), max(xs), max(ys))

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    obstacle_aabbs = []
    matched_roots = []
    keyword_hits = []
    accepted_paths = []
    rejected_paths = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(path.startswith(root + "/") for root in matched_roots):
            continue

        name = prim.GetName().lower()
        path_lower = path.lower()
        if not any(keyword in name or keyword in path_lower for keyword in lowered_keywords):
            continue
        keyword_hits.append(path)

        try:
            aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            rejected_paths.append((path, "bbox_failed"))
            _debug_log(f"[PEDESTRIAN][OBSTACLE] matched but bbox failed: {path}")
            continue

        if aligned_range.IsEmpty():
            rejected_paths.append((path, "bbox_empty"))
            _debug_log(f"[PEDESTRIAN][OBSTACLE] matched but bbox empty: {path}")
            continue

        min_pt = aligned_range.GetMin()
        max_pt = aligned_range.GetMax()
        if (max_pt[0] - min_pt[0]) < min_size or (max_pt[1] - min_pt[1]) < min_size:
            rejected_paths.append((path, "too_small"))
            _debug_log(
                f"[PEDESTRIAN][OBSTACLE] matched but too small: {path}, "
                f"raw_xy=({min_pt[0]:.2f},{min_pt[1]:.2f})-({max_pt[0]:.2f},{max_pt[1]:.2f})"
            )
            continue

        aabb = inflate_aabb_2d((min_pt[0], min_pt[1], max_pt[0], max_pt[1]), margin)
        if polygon_aabb is not None and not _aabb_overlaps(aabb, polygon_aabb):
            rejected_paths.append((path, "outside_walk_area"))
            if log_matches and len(keyword_hits) <= max_logged_matches:
                _debug_log(
                    f"[PEDESTRIAN][OBSTACLE] outside walk area: {path}, "
                    f"inflated_xy=({aabb[0]:.2f},{aabb[1]:.2f})-({aabb[2]:.2f},{aabb[3]:.2f})"
                )
            continue

        obstacle_aabbs.append(aabb)
        matched_roots.append(path)
        accepted_paths.append(path)
        if log_matches and len(obstacle_aabbs) <= max_logged_matches:
            _debug_log(
                f"[PEDESTRIAN][OBSTACLE] accepted #{len(obstacle_aabbs)}: {path}, "
                f"inflated_xy=({aabb[0]:.2f},{aabb[1]:.2f})-({aabb[2]:.2f},{aabb[3]:.2f})"
            )

    _debug_log(
        f"[PEDESTRIAN][OBSTACLE] keyword_hits={len(keyword_hits)}, "
        f"accepted={len(obstacle_aabbs)}, keywords={list(keywords)}"
    )
    if log_matches and len(obstacle_aabbs) > max_logged_matches:
        _debug_log(
            f"[PEDESTRIAN][OBSTACLE] {len(obstacle_aabbs) - max_logged_matches} more "
            "accepted obstacles not printed."
        )
    return obstacle_aabbs


def scan_obstacles(
    stage,
    keywords,
    walk_polygon=None,
    margin=0.7,
    min_size=0.05,
    log_matches=True,
    max_logged_matches=80,
    root_paths=None,
    exclude_keywords=None,
    max_bottom_z=None,
    min_top_z=None,
):
    aabbs = []
    keyword_hits = []
    accepted_paths = []
    rejected_paths = []

    polygon_aabb = None
    if walk_polygon is not None:
        xs = [p[0] for p in walk_polygon]
        ys = [p[1] for p in walk_polygon]
        polygon_aabb = (min(xs), min(ys), max(xs), max(ys))

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    lowered_excludes = tuple(
        keyword.lower() for keyword in (exclude_keywords or ())
    )
    matched_roots = []

    for prim in _iter_obstacle_prims(stage, root_paths):
        path = str(prim.GetPath())
        if any(path.startswith(root + "/") for root in matched_roots):
            continue

        name = prim.GetName().lower()
        path_lower = path.lower()
        if any(keyword in name or keyword in path_lower for keyword in lowered_excludes):
            rejected_paths.append((path, "excluded_structure"))
            continue
        if not any(keyword in name or keyword in path_lower for keyword in lowered_keywords):
            continue

        keyword_hits.append(path)
        try:
            aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            rejected_paths.append((path, "bbox_failed"))
            continue
        if aligned_range.IsEmpty():
            rejected_paths.append((path, "bbox_empty"))
            continue

        min_pt = aligned_range.GetMin()
        max_pt = aligned_range.GetMax()
        if max_bottom_z is not None and float(min_pt[2]) > float(max_bottom_z):
            rejected_paths.append((path, "above_pedestrian_ground_zone"))
            continue
        if min_top_z is not None and float(max_pt[2]) < float(min_top_z):
            rejected_paths.append((path, "below_pedestrian_body_zone"))
            continue
        if (max_pt[0] - min_pt[0]) < min_size or (max_pt[1] - min_pt[1]) < min_size:
            rejected_paths.append((path, "too_small"))
            continue

        aabb = inflate_aabb_2d((min_pt[0], min_pt[1], max_pt[0], max_pt[1]), margin)
        if polygon_aabb is not None and not _aabb_overlaps(aabb, polygon_aabb):
            rejected_paths.append((path, "outside_walk_area"))
            continue

        aabbs.append(aabb)
        accepted_paths.append(path)
        matched_roots.append(path)

    return ObstacleScanResult(aabbs, keyword_hits, accepted_paths, rejected_paths)


def _iter_obstacle_prims(stage, root_paths=None):
    seen = set()

    if root_paths:
        for root_path in root_paths:
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                _debug_log(f"[APP][OBSTACLE] root path not found: {root_path}")
                continue
            try:
                root.Load()
            except Exception:
                pass
            for prim in _iter_prim_tree(root):
                path = str(prim.GetPath())
                if path not in seen:
                    seen.add(path)
                    yield prim
        return

    traverse = getattr(stage, "TraverseAll", None)
    prim_iter = traverse() if traverse is not None else stage.Traverse()
    for prim in prim_iter:
        path = str(prim.GetPath())
        if path not in seen:
            seen.add(path)
            yield prim


def _iter_prim_tree(root):
    stack = [root]
    while stack:
        prim = stack.pop()
        yield prim
        children = prim.GetAllChildren()
        for child in reversed(children):
            stack.append(child)


def sample_paths_under_roots(stage, root_paths, limit=80):
    paths = []
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            paths.append(f"{root_path}  <root not found>")
            continue
        for prim in _iter_prim_tree(root):
            paths.append(str(prim.GetPath()))
            if len(paths) >= limit:
                return paths
    return paths


def _aabb_overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0
