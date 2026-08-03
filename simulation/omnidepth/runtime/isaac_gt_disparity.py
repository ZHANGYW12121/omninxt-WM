#!/usr/bin/env python3
"""Convert synchronized Isaac fisheye range mosaics to rectified disparity.

The input range is Isaac ``distance_to_camera`` (Euclidean optical-centre
range), not optical-Z.  Each stereo output therefore samples the physical
left-camera range along its calibrated rectified ray, converts that range to
rectified optical-Z, and algebraically inverts the pair's OpenCV Q matrix.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import yaml


PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))
RECTIFIED_SIZE = (320, 240)
RAW_CAMERA_SIZE = (1280, 720)


class IsaacGtDisparity:
    def __init__(self, config_dir, min_z=0.1, max_z=10.0):
        self.config_dir = Path(config_dir)
        self.min_z = float(min_z)
        self.max_z = float(max_z)
        fisheye_path = self.config_dir / "fisheye_cams.yaml"
        self.fisheye = yaml.safe_load(fisheye_path.read_text(encoding="utf-8"))
        self._sectors = [self._build_sector(sid, pair)
                         for sid, pair in enumerate(PAIRS)]

    @staticmethod
    def _distortion(node):
        values = node.get("distortion_coeffs", node.get("distortion"))
        if values is None or len(values) != 4:
            raise ValueError("Mei camera requires four radtan coefficients")
        return tuple(float(value) for value in values)

    def _build_sector(self, stereo_id, pair):
        left, right = pair
        stereo_path = self.config_dir / (
            f"stereo_calib_{left}_{right}_240_320.yaml"
        )
        stereo = yaml.safe_load(stereo_path.read_text(encoding="utf-8"))
        intr = tuple(float(value) for value in stereo["cam0"]["intrinsics"])
        k = np.array(((intr[0], 0.0, intr[2]),
                      (0.0, intr[1], intr[3]),
                      (0.0, 0.0, 1.0)), dtype=np.float64)
        d0 = np.asarray(stereo["cam0"]["distortion_coeffs"], np.float64)
        d1 = np.asarray(stereo["cam1"]["distortion_coeffs"], np.float64)
        baseline = np.asarray(stereo["cam1"]["T_cn_cnm1"], np.float64)
        r1, _, p1, _, q, _, _ = cv2.stereoRectify(
            k, d0, k, d1, RECTIFIED_SIZE,
            baseline[:3, :3], baseline[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
        )

        pixel_u, pixel_v = np.meshgrid(
            np.arange(RECTIFIED_SIZE[0], dtype=np.float32),
            np.arange(RECTIFIED_SIZE[1], dtype=np.float32),
        )
        rectified_ray = np.stack((
            (pixel_u - p1[0, 2]) / p1[0, 0],
            (pixel_v - p1[1, 2]) / p1[1, 1],
            np.ones_like(pixel_u),
        ), axis=-1)

        # FisheyeUndist half id 1 is the +45 degree virtual-left view.
        angle = math.pi / 4.0
        virtual_to_fisheye = np.array((
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        ), dtype=np.float64)
        mei_ray = rectified_ray @ r1 @ virtual_to_fisheye.T
        mei_ray /= np.linalg.norm(mei_ray, axis=2, keepdims=True)

        camera = self.fisheye[f"cam{left}"]
        xi, fx, fy, cx, cy = (float(value)
                              for value in camera["intrinsics"])
        k1, k2, p_tan1, p_tan2 = self._distortion(camera)
        denominator = mei_ray[..., 2] + xi
        x = mei_ray[..., 0] / denominator
        y = mei_ray[..., 1] / denominator
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
        distorted_x = (x * radial + 2.0 * p_tan1 * x * y
                       + p_tan2 * (radius2 + 2.0 * x * x))
        distorted_y = (y * radial + p_tan1 * (radius2 + 2.0 * y * y)
                       + 2.0 * p_tan2 * x * y)
        map_x = np.asarray(fx * distorted_x + cx, dtype=np.float32)
        map_y = np.asarray(fy * distorted_y + cy, dtype=np.float32)

        # For a unit Euclidean range along mei_ray, this is rectified optical-Z.
        axial_factor = (mei_ray @ virtual_to_fisheye @ r1.T)[..., 2]
        map_valid = (
            np.isfinite(map_x) & np.isfinite(map_y)
            & (map_x >= 0.0) & (map_x < RAW_CAMERA_SIZE[0] - 1)
            & (map_y >= 0.0) & (map_y < RAW_CAMERA_SIZE[1] - 1)
            & np.isfinite(axial_factor) & (axial_factor > 1e-6)
        )
        return {
            "id": stereo_id,
            "left": left,
            "right": right,
            "map_x": map_x,
            "map_y": map_y,
            "map_valid": map_valid,
            "axial_factor": np.asarray(axial_factor, dtype=np.float32),
            "q": q,
        }

    def convert(self, quad_range):
        quad_range = np.asarray(quad_range, dtype=np.float32)
        expected = (RAW_CAMERA_SIZE[1], RAW_CAMERA_SIZE[0] * 4)
        if quad_range.shape != expected:
            raise ValueError(
                f"GT range mosaic must be {expected}, got {quad_range.shape}"
            )
        outputs = []
        stats = []
        for sector in self._sectors:
            left = sector["left"]
            physical_range = quad_range[:,
                left * RAW_CAMERA_SIZE[0]:(left + 1) * RAW_CAMERA_SIZE[0]]
            sampled_range = cv2.remap(
                physical_range, sector["map_x"], sector["map_y"],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=float("nan"),
            )
            axial_z = sampled_range * sector["axial_factor"]
            q = sector["q"]
            disparity = np.full(axial_z.shape, np.nan, dtype=np.float32)
            valid = (
                sector["map_valid"] & np.isfinite(sampled_range)
                & (sampled_range > 0.03) & np.isfinite(axial_z)
                & (axial_z > self.min_z) & (axial_z < self.max_z)
            )
            disparity[valid] = np.asarray(
                (q[2, 3] / axial_z[valid] - q[3, 3]) / q[3, 2],
                dtype=np.float32,
            )
            valid &= np.isfinite(disparity) & (disparity > 0.0)
            disparity[~valid] = np.nan
            outputs.append(disparity)
            finite = disparity[valid]
            stats.append({
                "stereo": sector["id"],
                "valid": int(finite.size),
                "median_px": float(np.median(finite)) if finite.size else None,
            })
        return np.ascontiguousarray(np.vstack(outputs), dtype=np.float32), stats
