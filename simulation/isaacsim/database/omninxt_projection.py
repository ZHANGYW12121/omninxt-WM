#!/usr/bin/env python3
"""Projection helpers for the 2026-08-02 OmniNxt Sim2Real camera rig.

Isaac Sim does not render a native Mei unified camera with Radtan distortion.
The production profile therefore renders an ideal equidistant F-theta source
and resamples it into the formally calibrated Mei image.  The resampling maps
are static and are built once per camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RemapDiagnostics:
    valid_fraction: float
    median_forward_error_px: float
    max_forward_error_px: float
    max_ray_angle_deg: float


class MeiRadtanRemapper:
    """Map an ideal equidistant render into a Mei+radtan target image."""

    def __init__(
        self,
        intrinsics,
        distortion,
        target_resolution,
        source_resolution,
        source_fov_deg,
        newton_iterations=32,
        convergence_px=0.02,
    ):
        self.xi, self.fx, self.fy, self.cx, self.cy = (
            float(value) for value in intrinsics
        )
        self.k1, self.k2, self.p1, self.p2 = (
            float(value) for value in distortion
        )
        self.target_width, self.target_height = (
            int(value) for value in target_resolution
        )
        self.source_width, self.source_height = (
            int(value) for value in source_resolution
        )
        self.source_fov_deg = float(source_fov_deg)
        self.source_focal_px = min(self.source_width, self.source_height) / (
            2.0 * math.radians(self.source_fov_deg * 0.5)
        )
        self.source_cx = (self.source_width - 1.0) * 0.5
        self.source_cy = (self.source_height - 1.0) * 0.5
        self.map_x, self.map_y, self.valid_mask, self.diagnostics = self._build_maps(
            int(newton_iterations), float(convergence_px)
        )

    def _build_maps(self, iterations, convergence_px):
        pixel_y, pixel_x = np.indices(
            (self.target_height, self.target_width), dtype=np.float64
        )
        distorted_x = (pixel_x - self.cx) / self.fx
        distorted_y = (pixel_y - self.cy) / self.fy
        x = distorted_x.copy()
        y = distorted_y.copy()

        # Vectorized damped Newton inversion of Radtan.  CAM_D has a strongly
        # curved calibration near the image boundary, so a fixed-point inverse
        # is not sufficient and would silently select invalid rays.
        for _ in range(max(1, iterations)):
            radius2 = x * x + y * y
            radial = 1.0 + self.k1 * radius2 + self.k2 * radius2 * radius2
            radial_x = 2.0 * self.k1 * x + 4.0 * self.k2 * radius2 * x
            radial_y = 2.0 * self.k1 * y + 4.0 * self.k2 * radius2 * y
            value_x = (
                x * radial
                + 2.0 * self.p1 * x * y
                + self.p2 * (radius2 + 2.0 * x * x)
                - distorted_x
            )
            value_y = (
                y * radial
                + self.p1 * (radius2 + 2.0 * y * y)
                + 2.0 * self.p2 * x * y
                - distorted_y
            )
            jac_xx = radial + x * radial_x + 2.0 * self.p1 * y + 6.0 * self.p2 * x
            jac_xy = x * radial_y + 2.0 * self.p1 * x + 2.0 * self.p2 * y
            jac_yx = y * radial_x + 2.0 * self.p1 * x + 2.0 * self.p2 * y
            jac_yy = radial + y * radial_y + 6.0 * self.p1 * y + 2.0 * self.p2 * x
            determinant = jac_xx * jac_yy - jac_xy * jac_yx
            invertible = np.abs(determinant) > 1.0e-12
            step_x = np.where(
                invertible,
                (jac_yy * value_x - jac_xy * value_y) / determinant,
                0.0,
            )
            step_y = np.where(
                invertible,
                (-jac_yx * value_x + jac_xx * value_y) / determinant,
                0.0,
            )
            # Limit a Newton jump so boundary pixels cannot poison a large
            # vector block with infinities before being marked invalid.
            step_norm = np.sqrt(step_x * step_x + step_y * step_y)
            damping = np.maximum(1.0, step_norm / 0.35)
            x -= step_x / damping
            y -= step_y / damping

        radius2 = x * x + y * y
        radial = 1.0 + self.k1 * radius2 + self.k2 * radius2 * radius2
        forward_x = (
            x * radial
            + 2.0 * self.p1 * x * y
            + self.p2 * (radius2 + 2.0 * x * x)
        )
        forward_y = (
            y * radial
            + self.p1 * (radius2 + 2.0 * y * y)
            + 2.0 * self.p2 * x * y
        )
        forward_error_px = np.sqrt(
            ((forward_x - distorted_x) * self.fx) ** 2
            + ((forward_y - distorted_y) * self.fy) ** 2
        )

        root_term = 1.0 + (1.0 - self.xi * self.xi) * radius2
        valid = np.isfinite(forward_error_px) & (forward_error_px <= convergence_px)
        valid &= root_term >= 0.0
        root = np.sqrt(np.maximum(root_term, 0.0))
        scale = (self.xi + root) / (1.0 + radius2)
        ray_x = scale * x
        ray_y = scale * y
        ray_z = scale - self.xi
        ray_angle = np.arctan2(np.sqrt(ray_x * ray_x + ray_y * ray_y), ray_z)
        source_radius = self.source_focal_px * ray_angle
        azimuth = np.arctan2(ray_y, ray_x)
        map_x = self.source_cx + source_radius * np.cos(azimuth)
        map_y = self.source_cy + source_radius * np.sin(azimuth)
        valid &= ray_angle <= math.radians(self.source_fov_deg * 0.5)
        valid &= (map_x >= 0.0) & (map_x <= self.source_width - 1.0)
        valid &= (map_y >= 0.0) & (map_y <= self.source_height - 1.0)

        map_x = np.where(valid, map_x, -1.0).astype(np.float32)
        map_y = np.where(valid, map_y, -1.0).astype(np.float32)
        finite_error = forward_error_px[valid]
        finite_angle = ray_angle[valid]
        diagnostics = RemapDiagnostics(
            valid_fraction=float(np.mean(valid)),
            median_forward_error_px=(
                float(np.median(finite_error)) if finite_error.size else float("inf")
            ),
            max_forward_error_px=(
                float(np.max(finite_error)) if finite_error.size else float("inf")
            ),
            max_ray_angle_deg=(
                float(np.degrees(np.max(finite_angle))) if finite_angle.size else 0.0
            ),
        )
        return map_x, map_y, valid, diagnostics

    def remap(self, source, interpolation=None):
        """Return one exact-calibration target image from an Isaac render."""
        import cv2

        array = np.asarray(source)
        expected = (self.source_height, self.source_width)
        if array.shape[:2] != expected:
            raise ValueError(
                f"source shape {array.shape[:2]} does not match {expected}"
            )
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR
        output = cv2.remap(
            array,
            self.map_x,
            self.map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if output.ndim == 2:
            output[~self.valid_mask] = 0
        else:
            output[~self.valid_mask, ...] = 0
        return np.ascontiguousarray(output)


def ideal_ftheta_coefficients(source_resolution, source_fov_deg):
    """Isaac F-theta polynomial ``theta=A+B*r+...`` for equidistant rays."""
    width, height = (int(value) for value in source_resolution)
    focal = min(width, height) / (2.0 * math.radians(float(source_fov_deg) * 0.5))
    return [0.0, 1.0 / focal, 0.0, 0.0, 0.0]
