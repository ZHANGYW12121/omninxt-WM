"""Joint topology helpers shared by the compact dataset and world model."""

from __future__ import annotations

import numpy as np
import torch


COCO17_JOINT_COUNT = 17
COCO12_BODY_JOINT_COUNT = 12

# COCO17 indices 0..4 (nose, eyes and ears) are deliberately not recorded.
COCO12_BODY_SOURCE_INDICES = tuple(range(5, 17))
COCO12_BODY_JOINT_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
COCO12_BODY_EDGES = (
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),
)

# Causal observation sanitation shared by replay and deployment.  The bounds
# are 1.25 times the already-versioned tracker plausibility maxima, leaving a
# generous margin for perspective/depth error while excluding metre-scale
# pose explosions that cannot be a human body. A missing *middle* joint is
# interpolated only when the proximal and distal points of the same limb are
# directly measured in the current frame. Endpoints are never extrapolated.
# No temporal look-ahead, simulator identity or privileged collision geometry
# participates in this observation path.
COCO12_GEOMETRY_SANITATION_CONTRACT = (
    "coco12_causal_two_measured_anchor_middle_joint_interpolation_"
    "anthropometric_"
    "projection_margin1p25_root_scaled_v3")
COCO12_MAX_CENTER_RADIUS_M = 1.60
COCO12_MAX_HIP_ROOT_MEDIAN_OFFSET_M = 0.90
COCO12_MAX_EDGE_LENGTH_M = {
    (0, 1): 0.875,
    (0, 2): 0.6875,
    (2, 4): 0.625,
    (1, 3): 0.6875,
    (3, 5): 0.625,
    (0, 6): 1.0625,
    (1, 7): 1.0625,
    (6, 7): 0.750,
    (6, 8): 0.9375,
    (8, 10): 0.9375,
    (7, 9): 0.9375,
    (9, 11): 0.9375,
}

# Proximal, middle and distal indices.  The maximum-length ratios above are
# used only as anthropometric proportions; completion is anchored entirely to
# same-frame observed coordinates and remains deterministic.
COCO12_THREE_JOINT_LIMBS = (
    (0, 2, 4),
    (1, 3, 5),
    (6, 8, 10),
    (7, 9, 11),
)


def complete_coco12_topology_numpy(
    positions: np.ndarray, valid: np.ndarray,
    measured: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causally interpolate a missing middle point from two measured anchors.

    Endpoint extrapolation produced metre-scale errors against simulator
    geometry and is deliberately forbidden. The returned ``completed`` mask
    lets callers mark inferred points as predicted and exclude them from
    measurement-only supervision.
    """
    xyz = np.asarray(positions, np.float32)
    mask = np.asarray(valid, np.bool_)
    if xyz.shape[-2:] != (COCO12_BODY_JOINT_COUNT, 3):
        raise ValueError("COCO12 positions must end in [12,3]")
    if mask.shape != xyz.shape[:-1]:
        raise ValueError("COCO12 validity must match positions")
    anchor_mask = (
        mask.copy()
        if measured is None else np.asarray(measured, np.bool_).copy())
    if anchor_mask.shape != mask.shape:
        raise ValueError("COCO12 measured mask must match validity")
    anchor_mask &= mask
    if not np.isfinite(xyz).all():
        raise ValueError("COCO12 positions must be finite")
    result = xyz.copy()
    output_valid = mask.copy()
    completed = np.zeros_like(output_valid)
    for first, middle, last in COCO12_THREE_JOINT_LIMBS:
        first_length = float(COCO12_MAX_EDGE_LENGTH_M[(first, middle)])
        last_length = float(COCO12_MAX_EDGE_LENGTH_M[(middle, last)])
        total_length = first_length + last_length

        fill_middle = (
            ~output_valid[..., middle]
            & anchor_mask[..., first]
            & anchor_mask[..., last]
        )
        middle_value = (
            result[..., first, :]
            + (result[..., last, :] - result[..., first, :])
            * (first_length / total_length)
        )
        result[..., middle, :] = np.where(
            fill_middle[..., None], middle_value, result[..., middle, :])
        output_valid[..., middle] |= fill_middle
        completed[..., middle] |= fill_middle
    return (
        result.astype(np.float32, copy=False),
        output_valid,
        completed,
    )


def _coco12_robust_center(
    positions: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    return _coco12_robust_centers_numpy(
        np.asarray(positions, np.float32)[None],
        np.asarray(valid, np.bool_)[None],
    )[0]


def _coco12_robust_centers_numpy(
    positions: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    """Vectorized coordinate median over the at-most-twelve valid joints."""
    xyz = np.asarray(positions, np.float32)
    mask = np.asarray(valid, np.bool_)
    if xyz.shape[-2:] != (COCO12_BODY_JOINT_COUNT, 3):
        raise ValueError("COCO12 positions must end in [12,3]")
    if mask.shape != xyz.shape[:-1]:
        raise ValueError("COCO12 validity must match positions")
    # Sorting twelve elements is faster than a Python loop over every person
    # frame and avoids nanmedian's all-invalid warnings for padded slots.
    ordered = np.sort(
        np.where(mask[..., None], xyz, np.inf), axis=-2)
    count = mask.sum(axis=-1)
    lower = np.maximum((count - 1) // 2, 0)
    upper = np.maximum(count // 2, 0)
    gather_shape = lower.shape + (1, 3)
    lower_value = np.take_along_axis(
        ordered,
        np.broadcast_to(lower[..., None, None], gather_shape),
        axis=-2,
    )[..., 0, :]
    upper_value = np.take_along_axis(
        ordered,
        np.broadcast_to(upper[..., None, None], gather_shape),
        axis=-2,
    )[..., 0, :]
    center = 0.5 * (lower_value + upper_value)
    return np.where(count[..., None] > 0, center, 0.0).astype(np.float32)


def coco12_robust_root_numpy(
    positions: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    """Return a pelvis root unless the hip pair is geometrically corrupted."""
    xyz = np.asarray(positions, np.float32)
    mask = np.asarray(valid, np.bool_)
    if xyz.shape[-2:] != (COCO12_BODY_JOINT_COUNT, 3):
        raise ValueError("COCO12 positions must end in [12,3]")
    if mask.shape != xyz.shape[:-1]:
        raise ValueError("COCO12 validity must match positions")
    center = _coco12_robust_centers_numpy(xyz, mask)
    hips_valid = mask[..., 6] & mask[..., 7]
    hip_delta = xyz[..., 6, :] - xyz[..., 7, :]
    hip_midpoint = 0.5 * (xyz[..., 6, :] + xyz[..., 7, :])
    robust_hips = (
        hips_valid
        & (np.linalg.norm(hip_delta, axis=-1)
           <= COCO12_MAX_EDGE_LENGTH_M[(6, 7)] + 1.0e-6)
        & (np.linalg.norm(hip_midpoint - center, axis=-1)
           <= COCO12_MAX_HIP_ROOT_MEDIAN_OFFSET_M + 1.0e-6)
    )
    return np.where(robust_hips[..., None], hip_midpoint, center).astype(
        np.float32)


def sanitize_coco12_geometry_numpy(
    positions: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project impossible COCO12 geometry and report every changed joint.

    The coordinate-wise median supplies a robust same-frame body centre.  For
    an overlong edge, the endpoint farther from that centre is moved while the
    more central endpoint stays fixed.  This retains a conservative physical
    obstacle instead of deleting a limb, but changed measurements must be
    labelled as predicted by the caller and excluded from factual pose loss.
    """
    xyz = np.asarray(positions, np.float32)
    mask = np.asarray(valid, np.bool_)
    if xyz.shape[-2:] != (COCO12_BODY_JOINT_COUNT, 3):
        raise ValueError("COCO12 positions must end in [12,3]")
    if mask.shape != xyz.shape[:-1]:
        raise ValueError("COCO12 validity must match positions")
    if not np.isfinite(xyz).all():
        raise ValueError("COCO12 positions must be finite")
    original_shape = xyz.shape
    result = xyz.reshape(-1, COCO12_BODY_JOINT_COUNT, 3).copy()
    flat_mask = mask.reshape(-1, COCO12_BODY_JOINT_COUNT)
    active = flat_mask.sum(axis=-1) >= 2
    center = _coco12_robust_centers_numpy(result, flat_mask)
    for _ in range(6):
        displacement = result - center[:, None, :]
        radius = np.linalg.norm(displacement, axis=-1)
        scale = np.minimum(
            1.0,
            COCO12_MAX_CENTER_RADIUS_M / np.maximum(radius, 1.0e-8),
        )
        radial_valid = flat_mask & active[:, None]
        result = np.where(
            radial_valid[..., None],
            center[:, None, :] + displacement * scale[..., None],
            result,
        )

        hips_valid = flat_mask[:, 6] & flat_mask[:, 7] & active
        hip_midpoint = 0.5 * (result[:, 6] + result[:, 7])
        hip_offset = hip_midpoint - center
        hip_norm = np.linalg.norm(hip_offset, axis=-1)
        hip_scale = np.maximum(
            0.0,
            1.0 - COCO12_MAX_HIP_ROOT_MEDIAN_OFFSET_M
            / np.maximum(hip_norm, 1.0e-8),
        )
        hip_shift = hip_offset * hip_scale[:, None]
        hip_shift[~hips_valid] = 0.0
        result[:, 6] -= hip_shift
        result[:, 7] -= hip_shift

        for (first, second), maximum in COCO12_MAX_EDGE_LENGTH_M.items():
            edge_valid = (
                flat_mask[:, first] & flat_mask[:, second] & active)
            first_point = result[:, first].copy()
            second_point = result[:, second].copy()
            edge = second_point - first_point
            length = np.linalg.norm(edge, axis=-1)
            violation = edge_valid & (length > maximum)
            first_radius = np.linalg.norm(first_point - center, axis=-1)
            second_radius = np.linalg.norm(second_point - center, axis=-1)
            move_second = first_radius <= second_radius
            edge_scale = float(maximum) / np.maximum(length, 1.0e-8)
            projected_second = first_point + edge * edge_scale[:, None]
            projected_first = second_point - edge * edge_scale[:, None]
            move_first_mask = violation & ~move_second
            move_second_mask = violation & move_second
            result[move_first_mask, first] = projected_first[move_first_mask]
            result[move_second_mask, second] = projected_second[move_second_mask]

    # Shared joints make sequential edge projections only asymptotically
    # feasible. A final uniform contraction about the authoritative robust
    # root preserves pose directions and satisfies all constraints together.
    for _ in range(2):
        root = coco12_robust_root_numpy(result, flat_mask)
        displacement = result - root[:, None, :]
        radial_ratio = (
            np.linalg.norm(displacement, axis=-1)
            / COCO12_MAX_CENTER_RADIUS_M)
        radial_ratio[~flat_mask] = 0.0
        maximum_ratio = np.maximum(1.0, radial_ratio.max(axis=-1))
        for (first, second), maximum in COCO12_MAX_EDGE_LENGTH_M.items():
            edge_ratio = (
                np.linalg.norm(
                    result[:, second] - result[:, first], axis=-1)
                / float(maximum))
            edge_ratio[~(flat_mask[:, first] & flat_mask[:, second])] = 0.0
            maximum_ratio = np.maximum(maximum_ratio, edge_ratio)
        hips_valid = flat_mask[:, 6] & flat_mask[:, 7]
        hip_ratio = (
            np.linalg.norm(
                0.5 * (result[:, 6] + result[:, 7]) - root, axis=-1)
            / COCO12_MAX_HIP_ROOT_MEDIAN_OFFSET_M)
        hip_ratio[~hips_valid] = 0.0
        maximum_ratio = np.maximum(maximum_ratio, hip_ratio)
        contracted = (
            root[:, None, :] + displacement / maximum_ratio[:, None, None])
        result = np.where(
            (flat_mask & active[:, None])[..., None], contracted, result)

    output = result.reshape(original_shape)
    changed = mask & np.any(np.abs(output - xyz) > 1.0e-6, axis=-1)
    return output.astype(np.float32, copy=False), changed


def project_coco12_geometry_torch(
    positions: torch.Tensor,
    valid: torch.Tensor,
    root: torch.Tensor,
) -> torch.Tensor:
    """Differentiably keep recursive imagined joints inside the same body."""
    if positions.shape[-2:] != (COCO12_BODY_JOINT_COUNT, 3):
        raise ValueError("COCO12 positions must end in [12,3]")
    if valid.shape != positions.shape[:-1]:
        raise ValueError("COCO12 validity must match positions")
    if root.shape != positions.shape[:-2] + (3,):
        raise ValueError("COCO12 root must match leading person dimensions")
    result = positions
    mask = valid.bool()
    root_point = root.unsqueeze(-2)
    for _ in range(6):
        displacement = result - root_point
        radius = torch.linalg.vector_norm(
            displacement, dim=-1, keepdim=True)
        radial_scale = torch.clamp(
            result.new_tensor(COCO12_MAX_CENTER_RADIUS_M)
            / radius.clamp_min(1.0e-8),
            max=1.0,
        )
        result = torch.where(
            mask.unsqueeze(-1),
            root_point + displacement * radial_scale,
            result,
        )
        hips_valid = (mask[..., 6] & mask[..., 7]).unsqueeze(-1)
        hip_midpoint = 0.5 * (
            result[..., 6, :] + result[..., 7, :])
        hip_offset = hip_midpoint - root
        hip_offset_norm = torch.linalg.vector_norm(
            hip_offset, dim=-1, keepdim=True)
        hip_shift = hip_offset * torch.clamp(
            1.0
            - result.new_tensor(COCO12_MAX_HIP_ROOT_MEDIAN_OFFSET_M)
            / hip_offset_norm.clamp_min(1.0e-8),
            min=0.0,
        )
        hip_shift = torch.where(
            hips_valid, hip_shift, torch.zeros_like(hip_shift))
        replacement = torch.zeros_like(result)
        replacement[..., 6, :] = -hip_shift
        replacement[..., 7, :] = -hip_shift
        result = result + replacement
        for (first, second), maximum in COCO12_MAX_EDGE_LENGTH_M.items():
            edge_valid = mask[..., first] & mask[..., second]
            first_point = result[..., first, :]
            second_point = result[..., second, :]
            edge = second_point - first_point
            length = torch.linalg.vector_norm(edge, dim=-1, keepdim=True)
            violation = edge_valid.unsqueeze(-1) & (length > maximum)
            first_radius = torch.linalg.vector_norm(
                first_point - root, dim=-1, keepdim=True)
            second_radius = torch.linalg.vector_norm(
                second_point - root, dim=-1, keepdim=True)
            move_second = first_radius <= second_radius
            projected_second = (
                first_point
                + edge * (float(maximum) / length.clamp_min(1.0e-8)))
            projected_first = (
                second_point
                - edge * (float(maximum) / length.clamp_min(1.0e-8)))
            next_first = torch.where(
                violation & ~move_second, projected_first, first_point)
            next_second = torch.where(
                violation & move_second, projected_second, second_point)
            replacement = torch.zeros_like(result)
            replacement[..., first, :] = next_first - first_point
            replacement[..., second, :] = next_second - second_point
            result = result + replacement
    displacement = result - root_point
    radius_ratio = (
        torch.linalg.vector_norm(displacement, dim=-1)
        / float(COCO12_MAX_CENTER_RADIUS_M)
    ).masked_fill(~mask, 0.0)
    maximum_ratio = radius_ratio.amax(dim=-1).clamp_min(1.0)
    for (first, second), maximum in COCO12_MAX_EDGE_LENGTH_M.items():
        edge_ratio = (
            torch.linalg.vector_norm(
                result[..., second, :] - result[..., first, :], dim=-1)
            / float(maximum)
        )
        edge_ratio = edge_ratio.masked_fill(
            ~(mask[..., first] & mask[..., second]), 0.0)
        maximum_ratio = torch.maximum(maximum_ratio, edge_ratio)
    hips_valid = mask[..., 6] & mask[..., 7]
    hip_ratio = (
        torch.linalg.vector_norm(
            0.5 * (result[..., 6, :] + result[..., 7, :]) - root,
            dim=-1,
        ) / float(COCO12_MAX_HIP_ROOT_MEDIAN_OFFSET_M)
    ).masked_fill(~hips_valid, 0.0)
    maximum_ratio = torch.maximum(maximum_ratio, hip_ratio)
    contracted = root_point + displacement / maximum_ratio[..., None, None]
    result = torch.where(mask.unsqueeze(-1), contracted, result)
    return result


def hip_joint_indices(num_joints: int) -> tuple[int, int] | None:
    """Return left/right hip indices for a supported stored topology."""
    count = int(num_joints)
    if count == COCO12_BODY_JOINT_COUNT:
        return 6, 7
    if count >= COCO17_JOINT_COUNT:
        return 11, 12
    return None


def skeleton_edges(num_joints: int) -> tuple[tuple[int, int], ...]:
    """Return an exact topology; silently truncating COCO17 is incorrect."""
    count = int(num_joints)
    if count == COCO12_BODY_JOINT_COUNT:
        return COCO12_BODY_EDGES
    if count == COCO17_JOINT_COUNT:
        return COCO17_EDGES
    raise ValueError(f"Unsupported skeleton topology with {count} joints")
