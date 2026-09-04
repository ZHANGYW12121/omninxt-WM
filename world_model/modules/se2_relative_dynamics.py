"""Analytic Ego/Human relative-coordinate dynamics.

The module keeps its historical filename because checkpoints and callers
import it directly.  Human geometry is nevertheless full SO(3): compact-v3
stores skeletons and velocities in ``base_link`` while Ego14 stores roll,
pitch and relative yaw.  Transforming only yaw makes a banking vehicle invent
vertical/lateral Human motion, so every Human rollout now uses the complete
``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` rotation.
"""

from __future__ import annotations

import math

import torch
import numpy as np


ACTION_SCALE = (2.15, 2.15, 1.0, 0.8)
DEPLOYABLE_DRONE_COLLISION_RADIUS_M = 0.17

# Deployable COCO12 body observations are converted to the ten collision
# spheres used by the Isaac task.  The four limb pairs are direct keypoints;
# pelvis and head are deterministic virtual joints derived only from visible
# body keypoints.  This closes the old topology mismatch where Actor/Event
# clearance ignored the simulator's largest pelvis/head collision spheres.
COCO12_COLLISION_SPHERE_RADII_M = (
    0.14, 0.08, 0.08, 0.09, 0.09,
    0.09, 0.09, 0.08, 0.08, 0.12,
)


def _sqrt_nonnegative_with_tangent_subgradient(
    value: torch.Tensor,
    *,
    cancellation_scale: torch.Tensor,
) -> torch.Tensor:
    """Exact non-negative square root with a finite tangency subgradient.

    Swept sphere/point intersections use a quadratic discriminant.  At an
    exact tangency the forward root is well defined, but ``sqrt`` has an
    infinite derivative at zero.  A later inactive ``where`` branch can then
    turn that derivative into ``0 * inf = NaN``.  Near tangency the contact
    topology is itself non-differentiable, so use a zero subgradient inside a
    round-off-sized band while retaining the exact square-root value in the
    forward pass.
    """
    if value.shape != cancellation_scale.shape:
        raise ValueError("discriminant and cancellation scale must match")
    nonnegative = value.clamp_min(0.0)
    scale = cancellation_scale.detach().abs().clamp_min(1.0)
    tangent_tolerance = (
        32.0 * torch.finfo(nonnegative.dtype).eps * scale)
    regular = nonnegative > tangent_tolerance
    # Never evaluate sqrt at zero on the differentiable branch.  ``where``
    # alone is insufficient because autograd still visits both branch graphs.
    differentiable_input = torch.where(
        regular, nonnegative, torch.ones_like(nonnegative))
    differentiable_root = torch.where(
        regular,
        differentiable_input.sqrt(),
        torch.zeros_like(nonnegative),
    )
    exact_root = nonnegative.sqrt().detach()
    return exact_root + differentiable_root - differentiable_root.detach()


def coco12_collision_spheres(
    joints_body: torch.Tensor,
    joint_mask: torch.Tensor,
    joint_velocity_body: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map deployable COCO12 joints to Isaac's ten collision spheres.

    Output order is ``pelvis, right/left hand, right/left foot, right/left
    knee, right/left elbow, head``.  Head is estimated from the shoulder and
    pelvis centres as ``shoulder + .45 * (shoulder - pelvis)``.  A virtual
    sphere is valid only when every source joint needed to construct it is
    observed; missing detector joints are never invented during imagination.
    """
    if joints_body.shape[-2:] != (12, 3):
        raise ValueError("COCO12 collision conversion requires [...,12,3]")
    if joint_mask.shape != joints_body.shape[:-1]:
        raise ValueError("COCO12 joint mask must match joints without xyz")
    if joint_velocity_body is None:
        joint_velocity_body = torch.zeros_like(joints_body)
    if joint_velocity_body.shape != joints_body.shape:
        raise ValueError("COCO12 joint velocities must match joint positions")

    joints = joints_body.float()
    velocity = joint_velocity_body.float()
    valid = joint_mask.bool()
    pelvis = 0.5 * (joints[..., 6, :] + joints[..., 7, :])
    pelvis_velocity = 0.5 * (
        velocity[..., 6, :] + velocity[..., 7, :])
    pelvis_valid = valid[..., 6] & valid[..., 7]
    shoulder = 0.5 * (joints[..., 0, :] + joints[..., 1, :])
    shoulder_velocity = 0.5 * (
        velocity[..., 0, :] + velocity[..., 1, :])
    shoulder_valid = valid[..., 0] & valid[..., 1]
    head = shoulder + 0.45 * (shoulder - pelvis)
    head_velocity = (
        shoulder_velocity
        + 0.45 * (shoulder_velocity - pelvis_velocity))
    head_valid = shoulder_valid & pelvis_valid

    direct = (5, 4, 11, 10, 9, 8, 3, 2)
    sphere_joints = torch.stack((
        pelvis, *(joints[..., index, :] for index in direct), head,
    ), dim=-2)
    sphere_velocity = torch.stack((
        pelvis_velocity,
        *(velocity[..., index, :] for index in direct),
        head_velocity,
    ), dim=-2)
    sphere_mask = torch.stack((
        pelvis_valid, *(valid[..., index] for index in direct), head_valid,
    ), dim=-1)
    radii = joints.new_tensor(COCO12_COLLISION_SPHERE_RADII_M)
    sphere_joints = sphere_joints.masked_fill(~sphere_mask[..., None], 0.0)
    sphere_velocity = sphere_velocity.masked_fill(
        ~sphere_mask[..., None], 0.0)
    return sphere_joints, sphere_velocity, sphere_mask, radii


def persistent_rollout_joint_mask(
    human_mask: torch.Tensor,
    source_joint_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep only source-observed joints active throughout imagination.

    The Human prediction head emits a coordinate for every joint, but it does
    not predict whether a previously missing joint has become observable.
    Until such a lifecycle head exists, activating all joints from a
    person-level mask would turn unsupervised coordinates into collision
    geometry.  The source joint mask is therefore the complete deployable
    validity state for an imagined rollout.
    """
    if source_joint_mask.ndim != human_mask.ndim + 1:
        raise ValueError(
            "source_joint_mask must add one joint dimension to human_mask")
    if source_joint_mask.shape[:-1] != human_mask.shape:
        raise ValueError(
            "source_joint_mask Human dimensions must match human_mask")
    return source_joint_mask.bool() & human_mask.bool().unsqueeze(-1)


def swept_relative_point_signed_gap(
    relative_start: torch.Tensor,
    relative_end: torch.Tensor,
    valid: torch.Tensor,
    *,
    surface_radii_m: torch.Tensor | float,
    fallback_gap_m: float = 6.0,
    per_human: bool = False,
) -> torch.Tensor:
    """Minimum signed gap along one linearly swept relative transition.

    The two endpoints are Human-joint positions relative to the UAV centre in
    one common frame.  Minimising ``||r0 + alpha * (r1-r0)||`` for
    ``alpha in [0,1]`` detects contacts between 10 Hz recorder samples.  A
    joint is usable only when it exists at *both* endpoints.
    """
    if relative_start.shape != relative_end.shape or relative_start.shape[-1] != 3:
        raise ValueError("swept relative endpoints must share [...,3] shape")
    if valid.shape != relative_start.shape[:-1]:
        raise ValueError("swept valid mask must match endpoint rows")
    delta = relative_end.float() - relative_start.float()
    denominator = delta.square().sum(-1)
    alpha = -(
        relative_start.float() * delta
    ).sum(-1) / denominator.clamp_min(1.0e-12)
    alpha = torch.where(
        denominator > 1.0e-12,
        alpha.clamp(0.0, 1.0),
        torch.zeros_like(alpha),
    )
    closest = relative_start.float() + alpha[..., None] * delta
    radii = torch.as_tensor(
        surface_radii_m, dtype=closest.dtype, device=closest.device)
    signed = torch.linalg.vector_norm(closest, dim=-1) - radii
    signed = signed.masked_fill(~valid.bool(), torch.inf)
    # Convention: the penultimate dimension is joints and the one before it
    # is Human slots, matching compact-v3 [..., people, joints, xyz].
    slot_gap = signed.amin(-1)
    slot_valid = valid.bool().any(-1)
    slot_gap = torch.where(
        slot_valid,
        slot_gap,
        slot_gap.new_full(slot_gap.shape, float(fallback_gap_m)),
    )
    if per_human:
        return slot_gap
    return torch.where(
        slot_valid.any(-1),
        slot_gap.amin(-1),
        slot_gap.new_full(slot_gap.shape[:-1], float(fallback_gap_m)),
    )


def swept_relative_point_first_contact_fraction(
    relative_start: torch.Tensor,
    relative_end: torch.Tensor,
    valid: torch.Tensor,
    *,
    surface_radii_m: torch.Tensor | float,
    per_human: bool = False,
) -> torch.Tensor:
    """Earliest synchronous contact fraction in ``[0,1]`` or ``inf``.

    Unlike closest-distance time, this solves the first quadratic root of
    ``||r0 + alpha (r1-r0)|| = radius``.  Human and UAV therefore always use
    the same ``alpha``; intersecting spatial paths at different times cannot
    produce a contact.
    """
    if relative_start.shape != relative_end.shape or relative_start.shape[-1] != 3:
        raise ValueError("swept relative endpoints must share [...,3] shape")
    if valid.shape != relative_start.shape[:-1]:
        raise ValueError("swept valid mask must match endpoint rows")
    start = relative_start.float()
    delta = relative_end.float() - start
    radii = torch.as_tensor(
        surface_radii_m, dtype=start.dtype, device=start.device)
    quadratic = delta.square().sum(-1)
    linear = 2.0 * (start * delta).sum(-1)
    constant = start.square().sum(-1) - radii.square()
    quadratic_constant = 4.0 * quadratic * constant
    discriminant = linear.square() - quadratic_constant
    sqrt_discriminant = _sqrt_nonnegative_with_tangent_subgradient(
        discriminant,
        cancellation_scale=linear.square() + quadratic_constant.abs(),
    )
    moving = quadratic > 1.0e-12
    root = (
        -linear - sqrt_discriminant
    ) / (2.0 * quadratic.clamp_min(1.0e-12))
    already_inside = constant <= 0.0
    intersects = (
        valid.bool()
        & (already_inside | (
            moving & (discriminant >= 0.0) & (root >= 0.0) & (root <= 1.0)
        ))
    )
    fraction = torch.where(
        already_inside & valid.bool(), torch.zeros_like(root), root)
    fraction = fraction.masked_fill(~intersects, torch.inf)
    slot_fraction = fraction.amin(-1)
    if per_human:
        return slot_fraction
    return slot_fraction.amin(-1)


def swept_relative_point_signed_gap_numpy(
    relative_start: np.ndarray,
    relative_end: np.ndarray,
    valid: np.ndarray,
    *,
    surface_radii_m: np.ndarray | float,
    fallback_gap_m: float = 6.0,
) -> np.ndarray:
    """NumPy twin of :func:`swept_relative_point_signed_gap` for audits."""
    start = np.asarray(relative_start, np.float64)
    end = np.asarray(relative_end, np.float64)
    valid = np.asarray(valid, np.bool_)
    if start.shape != end.shape or start.shape[-1] != 3:
        raise ValueError("swept relative endpoints must share [...,3] shape")
    if valid.shape != start.shape[:-1]:
        raise ValueError("swept valid mask must match endpoint rows")
    delta = end - start
    denominator = np.sum(delta * delta, axis=-1)
    alpha = np.zeros_like(denominator)
    moving = denominator > 1.0e-12
    alpha[moving] = np.clip(
        -np.sum(start[moving] * delta[moving], axis=-1)
        / denominator[moving], 0.0, 1.0)
    closest = start + alpha[..., None] * delta
    signed = (
        np.linalg.norm(closest, axis=-1)
        - np.asarray(surface_radii_m, np.float64))
    signed[~valid] = np.inf
    result = np.min(signed.reshape(signed.shape[0], -1), axis=-1)
    return np.where(np.isfinite(result), result, float(fallback_gap_m))


def swept_segment_aabb_first_contact_fraction_2d(
    segment_start_xy: torch.Tensor,
    segment_end_xy: torch.Tensor,
    aabbs_xyxy: torch.Tensor,
    aabb_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Earliest AABB entry fraction in ``[0,1]`` or ``inf``.

    AABBs are ``[xmin, ymin, xmax, ymax]`` and are expected to already include
    the recorder's physical/conservative inflation.  This routine uses the
    exact slab interval, not endpoint containment or temporal subsampling.
    """
    if segment_start_xy.shape != segment_end_xy.shape \
            or segment_start_xy.shape[-1] != 2:
        raise ValueError("segment endpoints must share [...,2] shape")
    if aabbs_xyxy.shape[-1] != 4:
        raise ValueError("AABBs must end in xmin,ymin,xmax,ymax")
    start = segment_start_xy.float().unsqueeze(-2)
    delta = (segment_end_xy.float() - segment_start_xy.float()).unsqueeze(-2)
    minimum = aabbs_xyxy.float()[..., :2]
    maximum = aabbs_xyxy.float()[..., 2:]
    parallel = delta.abs() <= 1.0e-12
    inside_parallel = (start >= minimum) & (start <= maximum)
    reciprocal_input = torch.where(
        parallel, torch.ones_like(delta), delta)
    reciprocal = torch.where(
        parallel, torch.zeros_like(delta), reciprocal_input.reciprocal())
    first = (minimum - start) * reciprocal
    second = (maximum - start) * reciprocal
    enter_axis = torch.where(parallel, torch.full_like(first, -torch.inf),
                             torch.minimum(first, second))
    exit_axis = torch.where(parallel, torch.full_like(first, torch.inf),
                            torch.maximum(first, second))
    axis_possible = (~parallel) | inside_parallel
    enter = enter_axis.amax(-1).clamp_min(0.0)
    exit = exit_axis.amin(-1).clamp_max(1.0)
    intersects = axis_possible.all(-1) & (enter <= exit)
    if aabb_valid is not None:
        if aabb_valid.shape != intersects.shape:
            raise ValueError("AABB mask must match segment/AABB broadcast")
        intersects &= aabb_valid.bool()
    fraction = enter.masked_fill(~intersects, torch.inf)
    return fraction.amin(-1)


def swept_segment_intersects_aabb_2d(
    segment_start_xy: torch.Tensor,
    segment_end_xy: torch.Tensor,
    aabbs_xyxy: torch.Tensor,
    aabb_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility boolean wrapper around exact swept AABB entry time."""
    return torch.isfinite(swept_segment_aabb_first_contact_fraction_2d(
        segment_start_xy, segment_end_xy, aabbs_xyxy, aabb_valid))


def swept_segment_exit_aabb_fraction_2d(
    segment_start_xy: torch.Tensor,
    segment_end_xy: torch.Tensor,
    bounds_xyxy: torch.Tensor,
) -> torch.Tensor:
    """Earliest fraction at which a segment exits an allowed XY rectangle."""
    if segment_start_xy.shape != segment_end_xy.shape \
            or segment_start_xy.shape[-1] != 2:
        raise ValueError("segment endpoints must share [...,2] shape")
    if bounds_xyxy.shape[-1] != 4:
        raise ValueError("bounds must end in xmin,ymin,xmax,ymax")
    start = segment_start_xy.float()
    end = segment_end_xy.float()
    minimum, maximum = bounds_xyxy.float()[..., :2], bounds_xyxy.float()[..., 2:]
    start_inside = ((start >= minimum) & (start <= maximum)).all(-1)
    end_inside = ((end >= minimum) & (end <= maximum)).all(-1)
    delta = end - start
    fractions = []
    for axis in range(2):
        positive = delta[..., axis] > 1.0e-12
        negative = delta[..., axis] < -1.0e-12
        upper = (
            maximum[..., axis] - start[..., axis]
        ) / delta[..., axis].clamp_min(1.0e-12)
        lower = (
            minimum[..., axis] - start[..., axis]
        ) / delta[..., axis].clamp_max(-1.0e-12)
        fractions.append(torch.where(
            positive, upper, torch.where(negative, lower, torch.inf)))
    exit_fraction = torch.stack(fractions, -1).amin(-1).clamp(0.0, 1.0)
    return torch.where(
        ~start_inside, torch.zeros_like(exit_fraction),
        torch.where(~end_inside, exit_fraction, torch.inf),
    )


def swept_scalar_exit_interval_fraction(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Earliest fraction at which a scalar segment exits ``[min,max]``.

    The result is zero when the source is already outside, finite in
    ``[0,1]`` when this transition crosses a boundary, and ``inf`` when the
    complete transition remains inside.  It is used for altitude competing
    terminals so their time is comparable with swept Human/static contacts.
    """
    start = segment_start.float()
    end = segment_end.float()
    lower = minimum.float()
    upper = maximum.float()
    start_inside = (start >= lower) & (start <= upper)
    end_inside = (end >= lower) & (end <= upper)
    delta = end - start
    positive = delta > 1.0e-12
    negative = delta < -1.0e-12
    # Never form an arithmetic expression involving an inactive infinite
    # endpoint.  ``torch.where`` masks values in the forward pass but its
    # backward still multiplies the unselected branch Jacobian by zero;
    # ``0 * inf`` poisoned action gradients for the one-sided altitude
    # interval ``(-inf, z_max]``.  Sanitising numerator and denominator before
    # division keeps both branches differentiable and gives motion toward an
    # unbounded side the intended no-exit result.
    upper_active = positive & torch.isfinite(upper)
    lower_active = negative & torch.isfinite(lower)
    upper_fraction = torch.where(
        upper_active, upper - start, torch.zeros_like(start)
    ) / torch.where(upper_active, delta, torch.ones_like(delta))
    lower_fraction = torch.where(
        lower_active, lower - start, torch.zeros_like(start)
    ) / torch.where(lower_active, delta, torch.ones_like(delta))
    fraction = torch.where(
        upper_active,
        upper_fraction,
        torch.where(lower_active, lower_fraction, torch.inf),
    ).clamp(0.0, 1.0)
    return torch.where(
        ~start_inside, torch.zeros_like(fraction),
        torch.where(~end_inside, fraction, torch.inf),
    )


def swept_scalar_enter_lower_bound_fraction(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    minimum: torch.Tensor,
) -> torch.Tensor:
    """Earliest fraction where a linear scalar satisfies ``value >= min``.

    The closed comparison matches the recorder's elapsed-time and watchdog
    timer thresholds.  A row that never enters the half-line returns ``inf``.
    """
    start = segment_start.float()
    end = segment_end.float()
    threshold = minimum.float()
    already_inside = start >= threshold
    increasing = end > start
    crosses = increasing & (end >= threshold)
    fraction = (threshold - start) / (end - start).clamp_min(1.0e-12)
    return torch.where(
        already_inside,
        torch.zeros_like(fraction),
        torch.where(crosses, fraction.clamp(0.0, 1.0), torch.inf),
    )


def swept_scalar_enter_upper_bound_fraction(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Earliest fraction where a linear scalar satisfies ``value <= max``."""
    start = segment_start.float()
    end = segment_end.float()
    threshold = maximum.float()
    already_inside = start <= threshold
    decreasing = end < start
    crosses = decreasing & (end <= threshold)
    fraction = (start - threshold) / (start - end).clamp_min(1.0e-12)
    return torch.where(
        already_inside,
        torch.zeros_like(fraction),
        torch.where(crosses, fraction.clamp(0.0, 1.0), torch.inf),
    )


def swept_vector_exit_ball_fraction(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    radius: torch.Tensor | float,
) -> torch.Tensor:
    """Earliest fraction where a linear vector leaves a closed Euclidean ball.

    The returned boundary fraction represents the infimum of the recorder's
    strict ``norm(value) > radius`` condition.  It is therefore suitable for
    ordering competing continuous events; callers decide exact-tie priority.
    """
    if segment_start.shape != segment_end.shape:
        raise ValueError("vector endpoints must have identical shape")
    if segment_start.ndim == 0:
        raise ValueError("vector endpoints require a trailing vector dimension")
    start = segment_start.float()
    end = segment_end.float()
    threshold = torch.as_tensor(
        radius, dtype=start.dtype, device=start.device)
    start_norm = torch.linalg.vector_norm(start, dim=-1)
    end_norm = torch.linalg.vector_norm(end, dim=-1)
    already_outside = start_norm > threshold
    exits = (start_norm <= threshold) & (end_norm > threshold)
    delta = end - start
    quadratic = delta.square().sum(-1)
    linear = 2.0 * (start * delta).sum(-1)
    constant = start.square().sum(-1) - threshold.square()
    quadratic_constant = 4.0 * quadratic * constant
    discriminant = linear.square() - quadratic_constant
    sqrt_discriminant = _sqrt_nonnegative_with_tangent_subgradient(
        discriminant,
        cancellation_scale=linear.square() + quadratic_constant.abs(),
    )
    # A segment beginning inside a convex ball can have only one forward exit;
    # the larger root is that exit even when the infinite line entered earlier.
    root = (
        -linear + sqrt_discriminant
    ) / (2.0 * quadratic.clamp_min(1.0e-12))
    return torch.where(
        already_outside,
        torch.zeros_like(root),
        torch.where(
            exits & (quadratic > 1.0e-12) & (discriminant >= 0.0),
            root.clamp(0.0, 1.0),
            torch.inf,
        ),
    )


def ego_yaw(ego_state: torch.Tensor) -> torch.Tensor:
    if ego_state.shape[-1] != 14:
        raise ValueError("relative dynamics requires Ego14")
    return torch.atan2(ego_state[..., 12], ego_state[..., 13])


def ego_rotation_matrix(ego_state: torch.Tensor) -> torch.Tensor:
    """Return ``R_episode_from_body`` from Ego14 roll/pitch/relative-yaw."""
    if ego_state.shape[-1] != 14:
        raise ValueError("SO(3) dynamics requires Ego14")
    roll, pitch, yaw = (
        ego_state[..., 10], ego_state[..., 11], ego_yaw(ego_state))
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    # Expanded product Rz @ Ry @ Rx.  Constructing it explicitly avoids a
    # large chain of tiny batched matrices in 15-step imagined rollouts.
    return torch.stack((
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    ), dim=-1).reshape(*ego_state.shape[:-1], 3, 3)


def _expand_rotation_for_vectors(
    rotation: torch.Tensor, vectors: torch.Tensor,
) -> torch.Tensor:
    if vectors.shape[-1] != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation/vector shapes must end in [3,3] and [3]")
    while rotation.ndim < vectors.ndim + 1:
        rotation = rotation.unsqueeze(-3)
    return rotation


def rotate_body_to_episode(
    vectors_body: torch.Tensor, ego_state: torch.Tensor,
) -> torch.Tensor:
    """Rotate body-frame vectors into the episode-start frame."""
    rotation = _expand_rotation_for_vectors(
        ego_rotation_matrix(ego_state), vectors_body)
    return torch.matmul(
        vectors_body.unsqueeze(-2), rotation.transpose(-1, -2)
    ).squeeze(-2)


def rotate_episode_to_body(
    vectors_episode: torch.Tensor, ego_state: torch.Tensor,
) -> torch.Tensor:
    """Rotate episode-frame vectors into the current body frame."""
    rotation = _expand_rotation_for_vectors(
        ego_rotation_matrix(ego_state), vectors_episode)
    return torch.matmul(
        vectors_episode.unsqueeze(-2), rotation
    ).squeeze(-2)


def body_points_to_episode(
    points_body: torch.Tensor, ego_state: torch.Tensor,
) -> torch.Tensor:
    """Transform body-frame points into the episode-start frame."""
    position = ego_state[..., :3]
    while position.ndim < points_body.ndim:
        position = position.unsqueeze(-2)
    return rotate_body_to_episode(points_body, ego_state) + position


def interpolate_ego_state(
    source_ego_state: torch.Tensor,
    destination_ego_state: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """Interpolate an Ego14 state at a within-transition event time.

    Position, AGL, roll and pitch are linear over the structured transition.
    Horizontal velocity/acceleration are first put in the fixed episode frame,
    interpolated there, and only then expressed in the event yaw-heading
    frame.  Directly interpolating Ego14 indices 3:8 would mix two rotating
    coordinate systems and recreate the yaw propagation bug fixed in
    :func:`analytic_ego_step`.
    """
    if source_ego_state.shape != destination_ego_state.shape \
            or source_ego_state.shape[-1] != 14:
        raise ValueError("Ego interpolation requires paired Ego14 states")
    if fraction.shape != source_ego_state.shape[:-1] + (1,):
        raise ValueError("Ego interpolation fraction must end in one field")
    alpha = fraction.to(source_ego_state).clamp(0.0, 1.0)
    result = source_ego_state + alpha * (
        destination_ego_state - source_ego_state)
    source_yaw = ego_yaw(source_ego_state)
    destination_yaw = ego_yaw(destination_ego_state)
    yaw_delta = torch.atan2(
        torch.sin(destination_yaw - source_yaw),
        torch.cos(destination_yaw - source_yaw),
    )
    event_yaw = source_yaw + alpha.squeeze(-1) * yaw_delta
    result[..., 12] = torch.sin(event_yaw)
    result[..., 13] = torch.cos(event_yaw)
    for start, stop in ((3, 5), (6, 8)):
        source_episode = rotate_xy(
            source_ego_state[..., start:stop], source_yaw)
        destination_episode = rotate_xy(
            destination_ego_state[..., start:stop], destination_yaw)
        event_episode = source_episode + alpha * (
            destination_episode - source_episode)
        result[..., start:stop] = rotate_xy(event_episode, -event_yaw)
    return result


def interpolate_body_kinematics(
    source_points_body: torch.Tensor,
    destination_points_body: torch.Tensor,
    source_velocity_body: torch.Tensor,
    destination_velocity_body: torch.Tensor,
    source_ego_state: torch.Tensor,
    destination_ego_state: torch.Tensor,
    event_ego_state: torch.Tensor,
    fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate body-frame geometry through one common episode frame."""
    if source_points_body.shape != destination_points_body.shape \
            or source_velocity_body.shape != source_points_body.shape \
            or destination_velocity_body.shape != source_points_body.shape \
            or source_points_body.shape[-1] != 3:
        raise ValueError("interpolated body kinematics must share [...,3]")
    if source_ego_state.shape != destination_ego_state.shape \
            or event_ego_state.shape != source_ego_state.shape:
        raise ValueError("interpolated body kinematics require paired Ego14")
    if fraction.shape != source_ego_state.shape[:-1] + (1,):
        raise ValueError("kinematic interpolation fraction has wrong shape")
    alpha = fraction.to(source_points_body)
    while alpha.ndim < source_points_body.ndim:
        alpha = alpha.unsqueeze(-2)
    source_episode = body_points_to_episode(
        source_points_body, source_ego_state)
    destination_episode = body_points_to_episode(
        destination_points_body, destination_ego_state)
    event_episode = source_episode + alpha * (
        destination_episode - source_episode)
    event_position = event_ego_state[..., :3]
    while event_position.ndim < event_episode.ndim:
        event_position = event_position.unsqueeze(-2)
    event_points_body = rotate_episode_to_body(
        event_episode - event_position, event_ego_state)
    source_velocity_episode = rotate_body_to_episode(
        source_velocity_body, source_ego_state)
    destination_velocity_episode = rotate_body_to_episode(
        destination_velocity_body, destination_ego_state)
    event_velocity_episode = source_velocity_episode + alpha * (
        destination_velocity_episode - source_velocity_episode)
    event_velocity_body = rotate_episode_to_body(
        event_velocity_episode, event_ego_state)
    return event_points_body, event_velocity_body


def analytic_joint_surface_gap(
    joints_body: torch.Tensor,
    joint_mask: torch.Tensor,
    *,
    surface_radius_m: float,
    maximum_gap_m: float = 6.0,
    human_presence: torch.Tensor | None = None,
    per_human: bool = False,
) -> torch.Tensor:
    """Return non-negative UAV-to-Human free gap from predicted joints.

    ``surface_radius_m`` is the shared UAV-plus-joint envelope used by the
    privileged recorder target.  A soft presence does not create a hard
    threshold: it continuously moves a vanishing track toward the configured
    no-Human fallback distance.
    """
    if joints_body.shape[-1] != 3:
        raise ValueError("joints_body must end in xyz")
    if joint_mask.shape != joints_body.shape[:-1]:
        raise ValueError("joint_mask must match joints_body without xyz")
    if surface_radius_m < 0.0 or maximum_gap_m <= 0.0:
        raise ValueError("clearance radii must be non-negative/positive")
    valid = joint_mask.bool()
    distance = torch.linalg.vector_norm(joints_body.float(), dim=-1)
    slot_gap = distance.masked_fill(~valid, torch.inf).amin(-1)
    slot_valid = valid.any(-1)
    slot_gap = (slot_gap - float(surface_radius_m)).clamp(
        min=0.0, max=float(maximum_gap_m))
    slot_gap = torch.where(
        slot_valid, slot_gap,
        slot_gap.new_full(slot_gap.shape, float(maximum_gap_m)))
    if human_presence is not None:
        if human_presence.shape != slot_gap.shape:
            raise ValueError("human_presence must match Human slots")
        presence = human_presence.to(slot_gap).clamp(0.0, 1.0)
        slot_gap = (
            presence * slot_gap
            + (1.0 - presence) * float(maximum_gap_m))
    if per_human:
        return slot_gap
    any_human = slot_valid.any(-1)
    minimum = slot_gap.amin(-1)
    return torch.where(
        any_human, minimum,
        minimum.new_full(minimum.shape, float(maximum_gap_m)))


def analytic_joint_signed_surface_gap(
    joints_body: torch.Tensor,
    joint_mask: torch.Tensor,
    *,
    drone_radius_m: float,
    joint_radii_m: torch.Tensor | float,
    fallback_gap_m: float = 6.0,
    per_human: bool = False,
) -> torch.Tensor:
    """Return signed physical surface gap; negative values mean overlap."""
    if joints_body.shape[-1] != 3:
        raise ValueError("joints_body must end in xyz")
    if joint_mask.shape != joints_body.shape[:-1]:
        raise ValueError("joint_mask must match joints_body without xyz")
    radii = torch.as_tensor(
        joint_radii_m, dtype=joints_body.dtype, device=joints_body.device)
    if radii.ndim == 0:
        radii = radii.expand(joints_body.shape[-2])
    if radii.shape != (joints_body.shape[-2],):
        raise ValueError("joint_radii_m must be scalar or match joint count")
    valid = joint_mask.bool()
    signed = (
        torch.linalg.vector_norm(joints_body.float(), dim=-1)
        - float(drone_radius_m) - radii
    ).masked_fill(~valid, torch.inf)
    slot_gap = signed.amin(-1)
    slot_valid = valid.any(-1)
    slot_gap = torch.where(
        slot_valid, slot_gap,
        slot_gap.new_full(slot_gap.shape, float(fallback_gap_m)))
    if per_human:
        return slot_gap
    return torch.where(
        slot_valid.any(-1), slot_gap.amin(-1),
        slot_gap.new_full(slot_gap.shape[:-1], float(fallback_gap_m)))


def rotate_xy(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate the final XY coordinates by ``angle`` with broadcast support."""
    while angle.ndim < vector.ndim - 1:
        angle = angle.unsqueeze(-1)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack((cosine * x - sine * y,
                        sine * x + cosine * y), dim=-1)


def analytic_relative_human_step(
    root_position_body: torch.Tensor,
    human_velocity_body: torch.Tensor,
    ego_state: torch.Tensor,
    next_ego_state: torch.Tensor,
    *,
    dt_s: float | torch.Tensor = 0.1,
    position_residual_next_body: torch.Tensor | None = None,
    velocity_residual_current_body: torch.Tensor | None = None,
    maximum_velocity_mps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a human root into the *future* Ego body frame.

    Human velocity from compact-v3 is an Ego-motion-compensated estimate of
    Human world velocity, expressed in the current body axes. It is integrated
    once in that frame,
    lifted with the current full attitude, translated in the episode frame,
    and expressed with the next full attitude. Learned residuals are explicitly
    defined in their named frames to prevent double rotation.
    """
    if root_position_body.shape[-1] != 3:
        raise ValueError("root_position_body must end in xyz")
    if human_velocity_body.shape != root_position_body.shape:
        raise ValueError("human velocity shape must match root position")
    if ego_state.shape[:-1] != root_position_body.shape[:-2] or (
        next_ego_state.shape != ego_state.shape
    ):
        raise ValueError("Ego batch shape must match human root batch shape")
    dt = torch.as_tensor(
        dt_s, dtype=root_position_body.dtype,
        device=root_position_body.device)
    if not torch.isfinite(dt).all() or bool((dt <= 0.0).any()):
        raise ValueError("dt_s must be finite and positive")
    while dt.ndim < root_position_body.ndim:
        dt = dt.unsqueeze(-1)
    try:
        dt = torch.broadcast_to(dt, root_position_body.shape)
    except RuntimeError as error:
        raise ValueError(
            "dt_s cannot broadcast to Human kinematics") from error

    velocity_current = human_velocity_body
    if velocity_residual_current_body is not None:
        velocity_current = velocity_current + velocity_residual_current_body
    if maximum_velocity_mps is not None:
        maximum_velocity = float(maximum_velocity_mps)
        if not math.isfinite(maximum_velocity) or maximum_velocity <= 0.0:
            raise ValueError("maximum_velocity_mps must be finite and positive")
        # A per-step residual bound alone does not bound a recursive rollout:
        # the Actor could otherwise learn to accumulate the same correction
        # until imagined pedestrians leave the scene at unsupported speeds.
        # Project the complete velocity vector, not each coordinate, onto the
        # simulator-supported speed ball before both integration and rotation.
        speed = torch.linalg.vector_norm(
            velocity_current, dim=-1, keepdim=True)
        scale = torch.clamp(
            velocity_current.new_tensor(maximum_velocity)
            / speed.clamp_min(1.0e-8),
            max=1.0,
        )
        velocity_current = velocity_current * scale
    future_point_current_body = (
        root_position_body + velocity_current * dt)
    future_point_episode = body_points_to_episode(
        future_point_current_body, ego_state)
    next_position_episode_relative = (
        future_point_episode - next_ego_state[..., :3].unsqueeze(-2))
    next_position = rotate_episode_to_body(
        next_position_episode_relative, next_ego_state)
    velocity_episode = rotate_body_to_episode(velocity_current, ego_state)
    next_velocity = rotate_episode_to_body(
        velocity_episode, next_ego_state)
    if position_residual_next_body is not None:
        next_position = next_position + position_residual_next_body
    return next_position, next_velocity


def ego_velocity_full_body(ego_state: torch.Tensor) -> torch.Tensor:
    """Express Ego14 world velocity in the same full body frame as Humans.

    Ego14 stores horizontal velocity in the yaw-heading frame and vertical
    velocity in the episode/world axis, while skeleton tracker velocity is in
    the full roll/pitch/yaw ``base_link`` frame.  Lift Ego velocity to the
    episode frame and rotate it back with the full attitude before forming a
    relative Human velocity for TTC.
    """
    if ego_state.shape[-1] != 14:
        raise ValueError("Ego velocity conversion requires Ego14")
    yaw = ego_yaw(ego_state)
    episode_xy = rotate_xy(ego_state[..., 3:5], yaw)
    episode_velocity = torch.cat((
        episode_xy, ego_state[..., 5:6]), -1)
    return rotate_episode_to_body(episode_velocity, ego_state)


def analytic_ego_step(
    ego_state: torch.Tensor,
    applied_action: torch.Tensor,
    *,
    dt_s: float | torch.Tensor = 0.1,
    velocity_response: float | tuple[float, float, float] = 1.0,
    attitude_coefficients: torch.Tensor | tuple[tuple[float, float], ...]
    | None = None,
) -> torch.Tensor:
    """Integrate one applied velocity command into Ego14.

    ``velocity_response`` is the measured discrete closed-loop translational
    response ``v_next = v + alpha * (v_command - v)`` is evaluated in the
    *source* yaw-heading frame, where the body-frame command is issued.  Ego14
    velocity at the destination must then be re-expressed in the destination
    yaw-heading frame.  Omitting that last rotation silently changes the
    physical world velocity whenever yaw changes and compounds position, TTC
    and smoothness errors over a 15-step rollout. Position uses the resulting
    episode-frame interval velocity. Yaw-rate remains an immediately applied
    command (the replay identifies that response as effectively one). The
    whole update is analytic and differentiable with respect to the action.
    """
    if ego_state.shape[-1] != 14 or applied_action.shape[-1] != 4:
        raise ValueError("analytic Ego dynamics requires Ego14 and applied4")
    if ego_state.shape[:-1] != applied_action.shape[:-1]:
        raise ValueError("Ego and applied action batch shapes must match")
    response = torch.as_tensor(
        velocity_response, dtype=applied_action.dtype,
        device=applied_action.device).reshape(-1)
    if response.numel() == 1:
        response = response.expand(3)
    if response.numel() != 3:
        raise ValueError("velocity_response must be scalar or contain xyz")
    if not torch.isfinite(response).all() or bool(
        ((response <= 0.0) | (response > 1.0)).any()
    ):
        raise ValueError("velocity_response entries must lie in (0,1]")
    dt = torch.as_tensor(
        dt_s, dtype=applied_action.dtype, device=applied_action.device)
    if dt.shape == ego_state.shape[:-1] + (1,):
        dt = dt.squeeze(-1)
    try:
        dt = torch.broadcast_to(dt, ego_state.shape[:-1])
    except RuntimeError as error:
        raise ValueError("dt_s cannot broadcast to Ego dynamics") from error
    if not torch.isfinite(dt).all() or bool((dt <= 0.0).any()):
        raise ValueError("analytic Ego dt_s must be finite and positive")
    # The configured response is identified for the nominal 0.1 s controller
    # interval. Preserve the same continuous first-order response on shorter
    # force-written terminal rows instead of applying a full-tick velocity
    # jump in as little as 16 ms.
    response_fraction = dt[..., None] / 0.1
    effective_response = 1.0 - torch.pow(
        (1.0 - response).clamp_min(0.0), response_fraction)
    scale = applied_action.new_tensor(ACTION_SCALE)
    command = applied_action * scale
    yaw = ego_yaw(ego_state)
    previous_velocity = ego_state[..., 3:6]
    response_velocity_source = previous_velocity + effective_response * (
        command[..., :3] - previous_velocity)
    previous_velocity_episode_xy = rotate_xy(
        previous_velocity[..., :2], yaw)
    velocity_episode_xy = rotate_xy(
        response_velocity_source[..., :2], yaw)
    next_yaw = yaw + command[..., 3] * dt
    next_velocity_heading_xy = rotate_xy(velocity_episode_xy, -next_yaw)
    next_velocity = torch.cat((
        next_velocity_heading_xy,
        response_velocity_source[..., 2:3],
    ), -1)
    result = ego_state.clone()
    result[..., :2] = (
        ego_state[..., :2] + velocity_episode_xy * dt[..., None])
    result[..., 2] = (
        ego_state[..., 2]
        + response_velocity_source[..., 2] * dt)
    result[..., 3:6] = next_velocity
    acceleration_episode_xy = (
        velocity_episode_xy - previous_velocity_episode_xy) / dt[..., None]
    result[..., 6:8] = rotate_xy(
        acceleration_episode_xy, -next_yaw)
    result[..., 8] = (
        response_velocity_source[..., 2]
        - previous_velocity[..., 2]) / dt
    result[..., 9] = (
        ego_state[..., 9]
        + response_velocity_source[..., 2] * dt).clamp_min(0.0)
    if attitude_coefficients is not None:
        coefficients = torch.as_tensor(
            attitude_coefficients,
            dtype=applied_action.dtype,
            device=applied_action.device,
        )
        if coefficients.shape != (7, 2) or not torch.isfinite(
            coefficients
        ).all():
            raise ValueError(
                "attitude_coefficients must be a finite [7,2] matrix")
        # Held-out replay ARX model:
        # [1, roll, pitch, applied_forward, applied_left,
        #  source_forward_velocity, source_left_velocity] @ C.
        # Actions are normalized because those are the values stored in replay
        # and passed through Dreamer.  Short force-written terminal rows receive
        # only their actual fraction of a nominal 100 ms attitude response.
        attitude_design = torch.cat((
            torch.ones_like(ego_state[..., :1]),
            ego_state[..., 10:12],
            applied_action[..., :2],
            ego_state[..., 3:5],
        ), -1)
        nominal_attitude = attitude_design @ coefficients
        attitude_fraction = (dt / 0.1).clamp(0.0, 1.0)[..., None]
        result[..., 10:12] = (
            ego_state[..., 10:12]
            + attitude_fraction * (
                nominal_attitude - ego_state[..., 10:12])
        )
    result[..., 12] = torch.sin(next_yaw)
    result[..., 13] = torch.cos(next_yaw)
    return result
