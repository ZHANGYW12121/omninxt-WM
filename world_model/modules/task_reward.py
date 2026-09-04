"""Task-reward algebra shared by replay supervision and imagination.

Compact-v3 stores transition rewards on the destination row.  The helpers in
this module preserve that convention: index zero is a context row and the
progress caused by action ``t`` is written at index ``t + 1``.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


EVENT_COMPONENT_INDEX = 0
PROGRESS_COMPONENT_INDEX = 1


def continuous_residual_target(
    reward: torch.Tensor,
    reward_components: torch.Tensor,
) -> torch.Tensor:
    """Remove the already weighted progress and event components.

    ``reward_components[..., progress]`` already contains the configured
    metres-to-reward weight.  Applying that weight again would silently change
    the task objective.
    """
    if reward.shape[:-1] != reward_components.shape[:-1]:
        raise ValueError("reward and reward_components leading shapes differ")
    if (
        reward.shape[-1] != 1
        or reward_components.shape[-1] <= PROGRESS_COMPONENT_INDEX
    ):
        raise ValueError("expected reward [...,1] and at least two components")
    event = reward_components[..., EVENT_COMPONENT_INDEX:EVENT_COMPONENT_INDEX + 1]
    progress = reward_components[
        ..., PROGRESS_COMPONENT_INDEX:PROGRESS_COMPONENT_INDEX + 1]
    return reward.float() - event.float() - progress.float()


def point_goal_distance(
    ego_state: torch.Tensor,
    goal_position: torch.Tensor,
) -> torch.Tensor:
    """Euclidean point-goal distance in the shared episode-local frame."""
    if ego_state.shape[-1] < 3 or goal_position.shape[-1] != 3:
        raise ValueError("expected Ego state and 3-D goal position")
    if ego_state.shape[:-1] != goal_position.shape[:-1]:
        raise ValueError("Ego and goal leading shapes differ")
    return torch.linalg.vector_norm(
        goal_position.to(ego_state) - ego_state[..., :3], dim=-1, keepdim=True)


def analytic_progress_reward(
    ego_states: torch.Tensor,
    goal_position: torch.Tensor,
    *,
    dt_s: float | torch.Tensor,
    progress_weight_per_m: float,
    max_progress_speed_mps: float,
) -> torch.Tensor:
    """Return destination-row-aligned analytic progress rewards.

    The output has the same leading time shape as ``ego_states``.  Element
    zero is exactly zero; elements ``1:`` describe transitions ``[:-1] ->
    [1:]``.  This matches Compact-v3 and Dreamer's ``reward[:, 1:]`` return
    convention.
    """
    if ego_states.ndim < 3 or ego_states.shape[-1] < 3:
        raise ValueError("ego_states must be [...,time,Ego]")
    if ego_states.shape[-2] < 2:
        raise ValueError("analytic progress requires at least two states")
    if goal_position.shape == ego_states.shape[:-2] + (3,):
        goal = goal_position[..., None, :].expand(*ego_states.shape[:-1], 3)
    elif goal_position.shape == ego_states.shape[:-1] + (3,):
        goal = goal_position
    else:
        raise ValueError("goal_position must be [...,3] or [...,time,3]")

    distance = point_goal_distance(ego_states, goal)
    raw_progress = distance[..., :-1, :] - distance[..., 1:, :]
    dt = torch.as_tensor(dt_s, dtype=ego_states.dtype, device=ego_states.device)
    if bool((dt <= 0).any()):
        raise ValueError("dt_s must be positive")
    max_progress = float(max_progress_speed_mps) * dt
    used_progress = torch.clamp(raw_progress, min=-max_progress, max=max_progress)
    progress = float(progress_weight_per_m) * used_progress
    return torch.cat((torch.zeros_like(progress[..., :1, :]), progress), dim=-2)


def analytic_fractional_progress_reward(
    source_ego_state: torch.Tensor,
    destination_ego_state: torch.Tensor,
    goal_position: torch.Tensor,
    transition_fraction: torch.Tensor,
    *,
    dt_s: float,
    progress_weight_per_m: float,
    max_progress_speed_mps: float,
) -> torch.Tensor:
    """Progress up to an event occurring part-way through each transition.

    Ego position is linear under the authoritative transition model.  The
    recorder's speed-based progress clip uses the actual partial ``dt``, so a
    terminal at fraction ``f`` clips by ``max_speed * dt * f`` rather than by
    the complete policy period.
    """
    if source_ego_state.shape != destination_ego_state.shape \
            or source_ego_state.shape[-1] < 3:
        raise ValueError("fractional progress requires paired Ego states")
    expected_fraction_shape = source_ego_state.shape[:-1] + (1,)
    if transition_fraction.shape != expected_fraction_shape:
        raise ValueError("transition fraction must match Ego transition rows")
    if goal_position.shape == source_ego_state.shape[:-2] + (3,):
        goal = goal_position[..., None, :].expand(
            *source_ego_state.shape[:-1], 3)
    elif goal_position.shape == source_ego_state.shape[:-1] + (3,):
        goal = goal_position
    else:
        raise ValueError("goal position must be [...,3] or [...,time,3]")
    if float(dt_s) <= 0.0 or float(max_progress_speed_mps) <= 0.0:
        raise ValueError("fractional progress dt/speed must be positive")
    fraction = transition_fraction.float().clamp(0.0, 1.0)
    event_position = (
        source_ego_state[..., :3]
        + fraction * (
            destination_ego_state[..., :3] - source_ego_state[..., :3])
    )
    source_distance = torch.linalg.vector_norm(
        goal.to(source_ego_state) - source_ego_state[..., :3],
        dim=-1, keepdim=True)
    event_distance = torch.linalg.vector_norm(
        goal.to(event_position) - event_position, dim=-1, keepdim=True)
    raw = source_distance - event_distance
    limit = float(max_progress_speed_mps) * float(dt_s) * fraction
    return float(progress_weight_per_m) * torch.maximum(
        torch.minimum(raw, limit), -limit)


def expected_event_reward(
    event_probability: torch.Tensor,
    event_rewards: torch.Tensor,
) -> torch.Tensor:
    """Convert a categorical transition distribution to expected reward."""
    if event_probability.shape[-1] != event_rewards.numel():
        raise ValueError("event reward count does not match event probabilities")
    return (
        event_probability.float()
        * event_rewards.to(event_probability).reshape(
            *((1,) * (event_probability.ndim - 1)), -1)
    ).sum(dim=-1, keepdim=True)


def rescale_interval_probability(
    nominal_probability: torch.Tensor,
    exposure_fraction: torch.Tensor,
) -> torch.Tensor:
    """Rescale an interval probability under a constant conditional hazard.

    ``nominal_probability`` is the probability for one model interval (0.1 s
    in the current task). ``exposure_fraction`` is the actual interval length
    divided by that nominal duration. This is the survival-consistent map

    ``p(f) = 1 - (1 - p(1)) ** f``.

    Compact-v3 force-writes terminal rows at the physics event time, so their
    exposure is often shorter than a policy interval. Treating those rows as
    equal-duration Bernoulli samples biases both factual calibration and the
    hazard accumulated by a fixed-step imagination.
    """
    probability = nominal_probability.float()
    fraction = exposure_fraction.float()
    if not torch.isfinite(probability).all() \
            or not torch.isfinite(fraction).all():
        raise ValueError("interval probability/exposure must be finite")
    if bool((probability < 0.0).any()) or bool((probability > 1.0).any()):
        raise ValueError("nominal interval probability must lie in [0,1]")
    if bool((fraction < 0.0).any()):
        raise ValueError("exposure fraction must be non-negative")
    probability, fraction = torch.broadcast_tensors(probability, fraction)
    integrated_hazard = -torch.log1p(
        -probability.clamp(max=1.0 - 1.0e-7))
    return (-torch.expm1(-integrated_hazard * fraction)).clamp(0.0, 1.0)


def human_event_time_quadrature(
    integrated_hazard: torch.Tensor,
    stop_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable event fractions and exact-mass quadrature weights.

    A constant within-step Human hazard has density ``H exp(-H u)`` before
    the deterministic stop time.  Nonlinear partial-transition rewards must
    be integrated against that density; evaluating them only at
    ``E[u | event]`` is generally biased.  Eight-point Gauss-Legendre nodes
    provide a deterministic differentiable integral, and normalizing their
    conditional weights to the analytic Human event mass keeps the competing
    distribution exactly closed even in finite precision.

    Inputs end in a singleton field.  Outputs insert an eight-node dimension
    immediately before that field and have shape ``[..., 8, 1]``.  The second
    output sums exactly to ``1-exp(-H*stop)`` along the node dimension.
    """
    if integrated_hazard.shape != stop_fraction.shape \
            or integrated_hazard.shape[-1] != 1:
        raise ValueError(
            "Human hazard/stop fraction must share a trailing singleton")
    if not torch.isfinite(integrated_hazard).all() \
            or not torch.isfinite(stop_fraction).all():
        raise ValueError("Human event-time quadrature inputs must be finite")
    if bool((integrated_hazard < 0.0).any()):
        raise ValueError("integrated Human hazard must be non-negative")
    if bool((stop_fraction < 0.0).any()) \
            or bool((stop_fraction > 1.0).any()):
        raise ValueError("Human stop fraction must lie in [0,1]")

    nodes = integrated_hazard.new_tensor((
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346424956498,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ))
    base_weights = integrated_hazard.new_tensor((
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ))
    view_shape = (1,) * (integrated_hazard.ndim - 1) + (8, 1)
    unit_nodes = 0.5 * (nodes + 1.0).reshape(view_shape)
    fractions = stop_fraction.unsqueeze(-2) * unit_nodes
    density_shape = base_weights.reshape(view_shape) * torch.exp(
        -integrated_hazard.unsqueeze(-2) * fractions)
    conditional_weights = density_shape / density_shape.sum(
        -2, keepdim=True).clamp_min(1.0e-12)
    human_mass = -torch.expm1(
        -integrated_hazard * stop_fraction).unsqueeze(-2)
    event_mass_weights = human_mass * conditional_weights
    return fractions, event_mass_weights


def replace_human_event_probability(
    event_probability: torch.Tensor,
    human_collision_probability: torch.Tensor,
    *,
    human_index: int = 1,
) -> torch.Tensor:
    """Insert an explicit Human hazard into a categorical Event prediction.

    The remaining classes keep the conditional proportions learned by the
    factual categorical head. This produces one closed distribution and avoids
    counting Human collision once in the Event reward and again as an
    unrelated risk penalty.
    """
    probability = event_probability.float()
    human = human_collision_probability.float()
    if probability.ndim < 1 or not 0 <= int(human_index) < probability.shape[-1]:
        raise ValueError("human_index is outside the Event distribution")
    if human.shape != probability.shape[:-1] + (1,):
        raise ValueError("Human probability must match Event leading dimensions")
    if not torch.isfinite(probability).all() or not torch.isfinite(human).all():
        raise ValueError("Event probabilities must be finite")
    human = human.clamp(0.0, 1.0)
    non_human = probability.clone()
    non_human[..., int(human_index)] = 0.0
    denominator = non_human.sum(-1, keepdim=True).clamp_min(1.0e-8)
    combined = (1.0 - human) * non_human / denominator
    combined[..., int(human_index):int(human_index) + 1] = human
    return combined


def cross_track_potential_reward(
    ego_states: torch.Tensor,
    goal_position: torch.Tensor,
    *,
    discount: float,
    tolerance_m: float,
    potential_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return destination-aligned route shaping and cross-track distance.

    Ego position and the fixed goal are both in the episode-start frame, whose
    origin is the route start. Inside ``tolerance_m`` the potential is exactly
    zero, so ordinary Human avoidance is unconstrained. Outside it, standard
    discounted potential shaping rewards returning toward the route and
    penalizes continued boundary-seeking drift.
    """
    if ego_states.ndim < 3 or ego_states.shape[-1] < 2:
        raise ValueError("ego_states must be [...,time,Ego]")
    if goal_position.shape != ego_states.shape[:-2] + (3,):
        raise ValueError("goal_position must be [...,3]")
    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must lie in (0,1]")
    if float(tolerance_m) < 0.0 or float(potential_scale) < 0.0:
        raise ValueError("route tolerance/scale must be non-negative")
    goal_xy = goal_position[..., :2].to(ego_states)
    position_xy = ego_states[..., :2]
    goal_norm = torch.linalg.vector_norm(goal_xy, dim=-1).clamp_min(1.0e-6)
    cross = (
        goal_xy[..., 0, None] * position_xy[..., 1]
        - goal_xy[..., 1, None] * position_xy[..., 0]
    ).abs() / goal_norm[..., None]
    excess = F.relu(cross - float(tolerance_m))
    # Use a linear potential so ``potential_scale`` has the same physical
    # unit as the simulator's progress coefficient (reward / metre).  This
    # avoids an otherwise arbitrary choice of the cross-track distance at
    # which a quadratic penalty should equal forward progress.  It remains a
    # standard discounted potential F(s,s') = gamma*Phi(s') - Phi(s), rather
    # than a per-step corridor constraint.
    potential = -float(potential_scale) * excess
    transition = (
        float(discount) * potential[..., 1:] - potential[..., :-1]
    )[..., None]
    aligned = torch.cat((torch.zeros_like(transition[..., :1, :]), transition), -2)
    return aligned, cross[..., None]


def analytic_route_deviation_reward(
    ego_state: torch.Tensor,
    goal_position: torch.Tensor,
    *,
    tolerance_m: float,
    weight_per_m_per_sec: float,
    dt_s: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize time spent outside the destination-aligned free corridor.

    Potential shaping alone telescopes under Dreamer's discounted return: it
    improves credit assignment, but deliberately cannot change the preferred
    task policy. The crowd task, however, genuinely prefers completing the
    traversal without drifting toward a warehouse edge. This term expresses
    that objective directly. It is zero throughout the physical avoidance
    allowance and linear outside it, so the Actor remains free to pass on
    either side of a person while persistent one-sided drift has a dense
    restoring gradient.

    ``dt_s`` is the active duration of the state cost. It may be a scalar or a
    tensor broadcastable to the returned ``[...,1]`` shape; partial terminal
    transitions therefore use their actual exposure rather than paying a full
    control tick after contact.
    """
    if ego_state.shape[-1] < 2 or goal_position.shape[-1] != 3:
        raise ValueError("route deviation requires Ego state and Goal3")
    try:
        goal = torch.broadcast_to(
            goal_position.to(ego_state), ego_state.shape[:-1] + (3,))
    except RuntimeError as error:
        raise ValueError(
            "goal_position cannot broadcast to route-deviation states"
        ) from error
    tolerance = float(tolerance_m)
    weight = float(weight_per_m_per_sec)
    if (
        not math.isfinite(tolerance)
        or not math.isfinite(weight)
        or tolerance < 0.0
        or weight < 0.0
    ):
        raise ValueError("route deviation parameters must be finite/non-negative")
    dt = torch.as_tensor(dt_s, dtype=ego_state.dtype, device=ego_state.device)
    try:
        dt = torch.broadcast_to(dt, ego_state.shape[:-1] + (1,))
    except RuntimeError as error:
        raise ValueError(
            "dt_s cannot broadcast to route-deviation states"
        ) from error
    if not torch.isfinite(dt).all() or bool((dt < 0.0).any()):
        raise ValueError("route-deviation dt_s must be finite/non-negative")
    goal_xy = goal[..., :2]
    goal_norm = torch.linalg.vector_norm(
        goal_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    cross = (
        goal_xy[..., 0:1] * ego_state[..., 1:2]
        - goal_xy[..., 1:2] * ego_state[..., 0:1]
    ).abs() / goal_norm
    excess = F.relu(cross - tolerance)
    return -weight * dt * excess, cross


def route_heading_potential(
    ego_states: torch.Tensor,
    goal_position: torch.Tensor,
    *,
    potential_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a bounded potential for facing the fixed episode route.

    The vehicle is holonomic: body-forward and body-lateral velocity can reach
    an off-axis point without rotating the airframe.  Facing the instantaneous
    point goal therefore prescribed unnecessary yaw after a lateral avoidance
    manoeuvre, precisely when the Actor should combine lateral correction with
    forward progress.  The fixed episode-start-to-goal vector instead defines
    the task route. ``cos(error) - 1`` still exposes circular yaw inside H15,
    but cannot reward turning the nose diagonally or backwards merely because
    the vehicle is cross-track near the destination.  At zero route length the
    heading is undefined and the potential is exactly zero.

    This function deliberately returns a state potential rather than a reward.
    Callers must use the same terminal-aware discounted conversion as route and
    boundary shaping so the absorbing-task objective remains well defined.
    """
    if ego_states.ndim < 3 or ego_states.shape[-1] < 14:
        raise ValueError("route heading potential requires Ego14 time states")
    if goal_position.shape != ego_states.shape[:-2] + (3,):
        raise ValueError("goal_position must be [...,3]")
    scale = float(potential_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("route heading potential scale must be finite/non-negative")

    route_xy = goal_position[..., :2].to(ego_states)
    route_length = torch.linalg.vector_norm(
        route_xy, dim=-1, keepdim=True)
    route_xy = route_xy[..., None, :]
    sin_yaw = ego_states[..., 12:13]
    cos_yaw = ego_states[..., 13:14]
    forward_projection = (
        cos_yaw * route_xy[..., 0:1]
        + sin_yaw * route_xy[..., 1:2]
    )
    lateral_projection = (
        -sin_yaw * route_xy[..., 0:1]
        + cos_yaw * route_xy[..., 1:2]
    )
    cosine_error = forward_projection / route_length[..., None].clamp_min(
        1.0e-6)
    cosine_error = cosine_error.clamp(-1.0, 1.0)
    cosine_error = torch.where(
        route_length[..., None] > 1.0e-6,
        cosine_error,
        torch.ones_like(cosine_error),
    )
    potential = scale * (cosine_error - 1.0)
    # This is a diagnostic, but keep it numerically regular at exact
    # alignment/anti-alignment as it is returned from the live objective.
    # Unlike acos(clamp(cos)), atan2 does not acquire an infinite derivative
    # at either endpoint.
    heading_error = torch.atan2(
        lateral_projection.abs(), forward_projection)
    heading_error = torch.where(
        route_length[..., None] > 1.0e-6,
        heading_error,
        torch.zeros_like(heading_error),
    )
    return potential, heading_error


def flight_bounds_potential_reward(
    signed_clearance_m: torch.Tensor,
    *,
    discount: float,
    lookahead_margin_m: float,
    potential_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a differentiable discounted potential near exact flight bounds.

    ``signed_clearance_m`` is the exact minimum axis-aligned distance to the
    recorded flight box (positive inside, negative outside). The margin is
    supplied by the caller from the maximum distance that an action can cover
    over its imagination horizon. This adds an inward gradient before the hard
    out-of-bounds event while remaining ordinary state-reward shaping inside
    Dreamer's differentiable rollout; it is not a controller guard.
    """
    if signed_clearance_m.ndim < 3 or signed_clearance_m.shape[-1] != 1:
        raise ValueError(
            "signed flight-bound clearance must be [...,time,1]")
    if signed_clearance_m.shape[-2] < 2:
        raise ValueError("flight-bound potential requires at least two states")
    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must lie in (0,1]")
    if float(lookahead_margin_m) < 0.0 or float(potential_scale) < 0.0:
        raise ValueError("flight-bound potential parameters must be non-negative")
    excess = F.relu(
        float(lookahead_margin_m) - signed_clearance_m.float())
    potential = -float(potential_scale) * excess
    transition = (
        float(discount) * potential[..., 1:, :] - potential[..., :-1, :])
    aligned = torch.cat((
        torch.zeros_like(transition[..., :1, :]), transition), dim=-2)
    return aligned, excess


def analytic_boundary_proximity_reward(
    signed_clearance_m: torch.Tensor,
    continuation: torch.Tensor,
    *,
    lookahead_margin_m: float,
    weight_per_second: float,
    dt_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge non-refundable time spent near a flight boundary.

    The exact out-of-bounds event remains the terminal authority.  This
    continuous term exists only to expose an inward gradient before that hard
    event is crossed.  Unlike a discounted potential difference, moving back
    inward does not refund an earlier approach and an absorbing terminal can
    never turn a negative near-boundary state into a positive reward.

    ``signed_clearance_m`` contains states, while ``continuation`` contains
    the intervening transitions.  The cost is evaluated at the destination
    only for probability mass that remains active there.  Consequently a
    goal, Human contact, crash, or boundary terminal does not accrue a
    fictitious full-step state cost after its physical stop time.
    """
    if signed_clearance_m.ndim < 3 or signed_clearance_m.shape[-1] != 1:
        raise ValueError(
            "signed flight-bound clearance must be [...,time,1]")
    if continuation.shape != signed_clearance_m.shape[:-2] + (
        signed_clearance_m.shape[-2] - 1, 1,
    ):
        raise ValueError(
            "continuation must match flight-bound clearance transitions")
    margin = float(lookahead_margin_m)
    weight = float(weight_per_second)
    duration = float(dt_s)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("boundary lookahead margin must be finite/positive")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("boundary proximity weight must be finite/non-negative")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("boundary proximity dt_s must be finite/positive")
    if not torch.isfinite(signed_clearance_m).all():
        raise ValueError("signed flight-bound clearance must be finite")
    if not torch.isfinite(continuation).all():
        raise ValueError("boundary continuation must be finite")

    destination = signed_clearance_m.float()[..., 1:, :]
    excess = F.relu(destination.new_tensor(margin) - destination)
    normalized_proximity = (excess / margin).clamp(0.0, 1.0)
    survival = continuation.float().clamp(0.0, 1.0)
    reward = (
        -weight * duration * survival * normalized_proximity.square())
    return reward, excess


def terminal_aware_potential_reward(
    potential: torch.Tensor,
    continuation: torch.Tensor,
    *,
    discount: float,
) -> torch.Tensor:
    """Return ``gamma*c*Phi(next) - Phi(source)`` for an absorbing terminal.

    Ordinary potential shaping assumes a terminal state's successor potential
    is zero.  Applying ``gamma*Phi(next)-Phi(source)`` unchanged on the final
    transition instead leaks an arbitrary terminal-state potential into the
    task objective. ``continuation`` is the same closed competing-risk
    survival probability used by Dreamer's return recursion.
    """
    if potential.ndim < 2 or potential.shape[-1] != 1:
        raise ValueError("potential must be [...,time,1]")
    if continuation.shape != potential.shape[:-2] + (
        potential.shape[-2] - 1, 1,
    ):
        raise ValueError(
            "continuation must match the transitions between potential states")
    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must lie in (0,1]")
    if not torch.isfinite(potential).all():
        raise ValueError("potential must be finite")
    if not torch.isfinite(continuation).all():
        raise ValueError("continuation must be finite")
    continuation = continuation.float().clamp(0.0, 1.0)
    potential = potential.float()
    return (
        float(discount) * continuation * potential[..., 1:, :]
        - potential[..., :-1, :]
    )


def analytic_human_clearance_reward(
    clearance_m: torch.Tensor,
    *,
    hard_clearance_m: float,
    safe_clearance_m: float,
    weight_per_second: float,
    dt_s: float,
) -> torch.Tensor:
    """Differentiable instantaneous-clearance term matching simulator units."""
    hard = float(hard_clearance_m)
    safe = float(safe_clearance_m)
    if not safe > hard:
        raise ValueError("safe Human clearance must exceed hard clearance")
    if float(weight_per_second) < 0.0 or float(dt_s) <= 0.0:
        raise ValueError("clearance weight/dt must be valid")
    normalized = ((safe - clearance_m.float()) / (safe - hard)).clamp(0.0, 1.0)
    return -float(weight_per_second) * float(dt_s) * normalized.square()


def analytic_human_risk_reward(
    per_human_signed_clearance_m: torch.Tensor,
    human_root_body: torch.Tensor,
    human_velocity_body: torch.Tensor,
    human_valid: torch.Tensor,
    pelvis_valid: torch.Tensor | None = None,
    human_presence: torch.Tensor | None = None,
    *,
    ego_velocity_body: torch.Tensor | None = None,
    hard_clearance_m: float,
    safe_clearance_m: float,
    predictive_safe_clearance_m: float,
    predictive_horizon_s: float,
    corridor_half_width_m: float,
    corridor_lookahead_m: float,
    drone_radius_m: float,
    pelvis_radius_m: float,
    weight_per_second: float,
    dt_s: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return action-conditioned Human risk without a proximity-only veto.

    All inputs are deployable body-frame state. Compact-v3 Human velocity is
    an Ego-motion-compensated estimate of Human *world* velocity, not relative
    velocity. ``ego_velocity_body`` is therefore subtracted explicitly so the
    closest-approach calculation is identical to the recorder after a rigid
    rotation. During
    imagination, soft per-track existence is integrated as the exact expected
    maximum under the same independent-slot assumption as Event noisy-OR.

    ``corridor_risk`` is retained as a diagnostic for the recorder's legacy
    front-corridor heuristic, but it is deliberately excluded from the Actor
    reward.  A Human merely being ahead does not imply that slowing down is the
    safest response: for a crossing trajectory, maintaining or increasing
    forward speed can pass before the Human while a lateral move can pass
    behind.  Instantaneous articulated clearance and signed relative-velocity
    closest approach remain symmetric over every visible Human and decide
    whether the trajectories actually conflict.
    """
    if per_human_signed_clearance_m.shape != human_valid.shape:
        raise ValueError("per-Human clearance and validity shapes differ")
    if human_root_body.shape != human_velocity_body.shape \
            or human_root_body.shape[:-1] != human_valid.shape \
            or human_root_body.shape[-1] != 3:
        raise ValueError("Human root/velocity must be [...,N,3]")
    hard = float(hard_clearance_m)
    safe = float(safe_clearance_m)
    predictive_safe = float(predictive_safe_clearance_m)
    if not safe > hard or not predictive_safe > hard:
        raise ValueError("Human safe clearances must exceed hard clearance")
    if float(predictive_horizon_s) <= 0.0:
        raise ValueError("predictive horizon must be positive")
    if float(corridor_half_width_m) <= 0.0 \
            or float(corridor_lookahead_m) <= 0.0:
        raise ValueError("Human corridor dimensions must be positive")

    valid = human_valid.bool()
    presence = (
        valid.to(per_human_signed_clearance_m)
        if human_presence is None else human_presence.float().clamp(0.0, 1.0)
    )
    if presence.shape != valid.shape:
        raise ValueError("Human presence must match Human slots")
    presence = presence * valid.to(presence)

    def expected_maximum(slot_risk: torch.Tensor) -> torch.Tensor:
        """Expected maximum under independent per-slot existence."""
        ordered_risk, order = slot_risk.sort(dim=-1, descending=True)
        ordered_presence = presence.gather(-1, order)
        previous_absent = torch.cat((
            torch.ones_like(ordered_presence[..., :1]),
            torch.cumprod(
                1.0 - ordered_presence[..., :-1], dim=-1),
        ), -1)
        return (
            ordered_risk * ordered_presence * previous_absent
        ).sum(-1, keepdim=True)

    pelvis_is_valid = valid if pelvis_valid is None else pelvis_valid.bool()
    if pelvis_is_valid.shape != valid.shape:
        raise ValueError("pelvis validity must match Human slots")
    clearance = per_human_signed_clearance_m.float()
    instant_normalized = (
        (safe - clearance) / (safe - hard)
    ).clamp(0.0, 1.0)
    instant_slot = instant_normalized.square().masked_fill(~valid, 0.0)
    instant = expected_maximum(instant_slot)

    relative_position = human_root_body.float()
    if ego_velocity_body is None:
        ego_velocity_body = torch.zeros_like(human_root_body[..., 0, :])
    if ego_velocity_body.shape != human_root_body.shape[:-2] + (3,):
        raise ValueError(
            "Ego velocity must match Human leading dimensions without slots")
    relative_velocity = (
        human_velocity_body.float() - ego_velocity_body[..., None, :].float())
    speed_sq = relative_velocity.square().sum(-1)
    ttc = -(
        relative_position * relative_velocity
    ).sum(-1) / speed_sq.clamp_min(1.0e-8)
    ttc = ttc.clamp(0.0, float(predictive_horizon_s))
    closest = relative_position + relative_velocity * ttc[..., None]
    closest_clearance = (
        torch.linalg.vector_norm(closest, dim=-1)
        - float(drone_radius_m) - float(pelvis_radius_m)
    )
    predictive_spatial = (
        (predictive_safe - closest_clearance)
        / (predictive_safe - hard)
    ).clamp(0.0, 1.0).square()
    predictive_temporal = (
        1.0 - ttc / float(predictive_horizon_s)
    ).square()
    predictive_valid = (
        pelvis_is_valid
        & speed_sq.gt(1.0e-8)
        & ttc.gt(0.0)
        & ttc.lt(float(predictive_horizon_s))
        & closest_clearance.lt(predictive_safe)
    )
    predictive_slot = (
        predictive_spatial * predictive_temporal
    ).masked_fill(~predictive_valid, 0.0)
    predictive = expected_maximum(predictive_slot)

    forward = relative_position[..., 0]
    lateral = relative_position[..., 1].abs()
    corridor_valid = (
        pelvis_is_valid
        & forward.gt(0.0)
        & forward.lt(float(corridor_lookahead_m))
        & lateral.lt(float(corridor_half_width_m))
    )
    corridor_slot = (
        (1.0 - lateral / float(corridor_half_width_m)).square()
        * (1.0 - 0.5 * forward / float(corridor_lookahead_m))
    ).masked_fill(~corridor_valid, 0.0)
    corridor = expected_maximum(corridor_slot)

    # Do not let the front-only corridor become an implicit reverse-action
    # reward.  The two active terms are scene symmetric and depend on physical
    # separation or time-to-closest-approach rather than a hard-coded side.
    combined_slot = torch.maximum(instant_slot, predictive_slot)
    combined = expected_maximum(combined_slot)
    reward = (
        -float(weight_per_second) * float(dt_s) * combined)
    return reward, {
        "combined_risk": combined,
        "instant_risk": instant,
        "predictive_risk": predictive,
        "corridor_risk": corridor,
        "minimum_ttc_s": ttc.masked_fill(
            ~predictive_valid, torch.inf).amin(-1, keepdim=True),
    }


def analytic_smoothness_reward(
    source_task_memory: torch.Tensor,
    destination_task_memory: torch.Tensor,
    source_ego_state: torch.Tensor,
    destination_ego_state: torch.Tensor,
    *,
    dt_s: float,
    acceleration_weight_per_sec: float,
    acceleration_scale_mps2: float,
    jerk_weight_per_sec: float,
    jerk_scale_mps3: float,
    yaw_rate_weight_per_sec: float,
    yaw_rate_scale_rps: float,
    term_clip: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact structured counterpart of ``CrowdRewardCalculator`` smoothing."""
    if source_task_memory.shape != destination_task_memory.shape \
            or source_task_memory.shape[-1] < 6:
        raise ValueError("smoothness reward requires paired task memory")
    if source_ego_state.shape != destination_ego_state.shape \
            or source_ego_state.shape[-1] != 14 \
            or source_ego_state.shape[:-1] != source_task_memory.shape[:-1]:
        raise ValueError("smoothness reward Ego/memory shapes differ")
    if float(dt_s) <= 0.0:
        raise ValueError("smoothness dt must be positive")
    filtered = destination_task_memory[..., 4:6]
    previous_filtered = source_task_memory[..., 4:6]
    acceleration_norm = torch.linalg.vector_norm(filtered, dim=-1, keepdim=True)
    jerk_norm = torch.linalg.vector_norm(
        (filtered - previous_filtered) / float(dt_s), dim=-1, keepdim=True)
    source_yaw = torch.atan2(
        source_ego_state[..., 12], source_ego_state[..., 13])
    destination_yaw = torch.atan2(
        destination_ego_state[..., 12], destination_ego_state[..., 13])
    yaw_delta = torch.atan2(
        torch.sin(destination_yaw - source_yaw),
        torch.cos(destination_yaw - source_yaw),
    )
    yaw_rate = (yaw_delta / float(dt_s)).abs()[..., None]
    acceleration_term = (
        acceleration_norm / float(acceleration_scale_mps2)
    ).square().clamp(max=float(term_clip))
    jerk_term = (
        jerk_norm / float(jerk_scale_mps3)
    ).square().clamp(max=float(term_clip))
    yaw_term = (
        yaw_rate / float(yaw_rate_scale_rps)
    ).square().clamp(max=float(term_clip))
    reward = -float(dt_s) * (
        float(acceleration_weight_per_sec) * acceleration_term
        + float(jerk_weight_per_sec) * jerk_term
        + float(yaw_rate_weight_per_sec) * yaw_term
    )
    return reward, {
        "acceleration_norm_mps2": acceleration_norm,
        "jerk_norm_mps3": jerk_norm,
        "yaw_rate_rps": yaw_rate,
    }


def analytic_fractional_smoothness_reward(
    source_task_memory: torch.Tensor,
    source_ego_state: torch.Tensor,
    event_ego_state: torch.Tensor,
    transition_fraction: torch.Tensor,
    *,
    dt_s: float,
    acceleration_filter_alpha: float,
    acceleration_weight_per_sec: float,
    acceleration_scale_mps2: float,
    jerk_weight_per_sec: float,
    jerk_scale_mps3: float,
    yaw_rate_weight_per_sec: float,
    yaw_rate_scale_rps: float,
    term_clip: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Recorder-exact smoothing up to an intra-transition event time.

    A short force-written terminal row is not the full-step smoothness reward
    multiplied by its duration fraction.  The recorder updates filtered
    acceleration once at the event state and divides its change and yaw change
    by the *actual* partial duration.  In particular, jerk can reach its clip on
    a 16 ms row even when the corresponding 100 ms value would not.
    """
    if source_task_memory.shape[-1] < 6:
        raise ValueError("fractional smoothness requires task memory")
    if source_ego_state.shape != event_ego_state.shape \
            or source_ego_state.shape[-1] != 14 \
            or source_ego_state.shape[:-1] != source_task_memory.shape[:-1]:
        raise ValueError("fractional smoothness Ego/memory shapes differ")
    if transition_fraction.shape != source_ego_state.shape[:-1] + (1,):
        raise ValueError("fractional smoothness fraction has wrong shape")
    if float(dt_s) <= 0.0:
        raise ValueError("fractional smoothness dt must be positive")
    alpha = float(acceleration_filter_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("acceleration filter alpha must lie in [0,1]")

    fraction = transition_fraction.float().clamp(0.0, 1.0)
    partial_dt = float(dt_s) * fraction
    safe_dt = partial_dt.clamp_min(1.0e-8)
    event_yaw = torch.atan2(
        event_ego_state[..., 12], event_ego_state[..., 13])
    cosine, sine = event_yaw.cos(), event_yaw.sin()
    acceleration_episode = torch.stack((
        cosine * event_ego_state[..., 6]
        - sine * event_ego_state[..., 7],
        sine * event_ego_state[..., 6]
        + cosine * event_ego_state[..., 7],
    ), -1)
    previous_filtered = source_task_memory[..., 4:6]
    filtered = (
        alpha * acceleration_episode
        + (1.0 - alpha) * previous_filtered)
    jerk = (filtered - previous_filtered) / safe_dt
    jerk = torch.where(
        partial_dt.gt(0.0), jerk, torch.zeros_like(jerk))

    source_yaw = torch.atan2(
        source_ego_state[..., 12], source_ego_state[..., 13])
    yaw_delta = torch.atan2(
        torch.sin(event_yaw - source_yaw),
        torch.cos(event_yaw - source_yaw),
    )[..., None]
    yaw_rate = torch.where(
        partial_dt.gt(0.0), yaw_delta / safe_dt,
        torch.zeros_like(yaw_delta)).abs()
    acceleration_norm = torch.linalg.vector_norm(
        filtered, dim=-1, keepdim=True)
    jerk_norm = torch.linalg.vector_norm(jerk, dim=-1, keepdim=True)
    acceleration_term = (
        acceleration_norm / float(acceleration_scale_mps2)
    ).square().clamp(max=float(term_clip))
    jerk_term = (
        jerk_norm / float(jerk_scale_mps3)
    ).square().clamp(max=float(term_clip))
    yaw_term = (
        yaw_rate / float(yaw_rate_scale_rps)
    ).square().clamp(max=float(term_clip))
    reward = -partial_dt * (
        float(acceleration_weight_per_sec) * acceleration_term
        + float(jerk_weight_per_sec) * jerk_term
        + float(yaw_rate_weight_per_sec) * yaw_term)
    return reward, {
        "acceleration_norm_mps2": acceleration_norm,
        "jerk_norm_mps3": jerk_norm,
        "yaw_rate_rps": yaw_rate,
    }


def compose_analytic_goal_event(
    non_goal_probability: torch.Tensor,
    next_goal_distance: torch.Tensor,
    *,
    non_goal_event_rewards: torch.Tensor,
    goal_radius_m: float,
    goal_reward: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose mutually exclusive analytic-goal and learned non-goal events.

    ``non_goal_probability`` is ordered as ``continue, human, static, other``.
    A goal is possible only from the non-goal ``continue`` mass.  Consequently
    collisions keep priority over reaching the geometric goal, all terminal
    probabilities remain mutually exclusive, and no learned goal class can
    duplicate the analytic goal bonus.

    Returns ``(expected_reward, continuation, goal_probability)``.
    """
    if non_goal_probability.shape[-1] != non_goal_event_rewards.numel():
        raise ValueError(
            "non-goal probability/reward class counts must match")
    if next_goal_distance.shape != non_goal_probability.shape[:-1] + (1,):
        raise ValueError("goal distance must match event leading dimensions")
    if float(goal_radius_m) <= 0.0:
        raise ValueError("goal_radius_m must be positive")
    probability = non_goal_probability.float()
    if bool((probability < 0.0).any()):
        raise ValueError("event probabilities must be non-negative")
    inside_goal = (next_goal_distance <= float(goal_radius_m)).to(probability)
    non_goal_continue = probability[..., :1]
    goal_probability = inside_goal * non_goal_continue
    continuation = (1.0 - inside_goal) * non_goal_continue
    reward = expected_event_reward(
        probability, non_goal_event_rewards) + goal_probability * float(
            goal_reward)
    return reward, continuation, goal_probability


def compose_analytic_goal_boundary_event(
    non_goal_probability: torch.Tensor,
    next_goal_distance: torch.Tensor,
    boundary_signed_clearance_m: torch.Tensor,
    *,
    non_goal_event_rewards: torch.Tensor,
    goal_radius_m: float | torch.Tensor,
    goal_reward: float,
    boundary_reward: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compose exact goal/bounds events with learned non-goal risks.

    ``non_goal_probability`` is ordered ``continue, human, static, other``.
    Human/static collision probability keeps the simulator's first priority.
    Exact goal is next and consumes all remaining non-collision mass, including
    a learned watchdog ``other`` probability. Exact out-of-bounds is last and
    consumes remaining ``continue`` mass. This matches
    ``pegasus_app._detect_recording_event`` followed by its watchdog call.

    ``boundary_signed_clearance_m`` is positive/on-zero inside the flight box
    and negative outside it.  The comparison therefore matches the simulator
    watchdog's strict ``< xmin``/``> xmax`` contract.
    """
    if non_goal_probability.shape[-1] != non_goal_event_rewards.numel():
        raise ValueError(
            "non-goal probability/reward class counts must match")
    expected_shape = non_goal_probability.shape[:-1] + (1,)
    if next_goal_distance.shape != expected_shape:
        raise ValueError("goal distance must match event leading dimensions")
    if boundary_signed_clearance_m.shape != expected_shape:
        raise ValueError(
            "boundary signed clearance must match event leading dimensions")
    probability = non_goal_probability.float()
    if not torch.isfinite(probability).all():
        raise ValueError("event probabilities must be finite")
    if bool((probability < 0.0).any()):
        raise ValueError("event probabilities must be non-negative")
    probability_sum = probability.sum(-1, keepdim=True)
    if not torch.allclose(
        probability_sum, torch.ones_like(probability_sum),
        atol=1.0e-5, rtol=1.0e-5,
    ):
        raise ValueError("non-goal probabilities must form a closed distribution")

    radius = torch.as_tensor(
        goal_radius_m,
        dtype=next_goal_distance.dtype,
        device=next_goal_distance.device,
    )
    while radius.ndim < next_goal_distance.ndim:
        radius = radius.unsqueeze(-2)
    radius = torch.broadcast_to(radius, next_goal_distance.shape)
    if not torch.isfinite(radius).all() or bool((radius <= 0.0).any()):
        raise ValueError("goal radius must be finite and positive")

    inside_goal = (next_goal_distance <= radius).to(probability)
    outside_bounds = boundary_signed_clearance_m.lt(0.0).to(probability)
    continue_mass = probability[..., :1]
    human_probability = probability[..., 1:2]
    static_probability = probability[..., 2:3]
    other_mass = probability[..., 3:4]
    goal_probability = (continue_mass + other_mass) * inside_goal
    learned_other_probability = other_mass * (1.0 - inside_goal)
    boundary_probability = (
        continue_mass * (1.0 - inside_goal) * outside_bounds)
    continuation = (
        continue_mass * (1.0 - inside_goal) * (1.0 - outside_bounds))
    rewards = non_goal_event_rewards.to(probability)
    reward = (
        human_probability * rewards[1]
        + static_probability * rewards[2]
        + learned_other_probability * rewards[3]
        + goal_probability * float(goal_reward)
        + boundary_probability * float(boundary_reward)
    )
    closed_sum = (
        human_probability + static_probability + learned_other_probability
        + goal_probability + boundary_probability + continuation)
    return reward, continuation, {
        "goal": goal_probability,
        "boundary": boundary_probability,
        "learned_other": learned_other_probability,
        "continue": continuation,
        "inside_goal": inside_goal,
        "outside_bounds": outside_bounds,
        "probability_sum": closed_sum,
    }


def compose_human_analytic_events(
    human_hazard: torch.Tensor,
    analytic_terminal: torch.Tensor,
    *,
    human_collision_reward: float,
    analytic_event_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compose the v16 learned-Human/analytic-terminal contract.

    ``human_hazard`` is the sole learned terminal probability and has shape
    ``[..., 1]``. ``analytic_terminal`` contains mutually-exclusive analytic
    first-event indicators/probabilities with shape ``[..., K]``.  Human
    contact has priority at an exact tie, matching the simulator termination
    order.  Learned continuation is deliberately absent: multiplying it here
    would double-count terminals already represented by the Human/analytic
    branches.

    Returns ``(event_reward, continuation, probabilities)``.  The returned
    probability fields form a closed distribution at every transition.
    """
    if human_hazard.shape[-1] != 1:
        raise ValueError("human_hazard must have trailing singleton dimension")
    if analytic_terminal.shape[:-1] != human_hazard.shape[:-1]:
        raise ValueError("analytic terminal leading dimensions must match")
    if analytic_terminal.shape[-1] != analytic_event_rewards.numel():
        raise ValueError("analytic event reward count does not match terminals")
    if not torch.isfinite(human_hazard).all():
        raise ValueError("human_hazard must be finite")
    if not torch.isfinite(analytic_terminal).all():
        raise ValueError("analytic terminal probabilities must be finite")
    human = human_hazard.float().clamp(0.0, 1.0)
    analytic = analytic_terminal.float()
    if bool((analytic < 0.0).any()) or bool((analytic > 1.0).any()):
        raise ValueError("analytic terminal probabilities must be in [0,1]")
    analytic_any = analytic.sum(-1, keepdim=True)
    if bool((analytic_any > 1.0 + 1.0e-6).any()):
        raise ValueError("analytic terminal events must be mutually exclusive")
    analytic_any = analytic_any.clamp(0.0, 1.0)
    no_human = 1.0 - human
    analytic_probability = no_human * analytic
    continuation = no_human * (1.0 - analytic_any)
    reward = human * float(human_collision_reward)
    reward = reward + (
        analytic_probability
        * analytic_event_rewards.to(analytic_probability).reshape(
            *((1,) * (analytic_probability.ndim - 1)), -1)
    ).sum(-1, keepdim=True)
    closed_sum = human + analytic_probability.sum(-1, keepdim=True) + continuation
    result = {
        "human": human,
        "analytic": analytic_probability,
        "continue": continuation,
        "probability_sum": closed_sum,
    }
    return reward, continuation, result


def compose_human_analytic_competing_events(
    nominal_human_probability: torch.Tensor,
    analytic_terminal: torch.Tensor,
    analytic_first_fraction: torch.Tensor,
    *,
    human_collision_reward: float,
    analytic_event_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compose Human hazard with deterministic within-step task terminals.

    ``nominal_human_probability`` is the calibrated Human collision
    probability over one complete model step. It defines a constant
    conditional integrated hazard ``H = -log(1-p)`` within that step.
    ``analytic_terminal`` is a mutually-exclusive deterministic event and
    ``analytic_first_fraction`` is its exact occurrence fraction. If an
    analytic event occurs at ``f``, Human risk is accumulated only until
    ``f``; otherwise it is accumulated through the complete interval.

    This is a closed competing-risk distribution. It does not give a Human
    classifier artificial priority over a goal or boundary that physically
    occurred earlier in the same 0.1 s controller interval. Exact ties have
    zero probability under the continuous Human time model.

    The returned ``expected_active_fraction`` is
    ``integral_0^stop S(u) du`` and can consistently scale dense per-time
    rewards. ``human_mean_event_fraction`` is the conditional mean Human event
    time, used only when the Human branch has non-zero probability mass.
    """
    if nominal_human_probability.shape[-1] != 1:
        raise ValueError(
            "nominal Human probability must have trailing singleton dimension")
    if analytic_terminal.shape[:-1] != nominal_human_probability.shape[:-1]:
        raise ValueError("analytic terminal leading dimensions must match")
    if analytic_terminal.shape[-1] != analytic_event_rewards.numel():
        raise ValueError("analytic event reward count does not match terminals")
    if analytic_first_fraction.shape != nominal_human_probability.shape:
        raise ValueError("analytic first fraction must match Human probability")
    if not torch.isfinite(nominal_human_probability).all() \
            or not torch.isfinite(analytic_terminal).all() \
            or not torch.isfinite(analytic_first_fraction).all():
        raise ValueError("competing-event inputs must be finite")
    probability = nominal_human_probability.float()
    analytic = analytic_terminal.float()
    fraction = analytic_first_fraction.float()
    if bool((probability < 0.0).any()) or bool((probability > 1.0).any()):
        raise ValueError("Human probability must lie in [0,1]")
    if bool((analytic < 0.0).any()) or bool((analytic > 1.0).any()):
        raise ValueError("analytic terminal probabilities must lie in [0,1]")
    analytic_any = analytic.sum(-1, keepdim=True)
    if bool((analytic_any > 1.0 + 1.0e-6).any()):
        raise ValueError("analytic task terminals must be mutually exclusive")
    if bool((fraction < 0.0).any()) or bool((fraction > 1.0).any()):
        raise ValueError("analytic first fraction must lie in [0,1]")

    analytic_any = analytic_any.clamp(0.0, 1.0)
    stop_fraction = torch.where(
        analytic_any > 0.0, fraction, torch.ones_like(fraction))
    integrated_hazard = -torch.log1p(
        -probability.clamp(max=1.0 - 1.0e-7))
    stopped_hazard = integrated_hazard * stop_fraction
    survival_to_stop = torch.exp(-stopped_hazard)
    human = (-torch.expm1(-stopped_hazard)).clamp(0.0, 1.0)
    analytic_probability = survival_to_stop * analytic
    continuation = survival_to_stop * (1.0 - analytic_any)

    # E[min(T_human, stop)] in units of one model interval.  Use the analytic
    # limit at zero hazard to avoid a 0/0 branch and cancellation near zero.
    small_hazard = integrated_hazard.abs() < 1.0e-6
    expected_active_fraction = torch.where(
        small_hazard,
        stop_fraction,
        human / integrated_hazard.clamp_min(1.0e-12),
    )
    # Integral u * H exp(-H u) du over [0, stop]. Dividing by the Human event
    # mass gives E[T | T < stop]. The small-hazard conditional limit is stop/2.
    event_time_mass = torch.where(
        small_hazard,
        0.5 * integrated_hazard * stop_fraction.square(),
        (
            human - stopped_hazard * survival_to_stop
        ) / integrated_hazard.clamp_min(1.0e-12),
    )
    human_mean_fraction = torch.where(
        human > 1.0e-8,
        event_time_mass / human.clamp_min(1.0e-12),
        0.5 * stop_fraction,
    ).clamp(0.0, 1.0)

    reward = human * float(human_collision_reward)
    reward = reward + (
        analytic_probability
        * analytic_event_rewards.to(analytic_probability).reshape(
            *((1,) * (analytic_probability.ndim - 1)), -1)
    ).sum(-1, keepdim=True)
    closed_sum = human + analytic_probability.sum(-1, keepdim=True) + continuation
    result = {
        "human": human,
        "analytic": analytic_probability,
        "continue": continuation,
        "probability_sum": closed_sum,
        "nominal_human_probability": probability,
        "integrated_human_hazard": integrated_hazard,
        "stop_fraction": stop_fraction,
        "expected_active_fraction": expected_active_fraction,
        "human_mean_event_fraction": human_mean_fraction,
    }
    if analytic_probability.shape[-1] == 4:
        result.update({
            "goal": analytic_probability[..., 0:1],
            "boundary": analytic_probability[..., 1:2],
            "crash": analytic_probability[..., 2:3],
            "stuck": analytic_probability[..., 3:4],
        })
    return reward, continuation, result


def compose_swept_human_analytic_events(
    human_terminal: torch.Tensor,
    human_first_fraction: torch.Tensor,
    analytic_terminal: torch.Tensor,
    analytic_first_fraction: torch.Tensor,
    *,
    human_collision_reward: float,
    analytic_event_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compose deterministic swept-Human and analytic task events.

    Human collision geometry is known: the simulator reports contact when a
    swept UAV/joint sphere pair enters its serialized contact shell.  Once the
    World Model has recursively predicted the articulated Human endpoints,
    asking a second unconstrained action-conditioned classifier to rediscover
    that contact rule makes counterfactual action ordering non-identifiable.
    This helper instead orders the predicted swept contact and the existing
    analytic goal/boundary/crash/stuck event by their within-step fractions.

    ``human_terminal`` is a deterministic 0/1 tensor.  The comparison itself
    is intentionally piecewise: clearance shaping supplies the useful
    pre-contact action gradient, while the terminal reward preserves the
    simulator's literal event rather than inventing a soft collision label.
    """
    if human_terminal.shape[-1] != 1:
        raise ValueError("swept Human terminal must have a singleton tail")
    if human_first_fraction.shape != human_terminal.shape:
        raise ValueError("swept Human fraction must match its terminal")
    if analytic_terminal.shape[:-1] != human_terminal.shape[:-1]:
        raise ValueError("analytic terminal leading dimensions must match")
    if analytic_terminal.shape[-1] != analytic_event_rewards.numel():
        raise ValueError("analytic event reward count does not match terminals")
    if analytic_first_fraction.shape != human_terminal.shape:
        raise ValueError("analytic first fraction must match Human terminal")
    if not torch.isfinite(human_terminal).all() \
            or not torch.isfinite(human_first_fraction).all() \
            or not torch.isfinite(analytic_terminal).all() \
            or not torch.isfinite(analytic_first_fraction).all():
        raise ValueError("swept competing-event inputs must be finite")
    human = human_terminal.float()
    analytic = analytic_terminal.float()
    human_fraction = human_first_fraction.float()
    analytic_fraction = analytic_first_fraction.float()
    if bool((human < 0.0).any()) or bool((human > 1.0).any()):
        raise ValueError("swept Human terminal must lie in [0,1]")
    if bool((analytic < 0.0).any()) or bool((analytic > 1.0).any()):
        raise ValueError("analytic terminal probabilities must lie in [0,1]")
    if bool((human_fraction < 0.0).any()) or bool((human_fraction > 1.0).any()):
        raise ValueError("swept Human fraction must lie in [0,1]")
    if bool((analytic_fraction < 0.0).any()) \
            or bool((analytic_fraction > 1.0).any()):
        raise ValueError("analytic first fraction must lie in [0,1]")

    analytic_any = analytic.sum(-1, keepdim=True)
    if bool((analytic_any > 1.0 + 1.0e-6).any()):
        raise ValueError("analytic task terminals must be mutually exclusive")
    analytic_any = analytic_any.clamp(0.0, 1.0)
    # A physical contact at the exact same fraction wins the recorder event
    # tie. Exact ties have measure zero, but the <= rule is deterministic.
    human_first = (
        human.gt(0.0)
        & (analytic_any.le(0.0) | human_fraction.le(analytic_fraction))
    )
    human_probability = human_first.to(human)
    analytic_probability = analytic * (~human_first).to(analytic)
    analytic_wins = analytic_probability.sum(-1, keepdim=True).clamp(0.0, 1.0)
    continuation = 1.0 - human_probability - analytic_wins
    continuation = continuation.clamp(0.0, 1.0)
    stop_fraction = torch.where(
        human_first,
        human_fraction,
        torch.where(analytic_wins.gt(0.0), analytic_fraction,
                    torch.ones_like(human_fraction)),
    )
    reward = human_probability * float(human_collision_reward)
    reward = reward + (
        analytic_probability
        * analytic_event_rewards.to(analytic_probability).reshape(
            *((1,) * (analytic_probability.ndim - 1)), -1)
    ).sum(-1, keepdim=True)
    closed_sum = (
        human_probability + analytic_probability.sum(-1, keepdim=True)
        + continuation)
    result = {
        "human": human_probability,
        "analytic": analytic_probability,
        "continue": continuation,
        "probability_sum": closed_sum,
        "nominal_human_probability": human,
        # Retain common metric keys without pretending that a deterministic
        # swept contact is a constant stochastic hazard.
        "integrated_human_hazard": torch.zeros_like(human),
        "stop_fraction": stop_fraction,
        "expected_active_fraction": stop_fraction,
        "human_mean_event_fraction": torch.where(
            human_first, human_fraction, 0.5 * stop_fraction),
    }
    if analytic_probability.shape[-1] == 4:
        result.update({
            "goal": analytic_probability[..., 0:1],
            "boundary": analytic_probability[..., 1:2],
            "crash": analytic_probability[..., 2:3],
            "stuck": analytic_probability[..., 3:4],
        })
    return reward, continuation, result


def swept_human_contact_probability(
    swept_signed_gap_m: torch.Tensor,
    *,
    prediction_bias_m: float | torch.Tensor,
    prediction_scale_m: float | torch.Tensor,
) -> torch.Tensor:
    """Map learned swept-gap residual uncertainty to contact probability.

    Conditional on exact articulated endpoints, PhysX contact is
    deterministic.  Learned endpoints are not exact, so the signed-gap
    residual is modelled as a logistic distribution.  This yields
    ``sigmoid((bias - predicted_gap) / scale)`` and supplies a continuous
    action gradient for narrow predicted misses without introducing another
    action-conditioned Event classifier.
    """
    bias = torch.as_tensor(
        prediction_bias_m, dtype=torch.float32,
        device=swept_signed_gap_m.device)
    scale = torch.as_tensor(
        prediction_scale_m, dtype=torch.float32,
        device=swept_signed_gap_m.device)
    if not torch.isfinite(bias).all() or bool((bias < 0.0).any()):
        raise ValueError("swept-gap prediction bias must be finite/non-negative")
    if not torch.isfinite(scale).all() or bool((scale <= 0.0).any()):
        raise ValueError("swept-gap prediction scale must be finite/positive")
    if not torch.isfinite(swept_signed_gap_m).all():
        raise ValueError("swept signed gap must be finite")
    try:
        bias, scale, gap = torch.broadcast_tensors(
            bias, scale, swept_signed_gap_m.float())
    except RuntimeError as error:
        raise ValueError(
            "swept-gap uncertainty cannot broadcast to rollout") from error
    return torch.sigmoid((bias - gap) / scale)


def independent_human_contact_union_probability(
    slot_probability: torch.Tensor,
    slot_valid: torch.Tensor,
) -> torch.Tensor:
    """Probability of contact with any valid Human slot.

    The articulated swept-gap residual is defined per person.  Collapsing the
    predicted gaps with a hard minimum before applying the residual model
    makes only the currently nearest person differentiable.  That is not the
    probability of the union of multiple possible contacts and creates
    discontinuous avoidance gradients when the nearest identity changes.
    Under the same independent-slot assumption used by the Human lifecycle
    and Event noisy-OR paths, the closed probability is
    ``1 - product_i(1 - p_i)``.  Invalid/padded slots contribute exactly zero.
    """
    if slot_probability.shape != slot_valid.shape:
        raise ValueError(
            "Human contact probabilities and validity masks must match")
    if slot_probability.ndim == 0:
        raise ValueError("Human contact probabilities require a slot axis")
    if not torch.isfinite(slot_probability).all():
        raise ValueError("Human contact probabilities must be finite")
    if bool((slot_probability < 0.0).any()) or bool((
        slot_probability > 1.0
    ).any()):
        raise ValueError("Human contact probabilities must lie in [0,1]")
    probability = torch.where(
        slot_valid.bool(), slot_probability.float(),
        torch.zeros_like(slot_probability, dtype=torch.float32),
    )
    return 1.0 - torch.prod(1.0 - probability, dim=-1, keepdim=True)


def trajectory_correlated_human_contact_hazard(
    slot_probability: torch.Tensor,
    slot_valid: torch.Tensor,
) -> torch.Tensor:
    """Convert per-slot contact CDFs into non-repeated temporal hazards.

    ``swept_human_contact_probability`` estimates the chance that the shared
    prediction error of one imagined Human trajectory is large enough to turn
    a predicted swept gap into contact.  Adjacent 100 ms gaps come from that
    same trajectory error; treating every value as a fresh independent hazard
    repeatedly charges one persistent near miss and makes the result depend on
    how finely the horizon is discretized.

    Interpret each slot value as a marginal contact CDF, retain the largest
    CDF reached so far for that person, form the independent union across
    people, and difference the cumulative union into a conditional per-step
    hazard.  Consequently the returned hazards reproduce the final cumulative
    union exactly while preserving gradients through every person that sets a
    new closest-approach envelope.

    Inputs have shape ``[..., time, people]`` and the output has shape
    ``[..., time, 1]``.
    """
    if slot_probability.shape != slot_valid.shape:
        raise ValueError(
            "Human contact probabilities and validity masks must match")
    if slot_probability.ndim < 2:
        raise ValueError(
            "trajectory-correlated Human contact needs time and slot axes")
    if not torch.isfinite(slot_probability).all():
        raise ValueError("Human contact probabilities must be finite")
    if bool((slot_probability < 0.0).any()) or bool((
        slot_probability > 1.0
    ).any()):
        raise ValueError("Human contact probabilities must lie in [0,1]")

    probability = torch.where(
        slot_valid.bool(), slot_probability.float(),
        torch.zeros_like(slot_probability, dtype=torch.float32),
    )
    cumulative_slot_probability = torch.cummax(
        probability, dim=-2).values
    cumulative_union = 1.0 - torch.prod(
        1.0 - cumulative_slot_probability, dim=-1, keepdim=True)
    previous_union = torch.cat((
        torch.zeros_like(cumulative_union[..., :1, :]),
        cumulative_union[..., :-1, :],
    ), dim=-2)
    event_increment = (cumulative_union - previous_union).clamp_min(0.0)
    previous_survival = 1.0 - previous_union
    hazard = torch.where(
        previous_survival > 1.0e-7,
        event_increment / previous_survival.clamp_min(1.0e-7),
        torch.zeros_like(event_increment),
    )
    return hazard.clamp(0.0, 1.0)


def compose_uncertain_swept_human_analytic_events(
    exact_human_terminal: torch.Tensor,
    exact_human_first_fraction: torch.Tensor,
    uncertain_human_probability: torch.Tensor,
    analytic_terminal: torch.Tensor,
    analytic_first_fraction: torch.Tensor,
    *,
    human_collision_reward: float,
    analytic_event_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compose literal contact and learned-geometry uncertainty once.

    Predicted physical intersections retain their exact swept contact time and
    deterministic simulator ordering.  Predicted misses use the calibrated
    geometry-error probability as a within-step hazard competing with the
    exact analytic task-event time.  Both branches form a closed distribution
    and feed the same ordinary Dreamer return.
    """
    if uncertain_human_probability.shape != exact_human_terminal.shape:
        raise ValueError(
            "uncertain Human probability must match exact swept terminal")
    if not torch.isfinite(uncertain_human_probability).all():
        raise ValueError("uncertain Human probability must be finite")
    if bool((uncertain_human_probability < 0.0).any()) or bool((
        uncertain_human_probability > 1.0
    ).any()):
        raise ValueError("uncertain Human probability must lie in [0,1]")

    exact_reward, exact_continuation, exact_event = (
        compose_swept_human_analytic_events(
            exact_human_terminal,
            exact_human_first_fraction,
            analytic_terminal,
            analytic_first_fraction,
            human_collision_reward=human_collision_reward,
            analytic_event_rewards=analytic_event_rewards,
        ))
    uncertain_reward, uncertain_continuation, uncertain_event = (
        compose_human_analytic_competing_events(
            uncertain_human_probability,
            analytic_terminal,
            analytic_first_fraction,
            human_collision_reward=human_collision_reward,
            analytic_event_rewards=analytic_event_rewards,
        ))
    exact_selector = exact_human_terminal.bool()

    def select(
        exact_value: torch.Tensor,
        uncertain_value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(exact_selector, exact_value, uncertain_value)

    result = {
        key: select(exact_event[key], uncertain_event[key])
        for key in exact_event
    }
    result["exact_swept_human_contact"] = exact_human_terminal.float()
    result["uncertain_human_probability"] = (
        uncertain_human_probability.float())
    return (
        select(exact_reward, uncertain_reward),
        select(exact_continuation, uncertain_continuation),
        result,
    )


def compose_residual_analytic_events(
    residual_probability: torch.Tensor,
    analytic_terminal: torch.Tensor,
    *,
    event_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compose learned residual risks with exact goal/boundary events.

    ``residual_probability`` keeps the compatibility layout ``continue,
    human, static, crash, stuck``, but the last two entries must be exactly
    zero. ``analytic_terminal`` contains mutually exclusive ``goal, boundary,
    crash, stuck`` indicators derived from deployable geometry/task memory.
    Human/static collision retains simulator priority, then the analytic task
    events consume residual continuation mass. This avoids asking a classifier
    to infer watchdog history that is absent from an ordinary latent state.
    """
    if residual_probability.shape[-1] != 5:
        raise ValueError("residual event probability must contain five classes")
    if analytic_terminal.shape != residual_probability.shape[:-1] + (4,):
        raise ValueError(
            "analytic terminal must be [...,goal,boundary,crash,stuck]")
    if event_rewards.numel() != 6:
        raise ValueError(
            "event rewards must be human,static,goal,boundary,crash,stuck")
    probability = residual_probability.float()
    analytic = analytic_terminal.float()
    if not torch.isfinite(probability).all() or bool((probability < 0).any()):
        raise ValueError("residual probabilities must be finite/non-negative")
    probability_sum = probability.sum(-1, keepdim=True)
    if not torch.allclose(
        probability_sum, torch.ones_like(probability_sum),
        atol=1.0e-5, rtol=1.0e-5,
    ):
        raise ValueError("residual probabilities must form a closed distribution")
    if bool((analytic < 0).any()) or bool((analytic > 1).any()):
        raise ValueError("analytic terminal indicators must be in [0,1]")
    if bool((analytic.sum(-1) > 1.0 + 1.0e-6).any()):
        raise ValueError("analytic task terminal events must be exclusive")
    if not torch.allclose(
        probability[..., 3:], torch.zeros_like(probability[..., 3:]),
        atol=1.0e-7, rtol=0.0,
    ):
        raise ValueError("learned crash/stuck mass is forbidden")

    human = probability[..., 1:2]
    static = probability[..., 2:3]
    residual_mass = probability[..., 0:1]
    no_analytic = 1.0 - analytic.sum(-1, keepdim=True)
    goal = residual_mass * analytic[..., 0:1]
    boundary = residual_mass * analytic[..., 1:2]
    continuation = probability[..., 0:1] * no_analytic
    crash = residual_mass * analytic[..., 2:3]
    stuck = residual_mass * analytic[..., 3:4]
    rewards = event_rewards.to(probability)
    reward = (
        human * rewards[0]
        + static * rewards[1]
        + goal * rewards[2]
        + boundary * rewards[3]
        + crash * rewards[4]
        + stuck * rewards[5]
    )
    closed_sum = human + static + goal + boundary + crash + stuck + continuation
    return reward, continuation, {
        "human": human,
        "static": static,
        "goal": goal,
        "boundary": boundary,
        "crash": crash,
        "stuck": stuck,
        "continue": continuation,
        "probability_sum": closed_sum,
    }


def first_analytic_task_event(
    event_fraction: torch.Tensor,
) -> torch.Tensor:
    """Select the first task terminal with simulator-compatible tie priority.

    ``event_fraction`` is ordered ``goal, boundary, crash, stuck``.  Finite
    values are continuous transition fractions in ``[0,1]`` and ``inf`` means
    the event does not occur.  ``torch.min`` returns the first index on a tie,
    which exactly reproduces the simulator order: the main goal detector runs
    before the watchdog, whose checks are boundary, crash, then stuck.
    """
    if event_fraction.shape[-1] != 4:
        raise ValueError(
            "task event fractions must be [...,goal,boundary,crash,stuck]")
    fraction = event_fraction.float()
    if torch.isnan(fraction).any() or torch.isneginf(fraction).any():
        raise ValueError("task event fractions cannot contain NaN/-Inf")
    if bool((torch.isfinite(fraction) & (
        (fraction < 0.0) | (fraction > 1.0)
    )).any()):
        raise ValueError("finite task event fractions must lie in [0,1]")
    first_fraction, first_index = fraction.min(-1)
    selected = F.one_hot(first_index, num_classes=4).to(fraction)
    return selected * torch.isfinite(first_fraction)[..., None].to(fraction)
