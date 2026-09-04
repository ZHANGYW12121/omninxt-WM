from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.task_reward import (  # noqa: E402
    analytic_boundary_proximity_reward,
    analytic_fractional_progress_reward,
    analytic_fractional_smoothness_reward,
    analytic_human_clearance_reward,
    analytic_human_risk_reward,
    analytic_progress_reward,
    analytic_route_deviation_reward,
    analytic_smoothness_reward,
    compose_analytic_goal_boundary_event,
    compose_analytic_goal_event,
    compose_human_analytic_events,
    compose_human_analytic_competing_events,
    compose_swept_human_analytic_events,
    compose_uncertain_swept_human_analytic_events,
    continuous_residual_target,
    expected_event_reward,
    first_analytic_task_event,
    flight_bounds_potential_reward,
    route_heading_potential,
    human_event_time_quadrature,
    independent_human_contact_union_probability,
    cross_track_potential_reward,
    replace_human_event_probability,
    rescale_interval_probability,
    swept_human_contact_probability,
    terminal_aware_potential_reward,
    trajectory_correlated_human_contact_hazard,
)


class TaskRewardTest(unittest.TestCase):
    def test_human_event_quadrature_preserves_mass_and_nonlinear_moment(self):
        hazard = torch.tensor([[[1.6]]], requires_grad=True)
        stop = torch.tensor([[[0.75]]])
        fraction, mass_weight = human_event_time_quadrature(hazard, stop)
        human_mass = -torch.expm1(-hazard * stop)
        torch.testing.assert_close(
            mass_weight.sum(-2), human_mass, atol=1.0e-7, rtol=0.0)

        # Integral_0^s H exp(-H u) u^2 du. A nonlinear partial reward cannot
        # in general be replaced by P(event) * reward(E[u | event]).
        hs = hazard * stop
        expected_second_moment_mass = (
            2.0 / hazard.square()
            * (1.0 - torch.exp(-hs) * (1.0 + hs + 0.5 * hs.square()))
        )
        quadrature_second_moment_mass = (
            mass_weight * fraction.square()).sum(-2)
        torch.testing.assert_close(
            quadrature_second_moment_mass,
            expected_second_moment_mass,
            atol=2.0e-7,
            rtol=2.0e-6,
        )
        quadrature_second_moment_mass.sum().backward()
        self.assertTrue(torch.isfinite(hazard.grad).all())
        self.assertGreater(float(hazard.grad.abs().sum()), 0.0)

    def test_interval_probability_rescales_survival_not_probability(self):
        probability = torch.tensor([0.36], requires_grad=True)
        half = rescale_interval_probability(
            probability, torch.tensor([0.5]))
        double = rescale_interval_probability(
            probability, torch.tensor([2.0]))
        # Survival 0.64 becomes sqrt(.64)=.8 over half an interval and
        # .64**2 over two intervals.
        torch.testing.assert_close(half, torch.tensor([0.20]))
        torch.testing.assert_close(double, torch.tensor([0.5904]))
        (half + double).sum().backward()
        self.assertTrue(torch.isfinite(probability.grad).all())
        self.assertGreater(float(probability.grad.abs().sum()), 0.0)

    def test_continuous_human_hazard_stops_at_earlier_analytic_event(self):
        human = torch.tensor([[[0.36]]], requires_grad=True)
        analytic = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        reward, continuation, event = (
            compose_human_analytic_competing_events(
                human,
                analytic,
                torch.tensor([[[0.5]]]),
                human_collision_reward=-120.0,
                analytic_event_rewards=torch.tensor(
                    [100.0, -100.0, -100.0, -20.0]),
            )
        )
        # A nominal .36 hazard has .20 probability by half an interval. Goal
        # receives the surviving .80 mass; a whole-step priority rule would
        # incorrectly retain .36 Human probability here.
        torch.testing.assert_close(event["human"], torch.tensor([[[0.20]]]))
        torch.testing.assert_close(event["goal"], torch.tensor([[[0.80]]]))
        torch.testing.assert_close(continuation, torch.zeros_like(continuation))
        torch.testing.assert_close(
            event["probability_sum"],
            torch.ones_like(event["probability_sum"]),
        )
        torch.testing.assert_close(reward, torch.tensor([[[56.0]]]))
        reward.sum().backward()
        self.assertTrue(torch.isfinite(human.grad).all())
        self.assertGreater(float(human.grad.abs().sum()), 0.0)

    def test_swept_human_and_analytic_events_use_first_physical_fraction(self):
        human = torch.tensor([[[1.0]], [[1.0]], [[0.0]]])
        human_fraction = torch.tensor([[[0.25]], [[0.75]], [[1.0]]])
        analytic = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
        ])
        analytic_fraction = torch.tensor([[[0.5]], [[0.5]], [[1.0]]])
        reward, continuation, event = compose_swept_human_analytic_events(
            human, human_fraction, analytic, analytic_fraction,
            human_collision_reward=-120.0,
            analytic_event_rewards=torch.tensor(
                [100.0, -100.0, -100.0, -20.0]),
        )
        torch.testing.assert_close(
            event["human"], torch.tensor([[[1.0]], [[0.0]], [[0.0]]]))
        torch.testing.assert_close(
            event["goal"], torch.tensor([[[0.0]], [[1.0]], [[0.0]]]))
        torch.testing.assert_close(
            continuation, torch.tensor([[[0.0]], [[0.0]], [[1.0]]]))
        torch.testing.assert_close(
            event["expected_active_fraction"],
            torch.tensor([[[0.25]], [[0.5]], [[1.0]]]),
        )
        torch.testing.assert_close(
            reward, torch.tensor([[[-120.0]], [[100.0]], [[0.0]]]))
        torch.testing.assert_close(
            event["probability_sum"], torch.ones_like(continuation))

    def test_swept_contact_probability_is_calibrated_and_differentiable(self):
        gap = torch.tensor([0.03, 0.10, 0.17], requires_grad=True)
        probability = swept_human_contact_probability(
            gap, prediction_bias_m=0.03, prediction_scale_m=0.07)
        torch.testing.assert_close(
            probability[0], torch.tensor(0.5), atol=1.0e-7, rtol=0.0)
        self.assertTrue(bool((probability[:-1] > probability[1:]).all()))
        probability.sum().backward()
        self.assertTrue(torch.isfinite(gap.grad).all())
        self.assertTrue(bool((gap.grad < 0.0).all()))

    def test_swept_contact_probability_broadcasts_horizon_uncertainty(self):
        gap = torch.full((2, 3, 1), 0.10, requires_grad=True)
        bias = torch.tensor((0.03, 0.05, 0.07)).view(1, 3, 1)
        probability = swept_human_contact_probability(
            gap, prediction_bias_m=bias, prediction_scale_m=0.07)
        self.assertEqual(probability.shape, gap.shape)
        self.assertTrue(bool(
            (probability[:, 1:] > probability[:, :-1]).all()))
        probability.sum().backward()
        self.assertTrue(torch.isfinite(gap.grad).all())
        self.assertTrue(bool((gap.grad < 0.0).all()))

    def test_independent_human_contact_union_keeps_every_slot_gradient(self):
        slot = torch.tensor(
            [[[0.20, 0.30, 0.90], [0.10, 0.40, 0.80]]],
            requires_grad=True,
        )
        valid = torch.tensor(
            [[[True, True, False], [False, False, False]]])
        union = independent_human_contact_union_probability(slot, valid)
        torch.testing.assert_close(
            union, torch.tensor([[[0.44], [0.0]]]),
            atol=1.0e-7, rtol=0.0,
        )
        union.sum().backward()
        torch.testing.assert_close(
            slot.grad,
            torch.tensor([[[0.70, 0.80, 0.0], [0.0, 0.0, 0.0]]]),
            atol=1.0e-7, rtol=0.0,
        )

    def test_trajectory_correlated_hazard_does_not_repeat_persistent_near_miss(self):
        slot = torch.tensor(
            [[[0.20, 0.30], [0.20, 0.30], [0.40, 0.10]]],
            requires_grad=True,
        )
        valid = torch.ones_like(slot, dtype=torch.bool)
        hazard = trajectory_correlated_human_contact_hazard(slot, valid)
        torch.testing.assert_close(
            hazard,
            torch.tensor([[[0.44], [0.0], [0.25]]]),
            atol=1.0e-6, rtol=0.0,
        )
        cumulative = 1.0 - torch.prod(1.0 - hazard, dim=-2)
        torch.testing.assert_close(
            cumulative, torch.tensor([[0.58]]), atol=1.0e-6, rtol=0.0)
        cumulative.sum().backward()
        self.assertTrue(torch.isfinite(slot.grad).all())
        self.assertGreater(float(slot.grad.abs().sum()), 0.0)

    def test_trajectory_correlated_hazard_masks_invalid_slots(self):
        slot = torch.tensor([[[0.10, 0.99], [0.30, 0.99]]])
        valid = torch.tensor([[[True, False], [True, False]]])
        hazard = trajectory_correlated_human_contact_hazard(slot, valid)
        torch.testing.assert_close(
            hazard,
            torch.tensor([[[0.10], [2.0 / 9.0]]]),
            atol=1.0e-6, rtol=0.0,
        )

    def test_uncertain_swept_contact_preserves_exact_order_and_gap_gradient(self):
        gap = torch.tensor([[[0.058]], [[-0.01]]], requires_grad=True)
        probability = swept_human_contact_probability(
            gap, prediction_bias_m=0.03, prediction_scale_m=0.07)
        exact = torch.tensor([[[0.0]], [[1.0]]])
        exact_fraction = torch.tensor([[[1.0]], [[0.25]]])
        analytic = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
        ])
        reward, continuation, event = (
            compose_uncertain_swept_human_analytic_events(
                exact,
                exact_fraction,
                probability,
                analytic,
                torch.tensor([[[0.5]], [[0.5]]]),
                human_collision_reward=-120.0,
                analytic_event_rewards=torch.tensor(
                    [100.0, -100.0, -100.0, -20.0]),
            ))
        # gap=0.058 gives nominal p(sigmoid(-0.4)) ~= .4013, whose half-step
        # competing hazard is 1-sqrt(1-p). The exact second row still uses the
        # literal 0.25 contact and wins over the goal at 0.5.
        expected_human = 1.0 - torch.sqrt(1.0 - probability[0])
        torch.testing.assert_close(event["human"][0], expected_human)
        torch.testing.assert_close(event["human"][1], torch.ones((1, 1)))
        torch.testing.assert_close(event["goal"][1], torch.zeros((1, 1)))
        torch.testing.assert_close(
            event["probability_sum"], torch.ones_like(continuation))
        reward.sum().backward()
        self.assertTrue(torch.isfinite(gap.grad).all())
        self.assertGreater(float(gap.grad[0]), 0.0)
        # Exact-contact ordering is intentionally independent of the soft
        # uncertainty branch.
        self.assertEqual(float(gap.grad[1]), 0.0)

    def test_first_task_event_uses_time_then_simulator_tie_priority(self):
        fractions = torch.tensor([
            [0.8, float("inf"), 0.2, float("inf")],
            [0.4, 0.4, 0.4, 0.4],
            [float("inf"), 0.3, 0.3, 0.3],
            [float("inf"), float("inf"), float("inf"), float("inf")],
        ])
        expected = torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        torch.testing.assert_close(first_analytic_task_event(fractions), expected)

    def test_residual_subtracts_weighted_progress_and_event_once(self):
        reward = torch.tensor([[[8.0], [-5.0]]])
        components = torch.zeros(1, 2, 6)
        components[..., 0] = torch.tensor([[10.0, -4.0]])
        components[..., 1] = torch.tensor([[2.0, -2.0]])
        expected = torch.tensor([[[-4.0], [1.0]]])
        torch.testing.assert_close(
            continuous_residual_target(reward, components), expected)

    def test_progress_is_written_on_destination_row_and_uses_dynamic_dt_clip(self):
        ego = torch.zeros(1, 3, 14)
        ego[0, 1, 0] = 1.0
        ego[0, 2, 0] = -1.0
        goal = torch.tensor([[10.0, 0.0, 0.0]])
        reward = analytic_progress_reward(
            ego, goal, dt_s=0.1,
            progress_weight_per_m=2.0,
            max_progress_speed_mps=3.0,
        )
        # Raw +/- displacement exceeds 0.3 m and is clipped before the 2x
        # reward weight. Index zero remains context-only.
        torch.testing.assert_close(
            reward, torch.tensor([[[0.0], [0.6], [-0.6]]]))

    def test_fractional_terminal_progress_uses_event_position_and_partial_dt(self):
        source = torch.zeros(1, 1, 14)
        destination = source.clone()
        destination[..., 0] = 2.0
        goal = torch.tensor([[10.0, 0.0, 0.0]])
        reward = analytic_fractional_progress_reward(
            source, destination, goal, torch.tensor([[[0.25]]]),
            dt_s=0.1,
            progress_weight_per_m=2.0,
            max_progress_speed_mps=3.0,
        )
        # Event position advances .5 m, but the recorder clip over .025 s is
        # .075 m; the 2 reward/metre coefficient therefore yields .15.
        torch.testing.assert_close(reward, torch.tensor([[[0.15]]]))

    def test_fractional_terminal_smoothness_recomputes_partial_jerk(self):
        memory = torch.zeros(1, 1, 9)
        source = torch.zeros(1, 1, 14)
        source[..., 13] = 1.0
        event = source.clone()
        event[..., 6] = 4.0
        angle = torch.tensor(0.05)
        event[..., 12] = angle.sin()
        event[..., 13] = angle.cos()
        fraction = torch.tensor([[[0.25]]])
        reward, state = analytic_fractional_smoothness_reward(
            memory, source, event, fraction,
            dt_s=0.1,
            acceleration_filter_alpha=0.25,
            acceleration_weight_per_sec=0.15,
            acceleration_scale_mps2=3.0,
            jerk_weight_per_sec=0.10,
            jerk_scale_mps3=10.0,
            yaw_rate_weight_per_sec=0.05,
            yaw_rate_scale_rps=1.5,
            term_clip=4.0,
        )
        # Filtered acceleration is 1 m/s^2. Over 25 ms its change is
        # 40 m/s^3, so the jerk term clips at four; merely taking one quarter
        # of the 100 ms reward would incorrectly use a 10 m/s^3 jerk.
        torch.testing.assert_close(
            state["jerk_norm_mps3"], torch.tensor([[[40.0]]]),
            atol=1.0e-5, rtol=0.0)
        expected = -0.025 * (
            0.15 * (1.0 / 3.0) ** 2
            + 0.10 * 4.0
            + 0.05 * (2.0 / 1.5) ** 2)
        torch.testing.assert_close(
            reward, torch.tensor([[[expected]]]), atol=1.0e-6, rtol=0.0)

    def test_full_fractional_smoothness_matches_full_step_formula(self):
        memory = torch.zeros(1, 1, 9)
        source = torch.zeros(1, 1, 14)
        source[..., 13] = 1.0
        destination = source.clone()
        destination[..., 6] = 4.0
        angle = torch.tensor(0.20)
        destination[..., 12] = angle.sin()
        destination[..., 13] = angle.cos()
        destination_memory = memory.clone()
        destination_memory[..., 4] = 1.0
        full, _ = analytic_smoothness_reward(
            memory, destination_memory, source, destination,
            dt_s=0.1,
            acceleration_weight_per_sec=0.15,
            acceleration_scale_mps2=3.0,
            jerk_weight_per_sec=0.10,
            jerk_scale_mps3=10.0,
            yaw_rate_weight_per_sec=0.05,
            yaw_rate_scale_rps=1.5,
            term_clip=4.0,
        )
        fractional, _ = analytic_fractional_smoothness_reward(
            memory, source, destination, torch.ones(1, 1, 1),
            dt_s=0.1,
            acceleration_filter_alpha=0.25,
            acceleration_weight_per_sec=0.15,
            acceleration_scale_mps2=3.0,
            jerk_weight_per_sec=0.10,
            jerk_scale_mps3=10.0,
            yaw_rate_weight_per_sec=0.05,
            yaw_rate_scale_rps=1.5,
            term_clip=4.0,
        )
        torch.testing.assert_close(fractional, full)

    def test_event_reward_uses_probability_without_a_decision_threshold(self):
        probability = torch.tensor([[[0.5, 0.2, 0.1, 0.1, 0.1]]])
        rewards = torch.tensor([0.0, -120.0, -120.0, 100.0, -100.0])
        torch.testing.assert_close(
            expected_event_reward(probability, rewards),
            torch.tensor([[[-36.0]]]),
        )

    def test_explicit_human_probability_replaces_instead_of_duplicates_event(self):
        original = torch.tensor([[[0.50, 0.20, 0.10, 0.10, 0.10]]])
        combined = replace_human_event_probability(
            original, torch.tensor([[[0.40]]]))
        torch.testing.assert_close(
            combined.sum(-1), torch.ones_like(combined[..., 0]))
        torch.testing.assert_close(combined[..., 1:2], torch.tensor([[[0.40]]]))
        # Non-Human conditional proportions 5:1:1:1 are preserved in the
        # remaining 0.60 probability mass.
        torch.testing.assert_close(
            combined[..., (0, 2, 3, 4)],
            torch.tensor([[[0.375, 0.075, 0.075, 0.075]]]),
        )

    def test_cross_track_potential_has_tolerance_and_return_gradient(self):
        ego = torch.zeros(1, 4, 14, requires_grad=True)
        with torch.no_grad():
            ego[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
            ego[0, :, 1] = torch.tensor([0.0, 3.0, 5.0, 4.0])
        reward, distance = cross_track_potential_reward(
            ego, torch.tensor([[10.0, 0.0, 0.0]]),
            discount=1.0, tolerance_m=4.0, potential_scale=1.0)
        torch.testing.assert_close(
            distance[..., 0], torch.tensor([[0.0, 3.0, 5.0, 4.0]]))
        torch.testing.assert_close(
            reward[..., 0], torch.tensor([[0.0, 0.0, -1.0, 1.0]]))
        reward[0, 2, 0].backward()
        self.assertGreater(float(ego.grad[0, 2, 1].abs()), 0.0)

    def test_route_potential_is_left_right_symmetric(self):
        ego = torch.zeros(2, 2, 14, requires_grad=True)
        with torch.no_grad():
            ego[:, 1, 0] = 1.0
            ego[0, 1, 1] = 2.0
            ego[1, 1, 1] = -2.0
        reward, distance = cross_track_potential_reward(
            ego,
            torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            discount=0.997,
            tolerance_m=1.21,
            potential_scale=2.0,
        )
        torch.testing.assert_close(distance[0], distance[1])
        torch.testing.assert_close(reward[0], reward[1])
        gradient, = torch.autograd.grad(reward[:, 1].sum(), ego)
        torch.testing.assert_close(
            gradient[0, 1, 1], -gradient[1, 1, 1])

    def test_route_deviation_is_dense_symmetric_and_inward(self):
        ego = torch.zeros(2, 3, 14, requires_grad=True)
        with torch.no_grad():
            ego[0, :, 1] = torch.tensor([0.5, 2.0, 2.0])
            ego[1, :, 1] = torch.tensor([-0.5, -2.0, -2.0])
        goal = torch.tensor(
            [[[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])
        reward, distance = analytic_route_deviation_reward(
            ego, goal,
            tolerance_m=1.21,
            weight_per_m_per_sec=2.0,
            dt_s=0.1,
        )
        torch.testing.assert_close(distance[0], distance[1])
        torch.testing.assert_close(reward[0], reward[1])
        torch.testing.assert_close(reward[:, 0], torch.zeros(2, 1))
        torch.testing.assert_close(
            reward[:, 1:, 0], torch.full((2, 2), -0.158),
            atol=1.0e-6, rtol=0.0)
        gradient, = torch.autograd.grad(reward[:, 1].sum(), ego)
        self.assertLess(float(gradient[0, 1, 1]), 0.0)
        self.assertGreater(float(gradient[1, 1, 1]), 0.0)

    def test_route_deviation_uses_partial_terminal_exposure(self):
        ego = torch.zeros(1, 2, 14)
        ego[..., 1] = 2.0
        goal = torch.tensor([[[10.0, 0.0, 0.0]]])
        full, _ = analytic_route_deviation_reward(
            ego, goal, tolerance_m=1.21,
            weight_per_m_per_sec=2.0, dt_s=0.1)
        partial, _ = analytic_route_deviation_reward(
            ego, goal, tolerance_m=1.21,
            weight_per_m_per_sec=2.0,
            dt_s=torch.tensor([[[0.025], [0.075]]]),
        )
        torch.testing.assert_close(
            partial, full * torch.tensor([[[0.25], [0.75]]]))

    def test_route_deviation_penalizes_persistent_offset_not_safe_avoidance(self):
        ego = torch.zeros(2, 4, 14)
        ego[0, :, 1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        ego[1, :, 1] = torch.tensor([2.0, 1.7, 1.4, 1.1])
        reward, _ = analytic_route_deviation_reward(
            ego, torch.tensor([[[10.0, 0.0, 0.0]]]),
            tolerance_m=1.21,
            weight_per_m_per_sec=2.0,
            dt_s=0.1,
        )
        torch.testing.assert_close(reward[0], torch.zeros_like(reward[0]))
        self.assertLess(float(reward[1].detach().sum()), 0.0)
        self.assertEqual(float(reward[1, -1]), 0.0)

    def test_route_heading_potential_is_bounded_symmetric_and_directional(self):
        yaw = torch.tensor(
            [[0.0, 0.0], [0.0, torch.pi / 2], [0.0, -torch.pi / 2]],
            requires_grad=True,
        )
        ego = torch.cat((
            torch.zeros(3, 2, 12),
            yaw.sin()[..., None],
            yaw.cos()[..., None],
        ), -1)
        goal = torch.tensor([[10.0, 0.0, 0.0]]).expand(3, -1)
        potential, error = route_heading_potential(
            ego, goal, potential_scale=8.0)
        torch.testing.assert_close(
            potential[:, 0, 0], torch.zeros(3))
        torch.testing.assert_close(potential[0, 1, 0], torch.tensor(0.0))
        torch.testing.assert_close(
            potential[1:, 1, 0], torch.tensor([-8.0, -8.0]))
        torch.testing.assert_close(
            error[1:, 1, 0], torch.full((2,), torch.pi / 2))
        reward = terminal_aware_potential_reward(
            potential, torch.ones(3, 1, 1), discount=1.0)
        torch.testing.assert_close(
            reward[:, 0, 0], torch.tensor([0.0, -8.0, -8.0]))
        gradient, = torch.autograd.grad(reward[1:, 0].sum(), yaw)
        self.assertLess(float(gradient[1, 1]), 0.0)
        self.assertGreater(float(gradient[2, 1]), 0.0)

    def test_route_heading_potential_ignores_zero_length_route(self):
        ego = torch.zeros(1, 2, 14)
        ego[..., 12] = 1.0
        potential, error = route_heading_potential(
            ego, torch.zeros(1, 3), potential_scale=8.0)
        torch.testing.assert_close(potential, torch.zeros_like(potential))
        torch.testing.assert_close(error, torch.zeros_like(error))

    def test_route_heading_does_not_turn_toward_off_axis_point_goal(self):
        ego = torch.zeros(1, 3, 14)
        ego[..., 13] = 1.0
        # The vehicle stays aligned with the episode X route while moving
        # cross-track and even beyond the point goal. A point-goal bearing
        # would rotate sharply here; the holonomic route potential must not.
        ego[0, :, :2] = torch.tensor((
            (0.0, 0.0), (8.0, 3.0), (12.0, -2.0)))
        potential, error = route_heading_potential(
            ego, torch.tensor([[10.0, 0.0, 0.0]]), potential_scale=8.0)
        torch.testing.assert_close(potential, torch.zeros_like(potential))
        torch.testing.assert_close(error, torch.zeros_like(error))

    def test_flight_bound_potential_has_derived_margin_and_inward_gradient(self):
        signed = torch.tensor(
            [[[4.0], [3.0], [3.5]], [[4.0], [3.0], [3.5]]],
            requires_grad=True)
        reward, excess = flight_bounds_potential_reward(
            signed, discount=1.0,
            lookahead_margin_m=3.225, potential_scale=2.0)
        torch.testing.assert_close(
            excess[..., 0],
            torch.tensor([[0.0, 0.225, 0.0], [0.0, 0.225, 0.0]]),
            atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(reward[0], reward[1])
        # Moving toward the wall is negative; moving back inward recovers the
        # same discounted potential without choosing a left/right direction.
        torch.testing.assert_close(
            reward[0, :, 0], torch.tensor([0.0, -0.45, 0.45]),
            atol=1.0e-6, rtol=0.0)
        gradient, = torch.autograd.grad(reward[:, 1].sum(), signed)
        self.assertGreater(float(gradient[:, 1].sum()), 0.0)

    def test_boundary_proximity_cost_is_nonrefundable_and_terminal_safe(self):
        signed = torch.tensor(
            [[[4.0], [3.0], [2.0]], [[4.0], [3.0], [2.0]]],
            requires_grad=True,
        )
        continuation = torch.tensor(
            [[[1.0], [1.0]], [[1.0], [0.0]]])
        reward, excess = analytic_boundary_proximity_reward(
            signed,
            continuation,
            lookahead_margin_m=4.0,
            weight_per_second=2.0,
            dt_s=0.1,
        )
        torch.testing.assert_close(
            excess[..., 0], torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
        torch.testing.assert_close(
            reward[..., 0],
            torch.tensor([[-0.0125, -0.05], [-0.0125, 0.0]]),
        )
        # An absorbing terminal never refunds the earlier negative cost.
        self.assertLess(float(reward[1].detach().sum()), 0.0)
        gradient, = torch.autograd.grad(reward.sum(), signed)
        self.assertGreater(float(gradient[:, 1:, 0].sum()), 0.0)

    def test_terminal_potential_uses_absorbing_zero_successor(self):
        potential = torch.tensor([[[-1.0], [-3.0], [-4.0]]])
        continuation = torch.tensor([[[1.0], [0.0]]])
        reward = terminal_aware_potential_reward(
            potential, continuation, discount=0.9)
        # Nonterminal: .9*(-3)-(-1)=-1.7. Terminal: Phi(successor)=0,
        # so the reward is 0-(-3)=+3 rather than .9*(-4)-(-3)=-.6.
        torch.testing.assert_close(
            reward, torch.tensor([[[-1.7], [3.0]]]))

    def test_human_clearance_reward_matches_simulator_instantaneous_scale(self):
        clearance = torch.tensor([0.10, 0.40, 0.70])
        reward = analytic_human_clearance_reward(
            clearance,
            hard_clearance_m=0.10,
            safe_clearance_m=0.70,
            weight_per_second=4.0,
            dt_s=0.10,
        )
        torch.testing.assert_close(
            reward, torch.tensor([-0.40, -0.10, 0.0]))

    def test_human_dense_risk_uses_imagined_existence_probability(self):
        reward, state = analytic_human_risk_reward(
            torch.tensor([[0.10, 0.70]]),
            torch.tensor([[[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]]),
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            human_presence=torch.tensor([[0.25, 1.0]]),
            hard_clearance_m=0.10,
            safe_clearance_m=0.70,
            predictive_safe_clearance_m=0.90,
            predictive_horizon_s=2.5,
            corridor_half_width_m=0.75,
            corridor_lookahead_m=4.0,
            drone_radius_m=0.17,
            pelvis_radius_m=0.14,
            weight_per_second=4.0,
            dt_s=0.10,
        )
        torch.testing.assert_close(reward, torch.tensor([[-0.10]]))
        torch.testing.assert_close(
            state["combined_risk"], torch.tensor([[0.25]]))

    def test_front_corridor_is_diagnostic_not_a_proximity_slowdown(self):
        reward, state = analytic_human_risk_reward(
            torch.tensor([[1.50]]),
            torch.tensor([[[2.0, 0.0, 0.0]]]),
            torch.zeros(1, 1, 3),
            torch.ones(1, 1, dtype=torch.bool),
            ego_velocity_body=torch.zeros(1, 3),
            hard_clearance_m=0.10,
            safe_clearance_m=0.70,
            predictive_safe_clearance_m=0.90,
            predictive_horizon_s=2.5,
            corridor_half_width_m=0.75,
            corridor_lookahead_m=4.0,
            drone_radius_m=0.17,
            pelvis_radius_m=0.14,
            weight_per_second=4.0,
            dt_s=0.10,
        )
        self.assertGreater(float(state["corridor_risk"]), 0.0)
        torch.testing.assert_close(state["combined_risk"], torch.zeros(1, 1))
        torch.testing.assert_close(reward, torch.zeros(1, 1))

    def test_predictive_risk_is_symmetric_for_a_rear_closing_human(self):
        reward, state = analytic_human_risk_reward(
            torch.tensor([[1.0]]),
            torch.tensor([[[-1.0, 0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0, 0.0]]]),
            torch.ones(1, 1, dtype=torch.bool),
            ego_velocity_body=torch.zeros(1, 3),
            hard_clearance_m=0.10,
            safe_clearance_m=0.70,
            predictive_safe_clearance_m=0.90,
            predictive_horizon_s=2.5,
            corridor_half_width_m=0.75,
            corridor_lookahead_m=4.0,
            drone_radius_m=0.17,
            pelvis_radius_m=0.14,
            weight_per_second=4.0,
            dt_s=0.10,
        )
        self.assertGreater(float(state["predictive_risk"]), 0.0)
        torch.testing.assert_close(
            state["combined_risk"], state["predictive_risk"])
        self.assertLess(float(reward), 0.0)

    def test_crossing_risk_can_prefer_safe_acceleration(self):
        common = {
            "per_human_signed_clearance_m": torch.tensor([[2.0]]),
            "human_root_body": torch.tensor([[[0.5, -1.5, 0.0]]]),
            "human_velocity_body": torch.tensor([[[0.0, 1.0, 0.0]]]),
            "human_valid": torch.ones(1, 1, dtype=torch.bool),
            "hard_clearance_m": 0.10,
            "safe_clearance_m": 0.70,
            "predictive_safe_clearance_m": 0.90,
            "predictive_horizon_s": 2.5,
            "corridor_half_width_m": 0.75,
            "corridor_lookahead_m": 4.0,
            "drone_radius_m": 0.17,
            "pelvis_radius_m": 0.14,
            "weight_per_second": 4.0,
            "dt_s": 0.10,
        }
        slow_reward, slow = analytic_human_risk_reward(
            **common, ego_velocity_body=torch.tensor([[0.2, 0.0, 0.0]]))
        fast_reward, fast = analytic_human_risk_reward(
            **common, ego_velocity_body=torch.tensor([[3.0, 0.0, 0.0]]))
        self.assertGreater(float(slow["predictive_risk"]), 0.0)
        torch.testing.assert_close(
            fast["predictive_risk"], torch.zeros(1, 1))
        self.assertGreater(float(fast_reward), float(slow_reward))

    def test_analytic_goal_uses_only_continue_mass_and_keeps_collision_priority(self):
        probability = torch.tensor([[
            [0.7, 0.2, 0.1, 0.0, 0.0],
            [0.6, 0.3, 0.0, 0.05, 0.05],
        ]])
        distance = torch.tensor([[[0.8], [1.2]]])
        reward, continuation, goal = compose_analytic_goal_event(
            probability,
            distance,
            non_goal_event_rewards=torch.tensor(
                [0.0, -120.0, -120.0, -100.0, -20.0]),
            goal_radius_m=1.0,
            goal_reward=100.0,
        )
        # Inside the goal, only the 0.7 continue mass becomes goal.  Human and
        # static collision mass stays terminal and retains its penalty.
        torch.testing.assert_close(goal, torch.tensor([[[0.7], [0.0]]]))
        torch.testing.assert_close(
            continuation, torch.tensor([[[0.0], [0.6]]]))
        torch.testing.assert_close(
            reward, torch.tensor([[[34.0], [-42.0]]]))

    def test_exact_goal_and_boundary_consume_continue_mass_and_stay_closed(self):
        # continue, Human, static, other -- already conditioned on non-goal.
        probability = torch.tensor([[
            [0.70, 0.20, 0.05, 0.05],
            [0.80, 0.10, 0.05, 0.05],
            [0.60, 0.20, 0.10, 0.10],
        ]])
        distance = torch.tensor([[[0.5], [2.0], [0.5]]])
        signed_boundary = torch.tensor([[[1.0], [-0.1], [-0.2]]])
        reward, continuation, event = compose_analytic_goal_boundary_event(
            probability,
            distance,
            signed_boundary,
            non_goal_event_rewards=torch.tensor(
                [0.0, -120.0, -120.0, -100.0]),
            goal_radius_m=torch.tensor([[1.0]]),
            goal_reward=100.0,
            boundary_reward=-100.0,
        )
        # Human/static collision mass is untouched. Goal consumes all
        # remaining non-collision mass and has priority over bounds/other,
        # matching pegasus_app._detect_recording_event before its watchdog.
        torch.testing.assert_close(
            event["goal"], torch.tensor([[[0.75], [0.0], [0.70]]]))
        torch.testing.assert_close(
            event["boundary"], torch.tensor([[[0.0], [0.80], [0.0]]]))
        torch.testing.assert_close(continuation, torch.zeros_like(continuation))
        torch.testing.assert_close(
            event["probability_sum"],
            torch.ones_like(event["probability_sum"]),
        )
        torch.testing.assert_close(
            reward, torch.tensor([[[45.0], [-103.0], [34.0]]]))

    def test_v16_human_and_analytic_terminals_form_closed_distribution(self):
        human = torch.tensor([[[0.25], [0.10]]])
        # goal / out-of-bounds / reliable static are mutually exclusive.
        analytic = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]])
        reward, continuation, probability = compose_human_analytic_events(
            human,
            analytic,
            human_collision_reward=-120.0,
            analytic_event_rewards=torch.tensor([100.0, -100.0, -120.0]),
        )
        # Human owns 0.25 and analytic goal receives the remaining 0.75.
        torch.testing.assert_close(reward[0, 0], torch.tensor([45.0]))
        # Human owns 0.10 and static receives the remaining 0.90.
        torch.testing.assert_close(reward[0, 1], torch.tensor([-120.0]))
        torch.testing.assert_close(continuation, torch.zeros_like(continuation))
        torch.testing.assert_close(
            probability["probability_sum"],
            torch.ones_like(probability["probability_sum"]),
        )

    def test_v16_contract_rejects_overlapping_analytic_events(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            compose_human_analytic_events(
                torch.zeros(1, 1),
                torch.tensor([[1.0, 1.0]]),
                human_collision_reward=-120.0,
                analytic_event_rewards=torch.tensor([100.0, -100.0]),
            )


if __name__ == "__main__":
    unittest.main()
