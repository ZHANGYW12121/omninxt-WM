from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.action_smoother import (  # noqa: E402
    AFFINE_REACHABLE_INTERVAL,
    ActionSmootherConfig,
    REACHABLE_INTERVAL_FRACTION,
    SOFT_ABSOLUTE_TARGET,
    StatefulActionSmoother,
)


class StatefulActionSmootherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.smoother = StatefulActionSmoother(ActionSmootherConfig(
            time_step_s=0.1,
            time_constant_s=0.35,
            slew_rate_per_s=(1.2, 1.0, 0.45),
        ))

    def test_limits_each_applied_step(self) -> None:
        previous = torch.zeros(1, 3)
        raw = torch.tensor([[1.0, -1.0, 1.0]])
        applied = self.smoother(raw, previous)
        torch.testing.assert_close(
            applied,
            torch.tensor([[0.12, -0.10, 0.045]]),
        )
        second = self.smoother(raw, applied)
        self.assertTrue(bool((second.abs() > applied.abs()).all()))
        self.assertTrue(bool((second - applied).abs().le(
            self.smoother.maximum_step_delta + 1.0e-7).all()))

    def test_previous_action_is_explicit_state(self) -> None:
        raw = torch.zeros(2, 3)
        previous = torch.tensor([
            [0.6, 0.0, 0.0],
            [-0.6, 0.0, 0.0],
        ])
        applied = self.smoother(raw, previous)
        self.assertGreater(float(applied[0, 0]), 0.0)
        self.assertLess(float(applied[1, 0]), 0.0)

    def test_filter_preserves_raw_target_gradient(self) -> None:
        raw = torch.tensor([[0.1, -0.1, 0.02]], requires_grad=True)
        applied = self.smoother(raw, torch.zeros_like(raw))
        applied.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertTrue(bool(raw.grad.abs().gt(0.0).all()))

    def test_inverse_reconstructs_reachable_applied_target(self) -> None:
        previous = torch.tensor([[0.20, -0.15, 0.04]])
        desired = torch.tensor([[0.25, -0.20, 0.05]])
        raw, reachable = self.smoother.raw_target_for_smoothed(
            desired, previous)
        self.assertTrue(bool(reachable.item()))
        torch.testing.assert_close(
            self.smoother(raw, previous), desired,
            atol=2.0e-5, rtol=0.0)
        self.assertFalse(torch.equal(raw, desired))

    def test_inverse_rejects_transition_beyond_current_slew_limit(self) -> None:
        previous = torch.zeros(1, 3)
        desired = torch.tensor([[0.121, 0.0, 0.0]])
        raw, reachable = self.smoother.raw_target_for_smoothed(
            desired, previous)
        self.assertFalse(bool(reachable.item()))
        self.assertLess(float(self.smoother(raw, previous)[0, 0]), 0.121)

    def test_inverse_accepts_exact_slew_limited_boundary(self) -> None:
        previous = torch.zeros(1, 3)
        desired = torch.tensor([[0.12, -0.10, 0.045]])
        raw, reachable = self.smoother.raw_target_for_smoothed(
            desired, previous)
        self.assertTrue(bool(reachable.item()))
        torch.testing.assert_close(
            self.smoother(raw, previous), desired,
            atol=2.0e-5, rtol=0.0)

    def test_reachable_fraction_has_no_interior_hard_clamp_dead_zone(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=REACHABLE_INTERVAL_FRACTION))
        raw = torch.tensor(
            [[0.99, -0.99, 0.75], [-0.80, 0.80, -0.50]],
            requires_grad=True,
        )
        previous = torch.tensor(
            [[0.8, -0.8, 0.9], [-0.9, 0.9, -0.9]])
        applied = smoother(raw, previous)
        applied.sum().backward()
        self.assertTrue(bool(raw.grad.abs().gt(0.0).all()))
        self.assertTrue(bool(applied.abs().le(1.0 + 1.0e-7).all()))
        self.assertTrue(bool((applied - previous).abs().le(
            smoother.maximum_step_delta + 1.0e-7).all()))

    def test_reachable_fraction_zero_holds_and_inverse_is_exact(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=REACHABLE_INTERVAL_FRACTION))
        previous = torch.tensor([[0.95, -0.95, 0.25]])
        torch.testing.assert_close(smoother(torch.zeros_like(previous), previous),
                                   previous)
        desired = torch.tensor([[0.98, -0.90, 0.22]])
        raw, reachable = smoother.raw_target_for_smoothed(desired, previous)
        self.assertTrue(bool(reachable.item()))
        torch.testing.assert_close(
            smoother(raw, previous), desired, atol=2.0e-5, rtol=0.0)
        self.assertTrue(bool(raw.abs().le(1.0).all()))

    def test_affine_interval_reverses_exact_command_bounds_with_gradient(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=AFFINE_REACHABLE_INTERVAL))
        previous = torch.tensor([[1.0, -1.0, 1.0]])
        raw = torch.tensor([[-1.0, 1.0, -1.0]], requires_grad=True)
        applied = smoother(raw, previous)
        torch.testing.assert_close(
            applied, torch.tensor([[0.88, -0.90, 0.955]]),
            atol=1.0e-7, rtol=0.0)
        applied.sum().backward()
        expected_scale = torch.tensor([[0.06, 0.05, 0.0225]])
        torch.testing.assert_close(raw.grad, expected_scale)
        torch.testing.assert_close(
            smoother.local_action_scale(raw.detach(), previous),
            expected_scale)

    def test_affine_interval_is_bijective_over_complete_reachable_interval(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=AFFINE_REACHABLE_INTERVAL))
        previous = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.95, -0.95, 1.0],
            [-1.0, 1.0, -0.98],
        ])
        desired = torch.tensor([
            [0.08, -0.04, 0.02],
            [0.90, -0.99, 0.98],
            [-0.94, 0.93, -1.0],
        ])
        raw, reachable = smoother.raw_target_for_smoothed(desired, previous)
        self.assertTrue(bool(reachable.all()))
        self.assertTrue(bool(raw.abs().le(1.0).all()))
        torch.testing.assert_close(
            smoother(raw, previous), desired, atol=2.0e-5, rtol=0.0)

    def test_affine_interval_rejects_target_outside_reachable_interval(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=AFFINE_REACHABLE_INTERVAL))
        previous = torch.tensor([[1.0, -1.0, 1.0]])
        desired = torch.tensor([[0.879, -0.899, 0.954]])
        _, reachable = smoother.raw_target_for_smoothed(desired, previous)
        self.assertFalse(bool(reachable.item()))

    def test_soft_absolute_zero_mean_reverts_existing_command(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=SOFT_ABSOLUTE_TARGET))
        previous = torch.tensor([[0.8, -0.7, 0.6]])
        magnitudes = []
        for _ in range(20):
            previous = smoother(torch.zeros_like(previous), previous)
            magnitudes.append(previous.abs().clone())
        self.assertTrue(bool((magnitudes[-1] < magnitudes[0]).all()))
        self.assertTrue(bool((magnitudes[-1] < 0.1).all()))

    def test_soft_absolute_output_is_between_previous_and_target(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=SOFT_ABSOLUTE_TARGET))
        previous = torch.tensor([
            [1.0, -1.0, 0.7],
            [-0.8, 0.9, -1.0],
        ])
        target = torch.tensor([
            [-1.0, 1.0, -0.9],
            [0.7, -0.6, 1.0],
        ], requires_grad=True)
        output = smoother(target, previous)
        lower = torch.minimum(previous, target.detach())
        upper = torch.maximum(previous, target.detach())
        self.assertTrue(bool((output >= lower).all()))
        self.assertTrue(bool((output <= upper).all()))
        self.assertTrue(bool((output - previous).abs().lt(
            smoother.maximum_step_delta).all()))
        output.sum().backward()
        self.assertTrue(bool(target.grad.gt(0.0).all()))
        torch.testing.assert_close(
            target.grad,
            smoother.local_action_scale(target.detach(), previous),
            atol=1.0e-7, rtol=1.0e-6,
        )

    def test_soft_absolute_inverse_reconstructs_reachable_target(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=SOFT_ABSOLUTE_TARGET))
        previous = torch.tensor([[0.20, -0.15, 0.04]])
        raw_truth = torch.tensor([[-0.10, 0.25, -0.20]])
        desired = smoother(raw_truth, previous)
        raw, reachable = smoother.raw_target_for_smoothed(
            desired, previous)
        self.assertTrue(bool(reachable.item()))
        torch.testing.assert_close(raw, raw_truth, atol=2.0e-5, rtol=0.0)
        torch.testing.assert_close(
            smoother(raw, previous), desired, atol=2.0e-5, rtol=0.0)

    def test_soft_absolute_stochastic_commands_do_not_random_walk(self):
        smoother = StatefulActionSmoother(ActionSmootherConfig(
            parameterization=SOFT_ABSOLUTE_TARGET))
        generator = torch.Generator().manual_seed(17)
        batch = 4096
        command = torch.zeros(batch, 3)
        for _ in range(200):
            target = torch.randn(batch, 3, generator=generator) * 0.35
            command = smoother(target.clamp(-1.0, 1.0), command)
        # A stationary zero-mean absolute target has a stationary command
        # distribution.  The historical affine-increment map grows toward the
        # command bounds over the same 200 samples.
        self.assertLess(float(command.mean(0).abs().max()), 0.015)
        self.assertLess(float(command.std(0).max()), 0.20)


if __name__ == "__main__":
    unittest.main()
