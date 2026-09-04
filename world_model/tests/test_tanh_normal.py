from __future__ import annotations

import math
import unittest

import torch

from distributions import TanhNormal, bounded_normal


class TanhNormalTest(unittest.TestCase):

    def test_samples_are_strictly_bounded_and_finite(self):
        raw = torch.randn(128, 8, requires_grad=True)
        dist = bounded_normal(raw, min_std=0.05, max_std=1.0)
        action = dist.rsample()
        self.assertTrue(bool((action.abs() < 1.0).all()))
        loss = -(dist.log_prob(action) + 1.0e-3 * dist.entropy()).mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_log_prob_matches_change_of_variables(self):
        mean = torch.tensor([[0.2, -0.4]])
        std = torch.tensor([[0.7, 0.3]])
        pre_tanh = torch.tensor([[0.1, -0.8]])
        action = torch.tanh(pre_tanh)
        dist = TanhNormal(mean, std)
        expected = (
            torch.distributions.Normal(mean, std).log_prob(pre_tanh)
            - torch.log1p(-action.square())
        ).sum(-1)
        torch.testing.assert_close(dist.log_prob(action), expected)

    def test_mode_is_squashed_once(self):
        mean = torch.tensor([[2.0, -2.0]])
        dist = TanhNormal(mean, torch.ones_like(mean))
        torch.testing.assert_close(dist.mode, torch.tanh(mean))
        self.assertNotAlmostEqual(
            float(dist.mode[0, 0]), math.tanh(math.tanh(2.0)), places=4)

    def test_entropy_does_not_reward_float32_boundary_saturation(self):
        torch.manual_seed(7)
        mean = torch.tensor([[20.0, -20.0]], requires_grad=True)
        dist = TanhNormal(mean, torch.full_like(mean, 0.03))
        entropy = dist.entropy()
        self.assertTrue(bool(torch.isfinite(entropy).all()))
        self.assertLess(float(entropy.detach()), 0.0)

        # Gradient descent on -entropy (the Actor convention) must move both
        # saturated means back toward zero, not farther beyond the boundary.
        (-entropy.mean()).backward()
        self.assertGreater(float(mean.grad[0, 0]), 0.0)
        self.assertLess(float(mean.grad[0, 1]), 0.0)

    def test_stored_pre_tanh_log_prob_survives_float32_saturation(self):
        sampled_pre_tanh = torch.tensor([[20.01, -20.02]])
        action = torch.tanh(sampled_pre_tanh)
        self.assertTrue(bool((action.abs() == 1.0).all()))
        mean = torch.tensor([[20.0, -20.0]], requires_grad=True)
        dist = TanhNormal(mean, torch.full_like(mean, 0.03))
        stable = dist.log_prob_from_pre_tanh(sampled_pre_tanh)
        self.assertTrue(bool(torch.isfinite(stable).all()))
        self.assertGreater(float(stable.detach()), -100.0)
        (-stable.mean()).backward()
        self.assertTrue(bool(torch.isfinite(mean.grad).all()))

    def test_antithetic_pairs_have_opposite_standard_normal_noise(self):
        torch.manual_seed(17)
        mean = torch.tensor([
            [0.2, -0.1], [0.4, 0.3],
            [-0.2, 0.5], [0.1, -0.4],
        ], requires_grad=True)
        std = torch.full_like(mean, 0.25)
        dist = TanhNormal(mean, std)
        action, latent = dist.rsample_with_pre_tanh_antithetic_pairs()
        standardized = (latent - mean) / std
        torch.testing.assert_close(standardized[0], -standardized[1])
        torch.testing.assert_close(standardized[2], -standardized[3])
        torch.testing.assert_close(action, torch.tanh(latent))
        action.sum().backward()
        self.assertTrue(bool(torch.isfinite(mean.grad).all()))


if __name__ == "__main__":
    unittest.main()
