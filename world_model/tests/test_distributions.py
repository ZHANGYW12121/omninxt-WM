import unittest

import torch
from torch.distributions import kl_divergence

from distributions import (
    OneHotDist, kl, linear_twohot, symlog, symexp, symexp_twohot,
)


class SymexpTwoHotTest(unittest.TestCase):
    def test_bins_remain_linear_in_symlog_space(self):
        distribution = symexp_twohot(torch.zeros(2, 255), 255)
        expected = torch.linspace(-20.0, 20.0, 255)
        torch.testing.assert_close(distribution.bins.cpu(), expected)

    def test_mode_is_symexp_of_expected_symlog_bin(self):
        logits = torch.full((1, 255), -30.0)
        logits[0, 140] = 1.25
        logits[0, 141] = 0.50
        distribution = symexp_twohot(logits, 255)
        expected = symexp(
            (distribution.probs * distribution.bins).sum(-1, keepdim=True))
        torch.testing.assert_close(distribution.mode(), expected)

    def test_two_hot_target_interpolates_in_symlog_space(self):
        logits = torch.zeros(4, 255, requires_grad=True)
        target = torch.tensor([[-120.0], [-0.25], [0.0], [100.0]])
        distribution = symexp_twohot(logits, 255)
        loss = -distribution.log_prob(target).mean()
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

        transformed = symlog(target.squeeze(-1))
        below = (
            distribution.bins <= transformed[:, None]
        ).sum(-1).sub(1).clamp(0, 254)
        above = (
            255 - (distribution.bins > transformed[:, None]).sum(-1)
        ).clamp(0, 254)
        nonzero = logits.grad.ne(0.0)
        # Uniform logits also receive the softmax normalizer gradient.  The
        # target-specific positive correction, however, is confined to the
        # two neighbouring symlog bins; verify those indices bracket target.
        self.assertTrue(bool(
            (distribution.bins[below] <= transformed + 1.0e-6).all()))
        self.assertTrue(bool(
            (distribution.bins[above] >= transformed - 1.0e-6).all()))
        self.assertTrue(bool(nonzero.all()))

    def test_tiny_raw_endpoint_asymmetry_does_not_dominate_mode(self):
        probabilities = torch.zeros(1, 255)
        probabilities[0, 127] = 1.0 - 2.0e-4
        probabilities[0, 0] = 0.9999e-4
        probabilities[0, -1] = 1.0001e-4
        logits = probabilities.clamp_min(1.0e-30).log()
        distribution = symexp_twohot(logits, 255)
        # Expected symlog is 4e-7, so the decoded value remains near zero.
        self.assertLess(abs(float(distribution.mode())), 1.0e-5)


class LinearTwoHotTest(unittest.TestCase):
    def test_mode_is_expected_raw_return(self):
        probabilities = torch.full((1, 9), 1.0e-30)
        # Support is [-160,-120,...,160]. A ten-percent catastrophic return
        # and ninety-percent zero return must bootstrap to -12, not the much
        # smaller symexp(E[symlog(return)]) certainty equivalent.
        probabilities[0, 1] = 0.1
        probabilities[0, 4] = 0.9
        distribution = linear_twohot(
            probabilities.log(), 9, low=-160.0, high=160.0)
        torch.testing.assert_close(
            distribution.mode(), torch.tensor([[-12.0]]),
            atol=1.0e-5, rtol=0.0)

    def test_finite_task_targets_have_finite_two_hot_gradients(self):
        logits = torch.zeros(5, 255, requires_grad=True)
        target = torch.tensor(
            [[-160.0], [-120.0], [-3.25], [100.0], [160.0]])
        distribution = linear_twohot(
            logits, 255, low=-160.0, high=160.0)
        loss = -distribution.log_prob(target).mean()
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


class CategoricalUnimixKLTest(unittest.TestCase):
    def test_kl_matches_the_sampled_unimix_distribution(self):
        left = torch.tensor([[12.0, -8.0, 0.5]])
        right = torch.tensor([[-7.0, 11.0, 0.0]])
        actual = kl(left, right, unimix_ratio=0.01)
        expected = kl_divergence(
            OneHotDist(left, unimix_ratio=0.01),
            OneHotDist(right, unimix_ratio=0.01),
        )
        torch.testing.assert_close(actual, expected)

    def test_extreme_logits_have_finite_bounded_gradients(self):
        left = torch.tensor(
            [[1.0e6, -1.0e6, 0.0]], requires_grad=True)
        right = torch.tensor(
            [[-1.0e6, 1.0e6, 0.0]], requires_grad=True)
        loss = kl(left, right, unimix_ratio=0.01).sum()
        self.assertTrue(bool(torch.isfinite(loss)))
        # The shared 1% uniform mass bounds a three-class KL by the ratio of
        # its largest and smallest representable mixed probabilities.
        self.assertLess(float(loss.detach()), 6.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(left.grad).all()))
        self.assertTrue(bool(torch.isfinite(right.grad).all()))


if __name__ == "__main__":
    unittest.main()
