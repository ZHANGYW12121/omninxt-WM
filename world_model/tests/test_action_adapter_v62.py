from __future__ import annotations

import unittest

import torch

from modules.action_adapter import ActionAdapterConfig, HorizontalActionAdapter


class HorizontalActionAdapterTest(unittest.TestCase):

    def test_policy_and_applied_action_order(self):
        adapter = HorizontalActionAdapter(ActionAdapterConfig(
            target_altitude_agl_m=1.0,
            altitude_kp=0.8,
            vertical_velocity_kd=0.0,
            maximum_vertical_action=0.2,
        ))
        ego = torch.zeros(2, 14)
        ego[:, 9] = torch.tensor([0.5, 1.25])
        policy = torch.tensor([[0.4, -0.3, 0.2], [-0.1, 0.6, -0.4]])
        applied = adapter(policy, ego)
        expected = torch.tensor([
            [0.4, -0.3, 0.2, 0.2],
            [-0.1, 0.6, -0.2, -0.4],
        ])
        torch.testing.assert_close(applied, expected)
        torch.testing.assert_close(
            adapter.policy_from_applied(applied), policy)

    def test_vertical_damping_and_gradient_contract(self):
        adapter = HorizontalActionAdapter(ActionAdapterConfig(
            altitude_kp=0.5, vertical_velocity_kd=0.25,
            maximum_vertical_action=0.2,
        ))
        ego = torch.zeros(1, 14)
        ego[0, 9] = 1.0
        ego[0, 5] = 0.4
        policy = torch.tensor([[0.2, 0.1, -0.3]], requires_grad=True)
        applied = adapter(policy, ego)
        self.assertAlmostEqual(
            float(applied[0, 2].detach()), -0.1, places=6)
        applied.sum().backward()
        torch.testing.assert_close(policy.grad, torch.ones_like(policy))


if __name__ == "__main__":
    unittest.main()
