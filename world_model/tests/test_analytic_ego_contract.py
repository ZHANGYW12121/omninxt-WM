import unittest
from types import SimpleNamespace

import numpy as np

from analytic_ego_contract import (
    analytic_ego_transition_sha256,
    analytic_ego_transitions,
    load_analytic_ego_contract,
    validate_analytic_ego_coefficients,
)


class AnalyticEgoContractTest(unittest.TestCase):
    def test_transition_selection_uses_destination_applied_action(self):
        ego = np.zeros((4, 14), np.float32)
        ego[:, 13] = 1.0
        action = np.arange(16, dtype=np.float32).reshape(4, 4) / 20.0
        arrays = {
            "ego_state": ego,
            "action": action,
            "action_valid": np.asarray((False, True, True, True)),
            "simulation_time_s": np.asarray((0.0, 0.1, 0.2, 0.25)),
            "is_first": np.asarray((True, False, False, False)),
            "is_last": np.asarray((False, False, False, True)),
            "frame_index": np.arange(4, dtype=np.int64),
        }
        episode = SimpleNamespace(
            directory="synthetic",
            metadata={"episode": {"crowd_seed": 7}},
        )
        dataset = SimpleNamespace(
            episodes=[episode], _episode_arrays=lambda _: arrays)
        result = analytic_ego_transitions(dataset)
        self.assertEqual(result["source_frame"].tolist(), [0, 1, 2])
        np.testing.assert_allclose(result["applied_action"], action[1:])
        np.testing.assert_allclose(result["dt_s"][:, 0], (0.1, 0.1, 0.05))
        self.assertEqual(
            analytic_ego_transition_sha256(result),
            analytic_ego_transition_sha256(result),
        )

    def test_configured_coefficients_must_match_provenance(self):
        contract = load_analytic_ego_contract()
        model = SimpleNamespace(
            analytic_ego_velocity_response=tuple(
                contract["velocity_response"]),
            analytic_ego_attitude_coefficients=tuple(
                tuple(row) for row in contract["attitude_coefficients"]),
        )
        validate_analytic_ego_coefficients(contract, model)
        model.analytic_ego_velocity_response = (0.1, 0.2, 0.3)
        with self.assertRaisesRegex(ValueError, "train-only provenance"):
            validate_analytic_ego_coefficients(contract, model)


if __name__ == "__main__":
    unittest.main()
