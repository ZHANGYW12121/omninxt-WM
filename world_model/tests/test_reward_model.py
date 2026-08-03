import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from datasets.isaac_crowd import reward_components_to_vector


WORKSPACE_ROOT = Path(
    os.environ.get(
        "OMNINXT_WORKSPACE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).expanduser().resolve()
ISAAC_DIR = Path(
    os.environ.get(
        "ISAAC_APP_DIR",
        str(WORKSPACE_ROOT / "simulation" / "isaacsim" / "database"),
    )
).expanduser().resolve()
sys.path.insert(0, str(ISAAC_DIR))
PATH = ISAAC_DIR / "reward_model.py"
SPEC = importlib.util.spec_from_file_location("isaac_reward_model", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def config():
    return {"progress_weight_per_m": 2., "max_progress_speed_mps": 3.,
        "goal_height_tolerance_m": .2, "human_clearance_weight_per_sec": 4.,
        "human_hard_clearance_m": .1, "human_safe_clearance_m": .7,
        "drone_collision_radius_m": .32, "human_ttc_horizon_sec": 2.5,
        "human_predictive_safe_clearance_m": .9, "flight_corridor_half_width_m": .75,
        "flight_corridor_lookahead_m": 4., "joint_collision_radii": {"Pelvis": .14},
        "acceleration_filter_alpha": .25, "smooth_term_clip": 4.,
        "acceleration_scale_mps2": 3., "jerk_scale_mps3": 10., "yaw_rate_scale_rps": 1.5,
        "acceleration_weight_per_sec": .15, "jerk_weight_per_sec": .1,
        "yaw_rate_weight_per_sec": .05, "cruise_height_m": 1.,
        "height_tolerance_m": .15, "height_scale_m": .5, "height_weight_per_sec": .5,
        "time_cost_per_sec": .1, "event_rewards": {"recording": 0., "reached_goal": 100., "collision": -120.}}


class RewardModelTest(unittest.TestCase):
    def state(self, pos, vel=(0,0,0), agl=None):
        out = {"position": pos, "velocity": vel, "acceleration": [0,0,0], "yaw": 0.}
        if agl is not None: out["altitude_agl"] = agl
        return out

    def test_3d_progress_and_agl_validity(self):
        calc = MOD.CrowdRewardCalculator(config(), [0,0,2])
        calc.compute(0, self.state([0,0,0]))
        reward, comp, diag = calc.compute(1, self.state([0,0,1]))
        self.assertGreater(comp["progress"], 0)
        self.assertEqual(comp["height"], 0)
        self.assertFalse(diag["altitude_valid"])
        calc.reset(); calc.compute(0, self.state([0,0,0], agl=1))
        _, comp, diag = calc.compute(1, self.state([0,0,0], agl=2))
        self.assertLess(comp["height"], 0)
        self.assertTrue(diag["altitude_valid"])

    def test_predictive_and_corridor_human_risk(self):
        calc = MOD.CrowdRewardCalculator(config(), [10,0,0])
        pelvis0 = [{"pedestrian_id":"p", "position_world_m":[3,0,0]}]
        calc.compute(0, self.state([0,0,0], vel=(1,0,0)), pelvis_records=pelvis0)
        pelvis1 = [{"pedestrian_id":"p", "position_world_m":[2.5,0,0]}]
        _, comp, diag = calc.compute(1, self.state([1,0,0], vel=(1,0,0)), pelvis_records=pelvis1)
        self.assertGreater(diag["human_corridor_risk"], 0)
        self.assertGreater(diag["human_predictive_risk"], 0)
        self.assertLess(comp["human_clearance"], 0)

    def test_terminal_event(self):
        calc = MOD.CrowdRewardCalculator(config(), [0,0,0])
        reward, comp, _ = calc.compute(0, self.state([0,0,0]), collision=True)
        self.assertEqual(comp["event"], -120.)
        self.assertEqual(reward, -120.)
        self.assertEqual(reward_components_to_vector({"reward_components": comp}).shape, (6,))


if __name__ == "__main__": unittest.main()
