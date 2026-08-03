#!/usr/bin/env python3
"""Generate v2 reward sidecars for legacy Isaac records without mutating raw data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(
    os.environ.get(
        "OMNINXT_WORKSPACE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).expanduser().resolve()
ISAAC_DIR = Path(
    os.environ.get("ISAAC_APP_DIR", str(WORKSPACE_ROOT / "isaacsim/database"))
).expanduser().resolve()
sys.path.insert(0, str(ISAAC_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


reward_model = load_module("isaac_reward_model", ISAAC_DIR / "reward_model.py")
app_config = load_module("isaac_app_config", ISAAC_DIR / "app_config.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for record in sorted(args.data_root.glob("record_*")):
        frames = sorted((record / "frames").glob("frame_*.json"))
        if not frames: continue
        first = json.loads(frames[0].read_text())
        target = first.get("target_point", [0, 0, 1])
        cfg = dict(app_config.DATA_REWARD_CONFIG)
        cfg["joint_collision_radii"] = {
            "Pelvis": .14, "Head": .12, "R_Hand": .08, "L_Hand": .08,
            "R_Foot": .09, "L_Foot": .09, "R_KneeShareBone": .09,
            "L_KneeShareBone": .09, "R_ElbowShareBone": .08, "L_ElbowShareBone": .08,
        }
        calc = reward_model.CrowdRewardCalculator(cfg, target, first.get("goal_region"))
        out_dir = args.output_root / record.name; out_dir.mkdir(parents=True, exist_ok=True)
        for path in frames:
            frame = json.loads(path.read_text())
            out = out_dir / f"{path.stem}_reward.json"
            if out.exists() and not args.overwrite: continue
            reward, components, diagnostics = calc.compute(
                frame.get("timestamp", 0.0), frame.get("drone_state", {}),
                frame.get("pedestrian_joint_distances"),
                frame.get("pedestrian_pelvis_relative_positions"),
                collision=frame.get("collision", False),
                reached_goal=frame.get("reached_goal", False),
                termination_reason=frame.get("termination_reason", "recording"),
            )
            payload = {"reward": reward, "reward_components": components,
                       "reward_diagnostics": diagnostics, "reward_source": "offline_recomputed_v2"}
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{record.name}: {len(frames)} frames -> {out_dir}")


if __name__ == "__main__": main()
