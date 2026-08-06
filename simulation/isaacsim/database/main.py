#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys


def _apply_crowd_cli_environment():
    """Expose crowd-only CLI overrides before app_config is imported."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--crowd-mode", choices=("server_v2", "legacy"))
    parser.add_argument("--crowd-layout", choices=("sparse", "dense"))
    parser.add_argument(
        "--dense-profile", choices=("transverse40", "legacy")
    )
    parser.add_argument("--crowd-seed", type=int)
    parser.add_argument("--crowd-count", type=int)
    parser.add_argument("--benchmark-run", action="store_true")
    parser.add_argument("--benchmark-seed", type=int)
    parser.add_argument(
        "--benchmark-algorithm",
        choices=("ego", "navrl", "navrl_no_shield", "dpmpc"),
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.crowd_mode is not None:
        os.environ["WAREHOUSE_CROWD_MODE"] = args.crowd_mode
    if args.crowd_layout is not None:
        os.environ["WAREHOUSE_CROWD_LAYOUT"] = args.crowd_layout
    if args.dense_profile is not None:
        os.environ["WAREHOUSE_DENSE_PROFILE"] = args.dense_profile
    if args.crowd_seed is not None:
        os.environ["WAREHOUSE_CROWD_SEED"] = str(args.crowd_seed)
    if args.crowd_count is not None:
        os.environ["WAREHOUSE_CROWD_COUNT"] = str(args.crowd_count)
    if args.benchmark_seed is not None:
        os.environ["WAREHOUSE_CROWD_SEED"] = str(args.benchmark_seed)
    if args.benchmark_algorithm == "navrl":
        os.environ["NAVRL_SAFETY_SHIELD_ENABLED"] = "1"
    elif args.benchmark_algorithm == "navrl_no_shield":
        os.environ["NAVRL_SAFETY_SHIELD_ENABLED"] = "0"
    if args.benchmark_run:
        # Batch trials start without keyboard input and must not spend GPU time
        # exporting camera/depth products unrelated to navigation evaluation.
        # Dataset-v2 recording is the exception: its realistic 2D detector and
        # exact-timestamp 3D lift require the synchronized camera/GT-depth
        # stream configured by run_isaac_sync_live.sh.
        os.environ["CLASSIC_AUTO_START"] = "1"
        if os.environ.get("OMNINXT_DATA_RECORD_ENABLED", "0") != "1":
            os.environ["OMNINXT_DEPTH_EXPORT"] = "0"
            os.environ["OMNINXT_GT_RANGE_EXPORT"] = "0"
            os.environ.setdefault("OMNINXT_CAMERA_ENABLED", "0")
        os.environ.setdefault("OMNINXT_VISUAL_ENABLED", "1")


_apply_crowd_cli_environment()

from app_config import CONTROL_MODE, HEADLESS
from isaacsim import SimulationApp


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-mode",
        choices=(
            "gamepad", "classic", "ego", "navrl", "dpmpc",
            "px4_classic", "px4_ego", "px4_navrl", "px4_dpmpc",
        ),
        default=CONTROL_MODE,
        help=(
            "gamepad keeps joystick control; classic uses the local Pegasus "
            "backend; px4_classic uses PX4 SITL plus MAVSDK offboard commands; "
            "ego/navrl/dpmpc use local Pegasus control; "
            "px4_ego/px4_navrl/px4_dpmpc use the selected planner through "
            "PX4/MAVSDK."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=HEADLESS,
        help="run Isaac Sim without UI windows.",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="run Isaac Sim with UI windows.",
    )
    parser.add_argument(
        "--crowd-mode",
        choices=("server_v2", "legacy"),
        default=os.environ.get("WAREHOUSE_CROWD_MODE", "server_v2"),
        help="select the migrated Warehouse crowd map or the rollback implementation.",
    )
    parser.add_argument(
        "--crowd-layout",
        choices=("sparse", "dense"),
        default=os.environ.get("WAREHOUSE_CROWD_LAYOUT", "sparse"),
        help=(
            "select the original sparse Warehouse crowd layout or the "
            "40-person transverse dense benchmark layout."
        ),
    )
    parser.add_argument(
        "--dense-profile",
        choices=("transverse40", "legacy"),
        default=os.environ.get(
            "WAREHOUSE_DENSE_PROFILE", "transverse40"
        ),
        help=(
            "select the 40-person transverse dense map or the rollback dense "
            "map; ignored by sparse layout"
        ),
    )
    parser.add_argument("--crowd-seed", type=int)
    parser.add_argument("--crowd-count", type=int)
    parser.add_argument(
        "--benchmark-run",
        action="store_true",
        help="run one isolated navigation benchmark episode and exit.",
    )
    parser.add_argument("--benchmark-seed", type=int)
    parser.add_argument("--benchmark-repeat-index", type=int, default=1)
    parser.add_argument("--benchmark-run-id", default="")
    parser.add_argument("--benchmark-output", default="")
    parser.add_argument(
        "--benchmark-algorithm",
        choices=("ego", "navrl", "navrl_no_shield", "dpmpc"),
        default="",
        help="result label for the benchmark variant using this controller.",
    )
    parser.add_argument("--benchmark-eval-hz", type=float, default=10.0)
    parser.add_argument("--benchmark-timeout-sec", type=float, default=120.0)
    parser.add_argument("--benchmark-goal-radius-m", type=float, default=1.00)
    parser.add_argument(
        "--benchmark-human-intrusion-threshold-m",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--benchmark-human-intrusion-exit-threshold-m",
        type=float,
        default=0.60,
    )
    args, _ = parser.parse_known_args()
    if args.benchmark_run:
        if args.control_mode not in (
            "ego", "px4_ego", "navrl", "px4_navrl", "dpmpc", "px4_dpmpc"
        ):
            parser.error(
                "--benchmark-run requires an EGO, NavRL, or DPMPC control mode"
            )
        if args.benchmark_seed is None:
            parser.error("--benchmark-run requires --benchmark-seed")
        if not 0.0 < args.benchmark_eval_hz <= 25.0:
            parser.error("--benchmark-eval-hz must be in (0, 25]")
        if args.benchmark_timeout_sec <= 0.0:
            parser.error("--benchmark-timeout-sec must be positive")
        if args.benchmark_goal_radius_m <= 0.0:
            parser.error("--benchmark-goal-radius-m must be positive")
        if args.benchmark_human_intrusion_threshold_m <= 0.0:
            parser.error(
                "--benchmark-human-intrusion-threshold-m must be positive"
            )
        if (
            args.benchmark_human_intrusion_exit_threshold_m
            <= args.benchmark_human_intrusion_threshold_m
        ):
            parser.error(
                "--benchmark-human-intrusion-exit-threshold-m must exceed "
                "--benchmark-human-intrusion-threshold-m"
            )
    return args


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless})

# 注意：必须在 SimulationApp 创建之后再导入这些依赖 Isaac Sim / Omni 的模块
from pegasus_app import PegasusApp


def main():
    benchmark_config = None
    if ARGS.benchmark_run:
        benchmark_algorithm = ARGS.benchmark_algorithm or (
            "ego"
            if "ego" in ARGS.control_mode
            else "dpmpc"
            if "dpmpc" in ARGS.control_mode
            else "navrl"
        )
        benchmark_config = {
            "algorithm": benchmark_algorithm,
            "safety_layers_enabled": (
                None
                if benchmark_algorithm in ("ego", "dpmpc")
                else os.environ.get("NAVRL_SAFETY_SHIELD_ENABLED", "1") == "1"
            ),
            "seed": int(ARGS.benchmark_seed),
            "repeat_index": int(ARGS.benchmark_repeat_index),
            "run_id": str(ARGS.benchmark_run_id),
            "output_path": str(ARGS.benchmark_output),
            "evaluation_hz": float(ARGS.benchmark_eval_hz),
            "timeout_sec": float(ARGS.benchmark_timeout_sec),
            "goal_radius_m": float(ARGS.benchmark_goal_radius_m),
            "human_intrusion_threshold_m": float(
                ARGS.benchmark_human_intrusion_threshold_m
            ),
            "human_intrusion_exit_threshold_m": float(
                ARGS.benchmark_human_intrusion_exit_threshold_m
            ),
        }
    app = PegasusApp(
        simulation_app,
        control_mode=ARGS.control_mode,
        headless=ARGS.headless,
        benchmark_config=benchmark_config,
    )
    app.run()


if __name__ == "__main__":
    main()
