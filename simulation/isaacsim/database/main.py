#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os


def _apply_scene_argument_environment():
    """Expose scene args before app_config constructs the deterministic map."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--crowd-seed", type=int)
    parser.add_argument("--crowd-count", type=int)
    parser.add_argument("--crowd-mode", choices=("server_v2", "legacy"))
    parser.add_argument("--crowd-layout", choices=("sparse", "dense"))
    parser.add_argument("--dense-profile", choices=("transverse40", "legacy"))
    args, _ = parser.parse_known_args()
    if args.crowd_seed is not None:
        os.environ["WAREHOUSE_CROWD_SEED"] = str(args.crowd_seed)
    if args.crowd_count is not None:
        os.environ["WAREHOUSE_CROWD_COUNT"] = str(args.crowd_count)
    if args.crowd_mode is not None:
        os.environ["WAREHOUSE_CROWD_MODE"] = args.crowd_mode
    if args.crowd_layout is not None:
        os.environ["WAREHOUSE_CROWD_LAYOUT"] = args.crowd_layout
    if args.dense_profile is not None:
        os.environ["WAREHOUSE_DENSE_PROFILE"] = args.dense_profile


_apply_scene_argument_environment()

from app_config import CONTROL_MODE, HEADLESS
from isaacsim import SimulationApp


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-mode",
        choices=("gamepad", "classic", "px4_classic"),
        default=CONTROL_MODE,
        help=(
            "gamepad keeps joystick control; classic uses the local Pegasus "
            "backend; px4_classic uses PX4 SITL plus MAVSDK offboard commands."
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
        "--crowd-seed",
        type=int,
        help=(
            "deterministically fix crowd count, groups, arbitrary-angle routes, "
            "speeds, pauses, and crossing reservations."
        ),
    )
    parser.add_argument(
        "--crowd-count",
        type=int,
        choices=tuple(range(17, 24)) + (40,),
        metavar="17..23|40",
        help="override sparse count, or pass 40 for dense transverse40.",
    )
    parser.add_argument(
        "--crowd-mode", choices=("server_v2", "legacy"),
        help="select the formal V2 implementation or sparse legacy fallback.",
    )
    parser.add_argument(
        "--crowd-layout", choices=("sparse", "dense"),
        help="select the 17-23 person or formal 40-person map.",
    )
    parser.add_argument(
        "--dense-profile", choices=("transverse40", "legacy"),
        help="dense profile; transverse40 is the formal all-horizontal map.",
    )
    args, _ = parser.parse_known_args()
    return args


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless})

# 注意：必须在 SimulationApp 创建之后再导入这些依赖 Isaac Sim / Omni 的模块
from pegasus_app import PegasusApp


def main():
    app = PegasusApp(
        simulation_app,
        control_mode=ARGS.control_mode,
        headless=ARGS.headless,
    )
    app.run()


if __name__ == "__main__":
    main()
