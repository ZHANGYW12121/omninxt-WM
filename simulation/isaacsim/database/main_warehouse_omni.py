#!/usr/bin/env python3
"""Launch the existing OmniNxt drone in Isaac's built-in Simple Warehouse."""

import argparse

import app_config
from warehouse_omni_config import apply as apply_warehouse_omni_config

# pegasus_app imports names from app_config, so apply the isolated scene
# overrides before importing it. The original app_config.py is not modified.
apply_warehouse_omni_config(app_config)

from isaacsim import SimulationApp


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-mode",
        choices=("gamepad", "classic", "px4_classic"),
        default=app_config.CONTROL_MODE,
    )
    parser.add_argument("--headless", action="store_true", default=app_config.HEADLESS)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    args, _ = parser.parse_known_args()
    return args


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless})

# Isaac/Omni-dependent imports must remain after SimulationApp construction.
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
