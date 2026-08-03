#!/usr/bin/env python3
"""Generate and optionally display a Warehouse crowd map without Isaac Sim."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowd-layout", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--dense-profile", choices=("transverse40", "legacy"), default="transverse40")
    parser.add_argument("--crowd-seed", type=int, default=1)
    parser.add_argument("--crowd-count", type=int)
    parser.add_argument("--state")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-serve", action="store_true",
                        help="write JSON and exit; useful for automated checks")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["WAREHOUSE_CROWD_MODE"] = "server_v2"
    os.environ["WAREHOUSE_CROWD_LAYOUT"] = args.crowd_layout
    os.environ["WAREHOUSE_DENSE_PROFILE"] = args.dense_profile
    os.environ["WAREHOUSE_CROWD_SEED"] = str(args.crowd_seed)
    if args.crowd_count is not None:
        os.environ["WAREHOUSE_CROWD_COUNT"] = str(args.crowd_count)

    import app_config as config

    scene = config.ACTIVE_CROWD_SCENE
    state_path = Path(args.state or (
        f"/tmp/warehouse_crowd_{args.crowd_layout}_seed{args.crowd_seed}.json"
    ))
    groups = {}
    people = []
    for spec in scene.person_specs:
        groups.setdefault(int(spec.group_id), []).append(spec)
        people.append({
            "name": spec.name,
            "group_id": int(spec.group_id),
            "group_size": int(spec.group_size),
            "member_index": int(spec.member_index),
            "position": [float(value) for value in spec.init_pos],
            "traffic_waiting": False,
            "traffic_cycle_override": False,
            "planned_pause": False,
        })
    planned_routes = []
    for group_id, members in sorted(groups.items()):
        leader = min(members, key=lambda item: item.member_index)
        planned_routes.append({
            "group_id": group_id,
            "points": [[float(point[0]), float(point[1])]
                       for point in leader.waypoints],
            "gates": [list(gate) for gate in leader.traffic_gates],
        })
    state = {
        "ready": True,
        "seed": int(args.crowd_seed),
        "layout": args.crowd_layout,
        "dense_profile": args.dense_profile if args.crowd_layout == "dense" else "",
        "wall_time": time.time(),
        "simulation_time": 0.0,
        "walk_polygon": [list(point) for point in config.PEDESTRIAN_WALK_POLYGON],
        "obstacles": [],
        "people": people,
        "planned_routes": planned_routes,
        "drone": {"position": list(config.SPAWN_POS)},
        "goal": list(config.TARGET_POINT),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(
        f"layout={args.crowd_layout} seed={args.crowd_seed} "
        f"people={scene.num_people} groups={len(groups)} "
        f"state={state_path}",
        flush=True,
    )
    if args.no_serve:
        return

    import crowd_map_monitor
    sys.argv = [
        "crowd_map_monitor.py", "--state", str(state_path),
        *( ["--no-browser"] if args.no_browser else [] ),
    ]
    crowd_map_monitor.main()


if __name__ == "__main__":
    main()
