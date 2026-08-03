# Single-drone EGO-Planner in Isaac Sim

This integration runs the original ROS1 EGO-Planner in Docker and keeps ROS
libraries out of Isaac Sim. The first validation uses the Isaac scene geometry
as a world-frame obstacle point cloud. `px4_classic` remains unchanged.

## Start

Terminal 1:

```bash
cd omninxt-WM/navigation/ego_planner
./run_planner.sh
```

Terminal 2:

```bash
cd omninxt-WM/navigation/ego_planner
./run_isaac_pedestrians.sh
```

After PX4 and Isaac are ready, focus the Isaac window and press `T` once.
The vehicle performs the Pegasus local takeoff sequence, publishes its goal to
EGO-Planner, and then tracks `/planning/pos_cmd`. A stale planner command causes
an XY hold automatically.

The launcher uses `--control-mode px4_ego`. Static USD geometry is cached once;
runtime point-cloud work only updates moving people. The local `ego` mode remains
available for isolated planner debugging, while `px4_classic` and the existing
OmniDepth workflow are unchanged.

## Topics and coordinates

- `/isaac/odom`: Isaac world ENU, FLU body quaternion (`world` -> `base_link`).
- `/isaac/cloud`: obstacle points already expressed in Isaac world ENU.
- `/move_base_simple/goal`: world ENU goal generated from the current dataset target.
- `/planning/pos_cmd`: original EGO-Planner trajectory-server output.

UDP ports are `15100` (Isaac to ROS) and `15101` (ROS to Isaac). Both processes
must use host networking. Runtime settings are in
`simulation/isaacsim/database/app_config.py`
under `EGO_*`.

## Build log and checks

```bash
cd omninxt-WM/navigation/ego_planner
tail -f docker_build.log
python3 test_udp_protocol.py
```
