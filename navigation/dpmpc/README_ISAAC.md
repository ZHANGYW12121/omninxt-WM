# DPMPC-Planner in the Isaac Warehouse benchmark

This directory contains a pinned copy of the official ICRA 2021 DPMPC code,
its ACADO/QuadProg dependencies, and an isolated ROS Noetic sidecar for the
existing Isaac/Pegasus/PX4 benchmark.

## Layout

- `trajectory_optimization_official/`: untouched upstream snapshot
- `trajectory_optimization_isaac/`: portable build plus Isaac UDP node
- `dependencies/`: pinned ACADO stable and QuadProg++ sources
- `Dockerfile`: reproducible ROS Noetic image
- `UPSTREAM_AND_ADAPTATION.md`: exact retained parameters and adaptations
- `test_sidecar_loopback.py`: CPU-only static/dynamic planner smoke test

## Build and CPU-only verification

```bash
cd omninxt-WM/navigation/dpmpc
./build.sh
```

Terminal 1:

```bash
docker run --rm --network host \
  --name dpmpc_sidecar_test \
  dpmpc-planner-isaac:noetic
```

Terminal 2:

```bash
cd omninxt-WM/navigation/dpmpc
python3 test_sidecar_loopback.py
```

A healthy result contains `solver_status=0`. The current pinned build produces
a finite avoidance setpoint for the included moving-person test.

## One seeded Warehouse experiment

The benchmark launcher starts/stops the sidecar automatically:

```bash
cd omninxt-WM
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm dpmpc \
  --seeds 1 \
  --repeats 1 \
  --no-headless
```

Remove `--no-headless` for unattended runs. The normal GPU health gate,
infrastructure retry, process isolation, metrics, and reboot autoresume
mechanisms all apply to DPMPC.

## Four-way comparison

```bash
cd omninxt-WM
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm all \
  --seeds 1-30 \
  --repeats 1
```

`all` now means:

1. EGO-Planner
2. NavRL with its configured safety layers
3. NavRL without those safety layers
4. DPMPC

For each seed, all four algorithms share the same start, random 3-D goal,
crowd count, pedestrian trajectories, formations, and static scene.

## Outputs

The common experiment directory contains:

```text
runs/dpmpc_seedNNN_repeatNN.json
logs/dpmpc_seedNNN_repeatNN.log
logs/dpmpc_seedNNN_repeatNN_dpmpc_sidecar.log
runs.csv
summary.json
comparison.md
```

DPMPC decision latency is paired causally by `observation_id` from the atomic
Isaac state/person observation through the ACADO solve to the final PX4
velocity command.
