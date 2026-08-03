# Navigation baselines

This directory contains the reproducible algorithm sources used by the Isaac
Warehouse benchmark:

- `ego_planner/`: pinned EGO-Planner source plus the ROS Noetic UDP sidecar.
- `navrl/`: the official quick-demo policy definition used to audit the
  dependency-light Isaac inference wrapper. The checkpoint is downloaded
  separately and verified by SHA-256.
- `dpmpc/`: pinned DPMPC, ACADO and QuadProg++ sources plus the Isaac sidecar.

The Isaac-side controllers, shared benchmark evaluator and batch launcher live
in `simulation/isaacsim/database/`. Machine installations and outputs remain
outside Git.

## Fresh clone

```bash
git clone git@github.com:ZHANGYW12121/omninxt-WM.git
cd omninxt-WM
git switch import/alienware-algorithms

./tools/configure_machine.sh <machine-specific options>
./tools/fetch_navrl_checkpoint.sh
./navigation/ego_planner/build.sh
./navigation/dpmpc/build.sh
```

Run one dry scheduling check before starting Isaac:

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm all --seeds 1 --repeats 1 --dry-run
```

The normal four-way run is:

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm all --seeds 1-30 --repeats 1
```

`all` means EGO-Planner, NavRL with the safety shield, NavRL without the
shield, and DPMPC. See `NAVIGATION_BENCHMARK_GUIDE.md` for metrics and resume
behavior.
