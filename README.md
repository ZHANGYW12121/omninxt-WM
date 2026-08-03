# OmniNxt WM

This repository is the single source of truth shared by the server and the
Alienware laptop. It contains first-party simulation, camera/depth/skeleton,
world-model and interface code. Machine installations, generated data, model
weights and secrets stay outside Git.

Start with [docs/MULTI_MACHINE_SETUP.md](docs/MULTI_MACHINE_SETUP.md). The
authoritative collaboration rules are in
[docs/GIT_MULTI_MACHINE_CODEX_WORKFLOW.md](docs/GIT_MULTI_MACHINE_CODEX_WORKFLOW.md).

Key entry points:

- `simulation/isaacsim/database/run_people_warehouse_with_map.sh`
- `simulation/omnidepth/run_isaac_sync_live.sh`
- `simulation/isaacsim/database/run_navigation_benchmark.sh`
- `navigation/README.md`
- `world_model/README.md`
- `tools/configure_machine.sh`
- `tools/fetch_navrl_checkpoint.sh`
- `tools/doctor.sh`

The server copy imported on 2026-08-03 is authoritative for simulation,
calibration, Isaac ground-truth disparity and skeleton acquisition. The
Alienware copy must be used as the authoritative source for the next
`world_model/` import.
