# Navigation algorithm instructions

- Keep EGO-Planner, NavRL and DPMPC behavior aligned with the pinned upstream
  commits recorded in `UPSTREAM_COMPONENTS.yaml`.
- Isaac/PX4 integration lives in `simulation/isaacsim/database`; do not copy
  Isaac Sim, PX4, ROS installations, Docker images, logs or benchmark outputs.
- NavRL checkpoints are external artifacts fetched by
  `tools/fetch_navrl_checkpoint.sh`; never commit model weights.
- Run the protocol/loopback tests and Python syntax checks before committing.
