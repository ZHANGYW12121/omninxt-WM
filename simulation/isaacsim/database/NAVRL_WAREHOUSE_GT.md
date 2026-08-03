# NavRL pretrained baseline in the pedestrian Warehouse

This path loads the unmodified official checkpoint from the external asset
root (`$OMNINXT_ASSET_ROOT/models/navrl/navrl_checkpoint.pt`). Run
`tools/fetch_navrl_checkpoint.sh` once after cloning; the checkpoint is not
tracked by Git.

Inputs reproduce the upstream training definitions:

- state `[8]`: goal-frame unit goal vector, horizontal/vertical distance, and velocity;
- lidar `[1,36,4]`: `4 m - hit range`, measured with real PhysX scene
  raycasts (the drone's own colliders are filtered), 10-degree horizontal
  resolution and vertical angles `[-10, 0, 10, 20]`;
- dynamic obstacles `[1,5,10]`: the nearest five people within 4 m;
- action `[3]`: Beta-policy mean mapped to `[-1,1] m/s` and rotated from the
  goal frame into Isaac world ENU.

For the first Warehouse validation, altitude remains under the established
Pegasus/PX4 height loop. Static geometry and people use Isaac GT. No inflated
2-D obstacle AABBs participate in the NavRL lidar observation. The upstream
dynamic-obstacle ORCA/velocity-obstacle safety shield is enabled by default
after policy inference. It uses the official 2.0 s horizon, 0.05 s step,
0.30 m drone radius, and 0.30 m additional safety distance. Disable it only
for an explicit ablation with `NAVRL_SAFETY_SHIELD_ENABLED=0`.

The shield does not depend on PX4. PX4/MAVSDK and the local Pegasus backend
only execute the already-filtered velocity command.

Run with PX4/MAVSDK:

```bash
cd omninxt-WM/simulation/isaacsim/database
./run_warehouse_navrl_gt.sh
```

Press `T` once Isaac and PX4 are ready. No dataset is recorded.

State-source switch:

```bash
# Warehouse GT-state baseline (default): callback-free MAVSDK command channel.
NAVRL_STATE_SOURCE=isaac ./run_warehouse_navrl_gt.sh

# Sim-to-real/real-flight mode: consume PX4 EKF state through MAVSDK.
NAVRL_STATE_SOURCE=mavsdk ./run_warehouse_navrl_gt.sh
```

The default `isaac` mode creates no MAVSDK callback streams; Pegasus has already
confirmed the PX4 heartbeat before the command channel starts. The `mavsdk`
mode enables bounded 20 Hz position/velocity and 10 Hz attitude callbacks. Its
target coordinates must be expressed in the same PX4 local ENU frame as the EKF
telemetry. Both modes continue sending the final shielded velocity to PX4
through MAVSDK Offboard.

Because PX4 SITL heartbeats follow simulation time while MAVSDK timeouts follow
wall time, callback-free Isaac mode raises the MAVLink timeout to 60 seconds by
default. Override it with `MAVSDK_SIM_MAVLINK_TIMEOUT_SEC` only when needed;
real-flight `mavsdk` mode retains MAVSDK's normal timeout behavior.

Run with the local Pegasus backend for a lightweight policy/perception test:

```bash
NAVRL_USE_PX4=0 ./run_warehouse_navrl_gt.sh
```
