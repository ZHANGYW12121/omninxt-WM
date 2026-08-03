# DPMPC upstream and Isaac adaptation

## Pinned sources

- DPMPC: `Zhefan-Xu/trajectory_optimization`,
  commit `c8a89203bb11a5bce5a852052b240ce8638b6122`
- ACADO stable: commit `b8e586639fc714bf3263152637da3b0efce23a32`
- QuadProg++: commit `0c25447365c980876fdc395f55d60300a5e5793c`

`trajectory_optimization_official/` is the unmodified upstream snapshot.
`trajectory_optimization_isaac/` is the buildable Isaac sidecar copy.

## Algorithm parameters retained from the official MAVROS demo

- minimum-snap polynomial degree: 7
- desired static-trajectory speed: 2 m/s
- derivative order: 4
- regularization: 1
- sampling period: 0.1 s
- MPC horizon: 20
- mass: 1.5 kg
- roll/pitch gains and time constants: 10 and 1
- maximum thrust: `3 * 9.8`
- maximum roll/pitch command: `pi/6`
- chance-constraint probability parameter, obstacle gates, weights, and
  `forward_idx=10`: unchanged in `mpcPlanner.cpp`

## Necessary interface adaptations

- Isaac GT static points initialize the same OctoMap used by the official
  collision checks; no ROS/Gazebo map server is required.
- Isaac GT person position and velocity replace the official Gazebo
  `fakeDetector`. Each person remains the official 0.5 x 0.5 x 2.2 m
  ellipsoid, `root_z + 1.0 m` center, and position/velocity variance `1e-4`.
  The variance is retained for the chance constraint; an additional random
  measurement perturbation is not injected because all compared planners use
  Isaac GT perception in this benchmark.
- Isaac world ENU state replaces MAVROS odometry.
- The official demo publishes a position+yaw setpoint and asks PX4 to track
  it while ignoring velocity. The Isaac adapter converts that position
  setpoint to the project's existing MAVSDK velocity-offboard interface.
  It does not add a navigation guard, smoothing, fallback direction, or
  Safety Shield.

## Correctness fixes to definite upstream implementation defects

- Fixed `pose` constructors assigning `_z = z` instead of `z = _z`.
- Added the missing return of the ACADO solver status from `optimize()`.
- Prevented out-of-range reference-trajectory indexing.
- Initialized and owned the OctoMap pointer safely.
- Made header-defined utility functions `inline` to avoid multiple-definition
  errors in the new linked executable.
- Replaced hard-coded author-machine dependency paths with CMake cache paths.
