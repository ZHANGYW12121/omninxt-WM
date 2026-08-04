# Backend instructions

- Treat `interfaces/skeleton3d/` as the authoritative Nano wire contract; do
  not import runtime implementation code from `edge/jetson/`.
- Keep the TCP receive path bounded and non-blocking for the Nano producer.
  Reject malformed, oversized, stale, and non-finite packets before adapting
  them for the world model.
- Preserve ROS FLU `base_link` coordinates and metres. Do not silently convert
  the Human stream to world coordinates.
- Runtime snapshots, status files, datasets, checkpoints, credentials and
  machine addresses belong under `.local/` or environment configuration, not
  Git.
- Run `python3 -m unittest backend.tests.test_skeleton_receiver -v` after
  protocol, slotting, velocity, or receiver changes.
