# Simulation instructions

- Preserve Isaac physics at 250 Hz for PX4.
- Preserve the single Pegasus person physics callback. Person control is
  simulation-time limited to 25 Hz and skeleton/collision updates to 10 Hz.
- Preserve static crowd route planning as an event-triggered precomputation;
  never move planning into the 25 Hz control loop.
- Keep crowd seeds reproducible and run
  `simulation/isaacsim/database/test_warehouse_crowd_reproduction.py` after
  planning changes.
- Camera calibration comes from `calibration/rigs/`; generated Isaac geometry
  must pass `test_omninxt_sync_geometry.py`.
- Large USD/people/model assets stay machine-local and are resolved by
  `.local/machine.env`.
