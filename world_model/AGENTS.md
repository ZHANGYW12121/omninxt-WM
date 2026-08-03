# World-model instructions

- The first authoritative source import must come from Alienware, not from the
  older server R2Dreamer directory.
- Keep datasets, checkpoints, generated models and Conda environments outside
  Git.
- Read simulation data through versioned files under `interfaces/`; do not
  import simulation implementation modules or hard-code a machine path.
- Changes affecting simulation and world-model contracts belong in separate,
  reviewable commits.
