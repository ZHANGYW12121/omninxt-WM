# Repository instructions

Read `docs/GIT_MULTI_MACHINE_CODEX_WORKFLOW.md` before repository-wide work.

- Keep server and Alienware on one Git history; do not create machine-specific
  source copies.
- Never put absolute home-directory paths in tracked runtime code. Add a
  variable to `tools/configure_machine.sh` and `.local/machine.env` instead.
- Do not commit Isaac/PX4/Pegasus/D2SLAM installations, Conda environments,
  datasets, recordings, model weights, TensorRT engines, output or secrets.
- Server simulation/depth/skeleton is the current authority. Alienware
  world-model is the authority until its first import is reviewed and merged.
- Update an interface contract before changing both producers and consumers.
- Do not commit or push unless the user explicitly requests it.
