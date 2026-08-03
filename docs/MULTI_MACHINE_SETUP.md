# Server and Alienware workflow

## Resulting ownership

- `main` is the only shared truth.
- The server import in this repository is authoritative for Isaac simulation,
  crowd planning, four-fisheye calibration, Isaac ground-truth disparity and
  3D skeleton runtime.
- Alienware is authoritative for the next world-model import.
- Absolute installation paths live only in `.local/machine.env`; this file is
  ignored by Git.
- Isaac Sim, PX4, Pegasus, D2SLAM, people assets, USD assets, model weights,
  datasets and generated output are not copied into normal Git history.

## One-time configuration on the server

```bash
cd /home/neu/zyw/omninxt-WM
./tools/configure_machine.sh \
  --zyw-root /home/neu/zyw \
  --isaac-root /home/neu/zyw/isaacsim \
  --px4-root /home/neu/zyw/PX4-Autopilot \
  --legacy-database /home/neu/zyw/isaacsim/database \
  --d2slam-root /home/neu/zyw/our_omni_depth/D2SLAM \
  --omnidepth-gpu 1
```

## One-time configuration on Alienware

Use its actual paths; they do not need to match the server:

```bash
cd ~/zyw/omninxt-WM
# Rebuild the exact current D2SLAM source from the pinned upstream commit.
# The source directory below is the laptop's existing RTMPose ONNX directory.
./tools/bootstrap_d2slam.sh \
  --pose-models-from "$HOME/zyw/our_omni_depth/D2SLAM/models/rtmpose"

./tools/configure_machine.sh \
  --zyw-root "$HOME/zyw" \
  --isaac-root "$HOME/zyw/isaacsim" \
  --px4-root "$HOME/zyw/PX4-Autopilot" \
  --legacy-database "$HOME/zyw/isaacsim/database" \
  --d2slam-root "$PWD/.local/third_party/D2SLAM" \
  --omnidepth-gpu 0
./tools/apply_pegasus_patch.sh
./simulation/omnidepth/build_sync_engines.sh
```

The legacy database argument is only an asset source. Runtime Python comes from
this repository, so a later `git pull --ff-only` updates simulation code
without copying it into the Isaac installation.

## Import Alienware's newer world model once

```bash
git switch main
git pull --ff-only
git switch -c import/alienware-world-model
./tools/import_world_model.sh --source "$HOME/zyw/r2dreamer"
./tools/import_world_model.sh --source "$HOME/zyw/r2dreamer" --apply
./tools/doctor.sh
git status --short
```

Review, commit and merge that branch through the repository's normal review
process. Do not import a Conda directory, datasets, checkpoints or generated
models.

## Daily work

```bash
git switch main
git pull --ff-only
git switch -c sim/short-description       # or wm/short-description
# edit and test
./tools/doctor.sh
git add <explicit paths>
git commit
git push -u origin HEAD
```

After review/merge, the other machine runs:

```bash
git switch main
git pull --ff-only
./tools/doctor.sh
```

Do not make unrelated server and laptop changes directly on `main`. If both
components must change together, update the interface contract first and keep
simulation/world-model changes in separate commits.

## Run

```bash
# Isaac + crowd map
./simulation/isaacsim/database/run_people_warehouse_with_map.sh

# Isaac + ground-truth disparity + skeleton/depth viewer
./simulation/omnidepth/run_isaac_sync_live.sh
```

The second command uses the repository's latest camera, disparity and skeleton
code and the machine-local D2SLAM/models path.

## Large assets

`environments/components.yaml` pins required commits and SHA-256 values.
The 116 MB body USD, warehouse USD, rotors and offline people are resolved from
the existing machine asset directory. If an asset is replaced intentionally,
update both machines, update its hash in the manifest and explain the change in
the same pull request. Do not silently weaken hash checks.
