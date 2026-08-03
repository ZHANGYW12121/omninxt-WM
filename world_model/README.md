# World model import boundary

This directory intentionally does not contain the server's older R2Dreamer
copy. The Alienware laptop is the authoritative source for the next import.

On Alienware, create an import branch and run:

```bash
git switch -c import/alienware-world-model
./tools/import_world_model.sh --source /absolute/path/to/r2dreamer --apply
./tools/doctor.sh
git status --short
```

Review the result before committing. Conda environments, datasets, checkpoints,
logs, caches, generated models and machine-specific paths are excluded.
