#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE=""
APPLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 --source /absolute/path/to/r2dreamer [--apply]"
      echo "Without --apply, rsync performs a dry run."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SOURCE" && -d "$SOURCE" ]] || { echo "--source must be an existing directory" >&2; exit 2; }
SOURCE="$(realpath "$SOURCE")"
[[ "$SOURCE" != "$REPO_ROOT/world_model" ]] || { echo "Source cannot equal destination" >&2; exit 2; }
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }

if (( APPLY )); then
  branch="$(git -C "$REPO_ROOT" branch --show-current)"
  [[ -n "$branch" && "$branch" != "main" ]] || {
    echo "Refusing to import on main; create an import/alienware-world-model branch first." >&2
    exit 1
  }
fi

args=(-a --itemize-changes
  --exclude=.git/ --exclude=.agents/ --exclude=.codex/
  --include=.env.example --exclude='.env*'
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.venv/' --exclude='venv/'
  --exclude='build/' --exclude='devel/' --exclude='install/' --exclude='logs/'
  --exclude='logdir/' --exclude='checkpoints/' --exclude='weights/'
  --exclude='outputs/' --exclude='replay/' --exclude='trajectories/'
  --exclude='wandb/' --exclude='.hydra/'
  --exclude='conda_history*.yml' --exclude='pip_freeze*.txt'
  --exclude='*.bag' --exclude='*.pcd' --exclude='*.pkl' --exclude='*.npz'
  --exclude='*.npy' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
  --exclude='*.onnx' --exclude='*.engine' --exclude='*.trt'
)
(( APPLY )) || args+=(--dry-run)
rsync "${args[@]}" "$SOURCE/" "$REPO_ROOT/world_model/"
if (( APPLY )); then
  echo "Imported world-model source. Review git status before committing."
  if rg -n --glob '*.py' --glob '*.sh' --glob '*.yaml' '/home/[^/]+' "$REPO_ROOT/world_model"; then
    echo "WARNING: machine-specific paths remain in world_model; parameterize them before commit." >&2
  fi
else
  echo "Dry run only. Repeat with --apply after reviewing the list."
fi
