#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
active_file="${root}/runtime/calibration/photometric/active_run.txt"
test -f "${active_file}" || {
  echo "No active photometric run." >&2
  exit 1
}
run_dir="$(head -n 1 "${active_file}")"

python3 "${root}/scripts/94_generate_photometric_masks.py" \
  --run-dir "${run_dir}" "$@"
