#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/home/neu/OmniNxt/runtime/calibration"
active_file="${runtime_root}/ACTIVE_RUN"
[[ -f "${active_file}" ]] || {
  echo "No active formal calibration run." >&2
  exit 1
}
run_dir="$(<"${active_file}")"

python3 /home/neu/OmniNxt/scripts/82_generate_fisheye_config.py \
  --prepared-dir "${run_dir}/prepared"

