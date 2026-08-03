#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/home/neu/OmniNxt/runtime/calibration"
active_file="${runtime_root}/ACTIVE_RUN"
image_name="omninxt/tartancalib-noetic-arm64:jp512"

[[ -f "${active_file}" ]] || {
  echo "No active formal calibration run." >&2
  exit 1
}
run_dir="$(<"${active_file}")"
case "$(realpath -m "${run_dir}")" in
  "${runtime_root}/runs/"*) ;;
  *)
    echo "Invalid active run path: ${run_dir}" >&2
    exit 1
    ;;
esac

docker run --rm \
  -v /home/neu/OmniNxt/scripts:/scripts:ro \
  -v "${run_dir}:/run" \
  "${image_name}" \
  bash -lc "
    source /opt/ros/noetic/setup.bash
    source /catkin_ws/devel/setup.bash
    python3 /scripts/79_prepare_quarterkalibr_bags.py --run-dir /run
  "

