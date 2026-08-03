#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
active_file="${root}/runtime/calibration/photometric/active_run.txt"
camera="${1:-}"
kind="${2:-}"
frames="${PHOTOMETRIC_FRAMES:-120}"

case "${camera}" in
  CAM_A|CAM_B|CAM_C|CAM_D) ;;
  *)
    echo "Usage: $0 CAM_A|CAM_B|CAM_C|CAM_D flat|dark" >&2
    exit 2
    ;;
esac
case "${kind}" in
  flat|dark) ;;
  *)
    echo "Usage: $0 CAM_A|CAM_B|CAM_C|CAM_D flat|dark" >&2
    exit 2
    ;;
esac
[[ "${frames}" =~ ^[0-9]+$ ]] || {
  echo "PHOTOMETRIC_FRAMES must be an integer." >&2
  exit 2
}

test -f "${active_file}" || {
  echo "No active run. Start with ./scripts/92_begin_photometric_calibration.sh" >&2
  exit 1
}
run_dir="$(head -n 1 "${active_file}")"
case "${run_dir}" in
  "${root}/runtime/calibration/photometric/"run_*) ;;
  *)
    echo "Unsafe or invalid active run path: ${run_dir}" >&2
    exit 1
    ;;
esac

docker ps --format '{{.Names}}' | grep -Fxq omninxt_oak4p || {
  echo "omninxt_oak4p is not running. Restart the photometric run." >&2
  exit 1
}
docker cp \
  "${root}/scripts/93_capture_photometric_stage.py" \
  omninxt_oak4p:/tmp/93_capture_photometric_stage.py

host_output="${run_dir}/raw/${camera}"
container_output="/root/oak_ffc_ws/runtime${host_output#${root}/runtime}"
mkdir -p "${host_output}"

if [[ -e "${host_output}/${kind}_median.png" ]]; then
  backup="${host_output}/${kind}_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${backup}"
  find "${host_output}" -maxdepth 1 -type f -name "${kind}_*" \
    -exec mv -t "${backup}" -- {} +
  echo "Previous ${camera} ${kind} capture moved to ${backup}"
fi

echo "Capturing ${frames} ${kind} frames from ${camera}..."
docker exec omninxt_oak4p bash -lc "
  source /opt/ros/noetic/setup.bash
  python3 /tmp/93_capture_photometric_stage.py \
    --camera '${camera}' \
    --kind '${kind}' \
    --frames '${frames}' \
    --output-dir '${container_output}'
"
echo
echo "Saved ${camera} ${kind} capture under:"
echo "  ${host_output}"
