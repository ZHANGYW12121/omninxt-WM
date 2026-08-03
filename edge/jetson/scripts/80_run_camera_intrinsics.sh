#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/home/neu/OmniNxt/runtime/calibration"
active_file="${runtime_root}/ACTIVE_RUN"
image_name="omninxt/tartancalib-noetic-arm64:jp512"
tartan_calibrate_source="/home/neu/OmniNxt/source/deps/tartancalib/aslam_offline_calibration/kalibr/python/tartan_calibrate"

[[ -f "${active_file}" ]] || {
  echo "No active formal calibration run." >&2
  exit 1
}
run_dir="$(<"${active_file}")"
prepared="${run_dir}/prepared"
intrinsics_bags="${prepared}/intrinsics_5hz"

for camera in CAM_A CAM_B CAM_C CAM_D; do
  [[ -f "${intrinsics_bags}/${camera}.bag" ]] || {
    echo "Missing memory-safe intrinsics bag: ${intrinsics_bags}/${camera}.bag" >&2
    echo "Run scripts/79_prepare_intrinsics_subsets.py first." >&2
    exit 1
  }
done
[[ -f "${tartan_calibrate_source}" ]] || {
  echo "Missing patched TartanCalib entrypoint: ${tartan_calibrate_source}" >&2
  exit 1
}

docker run --rm \
  -v "${prepared}:/data" \
  -v "${tartan_calibrate_source}:/catkin_ws/src/tartancalib/aslam_offline_calibration/kalibr/python/tartan_calibrate:ro" \
  "${image_name}" \
  bash -lc '
    set -Eeuo pipefail
    source /opt/ros/noetic/setup.bash
    source /catkin_ws/devel/setup.bash
    export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
    export TARTAN_EXTRACT_PROCESSES=2
    export MPLBACKEND=Agg
    for camera in CAM_A CAM_B CAM_C CAM_D; do
      if test -s "/data/${camera}/log1-camchain.yaml"; then
        echo "=== ${camera} intrinsics already complete; skipping ==="
        continue
      fi
      echo "=== Calibrating ${camera} intrinsics ==="
      mkdir -p "/data/${camera}"
      rosrun kalibr tartan_calibrate \
        --bag "/data/intrinsics_5hz/${camera}.bag" \
        --target /data/april_6x6.yaml \
        --topics "/${camera}" \
        --models omni-radtan \
        --save_dir "/data/${camera}" \
        --dont-show-report
      test -s "/data/${camera}/log1-camchain.yaml"
    done
  '
