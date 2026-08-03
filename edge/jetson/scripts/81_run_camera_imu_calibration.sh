#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/home/neu/OmniNxt/runtime/calibration"
active_file="${runtime_root}/ACTIVE_RUN"
image_name="omninxt/tartancalib-noetic-arm64:jp512"
icc_sensors_source="/home/neu/OmniNxt/source/deps/tartancalib/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py"

[[ -f "${active_file}" ]] || {
  echo "No active formal calibration run." >&2
  exit 1
}
run_dir="$(<"${active_file}")"
prepared="${run_dir}/prepared"

for camera in CAM_A CAM_B CAM_C CAM_D; do
  [[ -s "${prepared}/${camera}/log1-camchain.yaml" ]] || {
    echo "Missing ${camera} intrinsic result." >&2
    exit 1
  }
done
[[ -s "${prepared}/imu.yaml" ]] || {
  echo "Missing IMU configuration." >&2
  exit 1
}
[[ -s "${icc_sensors_source}" ]] || {
  echo "Missing patched Kalibr IMU-camera sensor module." >&2
  exit 1
}

docker run --rm \
  -v "${prepared}:/data" \
  -v "${icc_sensors_source}:/catkin_ws/src/tartancalib/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py:ro" \
  "${image_name}" \
  bash -lc '
    set -Eeuo pipefail
    source /opt/ros/noetic/setup.bash
    source /catkin_ws/devel/setup.bash
    export MPLBACKEND=Agg
    export KALIBR_EXTRACT_PROCESSES=2
    cd /data
    for camera in CAM_A CAM_B CAM_C CAM_D; do
      if test -s "/data/${camera}-camchain-imucam.yaml"; then
        echo "=== ${camera} camera-IMU calibration already complete; skipping ==="
        continue
      fi
      echo "=== Calibrating ${camera} to PX4 IMU ==="
      rosrun kalibr kalibr_calibrate_imu_camera \
        --bag "/data/${camera}.bag" \
        --target /data/april_6x6.yaml \
        --cams "/data/${camera}/log1-camchain.yaml" \
        --imu /data/imu.yaml \
        --max-iter 30 \
        --dont-show-report
      test -s "/data/${camera}-camchain-imucam.yaml"
    done
  '
