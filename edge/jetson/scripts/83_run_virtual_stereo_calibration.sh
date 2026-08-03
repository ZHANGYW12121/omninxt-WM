#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/home/neu/OmniNxt/runtime/calibration"
active_file="${runtime_root}/ACTIVE_RUN"
image_name="omninxt/tartancalib-noetic-arm64:jp512"
quarterkalibr="/home/neu/OmniNxt/source/tools-quarterKalibr"
kalibr_camera_source="/home/neu/OmniNxt/source/deps/tartancalib/aslam_offline_calibration/kalibr/python/kalibr_calibrate_cameras"

[[ -f "${active_file}" ]] || {
  echo "No active formal calibration run." >&2
  exit 1
}
run_dir="$(<"${active_file}")"
prepared="${run_dir}/prepared"
[[ -s "${prepared}/fisheye_cams.yaml" ]] || {
  echo "Missing fisheye_cams.yaml" >&2
  exit 1
}
[[ -s "${prepared}/stereo_depth_calibration.bag" ]] || {
  echo "Missing stereo_depth_calibration.bag" >&2
  exit 1
}
[[ -s "${prepared}/april_6x6.yaml" ]] || {
  echo "Missing AprilGrid configuration." >&2
  exit 1
}
[[ -s "${kalibr_camera_source}" ]] || {
  echo "Missing patched Kalibr camera calibrator." >&2
  exit 1
}

docker run --rm \
  -v "${prepared}:/data" \
  -v "${quarterkalibr}:/quarterkalibr:ro" \
  -v "${kalibr_camera_source}:/catkin_ws/src/tartancalib/aslam_offline_calibration/kalibr/python/kalibr_calibrate_cameras:ro" \
  "${image_name}" \
  bash -lc '
    set -Eeuo pipefail
    source /opt/ros/noetic/setup.bash
    source /catkin_ws/devel/setup.bash
    export PYTHONPATH=/quarterkalibr:${PYTHONPATH}
    export QUARTERKALIBR_LOCAL_EXEC=1
    export QUARTERKALIBR_PROCESSES=1
    export KALIBR_EXTRACT_PROCESSES=2
    export KALIBR_OPT_THREADS=2
    export MPLBACKEND=Agg
    cd /quarterkalibr
    python3 - <<"PY"
from utils import VirtualStereoCalibration
VirtualStereoCalibration.calibrate_virtual_stereo(
    "/data/stereo_depth_calibration.bag",
    190,
    320,
    240,
    "/data/fisheye_cams.yaml",
    "/data/virtual_stereo_calibration_190",
    1,
    None,
    verbose=False,
)
PY
  ' 2>&1 | tee "${prepared}/virtual_stereo_calibration_190.log"
