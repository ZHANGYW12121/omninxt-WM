#!/usr/bin/env bash
set -Eeuo pipefail

stage="${1:-}"
runtime_root="/home/neu/OmniNxt/runtime/calibration"
active_file="${runtime_root}/ACTIVE_RUN"
oak_container="omninxt_oak4p"
mavros_container="omninxt_mavros"
preview_container="omninxt_stage_recorder"
preview_image="omninxt/oak4p-noetic-arm64:jp512"
record_duration_seconds=75

case "${stage}" in
  CAM_A|CAM_B|CAM_C|CAM_D|CAM_D-CAM_A|CAM_A-CAM_B|CAM_B-CAM_C|CAM_C-CAM_D) ;;
  *)
    echo "Usage: $0 {CAM_A|CAM_B|CAM_C|CAM_D|CAM_D-CAM_A|CAM_A-CAM_B|CAM_B-CAM_C|CAM_C-CAM_D}" >&2
    exit 2
    ;;
esac

[[ -f "${active_file}" ]] || {
  echo "No active run. Run ./scripts/77_begin_formal_calibration.sh first." >&2
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

docker ps --format '{{.Names}}' | grep -Fxq "${oak_container}" || {
  echo "${oak_container} is not running" >&2
  exit 1
}
docker ps --format '{{.Names}}' | grep -Fxq "${mavros_container}" || {
  echo "${mavros_container} is not running" >&2
  exit 1
}

for camera in CAM_A CAM_B CAM_C CAM_D; do
  docker exec "${oak_container}" bash -lc "
    source /opt/ros/noetic/setup.bash
    test \"\$(rostopic type /oak_ffc_4p/${camera})\" = sensor_msgs/Image
  "
done
docker exec "${oak_container}" bash -lc "
  source /opt/ros/noetic/setup.bash
  test \"\$(rostopic type /mavros/imu/data_raw)\" = sensor_msgs/Imu
"

host_bag="${run_dir}/raw/${stage}.bag"
[[ ! -e "${host_bag}" ]] || {
  echo "Refusing to overwrite existing stage bag: ${host_bag}" >&2
  exit 1
}
container_bag="/runtime/calibration/runs/$(basename "${run_dir}")/raw/${stage}"

[[ -n "${DISPLAY:-}" && -S "/tmp/.X11-unix/X${DISPLAY#:}" ]] || {
  echo "No local graphical display was found (DISPLAY=${DISPLAY:-unset})." >&2
  exit 1
}
command -v xhost >/dev/null || {
  echo "xhost is required for the camera preview." >&2
  exit 1
}

docker rm -f "${preview_container}" >/dev/null 2>&1 || true
xhost +SI:localuser:root >/dev/null
cleanup() {
  docker rm -f "${preview_container}" >/dev/null 2>&1 || true
  xhost -SI:localuser:root >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Preparing formal stage ${stage}"
echo "Move the entire rigid aircraft; keep the AprilGrid fixed."
echo "A preview window will open. Focus it and press SPACE to begin recording."
echo "Recording will stop and finalize automatically after ${record_duration_seconds} seconds."
echo "Ctrl-C remains available if you need to stop early."

set +e
docker run --rm -it \
  --name "${preview_container}" \
  --network host \
  -e ROS_MASTER_URI="http://127.0.0.1:11311" \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/neu/OmniNxt/scripts:/scripts:ro \
  -v /home/neu/OmniNxt/runtime:/runtime \
  "${preview_image}" \
  bash -lc "
  source /opt/ros/noetic/setup.bash
  exec python3 /scripts/78_record_formal_stage_interactive.py \
    --stage '${stage}' \
    --output-base '${container_bag}' \
    --duration '${record_duration_seconds}'
"
status=$?
set -e

if [[ -s "${host_bag}" ]]; then
  echo "Saved and finalized ${host_bag}"
  exit 0
fi

if (( status == 3 )); then
  echo "Stage ${stage} was cancelled before recording." >&2
else
  echo "Stage ${stage} did not produce a valid bag (recorder exit ${status})." >&2
fi
exit "${status}"
