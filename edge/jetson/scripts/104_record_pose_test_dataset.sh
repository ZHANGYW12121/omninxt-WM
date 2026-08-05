#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
label="${1:-}"
oak_container="omninxt_oak4p"
recorder_container="omninxt_pose_test_recorder"
image="omninxt/oak4p-noetic-arm64:jp512"
stamp="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${label}" || ! "${label}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Usage: $0 LABEL" >&2
  echo "Examples: $0 single_walk   or   $0 two_people_crossing" >&2
  exit 2
fi

docker ps --format '{{.Names}}' | grep -Fxq "${oak_container}" || {
  echo "${oak_container} is not running." >&2
  echo "Start it with: ${root}/scripts/20_start_oak4p_driver.sh normal" >&2
  exit 1
}
docker exec "${oak_container}" bash -lc '
  source /opt/ros/noetic/setup.bash
  test "$(rostopic type /oak_ffc_4p/assemble_image)" = sensor_msgs/Image
  timeout 8 rostopic echo -n 1 /oak_ffc_4p/assemble_image/width | grep -Fx 5120
'

[[ -n "${DISPLAY:-}" && -S "/tmp/.X11-unix/X${DISPLAY#:}" ]] || {
  echo "No local graphical display was found (DISPLAY=${DISPLAY:-unset})." >&2
  exit 1
}
command -v xhost >/dev/null || {
  echo "xhost is required for the preview window." >&2
  exit 1
}

output_dir="${root}/runtime/pose_test_recordings/${label}_${stamp}"
mkdir -p "${output_dir}"

# Raw capture needs only the OAK publisher. Stop compute consumers so the bag
# reflects the camera stream rather than a Nano overload condition.
"${root}/scripts/103_stop_live_stereo_pose.sh" || true
"${root}/scripts/88_stop_live_omnidepth.sh"

docker rm -f "${recorder_container}" >/dev/null 2>&1 || true
xhost +SI:localuser:root >/dev/null
cleanup() {
  docker rm -f "${recorder_container}" >/dev/null 2>&1 || true
  xhost -SI:localuser:root >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Output: ${output_dir}"
echo "The four-camera preview will open."
echo "Focus the preview and press SPACE. Recording begins after 5 seconds."
echo "Recording finalizes automatically after 30 seconds."
echo "Q/ESC cancels before recording; Ctrl-C safely stops early."

set +e
docker run --rm -it \
  --name "${recorder_container}" \
  --network host \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${root}/scripts:/scripts:ro" \
  -v "${root}/runtime:/runtime" \
  "${image}" bash -lc "
    source /opt/ros/noetic/setup.bash
    exec python3 /scripts/104_record_pose_test_dataset.py \
      --label '${label}' \
      --output-dir '/runtime/pose_test_recordings/${label}_${stamp}' \
      --countdown 5 \
      --duration 30
  "
status=$?
set -e

bag="${output_dir}/input.bag"
if [[ -s "${bag}" ]]; then
  echo
  echo "=== Bag verification ==="
  docker exec "${oak_container}" bash -lc "
    source /opt/ros/noetic/setup.bash
    rosbag info '/root/oak_ffc_ws/runtime/pose_test_recordings/${label}_${stamp}/input.bag'
  "
  echo "Saved raw bag: ${bag}"
  echo "Saved preview: ${output_dir}/preview_4cam.mp4"
  echo "Saved metadata: ${output_dir}/metadata.json"
  exit "${status}"
fi

echo "No valid bag was produced (exit ${status})." >&2
exit "${status}"
