#!/usr/bin/env bash
set -Eeuo pipefail

stage="${1:-}"
root="/home/neu/OmniNxt"
runtime_root="${root}/runtime/epipolar_validation"
active_file="${runtime_root}/ACTIVE_RUN"
depth_container="omninxt_omnidepth"
preview_container="omninxt_epipolar_recorder"
preview_image="omninxt/oak4p-noetic-arm64:jp512"
record_duration_seconds="${EPIPOLAR_DURATION_SECONDS:-25}"

case "${stage}" in
  AB_RIGHT) stereo_id=0; physical_side="right side: CAM_A + CAM_B" ;;
  BC_REAR) stereo_id=1; physical_side="rear side: CAM_B + CAM_C" ;;
  CD_LEFT) stereo_id=2; physical_side="left side: CAM_C + CAM_D" ;;
  DA_FRONT) stereo_id=3; physical_side="front side: CAM_D + CAM_A" ;;
  *)
    echo "Usage: $0 {AB_RIGHT|BC_REAR|CD_LEFT|DA_FRONT}" >&2
    exit 2
    ;;
esac

"${root}/scripts/85_start_live_omnidepth.sh"

[[ -f "${active_file}" ]] || {
  echo "No active validation run. Run ./scripts/98_begin_epipolar_validation.sh first." >&2
  exit 1
}
run_dir="$(<"${active_file}")"
case "$(realpath -m "${run_dir}")" in
  "${runtime_root}/runs/"*) ;;
  *)
    echo "Invalid active validation path: ${run_dir}" >&2
    exit 1
    ;;
esac

left_topic="/depth_estimation/stereo_${stereo_id}/left"
right_topic="/depth_estimation/stereo_${stereo_id}/right"
for topic in "${left_topic}" "${right_topic}"; do
  docker exec "${depth_container}" bash -lc "
    source /opt/ros/noetic/setup.bash
    test \"\$(rostopic type '${topic}')\" = sensor_msgs/Image
  "
done

host_bag="${run_dir}/raw/${stage}.bag"
[[ ! -e "${host_bag}" ]] || {
  echo "Refusing to overwrite existing validation bag: ${host_bag}" >&2
  exit 1
}
container_bag="/runtime/epipolar_validation/runs/$(basename "${run_dir}")/raw/${stage}"

[[ -n "${DISPLAY:-}" && -S "/tmp/.X11-unix/X${DISPLAY#:}" ]] || {
  echo "No local graphical display was found (DISPLAY=${DISPLAY:-unset})." >&2
  exit 1
}
command -v xhost >/dev/null || {
  echo "xhost is required for the preview window." >&2
  exit 1
}

docker rm -f "${preview_container}" >/dev/null 2>&1 || true
xhost +SI:localuser:root >/dev/null
cleanup() {
  docker rm -f "${preview_container}" >/dev/null 2>&1 || true
  xhost -SI:localuser:root >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Preparing ${stage} (${physical_side})"
echo "The window shows the runtime-rectified LEFT and RIGHT images."
echo "Place AprilGrid in their common view and align features against the horizontal guides."
echo "Focus the preview and press SPACE to begin."
echo "Recording stops automatically after ${record_duration_seconds} seconds."
echo "Q/ESC cancels during preview; Ctrl-C can finish a recording early."

set +e
docker run --rm -it \
  --name "${preview_container}" \
  --network host \
  -e ROS_MASTER_URI="http://127.0.0.1:11311" \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${root}/scripts:/scripts:ro" \
  -v "${root}/runtime:/runtime" \
  "${preview_image}" \
  bash -lc "
    source /opt/ros/noetic/setup.bash
    exec python3 /scripts/98_record_epipolar_stage_interactive.py \
      --stage '${stage}' \
      --left-topic '${left_topic}' \
      --right-topic '${right_topic}' \
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
