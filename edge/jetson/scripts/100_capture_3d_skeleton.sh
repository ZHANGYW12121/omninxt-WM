#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
pose_python="/home/neu/venvs/omninxt-pose/bin/python"
rtmlib_root="${root}/source/third_party/rtmlib"
model_dir="${root}/models/rtmpose"
config_dir="${root}/source/D2SLAM/config/quadcam_drone_nxt_tmp"
stamp="$(date +%Y%m%d_%H%M%S)"
output="${1:-${root}/runtime/pose_3d/capture_${stamp}}"
output="$(realpath -m "${output}")"

case "${output}" in
  "${root}/runtime/"*) ;;
  *) echo "Output must be below ${root}/runtime" >&2; exit 2 ;;
esac
[[ ! -e "${output}" ]] || { echo "Refusing to overwrite: ${output}" >&2; exit 2; }
[[ -x "${pose_python}" ]] || { echo "Missing pose environment: ${pose_python}" >&2; exit 2; }

"${root}/scripts/85_start_live_omnidepth.sh"
mkdir -p "${output}"
container_output="/runtime/${output#"${root}/runtime/"}"
docker exec "${container}" bash -lc "
  set -Eeuo pipefail
  source /opt/ros/noetic/setup.bash
  python3 /opt/omninxt/scripts/100_capture_pose_depth_frame.py \
    '${container_output}' --timeout '${CAPTURE_TIMEOUT_S:-45}'
  chown -R 1000:1000 '${container_output}'
"

PYTHONPATH="${rtmlib_root}" "${pose_python}" \
  "${root}/scripts/100_pose_depth_skeleton.py" "${output}" \
  --config-dir "${config_dir}" \
  --mode "${POSE_MODE:-balanced}" \
  --det-model "${model_dir}/yolox_m_humanart.onnx" \
  --pose-model "${model_dir}/rtmpose_m_body17_256x192.onnx" \
  --det-input-size "${POSE_DET_INPUT_SIZE:-640}" \
  --person-threshold "${PERSON_THRESHOLD:-0.35}" \
  --keypoint-threshold "${KEYPOINT_THRESHOLD:-0.30}" \
  --min-depth "${POSE_MIN_DEPTH_M:-0.20}" \
  --max-depth "${POSE_MAX_DEPTH_M:-8.0}" \
  --depth-radius "${POSE_DEPTH_RADIUS_PX:-2}" \
  --merge-distance "${POSE_MERGE_DISTANCE_M:-0.75}"

echo
echo "3D skeleton capture saved:"
echo "  2D pose/depth: ${output}/pose_depth_overview.png"
echo "  3D viewer:     ${output}/skeleton_3d_viewer.html"
echo "  Numeric JSON:  ${output}/skeletons_3d.json"
if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${output}/skeleton_3d_viewer.html" >/dev/null 2>&1 &
fi
