#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
pose_python="/home/neu/venvs/omninxt-pose/bin/python"
rtmlib_root="${root}/source/third_party/rtmlib"
model_dir="${root}/models/rtmpose"
config_dir="${root}/source/D2SLAM/config/quadcam_drone_nxt_tmp"
live_dir="${root}/runtime/pose_3d_live"
port="${POSE_LIVE_PORT:-8765}"

stop_stale_processes() {
  if [[ -f "${live_dir}/producer.pid" ]]; then
    stale_producer="$(cat "${live_dir}/producer.pid" 2>/dev/null || true)"
    if [[ "${stale_producer}" =~ ^[0-9]+$ ]]; then
      docker exec "${container}" kill "${stale_producer}" >/dev/null 2>&1 || true
    fi
  fi
  if [[ -f "${live_dir}/viewer.pid" ]]; then
    stale_viewer="$(cat "${live_dir}/viewer.pid" 2>/dev/null || true)"
    if [[ "${stale_viewer}" =~ ^[0-9]+$ ]] &&
       [[ -r "/proc/${stale_viewer}/cmdline" ]] &&
       tr '\0' ' ' < "/proc/${stale_viewer}/cmdline" | grep -Fq "101_live_pose_3d_web.py"; then
      kill "${stale_viewer}" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${live_dir}/producer.pid" "${live_dir}/viewer.pid"
}

cleanup() {
  if [[ -f "${live_dir}/producer.pid" ]]; then
    producer_pid="$(cat "${live_dir}/producer.pid" 2>/dev/null || true)"
    if [[ "${producer_pid}" =~ ^[0-9]+$ ]]; then
      docker exec "${container}" kill "${producer_pid}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

"${root}/scripts/85_start_live_omnidepth.sh"
mkdir -p "${live_dir}"
stop_stale_processes
rm -f "${live_dir}/latest.npz"
docker exec -d "${container}" bash -lc "
  source /opt/ros/noetic/setup.bash
  exec python3 /opt/omninxt/scripts/101_stream_pose_depth_frames.py \
    /runtime/pose_3d_live --max-hz '${POSE_INPUT_HZ:-10}' \
    >>/runtime/pose_3d_live/producer.log 2>&1
"

for _ in $(seq 1 100); do
  [[ -f "${live_dir}/latest.npz" ]] && break
  sleep .1
done
[[ -f "${live_dir}/latest.npz" ]] || { echo "Timed out waiting for live input" >&2; exit 1; }

echo "实时页面: http://127.0.0.1:${port}"
echo "按 Ctrl+C 结束测试；OAK/OmniDepth容器会保留运行。"
if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
  (sleep 1; xdg-open "http://127.0.0.1:${port}" >/dev/null 2>&1) &
fi

PYTHONPATH="/usr/lib/python3.8/dist-packages:${rtmlib_root}:${root}/scripts" "${pose_python}" \
  "${root}/scripts/101_live_pose_3d_web.py" "${live_dir}" \
  --config-dir "${config_dir}" \
  --det-model "${model_dir}/yolox_tiny_humanart.onnx" \
  --pose-model "${model_dir}/rtmpose_s_body17_256x192.onnx" \
  --pose-engine "${model_dir}/tensorrt/rtmpose_s_body17_256x192_fp16.engine" \
  --port "${port}" \
  --det-interval "${POSE_DET_INTERVAL:-20}" \
  --person-threshold "${PERSON_THRESHOLD:-0.35}" \
  --keypoint-threshold "${KEYPOINT_THRESHOLD:-0.30}" \
  --min-depth "${POSE_MIN_DEPTH_M:-0.20}" \
  --max-depth "${POSE_MAX_DEPTH_M:-5.0}"
