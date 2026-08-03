#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
config="/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp"
det_engine="/opt/omninxt/models/rtmpose/tensorrt/yolox_nano_coco_416_fp16.engine"
pose_variant="${STEREO_POSE_MODEL:-t}"
case "${pose_variant}" in
  t)
    pose_engine="/opt/omninxt/models/rtmpose/tensorrt/rtmpose_t_body17_256x192_fp16_dynamic_b8.engine"
    ;;
  s)
    pose_engine="/opt/omninxt/models/rtmpose/tensorrt/rtmpose_s_body17_256x192_fp16_dynamic_b8.engine"
    ;;
  *)
    echo "STEREO_POSE_MODEL must be t or s" >&2
    exit 2
    ;;
esac
port="${STEREO_POSE_PORT:-8766}"
web="${STEREO_POSE_WEB:-true}"
backend_host="${STEREO_POSE_BACKEND_HOST:-}"
backend_port="${STEREO_POSE_BACKEND_PORT:-9765}"

case "${web}" in
  true|false) ;;
  *) echo "STEREO_POSE_WEB must be true or false" >&2; exit 2 ;;
esac
[[ "${backend_port}" =~ ^[0-9]+$ ]] || {
  echo "STEREO_POSE_BACKEND_PORT must be an integer" >&2
  exit 2
}

cleanup() {
  if docker ps --format '{{.Names}}' | grep -Fxq "${container}"; then
    docker exec "${container}" bash -lc \
      "pkill -TERM -f '[1]02_live_stereo_pose.py' || true" \
      >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

[[ -s "${root}/models/rtmpose/tensorrt/yolox_nano_coco_416_fp16.engine" ]] || {
  echo "Missing YOLOX-Nano TensorRT engine" >&2
  exit 1
}
pose_engine_host="${root}${pose_engine#/opt/omninxt}"
[[ -s "${pose_engine_host}" ]] || {
  echo "Missing RTMPose-${pose_variant^^} TensorRT engine: ${pose_engine_host}" >&2
  exit 1
}

# Stop only older pose viewers/producers.  Their disk NPZ/JPEG path must not
# compete with this in-memory experiment.
pkill -TERM -f '[1]01_live_pose_3d_web.py' >/dev/null 2>&1 || true
if docker ps --format '{{.Names}}' | grep -Fxq "${container}"; then
  docker exec "${container}" bash -lc \
    "pkill -TERM -f '[1]01_stream_pose_depth_frames.py' || true
     pkill -TERM -f '[1]02_live_stereo_pose.py' || true" \
    >/dev/null 2>&1 || true
fi

# Recreate once when an older container lacks the read-only model/rtmlib
# mounts introduced for the in-container real-time path.
if docker container inspect "${container}" >/dev/null 2>&1; then
  if ! docker inspect "${container}" --format '{{range .Mounts}}{{println .Destination}}{{end}}' \
      | grep -Fxq /opt/omninxt/models; then
    "${root}/scripts/88_stop_live_omnidepth.sh"
    docker rm -f "${container}" >/dev/null
  fi
fi
"${root}/scripts/31_start_omnidepth_container.sh"

# This mode preserves HITNet metric-depth topics but skips dense PCL creation.
"${root}/scripts/88_stop_live_omnidepth.sh"
OMNIDEPTH_POINTCLOUD_OUTPUT=false \
  "${root}/scripts/85_start_live_omnidepth.sh"

if [[ "${web}" == true ]]; then
  echo "实时页面: http://127.0.0.1:${port}"
else
  echo "实时页面: disabled (formal no-encoding mode)"
fi
echo "ROS输出: /omninxt_pose/joints_3d, /omninxt_pose/skeleton_markers, /omninxt_pose/status"
echo "固定骨架输出: /omninxt_pose/skeleton_frame (17 joints/person)"
echo "姿态模型: RTMPose-${pose_variant^^} (${pose_engine##*/})"
if [[ -n "${backend_host}" ]]; then
  echo "骨架后端: ${backend_host}:${backend_port} (non-blocking TCP)"
else
  echo "骨架后端: disabled（设置STEREO_POSE_BACKEND_HOST后启用）"
fi
echo "运行时飞控输入: disabled（不启动MAVROS，不订阅/mavros/*）"
echo "三维坐标系: base_link（静态标定机体系，+X前、+Y左、+Z上）"
echo "按 Ctrl+C 结束骨架测试；OAK和OmniDepth容器会保留运行。"
if [[ "${web}" == true && -n "${DISPLAY:-}" ]] && \
   command -v xdg-open >/dev/null 2>&1; then
  (sleep 2; xdg-open "http://127.0.0.1:${port}" >/dev/null 2>&1) &
fi

docker exec \
  -e PYTHONUNBUFFERED=1 \
  -e STEREO_POSE_PORT="${port}" \
  -e STEREO_POSE_DISPLAY_HZ="${STEREO_POSE_DISPLAY_HZ:-4}" \
  -e STEREO_POSE_DET_INTERVAL="${STEREO_POSE_DET_INTERVAL:-10}" \
  -e STEREO_POSE_CENTER_DET_INTERVAL="${STEREO_POSE_CENTER_DET_INTERVAL:-10}" \
  -e STEREO_POSE_RESCUE_DET_INTERVAL="${STEREO_POSE_RESCUE_DET_INTERVAL:-3}" \
  -e STEREO_POSE_WEB="${web}" \
  -e STEREO_POSE_BACKEND_HOST="${backend_host}" \
  -e STEREO_POSE_BACKEND_PORT="${backend_port}" \
  -e STEREO_POSE_POSE_ENGINE="${pose_engine}" \
  "${container}" bash -lc '
    source /opt/ros/noetic/setup.bash
    source /root/swarm_ws/devel/setup.bash
    export PYTHONPATH=/opt/omninxt/rtmlib:/opt/omninxt/scripts:${PYTHONPATH:-}
    web_args=()
    if [[ "${STEREO_POSE_WEB}" == false ]]; then
      web_args+=(--no-web)
    fi
    backend_args=()
    if [[ -n "${STEREO_POSE_BACKEND_HOST}" ]]; then
      backend_args+=(
        --backend-host "${STEREO_POSE_BACKEND_HOST}"
        --backend-port "${STEREO_POSE_BACKEND_PORT}"
      )
    fi
    exec python3 /opt/omninxt/scripts/102_live_stereo_pose.py \
      --config-dir /root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp \
      --det-engine /opt/omninxt/models/rtmpose/tensorrt/yolox_nano_coco_416_fp16.engine \
      --pose-engine "${STEREO_POSE_POSE_ENGINE}" \
      --web-port "${STEREO_POSE_PORT}" \
      --display-hz "${STEREO_POSE_DISPLAY_HZ}" \
      --det-interval "${STEREO_POSE_DET_INTERVAL}" \
      --center-det-interval "${STEREO_POSE_CENTER_DET_INTERVAL}" \
      --rescue-det-interval "${STEREO_POSE_RESCUE_DET_INTERVAL}" \
      "${backend_args[@]}" \
      "${web_args[@]}"
  '
