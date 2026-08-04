#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
host="${DEPTH_BACKEND_HOST:-}"
port="${DEPTH_BACKEND_PORT:-9766}"
wire_format="${DEPTH_STREAM_FORMAT:-float32}"
compression="${DEPTH_STREAM_COMPRESSION:-zlib}"
compression_level="${DEPTH_STREAM_COMPRESSION_LEVEL:-1}"
max_hz="${DEPTH_STREAM_MAX_HZ:-0}"
image_max_hz="${PERCEPTION_IMAGE_MAX_HZ:-0}"

[[ -n "${host}" ]] || {
  echo "DEPTH_BACKEND_HOST is required" >&2
  exit 2
}
[[ "${port}" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || {
  echo "DEPTH_BACKEND_PORT must be an integer in [1, 65535]" >&2
  exit 2
}
case "${wire_format}" in
  float32|uint16_mm) ;;
  *) echo "DEPTH_STREAM_FORMAT must be float32 or uint16_mm" >&2; exit 2 ;;
esac
case "${compression}" in
  zlib|none) ;;
  *) echo "DEPTH_STREAM_COMPRESSION must be zlib or none" >&2; exit 2 ;;
esac
[[ "${compression_level}" =~ ^[0-9]+$ ]] &&   ((compression_level >= 0 && compression_level <= 9)) || {
  echo "DEPTH_STREAM_COMPRESSION_LEVEL must be in [0, 9]" >&2
  exit 2
}

cleanup() {
  if docker ps --format '{{.Names}}' | grep -Fxq "${container}"; then
    docker exec "${container}" bash -lc       "pkill -TERM -f '[1]09_stream_depth_to_backend.py' || true"       >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

"${root}/scripts/31_start_omnidepth_container.sh"
OMNIDEPTH_POINTCLOUD_OUTPUT=false   "${root}/scripts/85_start_live_omnidepth.sh"

echo "感知数据发送目标: ${host}:${port}"
echo "协议: OPB1 / omninxt.pose_images.v1 + omninxt.depth4.v1"
echo "图像: anchor mosaic + rectified stereo mosaic, max_hz=${image_max_hz}"
echo "深度: four synchronized maps, format=${wire_format}, compression=${compression}, max_hz=${max_hz}"
echo "状态话题: /omninxt_perception_stream/status"
echo "按 Ctrl+C 只停止发送器；OAK和OmniDepth容器保持运行。"

docker exec   -e PYTHONUNBUFFERED=1   "${container}" bash -lc "
    source /opt/ros/noetic/setup.bash
    source /root/swarm_ws/devel/setup.bash
    exec python3 /opt/omninxt/scripts/109_stream_depth_to_backend.py \
      --host '${host}' \
      --port '${port}' \
      --wire-format '${wire_format}' \
      --compression '${compression}' \
      --compression-level '${compression_level}' \
      --image-max-hz '${image_max_hz}' \
      --depth-max-hz '${max_hz}'
  "
