#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container_name="omninxt_omnidepth"
image_name="omninxt/omnidepth-orin:jp512"

if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "${container_name} is already running"
  exit 0
fi
if docker container inspect "${container_name}" >/dev/null 2>&1; then
  docker rm "${container_name}" >/dev/null
fi

docker run -d \
  --name "${container_name}" \
  --runtime nvidia \
  --network host \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -v "${root}/swarm_ws:/root/swarm_ws" \
  -v "${root}/source/D2SLAM:/root/swarm_ws/src/D2SLAM" \
  -v "${root}/source/deps/swarm_msgs:/root/swarm_ws/src/swarm_msgs:ro" \
  -v "${root}/source/deps/vision_opencv:/root/swarm_ws/src/vision_opencv:ro" \
  -v "${root}/runtime:/runtime" \
  -v "${root}/scripts:/opt/omninxt/scripts:ro" \
  -v "${root}/source/third_party/rtmlib:/opt/omninxt/rtmlib:ro" \
  -v "${root}/models:/opt/omninxt/models:ro" \
  "${image_name}" \
  bash -lc 'exec sleep infinity'

echo "Started ${container_name}"
