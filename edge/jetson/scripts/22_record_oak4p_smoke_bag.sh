#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_oak4p"
duration="${1:-20}"
stamp="$(date +%Y%m%d_%H%M%S)"
bag_path="/root/oak_ffc_ws/runtime/oak4p_smoke_${stamp}.bag"

if ! [[ "${duration}" =~ ^[0-9]+$ ]] || (( duration < 1 || duration > 300 )); then
  echo "Duration must be an integer from 1 to 300 seconds" >&2
  exit 2
fi

docker exec "${container_name}" bash -lc \
  "source /opt/ros/noetic/setup.bash
   source /root/oak_ffc_ws/devel/setup.bash
   timeout --signal=INT ${duration} rosbag record -O '${bag_path}' \
     /oak_ffc_4p/CAM_A \
     /oak_ffc_4p/CAM_B \
     /oak_ffc_4p/CAM_C \
     /oak_ffc_4p/CAM_D \
     /oak_ffc_4p/assemble_image" || status=$?

if [[ "${status:-0}" -ne 0 && "${status}" -ne 124 ]]; then
  exit "${status}"
fi

echo "Saved ${bag_path} (host: /home/neu/OmniNxt/runtime/)"
