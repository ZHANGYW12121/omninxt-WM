#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container_name="omninxt_omnidepth"
timestamp="$(date +%Y%m%d_%H%M%S)"
log="${root}/logs/32_build_workspace_${timestamp}.log"

docker ps --format '{{.Names}}' | grep -Fxq "${container_name}" || \
  "${root}/scripts/31_start_omnidepth_container.sh"

set -o pipefail
docker exec "${container_name}" bash -lc \
  'set -Eeuo pipefail
   source /opt/ros/noetic/setup.bash
   cd /root/swarm_ws
   catkin config --extend /opt/ros/noetic --merge-devel \
     --cmake-args \
       -DCMAKE_BUILD_TYPE=Release \
       -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
   catkin build quadcam_depth_est -j2 -p2 --no-status' \
  2>&1 | tee "${log}"

echo "Build log: ${log}"
