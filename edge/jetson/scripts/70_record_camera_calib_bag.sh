#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_oak4p"
host_dir="/home/neu/OmniNxt/runtime/calibration/bags"
container_dir="/root/oak_ffc_ws/runtime/calibration/bags"
topic="/oak_ffc_4p/assemble_image"
stamp="$(date +%Y%m%d_%H%M%S)"
bag_name="quarterkalibr_camera_${stamp}"

mkdir -p "${host_dir}"
docker ps --format '{{.Names}}' | grep -Fxq "${container_name}" || {
  echo "Start the OAK driver in normal mode first: 20_start_oak4p_driver.sh normal" >&2
  exit 1
}
docker exec "${container_name}" bash -lc \
  "source /opt/ros/noetic/setup.bash
   source /root/oak_ffc_ws/devel/setup.bash
   test \"\$(rostopic type '${topic}')\" = sensor_msgs/Image"

echo "Recording ${topic}. Press Ctrl-C after completing the required board sequence."
docker exec -it "${container_name}" bash -lc \
  "source /opt/ros/noetic/setup.bash
   source /root/oak_ffc_ws/devel/setup.bash
   exec rosbag record --lz4 -O '${container_dir}/${bag_name}' '${topic}'"

echo "Saved under ${host_dir}/${bag_name}.bag"
