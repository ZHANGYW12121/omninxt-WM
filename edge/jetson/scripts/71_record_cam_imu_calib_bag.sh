#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_oak4p"
host_dir="/home/neu/OmniNxt/runtime/calibration/bags"
container_dir="/root/oak_ffc_ws/runtime/calibration/bags"
image_topic="/oak_ffc_4p/assemble_image"
imu_topic="${IMU_TOPIC:-}"

if [[ -z "${imu_topic}" ]]; then
  echo "No IMU topic has been validated on this system." >&2
  echo "After connecting the final IMU, run: IMU_TOPIC=/validated/topic $0" >&2
  exit 2
fi

mkdir -p "${host_dir}"
docker ps --format '{{.Names}}' | grep -Fxq "${container_name}" || {
  echo "Start the OAK driver in normal mode first." >&2
  exit 1
}
docker exec "${container_name}" bash -lc \
  "source /opt/ros/noetic/setup.bash
   source /root/oak_ffc_ws/devel/setup.bash
   test \"\$(rostopic type '${image_topic}')\" = sensor_msgs/Image
   test \"\$(rostopic type '${imu_topic}')\" = sensor_msgs/Imu"

stamp="$(date +%Y%m%d_%H%M%S)"
bag_name="quarterkalibr_cam_imu_${stamp}"
echo "Recording synchronized camera/IMU topics. Press Ctrl-C when finished."
docker exec -it "${container_name}" bash -lc \
  "source /opt/ros/noetic/setup.bash
   source /root/oak_ffc_ws/devel/setup.bash
   exec rosbag record --lz4 -O '${container_dir}/${bag_name}' \
     '${image_topic}' '${imu_topic}'"

echo "Saved under ${host_dir}/${bag_name}.bag"
