#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_mavros"
imu_topic="/mavros/imu/data_raw"

docker ps --format '{{.Names}}' | grep -Fxq "${container_name}" || {
  echo "${container_name} is not running" >&2
  exit 1
}

docker exec "${container_name}" bash -lc "
source /opt/ros/noetic/setup.bash
echo '=== FCU state ==='
timeout 10 rostopic echo -n 1 /mavros/state
echo '=== IMU message type ==='
test \"\$(rostopic type '${imu_topic}')\" = sensor_msgs/Imu
rostopic type '${imu_topic}'
echo '=== IMU frequency (15 seconds) ==='
timeout 15 rostopic hz '${imu_topic}' || test \$? -eq 124
echo '=== IMU sample ==='
timeout 10 rostopic echo -n 1 '${imu_topic}'
echo '=== MAVROS time-reference sample ==='
timeout 10 rostopic echo -n 1 /mavros/time_reference || true
"

