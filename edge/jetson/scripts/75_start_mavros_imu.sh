#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_mavros"
image_name="omninxt/mavros-noetic-arm64:jp512"
serial_path="${PX4_SERIAL:-/dev/serial/by-id/usb-MicoAir_MicoAir743v2_0-if00}"
config_dir="/home/neu/OmniNxt/runtime/calibration/templates"
rate_request="/home/neu/OmniNxt/scripts/75_request_px4_imu_rate.py"

[[ -e "${serial_path}" ]] || {
  echo "PX4 serial device does not exist: ${serial_path}" >&2
  exit 1
}

docker image inspect "${image_name}" >/dev/null 2>&1 || {
  echo "Build the MAVROS image first: ./scripts/74_build_mavros_image.sh" >&2
  exit 1
}

docker rm -f "${container_name}" >/dev/null 2>&1 || true
/home/neu/venvs/mavlink/bin/python "${rate_request}" \
  --device "${serial_path}" \
  --rate "${PX4_IMU_RATE_HZ:-200}"
sleep 1
docker run -d \
  --name "${container_name}" \
  --network host \
  --device "${serial_path}:/dev/px4" \
  -v "${config_dir}:/config:ro" \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  "${image_name}" \
  bash -lc "source /opt/ros/noetic/setup.bash
exec roslaunch mavros node.launch \
  fcu_url:=/dev/px4:115200 \
  gcs_url:=udp://@127.0.0.1:14550 \
  tgt_system:=1 \
  tgt_component:=1 \
  pluginlists_yaml:=/config/mavros_imu_plugins.yaml \
  config_yaml:=/opt/ros/noetic/share/mavros/launch/px4_config.yaml"

echo "Started ${container_name} with ${serial_path}"
echo "This profile exposes PX4 status, raw IMU, and time synchronization only."
