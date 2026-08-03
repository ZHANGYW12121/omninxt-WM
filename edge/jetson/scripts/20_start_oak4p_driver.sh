#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_oak4p"
image_name="omninxt/oak4p-noetic-arm64:jp512"
workspace_root="/home/neu/OmniNxt"
mode="${1:-sharpness}"
fps="${OAK_FPS:-20}"
upside_down="${OAK_UPSIDE_DOWN:-false}"
auto_expose="${OAK_AUTO_EXPOSE:-false}"
expose_time_us="${OAK_EXPOSURE_US:-5000}"
iso="${OAK_ISO:-200}"
auto_awb="${OAK_AUTO_AWB:-true}"
awb_value="${OAK_AWB_VALUE:-3000}"

case "${mode}" in
  sharpness) sharpness_mode=true ;;
  normal) sharpness_mode=false ;;
  *)
    echo "Usage: $0 [sharpness|normal]" >&2
    exit 2
    ;;
esac

for boolean_value in "${upside_down}" "${auto_expose}" "${auto_awb}"; do
  case "${boolean_value}" in
    true|false) ;;
    *)
      echo "OAK_UPSIDE_DOWN, OAK_AUTO_EXPOSE, and OAK_AUTO_AWB must be true or false" >&2
      exit 2
      ;;
  esac
done

for numeric_value in "${fps}" "${expose_time_us}" "${iso}" "${awb_value}"; do
  [[ "${numeric_value}" =~ ^[0-9]+$ ]] || {
    echo "FPS, exposure, ISO, and white balance values must be integers" >&2
    exit 2
  }
done

if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "${container_name} is already running"
  exit 0
fi

if docker container inspect "${container_name}" >/dev/null 2>&1; then
  docker rm "${container_name}" >/dev/null
fi

docker run -d \
  --name "${container_name}" \
  --network host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "${workspace_root}/source/driver-oak_ffc_4p_ros:/root/oak_ffc_ws/src/oak_ffc_4p_ros:ro" \
  -v "${workspace_root}/camera_ws/build:/root/oak_ffc_ws/build:ro" \
  -v "${workspace_root}/camera_ws/devel:/root/oak_ffc_ws/devel:ro" \
  -v "${workspace_root}/camera_ws/logs:/root/oak_ffc_ws/logs" \
  -v "${workspace_root}/runtime:/root/oak_ffc_ws/runtime" \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  "${image_name}" \
  bash -lc "source /opt/ros/noetic/setup.bash
source /root/oak_ffc_ws/devel/setup.bash
roscore &
for attempt in \$(seq 1 30); do
  rosparam list >/dev/null 2>&1 && break
  sleep 0.2
done
exec roslaunch oak_ffc_4p_ros OV9782.launch \
  fps:=${fps} \
  resolution:=720 \
  auto_expose:=${auto_expose} \
  expose_time_us:=${expose_time_us} \
  iso:=${iso} \
  auto_awb:=${auto_awb} \
  awb_value:=${awb_value} \
  enable_upside_down:=${upside_down} \
  sharpness_calibration_mode:=${sharpness_mode}"

for attempt in $(seq 1 30); do
  if docker exec "${container_name}" bash -lc \
    "source /opt/ros/noetic/setup.bash; rosparam list >/dev/null 2>&1"; then
    break
  fi
  sleep 0.2
done

if docker ps --format '{{.Names}}' | grep -Fxq omninxt_mavros; then
  docker restart omninxt_mavros >/dev/null
  echo "Restarted omninxt_mavros to register with the new ROS master"
fi

echo "Started ${container_name} in ${mode} mode at ${fps} fps"
echo "180-degree image rotation: ${upside_down}"
echo "Exposure: auto=${auto_expose}, time_us=${expose_time_us}, ISO=${iso}"
echo "White balance: auto=${auto_awb}, value=${awb_value}K"
echo "Inspect with: docker logs -f ${container_name}"
