#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
oak_container="omninxt_oak4p"
depth_container="omninxt_omnidepth"
config="/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp/quadcam_depth.yaml"
stamp="$(date +%Y%m%d_%H%M%S)"
log="/runtime/live_omnidepth_${stamp}.log"
pointcloud_output="${OMNIDEPTH_POINTCLOUD_OUTPUT:-true}"

case "${pointcloud_output}" in
  true|false) ;;
  *) echo "OMNIDEPTH_POINTCLOUD_OUTPUT must be true or false" >&2; exit 2 ;;
esac

if ! docker ps --format '{{.Names}}' | grep -Fxq "${depth_container}"; then
  "${root}/scripts/31_start_omnidepth_container.sh"
fi
if ! docker ps --format '{{.Names}}' | grep -Fxq "${oak_container}"; then
  "${root}/scripts/20_start_oak4p_driver.sh" normal
fi

for _ in $(seq 1 60); do
  if docker exec "${depth_container}" bash -lc \
    'source /opt/ros/noetic/setup.bash
     test "$(rostopic type /oak_ffc_4p/assemble_image 2>/dev/null)" = sensor_msgs/Image'
  then
    break
  fi
  sleep 1
done
docker exec "${depth_container}" bash -lc \
  'source /opt/ros/noetic/setup.bash
   test "$(rostopic type /oak_ffc_4p/assemble_image)" = sensor_msgs/Image'

if docker exec "${depth_container}" bash -lc \
  "pgrep -f '[q]uadcam_depth_est_node' >/dev/null"; then
  echo "OmniDepth node is already running."
else
  docker exec -d \
    -e LIVE_CONFIG="${config}" \
    -e LIVE_LOG="${log}" \
    -e POINTCLOUD_OUTPUT="${pointcloud_output}" \
    "${depth_container}" bash -lc '
      source /opt/ros/noetic/setup.bash
      source /root/swarm_ws/devel/setup.bash
      exec roslaunch quadcam_depth_est depth-node.launch \
        depth_config:="${LIVE_CONFIG}" \
        enable_pointcloud_output:="${POINTCLOUD_OUTPUT}" \
        output:=screen >"${LIVE_LOG}" 2>&1
    '
fi

for _ in $(seq 1 60); do
  if [[ "${pointcloud_output}" == true ]]; then
    wait_topic="/depth_estimation/pointcloud/header"
  else
    wait_topic="/depth_estimation/stereo_0/depth/header"
  fi
  if docker exec -e WAIT_TOPIC="${wait_topic}" \
    "${depth_container}" bash -lc \
    'source /opt/ros/noetic/setup.bash
     timeout 2 rostopic echo -n 1 "${WAIT_TOPIC}" >/dev/null 2>&1'
  then
    echo "OmniDepth is publishing ${wait_topic}."
    echo "Node log: /home/neu/OmniNxt${log}"
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for /depth_estimation/pointcloud" >&2
echo "Inspect: /home/neu/OmniNxt${log}" >&2
exit 1
