#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
duration="${1:-1800}"
oak_container="omninxt_oak4p"
depth_container="omninxt_omnidepth"
config="/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp/quadcam_depth.yaml"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${root}/logs/50_live_pipeline_${stamp}.log"
tegrastats_log="${root}/logs/50_live_tegrastats_${stamp}.log"
node_log="${root}/logs/50_live_depth_node_${stamp}.log"

[[ "${duration}" =~ ^[0-9]+$ ]] && (( duration >= 60 )) || {
  echo "Duration must be an integer of at least 60 seconds (default: 1800)." >&2
  exit 2
}
docker ps --format '{{.Names}}' | grep -Fxq "${depth_container}" || {
  "${root}/scripts/31_start_omnidepth_container.sh"
}
if ! docker ps --format '{{.Names}}' | grep -Fxq "${oak_container}"; then
  "${root}/scripts/20_start_oak4p_driver.sh" normal
fi

for attempt in $(seq 1 60); do
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

timeout "${duration}" tegrastats --interval 2000 >"${tegrastats_log}" 2>&1 &
tegrastats_pid=$!
docker exec -d \
  -e TEST_CONFIG="${config}" \
  "${depth_container}" bash -lc \
  'source /opt/ros/noetic/setup.bash
   source /root/swarm_ws/devel/setup.bash
   exec roslaunch quadcam_depth_est depth-node.launch \
     depth_config:="${TEST_CONFIG}" output:=screen' \
  >"${node_log}" 2>&1

cleanup() {
  kill "${tegrastats_pid}" 2>/dev/null || true
  wait "${tegrastats_pid}" 2>/dev/null || true
  docker exec "${depth_container}" bash -lc \
    'pkill -TERM -f quadcam_depth_est_node || true' >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 15
{
  echo "Started=$(date --iso-8601=seconds)"
  echo "DurationSeconds=${duration}"
  docker exec "${depth_container}" bash -lc '
    source /opt/ros/noetic/setup.bash
    echo "=== input metadata ==="
    rostopic echo -n 1 /oak_ffc_4p/assemble_image/encoding
    rostopic echo -n 1 /oak_ffc_4p/assemble_image/width
    rostopic echo -n 1 /oak_ffc_4p/assemble_image/height
    echo "=== initial input hz ==="
    timeout 12 rostopic hz /oak_ffc_4p/assemble_image || test $? = 124
    echo "=== initial pointcloud hz ==="
    timeout 20 rostopic hz /depth_estimation/pointcloud || test $? = 124
    rostopic echo -n 1 /depth_estimation/pointcloud/header
  '
  sleep "$(( duration > 50 ? duration - 50 : 1 ))"
  docker exec "${depth_container}" bash -lc '
    source /opt/ros/noetic/setup.bash
    echo "=== final input hz ==="
    timeout 12 rostopic hz /oak_ffc_4p/assemble_image || test $? = 124
    echo "=== final pointcloud hz ==="
    timeout 20 rostopic hz /depth_estimation/pointcloud || test $? = 124
    rostopic echo -n 1 /depth_estimation/pointcloud/header
  '
  echo "Finished=$(date --iso-8601=seconds)"
  echo "=== OAK container tail ==="
  docker logs --tail 100 "${oak_container}"
  echo "=== ROS errors ==="
  docker exec "${depth_container}" bash -lc \
    'grep -RIE "CUDA.*error|illegal memory|deserialize.*error|out of memory|OOM" \
      /root/.ros/log 2>/dev/null || true'
} 2>&1 | tee "${log}"

echo "PASS: live pipeline remained available for ${duration} seconds."
echo "Runtime log: ${log}"
echo "tegrastats: ${tegrastats_log}"
