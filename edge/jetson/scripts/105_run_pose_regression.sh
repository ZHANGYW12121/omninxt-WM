#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
bag="${1:-}"
expected="${2:-}"
label="${3:-regression}"
master_port="${POSE_REGRESSION_MASTER_PORT:-11312}"

[[ -f "${bag}" ]] || {
  echo "Usage: $0 BAG EXPECTED_PEOPLE [LABEL]" >&2
  exit 2
}
[[ "${expected}" =~ ^[0-9]+$ ]] || {
  echo "EXPECTED_PEOPLE must be a non-negative integer" >&2
  exit 2
}
docker ps --format '{{.Names}}' | grep -Fxq "${container}" || \
  "${root}/scripts/31_start_omnidepth_container.sh"

stamp="$(date +%Y%m%d_%H%M%S)"
output_host="${root}/runtime/pose_regression/${label}_${stamp}"
mkdir -p "${output_host}"
output_container="/runtime/pose_regression/${label}_${stamp}"
bag_container="/runtime/${bag#${root}/runtime/}"
if [[ "${bag_container}" == "/runtime/${bag}" ]]; then
  echo "Bag must be under ${root}/runtime" >&2
  exit 2
fi

echo "Regression dataset: ${bag}"
echo "Expected people: ${expected}"
echo "Output: ${output_host}"

docker exec \
  -e ROS_MASTER_URI="http://127.0.0.1:${master_port}" \
  -e REG_BAG="${bag_container}" \
  -e REG_OUTPUT="${output_container}" \
  -e REG_EXPECTED="${expected}" \
  -e REG_MASTER_PORT="${master_port}" \
  "${container}" bash -lc '
    set -Eeuo pipefail
    source /opt/ros/noetic/setup.bash
    source /root/swarm_ws/devel/setup.bash
    export PYTHONPATH=/opt/omninxt/rtmlib:/opt/omninxt/scripts:${PYTHONPATH:-}
    mkdir -p "${REG_OUTPUT}"
    cleanup() {
      for pid in "${pose_pid:-}" "${depth_pid:-}" "${logger_pid:-}" \
                 "${master_pid:-}"; do
        [[ -n "${pid}" ]] && kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
      done
      sleep .5
      for pid in "${pose_pid:-}" "${depth_pid:-}" "${logger_pid:-}" \
                 "${master_pid:-}"; do
        [[ -n "${pid}" ]] && kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
      done
      wait >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM

    setsid roscore -p "${REG_MASTER_PORT}" >"${REG_OUTPUT}/roscore.log" 2>&1 &
    master_pid=$!
    for _ in $(seq 1 50); do
      rosnode list >/dev/null 2>&1 && break
      sleep .1
    done

    setsid roslaunch quadcam_depth_est depth-node.launch \
      depth_config:=/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp/quadcam_depth.yaml \
      enable_pointcloud_output:=false output:=screen \
      >"${REG_OUTPUT}/omnidepth.log" 2>&1 &
    depth_pid=$!

    setsid python3 /opt/omninxt/scripts/105_pose_regression_logger.py \
      --output-dir "${REG_OUTPUT}" \
      --expected-people "${REG_EXPECTED}" \
      >"${REG_OUTPUT}/logger.log" 2>&1 &
    logger_pid=$!

    setsid python3 /opt/omninxt/scripts/102_live_stereo_pose.py \
      --config-dir /root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp \
      --det-engine /opt/omninxt/models/rtmpose/tensorrt/yolox_nano_coco_416_fp16.engine \
      --pose-engine /opt/omninxt/models/rtmpose/tensorrt/rtmpose_t_body17_256x192_fp16_dynamic_b8.engine \
      --no-web >"${REG_OUTPUT}/pose.log" 2>&1 &
    pose_pid=$!

    for _ in $(seq 1 160); do
      nodes="$(rosnode list 2>/dev/null || true)"
      if grep -Fxq /quadcam_depth_est <<<"${nodes}" && \
         grep -Fxq /omninxt_sparse_stereo_pose <<<"${nodes}"; then
        break
      fi
      sleep .25
    done
    nodes="$(rosnode list)"
    grep -Fxq /quadcam_depth_est <<<"${nodes}"
    grep -Fxq /omninxt_sparse_stereo_pose <<<"${nodes}"

    rosbag play --quiet "${REG_BAG}" >"${REG_OUTPUT}/rosbag.log" 2>&1
    sleep 6
    wait "${logger_pid}"
  '

cat "${output_host}/summary.json"
echo "Completed: ${output_host}"
