#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container_name="omninxt_omnidepth"
config="/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp/quadcam_depth.yaml"
engine="${root}/source/D2SLAM/models/hitnet_series/hitnet_1x240x320_model_float16_quant_opt.trt"
bag="${1:-}"
duration="${SMOKE_DURATION:-30}"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${root}/runtime/offline_${stamp}"
log="${root}/logs/40_offline_smoke_${stamp}.log"

if [[ -z "${bag}" ]]; then
  bag="$(find "${root}/runtime" -maxdepth 1 -name 'oak4p_smoke_*.bag' -type f -printf '%T@ %p\n' |
    sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -f "${bag}" ]] || { echo "No input bag found" >&2; exit 2; }
case "$(realpath "${bag}")" in
  "${root}/runtime/"*) ;;
  *) echo "Input bag must be below ${root}/runtime" >&2; exit 2 ;;
esac
[[ -s "${engine}" ]] || { echo "Jetson-generated TensorRT engine is missing" >&2; exit 2; }
[[ "${duration}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SMOKE_DURATION must be a positive integer number of seconds" >&2
  exit 2
}
docker ps --format '{{.Names}}' | grep -Fxq "${container_name}" || {
  echo "Start ${container_name} first with 31_start_omnidepth_container.sh" >&2
  exit 2
}

mkdir -p "${output_dir}"
bag_in_container="/runtime/${bag#"${root}/runtime/"}"
pcd_in_container="/runtime/offline_${stamp}/pointcloud.pcd"

set -o pipefail
docker exec \
  -e TEST_BAG="${bag_in_container}" \
  -e TEST_CONFIG="${config}" \
  -e TEST_PCD="${pcd_in_container}" \
  -e TEST_DURATION="${duration}" \
  "${container_name}" bash -lc '
    set -Eeuo pipefail
    source /opt/ros/noetic/setup.bash
    source /root/swarm_ws/devel/setup.bash
    node_pid=""
    saver_pid=""
    cleanup() {
      [[ -z "${saver_pid}" ]] || kill "${saver_pid}" 2>/dev/null || true
      [[ -z "${node_pid}" ]] || kill "${node_pid}" 2>/dev/null || true
    }
    trap cleanup EXIT
    rostopic list >/dev/null 2>&1 || {
      roscore >/tmp/offline_roscore.log 2>&1 &
      for attempt in $(seq 1 30); do
        rostopic list >/dev/null 2>&1 && break
        sleep 0.2
      done
    }
    roslaunch quadcam_depth_est depth-node.launch \
      depth_config:="${TEST_CONFIG}" output:=screen &
    node_pid=$!
    sleep 5
    python3 /opt/omninxt/scripts/pointcloud_saver.py \
      "${TEST_PCD}" --timeout 180 &
    saver_pid=$!
    sleep 2
    rosbag play --quiet --duration="${TEST_DURATION}" "${TEST_BAG}"
    wait "${saver_pid}"
    saver_pid=""
  ' 2>&1 | tee "${log}"

pcd="${output_dir}/pointcloud.pcd"
[[ -s "${pcd}" ]] || { echo "PCD was not created" >&2; exit 1; }
points="$(awk '$1 == "POINTS" {print $2}' "${pcd}")"
[[ "${points}" =~ ^[1-9][0-9]*$ ]] || { echo "PCD POINTS is not positive" >&2; exit 1; }
echo "PASS: ${pcd}, POINTS=${points}"
echo "Log: ${log}"
