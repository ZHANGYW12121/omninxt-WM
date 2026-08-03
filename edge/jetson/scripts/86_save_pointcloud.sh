#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
stamp="$(date +%Y%m%d_%H%M%S)"
output="${1:-${root}/runtime/pointclouds/pointcloud_${stamp}.pcd}"
output="$(realpath -m "${output}")"

case "${output}" in
  "${root}/runtime/"*) ;;
  *)
    echo "Output must be below ${root}/runtime" >&2
    exit 2
    ;;
esac

docker ps --format '{{.Names}}' | grep -Fxq "${container}" || {
  echo "Start live OmniDepth first: ${root}/scripts/85_start_live_omnidepth.sh" >&2
  exit 2
}
docker exec "${container}" bash -lc \
  "pgrep -f '[q]uadcam_depth_est_node' >/dev/null" || {
  echo "OmniDepth node is not running. Run scripts/85_start_live_omnidepth.sh" >&2
  exit 2
}

mkdir -p "$(dirname "${output}")"
container_output="/runtime/${output#"${root}/runtime/"}"
docker exec "${container}" bash -lc "
  source /opt/ros/noetic/setup.bash
  python3 /opt/omninxt/scripts/pointcloud_saver.py \
    '${container_output}' --topic /depth_estimation/pointcloud --timeout 30
  chown 1000:1000 '${container_output}'
"

[[ -s "${output}" ]] || {
  echo "Point cloud was not created: ${output}" >&2
  exit 1
}
points="$(awk '$1 == "POINTS" {print $2}' "${output}")"
echo "Saved ${points} finite XYZ points:"
echo "${output}"
