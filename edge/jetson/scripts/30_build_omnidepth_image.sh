#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
source_root="${root}/source"
log_dir="${root}/logs"
base_tag="omninxt/jetson-orin-base:opencv-ccalib-r35.4.1"
final_tag="omninxt/omnidepth-orin:jp512"
timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"

if [[ "${REBUILD_OPENCV:-0}" == "1" ]]; then
  docker build \
    --build-arg USE_PROC="${USE_PROC:-2}" \
    -f "${source_root}/D2SLAM/docker/Dockerfile.jetson_orin_base_35.4.1" \
    -t "${base_tag}" \
    "${source_root}/deps" \
    2>&1 | tee "${log_dir}/30_rebuild_opencv_${timestamp}.log"
fi

docker image inspect "${base_tag}" >/dev/null 2>&1 || {
  echo "Missing ${base_tag}; run with REBUILD_OPENCV=1." >&2
  exit 1
}

docker build \
  --build-arg USE_PROC="${USE_PROC:-2}" \
  -f "${source_root}/D2SLAM/docker/Dockerfile.omnidepth_orin_jp512" \
  -t "${final_tag}" \
  "${source_root}" \
  2>&1 | tee "${log_dir}/30_build_omnidepth_${timestamp}.log"

docker image inspect "${final_tag}" \
  --format 'ID={{.Id}} Architecture={{.Architecture}}'
