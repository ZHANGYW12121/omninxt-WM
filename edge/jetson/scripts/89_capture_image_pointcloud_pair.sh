#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
stamp="$(date +%Y%m%d_%H%M%S)"
output="${1:-${root}/runtime/paired_captures/capture_${stamp}}"
output="$(realpath -m "${output}")"

case "${output}" in
  "${root}/runtime/"*) ;;
  *)
    echo "Output directory must be below ${root}/runtime" >&2
    exit 2
    ;;
esac
[[ ! -e "${output}" ]] || {
  echo "Refusing to overwrite existing output: ${output}" >&2
  exit 2
}

"${root}/scripts/85_start_live_omnidepth.sh"
container_output="/runtime/${output#"${root}/runtime/"}"
docker exec "${container}" bash -lc "
  source /opt/ros/noetic/setup.bash
  python3 /opt/omninxt/scripts/paired_image_pointcloud_saver.py \
    '${container_output}' --timeout 30 --max-delta-ms 60
  chown -R 1000:1000 '${container_output}'
"

MPLBACKEND=Agg python3 "${root}/scripts/87_view_pcd.py" \
  "${output}/pointcloud.pcd" \
  --save "${output}/pointcloud_preview.png"

echo "Saved matched image/point-cloud capture:"
echo "${output}"
echo "Image: ${output}/assembled_labeled_preview.png"
echo "Cloud: ${output}/pointcloud.pcd"
echo "3D preview: ${output}/pointcloud_preview.png"
echo "Metadata: ${output}/metadata.json"
