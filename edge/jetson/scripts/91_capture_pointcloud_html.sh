#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${root}/runtime/html_pointclouds/capture_${stamp}"
pcd="${output_dir}/pointcloud_sectors.pcd"
html="${output_dir}/pointcloud_viewer.html"

"${root}/scripts/85_start_live_omnidepth.sh"
mkdir -p "${output_dir}"

docker exec "${container}" bash -lc "
  source /opt/ros/noetic/setup.bash
  python3 /opt/omninxt/scripts/paired_image_colored_cloud_saver.py \
    /runtime/html_pointclouds/capture_${stamp} --timeout 30
"

python3 "${root}/scripts/pcd_to_standalone_html.py" \
  "${pcd}" "${html}" \
  --image "${output_dir}/assembled_labeled_preview.jpg"

echo "Image: ${output_dir}/assembled_CAM_A_B_C_D.png"
echo "Preview: ${output_dir}/assembled_labeled_preview.jpg"
echo "PCD: ${pcd}"
echo "HTML: ${html}"
if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${html}" >/dev/null 2>&1 &
else
  echo "Open the HTML file in Chrome/Chromium."
fi
