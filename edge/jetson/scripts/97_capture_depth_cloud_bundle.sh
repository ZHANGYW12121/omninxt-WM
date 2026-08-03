#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
stamp="$(date +%Y%m%d_%H%M%S)"
output="${1:-${root}/runtime/depth_cloud_bundles/capture_${stamp}}"
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
mkdir -p "${output}"
container_output="/runtime/${output#"${root}/runtime/"}"

docker exec "${container}" bash -lc "
  set -Eeuo pipefail
  source /opt/ros/noetic/setup.bash
  python3 /opt/omninxt/scripts/97_capture_depth_cloud_bundle.py \
    '${container_output}' \
    --timeout '${CAPTURE_TIMEOUT_S:-45}' \
    --min-depth '${DEPTH_VIEW_MIN_M:-0.2}' \
    --max-depth '${DEPTH_VIEW_MAX_M:-5.0}' \
    --max-disparity '${DISPARITY_VIEW_MAX_PX:-96}' \
    --max-cloud-delta-ns '${MAX_CLOUD_DELTA_NS:-1000}'
  chown -R 1000:1000 '${container_output}'
"

python3 "${root}/scripts/pcd_to_standalone_html.py" \
  "${output}/pointcloud_sectors.pcd" \
  "${output}/pointcloud_viewer.html" \
  --image "${output}/assembled_labeled_preview.jpg"

python3 - "${output}/metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)
if not metadata.get("image_debug_exact_header_match"):
    raise SystemExit("Image/debug topics do not have an exact header match")
if metadata.get("matched_topic_count") != 18:
    raise SystemExit("Expected 18 exact-header-matched topics")
if metadata["cloud_stamp_delta_ns"] > metadata["max_cloud_stamp_delta_ns"]:
    raise SystemExit("Point cloud exceeds the allowed PCL timestamp quantization")
print(
    "Image/debug ROS stamp:",
    metadata["stamp_ns"],
    "| cloud delta:",
    metadata["cloud_stamp_delta_ns"],
    "ns",
    "| points:",
    metadata["finite_xyzrgb_points"],
)
PY

echo
echo "Exact-frame depth and point-cloud bundle saved:"
echo "  Combined page: ${output}/capture_index.html"
echo "  Depth page:    ${output}/depth_viewer.html"
echo "  Point cloud:   ${output}/pointcloud_viewer.html"
echo "  Raw PCD:       ${output}/pointcloud_sectors.pcd"
echo "  Metadata:      ${output}/metadata.json"

if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${output}/capture_index.html" >/dev/null 2>&1 &
fi
