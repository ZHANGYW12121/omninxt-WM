#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
container="omninxt_omnidepth"
stamp="$(date +%Y%m%d_%H%M%S)"
output="${1:-${root}/runtime/stereo_depth_views/capture_${stamp}}"
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
  python3 /opt/omninxt/scripts/96_capture_stereo_depth_views.py \
    '${container_output}' \
    --timeout 30 \
    --min-depth '${DEPTH_VIEW_MIN_M:-0.2}' \
    --max-depth '${DEPTH_VIEW_MAX_M:-5.0}' \
    --max-disparity '${DISPARITY_VIEW_MAX_PX:-96}'
  chown -R 1000:1000 '${container_output}'
"

echo
echo "Four-pair virtual stereo diagnostics saved:"
echo "  Overview: ${output}/depth_overview.png"
echo "  HTML:     ${output}/depth_viewer.html"
echo "  Metadata: ${output}/metadata.json"
echo
echo "Raw depth is stored as uint16 millimetres in each */depth_mm.png."
echo "Raw disparity is stored as uint16 disparity*256 in each */disparity_x256.png."

if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${output}/depth_viewer.html" >/dev/null 2>&1 &
fi
