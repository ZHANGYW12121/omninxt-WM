#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAME="${1:-$(cat "$ROOT/shared/input/LATEST")}"

if python3 -c 'import cv2' >/dev/null 2>&1; then
  python3 "$ROOT/process_isaac_gt_disparity.py" "$FRAME"
  python3 "$ROOT/visualize_pointcloud_3d.py" \
    "$ROOT/shared/output/isaac_gt_disparity/${FRAME}_gt_disparity.pcd"
else
  docker exec d2slam_omni_depth python3 \
    /root/omninxt_offline/process_isaac_gt_disparity.py "$FRAME" \
    --input-root /root/omninxt_shared/input \
    --output-root /root/omninxt_shared/output/isaac_gt_disparity \
    --debug-root /root/omninxt_shared/debug/isaac_gt_disparity \
    --config /root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_isaac
  docker exec d2slam_omni_depth chown -R 1000:1000 \
    /root/omninxt_shared/output/isaac_gt_disparity \
    /root/omninxt_shared/debug/isaac_gt_disparity
  # Matplotlib is available on the host; the PCD reader/renderer does not need OpenCV.
  python3 "$ROOT/visualize_pointcloud_3d.py" \
    "$ROOT/shared/output/isaac_gt_disparity/${FRAME}_gt_disparity.pcd"
fi

echo "GT-disparity result: $ROOT/shared/output/isaac_gt_disparity/${FRAME}_gt_disparity.pcd"
