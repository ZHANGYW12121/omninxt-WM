#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAME="${1:-$(cat "$ROOT/shared/input/LATEST")}"

"$ROOT/process_isaac_gt_disparity.sh" "$FRAME"
docker exec d2slam_omni_depth python3 \
  /root/omninxt_offline/build_isaac_direct_gt.py "$FRAME" \
  --input-root /root/omninxt_shared/input \
  --output-root /root/omninxt_shared/output/isaac_direct_gt \
  --config /root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_isaac
docker exec d2slam_omni_depth python3 \
  /root/omninxt_offline/compare_gt_clouds.py "$FRAME" \
  --root /root/omninxt_shared/output
docker exec d2slam_omni_depth chown -R 1000:1000 \
  /root/omninxt_shared/output/isaac_direct_gt \
  /root/omninxt_shared/output/comparison

echo "Comparison: $ROOT/shared/output/comparison/$FRAME/comparison.png"
echo "Metrics:    $ROOT/shared/output/comparison/$FRAME/metrics.json"
