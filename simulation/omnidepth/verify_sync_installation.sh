#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
[[ -f "$REPO_ROOT/.local/machine.env" ]] && source "$REPO_ROOT/.local/machine.env"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-}}"
D2SLAM_ROOT="${D2SLAM_ROOT:-$ROOT/D2SLAM}"
IMAGE="${OMNIDEPTH_IMAGE:-omnidepth:ada-sync-20260802}"

require_file() {
  [[ -s "$1" ]] || { echo "[FAIL] missing or empty: $1" >&2; exit 1; }
  echo "[OK] $1"
}

echo "== Isaac geometry =="
require_file "$ROOT/geometry/sim_geometry.yaml"
require_file "$ROOT/../isaacsim/database/omninxt_sync_20260802_camera_config.json"
(cd "$ROOT/../isaacsim/database" && "$ISAACSIM_ROOT/python.sh" test_omninxt_sync_geometry.py)
grep -q 'mavlink stream -r 5 -s HEARTBEAT -u \$udp_offboard_port_local' \
  "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink"
grep -q 'mavlink stream -r 5 -s HEARTBEAT -u \$udp_offboard_port_local' \
  "$PX4_DIR/build/px4_sitl_default/etc/init.d-posix/px4-rc.mavlink"
echo "[OK] PX4 offboard heartbeat: 5 Hz"

echo "== Runtime configuration and engines =="
for file in \
  "$D2SLAM_ROOT/config/quadcam_drone_nxt_sync_20260802/fisheye_cams.yaml" \
  "$D2SLAM_ROOT/config/quadcam_drone_nxt_sync_20260802/quadcam_depth.yaml" \
  "$D2SLAM_ROOT/models/hitnet_series/hitnet_1x240x320_model_float16_quant_opt_sync_20260802_sm89.trt" \
  "$D2SLAM_ROOT/models/rtmpose/yolox_nano_coco_416_sync_20260802_sm89.engine" \
  "$D2SLAM_ROOT/models/rtmpose/rtmpose_s_body17_256x192_sync_20260802_sm89.engine"; do
  require_file "$file"
done

docker image inspect "$IMAGE" >/dev/null
echo "[OK] Docker image: $IMAGE"

if docker ps --format '{{.Names}}' | grep -qx d2slam_omni_depth; then
  docker exec d2slam_omni_depth bash -lc \
    'source /opt/ros/noetic/setup.bash; rostopic info /oak_ffc_4p/assemble_image' \
    | grep -q /quadcam_depth_est
  echo "[OK] running C++ depth subscriber"
else
  echo "[INFO] backend is stopped; live ROS check skipped"
fi

echo "SYNC_INSTALLATION_OK"
