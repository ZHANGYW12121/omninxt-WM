#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
if [[ -f "$REPO_ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.local/machine.env"
fi
D2SLAM_ROOT="${D2SLAM_ROOT:-$ROOT/D2SLAM}"
NAME="${OMNIDEPTH_CONTAINER:-d2slam_omni_depth}"
DET_ENGINE="/root/swarm_ws/src/D2SLAM/models/rtmpose/yolox_nano_coco_416_sync_20260802_sm89.engine"
POSE_ENGINE="/root/swarm_ws/src/D2SLAM/models/rtmpose/rtmpose_s_body17_256x192_sync_20260802_sm89.engine"
MAX_PERSON_RANGE="${OMNINXT_MAX_PERSON_RANGE:-0}"
POSE_DEPTH_SOURCE="${OMNINXT_POSE_DEPTH_SOURCE:-isaac_gt}"
SKELETON_BACKEND_HOST="${OMNINXT_SKELETON_BACKEND_HOST:-}"
SKELETON_BACKEND_PORT="${OMNINXT_SKELETON_BACKEND_PORT:-9765}"
if [[ "${OMNINXT_DATA_RECORD_ENABLED:-0}" == 1 && -z "$SKELETON_BACKEND_HOST" ]]; then
  SKELETON_BACKEND_HOST=127.0.0.1
fi
case "$POSE_DEPTH_SOURCE" in
  hybrid|isaac_gt) ;;
  *) echo "OMNINXT_POSE_DEPTH_SOURCE must be hybrid or isaac_gt" >&2; exit 2 ;;
esac

for file in \
  "$D2SLAM_ROOT/models/rtmpose/yolox_nano_coco_416_sync_20260802_sm89.engine" \
  "$D2SLAM_ROOT/models/rtmpose/rtmpose_s_body17_256x192_sync_20260802_sm89.engine"; do
  [[ -s "$file" ]] || { echo "Missing $file; run ./build_sync_engines.sh" >&2; exit 1; }
done

"$ROOT/start_omnidepth.sh"
docker exec "$NAME" bash -lc \
  "pkill -TERM -f '[r]os1_live_feeder.py' || true; pkill -TERM -f '[1]02_live_stereo_pose.py' || true"

docker exec -d "$NAME" bash -lc '
  source /opt/ros/noetic/setup.bash
  source /root/swarm_ws/devel/setup.bash
  exec env PYTHONUNBUFFERED=1 python3 /opt/omninxt/runtime/ros1_live_feeder.py \
    > /root/omninxt_shared/logs/sync_live_feeder.log 2>&1
'

if [[ "${OMNINXT_POSE_ENABLED:-1}" == 1 ]]; then
  BACKEND_ARGS=""
  if [[ -n "$SKELETON_BACKEND_HOST" ]]; then
    BACKEND_ARGS="--backend-host '$SKELETON_BACKEND_HOST' --backend-port '$SKELETON_BACKEND_PORT'"
  fi
  docker exec -d \
    -e "PYTHONPATH=/opt/omninxt/runtime/rtmlib:/opt/omninxt/runtime" \
    "$NAME" bash -lc "
      source /opt/ros/noetic/setup.bash
      source /root/swarm_ws/devel/setup.bash
      exec env PYTHONUNBUFFERED=1 python3 /opt/omninxt/runtime/102_live_stereo_pose.py \
        --config-dir /root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_sync_20260802 \
        --det-engine '$DET_ENGINE' \
        --pose-engine '$POSE_ENGINE' \
        --max-person-range '$MAX_PERSON_RANGE' \
        --joint-depth-source '$POSE_DEPTH_SOURCE' \
        $BACKEND_ARGS \
        > /root/omninxt_shared/logs/sync_pose.log 2>&1
    "
fi

echo "Sim2Real backend ready. Feeder waits for /dev/shm/omninxt_sync/live/LATEST."
echo "Logs: $ROOT/shared/logs/sync_live_feeder.log and sync_pose.log"
