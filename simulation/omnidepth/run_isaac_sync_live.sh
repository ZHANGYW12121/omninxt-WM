#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
if [[ -f "$REPO_ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.local/machine.env"
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-}}"
[[ -x "$ISAACSIM_ROOT/python.sh" ]] || {
  echo "ISAACSIM_ROOT must point to an Isaac Sim 5.1 installation." >&2
  exit 2
}
MODE="${OMNINXT_SENSOR_MODE:-raw_mei}"
case "$MODE" in
  raw_mei|rectified_validation) ;;
  *) echo "OMNINXT_SENSOR_MODE must be raw_mei or rectified_validation" >&2; exit 2 ;;
esac

rm -rf /dev/shm/omninxt_sync/live
mkdir -p /dev/shm/omninxt_sync/live
"$ROOT/start_sync_backend.sh"

if [[ "${OMNINXT_OPEN_VIEWER:-1}" == 1 ]]; then
  (
    for _ in $(seq 1 120); do
      if curl -fsS http://127.0.0.1:8766/status.json >/dev/null 2>&1; then
        xdg-open http://127.0.0.1:8766 >/dev/null 2>&1 || true
        exit 0
      fi
      sleep 0.5
    done
  ) &
fi

export OMNINXT_SENSOR_MODE="$MODE"
export OMNINXT_LIVE_STREAM=1
export OMNINXT_LIVE_STREAM_ROOT=/dev/shm/omninxt_sync
export OMNINXT_LIVE_STREAM_HZ="${OMNINXT_LIVE_STREAM_HZ:-10}"
export OMNINXT_DEPTH_EXPORT=0
# The live depth backend derives disparity from Isaac's synchronized
# distance_to_camera ground truth.  This also attaches the annotator before
# the four cameras start streaming.
export OMNINXT_GT_RANGE_EXPORT=1
# Keep 2D detection/tracking realistic while replacing only the unreliable
# simulation depth estimator with exact-timestamp Isaac ground-truth depth.
export OMNINXT_POSE_DEPTH_SOURCE="${OMNINXT_POSE_DEPTH_SOURCE:-isaac_gt}"

echo "Starting Isaac Sim: mode=$MODE, live=${OMNINXT_LIVE_STREAM_HZ}Hz"
echo "Depth + skeleton viewer: http://127.0.0.1:8766"
cd "$ROOT/../isaacsim/database"
exec "$ISAACSIM_ROOT/python.sh" main.py --no-headless \
  --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 "$@"
