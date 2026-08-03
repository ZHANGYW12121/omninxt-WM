#!/usr/bin/env bash
set -euo pipefail

# The account may have been added to the docker group after this terminal was
# opened. Re-enter the script with that supplementary group instead of turning
# a Docker permission error into the misleading "container is not running".
if [[ "${OMNIDEPTH_DOCKER_GROUP_REEXEC:-0}" != 1 ]] \
    && ! docker info >/dev/null 2>&1 \
    && getent group docker | awk -F: -v user="$USER" '
         { n = split($4, members, ","); for (i = 1; i <= n; i++) if (members[i] == user) found = 1 }
         END { exit !found }
       '; then
  export OMNIDEPTH_DOCKER_GROUP_REEXEC=1
  printf -v _omnidepth_cmd '%q ' "$0" "$@"
  exec sg docker -c "$_omnidepth_cmd"
fi

if ! docker info >/dev/null 2>&1; then
  echo "Cannot access Docker. Re-login or run: newgrp docker" >&2
  exit 1
fi

ROOT="${OMNIDEPTH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SHARED="$ROOT/shared"
LATEST="$SHARED/input/LATEST"
if [[ "${1:-}" == "--latest" ]]; then
  echo "Using the most recently exported Isaac frame: $LATEST"
  while [[ ! -s "$LATEST" ]]; do sleep 1; done
  NAME="$(tr -d '\r\n' < "$LATEST")"
elif [[ -n "${1:-}" ]]; then
  NAME="$1"
else
  PREVIOUS=""
  [[ ! -s "$LATEST" ]] || PREVIOUS="$(tr -d '\r\n' < "$LATEST")"
  echo "Waiting for a NEW Isaac synchronized frame: $LATEST"
  [[ -z "$PREVIOUS" ]] || echo "Ignoring existing frame: $PREVIOUS"
  while true; do
    CURRENT=""
    [[ ! -s "$LATEST" ]] || CURRENT="$(tr -d '\r\n' < "$LATEST")"
    if [[ -n "$CURRENT" && "$CURRENT" != "$PREVIOUS" \
        && -f "$SHARED/input/$CURRENT/metadata.json" ]]; then
      NAME="$CURRENT"
      break
    fi
    sleep 1
  done
  echo "Detected new Isaac frame: $NAME"
fi
FRAME_HOST="$SHARED/input/$NAME"
FRAME_DOCKER="/root/omninxt_shared/input/$NAME"
[[ -f "$FRAME_HOST/metadata.json" ]] || { echo "Invalid frame: $FRAME_HOST"; exit 1; }
docker ps --format '{{.Names}}' | grep -qx d2slam_omni_depth || {
  echo "Container d2slam_omni_depth is not running."; exit 1;
}
STAMP="${NAME#frame_}"
OUT="/root/omninxt_shared/output/${NAME}.pcd"
DEBUG_HOST="$SHARED/debug/$NAME"
DEBUG_LATEST="$SHARED/debug/latest"
mkdir -p "$DEBUG_HOST" "$DEBUG_LATEST"
docker exec -d d2slam_omni_depth bash -lc \
  "source /opt/ros/noetic/setup.bash; python3 /root/omninxt_offline/pointcloud_saver.py --output '$OUT' > /root/omninxt_shared/logs/saver_${STAMP}.log 2>&1"
for sector in 0 1 2 3; do
  docker exec -d d2slam_omni_depth bash -lc \
    "source /opt/ros/noetic/setup.bash; python3 /root/omninxt_offline/pointcloud_saver.py --topic '/depth_estimation/pointcloud_sector_${sector}' --output '/root/omninxt_shared/debug/${NAME}/stereo${sector}.pcd' > '/root/omninxt_shared/logs/saver_${STAMP}_sector${sector}.log' 2>&1"
done
sleep 1
docker exec d2slam_omni_depth bash -lc \
  "source /opt/ros/noetic/setup.bash; python3 /root/omninxt_offline/ros1_offline_feeder.py --input-dir '$FRAME_DOCKER' --repeat-count 100"
for sector in 0 1 2 3; do
  for kind in left_rect right_rect disparity; do
    cp "$DEBUG_LATEST/stereo${sector}_${kind}.png" "$DEBUG_HOST/stereo${sector}_${kind}.png"
  cp "$DEBUG_LATEST/stereo${sector}_disparity.tiff" "$DEBUG_HOST/stereo${sector}_disparity.tiff"
  done
done
echo "Waiting for DenseMap outputs..."
for _ in $(seq 1 120); do
  [[ -f "$SHARED/output/${NAME}_densemap.png" ]] && break
  sleep 1
done
python3 "$ROOT/visualize_pointcloud_3d.py" "$SHARED/output/${NAME}.pcd"
OWNER="$(stat -c '%u:%g' "$ROOT")"
docker exec d2slam_omni_depth chown "$OWNER" \
  "/root/omninxt_shared/output/${NAME}.pcd" \
  "/root/omninxt_shared/output/${NAME}.ply" \
  "/root/omninxt_shared/output/${NAME}_densemap.png" \
  "/root/omninxt_shared/output/${NAME}_3d.html" \
  "/root/omninxt_shared/output/${NAME}_3d.png"
ls -lh "$SHARED/output/${NAME}.pcd" "$SHARED/output/${NAME}.ply" "$SHARED/output/${NAME}_densemap.png"
ls -lh "$SHARED/output/${NAME}_3d.html" "$SHARED/output/${NAME}_3d.png"
