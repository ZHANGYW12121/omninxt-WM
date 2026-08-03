#!/usr/bin/env bash
set -euo pipefail

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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
if [[ -f "$REPO_ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.local/machine.env"
fi
DOCKER="${DOCKER:-docker}"
NAME="${OMNIDEPTH_CONTAINER:-d2slam_omni_depth}"
GPU="${OMNIDEPTH_GPU:-1}"
IMAGE="${OMNIDEPTH_IMAGE:-omnidepth:ada-sync-20260802}"
D2SLAM_ROOT="${D2SLAM_ROOT:-$ROOT/D2SLAM}"
[[ -d "$D2SLAM_ROOT/config" && -d "$D2SLAM_ROOT/models" ]] || {
  echo "D2SLAM_ROOT is not configured or incomplete: $D2SLAM_ROOT" >&2
  exit 2
}

mkdir -p "$ROOT/shared"/{input,output,logs,debug,config}
mkdir -p /dev/shm/omninxt_sync/live
"$DOCKER" rm -f "$NAME" >/dev/null 2>&1 || true
"$DOCKER" run -d --gpus "device=$GPU" --network host --ipc host \
  -e "OMNIDEPTH_POINTCLOUD_OUTPUT=${OMNIDEPTH_POINTCLOUD_OUTPUT:-true}" \
  -v "$D2SLAM_ROOT:/root/swarm_ws/src/D2SLAM/D2SLAM" \
  -v "$D2SLAM_ROOT/config:/root/swarm_ws/src/D2SLAM/config" \
  -v "$D2SLAM_ROOT/models:/root/swarm_ws/src/D2SLAM/models" \
  -v "$ROOT/shared:/root/omninxt_shared" \
  -v "$ROOT/runtime:/opt/omninxt/runtime:ro" \
  -v "$ROOT:/root/omninxt_offline:ro" \
  --entrypoint /bin/bash \
  --name "$NAME" "$IMAGE" -lc '
    source /opt/ros/noetic/setup.bash
    source /root/swarm_ws/devel/setup.bash
    roscore > /root/omninxt_shared/logs/roscore.log 2>&1 &
    until rosparam list >/dev/null 2>&1; do sleep 0.5; done
    exec roslaunch quadcam_depth_est depth-node.launch output:=screen \
      depth_config:=/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_sync_20260802/quadcam_depth.yaml \
      enable_pointcloud_output:=${OMNIDEPTH_POINTCLOUD_OUTPUT:-true}
  '

for _ in $(seq 1 180); do
  if ! "$DOCKER" ps --format '{{.Names}}' | grep -qx "$NAME"; then
    "$DOCKER" logs "$NAME" >&2 || true
    echo "OmniDepth container exited during startup." >&2
    exit 1
  fi
  if "$DOCKER" exec "$NAME" bash -lc \
      'source /opt/ros/noetic/setup.bash; rostopic info /oak_ffc_4p/assemble_image 2>/dev/null' \
      | grep -q '/quadcam_depth_est'; then
    echo "OmniDepth is ready: container=$NAME GPU=$GPU"
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for the OmniDepth image subscriber." >&2
"$DOCKER" logs --tail 100 "$NAME" >&2 || true
exit 1
