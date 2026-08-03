#!/usr/bin/env bash
set -Eeuo pipefail

container="omninxt_omnidepth"
if docker ps --format '{{.Names}}' | grep -Fxq "${container}"; then
  docker exec "${container}" bash -lc \
    "pkill -TERM -f '[q]uadcam_depth_est_node' || true"
  for _ in $(seq 1 50); do
    if ! docker exec "${container}" bash -lc \
      "pgrep -f '[q]uadcam_depth_est_node' >/dev/null"; then
      break
    fi
    sleep .1
  done
fi
echo "OmniDepth node stopped. OAK camera container was left running."
