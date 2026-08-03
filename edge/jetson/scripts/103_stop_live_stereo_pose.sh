#!/usr/bin/env bash
set -Eeuo pipefail

container="omninxt_omnidepth"

if ! docker ps --format '{{.Names}}' | grep -Fxq "${container}"; then
  echo "${container} is not running; no pose process to stop."
  exit 0
fi

docker exec "${container}" bash -lc \
  "pkill -TERM -f '[1]02_live_stereo_pose.py' || true"

for _ in $(seq 1 30); do
  if ! docker exec "${container}" bash -lc \
      "pgrep -f '[1]02_live_stereo_pose.py' >/dev/null"; then
    echo "Sparse stereo pose node stopped."
    exit 0
  fi
  sleep .1
done

echo "Timed out waiting for sparse stereo pose node to stop." >&2
exit 1
