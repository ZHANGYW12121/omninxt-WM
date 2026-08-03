#!/usr/bin/env bash
set -euo pipefail
NAME="${OMNIDEPTH_CONTAINER:-d2slam_omni_depth}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "Stopped $NAME"
