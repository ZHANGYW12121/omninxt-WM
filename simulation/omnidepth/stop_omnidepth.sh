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
DOCKER="${DOCKER:-docker}"
NAME="${OMNIDEPTH_CONTAINER:-d2slam_omni_depth}"
"$DOCKER" rm -f "$NAME" >/dev/null 2>&1 || true
echo "OmniDepth stopped: $NAME"
