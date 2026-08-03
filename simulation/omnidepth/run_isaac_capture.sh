#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
[[ -f "$REPO_ROOT/.local/machine.env" ]] && source "$REPO_ROOT/.local/machine.env"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-}}"
[[ -x "$ISAACSIM_ROOT/python.sh" ]] || { echo "ISAACSIM_ROOT is not configured" >&2; exit 2; }
SHARED="$ROOT/shared"
mkdir -p "$SHARED/input" "$SHARED/output" "$SHARED/logs"
export OMNINXT_DEPTH_EXPORT=1
export OMNINXT_DEPTH_EXPORT_ROOT="$SHARED"
export OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES="${OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES:-30}"
export OMNINXT_DEPTH_EXPORT_NAV_ONLY="${OMNINXT_DEPTH_EXPORT_NAV_ONLY:-1}"
cd "$ROOT/../isaacsim/database"
echo "Isaac will export one synchronized IN-FLIGHT frame to $SHARED/input"
echo "Capture waits for takeoff/navigation, then 30 rendered frames."
echo "Keep Isaac running until [APP][OMNI-DEPTH] Exported synchronized frame appears."
exec "$ISAACSIM_ROOT/python.sh" main.py --no-headless \
  --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 "$@"
