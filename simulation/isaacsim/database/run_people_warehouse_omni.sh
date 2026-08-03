#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -f "$REPO_ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.local/machine.env"
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-}}"
[[ -x "$ISAACSIM_ROOT/python.sh" ]] || {
  echo "ISAACSIM_ROOT must point to an Isaac Sim 5.1 installation (missing python.sh)." >&2
  echo "Run: $REPO_ROOT/tools/configure_machine.sh --help" >&2
  exit 2
}
cd "$SCRIPT_DIR"
unset ROS_DISTRO RMW_IMPLEMENTATION

# Existing custom warehouse scene with the randomized 17-23 person crowd.
# main.py/app_config.py retain the current drone model and quad-fisheye rig.
export OMNINXT_DEPTH_EXPORT="${OMNINXT_DEPTH_EXPORT:-1}"
export OMNINXT_GT_RANGE_EXPORT="${OMNINXT_GT_RANGE_EXPORT:-1}"
export OMNINXT_DEPTH_EXPORT_COUNT="${OMNINXT_DEPTH_EXPORT_COUNT:-3}"
export OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES="${OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES:-120}"
export OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES="${OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES:-0}"
export OMNINXT_DEPTH_EXPORT_NAV_ONLY="${OMNINXT_DEPTH_EXPORT_NAV_ONLY:-1}"
export CLASSIC_AUTO_START="${CLASSIC_AUTO_START:-0}"
export WAREHOUSE_CROWD_MODE="${WAREHOUSE_CROWD_MODE:-server_v2}"
export WAREHOUSE_CROWD_LAYOUT="${WAREHOUSE_CROWD_LAYOUT:-sparse}"
export WAREHOUSE_DENSE_PROFILE="${WAREHOUSE_DENSE_PROFILE:-transverse40}"
export PEGASUS_PEOPLE_ASSET_ROOT="${PEGASUS_PEOPLE_ASSET_ROOT:-$REPO_ROOT/.local/assets/people/Characters}"

exec "$ISAACSIM_ROOT/python.sh" main.py \
  --reset-user --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 "$@"
