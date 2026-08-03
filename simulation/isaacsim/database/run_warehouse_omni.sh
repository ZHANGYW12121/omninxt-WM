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
  echo "ISAACSIM_ROOT must point to an Isaac Sim 5.1 installation." >&2
  exit 2
}
cd "$SCRIPT_DIR"
# This workflow does not use ROS 2. Do not prepend Isaac ROS 2 libraries: PX4 is
# auto-launched as a child and must inherit its normal runtime libraries.
unset ROS_DISTRO RMW_IMPLEMENTATION

# Keep capture settings overridable by the caller. By default the Warehouse
# scene opens with Omni-Depth export enabled and waits for a manual T press.
export OMNINXT_DEPTH_EXPORT="${OMNINXT_DEPTH_EXPORT:-1}"
export OMNINXT_GT_RANGE_EXPORT="${OMNINXT_GT_RANGE_EXPORT:-1}"
export OMNINXT_DEPTH_EXPORT_COUNT="${OMNINXT_DEPTH_EXPORT_COUNT:-3}"
export OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES="${OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES:-120}"
export OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES="${OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES:-0}"
export OMNINXT_DEPTH_EXPORT_NAV_ONLY="${OMNINXT_DEPTH_EXPORT_NAV_ONLY:-1}"
export CLASSIC_AUTO_START="${CLASSIC_AUTO_START:-0}"

exec "$ISAACSIM_ROOT/python.sh" main_warehouse_omni.py --reset-user --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 "$@"
