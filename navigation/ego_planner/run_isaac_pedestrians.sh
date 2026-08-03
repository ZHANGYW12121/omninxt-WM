#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.local/machine.env"
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-}}"
[[ -x "$ISAACSIM_ROOT/python.sh" ]] || {
  echo "ISAACSIM_ROOT must point to an Isaac Sim 5.1 installation." >&2
  exit 2
}
cd "$ROOT/simulation/isaacsim/database"
CLASSIC_AUTO_START=0 EGO_POINT_SOURCE=isaac_gt EGO_TEST_FIXED_ROUTE=1 \
  "$ISAACSIM_ROOT/python.sh" main.py --control-mode px4_ego "$@"
