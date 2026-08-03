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
  echo "Run tools/configure_machine.sh first." >&2
  exit 2
}
export ISAACSIM_ROOT
exec /usr/bin/env python3 "$SCRIPT_DIR/run_navigation_benchmark.py" "$@"
