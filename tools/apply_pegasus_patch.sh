#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO_ROOT/.local/machine.env" ]] && source "$REPO_ROOT/.local/machine.env"
PEGASUS_ROOT="${PEGASUS_ROOT:-${ISAACSIM_ROOT:?configure ISAACSIM_ROOT first}/PegasusSimulator}"
PATCH="$REPO_ROOT/third_party/pegasus/patches/0001-portable-people-runtime.patch"
EXPECTED="bef3c57f2cfef9c6dacf3262969e01b59eeb7c7a"

[[ "$(git -C "$PEGASUS_ROOT" rev-parse HEAD)" == "$EXPECTED" ]] || {
  echo "Pegasus HEAD does not match $EXPECTED" >&2
  exit 1
}
if git -C "$PEGASUS_ROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "Pegasus runtime patch is already applied."
  exit 0
fi
git -C "$PEGASUS_ROOT" apply --check "$PATCH"
git -C "$PEGASUS_ROOT" apply "$PATCH"
echo "Applied Pegasus runtime patch."
