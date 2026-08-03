#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.local/machine.env"
[[ -f "$ENV_FILE" ]] || {
  echo "[FAIL] missing $ENV_FILE; run tools/configure_machine.sh" >&2
  exit 1
}
# shellcheck disable=SC1090
source "$ENV_FILE"

fail=0
check_file() {
  if [[ -e "$1" ]]; then echo "[OK] $1"; else echo "[FAIL] $1" >&2; fail=1; fi
}
check_hash() {
  local expected="$1" path="$2" actual
  check_file "$path"
  [[ -e "$path" ]] || return
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" == "$expected" ]]; then
    echo "[OK] sha256 $path"
  else
    echo "[FAIL] sha256 $path expected=$expected actual=$actual" >&2
    fail=1
  fi
}

check_file "$ISAACSIM_ROOT/python.sh"
check_file "$PX4_DIR/build/px4_sitl_default/bin/px4"
check_file "$D2SLAM_ROOT/config"
check_file "$D2SLAM_ROOT/models"
check_file "$PEGASUS_PEOPLE_ASSET_ROOT"
check_hash 2f3eb49cf492e521ee4a7b7c682be14b9dfaacab0176bc24ecd8d49bb113d7cd "$WAREHOUSE_USD"
check_hash 4c796f07f694b2f8a89201cadbc56ce64f605a514ac06698cf24a9805d29b8ba "$OMNINXT_VISUAL_ASSET_DIR/Omininxt_body.usdc"
check_hash e6d7c6169f78410a83f7ea763c799f9659ccfe101c65b4acbe296109ea4cc6a6 "$OMNINXT_VISUAL_ASSET_DIR/fl.usdc"
check_hash 95cf0952a57918a157d86747fe5be0b91489a518f8e691c0f520c1b12d9c63df "$OMNINXT_VISUAL_ASSET_DIR/fr.usdc"
check_hash 11f637b667ad3f12e497d223d51da0195eb6771b2d9d2fd88cca68f9ded87aad "$OMNINXT_VISUAL_ASSET_DIR/rl.usdc"
check_hash 973937f074de91bfe228447b75167ba96568b2e55f002c29c02c6c47f73303b4 "$OMNINXT_VISUAL_ASSET_DIR/rr.usdc"

px4_expected=6ea3539157ca358c70a515878b77077af7d4611d
px4_actual="$(git -C "$PX4_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ "$px4_actual" == "$px4_expected" ]]; then
  echo "[OK] PX4 commit $px4_actual"
else
  echo "[FAIL] PX4 commit expected=$px4_expected actual=$px4_actual" >&2
  fail=1
fi

pegasus_expected=bef3c57f2cfef9c6dacf3262969e01b59eeb7c7a
pegasus_root="${PEGASUS_ROOT:-$ISAACSIM_ROOT/PegasusSimulator}"
pegasus_actual="$(git -C "$pegasus_root" rev-parse HEAD 2>/dev/null || true)"
if [[ "$pegasus_actual" == "$pegasus_expected" ]]; then
  echo "[OK] Pegasus commit $pegasus_actual"
else
  echo "[FAIL] Pegasus commit expected=$pegasus_expected actual=$pegasus_actual" >&2
  fail=1
fi
if git -C "$pegasus_root" apply --reverse --check \
    "$REPO_ROOT/third_party/pegasus/patches/0001-portable-people-runtime.patch" >/dev/null 2>&1; then
  echo "[OK] Pegasus people runtime patch applied"
else
  echo "[FAIL] Pegasus people runtime patch is not applied" >&2
  fail=1
fi

python3 - "$REPO_ROOT" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = list(root.glob("simulation/**/*.py"))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"[OK] parsed {len(paths)} Python files")
PY

if (( fail )); then
  echo "DOCTOR_FAILED" >&2
  exit 1
fi
echo "DOCTOR_OK"
