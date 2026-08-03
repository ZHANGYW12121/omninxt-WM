#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.local/machine.env"
fi

COMMIT="3725bcc2e7c1be4ecf1455d922299ae85042603a"
EXPECTED="51fa3dbdc6ba89626b5dad3a4638deb53d40aa6f0caa9b289657da4a8e0b60c3"
URL="https://raw.githubusercontent.com/Zhefan-Xu/NavRL/${COMMIT}/quick-demos/ckpts/navrl_checkpoint.pt"
ASSET_ROOT="${OMNINXT_ASSET_ROOT:-$ROOT/.local/assets}"
DESTINATION="${NAVRL_CHECKPOINT:-$ASSET_ROOT/models/navrl/navrl_checkpoint.pt}"
mkdir -p "$(dirname "$DESTINATION")"

if [[ -f "$DESTINATION" ]] && echo "$EXPECTED  $DESTINATION" | sha256sum -c - >/dev/null 2>&1; then
  echo "NavRL checkpoint already verified: $DESTINATION"
  exit 0
fi

tmp="${DESTINATION}.tmp.$$"
trap 'rm -f "$tmp"' EXIT
curl --fail --location --retry 3 --output "$tmp" "$URL"
echo "$EXPECTED  $tmp" | sha256sum -c -
mv "$tmp" "$DESTINATION"
trap - EXIT
echo "Installed NavRL checkpoint: $DESTINATION"
