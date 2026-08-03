#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO_ROOT/simulation/isaacsim/database/systemd/navigation-benchmark-autoresume.service"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TARGET="$USER_UNIT_DIR/navigation-benchmark-autoresume.service"

[[ -f "$TEMPLATE" ]] || { echo "Missing service template: $TEMPLATE" >&2; exit 1; }
mkdir -p "$USER_UNIT_DIR"

python3 - "$TEMPLATE" "$TARGET" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

template, target, repo_root = map(Path, sys.argv[1:])
content = template.read_text(encoding="utf-8")
target.write_text(content.replace("@REPO_ROOT@", str(repo_root)), encoding="utf-8")
PY

systemctl --user daemon-reload
systemctl --user enable navigation-benchmark-autoresume.service
echo "Installed: $TARGET"
echo "Start with: systemctl --user start navigation-benchmark-autoresume.service"
