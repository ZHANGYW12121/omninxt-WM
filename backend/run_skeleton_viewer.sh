#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.local/machine.env"
fi

export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"
exec python3 -m backend.skeleton_viewer.server "$@"
