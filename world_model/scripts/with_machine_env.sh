#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env.local" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env.local"
    set +a
fi

if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi
exec "$@"
