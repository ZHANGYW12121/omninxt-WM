#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${SKELETON_RECEIVER_HOST:-0.0.0.0}"
PORT="${SKELETON_RECEIVER_PORT:-9765}"
MAX_PEOPLE="${SKELETON_MAX_PEOPLE:-20}"
RUNTIME_DIR="${SKELETON_RUNTIME_DIR:-${REPO_ROOT}/.local/run/skeleton_receiver}"

usage() {
  cat <<'EOF'
Usage: tools/configure_skeleton_receiver.sh [options]

  --host ADDRESS       Listen address (default: 0.0.0.0 for Nano LAN access)
  --port PORT          TCP port (default: 9765)
  --max-people COUNT   World-model Human slots (default: 20)
  --runtime-dir PATH   Machine-local latest-frame/status directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-people) MAX_PEOPLE="$2"; shift 2 ;;
    --runtime-dir) RUNTIME_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${PORT}" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || {
  echo "Invalid TCP port: ${PORT}" >&2; exit 2;
}
[[ "${MAX_PEOPLE}" =~ ^[0-9]+$ ]] && (( MAX_PEOPLE >= 1 )) || {
  echo "Invalid max-people: ${MAX_PEOPLE}" >&2; exit 2;
}
RUNTIME_DIR="$(realpath -m "${RUNTIME_DIR}")"

mkdir -p "${REPO_ROOT}/.local" "${RUNTIME_DIR}"
ENV_FILE="${REPO_ROOT}/.local/machine.env"
TEMP_FILE="${ENV_FILE}.tmp.$$"
touch "${ENV_FILE}"
awk '!/^export SKELETON_(RECEIVER_HOST|RECEIVER_PORT|MAX_PEOPLE|RUNTIME_DIR)=/' \
  "${ENV_FILE}" > "${TEMP_FILE}"
{
  printf 'export SKELETON_RECEIVER_HOST=%q\n' "${HOST}"
  printf 'export SKELETON_RECEIVER_PORT=%q\n' "${PORT}"
  printf 'export SKELETON_MAX_PEOPLE=%q\n' "${MAX_PEOPLE}"
  printf 'export SKELETON_RUNTIME_DIR=%q\n' "${RUNTIME_DIR}"
} >> "${TEMP_FILE}"
mv "${TEMP_FILE}" "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

echo "Configured ${ENV_FILE}"
echo "Receiver: ${HOST}:${PORT}"
echo "Runtime output: ${RUNTIME_DIR}"
