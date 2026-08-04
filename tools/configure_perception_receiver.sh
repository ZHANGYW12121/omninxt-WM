#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${PERCEPTION_RECEIVER_HOST:-0.0.0.0}"
PORT="${PERCEPTION_RECEIVER_PORT:-9766}"
RUNTIME_DIR="${PERCEPTION_RUNTIME_DIR:-${REPO_ROOT}/.local/run/perception_receiver}"
MATCH_TOLERANCE_MS="${PERCEPTION_MATCH_TOLERANCE_MS:-120}"

usage() {
  cat <<'EOF'
Usage: tools/configure_perception_receiver.sh [options]

  --host ADDRESS             Listen address (default: 0.0.0.0)
  --port PORT                OPB1 TCP port (default: 9766)
  --runtime-dir PATH         Machine-local atomic snapshot directory
  --match-tolerance-ms MS    Maximum image/depth timestamp delta (default: 120)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --runtime-dir) RUNTIME_DIR="$2"; shift 2 ;;
    --match-tolerance-ms) MATCH_TOLERANCE_MS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${PORT}" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || {
  echo "Invalid TCP port: ${PORT}" >&2; exit 2;
}
[[ "${MATCH_TOLERANCE_MS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "Invalid match tolerance: ${MATCH_TOLERANCE_MS}" >&2; exit 2;
}
RUNTIME_DIR="$(realpath -m "${RUNTIME_DIR}")"

mkdir -p "${REPO_ROOT}/.local" "${RUNTIME_DIR}"
ENV_FILE="${REPO_ROOT}/.local/machine.env"
TEMP_FILE="${ENV_FILE}.tmp.$$"
touch "${ENV_FILE}"
awk '!/^export PERCEPTION_(RECEIVER_HOST|RECEIVER_PORT|RUNTIME_DIR|MATCH_TOLERANCE_MS)=/' \
  "${ENV_FILE}" > "${TEMP_FILE}"
{
  printf 'export PERCEPTION_RECEIVER_HOST=%q\n' "${HOST}"
  printf 'export PERCEPTION_RECEIVER_PORT=%q\n' "${PORT}"
  printf 'export PERCEPTION_RUNTIME_DIR=%q\n' "${RUNTIME_DIR}"
  printf 'export PERCEPTION_MATCH_TOLERANCE_MS=%q\n' "${MATCH_TOLERANCE_MS}"
} >> "${TEMP_FILE}"
mv "${TEMP_FILE}" "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

echo "Configured ${ENV_FILE}"
echo "Perception receiver: ${HOST}:${PORT}"
echo "Runtime output: ${RUNTIME_DIR}"
echo "Image/depth match tolerance: ${MATCH_TOLERANCE_MS} ms"
