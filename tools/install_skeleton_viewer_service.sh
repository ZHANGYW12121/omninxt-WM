#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/backend/systemd/omninxt-skeleton-viewer.service.in"
USER_DIR="${HOME}/.config/systemd/user"
DESTINATION="${USER_DIR}/omninxt-skeleton-viewer.service"

mkdir -p "${USER_DIR}"
escaped="$(printf '%s' "${REPO_ROOT}" | sed 's/[&|]/\\&/g')"
sed "s|@REPO_ROOT@|${escaped}|g" "${SOURCE}" > "${DESTINATION}"
systemctl --user daemon-reload
systemctl --user enable --now omninxt-skeleton-viewer.service
echo "Installed and started: ${DESTINATION}"
echo "Open: http://${SKELETON_VIEWER_HOST:-127.0.0.1}:${SKELETON_VIEWER_PORT:-8767}"
echo "Logs: journalctl --user -u omninxt-skeleton-viewer.service -f"
