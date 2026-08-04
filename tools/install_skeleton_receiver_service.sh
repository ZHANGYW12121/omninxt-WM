#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/backend/systemd/omninxt-skeleton-receiver.service.in"
USER_DIR="${HOME}/.config/systemd/user"
DESTINATION="${USER_DIR}/omninxt-skeleton-receiver.service"

mkdir -p "${USER_DIR}"
escaped="$(printf '%s' "${REPO_ROOT}" | sed 's/[&|]/\\&/g')"
sed "s|@REPO_ROOT@|${escaped}|g" "${SOURCE}" > "${DESTINATION}"
systemctl --user daemon-reload
systemctl --user enable --now omninxt-skeleton-receiver.service
echo "Installed and started: ${DESTINATION}"
echo "Logs: journalctl --user -u omninxt-skeleton-receiver.service -f"
