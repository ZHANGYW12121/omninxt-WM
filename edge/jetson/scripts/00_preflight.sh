#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${HOME}/OmniNxt"
REPORT="${ROOT}/reports/system_snapshot.txt"
mkdir -p "${ROOT}"/{source,camera_ws,calibration_ws,docker,scripts,logs,reports,backups,runtime}

run() {
  printf '\n$ %s\n' "$*"
  "$@" 2>&1 || true
}

{
  printf 'OmniDepth pre-calibration system snapshot\n'
  run date --iso-8601=seconds
  run cat /proc/device-tree/model
  run uname -a
  run uname -m
  run cat /etc/nv_tegra_release
  run cat /etc/os-release
  run dpkg-query -W nvidia-jetpack
  run nvcc --version
  run dpkg-query -W 'libnvinfer*'
  run dpkg-query -W 'libcudnn*'
  run dpkg-query -W 'nvidia-container*'
  run docker --version
  run docker info
  run lsblk -f
  run df -h
  run free -h
  run swapon --show
  run nvpmodel -q
  run lsusb
  run lsusb -t
  run ip -br addr
} | tee "${REPORT}"

printf '\nSnapshot saved to %s\n' "${REPORT}"
