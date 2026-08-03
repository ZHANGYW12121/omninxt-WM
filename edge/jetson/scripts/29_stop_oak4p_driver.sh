#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_oak4p"

if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  docker stop --time 10 "${container_name}" >/dev/null
  echo "Stopped ${container_name}; container and logs were retained."
else
  echo "${container_name} is not running."
fi
