#!/usr/bin/env bash
set -Eeuo pipefail

for container_name in omninxt_oak4p omninxt_omnidepth; do
  if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
    docker stop --time 15 "${container_name}" >/dev/null
    echo "Stopped ${container_name}; retained container, data, and logs."
  else
    echo "${container_name} is not running."
  fi
done
