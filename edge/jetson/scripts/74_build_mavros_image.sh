#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/neu/OmniNxt
docker build \
  --file docker/mavros/Dockerfile \
  --tag omninxt/mavros-noetic-arm64:jp512 \
  .

