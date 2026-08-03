#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${DPMPC_IMAGE:-dpmpc-planner-isaac:noetic}"
LOG="${DPMPC_BUILD_LOG:-${ROOT}/docker_build.log}"

docker build \
    --progress=plain \
    --file "${ROOT}/Dockerfile" \
    --tag "${IMAGE}" \
    "${ROOT}" 2>&1 | tee "${LOG}"
