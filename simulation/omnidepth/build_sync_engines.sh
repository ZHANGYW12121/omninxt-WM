#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
if [[ -f "$REPO_ROOT/.local/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.local/machine.env"
fi
IMAGE="${OMNIDEPTH_IMAGE:-omnidepth:ada-sync-20260802}"
GPU="${OMNIDEPTH_GPU:-1}"
MODELS="${D2SLAM_ROOT:-$ROOT/D2SLAM}/models"

build_engine() {
  local onnx="$1" engine="$2"
  local temporary="${engine}.building"
  if [[ -s "$engine" && "$engine" -nt "$onnx" ]]; then
    echo "Engine is current: $engine"
    return
  fi
  rm -f "$temporary"
  docker run --rm --gpus "device=$GPU" \
    -v "$MODELS:/models" \
    --entrypoint bash "$IMAGE" -lc \
    "/usr/src/tensorrt/bin/trtexec --onnx='/models/${onnx#"$MODELS/"}' --saveEngine='/models/${temporary#"$MODELS/"}' --fp16 --workspace=2048 --buildOnly"
  [[ -s "$temporary" ]] || { echo "Engine build failed: $engine" >&2; exit 1; }
  mv "$temporary" "$engine"
}

# HITNet is built by the TensorRT-10 C++ node on first startup.  This image
# intentionally retains TensorRT-8 Python for the two pose engines because the
# base image ships both runtimes side by side.
build_engine \
  "$MODELS/rtmpose/yolox_nano_coco_416.onnx" \
  "$MODELS/rtmpose/yolox_nano_coco_416_sync_20260802_sm89.engine"
build_engine \
  "$MODELS/rtmpose/rtmpose_s_body17_256x192.onnx" \
  "$MODELS/rtmpose/rtmpose_s_body17_256x192_sync_20260802_sm89.engine"

echo "All TensorRT engines are ready for GPU $GPU."
