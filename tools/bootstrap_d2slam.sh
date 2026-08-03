#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO_ROOT/.local/third_party/D2SLAM"
POSE_MODELS_FROM=""
COMMIT=3a4b8071f6f9c40a151c7685aa738b717ce8916a
URL=https://github.com/HKUST-Aerial-Robotics/D2SLAM.git
PATCH="$REPO_ROOT/simulation/omnidepth/patches/d2slam_current_geometry_and_depth.patch"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --pose-models-from) POSE_MODELS_FROM="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/bootstrap_d2slam.sh [--target PATH] [--pose-models-from PATH]

Clones the pinned official D2SLAM commit and applies the repository patch.
--pose-models-from points to a directory containing the two RTMPose ONNX files.
TensorRT engines are intentionally rebuilt on each machine/GPU.
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

TARGET="$(realpath -m "$TARGET")"
if [[ ! -d "$TARGET/.git" ]]; then
  [[ ! -e "$TARGET" ]] || { echo "Target exists but is not a Git checkout: $TARGET" >&2; exit 1; }
  mkdir -p "$(dirname "$TARGET")"
  git clone "$URL" "$TARGET"
fi
git -C "$TARGET" fetch origin "$COMMIT"
git -C "$TARGET" checkout --detach "$COMMIT"

if git -C "$TARGET" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "D2SLAM patch is already applied."
else
  git -C "$TARGET" apply --check --whitespace=error "$PATCH"
  git -C "$TARGET" apply "$PATCH"
  echo "Applied current OmniNxt depth/config patch."
fi
# This prevents the source tree mounted under D2SLAM/D2SLAM from being
# rediscovered beside the built quadcam_depth_est package in the container.
touch "$TARGET/quadcam_depth_est/CATKIN_IGNORE"

if [[ -n "$POSE_MODELS_FROM" ]]; then
  POSE_MODELS_FROM="$(realpath "$POSE_MODELS_FROM")"
  mkdir -p "$TARGET/models/rtmpose"
  for model in yolox_nano_coco_416.onnx rtmpose_s_body17_256x192.onnx; do
    [[ -s "$POSE_MODELS_FROM/$model" ]] || { echo "Missing $POSE_MODELS_FROM/$model" >&2; exit 1; }
    ln -sfn "$POSE_MODELS_FROM/$model" "$TARGET/models/rtmpose/$model"
  done
  echo "Linked RTMPose ONNX models from $POSE_MODELS_FROM"
else
  echo "RTMPose models were not configured; use --pose-models-from before building engines."
fi

echo "D2SLAM_ROOT=$TARGET"
echo "Re-run tools/configure_machine.sh with --d2slam-root '$TARGET'."
