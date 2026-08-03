#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
variant="${1:-t}"
min_batch="${RTMPOSE_MIN_BATCH:-1}"
opt_batch="${RTMPOSE_OPT_BATCH:-4}"
max_batch="${RTMPOSE_MAX_BATCH:-8}"
trtexec_bin="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"

case "${variant}" in
  t|s) ;;
  *) echo "Usage: $0 [t|s]" >&2; exit 2 ;;
esac
for value in "${min_batch}" "${opt_batch}" "${max_batch}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Batch sizes must be positive integers" >&2
    exit 2
  }
done
(( min_batch <= opt_batch && opt_batch <= max_batch )) || {
  echo "Expected min_batch <= opt_batch <= max_batch" >&2
  exit 2
}

onnx="${root}/models/rtmpose/rtmpose_${variant}_body17_256x192.onnx"
engine="${root}/models/rtmpose/tensorrt/rtmpose_${variant}_body17_256x192_fp16_dynamic_b${max_batch}.engine"
log="${root}/runtime/rtmpose_${variant}_dynamic_b${max_batch}_build.log"

[[ -x "${trtexec_bin}" ]] || {
  echo "TensorRT trtexec not found: ${trtexec_bin}" >&2
  exit 1
}
[[ -s "${onnx}" ]] || {
  echo "ONNX model not found: ${onnx}" >&2
  exit 1
}
mkdir -p "$(dirname "${engine}")" "$(dirname "${log}")"

echo "Building RTMPose-${variant^^} FP16 dynamic engine"
echo "ONNX: ${onnx}"
echo "Profile: min=${min_batch}, opt=${opt_batch}, max=${max_batch}"
echo "Engine: ${engine}"

"${trtexec_bin}" \
  --onnx="${onnx}" \
  --saveEngine="${engine}" \
  --fp16 \
  --minShapes="input:${min_batch}x3x256x192" \
  --optShapes="input:${opt_batch}x3x256x192" \
  --maxShapes="input:${max_batch}x3x256x192" \
  --memPoolSize=workspace:2048 \
  --buildOnly \
  --noDataTransfers \
  2>&1 | tee "${log}"

[[ -s "${engine}" ]] || {
  echo "Engine build did not produce a non-empty file" >&2
  exit 1
}
sha256sum "${onnx}" "${engine}"
