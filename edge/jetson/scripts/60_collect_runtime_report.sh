#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
stamp="$(date +%Y%m%d_%H%M%S)"
report="${root}/reports/runtime_${stamp}.txt"
binary="/root/swarm_ws/devel/lib/quadcam_depth_est/quadcam_depth_est_node"
engine="${root}/source/D2SLAM/models/hitnet_series/hitnet_1x240x320_model_float16_quant_opt.trt"
onnx="${root}/source/D2SLAM/models/hitnet_series/hitnet_1x240x320_model_float16_quant_opt.onnx"

{
  date --iso-8601=seconds
  uname -a
  dpkg-query -W nvidia-l4t-core nvidia-jetpack 2>/dev/null || true
  nvpmodel -q || true
  free -h
  df -h "${root}"
  docker version --format 'Docker client={{.Client.Version}} server={{.Server.Version}}'
  docker info --format \
    'Docker arch={{.Architecture}} default={{.DefaultRuntime}}'
  docker info --format \
    'Docker runtimes={{range $name, $runtime := .Runtimes}}{{$name}} {{end}}'
  docker image inspect omninxt/omnidepth-orin:jp512 --format \
    'OmniDepth image={{.Id}} arch={{.Architecture}}'
  docker exec omninxt_omnidepth bash -lc \
    'nvcc --version | tail -1
     dpkg-query -W libnvinfer8 libcudnn8'
  sha256sum "${onnx}"
  if [[ -s "${engine}" ]]; then
    stat --printf='Engine bytes=%s mtime=%y\n' "${engine}"
    sha256sum "${engine}"
  else
    echo "Engine=MISSING"
  fi
  git -C "${root}/source/D2SLAM" rev-parse HEAD
  git -C "${root}/source/D2SLAM" status --short
  docker exec omninxt_omnidepth bash -lc \
    "source /opt/ros/noetic/setup.bash
     source /root/swarm_ws/devel/setup.bash
     rospack find quadcam_depth_est
     roslaunch --files quadcam_depth_est depth-node.launch
     ldd '${binary}' | grep -E 'opencv|nvinfer|cudart' | sort -u"
} 2>&1 | tee "${report}"

echo "Saved ${report}"
