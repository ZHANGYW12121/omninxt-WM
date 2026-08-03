#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${root}/logs/10_prepare_docker_${stamp}.log"
required_images=(
  "nvcr.io/nvidia/l4t-jetpack:r35.4.1"
  "omninxt/oak4p-noetic-arm64:jp512"
)

mkdir -p "${root}/logs"
{
  docker version
  docker info --format \
    'Architecture={{.Architecture}} DefaultRuntime={{.DefaultRuntime}} Runtimes={{json .Runtimes}}'
  for image in "${required_images[@]}"; do
    docker image inspect "${image}" --format \
      'Image={{.RepoTags}} ID={{.Id}} Architecture={{.Architecture}}'
  done
  docker run --rm --runtime nvidia \
    nvcr.io/nvidia/l4t-jetpack:r35.4.1 \
    bash -lc 'uname -m; nvcc --version | tail -1; dpkg-query -W nvinfer8 libcudnn8'
} 2>&1 | tee "${log}"

echo "Docker/NVIDIA runtime is ready. Log: ${log}"
