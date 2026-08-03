#!/usr/bin/env bash
set -Eeuo pipefail

runtime_root="/home/neu/OmniNxt/runtime/calibration"
template_root="${runtime_root}/templates"
stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${runtime_root}/runs/quarterkalibr_${stamp}"
active_file="${runtime_root}/ACTIVE_RUN"

mkdir -p "${run_dir}/raw" "${run_dir}/prepared" "${run_dir}/results"
cp "${template_root}/aprilgrid_USER_MUST_FILL.yaml" \
  "${run_dir}/april_6x6.yaml"
cp "${template_root}/imu_BMI088_PROVISIONAL.yaml" \
  "${run_dir}/imu_BMI088_PROVISIONAL.yaml"
printf '%s\n' "${run_dir}" >"${active_file}"

echo "Formal calibration run created:"
echo "${run_dir}"
echo
echo "Target: 6x6 AprilGrid, tagSize=0.088 m, tagSpacing=0.3"
echo "IMU: PX4 /mavros/imu/data_raw (BMI088, provisional noise model)"
echo "ACTIVE_RUN=${active_file}"

