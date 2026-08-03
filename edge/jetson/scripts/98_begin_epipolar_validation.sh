#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt/runtime/epipolar_validation"
stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${root}/runs/epipolar_${stamp}"

mkdir -p "${run_dir}/raw"
printf '%s\n' "${run_dir}" >"${root}/ACTIVE_RUN"

echo "Epipolar validation run created:"
echo "  ${run_dir}"
echo
echo "Record in this order:"
echo "  ./scripts/98_record_epipolar_stage.sh AB_RIGHT"
echo "  ./scripts/98_record_epipolar_stage.sh BC_REAR"
echo "  ./scripts/98_record_epipolar_stage.sh CD_LEFT"
echo "  ./scripts/98_record_epipolar_stage.sh DA_FRONT"
