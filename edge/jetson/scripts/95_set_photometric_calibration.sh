#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
config="${root}/source/D2SLAM/config/quadcam_drone_nxt_tmp/quadcam_depth.yaml"
local_masks="${root}/source/D2SLAM/config/quadcam_drone_nxt_tmp/camera_vig_mask_local"
container_masks="/root/swarm_ws/src/D2SLAM/config/quadcam_drone_nxt_tmp/camera_vig_mask_local"
active_file="${root}/runtime/calibration/photometric/active_run.txt"
action="${1:-}"

case "${action}" in
  enable|disable) ;;
  *)
    echo "Usage: $0 enable|disable" >&2
    exit 2
    ;;
esac

stamp="$(date +%Y%m%d_%H%M%S)"
backup="${root}/runtime/backups/quadcam_depth_before_photometric_${stamp}.yaml"
cp -- "${config}" "${backup}"

if [[ "${action}" == "enable" ]]; then
  test -f "${active_file}" || {
    echo "No active photometric run." >&2
    exit 1
  }
  run_dir="$(head -n 1 "${active_file}")"
  generated="${run_dir}/generated"
  validation="${generated}/photometric_validation.json"
  test -f "${validation}" || {
    echo "Generate masks first with ./scripts/94_generate_photometric_masks.sh" >&2
    exit 1
  }
  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${validation}")"
  if [[ "${status}" == "FAIL" ]]; then
    echo "Refusing failed photometric calibration: ${validation}" >&2
    exit 1
  fi
  if [[ "${status}" == "WARN" && "${ALLOW_PHOTOMETRIC_WARN:-false}" != "true" ]]; then
    echo "Calibration has warnings. Review ${validation}" >&2
    echo "After review, enable with ALLOW_PHOTOMETRIC_WARN=true $0 enable" >&2
    exit 1
  fi

  for index in 0 1 2 3; do
    test -f "${generated}/masks/cam_${index}_vig_mask.png" || {
      echo "Missing cam_${index}_vig_mask.png" >&2
      exit 1
    }
  done
  if [[ -d "${local_masks}" ]]; then
    mv -- "${local_masks}" "${local_masks}_backup_${stamp}"
  fi
  mkdir -p "${local_masks}"
  cp -- "${generated}/masks"/cam_[0-3]_vig_mask.png "${local_masks}/"
  cp -- "${validation}" "${local_masks}/photometric_validation.json"
  cp -- "${run_dir}/capture_settings.json" "${local_masks}/capture_settings.json"

  python3 - "${config}" true "${container_masks}" <<'PY'
import re
import sys

path, enabled, mask_path = sys.argv[1:]
text = open(path, encoding="utf-8").read()
text, count_enabled = re.subn(
    r"(?m)^enable_photometric_calib:\s*(?:true|false)\s*$",
    f"enable_photometric_calib: {enabled}",
    text,
)
text, count_path = re.subn(
    r'(?m)^photometric_calib_path:\s*".*"\s*$',
    f'photometric_calib_path: "{mask_path}"',
    text,
)
if count_enabled != 1 or count_path != 1:
    raise SystemExit(
        f"Expected exactly one config key; enabled={count_enabled}, path={count_path}"
    )
with open(path, "w", encoding="utf-8") as handle:
    handle.write(text)
PY
  echo "Enabled local photometric calibration."
else
  python3 - "${config}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
text, count = re.subn(
    r"(?m)^enable_photometric_calib:\s*(?:true|false)\s*$",
    "enable_photometric_calib: false",
    text,
)
if count != 1:
    raise SystemExit(f"Expected exactly one enable key, found {count}")
with open(path, "w", encoding="utf-8") as handle:
    handle.write(text)
PY
  echo "Disabled photometric calibration; local masks were retained."
fi

echo "Configuration backup: ${backup}"
echo "Restart the OAK normal-mode driver and OmniDepth before testing."
