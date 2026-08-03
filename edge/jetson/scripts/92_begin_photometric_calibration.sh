#!/usr/bin/env bash
set -Eeuo pipefail

root="/home/neu/OmniNxt"
runtime_root="${root}/runtime/calibration/photometric"
active_file="${runtime_root}/active_run.txt"
stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${runtime_root}/run_${stamp}"

exposure_us="${OAK_EXPOSURE_US:-5000}"
iso="${OAK_ISO:-100}"
awb_value="${OAK_AWB_VALUE:-3000}"

for value in "${exposure_us}" "${iso}" "${awb_value}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "Exposure, ISO, and white balance must be integers." >&2
    exit 2
  }
done

mkdir -p "${run_dir}/raw"
printf '%s\n' "${run_dir}" >"${active_file}"

if docker ps --format '{{.Names}}' | grep -Fxq omninxt_omnidepth; then
  if docker exec omninxt_omnidepth bash -lc \
    "pgrep -f '[q]uadcam_depth_est_node' >/dev/null"; then
    "${root}/scripts/88_stop_live_omnidepth.sh"
  fi
fi

"${root}/scripts/29_stop_oak4p_driver.sh"
OAK_AUTO_EXPOSE=false \
OAK_EXPOSURE_US="${exposure_us}" \
OAK_ISO="${iso}" \
OAK_AUTO_AWB=false \
OAK_AWB_VALUE="${awb_value}" \
OAK_UPSIDE_DOWN=false \
  "${root}/scripts/20_start_oak4p_driver.sh" sharpness

for _ in $(seq 1 60); do
  if docker exec omninxt_oak4p bash -lc \
    'source /opt/ros/noetic/setup.bash
     test "$(rostopic type /oak_ffc_4p/CAM_A 2>/dev/null)" = sensor_msgs/Image
     test "$(rostopic type /oak_ffc_4p/CAM_D 2>/dev/null)" = sensor_msgs/Image'
  then
    break
  fi
  sleep 1
done

docker exec omninxt_oak4p bash -lc \
  'source /opt/ros/noetic/setup.bash
   test "$(rostopic type /oak_ffc_4p/CAM_A)" = sensor_msgs/Image
   test "$(rostopic type /oak_ffc_4p/CAM_B)" = sensor_msgs/Image
   test "$(rostopic type /oak_ffc_4p/CAM_C)" = sensor_msgs/Image
   test "$(rostopic type /oak_ffc_4p/CAM_D)" = sensor_msgs/Image'

docker cp \
  "${root}/scripts/93_capture_photometric_stage.py" \
  omninxt_oak4p:/tmp/93_capture_photometric_stage.py

python3 - "${run_dir}/capture_settings.json" "${stamp}" \
  "${exposure_us}" "${iso}" "${awb_value}" <<'PY'
import json
import sys

path, stamp, exposure, iso, awb = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "run_stamp": stamp,
            "camera_order": ["CAM_A", "CAM_B", "CAM_C", "CAM_D"],
            "exposure_us": int(exposure),
            "iso": int(iso),
            "awb_value": int(awb),
            "auto_exposure": False,
            "auto_awb": False,
            "enable_upside_down": False,
            "resolution": [1280, 720],
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo
echo "Photometric calibration run created:"
echo "  ${run_dir}"
echo
echo "Locked camera settings:"
echo "  exposure=${exposure_us} us, ISO=${iso}, white_balance=${awb_value} K"
echo
echo "Next, present a uniform diffuse field to CAM_A and run:"
echo "  ./scripts/93_capture_photometric_stage.sh CAM_A flat"
echo
echo "Use a real lens cap/opaque cover for the dark stage:"
echo "  ./scripts/93_capture_photometric_stage.sh CAM_A dark"
