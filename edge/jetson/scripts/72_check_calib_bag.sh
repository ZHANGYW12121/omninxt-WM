#!/usr/bin/env bash
set -Eeuo pipefail

bag_path="${1:-}"
runtime_root="/home/neu/OmniNxt/runtime"
container_name="omninxt_calib_bag_check"

if [[ -z "${bag_path}" || ! -f "${bag_path}" ]]; then
  echo "Usage: $0 /home/neu/OmniNxt/runtime/calibration/bags/FILE.bag" >&2
  exit 2
fi
case "$(realpath "${bag_path}")" in
  "${runtime_root}"/*) ;;
  *)
    echo "Bag must be under ${runtime_root} so it can be mounted read-only." >&2
    exit 2
    ;;
esac

relative_path="${bag_path#${runtime_root}/}"
docker rm "${container_name}" >/dev/null 2>&1 || true
docker run --rm \
  --name "${container_name}" \
  -v "${runtime_root}:/runtime:ro" \
  omninxt/oak4p-noetic-arm64:jp512 \
  bash -lc "source /opt/ros/noetic/setup.bash
            rosbag info --yaml '/runtime/${relative_path}'"

echo "Required quarterKalibr image topic: /oak_ffc_4p/assemble_image"
echo "Expected image contract: sensor_msgs/Image, bgr8, 5120x720"
echo "IMU calibration additionally requires one validated sensor_msgs/Imu topic."
