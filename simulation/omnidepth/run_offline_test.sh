#!/usr/bin/env bash
set -eo pipefail
FRAME_DIR="${1:-/root/omninxt_shared/input/frame_000000}"
FRAME_NAME="$(basename "$FRAME_DIR")"
source /opt/ros/noetic/setup.bash
source /root/swarm_ws/devel/setup.bash
mkdir -p /root/omninxt_shared/{output,logs}
roscore > /root/omninxt_shared/logs/roscore.log 2>&1 & ROSCORE_PID=$!
trap 'kill "$ROSCORE_PID" "$DEPTH_PID" "$SAVER_PID" 2>/dev/null || true' EXIT
sleep 2
roslaunch quadcam_depth_est depth-node.launch output:=screen > /root/omninxt_shared/logs/depth_node.log 2>&1 & DEPTH_PID=$!
sleep 5
python3 /root/omninxt_offline/pointcloud_saver.py --timeout 600 \
  --output "/root/omninxt_shared/output/${FRAME_NAME}.pcd" & SAVER_PID=$!
python3 /root/omninxt_offline/ros1_offline_feeder.py --input-dir "$FRAME_DIR" --repeat-count 100
wait "$SAVER_PID"
