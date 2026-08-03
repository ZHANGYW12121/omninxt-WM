#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/noetic/setup.bash
source /dpmpc_ws/devel/setup.bash
set -u

roscore >/tmp/dpmpc_roscore.log 2>&1 &
ROSCORE_PID=$!
cleanup() {
    kill "${ROSCORE_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 50); do
    if rosparam list >/dev/null 2>&1; then
        exec "$@"
    fi
    sleep 0.1
done

echo "[DPMPC] roscore did not become ready; see /tmp/dpmpc_roscore.log" >&2
exit 1
