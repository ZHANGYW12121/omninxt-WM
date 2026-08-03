#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_PATH="${CROWD_MAP_STATE_PATH:-/tmp/isaac_crowd_map_${USER:-user}_$$.json}"
SIM_PID=""
MAP_PID=""

cleanup() {
  if [[ -n "$MAP_PID" ]] && kill -0 "$MAP_PID" 2>/dev/null; then
    kill "$MAP_PID" 2>/dev/null || true
  fi
  if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill -TERM "$SIM_PID" 2>/dev/null || true
  fi
  rm -f "$STATE_PATH" "$STATE_PATH".tmp.*
}
trap cleanup EXIT INT TERM

rm -f "$STATE_PATH" "$STATE_PATH".tmp.*
export CROWD_MAP_STATE_PATH="$STATE_PATH"

"$SCRIPT_DIR/run_people_warehouse_omni.sh" "$@" &
SIM_PID=$!

echo "Isaac Sim PID: $SIM_PID"
echo "Waiting for the simulation UI and crowd state..."
for _ in $(seq 1 1800); do
  if [[ -s "$STATE_PATH" ]]; then
    break
  fi
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    wait "$SIM_PID"
    exit $?
  fi
  sleep 0.1
done

if [[ ! -s "$STATE_PATH" ]]; then
  echo "Timed out waiting for Isaac Sim crowd state: $STATE_PATH" >&2
  exit 1
fi

/usr/bin/python3 "$SCRIPT_DIR/crowd_map_monitor.py" \
  --state "$STATE_PATH" --sim-pid "$SIM_PID" &
MAP_PID=$!
echo "Crowd map window PID: $MAP_PID"

set +e
wait "$SIM_PID"
STATUS=$?
set -e
SIM_PID=""
exit "$STATUS"
