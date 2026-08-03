#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -f "$REPO_ROOT/.local/machine.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.local/machine.env"
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-}}"
LOCAL_PEOPLE_ROOT="${PEGASUS_PEOPLE_ASSET_ROOT:-}"

if [[ ! -x "$ISAACSIM_ROOT/python.sh" ]]; then
    echo "ISAACSIM_ROOT must point to an Isaac Sim 5.1 installation." >&2
    exit 2
fi
if [[ -z "$LOCAL_PEOPLE_ROOT" || ! -f "$LOCAL_PEOPLE_ROOT/Biped_Setup.usd" ]]; then
    echo "Offline people assets are missing: $LOCAL_PEOPLE_ROOT" >&2
    echo "Run tools/configure_machine.sh with the correct legacy asset directory." >&2
    exit 2
fi

cd "$SCRIPT_DIR"
unset ROS_DISTRO RMW_IMPLEMENTATION
export PEGASUS_PEOPLE_ASSET_ROOT="$LOCAL_PEOPLE_ROOT"

# Flight-only NavRL baseline: no dataset recording and no OmniDepth export.
# Manual T starts takeoff. Select local Pegasus with NAVRL_USE_PX4=0 when PX4
# timing is not part of the experiment.
export CLASSIC_AUTO_START="${CLASSIC_AUTO_START:-0}"
export OMNINXT_DEPTH_EXPORT=0
export OMNINXT_GT_RANGE_EXPORT=0
export DATA_RECORD_ENABLED=0
# Simulation default. Set NAVRL_STATE_SOURCE=mavsdk to consume PX4 EKF
# position/velocity/attitude instead of Isaac rigid-body ground truth.
export NAVRL_STATE_SOURCE="${NAVRL_STATE_SOURCE:-isaac}"

if [[ "${NAVRL_USE_PX4:-1}" == "1" ]]; then
    control_mode="px4_navrl"
else
    control_mode="navrl"
fi

exec "$ISAACSIM_ROOT/python.sh" main.py \
    --control-mode "$control_mode" \
    --reset-user \
    --/renderer/multiGpu/enabled=false \
    --/renderer/activeGpu=0 \
    "$@"
