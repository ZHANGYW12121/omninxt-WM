#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZYW_ROOT="${ZYW_ROOT:-$HOME/zyw}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAAC_ROOT:-$ZYW_ROOT/isaacsim}}"
PX4_DIR="${PX4_DIR:-$ZYW_ROOT/PX4-Autopilot}"
LEGACY_DATABASE="${LEGACY_DATABASE:-$ISAACSIM_ROOT/database}"
D2SLAM_ROOT="${D2SLAM_ROOT:-$ZYW_ROOT/our_omni_depth/D2SLAM}"
OMNINXT_DATASET_ROOT="${OMNINXT_DATASET_ROOT:-$ZYW_ROOT/database_quadcamera}"
OMNIDEPTH_GPU="${OMNIDEPTH_GPU:-0}"

usage() {
  cat <<'EOF'
Usage: tools/configure_machine.sh [options]

  --zyw-root PATH          Workspace root (default: ~/zyw)
  --isaac-root PATH        Isaac Sim 5.1 installation containing python.sh
  --px4-root PATH          PX4-Autopilot checkout
  --legacy-database PATH   Existing Isaac database containing large assets
  --d2slam-root PATH       Patched D2SLAM checkout containing config and models
  --dataset-root PATH      Dataset/output location
  --omnidepth-gpu INDEX    Docker GPU index (server normally 1, laptop normally 0)

This writes only .local/machine.env, which is ignored by Git.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zyw-root) ZYW_ROOT="$2"; shift 2 ;;
    --isaac-root) ISAACSIM_ROOT="$2"; shift 2 ;;
    --px4-root) PX4_DIR="$2"; shift 2 ;;
    --legacy-database) LEGACY_DATABASE="$2"; shift 2 ;;
    --d2slam-root) D2SLAM_ROOT="$2"; shift 2 ;;
    --dataset-root) OMNINXT_DATASET_ROOT="$2"; shift 2 ;;
    --omnidepth-gpu) OMNIDEPTH_GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ZYW_ROOT="$(realpath -m "$ZYW_ROOT")"
ISAACSIM_ROOT="$(realpath -m "$ISAACSIM_ROOT")"
PX4_DIR="$(realpath -m "$PX4_DIR")"
LEGACY_DATABASE="$(realpath -m "$LEGACY_DATABASE")"
D2SLAM_ROOT="$(realpath -m "$D2SLAM_ROOT")"
OMNINXT_DATASET_ROOT="$(realpath -m "$OMNINXT_DATASET_ROOT")"

WAREHOUSE_USD="$LEGACY_DATABASE/warehouse/warehouse.usd"
OMNINXT_VISUAL_ASSET_DIR="$LEGACY_DATABASE/usd"
PEGASUS_PEOPLE_ASSET_ROOT="$LEGACY_DATABASE/assets/people/Characters"

required=(
  "$ISAACSIM_ROOT/python.sh"
  "$PX4_DIR/build/px4_sitl_default/bin/px4"
  "$WAREHOUSE_USD"
  "$OMNINXT_VISUAL_ASSET_DIR/Omininxt_body.usdc"
  "$OMNINXT_VISUAL_ASSET_DIR/fl.usdc"
  "$OMNINXT_VISUAL_ASSET_DIR/fr.usdc"
  "$OMNINXT_VISUAL_ASSET_DIR/rl.usdc"
  "$OMNINXT_VISUAL_ASSET_DIR/rr.usdc"
  "$PEGASUS_PEOPLE_ASSET_ROOT"
  "$D2SLAM_ROOT/config"
  "$D2SLAM_ROOT/models"
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

mkdir -p "$REPO_ROOT/.local"
tmp="$REPO_ROOT/.local/machine.env.tmp.$$"
umask 077
{
  printf 'export ZYW_ROOT=%q\n' "$ZYW_ROOT"
  printf 'export ISAACSIM_ROOT=%q\n' "$ISAACSIM_ROOT"
  printf 'export ISAAC_ROOT=%q\n' "$ISAACSIM_ROOT"
  printf 'export PX4_DIR=%q\n' "$PX4_DIR"
  printf 'export D2SLAM_ROOT=%q\n' "$D2SLAM_ROOT"
  printf 'export WAREHOUSE_USD=%q\n' "$WAREHOUSE_USD"
  printf 'export OMNINXT_VISUAL_ASSET_DIR=%q\n' "$OMNINXT_VISUAL_ASSET_DIR"
  printf 'export PEGASUS_PEOPLE_ASSET_ROOT=%q\n' "$PEGASUS_PEOPLE_ASSET_ROOT"
  printf 'export OMNINXT_DATASET_ROOT=%q\n' "$OMNINXT_DATASET_ROOT"
  printf 'export OMNINXT_DEPTH_EXPORT_ROOT=%q\n' "$REPO_ROOT/simulation/omnidepth/shared"
  printf 'export OMNIDEPTH_GPU=%q\n' "$OMNIDEPTH_GPU"
} > "$tmp"
mv "$tmp" "$REPO_ROOT/.local/machine.env"
chmod 600 "$REPO_ROOT/.local/machine.env"

echo "Configured: $REPO_ROOT/.local/machine.env"
"$REPO_ROOT/tools/doctor.sh"
