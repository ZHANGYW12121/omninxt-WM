#!/usr/bin/env python3
"""Compute ego-state normalization statistics for Isaac crowd records.

The script only reads raw ``record_xxx/frames/frame_*.json`` files or an
existing derived episode index.  It does not modify the raw dataset.

Example:

    python scripts/compute_ego_state_stats.py \
      --index_path crowd_dataset_index_lenovo.json \
      --out ego_state_stats_lenovo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.isaac_crowd import (
    DEFAULT_EGO_KEYS,
    EpisodeInfo,
    build_episode_index,
    ego_state_reference_from_frame,
    ego_state_to_vector,
    load_episode_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help=(
            "Raw dataset root. Required when --index_path is omitted. "
            "When --index_path is provided, this optionally overrides record_dir "
            "paths inside the index via data_root/record_name."
        ),
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=None,
        help="Optional episode index from build_crowd_dataset_index.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path to save the computed statistics.",
    )
    parser.add_argument("--min_frames", type=int, default=30, help="Used only when --index_path is omitted.")
    parser.add_argument(
        "--exclude_termination",
        nargs="*",
        default=["stuck_timeout"],
        help="Used only when --index_path is omitted.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Optional record limit for quick tests.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every Nth frame inside each episode. Keep 1 for final statistics.",
    )
    parser.add_argument(
        "--std_floor",
        type=float,
        default=1.0e-3,
        help="Minimum std written to ego_state_std to avoid division by tiny values.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal digits printed in the YAML snippet.",
    )
    return parser.parse_args()


def load_episodes(args: argparse.Namespace) -> list[EpisodeInfo]:
    if args.index_path is not None:
        episodes = load_episode_index(args.index_path)
        if args.data_root is not None:
            root = args.data_root.expanduser()
            episodes = [replace(ep, record_dir=str(root / ep.record_name)) for ep in episodes]
    else:
        if args.data_root is None:
            raise ValueError("Provide --data_root when --index_path is omitted.")
        episodes = build_episode_index(
            args.data_root,
            min_frames=args.min_frames,
            exclude_termination=args.exclude_termination,
        )
    if args.max_records is not None:
        episodes = episodes[: int(args.max_records)]
    if not episodes:
        raise ValueError("No episodes available for ego-state statistics.")
    return episodes


def load_frame_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class RunningStats:
    """Numerically stable vector mean/std/min/max with Welford updates."""

    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros((dim,), dtype=np.float64)
        self.m2 = np.zeros((dim,), dtype=np.float64)
        self.min = np.full((dim,), np.inf, dtype=np.float64)
        self.max = np.full((dim,), -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.mean.shape:
            raise ValueError(f"Expected vector shape {self.mean.shape}, got {x.shape}.")
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    @property
    def std(self) -> np.ndarray:
        if self.count <= 0:
            return np.zeros_like(self.mean)
        return np.sqrt(np.maximum(self.m2 / self.count, 0.0))


def iter_episode_frame_jsons(episode: EpisodeInfo, frame_stride: int) -> Iterable[Path]:
    frames_dir = Path(episode.record_dir) / "frames"
    for frame_id in range(int(episode.first_frame), int(episode.last_frame) + 1, int(frame_stride)):
        yield frames_dir / f"frame_{frame_id:06d}.json"


def fmt_list(values: np.ndarray, precision: int) -> str:
    fmt = f"{{:.{int(precision)}f}}"
    return "[" + ", ".join(fmt.format(float(x)) for x in values.tolist()) + "]"


def main() -> None:
    args = parse_args()
    frame_stride = max(1, int(args.frame_stride))
    std_floor = max(0.0, float(args.std_floor))
    episodes = load_episodes(args)

    stats = RunningStats(dim=len(DEFAULT_EGO_KEYS))
    missing = 0
    malformed = 0
    for ep in episodes:
        reference_path = Path(ep.record_dir) / "frames" / f"frame_{int(ep.first_frame):06d}.json"
        if not reference_path.is_file():
            missing += 1
            continue
        try:
            reference = ego_state_reference_from_frame(load_frame_json(reference_path))
        except Exception:  # noqa: BLE001
            malformed += 1
            continue
        for frame_json in iter_episode_frame_jsons(ep, frame_stride):
            if not frame_json.is_file():
                missing += 1
                continue
            try:
                frame = load_frame_json(frame_json)
                stats.update(ego_state_to_vector(frame, reference=reference))
            except Exception:  # noqa: BLE001 - keep scanning and report aggregate count
                malformed += 1

    if stats.count == 0:
        raise RuntimeError(
            "No valid frame JSON files were found. "
            f"missing_frames={missing}, malformed_frames={malformed}. "
            "If you used an index whose record_dir paths have moved, rerun with "
            "--data_root <current_dataset_root> --index_path <index>."
        )

    std = stats.std
    std_clamped = np.maximum(std, std_floor)
    payload = {
        "schema": "isaac_crowd_ego_state_stats_v2_local17",
        "ego_state_keys": list(DEFAULT_EGO_KEYS),
        "num_episodes": len(episodes),
        "num_frames": int(stats.count),
        "frame_stride": frame_stride,
        "missing_frames": int(missing),
        "malformed_frames": int(malformed),
        "std_floor": std_floor,
        "ego_state_mean": stats.mean.tolist(),
        "ego_state_std_raw": std.tolist(),
        "ego_state_std": std_clamped.tolist(),
        "ego_state_min": stats.min.tolist(),
        "ego_state_max": stats.max.tolist(),
    }

    print("Ego-state statistics")
    print(f"  Episodes:        {len(episodes)}")
    print(f"  Frames used:     {stats.count}")
    print(f"  Frame stride:    {frame_stride}")
    print(f"  Missing frames:  {missing}")
    print(f"  Malformed frames:{malformed}")
    print(f"  Keys:            {list(DEFAULT_EGO_KEYS)}")
    print()
    print(f"ego_state_mean:    {fmt_list(stats.mean, args.precision)}")
    print(f"ego_state_std_raw: {fmt_list(std, args.precision)}")
    print(f"ego_state_std:     {fmt_list(std_clamped, args.precision)}")
    print(f"ego_state_min:     {fmt_list(stats.min, args.precision)}")
    print(f"ego_state_max:     {fmt_list(stats.max, args.precision)}")
    print()
    print("YAML snippet for configs/model/_base_.yaml:")
    print("prediction_heads:")
    print(f"  ego_state_mean: {fmt_list(stats.mean, args.precision)}")
    print(f"  ego_state_std:  {fmt_list(std_clamped, args.precision)}")
    print()
    print("Copy these ego_state_mean/std values into configs/model/_base_.yaml and keep them frozen")
    print("for this task definition unless the 17D ego-state definition or task scale changes.")

    if args.out is not None:
        out = args.out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()
