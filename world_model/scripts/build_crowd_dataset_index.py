#!/usr/bin/env python3
"""Build a read-only index for Isaac Sim crowd navigation records."""

from __future__ import annotations

import argparse
import sys
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.isaac_crowd import build_episode_index, save_episode_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Raw dataset root containing record_xxx folders. This directory is only read.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("crowd_dataset_index.json"),
        help="Where to save the derived index JSON.",
    )
    parser.add_argument("--min_frames", type=int, default=30, help="Drop episodes shorter than this.")
    parser.add_argument(
        "--exclude_termination",
        nargs="*",
        default=["stuck_timeout"],
        help="Episode termination reasons to exclude.",
    )
    parser.add_argument(
        "--allow_incomplete_triplets",
        action="store_true",
        help="Do not require json/camera/lidar triplets for every frame.",
    )
    parser.add_argument(
        "--allow_non_contiguous",
        action="store_true",
        help="Do not require contiguous frame indices.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Optional scan limit for quick tests.")
    return parser.parse_args()


def _stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    values = sorted(values)
    return (
        f"n={len(values)} min={values[0]:.4g} "
        f"p50={values[len(values)//2]:.4g} "
        f"mean={statistics.mean(values):.4g} "
        f"max={values[-1]:.4g}"
    )


def main() -> None:
    args = parse_args()
    episodes, rejected = build_episode_index(
        args.data_root,
        min_frames=args.min_frames,
        exclude_termination=args.exclude_termination,
        require_complete_triplets=not args.allow_incomplete_triplets,
        require_contiguous=not args.allow_non_contiguous,
        max_records=args.max_records,
        return_rejected=True,
    )
    save_episode_index(episodes, args.out, rejected=rejected)

    term_counter = Counter(ep.termination_reason for ep in episodes)
    reject_counter = Counter(item["reason"] for item in rejected)
    frame_counts = [ep.num_frames for ep in episodes]
    durations = [ep.duration_sec for ep in episodes if ep.duration_sec is not None]
    returns = [ep.episode_return for ep in episodes if ep.episode_return is not None]

    print(f"Raw dataset root: {args.data_root}")
    print(f"Saved index:      {args.out}")
    print(f"Accepted episodes: {len(episodes)}")
    print(f"Rejected episodes: {len(rejected)}")
    print(f"Accepted terminations: {dict(term_counter)}")
    print(f"Rejected reasons:      {dict(reject_counter)}")
    print(f"Frames:   {_stats([float(x) for x in frame_counts])}")
    print(f"Duration: {_stats([float(x) for x in durations])}")
    print(f"Return:   {_stats([float(x) for x in returns])}")


if __name__ == "__main__":
    main()
