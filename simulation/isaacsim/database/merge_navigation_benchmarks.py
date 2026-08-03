#!/usr/bin/env python3
"""Merge completed navigation benchmark result directories losslessly."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from run_navigation_benchmark import write_aggregate


CONFIG_KEYS_THAT_MUST_MATCH = (
    "schema_version",
    "algorithm",
    "repeats",
    "evaluation_hz",
    "human_intrusion_threshold_m",
    "human_intrusion_exit_threshold_m",
    "algorithms",
    "isolation",
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    if len(args.sources) < 2:
        raise SystemExit("at least two source result directories are required")
    source_dirs = [path.resolve() for path in args.sources]
    configs = [
        _read_json(path / "experiment_config.json") for path in source_dirs
    ]
    reference = configs[0]
    for source, config in zip(source_dirs[1:], configs[1:]):
        for key in CONFIG_KEYS_THAT_MUST_MATCH:
            if config.get(key) != reference.get(key):
                raise SystemExit(
                    f"incompatible configuration key {key!r}: {source}"
                )

    scheduled = []
    expected_run_ids = []
    seeds = set()
    for config in configs:
        scheduled.extend(config.get("scheduled_trials", []))
        expected_run_ids.extend(config.get("expected_run_ids", []))
        seeds.update(int(seed) for seed in config.get("seeds", []))
    if len(expected_run_ids) != len(set(expected_run_ids)):
        raise SystemExit("source experiments contain duplicate expected run IDs")

    output = args.output.resolve()
    runs_output = output / "runs"
    output.mkdir(parents=True, exist_ok=True)
    runs_output.mkdir(parents=True, exist_ok=True)

    copied_run_ids = set()
    source_result_counts = {}
    for source in source_dirs:
        source_count = 0
        for result_path in sorted((source / "runs").glob("*.json")):
            result = _read_json(result_path)
            run_id = str(result.get("run_id"))
            if run_id in copied_run_ids:
                raise SystemExit(f"duplicate result run_id: {run_id}")
            copied_run_ids.add(run_id)
            source_count += 1
            shutil.copy2(result_path, runs_output / result_path.name)
        source_result_counts[str(source)] = source_count

    missing = sorted(set(expected_run_ids) - copied_run_ids)
    unexpected = sorted(copied_run_ids - set(expected_run_ids))
    if missing or unexpected:
        raise SystemExit(
            f"result coverage mismatch: missing={missing}, unexpected={unexpected}"
        )

    merged_config = dict(reference)
    merged_config.update(
        {
            "experiment_id": output.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": sorted(seeds),
            "scheduled_trials": sorted(
                scheduled,
                key=lambda item: (
                    int(item["seed"]),
                    int(item["repeat_index"]),
                    str(item["algorithm"]),
                ),
            ),
            "expected_run_ids": sorted(expected_run_ids),
            "merged_from": [str(path) for path in source_dirs],
        }
    )
    merged_config.pop("configuration_sha256", None)
    (output / "experiment_config.json").write_text(
        json.dumps(merged_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "merge_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "sources": source_result_counts,
                "result_count": len(copied_run_ids),
                "seed_min": min(seeds),
                "seed_max": max(seeds),
                "logs_remain_in_source_directories": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = write_aggregate(output, expected_run_ids=expected_run_ids)
    if summary["missing_run_ids"]:
        raise SystemExit(f"aggregate has missing results: {summary['missing_run_ids']}")
    if summary["paired_scene_mismatch_count"]:
        raise SystemExit(
            "paired scene fingerprints do not match: "
            f"{summary['paired_scene_mismatch_count']}"
        )
    print(json.dumps(summary["algorithms"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
