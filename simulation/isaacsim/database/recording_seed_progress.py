"""Persistent, simulator-independent seed-range progress for dataset recording."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "omninxt.recording_seed_progress.v1"
DATASET_SCHEMA = "omninxt.crowd_skeleton_state.v3"
SEED_COMPLETING_REASONS = frozenset({
    "reached_goal", "human_collision", "static_collision",
    "out_of_bounds", "crash", "stuck_timeout", "time_limit",
})


def validate_seed_range(seed_start: int, seed_end: int) -> tuple[int, int]:
    start, end = int(seed_start), int(seed_end)
    if start < 0 or end < 0:
        raise ValueError("recording seed bounds must be non-negative")
    if end < start:
        raise ValueError(
            f"recording seed end ({end}) must be >= start ({start})")
    return start, end


def normalize_run_id(run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id).strip()).strip("._")
    if not value:
        raise ValueError("recording run ID must contain a filename-safe character")
    return value[:96]


def seed_progress_path(dataset_root: str | Path, run_id: str,
                       seed_start: int, seed_end: int) -> Path:
    start, end = validate_seed_range(seed_start, seed_end)
    safe_id = normalize_run_id(run_id)
    return Path(dataset_root).expanduser() / "recording_progress" / (
        f"{safe_id}_seed_{start}_{end}.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _completed_from_episode_summaries(
    dataset_root: Path, run_id: str, seed_start: int, seed_end: int,
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    episodes_root = dataset_root / "episodes"
    if not episodes_root.is_dir():
        return completed
    for directory in sorted(path for path in episodes_root.iterdir() if path.is_dir()):
        metadata_path, summary_path = directory / "metadata.json", directory / "summary.json"
        if not metadata_path.is_file() or not summary_path.is_file():
            continue
        try:
            metadata, summary = _read_json(metadata_path), _read_json(summary_path)
            episode = metadata.get("episode", {})
            if metadata.get("schema") != DATASET_SCHEMA or not isinstance(episode, Mapping):
                continue
            if str(episode.get("recording_run_id", "")) != run_id:
                continue
            if int(episode.get("seed_start")) != seed_start \
                    or int(episode.get("seed_end")) != seed_end:
                continue
            seed = int(episode.get("crowd_seed"))
            reason = str(summary.get("termination_reason", ""))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        # A controlled/forced application shutdown means the current seed was
        # interrupted. Keep its partial trajectory for audit, but rerun it.
        if seed_start <= seed <= seed_end and reason in SEED_COMPLETING_REASONS:
            completed[seed] = {
                "seed": seed,
                "reason": reason,
                "episode_id": directory.name,
            }
    return completed


def resolve_seed_progress(
    dataset_root: str | Path,
    run_id: str,
    seed_start: int,
    seed_end: int,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Resolve the first unfinished seed using state plus durable summaries."""
    start, end = validate_seed_range(seed_start, seed_end)
    safe_id = normalize_run_id(run_id)
    root = Path(dataset_root).expanduser()
    path = seed_progress_path(root, safe_id, start, end)
    completed: dict[int, dict[str, Any]] = {}
    if resume and path.is_file():
        state = _read_json(path)
        if (state.get("schema") != SCHEMA
                or str(state.get("run_id")) != safe_id
                or int(state.get("seed_start", -1)) != start
                or int(state.get("seed_end", -1)) != end):
            raise ValueError(f"{path}: progress identity/range mismatch")
        for item in state.get("completed", ()):
            if isinstance(item, Mapping):
                try:
                    seed = int(item["seed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if start <= seed <= end:
                    completed[seed] = dict(item)
    if resume:
        completed.update(_completed_from_episode_summaries(
            root, safe_id, start, end))
    current = start
    while current <= end and current in completed:
        current += 1
    return {
        "schema": SCHEMA,
        "run_id": safe_id,
        "seed_start": start,
        "seed_end": end,
        "current_seed": current,
        "status": "complete" if current > end else "recording",
        "completed": [completed[seed] for seed in sorted(completed)],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def mark_seed_completed(state: Mapping[str, Any], seed: int,
                        reason: str) -> dict[str, Any]:
    start, end = validate_seed_range(state["seed_start"], state["seed_end"])
    seed = int(seed)
    if not start <= seed <= end:
        raise ValueError(f"completed seed {seed} lies outside [{start}, {end}]")
    completed = {
        int(item["seed"]): dict(item)
        for item in state.get("completed", ())
        if isinstance(item, Mapping) and "seed" in item
    }
    reason = str(reason)
    if reason in SEED_COMPLETING_REASONS:
        completed[seed] = {"seed": seed, "reason": reason}
    current = start
    while current <= end and current in completed:
        current += 1
    return {
        "schema": SCHEMA,
        "run_id": normalize_run_id(state["run_id"]),
        "seed_start": start,
        "seed_end": end,
        "current_seed": current,
        "status": "complete" if current > end else "recording",
        "completed": [completed[value] for value in sorted(completed)],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_seed_progress(path: str | Path, state: Mapping[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(state), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
