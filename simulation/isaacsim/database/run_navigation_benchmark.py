#!/usr/bin/env python3
"""Run isolated, reproducible Warehouse navigation benchmark trials."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ISAAC_ROOT = Path(
    os.environ.get("ISAACSIM_ROOT", os.environ.get("ISAAC_ROOT", "/nonexistent"))
).expanduser().resolve()
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "navigation_benchmark_results"
AUTORESUME_STATE_PATH = (
    SCRIPT_DIR / "navigation_benchmark_autoresume_state.json"
)
BENCHMARK_LAUNCH_LOCK_PATH = (
    SCRIPT_DIR / "navigation_benchmark_launcher.lock"
)
DEFAULT_EGO_IMAGE = "ego-planner-isaac:noetic"
DEFAULT_DPMPC_IMAGE = "dpmpc-planner-isaac:noetic"
BENCHMARK_ALGORITHMS = ("ego", "navrl", "navrl_no_shield", "dpmpc")
DEFAULT_HUMAN_INTRUSION_THRESHOLD_M = 0.50
DEFAULT_HUMAN_INTRUSION_EXIT_THRESHOLD_M = 0.60
DEFAULT_GPU_HEALTH_ATTEMPTS = 0
DEFAULT_GPU_HEALTH_TIMEOUT_SEC = 10.0
DEFAULT_GPU_HEALTH_RETRY_SEC = 5.0
DEFAULT_INTER_TRIAL_COOLDOWN_SEC = 10.0
DEFAULT_GPU_MONITOR_INTERVAL_SEC = 5.0
DEFAULT_GPU_MONITOR_FAILURES = 1
INFRASTRUCTURE_TERMINATION_REASONS = (
    "process_error",
    "wall_timeout",
    "gpu_failure",
)


class GPUHealthError(RuntimeError):
    """Raised before a trial when the selected GPU is not responsive."""


def _controller_algorithm(algorithm):
    """Map a benchmark variant to the underlying Pegasus controller."""
    return "navrl" if algorithm == "navrl_no_shield" else algorithm


def _safety_layers_enabled(algorithm):
    """Return the NavRL safety-layer state, or None for non-NavRL trials."""
    if algorithm in ("ego", "dpmpc"):
        return None
    return algorithm == "navrl"


def parse_seed_spec(text):
    values = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            items = range(start, end + step, step)
        else:
            items = (int(part),)
        for value in items:
            if value <= 0:
                raise argparse.ArgumentTypeError("seeds must be positive integers")
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run navigation variants over isolated seeded "
            "Warehouse trials and aggregate the metrics automatically."
        )
    )
    parser.add_argument(
        "--algorithm",
        choices=(
            "ego",
            "navrl",
            "navrl_no_shield",
            "dpmpc",
            "both",
            "navrl_pair",
            "all",
        ),
        required=True,
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_spec,
        default=parse_seed_spec("1-30"),
        help="seed list/range, e.g. 1-30 or 1,3,8-12 (default: 1-30)",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--eval-hz", type=float, default=10.0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--goal-radius-m", type=float, default=1.00)
    parser.add_argument(
        "--human-intrusion-threshold-m",
        type=float,
        default=DEFAULT_HUMAN_INTRUSION_THRESHOLD_M,
        help=(
            "surface-clearance threshold used for intrusion duration and "
            "event entry (default: 0.50 m)"
        ),
    )
    parser.add_argument(
        "--human-intrusion-exit-threshold-m",
        type=float,
        default=DEFAULT_HUMAN_INTRUSION_EXIT_THRESHOLD_M,
        help=(
            "surface-clearance threshold that ends an intrusion event; "
            "must exceed the entry threshold (default: 0.60 m)"
        ),
    )
    parser.add_argument(
        "--wall-timeout-sec",
        type=float,
        help=(
            "wall-clock watchdog per trial. Default is max(600, "
            "15*timeout-sec+300), allowing for slow Isaac RTF."
        ),
    )
    parser.add_argument(
        "--infrastructure-retries",
        type=int,
        default=1,
        help=(
            "automatic retries after process_error, wall_timeout, or "
            "gpu_failure (default: 1)"
        ),
    )
    parser.add_argument("--crowd-count", type=int)
    parser.add_argument(
        "--crowd-layout",
        choices=("sparse", "dense"),
        default="sparse",
        help=(
            "Warehouse crowd layout used by every trial in this experiment "
            "(default: sparse)."
        ),
    )
    parser.add_argument(
        "--dense-profile",
        choices=("transverse40", "legacy"),
        default="transverse40",
        help=(
            "dense map implementation (default: transverse40); ignored by "
            "sparse layout"
        ),
    )
    parser.add_argument(
        "--output-root",
        help="experiment directory; defaults to a timestamped directory.",
    )
    parser.add_argument(
        "--no-px4",
        action="store_true",
        help="use local Pegasus controllers instead of PX4/MAVSDK.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="show the Isaac Sim window (headless is the batch default).",
    )
    parser.add_argument("--ego-image", default=DEFAULT_EGO_IMAGE)
    parser.add_argument("--dpmpc-image", default=DEFAULT_DPMPC_IMAGE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip run JSON files already present in output-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned commands without starting Isaac or Docker.",
    )
    parser.add_argument(
        "--register-autoresume-only",
        action="store_true",
        help=(
            "register this experiment as the active reboot-resume job "
            "without starting any trial"
        ),
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index checked before every trial (default: 0)",
    )
    parser.add_argument(
        "--gpu-health-attempts",
        type=int,
        default=DEFAULT_GPU_HEALTH_ATTEMPTS,
        help=(
            "nvidia-smi attempts before aborting the batch; 0 waits "
            "indefinitely for recovery (default: 0)"
        ),
    )
    parser.add_argument(
        "--gpu-health-timeout-sec",
        type=float,
        default=DEFAULT_GPU_HEALTH_TIMEOUT_SEC,
        help="timeout for each nvidia-smi health query (default: 10 s)",
    )
    parser.add_argument(
        "--gpu-health-retry-sec",
        type=float,
        default=DEFAULT_GPU_HEALTH_RETRY_SEC,
        help="delay between failed GPU health queries (default: 5 s)",
    )
    parser.add_argument(
        "--inter-trial-cooldown-sec",
        type=float,
        default=DEFAULT_INTER_TRIAL_COOLDOWN_SEC,
        help=(
            "cooldown before checking the GPU and launching the next "
            "non-skipped trial (default: 10 s)"
        ),
    )
    parser.add_argument(
        "--gpu-monitor-interval-sec",
        type=float,
        default=DEFAULT_GPU_MONITOR_INTERVAL_SEC,
        help=(
            "nvidia-smi interval while Isaac is running; 0 disables "
            "runtime monitoring (default: 5 s)"
        ),
    )
    parser.add_argument(
        "--gpu-monitor-failures",
        type=int,
        default=DEFAULT_GPU_MONITOR_FAILURES,
        help=(
            "consecutive runtime GPU query failures that terminate and "
            "retry the trial (default: 1)"
        ),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not 0.0 < args.eval_hz <= 25.0:
        parser.error("--eval-hz must be in (0, 25]")
    if args.timeout_sec <= 0.0:
        parser.error("--timeout-sec must be positive")
    if args.goal_radius_m <= 0.0:
        parser.error("--goal-radius-m must be positive")
    if args.human_intrusion_threshold_m <= 0.0:
        parser.error("--human-intrusion-threshold-m must be positive")
    if (
        args.human_intrusion_exit_threshold_m
        <= args.human_intrusion_threshold_m
    ):
        parser.error(
            "--human-intrusion-exit-threshold-m must exceed "
            "--human-intrusion-threshold-m"
        )
    if args.wall_timeout_sec is not None and args.wall_timeout_sec <= 0.0:
        parser.error("--wall-timeout-sec must be positive")
    if args.infrastructure_retries < 0:
        parser.error("--infrastructure-retries cannot be negative")
    if args.crowd_count is not None:
        if (
            args.crowd_layout == "dense"
            and args.dense_profile == "transverse40"
            and args.crowd_count != 40
        ):
            parser.error(
                "transverse40 dense layout requires --crowd-count 40"
            )
        if (
            (
                args.crowd_layout == "sparse"
                or args.dense_profile == "legacy"
            )
            and not 17 <= args.crowd_count <= 23
        ):
            parser.error(
                "sparse/legacy layout requires --crowd-count between 17 and 23"
            )
    if args.dry_run and args.register_autoresume_only:
        parser.error(
            "--dry-run and --register-autoresume-only cannot be combined"
        )
    if args.gpu_index < 0:
        parser.error("--gpu-index cannot be negative")
    if args.gpu_health_attempts < 0:
        parser.error("--gpu-health-attempts cannot be negative")
    if args.gpu_health_timeout_sec <= 0.0:
        parser.error("--gpu-health-timeout-sec must be positive")
    if args.gpu_health_retry_sec < 0.0:
        parser.error("--gpu-health-retry-sec cannot be negative")
    if args.inter_trial_cooldown_sec < 0.0:
        parser.error("--inter-trial-cooldown-sec cannot be negative")
    if args.gpu_monitor_interval_sec < 0.0:
        parser.error("--gpu-monitor-interval-sec cannot be negative")
    if args.gpu_monitor_failures <= 0:
        parser.error("--gpu-monitor-failures must be positive")
    return args


def _autoresume_command(args, output_root):
    """Build a complete, stable resume command for the active experiment."""
    command = [
        str(Path(__file__).resolve()),
        "--algorithm",
        str(args.algorithm),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--repeats",
        str(args.repeats),
        "--eval-hz",
        str(args.eval_hz),
        "--timeout-sec",
        str(args.timeout_sec),
        "--goal-radius-m",
        str(args.goal_radius_m),
        "--human-intrusion-threshold-m",
        str(args.human_intrusion_threshold_m),
        "--human-intrusion-exit-threshold-m",
        str(args.human_intrusion_exit_threshold_m),
        "--crowd-layout",
        str(args.crowd_layout),
        "--dense-profile",
        str(args.dense_profile),
        "--infrastructure-retries",
        str(args.infrastructure_retries),
        "--output-root",
        str(output_root),
        "--ego-image",
        str(args.ego_image),
        "--dpmpc-image",
        str(args.dpmpc_image),
        "--gpu-index",
        str(args.gpu_index),
        "--gpu-health-attempts",
        str(args.gpu_health_attempts),
        "--gpu-health-timeout-sec",
        str(args.gpu_health_timeout_sec),
        "--gpu-health-retry-sec",
        str(args.gpu_health_retry_sec),
        "--inter-trial-cooldown-sec",
        str(args.inter_trial_cooldown_sec),
        "--gpu-monitor-interval-sec",
        str(args.gpu_monitor_interval_sec),
        "--gpu-monitor-failures",
        str(args.gpu_monitor_failures),
        "--resume",
    ]
    if args.wall_timeout_sec is not None:
        command.extend(
            ["--wall-timeout-sec", str(args.wall_timeout_sec)]
        )
    if args.crowd_count is not None:
        command.extend(["--crowd-count", str(args.crowd_count)])
    if args.no_px4:
        command.append("--no-px4")
    if args.no_headless:
        command.append("--no-headless")
    return command


def _pending_run_ids(output_root, scheduled_trials):
    """Return missing or retryable infrastructure-failure run IDs."""
    pending = []
    for trial in scheduled_trials:
        run_id = trial["run_id"]
        path = output_root / "runs" / f"{run_id}.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            identity_matches = result.get("run_id") == run_id
            retryable = (
                result.get("termination_reason")
                in INFRASTRUCTURE_TERMINATION_REASONS
            )
            if not identity_matches or retryable:
                pending.append(run_id)
        except Exception:
            pending.append(run_id)
    return pending


def _read_autoresume_state():
    try:
        return json.loads(
            AUTORESUME_STATE_PATH.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def _write_autoresume_state(state):
    """Atomically persist the active experiment across crashes and reboots."""
    AUTORESUME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUTORESUME_STATE_PATH.with_name(
        f".{AUTORESUME_STATE_PATH.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, AUTORESUME_STATE_PATH)


def _acquire_benchmark_launch_lock():
    """Prevent a reboot-resumed job and a manual job from overlapping."""
    handle = BENCHMARK_LAUNCH_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "acquired_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _register_autoresume_job(args, output_root, config, pending_run_ids):
    previous = _read_autoresume_state()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    configuration_sha256 = config.get("configuration_sha256")
    same_job = (
        isinstance(previous, dict)
        and previous.get("output_root") == str(output_root)
        and previous.get("configuration_sha256") == configuration_sha256
    )
    if (
        isinstance(previous, dict)
        and previous.get("status") == "active"
        and not same_job
    ):
        print(
            "[BENCH-RESUME] replacing previously active experiment: "
            f"{previous.get('output_root')}",
            file=sys.stderr,
        )
    state = {
        "schema_version": 1,
        "status": "active" if pending_run_ids else "completed",
        "created_at": (
            previous.get("created_at")
            if same_job and previous.get("created_at")
            else now
        ),
        "updated_at": now,
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "output_root": str(output_root),
        "experiment_config": str(
            output_root / "experiment_config.json"
        ),
        "configuration_sha256": configuration_sha256,
        "command": _autoresume_command(args, output_root),
        "pending_count": len(pending_run_ids),
        "next_pending_run_id": (
            pending_run_ids[0] if pending_run_ids else None
        ),
        "last_event": (
            "registered" if pending_run_ids else "already_completed"
        ),
    }
    _write_autoresume_state(state)
    print(
        f"[BENCH-RESUME] {state['status']} job registered: "
        f"pending={state['pending_count']}, "
        f"next={state['next_pending_run_id']}, "
        f"state={AUTORESUME_STATE_PATH}"
    )
    return state


def _update_autoresume_job(
    output_root,
    config,
    pending_run_ids,
    status=None,
    last_event=None,
):
    """Update only if the state still belongs to this experiment."""
    state = _read_autoresume_state()
    if not isinstance(state, dict):
        return
    if (
        state.get("output_root") != str(output_root)
        or state.get("configuration_sha256")
        != config.get("configuration_sha256")
    ):
        return
    state.update(
        {
            "status": (
                status
                if status is not None
                else "active" if pending_run_ids else "completed"
            ),
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "pid": os.getpid(),
            "pending_count": len(pending_run_ids),
            "next_pending_run_id": (
                pending_run_ids[0] if pending_run_ids else None
            ),
            "last_event": last_event,
        }
    )
    _write_autoresume_state(state)


def _mean(values):
    finite = [
        float(value) for value in values
        if value is not None
    ]
    return statistics.fmean(finite) if finite else None


def _percentile(values, percentile):
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * float(percentile) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(data) - 1)
    fraction = rank - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


def _load_results(runs_dir, expected_run_ids=None):
    expected = None if expected_run_ids is None else set(expected_run_ids)
    results = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            if expected is not None and result.get("run_id") not in expected:
                continue
            result["_result_path"] = str(path)
            results.append(result)
        except Exception as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
    return results


def _algorithm_summary(results):
    def intrusion_duration_sec(item):
        ratio = item.get("human_intrusion_ratio_valid_coverage")
        valid_duration = item.get("human_clearance_valid_duration_sec")
        if not isinstance(ratio, (int, float)) or not isinstance(
            valid_duration, (int, float)
        ):
            return None
        return float(ratio) * float(valid_duration)

    infrastructure = [
        item for item in results
        if item.get("termination_reason")
        in INFRASTRUCTURE_TERMINATION_REASONS
    ]
    valid = [item for item in results if item not in infrastructure]
    success = [item for item in results if item.get("success")]
    clearance_valid = [
        item for item in results
        if item.get("human_intrusion_ratio_valid_coverage") is not None
    ]
    clearance_valid_success = [
        item for item in success
        if item.get("human_intrusion_ratio_valid_coverage") is not None
    ]
    intrusion_event_valid = [
        item for item in results
        if item.get("human_intrusion_event_count") is not None
    ]
    intrusion_event_valid_success = [
        item for item in success
        if item.get("human_intrusion_event_count") is not None
    ]
    latency_valid_success = [
        item for item in success
        if item.get("decision_latency_p95_ms") is not None
    ]
    pooled_latencies = [
        latency
        for item in success
        for latency in item.get("decision_latency_samples_ms", [])
    ]
    pooled_policy_latencies = [
        float(record["latency_ms"])
        for item in success
        for record in item.get("decision_latency_records", [])
        if record.get("source") == "policy"
    ]
    pooled_direct_goal_latencies = [
        float(record["latency_ms"])
        for item in success
        for record in item.get("decision_latency_records", [])
        if record.get("source") == "direct_goal"
    ]
    count = len(results)
    valid_count = len(valid)
    return {
        "total_trials": count,
        "valid_experimental_trials": valid_count,
        "infrastructure_failure_trials": len(infrastructure),
        "successful_trials": len(success),
        "success_rate": len(success) / valid_count if valid_count else None,
        "collision_failures": sum(
            item.get("termination_reason") == "collision" for item in results
        ),
        "human_collision_failures": sum(
            bool(item.get("collision_human")) for item in results
        ),
        "static_collision_failures": sum(
            bool(item.get("collision_static")) for item in results
        ),
        "timeout_failures": sum(
            item.get("termination_reason") == "timeout" for item in results
        ),
        "other_failures": sum(
            not item.get("success")
            and item.get("termination_reason") not in (
                "collision",
                "timeout",
                *INFRASTRUCTURE_TERMINATION_REASONS,
            )
            for item in results
        ),
        "mean_human_intrusion_ratio_all": _mean(
            item.get("human_intrusion_ratio") for item in results
        ),
        "mean_human_intrusion_ratio_success": _mean(
            item.get("human_intrusion_ratio") for item in success
        ),
        "mean_human_intrusion_ratio_valid_coverage_all": _mean(
            item.get("human_intrusion_ratio_valid_coverage")
            for item in clearance_valid
        ),
        "mean_human_intrusion_ratio_valid_coverage_success": _mean(
            item.get("human_intrusion_ratio_valid_coverage")
            for item in clearance_valid_success
        ),
        "mean_human_intrusion_duration_sec_all": _mean(
            intrusion_duration_sec(item) for item in clearance_valid
        ),
        "mean_human_intrusion_duration_sec_success": _mean(
            intrusion_duration_sec(item) for item in clearance_valid_success
        ),
        "mean_human_intrusion_event_count_all": _mean(
            item.get("human_intrusion_event_count")
            for item in intrusion_event_valid
        ),
        "mean_human_intrusion_event_count_success": _mean(
            item.get("human_intrusion_event_count")
            for item in intrusion_event_valid_success
        ),
        "human_intrusion_event_trial_count_all": len(intrusion_event_valid),
        "human_intrusion_any_trial_count_all": sum(
            int(item.get("human_intrusion_event_count", 0)) > 0
            for item in intrusion_event_valid
        ),
        "human_intrusion_any_trial_count_success": sum(
            int(item.get("human_intrusion_event_count", 0)) > 0
            for item in intrusion_event_valid_success
        ),
        "human_clearance_valid_trial_count_all": len(clearance_valid),
        "mean_human_clearance_valid_ratio_all": _mean(
            item.get("human_clearance_valid_ratio") for item in results
        ),
        "mean_task_time_sim_sec_success": _mean(
            item.get("task_time_sim_sec") for item in success
        ),
        "mean_average_speed_3d_mps_success": _mean(
            item.get("average_speed_3d_mps") for item in success
        ),
        "mean_path_length_3d_m_success": _mean(
            item.get("path_length_3d_m") for item in success
        ),
        "mean_efficiency_success": _mean(
            item.get("efficiency") for item in success
        ),
        "mean_episode_p95_latency_ms_success": _mean(
            item.get("decision_latency_p95_ms") for item in success
        ),
        "pooled_p95_latency_ms_success": _percentile(
            pooled_latencies, 95
        ),
        "successful_latency_sample_count": len(pooled_latencies),
        "successful_episode_latency_count": len(latency_valid_success),
        "successful_episode_missing_latency_count": (
            len(success) - len(latency_valid_success)
        ),
        "pooled_policy_p95_latency_ms_success": _percentile(
            pooled_policy_latencies, 95
        ),
        "policy_latency_sample_count_success": len(pooled_policy_latencies),
        "pooled_direct_goal_p95_latency_ms_success": _percentile(
            pooled_direct_goal_latencies, 95
        ),
        "direct_goal_latency_sample_count_success": len(
            pooled_direct_goal_latencies
        ),
    }


def write_aggregate(output_root, expected_run_ids=None):
    experiment_config = None
    config_path = output_root / "experiment_config.json"
    if config_path.is_file():
        try:
            experiment_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except Exception:
            experiment_config = None
    if expected_run_ids is None:
        if experiment_config is not None:
            expected_run_ids = experiment_config.get("expected_run_ids")
    runs_dir = output_root / "runs"
    results = _load_results(runs_dir, expected_run_ids)
    grouped = {}
    for result in results:
        grouped.setdefault(str(result.get("algorithm", "unknown")), []).append(result)
    paired_scene_checks = []
    paired = {}
    expected_algorithms = {}
    if experiment_config is not None:
        for trial in experiment_config.get("scheduled_trials", []):
            key = (trial.get("seed"), trial.get("repeat_index"))
            expected_algorithms.setdefault(key, set()).add(
                str(trial.get("algorithm"))
            )
    for result in results:
        key = (result.get("seed"), result.get("repeat_index"))
        paired.setdefault(key, {})[str(result.get("algorithm"))] = result
    for (seed, repeat), pair in sorted(paired.items()):
        expected = expected_algorithms.get((seed, repeat), set(pair))
        if len(expected) < 2 or not expected.issubset(pair):
            continue
        fingerprints = {
            algorithm: pair[algorithm].get("scene_fingerprint_sha256")
            for algorithm in sorted(expected)
        }
        values = list(fingerprints.values())
        paired_scene_checks.append(
            {
                "seed": seed,
                "repeat_index": repeat,
                "algorithms": sorted(expected),
                "match": (
                    all(value is not None for value in values)
                    and len(set(values)) == 1
                ),
                "fingerprints": fingerprints,
            }
        )
    summary = {
        "schema_version": 3,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "human_intrusion_threshold_m": (
            experiment_config.get("human_intrusion_threshold_m")
            if experiment_config is not None
            else None
        ),
        "human_intrusion_exit_threshold_m": (
            experiment_config.get("human_intrusion_exit_threshold_m")
            if experiment_config is not None
            else None
        ),
        "result_count": len(results),
        "expected_result_count": (
            len(expected_run_ids) if expected_run_ids is not None else None
        ),
        "missing_run_ids": (
            sorted(
                set(expected_run_ids)
                - {str(item.get("run_id")) for item in results}
            )
            if expected_run_ids is not None
            else []
        ),
        "paired_scene_checks": paired_scene_checks,
        "paired_scene_mismatch_count": sum(
            not item["match"] for item in paired_scene_checks
        ),
        "algorithms": {
            algorithm: _algorithm_summary(items)
            for algorithm, items in sorted(grouped.items())
        },
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    columns = (
        "run_id",
        "algorithm",
        "safety_layers_enabled",
        "control_mode",
        "seed",
        "repeat_index",
        "crowd_count",
        "crowd_layout",
        "dense_profile",
        "success",
        "termination_reason",
        "collision_human",
        "collision_static",
        "task_time_sim_sec",
        "average_speed_3d_mps",
        "path_length_3d_m",
        "efficiency",
        "human_intrusion_ratio",
        "human_intrusion_ratio_valid_coverage",
        "human_intrusion_threshold_m",
        "human_intrusion_exit_threshold_m",
        "human_intrusion_event_count",
        "human_intrusion_event_definition",
        "human_clearance_valid_ratio",
        "min_human_surface_clearance_m",
        "reference_length_3d_m",
        "decision_latency_p95_ms",
        "decision_latency_policy_p95_ms",
        "decision_latency_direct_goal_p95_ms",
        "decision_latency_causal_pairing",
        "final_goal_distance_3d_m",
        "scene_fingerprint_sha256",
        "_result_path",
    )
    with (output_root / "runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    lines = [
        "# Warehouse navigation benchmark",
        "",
        f"Recorded trials: {len(results)}",
        "",
        (
            "Human intrusion: surface clearance < "
            f"{summary['human_intrusion_threshold_m']} m; an event resets at "
            f">= {summary['human_intrusion_exit_threshold_m']} m."
        ),
        "",
        "## Task outcome and safety",
        "",
        "| Algorithm | Success (valid) | Infrastructure failures | "
        "Human collisions | Static collisions | Timeouts | Other failures | "
        "Intrusion ratio all | Intrusion ratio success | "
        "Intrusion seconds/task all | Intrusion seconds/task success | "
        "Events/task all | Events/task success | "
        "Trials with intrusion all | Trials with intrusion success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|",
    ]
    for algorithm, values in summary["algorithms"].items():
        def fmt(key, digits=4):
            value = values.get(key)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"| {algorithm} | {values['successful_trials']}/"
            f"{values['valid_experimental_trials']} "
            f"({fmt('success_rate', 3)}) | "
            f"{values['infrastructure_failure_trials']} | "
            f"{values['human_collision_failures']} | "
            f"{values['static_collision_failures']} | "
            f"{values['timeout_failures']} | "
            f"{values['other_failures']} | "
            f"{fmt('mean_human_intrusion_ratio_all', 5)} | "
            f"{fmt('mean_human_intrusion_ratio_success', 5)} | "
            f"{fmt('mean_human_intrusion_duration_sec_all', 3)} | "
            f"{fmt('mean_human_intrusion_duration_sec_success', 3)} | "
            f"{fmt('mean_human_intrusion_event_count_all', 3)} | "
            f"{fmt('mean_human_intrusion_event_count_success', 3)} | "
            f"{values['human_intrusion_any_trial_count_all']}/"
            f"{values['human_intrusion_event_trial_count_all']} | "
            f"{values['human_intrusion_any_trial_count_success']}/"
            f"{values['successful_trials']} |"
        )

    lines.extend(
        [
            "",
            "## Successful-task navigation performance",
            "",
            "| Algorithm | Successful trials | Time (sim s) | "
            "3D speed (m/s) | 3D path length (m) | Efficiency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for algorithm, values in summary["algorithms"].items():
        def fmt(key, digits=4):
            value = values.get(key)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"| {algorithm} | {values['successful_trials']} | "
            f"{fmt('mean_task_time_sim_sec_success', 3)} | "
            f"{fmt('mean_average_speed_3d_mps_success', 3)} | "
            f"{fmt('mean_path_length_3d_m_success', 3)} | "
            f"{fmt('mean_efficiency_success', 4)} |"
        )

    lines.extend(
        [
            "",
            "## Successful-task decision latency",
            "",
            "| Algorithm | Mean episode P95 (ms) | Pooled P95 (ms) | "
            "Policy pooled P95 (ms) | Direct-goal pooled P95 (ms) | "
            "Latency samples | Episodes with/missing latency |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for algorithm, values in summary["algorithms"].items():
        def fmt(key, digits=4):
            value = values.get(key)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"| {algorithm} | "
            f"{fmt('mean_episode_p95_latency_ms_success', 3)} | "
            f"{fmt('pooled_p95_latency_ms_success', 3)} | "
            f"{fmt('pooled_policy_p95_latency_ms_success', 3)} | "
            f"{fmt('pooled_direct_goal_p95_latency_ms_success', 3)} | "
            f"{values['successful_latency_sample_count']} | "
            f"{values['successful_episode_latency_count']}/"
            f"{values['successful_episode_missing_latency_count']} |"
        )

    lines.extend(
        [
            "",
            "## Measurement coverage",
            "",
            "| Algorithm | Human-clearance valid trials | "
            "Mean valid-clearance coverage | Event-count valid trials |",
            "|---|---:|---:|---:|",
        ]
    )
    for algorithm, values in summary["algorithms"].items():
        def fmt(key, digits=4):
            value = values.get(key)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"| {algorithm} | "
            f"{values['human_clearance_valid_trial_count_all']} | "
            f"{fmt('mean_human_clearance_valid_ratio_all', 4)} | "
            f"{values['human_intrusion_event_trial_count_all']} |"
        )
    (output_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def _write_process_failure(
    path,
    algorithm,
    control_mode,
    seed,
    repeat,
    run_id,
    exit_code,
    termination_reason="process_error",
    details=None,
    human_intrusion_threshold_m=DEFAULT_HUMAN_INTRUSION_THRESHOLD_M,
    human_intrusion_exit_threshold_m=(
        DEFAULT_HUMAN_INTRUSION_EXIT_THRESHOLD_M
    ),
):
    result = {
        "schema_version": 3,
        "run_id": run_id,
        "algorithm": algorithm,
        "safety_layers_enabled": _safety_layers_enabled(algorithm),
        "control_mode": control_mode,
        "seed": seed,
        "repeat_index": repeat,
        "success": False,
        "termination_reason": str(termination_reason),
        "process_exit_code": exit_code,
        "collision": False,
        "collision_human": False,
        "collision_static": False,
        "human_intrusion_threshold_m": human_intrusion_threshold_m,
        "human_intrusion_exit_threshold_m": (
            human_intrusion_exit_threshold_m
        ),
        "human_intrusion_ratio": None,
        "human_intrusion_event_count": None,
        "decision_latency_samples_ms": [],
        "details": details or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class TrialRunner:
    def __init__(self, args, output_root):
        self.args = args
        self.output_root = output_root
        self.current_process = None
        self.sidecar_process = None
        self.sidecar_name = None
        self._launched_trial_count = 0

    def stop(self):
        if self.current_process is not None and self.current_process.poll() is None:
            self._terminate_process_tree(self.current_process)
        self._stop_sidecar()

    def run(self, algorithm, seed, repeat):
        run_id = f"{algorithm}_seed{seed:03d}_repeat{repeat:02d}"
        result_path = self.output_root / "runs" / f"{run_id}.json"
        log_path = self.output_root / "logs" / f"{run_id}.log"
        controller_algorithm = _controller_algorithm(algorithm)
        control_mode = (
            controller_algorithm
            if self.args.no_px4
            else f"px4_{controller_algorithm}"
        )
        if self.args.resume and self._result_is_resumable(
            result_path, algorithm, control_mode, seed, repeat, run_id
        ):
            print(f"[BENCH-LAUNCH] skip valid existing {run_id}")
            return self._result_termination_reason(result_path)

        if not self.args.dry_run:
            if (
                self._launched_trial_count > 0
                and self.args.inter_trial_cooldown_sec > 0.0
            ):
                print(
                    f"[BENCH-GPU] cooling down for "
                    f"{self.args.inter_trial_cooldown_sec:.1f}s before "
                    f"{run_id}."
                )
                time.sleep(self.args.inter_trial_cooldown_sec)
            self._require_healthy_gpu(run_id)

        if result_path.exists():
            self._archive_stale_result(result_path)

        command = [
            str(ISAAC_ROOT / "python.sh"),
            str(SCRIPT_DIR / "main.py"),
            "--control-mode",
            control_mode,
            "--benchmark-algorithm",
            algorithm,
            "--crowd-mode",
            "server_v2",
            "--crowd-layout",
            self.args.crowd_layout,
            "--dense-profile",
            self.args.dense_profile,
            "--crowd-seed",
            str(seed),
            "--benchmark-run",
            "--benchmark-seed",
            str(seed),
            "--benchmark-repeat-index",
            str(repeat),
            "--benchmark-run-id",
            run_id,
            "--benchmark-output",
            str(result_path),
            "--benchmark-eval-hz",
            str(self.args.eval_hz),
            "--benchmark-timeout-sec",
            str(self.args.timeout_sec),
            "--benchmark-goal-radius-m",
            str(self.args.goal_radius_m),
            "--benchmark-human-intrusion-threshold-m",
            str(self.args.human_intrusion_threshold_m),
            "--benchmark-human-intrusion-exit-threshold-m",
            str(self.args.human_intrusion_exit_threshold_m),
            "--reset-user",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/activeGpu=0",
        ]
        if not self.args.no_headless:
            command.append("--headless")
        if self.args.crowd_count is not None:
            command.extend(["--crowd-count", str(self.args.crowd_count)])

        environment = os.environ.copy()
        environment.update(
            {
                "WAREHOUSE_CROWD_MODE": "server_v2",
                "WAREHOUSE_CROWD_LAYOUT": self.args.crowd_layout,
                "WAREHOUSE_DENSE_PROFILE": self.args.dense_profile,
                "WAREHOUSE_CROWD_SEED": str(seed),
                "CLASSIC_AUTO_START": "1",
                "OMNINXT_CAMERA_ENABLED": "0",
                # Keep the latest OmniNxt visual overlay in GUI benchmark runs.
                # Camera/depth rendering remains disabled independently.
                "OMNINXT_VISUAL_ENABLED": "1",
                "OMNINXT_DEPTH_EXPORT": "0",
                "OMNINXT_GT_RANGE_EXPORT": "0",
                "NAVRL_DIAGNOSTICS_ENABLED": "0",
                "NAVRL_SAFETY_SHIELD_ENABLED": (
                    "0" if algorithm == "navrl_no_shield" else "1"
                ),
            }
        )
        if self.args.crowd_count is not None:
            environment["WAREHOUSE_CROWD_COUNT"] = str(self.args.crowd_count)
        environment.pop("ROS_DISTRO", None)
        environment.pop("RMW_IMPLEMENTATION", None)
        people_root = SCRIPT_DIR / "assets" / "people" / "Characters"
        environment["PEGASUS_PEOPLE_ASSET_ROOT"] = str(people_root)

        print(
            f"\n[BENCH-LAUNCH] {run_id}: control={control_mode}, "
            f"safety_layers="
            f"{'off' if algorithm == 'navrl_no_shield' else 'on' if algorithm == 'navrl' else 'n/a'}, "
            f"result={result_path}"
        )
        print("[BENCH-LAUNCH] command:", " ".join(command))
        if self.args.dry_run:
            return None

        self._launched_trial_count += 1

        if algorithm in ("ego", "dpmpc"):
            try:
                self._start_sidecar(run_id, algorithm)
            except Exception as exc:
                print(
                    f"[BENCH-LAUNCH] {algorithm.upper()} sidecar failed "
                    f"for {run_id}: {exc}",
                    file=sys.stderr,
                )
                _write_process_failure(
                    result_path,
                    algorithm,
                    control_mode,
                    seed,
                    repeat,
                    run_id,
                    exit_code=None,
                    termination_reason="process_error",
                    details={
                        "stage": f"{algorithm}_sidecar_start",
                        "error": repr(exc),
                    },
                    human_intrusion_threshold_m=(
                        self.args.human_intrusion_threshold_m
                    ),
                    human_intrusion_exit_threshold_m=(
                        self.args.human_intrusion_exit_threshold_m
                    ),
                )
                self._stop_sidecar()
                write_aggregate(self.output_root)
                return "process_error"

        log_path.parent.mkdir(parents=True, exist_ok=True)
        exit_code = -1
        watchdog_fired = False
        gpu_monitor_failure = None
        launch_error = None
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                self.current_process = subprocess.Popen(
                    command,
                    cwd=str(SCRIPT_DIR),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                assert self.current_process.stdout is not None
                output_thread = threading.Thread(
                    target=self._stream_output,
                    args=(self.current_process.stdout, log),
                    daemon=True,
                )
                output_thread.start()
                wall_timeout = (
                    float(self.args.wall_timeout_sec)
                    if self.args.wall_timeout_sec is not None
                    else max(600.0, 15.0 * float(self.args.timeout_sec) + 300.0)
                )
                wall_deadline = time.monotonic() + wall_timeout
                gpu_monitor_interval = float(
                    self.args.gpu_monitor_interval_sec
                )
                next_gpu_check = (
                    time.monotonic() + gpu_monitor_interval
                    if gpu_monitor_interval > 0.0
                    else None
                )
                gpu_monitor_attempt = 0
                consecutive_gpu_failures = 0
                while self.current_process.poll() is None:
                    now = time.monotonic()
                    if now >= wall_deadline:
                        watchdog_fired = True
                        print(
                            f"\n[BENCH-LAUNCH] wall watchdog expired for {run_id} "
                            f"after {wall_timeout:.1f}s; terminating the trial.",
                            file=sys.stderr,
                        )
                        self._terminate_process_tree(self.current_process)
                        break
                    if (
                        next_gpu_check is not None
                        and now >= next_gpu_check
                    ):
                        gpu_monitor_attempt += 1
                        healthy, record, error = self._query_gpu_health_once(
                            run_id,
                            phase="runtime",
                            attempt=gpu_monitor_attempt,
                            max_attempts=None,
                        )
                        next_gpu_check = (
                            time.monotonic() + gpu_monitor_interval
                        )
                        if healthy:
                            if consecutive_gpu_failures:
                                print(
                                    f"[BENCH-GPU] runtime GPU query "
                                    f"recovered during {run_id}."
                                )
                            consecutive_gpu_failures = 0
                        else:
                            consecutive_gpu_failures += 1
                            print(
                                f"[BENCH-GPU] runtime GPU query failed "
                                f"during {run_id} "
                                f"({consecutive_gpu_failures}/"
                                f"{self.args.gpu_monitor_failures}): "
                                f"{error}",
                                file=sys.stderr,
                            )
                            if (
                                consecutive_gpu_failures
                                >= self.args.gpu_monitor_failures
                            ):
                                gpu_monitor_failure = {
                                    "stage": "runtime_gpu_monitor",
                                    "failure_reason": error,
                                    "health_record": record,
                                }
                                print(
                                    f"[BENCH-GPU] terminating {run_id}; "
                                    "the GPU must recover before this "
                                    "trial is retried.",
                                    file=sys.stderr,
                                )
                                self._terminate_process_tree(
                                    self.current_process
                                )
                                break
                    time.sleep(0.25)
                try:
                    exit_code = self.current_process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    exit_code = -9
                    if launch_error is None:
                        launch_error = (
                            "Isaac process did not exit within 15 seconds"
                        )
                output_thread.join(timeout=5.0)
        except Exception as exc:
            launch_error = repr(exc)
            print(
                f"[BENCH-LAUNCH] Isaac trial failed for {run_id}: {exc}",
                file=sys.stderr,
            )
            if self.current_process is not None and self.current_process.poll() is None:
                self._terminate_process_tree(self.current_process)
        finally:
            self.current_process = None
            self._stop_sidecar()

        if not result_path.is_file():
            termination_reason = (
                "gpu_failure"
                if gpu_monitor_failure is not None
                else "wall_timeout"
                if watchdog_fired
                else "process_error"
            )
            _write_process_failure(
                result_path,
                algorithm,
                control_mode,
                seed,
                repeat,
                run_id,
                exit_code,
                termination_reason=termination_reason,
                details={
                    "wall_watchdog_fired": watchdog_fired,
                    "gpu_monitor_failure": gpu_monitor_failure,
                    "launch_error": launch_error,
                },
                human_intrusion_threshold_m=(
                    self.args.human_intrusion_threshold_m
                ),
                human_intrusion_exit_threshold_m=(
                    self.args.human_intrusion_exit_threshold_m
                ),
            )
        write_aggregate(self.output_root)
        return self._result_termination_reason(result_path)

    def _require_healthy_gpu(self, run_id):
        """Wait for, or optionally require, a healthy selected GPU."""
        max_attempts = int(self.args.gpu_health_attempts)
        last_error = "unknown GPU health error"
        attempt = 0
        while True:
            attempt += 1
            healthy, record, last_error = self._query_gpu_health_once(
                run_id,
                phase="pre_trial",
                attempt=attempt,
                max_attempts=max_attempts or None,
            )
            if healthy:
                fields = record["fields"]
                print(
                    f"[BENCH-GPU] healthy before {run_id}: "
                    f"gpu={fields[0]} {fields[2]}, "
                    f"driver={fields[3]}, temp={fields[4]}C, "
                    f"memory={fields[5]}/{fields[6]} MiB, "
                    f"util={fields[7]}%."
                )
                return record
            attempt_label = (
                f"{attempt}/{max_attempts}"
                if max_attempts > 0
                else f"{attempt}/unlimited"
            )
            print(
                f"[BENCH-GPU] unhealthy before {run_id} "
                f"(attempt {attempt_label}): {last_error}",
                file=sys.stderr,
            )
            if max_attempts > 0 and attempt >= max_attempts:
                health_log = self.output_root / "gpu_health.jsonl"
                raise GPUHealthError(
                    f"GPU {self.args.gpu_index} failed all {max_attempts} "
                    f"health checks before {run_id}; no Isaac process was "
                    f"launched. Last error: {last_error}. "
                    f"Health log: {health_log}"
                )
            wait_sec = float(self.args.gpu_health_retry_sec)
            print(
                f"[BENCH-GPU] waiting {wait_sec:.1f}s for GPU recovery; "
                "Isaac will not be launched. Press Ctrl+C to stop safely.",
                file=sys.stderr,
            )
            if wait_sec > 0.0:
                time.sleep(wait_sec)

    def _query_gpu_health_once(
        self, run_id, phase, attempt, max_attempts=None
    ):
        """Run and record exactly one bounded nvidia-smi health query."""
        command = [
            "nvidia-smi",
            f"--id={self.args.gpu_index}",
            "--query-gpu="
            "index,uuid,name,driver_version,temperature.gpu,"
            "memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        record = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "run_id": run_id,
            "phase": str(phase),
            "gpu_index": int(self.args.gpu_index),
            "attempt": int(attempt),
            "max_attempts": max_attempts,
            "command": command,
            "healthy": False,
        }
        error = "unknown GPU health error"
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(self.args.gpu_health_timeout_sec),
                check=False,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            fields = [field.strip() for field in stdout.split(",")]
            numeric_values = None
            if len(fields) == 8:
                try:
                    numeric_values = {
                        "temperature_c": float(fields[4]),
                        "memory_used_mib": float(fields[5]),
                        "memory_total_mib": float(fields[6]),
                        "utilization_percent": float(fields[7]),
                    }
                except (TypeError, ValueError):
                    numeric_values = None
            lowered_output = f"{stdout}\n{stderr}".lower()
            fatal_tokens = (
                "gpu requires reset",
                "fallen off the bus",
                "unknown error",
                "error while waiting for gpu progress",
            )
            healthy = (
                completed.returncode == 0
                and len(fields) == 8
                and fields[0] == str(self.args.gpu_index)
                and numeric_values is not None
                and all(
                    value and value.lower() not in ("n/a", "[n/a]")
                    for value in fields
                )
                and not any(token in lowered_output for token in fatal_tokens)
                and 0.0 <= numeric_values["temperature_c"] <= 125.0
                and numeric_values["memory_used_mib"] >= 0.0
                and numeric_values["memory_total_mib"] > 0.0
                and 0.0 <= numeric_values["utilization_percent"] <= 100.0
            )
            record.update(
                {
                    "return_code": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "fields": fields,
                    "numeric_values": numeric_values,
                    "healthy": healthy,
                }
            )
            if not healthy:
                error = (
                    f"nvidia-smi returned {completed.returncode}: "
                    f"{stderr or stdout or 'empty output'}"
                )
        except subprocess.TimeoutExpired:
            error = (
                "nvidia-smi timed out after "
                f"{self.args.gpu_health_timeout_sec:.1f}s"
            )
            record["error"] = error
        except (OSError, ValueError) as exc:
            error = f"cannot execute nvidia-smi: {exc}"
            record["error"] = error
        if not record["healthy"]:
            record["failure_reason"] = error
        self._append_gpu_health_record(record)
        return record["healthy"], record, (
            None if record["healthy"] else error
        )

    def _append_gpu_health_record(self, record):
        path = self.output_root / "gpu_health.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    @staticmethod
    def _stream_output(pipe, log):
        try:
            for line in pipe:
                sys.stdout.write(line)
                log.write(line)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    @staticmethod
    def _terminate_process_tree(process):
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    f"[BENCH-LAUNCH] process group {process.pid} remains "
                    "blocked after SIGKILL; continuing infrastructure "
                    "recovery without waiting indefinitely.",
                    file=sys.stderr,
                )

    @staticmethod
    def _result_is_resumable(
        path, algorithm, control_mode, seed, repeat, run_id
    ):
        if not path.is_file():
            return False
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            identity_matches = (
                result.get("run_id") == run_id
                and result.get("algorithm") == algorithm
                and result.get("control_mode") == control_mode
                and int(result.get("seed")) == int(seed)
                and int(result.get("repeat_index")) == int(repeat)
            )
        except Exception:
            return False
        # Infrastructure failures are retried on resume; completed experiment
        # outcomes (including collision and simulation timeout) are preserved.
        return identity_matches and result.get("termination_reason") not in (
            *INFRASTRUCTURE_TERMINATION_REASONS,
        )

    def _archive_stale_result(self, path):
        stale_dir = self.output_root / "stale_results"
        stale_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = stale_dir / f"{path.stem}_{stamp}{path.suffix}"
        path.replace(destination)
        print(f"[BENCH-LAUNCH] archived stale result: {destination}")

    @staticmethod
    def _result_termination_reason(path):
        try:
            return json.loads(path.read_text(encoding="utf-8")).get(
                "termination_reason"
            )
        except Exception:
            return "process_error"

    def _start_sidecar(self, run_id, algorithm):
        if algorithm not in ("ego", "dpmpc"):
            raise ValueError(f"unsupported sidecar algorithm: {algorithm}")
        self.sidecar_name = (
            f"{algorithm}_planner_bench_{os.getpid()}_{run_id}"
        )
        sidecar_log = (
            self.output_root / "logs" / f"{run_id}_{algorithm}_sidecar.log"
        )
        sidecar_log.parent.mkdir(parents=True, exist_ok=True)
        handle = sidecar_log.open("w", encoding="utf-8")
        command = ["docker", "run", "--rm", "--network", "host", "--name",
                   self.sidecar_name]
        if algorithm == "ego":
            command.extend([
                "--volume",
                (
                    str(
                        (
                            REPO_ROOT
                            / "navigation"
                            / "ego_planner"
                            / "ego-planner"
                            / "src"
                            / "isaac_ego_bridge"
                            / "scripts"
                            / "isaac_udp_ros_bridge.py"
                        ).resolve()
                    )
                    + ":/ws/src/isaac_ego_bridge/scripts/"
                    "isaac_udp_ros_bridge.py:ro"
                ),
                self.args.ego_image,
                "roslaunch",
                "ego_planner",
                "isaac_single.launch",
            ])
        else:
            command.extend([self.args.dpmpc_image])
        try:
            self.sidecar_process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.sidecar_process._benchmark_log_handle = handle
            # Isaac startup is much longer than this; the short check catches
            # a missing image/port conflict without delaying a healthy run.
            time.sleep(1.0)
            if self.sidecar_process.poll() is not None:
                raise RuntimeError(
                    f"{algorithm.upper()} sidecar exited early; "
                    f"inspect {sidecar_log}"
                )
        except Exception:
            handle.close()
            self.sidecar_process = None
            raise

    def _stop_sidecar(self):
        if self.sidecar_name:
            subprocess.run(
                ["docker", "stop", "-t", "3", self.sidecar_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        process = self.sidecar_process
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            handle = getattr(process, "_benchmark_log_handle", None)
            if handle is not None:
                handle.close()
        self.sidecar_process = None
        self.sidecar_name = None


def main():
    args = parse_args()
    launch_lock = None
    if not args.dry_run:
        launch_lock = _acquire_benchmark_launch_lock()
        if launch_lock is None:
            print(
                "[BENCH-RESUME] another navigation benchmark launcher "
                f"already holds {BENCHMARK_LAUNCH_LOCK_PATH}; refusing "
                "to start a duplicate process.",
                file=sys.stderr,
            )
            return 75
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(
        args.output_root
        or (DEFAULT_RESULTS_ROOT / f"benchmark_{timestamp}")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.algorithm == "both":
        base_algorithms = ("ego", "navrl")
    elif args.algorithm == "navrl_pair":
        base_algorithms = ("navrl", "navrl_no_shield")
    elif args.algorithm == "all":
        base_algorithms = BENCHMARK_ALGORITHMS
    else:
        base_algorithms = (args.algorithm,)
    scheduled_trials = []
    for seed_index, seed in enumerate(args.seeds):
        # Rotate variant order by seed so machine warm-up or thermal drift
        # does not always favor the same controller.
        offset = seed_index % len(base_algorithms)
        algorithms_for_seed = (
            base_algorithms[offset:] + base_algorithms[:offset]
        )
        for repeat in range(1, args.repeats + 1):
            for algorithm in algorithms_for_seed:
                scheduled_trials.append(
                    {
                        "run_id": (
                            f"{algorithm}_seed{seed:03d}_repeat{repeat:02d}"
                        ),
                        "algorithm": algorithm,
                        "controller_algorithm": _controller_algorithm(algorithm),
                        "safety_layers_enabled": _safety_layers_enabled(algorithm),
                        "seed": seed,
                        "repeat_index": repeat,
                        "order_index": len(scheduled_trials),
                    }
                )

    config = {
        "schema_version": 3,
        "experiment_id": f"navigation_benchmark_{timestamp}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": args.algorithm,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "evaluation_hz": args.eval_hz,
        "timeout_sim_sec": args.timeout_sec,
        "goal_radius_3d_m": args.goal_radius_m,
        "human_intrusion_threshold_m": args.human_intrusion_threshold_m,
        "human_intrusion_exit_threshold_m": (
            args.human_intrusion_exit_threshold_m
        ),
        "crowd_count_override": args.crowd_count,
        "crowd_layout": args.crowd_layout,
        "dense_profile": args.dense_profile,
        "use_px4": not args.no_px4,
        "headless": not args.no_headless,
        "isolation": "fresh_isaac_px4_planner_process_per_trial",
        "wall_timeout_sec": args.wall_timeout_sec,
        "infrastructure_retries": args.infrastructure_retries,
        "scheduled_trials": scheduled_trials,
        "expected_run_ids": [item["run_id"] for item in scheduled_trials],
    }
    config["configuration_sha256"] = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    config_path = output_root / "experiment_config.json"
    if args.resume and config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "algorithm",
            "seeds",
            "repeats",
            "evaluation_hz",
            "timeout_sim_sec",
            "goal_radius_3d_m",
            "human_intrusion_threshold_m",
            "human_intrusion_exit_threshold_m",
            "crowd_count_override",
            "crowd_layout",
            "dense_profile",
            "use_px4",
            "headless",
            "wall_timeout_sec",
            "infrastructure_retries",
            "expected_run_ids",
        )
        # Experiments created before the layout selector are sparse. This
        # preserves reboot/resume compatibility with those active runs.
        existing.setdefault("crowd_layout", "sparse")
        existing.setdefault(
            "dense_profile",
            "legacy"
            if existing.get("crowd_layout") == "dense"
            else "transverse40",
        )
        mismatches = [
            key for key in comparable_keys
            if existing.get(key) != config.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "--resume configuration does not match experiment_config.json: "
                + ", ".join(mismatches)
            )
        config = existing
        scheduled_trials = list(existing["scheduled_trials"])
    elif not args.resume:
        # Remove same-id results before the first partial aggregate so an old
        # file can never masquerade as a trial from this experiment.
        stale_dir = output_root / "stale_results"
        for trial in scheduled_trials:
            old_path = output_root / "runs" / f"{trial['run_id']}.json"
            if old_path.is_file():
                stale_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                old_path.replace(
                    stale_dir / f"{old_path.stem}_{stamp}{old_path.suffix}"
                )

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not args.dry_run:
        pending_run_ids = _pending_run_ids(
            output_root, scheduled_trials
        )
        _register_autoresume_job(
            args, output_root, config, pending_run_ids
        )
        if args.register_autoresume_only:
            print(
                "[BENCH-RESUME] registration complete; no trial was "
                "started."
            )
            return 0

    runner = TrialRunner(args, output_root)

    def stop_handler(_signum, _frame):
        runner.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    gpu_health_error = None
    try:
        for trial in scheduled_trials:
            max_attempts = 1 + int(args.infrastructure_retries)
            for attempt in range(1, max_attempts + 1):
                reason = runner.run(
                    trial["algorithm"],
                    int(trial["seed"]),
                    int(trial["repeat_index"]),
                )
                if not args.dry_run:
                    _update_autoresume_job(
                        output_root,
                        config,
                        _pending_run_ids(
                            output_root, scheduled_trials
                        ),
                        last_event=(
                            f"trial_finished:{trial['run_id']}:{reason}"
                        ),
                    )
                if reason not in INFRASTRUCTURE_TERMINATION_REASONS:
                    break
                if attempt < max_attempts:
                    print(
                        f"[BENCH-LAUNCH] infrastructure failure for "
                        f"{trial['run_id']}; retrying automatically "
                        f"({attempt}/{max_attempts - 1})."
                    )
    except GPUHealthError as exc:
        gpu_health_error = exc
        if not args.dry_run:
            _update_autoresume_job(
                output_root,
                config,
                _pending_run_ids(output_root, scheduled_trials),
                status="active",
                last_event="gpu_health_wait_aborted",
            )
        print(
            f"\n[BENCH-GPU] batch stopped safely: {exc}",
            file=sys.stderr,
        )
    except KeyboardInterrupt:
        if not args.dry_run:
            _update_autoresume_job(
                output_root,
                config,
                _pending_run_ids(output_root, scheduled_trials),
                status="active",
                last_event="interrupted",
            )
        print("\n[BENCH-LAUNCH] interrupted; completed results were preserved.")
        return 130
    finally:
        runner.stop()

    if gpu_health_error is not None:
        if not args.dry_run:
            write_aggregate(output_root)
        print(
            "[BENCH-GPU] completed results were preserved. Restore GPU "
            "health, then run the same command with --resume.",
            file=sys.stderr,
        )
        return 3

    if not args.dry_run:
        summary = write_aggregate(output_root)
        pending_run_ids = _pending_run_ids(
            output_root, scheduled_trials
        )
        if pending_run_ids:
            _update_autoresume_job(
                output_root,
                config,
                pending_run_ids,
                status="active",
                last_event="incomplete_after_retry_budget",
            )
            print(
                "\n[BENCH-RESUME] requested pass ended with "
                f"{len(pending_run_ids)} retryable or missing trials. "
                "The auto-resume service will retry them.",
                file=sys.stderr,
            )
            print(f"[BENCH-LAUNCH] results: {output_root}")
            return 4
        _update_autoresume_job(
            output_root,
            config,
            [],
            status="completed",
            last_event="all_trials_completed",
        )
        print("\n[BENCH-LAUNCH] all requested trials finished.")
        print(f"[BENCH-LAUNCH] results: {output_root}")
        print(json.dumps(summary["algorithms"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
