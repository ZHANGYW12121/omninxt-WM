#!/usr/bin/env python3
"""Resume the single active navigation benchmark registered by the runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_PATH = SCRIPT_DIR / "navigation_benchmark_autoresume_state.json"
RUNNER_PATH = SCRIPT_DIR / "run_navigation_benchmark.py"


def main():
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"[BENCH-RESUME] no active-job state at {STATE_PATH}; idle."
        )
        return 0
    except Exception as exc:
        print(
            f"[BENCH-RESUME] cannot read {STATE_PATH}: {exc}",
            file=sys.stderr,
        )
        return 1

    if state.get("status") != "active":
        print(
            "[BENCH-RESUME] registered experiment is not active "
            f"(status={state.get('status')!r}); idle."
        )
        return 0

    command = state.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        print(
            "[BENCH-RESUME] active-job command is missing or invalid.",
            file=sys.stderr,
        )
        return 1
    if Path(command[0]).resolve() != RUNNER_PATH.resolve():
        print(
            f"[BENCH-RESUME] refusing unexpected runner: {command[0]}",
            file=sys.stderr,
        )
        return 1
    if "--resume" not in command:
        command.append("--resume")

    print(
        "[BENCH-RESUME] resuming active experiment: "
        f"output={state.get('output_root')}, "
        f"pending={state.get('pending_count')}, "
        f"next={state.get('next_pending_run_id')}"
    )
    print("[BENCH-RESUME] command:", " ".join(command))
    while True:
        try:
            docker = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
                check=False,
            )
            if docker.returncode == 0:
                break
        except (OSError, subprocess.TimeoutExpired):
            pass
        print(
            "[BENCH-RESUME] Docker is not ready; retrying in 10 seconds."
        )
        time.sleep(10.0)
    os.chdir(SCRIPT_DIR)
    os.execv(command[0], command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
