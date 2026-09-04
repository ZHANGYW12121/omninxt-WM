"""Train-only provenance contract for authoritative analytic Ego dynamics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ANALYTIC_EGO_CONTRACT_SCHEMA = "omninxt.analytic-ego.train-only.v1"
ANALYTIC_EGO_SELECTION_VERSION = (
    "episode-source-to-destination.applied-action-destination.v1")
DEFAULT_ANALYTIC_EGO_CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "configs" / "data" / "pure_dreamer_analytic_ego_formal_v4.json"
)


def analytic_ego_transitions(dataset: Any) -> dict[str, np.ndarray]:
    """Extract the exact physical transitions used to identify Ego dynamics.

    Compact replay stores the action applied over ``source -> destination`` on
    the destination row. Terminal destinations are retained (including their
    short actual ``dt``); terminal sources, reset destinations, invalid
    actions, and non-finite rows are excluded.
    """
    output: dict[str, list[np.ndarray]] = {
        "episode_seed": [],
        "source_frame": [],
        "source_ego": [],
        "destination_ego": [],
        "applied_action": [],
        "dt_s": [],
    }
    for episode_index, episode in enumerate(dataset.episodes):
        arrays = dataset._episode_arrays(episode_index)
        required = {
            "ego_state", "action", "action_valid", "simulation_time_s",
            "is_first", "is_last", "frame_index",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(
                f"{episode.directory}: analytic Ego fit lacks {missing}")
        ego = np.asarray(arrays["ego_state"], np.float64)
        action = np.asarray(arrays["action"], np.float64)
        action_valid = np.asarray(
            arrays["action_valid"], np.bool_).reshape(-1)
        is_first = np.asarray(arrays["is_first"], np.bool_).reshape(-1)
        is_last = np.asarray(arrays["is_last"], np.bool_).reshape(-1)
        frame = np.asarray(arrays["frame_index"], np.int64).reshape(-1)
        time_s = np.asarray(
            arrays["simulation_time_s"], np.float64).reshape(-1)
        length = ego.shape[0]
        if (
            ego.shape != (length, 14)
            or action.shape != (length, 4)
            or any(value.size != length for value in (
                action_valid, is_first, is_last, frame, time_s))
        ):
            raise ValueError(
                f"{episode.directory}: malformed analytic Ego fit arrays")
        dt_s = np.diff(time_s)
        valid = (
            ~is_last[:-1]
            & ~is_first[1:]
            & action_valid[1:]
            & np.isfinite(ego[:-1]).all(axis=-1)
            & np.isfinite(ego[1:]).all(axis=-1)
            & np.isfinite(action[1:]).all(axis=-1)
            & np.isfinite(dt_s)
            & (dt_s > 0.0)
        )
        if bool((np.abs(action[1:][valid]) > 1.0 + 1.0e-6).any()):
            raise ValueError(
                f"{episode.directory}: fit action exceeds normalized bounds")
        seed = int(episode.metadata.get("episode", {}).get("crowd_seed"))
        count = int(valid.sum())
        output["episode_seed"].append(
            np.full(count, seed, dtype=np.int64))
        output["source_frame"].append(frame[:-1][valid])
        output["source_ego"].append(ego[:-1][valid])
        output["destination_ego"].append(ego[1:][valid])
        output["applied_action"].append(action[1:][valid])
        output["dt_s"].append(dt_s[valid, None])
    if not output["episode_seed"]:
        raise ValueError("analytic Ego fit contains no eligible episodes")
    result = {
        key: np.concatenate(values, axis=0)
        for key, values in output.items()
    }
    if result["episode_seed"].size == 0:
        raise ValueError("analytic Ego fit contains no eligible transitions")
    return result


def analytic_ego_transition_sha256(
    transitions: Mapping[str, np.ndarray],
) -> str:
    """Hash canonical numeric fit inputs, including their row identities."""
    digest = hashlib.sha256()
    dtypes = {
        "episode_seed": "<i8",
        "source_frame": "<i8",
        "source_ego": "<f8",
        "destination_ego": "<f8",
        "applied_action": "<f8",
        "dt_s": "<f8",
    }
    for key, dtype in dtypes.items():
        if key not in transitions:
            raise KeyError(f"analytic Ego transition field {key!r} is missing")
        value = np.ascontiguousarray(
            np.asarray(transitions[key]).astype(dtype, copy=False))
        digest.update(key.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def analytic_ego_seed_sha256(seeds: Any) -> str:
    """Hash the ordered unique train-seed identity compactly."""
    values = np.asarray(sorted(map(int, seeds)), dtype="<i8")
    if values.size != np.unique(values).size:
        raise ValueError("analytic Ego training seeds must be unique")
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_analytic_ego_contract(
    path: str | Path = DEFAULT_ANALYTIC_EGO_CONTRACT_PATH,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: analytic Ego contract must be an object")
    if payload.get("schema") != ANALYTIC_EGO_CONTRACT_SCHEMA:
        raise ValueError(f"{path}: unsupported analytic Ego contract schema")
    if payload.get("transition_selection") != ANALYTIC_EGO_SELECTION_VERSION:
        raise ValueError(f"{path}: analytic Ego transition selection differs")
    result = dict(payload)
    result["contract_file_sha256"] = hashlib.sha256(
        path.read_bytes()).hexdigest()
    return result


def validate_analytic_ego_coefficients(
    contract: Mapping[str, Any], model: Any,
) -> None:
    velocity = np.asarray(
        contract.get("velocity_response"), np.float64).reshape(-1)
    attitude = np.asarray(
        contract.get("attitude_coefficients"), np.float64)
    configured_velocity = np.asarray(
        model.analytic_ego_velocity_response, np.float64).reshape(-1)
    configured_attitude = np.asarray(
        model.analytic_ego_attitude_coefficients, np.float64)
    if velocity.shape != (3,) or attitude.shape != (7, 2):
        raise ValueError("analytic Ego contract coefficient shapes are invalid")
    if not np.isfinite(velocity).all() or not np.isfinite(attitude).all():
        raise ValueError("analytic Ego contract coefficients are non-finite")
    if not np.allclose(
        velocity, configured_velocity, rtol=0.0, atol=1.0e-12,
    ) or not np.allclose(
        attitude, configured_attitude, rtol=0.0, atol=1.0e-12,
    ):
        raise ValueError(
            "configured analytic Ego dynamics differ from the train-only "
            "provenance contract")


def validate_train_only_analytic_ego_contract(
    dataset: Any,
    split_contract: Mapping[str, Any],
    model: Any,
    *,
    path: str | Path = DEFAULT_ANALYTIC_EGO_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate split identity, fit rows, input hash, and model coefficients."""
    contract = load_analytic_ego_contract(path)
    validate_analytic_ego_coefficients(contract, model)
    if contract.get("split_manifest_sha256") != split_contract.get(
        "manifest_sha256"):
        raise ValueError(
            "analytic Ego coefficients were identified under another split")
    split_seeds = sorted(map(
        int, split_contract.get("training_seeds", ())))
    if int(contract.get("training_seed_count", -1)) != len(split_seeds) \
            or contract.get("training_seed_sha256") != (
                analytic_ego_seed_sha256(split_seeds)):
        raise ValueError(
            "analytic Ego coefficient training seeds differ from replay split")
    transitions = analytic_ego_transitions(dataset)
    count = int(transitions["episode_seed"].size)
    if int(contract.get("transition_count", -1)) != count:
        raise ValueError(
            "analytic Ego coefficient transition count differs from replay")
    actual_hash = analytic_ego_transition_sha256(transitions)
    if contract.get("transition_sha256") != actual_hash:
        raise ValueError(
            "analytic Ego coefficient fit inputs differ from train replay")
    if not math.isfinite(float(contract.get("nominal_dt_s", float("nan")))):
        raise ValueError("analytic Ego contract lacks a finite nominal dt")
    return contract
