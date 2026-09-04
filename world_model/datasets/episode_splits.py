"""Reproducible episode-level splits for compact skeleton datasets."""

from __future__ import annotations

import json
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .compact_skeleton_v3 import CompactEpisode, CompactSkeletonV3Dataset


def episode_split_manifest_sha256(path: str | Path) -> str:
    """Hash the exact immutable manifest bytes used to assign seed roles."""
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


@dataclass(frozen=True)
class EpisodeSplitManifest:
    name: str
    seed_min: int
    seed_max: int
    excluded_seeds: frozenset[int]
    excluded_termination_reasons: frozenset[str]
    validation_seeds: frozenset[int]
    test_seeds: frozenset[int]
    expected: Mapping[str, Mapping[str, int]]
    evaluation_protocol: str = "development_posthoc_v1"
    validation_collection_contract: Mapping[str, Any] | None = None
    scene_contract: Mapping[str, Any] | None = None


def episode_seed(episode: CompactEpisode) -> int:
    value = episode.metadata.get("episode", {}).get("crowd_seed")
    if value is None:
        raise ValueError(f"{episode.directory}: metadata episode.crowd_seed is missing")
    return int(value)


def scene_generator_population_contract(
    contract: Mapping[str, Any], *, seed: int,
    generator_contract: str,
) -> dict[str, Any]:
    """Resolve the deterministic population rule for one generator revision.

    Scene schema v2 permits an older replay generator to coexist with the
    current generator after a capacity migration.  The explicit generator
    identity remains part of the scenario key, so a seed can never silently
    change population rules.  Legacy generators are bounded by their last
    admitted seed; all prospective collection uses the top-level generator.
    """
    schema = contract.get("schema")
    if schema == "omninxt.warehouse-crowd-scene.v1":
        if generator_contract and generator_contract != str(
            contract.get("generator_contract", "")
        ):
            raise ValueError("scene generator differs from the frozen contract")
        return {
            "generator_contract": str(contract["generator_contract"]),
            "minimum_people": int(contract["minimum_people"]),
            "maximum_people": int(contract["maximum_people"]),
            "population_rule": str(contract["population_rule"]),
        }
    if schema != "omninxt.warehouse-crowd-scene.v2":
        raise ValueError(
            "scene contract schema must be omninxt.warehouse-crowd-scene.v1 "
            "or omninxt.warehouse-crowd-scene.v2")
    if contract.get("population_rule") != (
        "generator_scoped_python_random_seed_choice_inclusive_v2"
    ):
        raise ValueError("scene contract has an unsupported population rule")
    current = {
        "generator_contract": str(contract["generator_contract"]),
        "minimum_people": int(contract["minimum_people"]),
        "maximum_people": int(contract["maximum_people"]),
        "population_rule": "python_random_seed_choice_inclusive_v1",
    }
    if not generator_contract or generator_contract == current[
        "generator_contract"
    ]:
        return current
    legacy = contract.get("legacy_generator_contracts", ())
    if not isinstance(legacy, list):
        raise ValueError("legacy_generator_contracts must be a list")
    for raw in legacy:
        if not isinstance(raw, Mapping) or str(raw.get(
            "generator_contract", ""
        )) != generator_contract:
            continue
        maximum_seed = int(raw["maximum_seed"])
        if int(seed) > maximum_seed:
            raise ValueError(
                "legacy scene generator was used after its frozen final seed")
        return {
            "generator_contract": generator_contract,
            "minimum_people": int(raw["minimum_people"]),
            "maximum_people": int(raw["maximum_people"]),
            "population_rule": "python_random_seed_choice_inclusive_v1",
        }
    raise ValueError("scene generator differs from the frozen contract")


def validate_episode_scene_contract(
    episode: CompactEpisode,
    contract: Mapping[str, Any] | None,
    *,
    require_explicit_metadata: bool,
) -> None:
    """Reject a seed whose crowd generator/configuration changed roles.

    A crowd seed is not a complete scenario identity: the same integer can
    generate sparse, dense, differently sized, or differently directed
    scenes.  Formal split membership is therefore conditional on this frozen
    scene contract.  Historical train-prefill metadata can be authenticated
    by its deterministic scene key; newly collected reserved/online episodes
    must also carry every explicit generator field.
    """
    if contract is None:
        return
    metadata = episode.metadata.get("episode") or {}
    seed = episode_seed(episode)
    label = str(episode.directory)
    try:
        people = int(metadata["crowd_num_people"])
        scene_key_template = str(contract["scene_key_template"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: scene identity lacks a valid seed/population contract"
        ) from error
    recorded_generator = str(metadata.get("crowd_scene_contract", ""))
    if (
        not recorded_generator
        and contract.get("schema") == "omninxt.warehouse-crowd-scene.v2"
    ):
        # Imported historical train episodes can predate the explicit
        # generator field.  Authenticate them by the old deterministic
        # seed->population mapping and its frozen final seed. Prospective
        # validation still requires the explicit field below.
        for raw in contract.get("legacy_generator_contracts", ()):
            if not isinstance(raw, Mapping) or seed > int(raw[
                "maximum_seed"
            ]):
                continue
            legacy_people = random.Random(seed).choice(tuple(range(
                int(raw["minimum_people"]),
                int(raw["maximum_people"]) + 1,
            )))
            if people == legacy_people:
                recorded_generator = str(raw["generator_contract"])
                break
    try:
        population = scene_generator_population_contract(
            contract, seed=seed, generator_contract=recorded_generator)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: {error}") from error
    minimum_people = int(population["minimum_people"])
    maximum_people = int(population["maximum_people"])
    if minimum_people <= 0 or maximum_people < minimum_people:
        raise ValueError("scene population bounds are invalid")
    if not minimum_people <= people <= maximum_people:
        raise ValueError(
            f"{label}: crowd population {people} lies outside the frozen "
            f"[{minimum_people}, {maximum_people}] scene contract")
    population_rule = population.get("population_rule")
    if population_rule != "python_random_seed_choice_inclusive_v1":
        raise ValueError("scene contract has an unsupported population rule")
    expected_people = random.Random(seed).choice(
        tuple(range(minimum_people, maximum_people + 1)))
    if people != expected_people:
        raise ValueError(
            f"{label}: seed {seed} crowd population differs from the frozen "
            f"generator: {people} != {expected_people}")
    expected_key = scene_key_template.format(
        crowd_seed=seed, crowd_num_people=people)
    if str(metadata.get("scene_key", "")) != expected_key:
        raise ValueError(
            f"{label}: scene_key differs from the frozen scenario identity")
    recorded_people = metadata.get("people")
    if isinstance(recorded_people, list) and len(recorded_people) != people:
        raise ValueError(
            f"{label}: recorded person definitions differ from crowd_num_people")

    explicit_fields = {
        "crowd_layout": "crowd_layout",
        "crowd_dense_profile": "dense_profile",
        "crowd_template_version": "template_version",
        "crowd_group_spacing": "group_spacing",
        "crowd_direction": "direction",
        "crowd_drone_distance": "drone_distance",
        "crowd_speed": "speed",
        "crowd_scene_contract": "generator_contract",
    }
    for metadata_key, contract_key in explicit_fields.items():
        if metadata_key not in metadata:
            if require_explicit_metadata:
                raise ValueError(
                    f"{label}: formal scene metadata lacks {metadata_key}")
            continue
        expected_value = (
            population["generator_contract"]
            if contract_key == "generator_contract" else
            contract[contract_key]
        )
        if str(metadata[metadata_key]) != str(expected_value):
            raise ValueError(
                f"{label}: {metadata_key} differs from the frozen scene contract")


def load_episode_split_manifest(path: str | Path) -> EpisodeSplitManifest:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    excluded_seeds = set(map(int, payload.get("excluded_seeds", ())))
    for index, raw_range in enumerate(payload.get("excluded_seed_ranges", ())):
        if (
            not isinstance(raw_range, (list, tuple))
            or len(raw_range) != 2
            or isinstance(raw_range[0], bool)
            or isinstance(raw_range[1], bool)
        ):
            raise ValueError(
                f"{path}: excluded_seed_ranges[{index}] must be [min,max]")
        lower, upper = map(int, raw_range)
        if lower > upper:
            raise ValueError(
                f"{path}: excluded_seed_ranges[{index}] is reversed")
        excluded_seeds.update(range(lower, upper + 1))
    manifest = EpisodeSplitManifest(
        name=str(payload["name"]),
        seed_min=int(payload["seed_min"]),
        seed_max=int(payload["seed_max"]),
        excluded_seeds=frozenset(excluded_seeds),
        excluded_termination_reasons=frozenset(
            map(str, payload.get("excluded_termination_reasons", ()))),
        validation_seeds=frozenset(map(int, payload["validation_seeds"])),
        test_seeds=frozenset(map(int, payload["test_seeds"])),
        expected=payload.get("expected", {}),
        evaluation_protocol=str(payload.get(
            "evaluation_protocol", "development_posthoc_v1")),
        validation_collection_contract=(
            dict(payload["validation_collection_contract"])
            if isinstance(
                payload.get("validation_collection_contract"), Mapping)
            else None
        ),
        scene_contract=(
            dict(payload["scene_contract"])
            if isinstance(payload.get("scene_contract"), Mapping)
            else None
        ),
    )
    if manifest.seed_min > manifest.seed_max:
        raise ValueError(f"{path}: seed_min must not exceed seed_max")
    if manifest.evaluation_protocol not in {
        "development_posthoc_v1",
        "prospective_uncollected_holdout_v1",
        "prospective_actor_independent_validation_v2",
    }:
        raise ValueError(
            f"{path}: unknown evaluation_protocol "
            f"{manifest.evaluation_protocol!r}")
    if (
        manifest.evaluation_protocol
        == "prospective_actor_independent_validation_v2"
        and (
            not manifest.validation_collection_contract
            or not manifest.scene_contract
        )
    ):
        raise ValueError(
            f"{path}: prospective v2 requires validation collection and "
            "scene contracts")
    overlap = manifest.validation_seeds & manifest.test_seeds
    if overlap:
        raise ValueError(f"{path}: validation/test seed overlap: {sorted(overlap)}")
    reserved = manifest.validation_seeds | manifest.test_seeds
    outside = {
        seed for seed in reserved
        if seed < manifest.seed_min or seed > manifest.seed_max
    }
    if outside:
        raise ValueError(f"{path}: split seeds outside range: {sorted(outside)}")
    excluded_reserved = reserved & manifest.excluded_seeds
    if excluded_reserved:
        raise ValueError(
            f"{path}: excluded seeds assigned to validation/test: "
            f"{sorted(excluded_reserved)}")
    return manifest


def split_compact_dataset_windows(
    dataset: CompactSkeletonV3Dataset,
    manifest: EpisodeSplitManifest,
    *,
    verify_expected: bool = True,
    required_splits: tuple[str, ...] = ("train", "validate", "test"),
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Return window indices and episode seeds without crossing episodes.

    The strict default is useful for diagnostics that intentionally inspect a
    combined replay.  Production learners and physically separate holdout
    evaluators pass exactly the role they are authorized to read; the complete
    reserved populations still come from the immutable manifest.
    """
    valid_split_names = frozenset(("train", "validate", "test"))
    required = tuple(dict.fromkeys(map(str, required_splits)))
    unknown = sorted(set(required).difference(valid_split_names))
    if unknown or not required:
        raise ValueError(
            "required_splits must be a non-empty subset of "
            f"{sorted(valid_split_names)}; got {unknown or required}")
    episode_split: dict[int, str] = {}
    split_seeds = {"train": [], "validate": [], "test": []}
    for episode_index, episode in enumerate(dataset.episodes):
        seed = episode_seed(episode)
        if seed < manifest.seed_min or seed > manifest.seed_max:
            continue
        reason = str(episode.summary.get("termination_reason", ""))
        if seed in manifest.excluded_seeds:
            continue
        if reason in manifest.excluded_termination_reasons:
            continue
        if seed in manifest.validation_seeds:
            split = "validate"
        elif seed in manifest.test_seeds:
            split = "test"
        else:
            split = "train"
        validate_episode_scene_contract(
            episode,
            manifest.scene_contract,
            require_explicit_metadata=(
                split in {"validate", "test"}
                and bool((manifest.scene_contract or {}).get(
                    "require_explicit_metadata_for_reserved_splits", False))
            ),
        )
        episode_split[episode_index] = split
        split_seeds[split].append(seed)

    window_indices = {"train": [], "validate": [], "test": []}
    for window_index, (episode_index, _) in enumerate(dataset.windows):
        split = episode_split.get(episode_index)
        if split is not None:
            window_indices[split].append(window_index)
    for name in window_indices:
        split_seeds[name].sort()
        if name in required and (
            not split_seeds[name] or not window_indices[name]
        ):
            raise ValueError(f"Split {name!r} has no eligible episodes/windows")
        expected = manifest.expected.get(name, {})
        if verify_expected and expected and name in required:
            actual = {
                "episodes": len(split_seeds[name]),
                "windows": len(window_indices[name]),
            }
            mismatch = {
                key: (int(value), actual[key]) for key, value in expected.items()
                if key in actual and int(value) != actual[key]
            }
            if mismatch:
                raise ValueError(
                    f"Split {name!r} differs from manifest expectations: {mismatch}")
    all_seeds = [seed for values in split_seeds.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Episode seed leakage detected between data splits")
    return window_indices, split_seeds


def validate_prospective_validation_collection(
    dataset: CompactSkeletonV3Dataset,
    selected_seeds: list[int],
    expected_contract: Mapping[str, Any],
    expected_scene_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that every validation action came from the frozen explorer.

    This validator is shared by every replay-based release audit.  Requiring
    the same physical replay root and the same collector evidence in those
    reports prevents one audit from silently substituting model-generated or
    selectively recollected validation episodes with the same seed labels.
    """
    selected = set(map(int, selected_seeds))
    episodes = [
        episode for episode in dataset.episodes
        if int(episode.metadata["episode"]["crowd_seed"]) in selected
    ]
    by_seed: dict[int, CompactEpisode] = {}
    phase_counts: dict[str, int] = defaultdict(int)
    checkpoint_hashes: set[str] = set()
    recording_run_ids: set[str] = set()
    saturation_rows = np.zeros(3, np.int64)
    acted_rows = 0
    for episode in episodes:
        episode_metadata = episode.metadata.get("episode") or {}
        seed = int(episode_metadata.get("crowd_seed", -1))
        if seed in by_seed:
            raise RuntimeError(
                f"prospective validation contains duplicate seed {seed}")
        by_seed[seed] = episode
        validate_episode_scene_contract(
            episode,
            expected_scene_contract,
            require_explicit_metadata=True,
        )
        evidence = episode_metadata.get("collection_policy_evidence")
        if not isinstance(evidence, Mapping):
            raise RuntimeError(
                f"validation seed {seed} lacks collection-policy evidence")
        if evidence.get("contract") != dict(expected_contract):
            raise RuntimeError(
                f"validation seed {seed} used a non-registered collector")
        episode_evidence = evidence.get("episode")
        source = evidence.get("source")
        if not isinstance(episode_evidence, Mapping) or not isinstance(
            source, Mapping
        ):
            raise RuntimeError(
                f"validation seed {seed} has malformed policy provenance")
        if int(episode_evidence.get("collection_scene_seed", -1)) != seed:
            raise RuntimeError(
                f"validation seed {seed} collector used a different scene key")
        target_count = len(expected_contract.get("target_sequence", ()))
        expected_phase = seed % target_count if target_count else -1
        phase = int(episode_evidence.get("initial_target_index", -1))
        if phase != expected_phase:
            raise RuntimeError(
                f"validation seed {seed} has non-reproducible start phase")
        phase_counts[str(phase)] += 1
        if source.get("evaluated_checkpoint_independent") is not True:
            raise RuntimeError(
                f"validation seed {seed} was checkpoint-policy dependent")
        checkpoint_hash = str(source.get("checkpoint_sha256", ""))
        if len(checkpoint_hash) != 64 or any(
            char not in "0123456789abcdef" for char in checkpoint_hash
        ):
            raise RuntimeError(
                f"validation seed {seed} lacks a source checkpoint hash")
        checkpoint_hashes.add(checkpoint_hash)
        recording_run_ids.add(str(episode_metadata.get("recording_run_id", "")))
        episode_has_action = False
        for chunk in episode.chunks:
            with np.load(chunk, allow_pickle=False) as arrays:
                try:
                    steps = np.asarray(
                        arrays["collector_policy_step"], np.int64).reshape(-1)
                    action_valid = np.asarray(
                        arrays["action_valid"], np.bool_).reshape(-1)
                    policy_valid = np.asarray(
                        arrays["policy_action_valid"], np.bool_).reshape(-1)
                    action = np.asarray(arrays["action"], np.float32)
                except KeyError as error:
                    raise RuntimeError(
                        f"validation seed {seed} chunk lacks collector fields"
                    ) from error
            if (
                action.shape != (steps.size, 4)
                or action_valid.shape != steps.shape
                or policy_valid.shape != steps.shape
                or bool((steps < -1).any())
                or bool((steps > 0).any())
            ):
                raise RuntimeError(
                    f"validation seed {seed} has invalid collector arrays")
            acted = steps == 0
            if not bool((action_valid[acted] & policy_valid[acted]).all()):
                raise RuntimeError(
                    f"validation seed {seed} has invalid explorer actions")
            episode_has_action |= bool(acted.any())
            policy_axes = action[acted][:, (0, 1, 3)]
            saturation_rows += np.sum(
                np.abs(policy_axes) >= 0.95, axis=0, dtype=np.int64)
            acted_rows += int(acted.sum())
        if not episode_has_action:
            raise RuntimeError(
                f"validation seed {seed} contains no explorer action")
    if set(by_seed) != selected:
        raise RuntimeError(
            "validation collection evidence does not cover the exact split")
    if len(checkpoint_hashes) != 1:
        raise RuntimeError(
            "validation collection used more than one observation-adapter "
            "checkpoint")
    if len(recording_run_ids) != 1 or "" in recording_run_ids:
        raise RuntimeError(
            "validation collection is not one immutable recording run")
    return {
        "contract": dict(expected_contract),
        "scene_contract": (
            None if expected_scene_contract is None
            else dict(expected_scene_contract)),
        "episode_count": len(episodes),
        "all_actions_actor_independent_step_zero": True,
        "acted_row_count": int(acted_rows),
        "physical_policy_axis_saturation_threshold": 0.95,
        "physical_policy_axis_saturation_row_counts": {
            "forward": int(saturation_rows[0]),
            "lateral": int(saturation_rows[1]),
            "yaw": int(saturation_rows[2]),
        },
        "initial_target_phase_episode_counts": dict(sorted(
            phase_counts.items())),
        "collector_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "recording_run_id": next(iter(recording_run_ids)),
    }
