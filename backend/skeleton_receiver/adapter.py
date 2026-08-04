"""Convert validated Nano packets into the current Ego--Human model schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.skeleton3d.protocol import COCO17_JOINT_NAMES, JOINT_FIELDS

from .optimizer import CausalSkeletonOptimizer, RefinedPerson


FIELD = {name: index for index, name in enumerate(JOINT_FIELDS)}
BODY_JOINT_SLICE = slice(5, 17)


@dataclass(frozen=True)
class WorldModelHumanFrame:
    """One online Human observation without an Ego/goal observation.

    The arrays omit batch/time dimensions.  A model caller adds ``[B,T]`` and
    obtains Ego14 and Goal inputs from the flight-state/task-state providers.
    """

    schema: str
    sequence: int
    timestamp_ns: int
    skeleton: np.ndarray       # [N,17,7] refined base_link position/velocity/confidence
    skeleton_raw: np.ndarray   # [N,17,7] transmitted position/confidence diagnostics
    human_root: np.ndarray     # [N,10] position, velocity, extent, confidence
    human_joints: np.ndarray   # [N,17,7] root-relative position/velocity/confidence
    human_mask: np.ndarray     # [N]
    joint_mask: np.ndarray     # [N,17]
    joint_inferred_mask: np.ndarray  # [N,17], kinematic rather than fresh depth
    human_ids: np.ndarray      # [N], -1 for padding
    human_is_first: np.ndarray # [N]
    joint_uncertainty_m: np.ndarray      # [N,17]
    joint_prediction_age_ms: np.ndarray  # [N,17]
    truncated_people: int

    def as_model_batch(self) -> dict[str, np.ndarray]:
        """Return Human tensors with singleton batch/time dimensions."""

        return {
            "skeleton": self.skeleton[None, None],
            "human_root": self.human_root[None, None],
            "human_joints": self.human_joints[None, None],
            "human_mask": self.human_mask[None, None],
            "joint_mask": self.joint_mask[None, None],
            "human_ids": self.human_ids[None, None],
            "human_is_first": self.human_is_first[None, None],
            "truncated_people": np.asarray(
                [[self.truncated_people]], dtype=np.int32
            ),
        }


def _root_and_relative(
    skeleton: np.ndarray, human_mask: np.ndarray, joint_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy equivalent of the world-model Human root/pose split."""

    xyz = skeleton[..., :3]
    velocity = skeleton[..., 3:6]
    confidence = skeleton[..., 6]
    valid = joint_mask & human_mask[:, None]
    weight = valid[..., None].astype(np.float32)
    denom = np.maximum(weight.sum(axis=-2), 1.0)
    mean_pos = (xyz * weight).sum(axis=-2) / denom
    mean_vel = (velocity * weight).sum(axis=-2) / denom
    hips_valid = valid[:, 11] & valid[:, 12]
    hip_pos = 0.5 * (xyz[:, 11] + xyz[:, 12])
    hip_vel = 0.5 * (velocity[:, 11] + velocity[:, 12])
    root_pos = np.where(hips_valid[:, None], hip_pos, mean_pos)
    root_vel = np.where(hips_valid[:, None], hip_vel, mean_vel)

    low = np.where(valid[..., None], xyz, np.inf).min(axis=-2)
    high = np.where(valid[..., None], xyz, -np.inf).max(axis=-2)
    extent = np.where(valid.any(axis=-1)[:, None], high - low, 0.0)
    root_confidence = (
        (confidence * valid.astype(np.float32)).sum(axis=-1)
        / np.maximum(valid.sum(axis=-1), 1)
    )
    root = np.concatenate(
        (root_pos, root_vel, extent, root_confidence[:, None]), axis=-1
    ).astype(np.float32)
    root[~human_mask] = 0.0

    relative = skeleton.copy()
    relative[..., :3] -= root_pos[:, None]
    relative[..., 3:6] -= root_vel[:, None]
    relative[~valid] = 0.0
    return root, relative.astype(np.float32)


class NanoHumanObservationAdapter:
    """Stable-slot adapter with causal kinematic skeleton refinement."""

    def __init__(
        self, max_people: int = 20, *, velocity_ema: float = 0.4,
        max_velocity_mps: float = 15.0, max_velocity_dt_s: float = 0.5,
    ) -> None:
        if max_people <= 0:
            raise ValueError("max_people must be positive")
        if not 0.0 < velocity_ema <= 1.0:
            raise ValueError("velocity_ema must be in (0,1]")
        self.max_people = int(max_people)
        self.velocity_ema = float(velocity_ema)
        self.max_velocity_mps = float(max_velocity_mps)
        self.max_velocity_dt_ns = int(float(max_velocity_dt_s) * 1e9)
        self.preferred_slot: dict[int, int] = {}
        self.optimizer = CausalSkeletonOptimizer(
            max_track_gap_s=max_velocity_dt_s,
        )
        self.previous_slot_ids = np.full(self.max_people, -1, dtype=np.int64)
        self.last_timestamp_ns: int | None = None

    def reset(self) -> None:
        """Reset temporal identity and velocity state after a source restart."""

        self.preferred_slot.clear()
        self.optimizer.reset()
        self.previous_slot_ids.fill(-1)
        self.last_timestamp_ns = None

    @staticmethod
    def _decode_person(person: dict) -> tuple[int, np.ndarray]:
        rows = np.asarray(person["joints"], dtype=np.float32)
        return int(person["person_id"]), rows

    @staticmethod
    def _rank_key(item: RefinedPerson):
        person_id, xyz, confidence, valid = (
            item.person_id, item.xyz, item.confidence, item.valid
        )
        valid = valid.copy()
        valid[:5] = False
        if valid[11] and valid[12]:
            root = 0.5 * (xyz[11] + xyz[12])
        else:
            root = xyz[valid].mean(axis=0)
        return (float(np.linalg.norm(root)), -float(confidence[valid].mean()), person_id)

    def adapt(self, packet: dict) -> WorldModelHumanFrame:
        timestamp_ns = int(packet["timestamp_ns"])
        if self.last_timestamp_ns is not None:
            delta = timestamp_ns - self.last_timestamp_ns
            if delta <= 0 or delta > self.max_velocity_dt_ns:
                self.reset()
        self.last_timestamp_ns = timestamp_ns

        candidates = []
        active_ids: set[int] = set()
        for person in packet["people"]:
            person_id, rows = self._decode_person(person)
            active_ids.add(person_id)
            coordinate_valid = rows[:, FIELD["coordinate_valid"]].astype(np.bool_)
            if (not coordinate_valid[BODY_JOINT_SLICE].any()
                    and person_id not in self.optimizer.states):
                continue
            refined = self.optimizer.update(person_id, timestamp_ns, rows)
            if refined.valid[BODY_JOINT_SLICE].any():
                candidates.append(refined)
        for person_id, state in list(self.optimizer.states.items()):
            if person_id in active_ids:
                continue
            if timestamp_ns - state.last_presence_ns <= self.optimizer.max_hold_ns:
                predicted = self.optimizer.predict_absent(person_id, timestamp_ns)
                if predicted.valid[BODY_JOINT_SLICE].any():
                    candidates.append(predicted)
        self.optimizer.prune(active_ids, timestamp_ns)
        stale_ids = [person_id for person_id in self.preferred_slot
                     if person_id not in self.optimizer.states]
        for person_id in stale_ids:
            self.preferred_slot.pop(person_id, None)
        candidates.sort(key=self._rank_key)
        selected = candidates[: self.max_people]

        assignment: dict[int, int] = {}
        used: set[int] = set()
        for index, item in enumerate(selected):
            slot = self.preferred_slot.get(item.person_id)
            if slot is not None and slot not in used:
                assignment[index] = slot
                used.add(slot)
        free_slots = iter(slot for slot in range(self.max_people) if slot not in used)
        for index in range(len(selected)):
            if index not in assignment:
                assignment[index] = next(free_slots)

        joint_count = len(COCO17_JOINT_NAMES)
        skeleton = np.zeros((self.max_people, joint_count, 7), np.float32)
        skeleton_raw = np.zeros((self.max_people, joint_count, 7), np.float32)
        human_mask = np.zeros(self.max_people, np.bool_)
        joint_mask = np.zeros((self.max_people, joint_count), np.bool_)
        joint_inferred_mask = np.zeros((self.max_people, joint_count), np.bool_)
        human_ids = np.full(self.max_people, -1, np.int64)
        human_is_first = np.zeros(self.max_people, np.bool_)
        joint_uncertainty = np.full((self.max_people, joint_count), 2.0, np.float32)
        prediction_age = np.zeros((self.max_people, joint_count), np.float32)
        for index, refined in enumerate(selected):
            person_id = refined.person_id
            slot = assignment[index]
            self.preferred_slot[person_id] = slot
            human_mask[slot] = True
            output_valid = refined.valid.copy()
            output_valid[:5] = False
            output_inferred = refined.inferred & output_valid
            joint_mask[slot] = output_valid
            joint_inferred_mask[slot] = output_inferred
            human_ids[slot] = person_id
            human_is_first[slot] = self.previous_slot_ids[slot] != person_id
            skeleton[slot, :, :3] = refined.xyz
            skeleton[slot, :, 3:6] = refined.velocity
            skeleton[slot, :, 6] = refined.confidence
            skeleton[slot, ~output_valid] = 0.0
            skeleton_raw[slot, :, :3] = refined.raw_xyz
            skeleton_raw[slot, :, 6] = refined.raw_confidence
            skeleton_raw[slot, ~refined.raw_valid] = 0.0
            joint_uncertainty[slot] = refined.uncertainty_m
            prediction_age[slot] = refined.prediction_age_ms
        self.previous_slot_ids = human_ids.copy()

        root, relative = _root_and_relative(skeleton, human_mask, joint_mask)
        return WorldModelHumanFrame(
            schema="omninxt.world_model.human.v1",
            sequence=int(packet["sequence"]),
            timestamp_ns=timestamp_ns,
            skeleton=skeleton,
            skeleton_raw=skeleton_raw,
            human_root=root,
            human_joints=relative,
            human_mask=human_mask,
            joint_mask=joint_mask,
            joint_inferred_mask=joint_inferred_mask,
            human_ids=human_ids,
            human_is_first=human_is_first,
            joint_uncertainty_m=joint_uncertainty,
            joint_prediction_age_ms=prediction_age,
            truncated_people=max(0, len(candidates) - self.max_people),
        )
