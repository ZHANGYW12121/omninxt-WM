"""Causal confidence-aware refinement for noisy transmitted COCO-17 poses.

The optimizer deliberately consumes only fields present in the formal Nano
wire packet.  It separates common root motion from root-relative articulation,
trusts radial (depth) measurements less than tangential measurements, rejects
large isolated innovations, and projects the result onto a slowly learned
person-specific skeleton.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from interfaces.skeleton3d.protocol import JOINT_FIELDS


FIELD = {name: index for index, name in enumerate(JOINT_FIELDS)}
TORSO_JOINTS = np.asarray((5, 6, 11, 12), dtype=np.int64)
BODY_JOINTS = np.arange(5, 17, dtype=np.int64)
BODY_JOINT_MASK = np.arange(17) >= 5
BODY_BONES = (
    (5, 6), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
SYMMETRIC_BONE_GROUPS = (
    ((5, 7), (6, 8)),
    ((7, 9), (8, 10)),
    ((5, 11), (6, 12)),
    ((11, 13), (12, 14)),
    ((13, 15), (14, 16)),
)
LENGTH_RANGES = {
    (5, 6): (0.18, 0.70),
    (11, 12): (0.12, 0.60),
    (5, 7): (0.16, 0.55), (6, 8): (0.16, 0.55),
    (7, 9): (0.14, 0.50), (8, 10): (0.14, 0.50),
    (5, 11): (0.25, 0.85), (6, 12): (0.25, 0.85),
    (11, 13): (0.22, 0.75), (12, 14): (0.22, 0.75),
    (13, 15): (0.20, 0.75), (14, 16): (0.20, 0.75),
}

# Upright COCO-17 template in a pelvis-centred body frame.  It is used only
# when a person is present but one or more joints have no usable depth.  The
# template is aligned to the person's measured torso/lateral axes and scaled
# from learned bone lengths; it is never treated as a new measurement.
DEFAULT_LOCAL_TEMPLATE = np.asarray([
    [0.00,  0.00,  0.82],  # nose
    [0.00,  0.04,  0.86], [0.00, -0.04,  0.86],
    [0.00,  0.09,  0.82], [0.00, -0.09,  0.82],
    [0.00,  0.20,  0.52], [0.00, -0.20,  0.52],
    [0.00,  0.34,  0.27], [0.00, -0.34,  0.27],
    [0.00,  0.40,  0.02], [0.00, -0.40,  0.02],
    [0.00,  0.15,  0.00], [0.00, -0.15,  0.00],
    [0.00,  0.15, -0.45], [0.00, -0.15, -0.45],
    [0.00,  0.15, -0.90], [0.00, -0.15, -0.90],
], dtype=np.float64)
MIRROR_JOINT = {
    1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7,
    9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15,
}
KINEMATIC_PARENT = {
    1: 0, 2: 0, 3: 1, 4: 2,
    5: 11, 6: 12, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 5, 12: 6, 13: 11, 14: 12, 15: 13, 16: 14,
}
FILL_ORDER = (11, 12, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16)
SOURCE_WEIGHT = {
    0: 0.0,   # invalid
    1: 1.0,   # direct stereo geometry
    2: 0.45,  # HITNet fallback
    3: 0.0,   # upstream temporal prediction is not a new measurement
    4: 0.35,  # other
    5: 1.0,   # Isaac ground truth, when used in simulation
}


@dataclass(frozen=True)
class RefinedPerson:
    person_id: int
    raw_xyz: np.ndarray
    raw_valid: np.ndarray
    raw_confidence: np.ndarray
    xyz: np.ndarray
    velocity: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    inferred: np.ndarray
    uncertainty_m: np.ndarray
    prediction_age_ms: np.ndarray


@dataclass
class _PersonState:
    timestamp_ns: int
    last_presence_ns: int
    root: np.ndarray
    root_velocity: np.ndarray
    local: np.ndarray
    local_velocity: np.ndarray
    confidence: np.ndarray
    ever_valid: np.ndarray
    last_measurement_ns: np.ndarray
    uncertainty_m: np.ndarray
    joint_outlier_count: np.ndarray
    root_outlier_count: int = 0
    lateral: np.ndarray | None = None
    torso_height: float | None = None
    bone_lengths: dict[tuple[int, int], float] = field(default_factory=dict)
    bone_samples: dict[tuple[int, int], deque[float]] = field(default_factory=dict)


def _unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8 or not np.isfinite(norm):
        return fallback.copy()
    return value / norm


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Huber-like robust centre for a handful of inferred root candidates."""

    if len(values) == 1:
        return values[0].copy()
    centre = np.median(values, axis=0)
    residual = np.linalg.norm(values - centre, axis=1)
    scale = max(0.05, float(np.median(residual)) * 1.4826)
    robust = np.minimum(1.0, (2.5 * scale) / np.maximum(residual, 1e-6))
    combined = np.maximum(weights, 0.0) * robust
    total = float(combined.sum())
    if total <= 1e-8:
        return centre
    return (values * combined[:, None]).sum(axis=0) / total


class CausalSkeletonOptimizer:
    """Per-track, causal, low-latency 3-D skeleton refinement."""

    def __init__(
        self, *, max_hold_s: float = 0.35, max_track_gap_s: float = 0.5,
        constraint_iterations: int = 5,
    ) -> None:
        self.max_hold_ns = int(max_hold_s * 1e9)
        self.max_track_gap_ns = int(max_track_gap_s * 1e9)
        self.constraint_iterations = max(1, int(constraint_iterations))
        self.states: dict[int, _PersonState] = {}

    def reset(self) -> None:
        self.states.clear()

    @staticmethod
    def _decode(rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.float64)
        xyz = rows[:, :3].copy()
        coordinate_valid = rows[:, FIELD["coordinate_valid"]].astype(bool)
        finite = np.isfinite(xyz).all(axis=1)
        valid = coordinate_valid & finite
        measured = rows[:, FIELD["measured"]].astype(bool) & valid
        predicted = rows[:, FIELD["predicted"]].astype(bool) & valid
        confidence = np.clip(rows[:, FIELD["confidence"]], 0.0, 1.0)
        sigma = np.clip(rows[:, FIELD["measurement_sigma_m"]], 0.0, 5.0)
        age_ms = np.clip(rows[:, FIELD["measurement_age_ms"]], 0.0, 10_000.0)
        source = rows[:, FIELD["source_code"]].astype(np.int64)
        source_weight = np.asarray(
            [SOURCE_WEIGHT.get(int(code), 0.25) for code in source], dtype=np.float64
        )
        sigma_weight = 1.0 / (1.0 + np.square(sigma / 0.15))
        age_weight = np.exp(-age_ms / 350.0)
        quality = confidence * source_weight * sigma_weight * age_weight
        quality[~valid] = 0.0
        quality[predicted | ~measured] = 0.0
        xyz[~valid] = 0.0
        return {
            "xyz": xyz, "valid": valid, "measured": measured,
            "predicted": predicted, "confidence": confidence,
            "sigma": sigma, "quality": np.clip(quality, 0.0, 1.0),
        }

    @staticmethod
    def _initial_root(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if valid[11] and valid[12]:
            return 0.5 * (xyz[11] + xyz[12])
        torso = TORSO_JOINTS[valid[TORSO_JOINTS]]
        if len(torso):
            return np.median(xyz[torso], axis=0)
        body = BODY_JOINTS[valid[BODY_JOINTS]]
        return np.median(xyz[body], axis=0)

    @staticmethod
    def _view_direction(root: np.ndarray) -> np.ndarray:
        return _unit(root, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))

    @staticmethod
    def _blend_anisotropic(
        prediction: np.ndarray, measurement: np.ndarray, direction: np.ndarray,
        radial_alpha: float, tangent_alpha: float,
    ) -> np.ndarray:
        residual = measurement - prediction
        radial = direction * float(np.dot(residual, direction))
        tangent = residual - radial
        return prediction + radial_alpha * radial + tangent_alpha * tangent

    @staticmethod
    def _bone_length(local: np.ndarray, bone: tuple[int, int]) -> float:
        return float(np.linalg.norm(local[bone[0]] - local[bone[1]]))

    def _initialize_bones(
        self, state: _PersonState, valid: np.ndarray,
    ) -> None:
        for bone in BODY_BONES:
            if not (valid[bone[0]] and valid[bone[1]]):
                continue
            length = self._bone_length(state.local, bone)
            low, high = LENGTH_RANGES[bone]
            if low <= length <= high:
                state.bone_lengths[bone] = length
                state.bone_samples[bone] = deque([length], maxlen=31)
        self._share_symmetric_lengths(state)

    @staticmethod
    def _share_symmetric_lengths(state: _PersonState) -> None:
        for group in SYMMETRIC_BONE_GROUPS:
            present = [state.bone_lengths[bone] for bone in group
                       if bone in state.bone_lengths]
            if not present:
                continue
            target = float(np.median(present))
            for bone in group:
                low, high = LENGTH_RANGES[bone]
                state.bone_lengths[bone] = float(np.clip(target, low, high))

    @staticmethod
    def _template_scale(state: _PersonState) -> float:
        if state.torso_height is not None:
            return float(np.clip(state.torso_height / 0.52, 0.65, 1.55))
        ratios = []
        for bone, length in state.bone_lengths.items():
            template_length = float(np.linalg.norm(
                DEFAULT_LOCAL_TEMPLATE[bone[0]] - DEFAULT_LOCAL_TEMPLATE[bone[1]]
            ))
            if template_length > 1e-6:
                ratios.append(length / template_length)
        return float(np.clip(np.median(ratios), 0.65, 1.55)) if ratios else 1.0

    def _aligned_template(
        self, state: _PersonState, local: np.ndarray, anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align the body template to current torso axes and measured anchors."""

        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if anchors[5] and anchors[6] and anchors[11] and anchors[12]:
            torso = 0.5 * (local[5] + local[6]) - 0.5 * (local[11] + local[12])
            up = self._cone_direction(torso, up, 35.0)
        lateral = state.lateral
        if lateral is None:
            lateral = self._candidate_lateral(local, anchors)
        if lateral is None:
            lateral = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        lateral = lateral - up * float(np.dot(lateral, up))
        lateral = _unit(lateral, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        forward = _unit(np.cross(lateral, up), np.asarray([1.0, 0.0, 0.0]))
        scale = self._template_scale(state)
        template = scale * (
            DEFAULT_LOCAL_TEMPLATE[:, 0, None] * forward
            + DEFAULT_LOCAL_TEMPLATE[:, 1, None] * lateral
            + DEFAULT_LOCAL_TEMPLATE[:, 2, None] * up
        )
        alignment_indices = TORSO_JOINTS[anchors[TORSO_JOINTS]]
        if not len(alignment_indices):
            alignment_indices = np.flatnonzero(anchors)
        offset = np.median(
            local[alignment_indices] - template[alignment_indices], axis=0
        )
        return template + offset, lateral, offset

    def _complete_missing_joints(
        self, state: _PersonState, local: np.ndarray, anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kinematically fill joints that lack depth while the person is present.

        ``anchors`` contains recent measured/predicted 3-D joints.  Completion
        requires at least three current or historical anchors. Whole-person
        dropouts call this only inside the adapter's bounded 0.35 s hold, so a
        departed track still expires normally.
        """

        valid = anchors.copy()
        inferred = np.zeros(17, dtype=bool)
        if max(
            int((anchors & BODY_JOINT_MASK).sum()),
            int((state.ever_valid & BODY_JOINT_MASK).sum()),
        ) < 3:
            return local, valid, inferred
        result = local.copy()
        template, lateral, centre = self._aligned_template(state, result, anchors)
        # A joint that was measured earlier keeps its motion-predicted local
        # pose after the short high-confidence hold expires. It is still
        # labelled inferred and its confidence decays; only never-seen joints
        # need a mirror/template position.
        historical = state.ever_valid & ~valid
        valid[historical] = True
        inferred[historical] = True
        for joint in FILL_ORDER:
            if valid[joint]:
                continue
            mirror = MIRROR_JOINT.get(joint)
            if mirror is not None and valid[mirror]:
                relative = result[mirror] - centre
                result[joint] = centre + relative - 2.0 * lateral * float(
                    np.dot(relative, lateral)
                )
            else:
                parent = KINEMATIC_PARENT.get(joint)
                if parent is not None and valid[parent]:
                    result[joint] = result[parent] + template[joint] - template[parent]
                else:
                    result[joint] = template[joint]
            valid[joint] = True
            inferred[joint] = True
        return result, valid, inferred

    def _ensure_completion_bones(
        self, state: _PersonState, local: np.ndarray, valid: np.ndarray,
    ) -> None:
        """Seed missing bone lengths from aligned completion without learning it."""

        for bone in BODY_BONES:
            if bone in state.bone_lengths or not (valid[bone[0]] and valid[bone[1]]):
                continue
            low, high = LENGTH_RANGES[bone]
            length = self._bone_length(local, bone)
            state.bone_lengths[bone] = float(np.clip(length, low, high))
        self._share_symmetric_lengths(state)

    def _update_bones(
        self, state: _PersonState, raw_local: np.ndarray,
        measured: np.ndarray, quality: np.ndarray,
    ) -> None:
        for bone in BODY_BONES:
            first, second = bone
            if not (measured[first] and measured[second]):
                continue
            if min(quality[first], quality[second]) < 0.42:
                continue
            length = self._bone_length(raw_local, bone)
            low, high = LENGTH_RANGES[bone]
            if not low <= length <= high:
                continue
            previous = state.bone_lengths.get(bone)
            if previous is not None and abs(length - previous) > 0.22 * previous:
                continue
            samples = state.bone_samples.setdefault(bone, deque(maxlen=31))
            samples.append(length)
            median = float(np.median(np.asarray(samples)))
            state.bone_lengths[bone] = median if previous is None else (
                0.98 * previous + 0.02 * median
            )
        self._share_symmetric_lengths(state)

    @staticmethod
    def _candidate_lateral(local: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
        vectors = []
        if valid[5] and valid[6]:
            vectors.append(local[5] - local[6])
        if valid[11] and valid[12]:
            vectors.append(local[11] - local[12])
        if not vectors:
            return None
        candidate = np.mean([_unit(v, np.zeros(3)) for v in vectors], axis=0)
        if np.linalg.norm(candidate) < 1e-5:
            return None
        if valid[5] and valid[6] and valid[11] and valid[12]:
            vertical = 0.5 * (local[5] + local[6]) - 0.5 * (local[11] + local[12])
            vertical = _unit(vertical, np.asarray([0.0, 0.0, 1.0]))
            candidate = candidate - vertical * float(np.dot(candidate, vertical))
        return _unit(candidate, np.asarray([0.0, 1.0, 0.0]))

    def _update_lateral(
        self, state: _PersonState, local: np.ndarray, valid: np.ndarray,
        quality: np.ndarray,
    ) -> None:
        candidate = self._candidate_lateral(local, valid)
        if candidate is None:
            return
        if state.lateral is None:
            state.lateral = candidate
            return
        # Left/right labels define the sign. A sudden opposite direction is a
        # depth flip, so keep the previous body orientation instead.
        if float(np.dot(candidate, state.lateral)) <= 0.0:
            return
        pair_quality = float(np.mean(quality[TORSO_JOINTS]))
        alpha = 0.04 + 0.18 * pair_quality
        state.lateral = _unit(
            (1.0 - alpha) * state.lateral + alpha * candidate, state.lateral
        )

    @staticmethod
    def _torso_height(local: np.ndarray, valid: np.ndarray) -> float | None:
        if not (valid[5] and valid[6] and valid[11] and valid[12]):
            return None
        shoulder_centre = 0.5 * (local[5] + local[6])
        hip_centre = 0.5 * (local[11] + local[12])
        height = float(np.linalg.norm(shoulder_centre - hip_centre))
        return height if 0.22 <= height <= 0.85 else None

    def _update_upright_model(
        self, state: _PersonState, local: np.ndarray, valid: np.ndarray,
        quality: np.ndarray,
    ) -> None:
        height = self._torso_height(local, valid)
        if height is None:
            return
        torso_quality = float(np.mean(quality[TORSO_JOINTS]))
        if state.torso_height is None:
            state.torso_height = height
        elif torso_quality >= 0.35 and abs(height - state.torso_height) <= (
            0.22 * state.torso_height
        ):
            state.torso_height = 0.985 * state.torso_height + 0.015 * height

    @staticmethod
    def _enforce_pair(
        local: np.ndarray, first: int, second: int, lateral: np.ndarray,
        width: float, strength: float,
    ) -> None:
        centre = 0.5 * (local[first] + local[second])
        target_first = centre + 0.5 * width * lateral
        target_second = centre - 0.5 * width * lateral
        local[first] += strength * (target_first - local[first])
        local[second] += strength * (target_second - local[second])

    @staticmethod
    def _cone_direction(
        delta: np.ndarray, axis: np.ndarray, maximum_angle_deg: float,
    ) -> np.ndarray:
        """Clamp a segment to a cone while retaining its horizontal heading."""

        direction = _unit(delta, axis)
        axial = float(np.clip(np.dot(direction, axis), -1.0, 1.0))
        minimum_axial = float(np.cos(np.deg2rad(maximum_angle_deg)))
        if axial >= minimum_axial:
            return direction
        tangent = direction - axial * axis
        if np.linalg.norm(tangent) < 1e-8:
            return axis.copy()
        tangent = _unit(tangent, axis)
        return tangent * np.sqrt(max(0.0, 1.0 - minimum_axial ** 2)) + axis * minimum_axial

    def _enforce_upright_walking(
        self, state: _PersonState, local: np.ndarray, valid: np.ndarray,
        quality: np.ndarray,
    ) -> None:
        """Soft standing/walking prior in base_link where +Z is up.

        The prior permits ordinary torso lean and leg swing. It never fixes a
        foot to the floor and never constrains whole-person translation.
        """

        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        down = -up
        if state.torso_height is not None and (
            valid[5] and valid[6] and valid[11] and valid[12]
        ):
            shoulder_centre = 0.5 * (local[5] + local[6])
            hip_centre = 0.5 * (local[11] + local[12])
            torso = shoulder_centre - hip_centre
            direction = self._cone_direction(torso, up, 35.0)
            target = hip_centre + state.torso_height * direction
            # Noisy shoulders move more than trusted shoulders.
            shoulder_quality = float(np.mean(quality[[5, 6]]))
            strength = 0.28 - 0.12 * shoulder_quality
            shift = strength * (target - shoulder_centre)
            local[5] += shift
            local[6] += shift

        # Warehouse pedestrians are upright walkers: thighs and shins point
        # predominantly down, while a 72-degree cone still permits a wide
        # stride and ordinary knee flexion.
        for parent, child in ((11, 13), (12, 14), (13, 15), (14, 16)):
            if not (valid[parent] and valid[child]):
                continue
            target_length = state.bone_lengths.get((parent, child))
            if target_length is None:
                continue
            direction = self._cone_direction(local[child] - local[parent], down, 72.0)
            target = local[parent] + target_length * direction
            strength = 0.24 - 0.10 * float(quality[child])
            local[child] += strength * (target - local[child])

        # Translate the facial cluster together if a gross depth error places
        # the head below the shoulder line. Normal head pitch remains free.
        face = np.asarray((0, 1, 2, 3, 4), dtype=np.int64)
        face_valid = face[valid[face]]
        if len(face_valid) and valid[5] and valid[6]:
            shoulder_z = float(0.5 * (local[5, 2] + local[6, 2]))
            highest_face = float(np.max(local[face_valid, 2]))
            deficit = shoulder_z + 0.08 - highest_face
            if deficit > 0.0:
                local[face_valid, 2] += 0.30 * deficit

    def _project_constraints(
        self, state: _PersonState, local: np.ndarray, valid: np.ndarray,
        quality: np.ndarray,
    ) -> np.ndarray:
        result = local.copy()
        for _ in range(self.constraint_iterations):
            for bone in BODY_BONES:
                target = state.bone_lengths.get(bone)
                first, second = bone
                if target is None or not (valid[first] and valid[second]):
                    continue
                delta = result[second] - result[first]
                length = float(np.linalg.norm(delta))
                if length <= 1e-7:
                    continue
                correction = delta * ((length - target) / length)
                # Low-confidence joints move more than trusted measurements.
                move_first = 1.0 / (0.15 + quality[first])
                move_second = 1.0 / (0.15 + quality[second])
                total = move_first + move_second
                result[first] += correction * (move_first / total)
                result[second] -= correction * (move_second / total)
            if state.lateral is not None:
                shoulder = state.bone_lengths.get((5, 6))
                hip = state.bone_lengths.get((11, 12))
                if shoulder is not None and valid[5] and valid[6]:
                    self._enforce_pair(result, 5, 6, state.lateral, shoulder, 0.30)
                if hip is not None and valid[11] and valid[12]:
                    self._enforce_pair(result, 11, 12, state.lateral, hip, 0.40)
            self._enforce_upright_walking(state, result, valid, quality)
            if valid[11] and valid[12]:
                # Keep the virtual pelvis root at the origin of local pose.
                result -= 0.5 * (result[11] + result[12])
        return result

    def _initialize(
        self, person_id: int, timestamp_ns: int, decoded: dict[str, np.ndarray],
    ) -> _PersonState:
        xyz = decoded["xyz"]
        valid = decoded["valid"]
        root = self._initial_root(xyz, valid)
        local = xyz - root
        local[~valid] = 0.0
        last_measurement = np.full(17, timestamp_ns - self.max_hold_ns - 1, np.int64)
        first_observation = decoded["measured"] | valid
        last_measurement[first_observation] = timestamp_ns
        uncertainty = np.full(17, 2.0, dtype=np.float64)
        uncertainty[valid] = np.maximum(0.04, decoded["sigma"][valid])
        state = _PersonState(
            timestamp_ns=timestamp_ns, last_presence_ns=timestamp_ns,
            root=root.copy(), root_velocity=np.zeros(3, dtype=np.float64),
            local=local.copy(), local_velocity=np.zeros((17, 3), dtype=np.float64),
            confidence=decoded["confidence"].copy(), ever_valid=valid.copy(),
            last_measurement_ns=last_measurement, uncertainty_m=uncertainty,
            joint_outlier_count=np.zeros(17, dtype=np.int16),
        )
        self._initialize_bones(state, valid)
        self._update_lateral(state, local, valid, decoded["quality"])
        self._update_upright_model(state, local, valid, decoded["quality"])
        state.local = self._project_constraints(
            state, state.local, valid, decoded["quality"]
        )
        return state

    def update(
        self, person_id: int, timestamp_ns: int, rows: np.ndarray,
        *, source_present: bool = True,
    ) -> RefinedPerson:
        decoded = self._decode(rows)
        raw_xyz = decoded["xyz"].copy()
        raw_valid = decoded["valid"].copy()
        state = self.states.get(person_id)
        if state is None or timestamp_ns <= state.timestamp_ns or (
            timestamp_ns - state.timestamp_ns > self.max_track_gap_ns
        ):
            state = self._initialize(person_id, timestamp_ns, decoded)
            self.states[person_id] = state
            age_ms = np.maximum(0, timestamp_ns - state.last_measurement_ns) / 1e6
            recent = state.ever_valid & (age_ms <= self.max_hold_ns / 1e6)
            state.local, valid, inferred = self._complete_missing_joints(
                state, state.local, recent,
            )
            completion_inferred = inferred.copy()
            inferred |= valid & ~decoded["measured"]
            if completion_inferred.any():
                self._ensure_completion_bones(state, state.local, valid)
            state.local = self._project_constraints(
                state, state.local, valid, decoded["quality"]
            )
            xyz = state.root + state.local
            xyz[~valid] = 0.0
            velocity = np.zeros_like(xyz)
            confidence = state.confidence.copy()
            support = float(np.median(confidence[recent])) if recent.any() else 0.0
            confidence[inferred] = max(0.03, 0.18 * support)
            confidence[~valid] = 0.0
            uncertainty = state.uncertainty_m.copy()
            uncertainty[inferred] = np.maximum(uncertainty[inferred], 0.75)
            state.confidence = confidence
            state.uncertainty_m = uncertainty
            return RefinedPerson(
                person_id, raw_xyz, raw_valid, decoded["confidence"].copy(),
                xyz, velocity, confidence, valid, inferred,
                uncertainty, age_ms.astype(np.float64),
            )

        dt = (timestamp_ns - state.timestamp_ns) * 1e-9
        previous_root = state.root.copy()
        previous_local = state.local.copy()
        root_prediction = state.root + state.root_velocity * dt
        local_prediction = state.local + state.local_velocity * dt
        direction = self._view_direction(root_prediction)

        measured = decoded["measured"].copy()
        quality = decoded["quality"].copy()
        sigma = decoded["sigma"]

        # Infer a common root measurement from torso joints after subtracting
        # the predicted local articulation. This rejects an isolated bad hip.
        root_indices = TORSO_JOINTS[measured[TORSO_JOINTS] & (quality[TORSO_JOINTS] > 0)]
        if len(root_indices) < 2:
            root_indices = BODY_JOINTS[
                measured[BODY_JOINTS] & (quality[BODY_JOINTS] > 0.25)
            ]
        root = root_prediction.copy()
        if len(root_indices):
            candidates = raw_xyz[root_indices] - local_prediction[root_indices]
            root_measurement = _weighted_center(candidates, quality[root_indices])
            root_quality = float(np.clip(np.mean(quality[root_indices]), 0.0, 1.0))
            root_residual = root_measurement - root_prediction
            root_radial_error = abs(float(np.dot(root_residual, direction)))
            root_tangent_error = float(np.linalg.norm(
                root_residual - direction * float(np.dot(root_residual, direction))
            ))
            root_sigma = float(np.median(sigma[root_indices]))
            root_outlier = (
                root_radial_error > max(0.45, 4.0 * root_sigma)
                or root_tangent_error > max(0.35, 4.0 * root_sigma)
            )
            if root_outlier:
                state.root_outlier_count += 1
                # Ignore two isolated common-depth jumps. A persistent shift
                # is allowed to reacquire after three frames, which prevents
                # locking when the drone/person genuinely moves quickly.
                if state.root_outlier_count < 3:
                    root_quality *= 0.02
                else:
                    root_quality = max(0.35, root_quality * 0.55)
                    state.root_outlier_count = 0
            else:
                state.root_outlier_count = 0
            root = self._blend_anisotropic(
                root_prediction, root_measurement, direction,
                radial_alpha=0.16 + 0.42 * root_quality,
                tangent_alpha=0.28 + 0.62 * root_quality,
            )

        local = local_prediction.copy()
        accepted_measurement = np.zeros(17, dtype=bool)
        raw_local = raw_xyz - root
        for joint in range(17):
            if not measured[joint] or quality[joint] <= 0.0:
                continue
            residual = raw_local[joint] - local_prediction[joint]
            radial_error = abs(float(np.dot(residual, direction)))
            tangent_error = float(np.linalg.norm(
                residual - direction * float(np.dot(residual, direction))
            ))
            radial_gate = max(0.28, 4.0 * float(sigma[joint]))
            # Distal joints move faster during an ordinary walking gesture.
            tangent_floor = 0.42 if joint in (9, 10, 15, 16) else (
                0.30 if joint in (7, 8, 13, 14) else 0.22
            )
            tangent_gate = max(tangent_floor, 4.0 * float(sigma[joint]))
            if radial_error > radial_gate or tangent_error > tangent_gate:
                state.joint_outlier_count[joint] += 1
                # Reject isolated errors, but reacquire a persistent valid
                # change on the third frame so a fast real limb motion cannot
                # leave the optimizer permanently frozen.
                if state.joint_outlier_count[joint] < 3:
                    quality[joint] *= 0.01
                else:
                    quality[joint] = max(0.25, quality[joint] * 0.45)
                    state.joint_outlier_count[joint] = 0
            else:
                state.joint_outlier_count[joint] = 0
            if quality[joint] < 0.02:
                continue
            local[joint] = self._blend_anisotropic(
                local_prediction[joint], raw_local[joint], direction,
                radial_alpha=0.06 + 0.28 * float(quality[joint]),
                tangent_alpha=0.18 + 0.68 * float(quality[joint]),
            )
            accepted_measurement[joint] = True
            state.last_measurement_ns[joint] = timestamp_ns
            state.ever_valid[joint] = True

        age_ms = np.maximum(0, timestamp_ns - state.last_measurement_ns) / 1e6
        recent = state.ever_valid & (age_ms <= self.max_hold_ns / 1e6)
        decayed_confidence = state.confidence * np.exp(-dt / 0.35)
        confidence = decayed_confidence
        confidence[accepted_measurement] = np.maximum(
            0.35 * decayed_confidence[accepted_measurement],
            quality[accepted_measurement],
        )
        self._update_bones(state, raw_local, accepted_measurement, quality)
        self._update_lateral(state, local, recent, quality)
        self._update_upright_model(state, local, recent, quality)
        local, valid, inferred = self._complete_missing_joints(
            state, local, recent,
        )
        completion_inferred = inferred.copy()
        inferred |= valid & ~accepted_measurement
        if completion_inferred.any():
            self._ensure_completion_bones(state, local, valid)
        support = float(np.median(confidence[recent])) if recent.any() else 0.0
        inferred_age_s = np.minimum(age_ms[inferred] / 1000.0, 4.0)
        confidence[inferred] = np.maximum(
            0.025, 0.20 * support * np.exp(-inferred_age_s / 2.0)
        )
        confidence[~valid] = 0.0
        local = self._project_constraints(state, local, valid, quality)

        root_velocity_measurement = (root - previous_root) / dt
        state.root_velocity = 0.45 * root_velocity_measurement + 0.55 * state.root_velocity
        local_velocity_measurement = (local - previous_local) / dt
        local_speed = np.linalg.norm(local_velocity_measurement, axis=1)
        plausible = valid & ~inferred & np.isfinite(local_velocity_measurement).all(axis=1)
        plausible &= local_speed <= 8.0
        state.local_velocity[plausible] = (
            0.40 * local_velocity_measurement[plausible]
            + 0.60 * state.local_velocity[plausible]
        )
        state.local_velocity[inferred | ~valid] *= np.exp(-dt / 0.20)

        uncertainty = state.uncertainty_m.copy()
        uncertainty += dt * 0.30
        measurement_uncertainty = np.clip(
            np.maximum(0.04, sigma) / np.maximum(quality, 0.08), 0.04, 2.0
        )
        uncertainty[accepted_measurement] = (
            0.35 * uncertainty[accepted_measurement]
            + 0.65 * measurement_uncertainty[accepted_measurement]
        )
        uncertainty[inferred] = np.maximum(
            uncertainty[inferred], np.minimum(2.0, 0.65 + age_ms[inferred] / 1000.0)
        )
        uncertainty[~valid] = 2.0

        state.timestamp_ns = timestamp_ns
        if source_present:
            state.last_presence_ns = timestamp_ns
        state.root = root
        state.local = local
        state.confidence = confidence
        state.uncertainty_m = uncertainty
        xyz = root + local
        velocity = state.root_velocity + state.local_velocity
        xyz[~valid] = 0.0
        velocity[~valid] = 0.0
        return RefinedPerson(
            person_id=person_id, raw_xyz=raw_xyz, raw_valid=raw_valid,
            raw_confidence=decoded["confidence"].copy(),
            xyz=xyz, velocity=velocity, confidence=confidence, valid=valid,
            inferred=inferred,
            uncertainty_m=uncertainty, prediction_age_ms=age_ms.astype(np.float64),
        )

    def predict_absent(self, person_id: int, timestamp_ns: int) -> RefinedPerson:
        """Advance one temporarily absent person without inventing a measurement."""

        rows = np.zeros((17, len(JOINT_FIELDS)), dtype=np.float64)
        return self.update(
            person_id, timestamp_ns, rows, source_present=False,
        )

    def prune(self, active_ids: set[int], timestamp_ns: int) -> None:
        stale = [person_id for person_id, state in self.states.items()
                 if person_id not in active_ids
                 and timestamp_ns - state.last_presence_ns > self.max_track_gap_ns]
        for person_id in stale:
            self.states.pop(person_id, None)
