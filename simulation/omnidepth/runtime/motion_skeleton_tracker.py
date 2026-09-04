#!/usr/bin/env python3
"""Motion-aware 3D skeleton identity tracking for the simulation runtime.

The camera/pose/depth front end first produces one fused 3D skeleton per
visible person.  This module assigns persistent positive IDs using predicted
body motion.  Small frames retain the exact dynamic-programming assignment
from the Jetson implementation; frames with more than 12 measurements use a
dependency-free Hungarian assignment so dense simulation crowds never fall
back to greedy matching.
"""

import math
import uuid
from collections import deque
from functools import lru_cache

import numpy as np


CORE_JOINTS = (5, 6, 11, 12)
EXACT_ASSIGNMENT_LIMIT = 12
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
BONE_LENGTH_RANGES = {
    (5, 6): (0.18, 0.70),
    (11, 12): (0.12, 0.60),
    (5, 7): (0.16, 0.55), (6, 8): (0.16, 0.55),
    (7, 9): (0.14, 0.50), (8, 10): (0.14, 0.50),
    (5, 11): (0.25, 0.85), (6, 12): (0.25, 0.85),
    (11, 13): (0.22, 0.75), (12, 14): (0.22, 0.75),
    (13, 15): (0.20, 0.75), (14, 16): (0.20, 0.75),
}
KALMAN_NIS_GATE = 25.0
KALMAN_ABSOLUTE_INNOVATION_GATE_M = 0.55
MIN_USABLE_BODY_JOINTS = 4
MIN_USABLE_CORE_JOINTS = 2
LOW_GEOMETRY_CONFIRMATION_HITS = 4
SYMMETRIC_JOINT_PAIRS = (
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
)
SYMMETRIC_SWAP_MARGIN_M = 0.08


def _person_center(person):
    preferred = person.get("range_gate_center_m")
    if preferred is not None:
        value = np.asarray(preferred, dtype=np.float64).reshape(-1)
        if value.size == 3 and np.isfinite(value).all():
            return value
    joints = person.get("joints", [])
    points = []
    for index in CORE_JOINTS:
        if index < len(joints):
            value = joints[index].get("xyz_imu_m")
            if value is not None:
                points.append(value)
    if len(points) < 2:
        points = [joint.get("xyz_imu_m") for joint in joints
                  if joint.get("xyz_imu_m") is not None]
    if not points:
        return None
    return np.median(np.asarray(points, dtype=np.float64), axis=0)


def _usable_joint_ids(person):
    """Return finite measured joints that can support an articulated track."""
    result = set()
    for joint in person.get("joints", []):
        xyz = joint.get("xyz_imu_m")
        if xyz is None:
            continue
        value = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if value.shape == (3,) and np.isfinite(value).all():
            result.add(int(joint["id"]))
    return result


def _has_usable_body(joint_ids):
    """Require an articulated body, never a face-only or two-point output."""
    joint_ids = set(joint_ids)
    body_count = sum(index >= 5 for index in joint_ids)
    core_count = sum(index in joint_ids for index in CORE_JOINTS)
    return (body_count >= MIN_USABLE_BODY_JOINTS and
            core_count >= MIN_USABLE_CORE_JOINTS)


def _limit(vector, maximum):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > maximum > 0.0:
        value = value * (maximum / norm)
    return value


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return None if norm <= 1e-6 else value / norm


def _bearing_angle(first, second):
    first_unit, second_unit = _unit(first), _unit(second)
    if first_unit is None or second_unit is None:
        return math.pi
    return math.acos(float(np.clip(np.dot(first_unit, second_unit), -1, 1)))


def _rotation_from_xyzw(quaternion):
    """Return R_world_from_body without depending on SciPy in the runtime."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("ego attitude must contain four finite xyzw values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-9:
        raise ValueError("ego attitude quaternion has zero norm")
    x, y, z, w = value / norm
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


def _decayed_step(position, velocity, dt, decay):
    """Advance without reversing while velocity exponentially approaches 0."""
    if dt <= 0.0:
        return position, velocity
    factor = math.exp(-decay * dt)
    travel = dt if decay <= 1e-9 else (1.0 - factor) / decay
    return position + velocity * travel, velocity * factor


def _robust_history_velocity(history, max_speed):
    """Return causal median-slope velocity and a residual uncertainty."""
    values = list(history)
    if len(values) < 3 or values[-1][0] - values[0][0] < 0.20:
        return np.zeros(3, np.float64), float("inf"), False
    slopes = []
    for first in range(len(values) - 1):
        for second in range(first + 1, len(values)):
            dt = float(values[second][0] - values[first][0])
            if dt >= 0.08:
                slopes.append((values[second][1] - values[first][1]) / dt)
    if len(slopes) < 3:
        return np.zeros(3, np.float64), float("inf"), False
    velocity = np.median(np.asarray(slopes, np.float64), axis=0)
    velocity = _limit(velocity, float(max_speed))
    last_time, last_position = values[-1]
    residuals = [
        np.linalg.norm(
            position - (last_position - velocity * (last_time - stamp)))
        for stamp, position in values
    ]
    sigma = float(np.median(np.asarray(residuals, np.float64)))
    valid = bool(np.isfinite(velocity).all() and math.isfinite(sigma))
    return velocity, sigma, valid


def _hungarian_rows(costs):
    """Assign every row of a finite matrix with rows <= columns."""
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] > values.shape[1]:
        raise ValueError("Hungarian solver requires rows <= columns")
    rows, columns = values.shape
    if rows == 0:
        return []
    # Shortest augmenting path formulation. Arrays use the conventional
    # one-based indexing from the Hungarian algorithm description.
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    column_row = np.zeros(columns + 1, dtype=np.int64)
    previous_column = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        column_row[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=np.bool_)
        column = 0
        while True:
            used[column] = True
            active_row = int(column_row[column])
            delta = np.inf
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (values[active_row - 1, candidate - 1] -
                           u[active_row] - v[candidate])
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            if not np.isfinite(delta):
                raise ValueError("Hungarian cost matrix has no finite assignment")
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[column_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while True:
            prior = int(previous_column[column])
            column_row[column] = column_row[prior]
            column = prior
            if column == 0:
                break
    return [(int(column_row[column] - 1), column - 1)
            for column in range(1, columns + 1)
            if column_row[column] != 0]


def _hungarian_assignment(costs):
    """Maximum-cardinality, minimum-cost rectangular assignment.

    Forbidden pairs are represented by infinity. Dummy columns give every real
    track an explicit unmatched option; unused real columns are unmatched
    measurements. The penalty makes cardinality dominate total real cost.
    """
    costs = np.asarray(costs, dtype=np.float64)
    tracks, measurements = costs.shape
    if not tracks or not measurements:
        return []
    finite = np.isfinite(costs)
    if not finite.any():
        return []
    size = tracks + measurements
    maximum = max(1.0, float(np.max(np.abs(costs[finite]))))
    unmatched = (maximum + 1.0) * (size + 1.0)
    forbidden = unmatched * (tracks + 2.0)
    # Each track receives either one real measurement or one of the dummy
    # unmatched columns. Measurements need no dummy rows: leaving a real
    # column unused already means that measurement was unmatched.
    augmented = np.full(
        (tracks, measurements + tracks), unmatched, dtype=np.float64)
    augmented[:, :measurements] = np.where(
        finite, costs, forbidden)
    assignment = _hungarian_rows(augmented)
    return [(row, col) for row, col in assignment
            if row < tracks and col < measurements and finite[row, col]]


def _optimal_assignment(costs):
    """Choose the exact small-frame or O(N^3) dense-frame assignment."""
    costs = np.asarray(costs, dtype=np.float64)
    tracks, measurements = costs.shape
    if not tracks or not measurements:
        return []
    if measurements > EXACT_ASSIGNMENT_LIMIT:
        return _hungarian_assignment(costs)

    @lru_cache(maxsize=None)
    def solve(row, used_mask):
        if row == tracks:
            return 0, 0.0, ()
        best = solve(row + 1, used_mask)
        for col in range(measurements):
            if used_mask & (1 << col) or not np.isfinite(costs[row, col]):
                continue
            count, total, pairs = solve(row + 1, used_mask | (1 << col))
            candidate = (count + 1, total + float(costs[row, col]),
                         ((row, col),) + pairs)
            if candidate[0] > best[0] or (candidate[0] == best[0] and
                                         candidate[1] < best[1]):
                best = candidate
        return best

    return list(solve(0, 0)[2])


def _joint_measurement_sigma(joint):
    """Return a conservative metric uncertainty for one 3-D keypoint."""
    try:
        sigma = float(joint.get("measurement_sigma_m") or 0.20)
    except (TypeError, ValueError):
        sigma = 0.20
    source = str(joint.get("source") or "unknown")
    # Isaac supplies exact depth at the selected pixel, not an exact semantic
    # joint location. Keep a floor so two-pixel RTMPose jitter is not mistaken
    # for centimetre-accurate body motion.
    floor = 0.035 if source.startswith("isaac_gt_depth") else 0.05
    if source.startswith("detection_only_"):
        floor = 0.20
    return float(np.clip(sigma, floor, 1.0))


class _JointState:
    """Constant-velocity Kalman state for one joint in the current body frame."""

    def __init__(self, joint, stamp):
        position = np.asarray(joint["xyz_imu_m"], dtype=np.float64)
        sigma = _joint_measurement_sigma(joint)
        self.state = np.r_[position, np.zeros(3, dtype=np.float64)]
        self.covariance = np.diag(
            [sigma * sigma] * 3 + [0.8 * 0.8] * 3).astype(np.float64)
        self.last_stamp = stamp
        self.last_observed = stamp
        self.template = dict(joint)
        self.last_score = float(joint.get("score", 0.0) or 0.0)
        self.last_sigma = sigma
        self.last_source = str(joint.get("source") or "unknown")
        self.rejected_measurements = 0

    @property
    def position(self):
        return self.state[:3]

    @property
    def velocity(self):
        return self.state[3:]

    def reexpress(self, rotation, translation):
        """Transform both mean and covariance into a new base_link frame."""
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.state[:3] = rotation.dot(self.state[:3]) + translation
        self.state[3:] = rotation.dot(self.state[3:])
        transform = np.zeros((6, 6), dtype=np.float64)
        transform[:3, :3] = rotation
        transform[3:, 3:] = rotation
        self.covariance = transform.dot(self.covariance).dot(transform.T)

    def predict(self, stamp, missing):
        dt = max(0.0, min(0.5, stamp - self.last_stamp))
        if dt <= 0.0:
            return
        decay = 1.35 if missing else 0.18
        velocity_factor = math.exp(-decay * dt)
        travel = ((1.0 - velocity_factor) / decay
                  if decay > 1e-9 else dt)
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3) * travel
        transition[3:, 3:] = np.eye(3) * velocity_factor
        # Allow ordinary limb acceleration without allowing one depth alias to
        # drag the state metres away. Missing joints receive slightly more
        # process uncertainty as their prediction ages.
        acceleration_sigma = 3.0 if missing else 2.2
        q_position = 0.25 * dt ** 4 * acceleration_sigma ** 2
        q_cross = 0.5 * dt ** 3 * acceleration_sigma ** 2
        q_velocity = dt ** 2 * acceleration_sigma ** 2
        process = np.zeros((6, 6), dtype=np.float64)
        process[:3, :3] = np.eye(3) * max(1e-6, q_position)
        process[:3, 3:] = np.eye(3) * q_cross
        process[3:, :3] = np.eye(3) * q_cross
        process[3:, 3:] = np.eye(3) * max(1e-5, q_velocity)
        self.state = transition.dot(self.state)
        self.covariance = transition.dot(
            self.covariance).dot(transition.T) + process
        self.last_stamp = stamp

    def can_observe(self, joint):
        """Check a measurement gate without mutating the Kalman state."""
        measurement = np.asarray(joint["xyz_imu_m"], dtype=np.float64)
        sigma = _joint_measurement_sigma(joint)
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3)
        innovation = measurement - observation.dot(self.state)
        noise = np.eye(3, dtype=np.float64) * sigma * sigma
        residual_covariance = observation.dot(self.covariance).dot(
            observation.T) + noise
        try:
            solved = np.linalg.solve(residual_covariance, innovation)
        except np.linalg.LinAlgError:
            return False
        nis = float(innovation.dot(solved))
        return (float(np.linalg.norm(innovation)) <=
                KALMAN_ABSOLUTE_INNOVATION_GATE_M and
                math.isfinite(nis) and nis <= KALMAN_NIS_GATE)

    def observe(self, joint, stamp):
        measurement = np.asarray(joint["xyz_imu_m"], dtype=np.float64)
        sigma = _joint_measurement_sigma(joint)
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3)
        innovation = measurement - observation.dot(self.state)
        noise = np.eye(3, dtype=np.float64) * sigma * sigma
        residual_covariance = observation.dot(self.covariance).dot(
            observation.T) + noise
        try:
            solved = np.linalg.solve(residual_covariance, innovation)
        except np.linalg.LinAlgError:
            self.rejected_measurements += 1
            return False
        nis = float(innovation.dot(solved))
        if (float(np.linalg.norm(innovation)) >
                KALMAN_ABSOLUTE_INNOVATION_GATE_M or
                not math.isfinite(nis) or nis > KALMAN_NIS_GATE):
            self.rejected_measurements += 1
            return False
        gain = self.covariance.dot(observation.T).dot(
            np.linalg.inv(residual_covariance))
        self.state += gain.dot(innovation)
        self.state[3:] = _limit(self.state[3:], 2.5)
        identity = np.eye(6, dtype=np.float64)
        remainder = identity - gain.dot(observation)
        # Joseph form keeps P symmetric positive semi-definite under float
        # roundoff during long online runs.
        self.covariance = (remainder.dot(self.covariance).dot(remainder.T) +
                           gain.dot(noise).dot(gain.T))
        self.last_observed = stamp
        self.last_stamp = stamp
        self.template = dict(joint)
        self.last_score = float(joint.get("score", 0.0) or 0.0)
        self.last_sigma = sigma
        self.last_source = str(joint.get("source") or "unknown")
        return True

    def output(self, stamp, predicted, position=None, kinematic_refined=False):
        value = dict(self.template)
        filtered_position = self.position if position is None else position
        xyz = np.asarray(filtered_position, dtype=np.float64).round(6).tolist()
        value["xyz_imu_m"] = xyz
        value["xyz_base_link_m"] = xyz
        value["predicted"] = bool(predicted)
        value["kinematic_refined"] = bool(kinematic_refined)
        value["filtered_sigma_m"] = round(float(np.sqrt(max(
            0.0, np.trace(self.covariance[:3, :3]) / 3.0))), 6)
        value["kalman_rejected_measurements"] = int(
            self.rejected_measurements)
        value["measurement_age_ms"] = round(
            max(0.0, stamp - self.last_observed) * 1000.0, 3)
        if predicted:
            value["source"] = "motion_prediction"
            value["measurement_source"] = self.last_source
            value["last_measurement_score"] = self.last_score
            value["last_measurement_sigma_m"] = self.last_sigma
            value["score"] = round(
                self.last_score *
                math.exp(-(stamp - self.last_observed) / 0.7), 6)
        return value


class MotionSkeletonTracker:
    """Assign IDs after 3D fusion and coast occluded people to a slow stop."""

    def __init__(self, joint_names, confirmation_hits=3,
                 prediction_timeout=1.2, deletion_timeout=4.0,
                 base_gate=0.70, max_speed=2.0, max_person_range=0.0,
                 enable_ray_recovery=True):
        self.joint_names = tuple(joint_names)
        self.confirmation_hits = int(confirmation_hits)
        self.prediction_timeout = float(prediction_timeout)
        self.deletion_timeout = float(deletion_timeout)
        self.base_gate = float(base_gate)
        self.max_speed = float(max_speed)
        self.max_person_range = max(0.0, float(max_person_range))
        self.enable_ray_recovery = bool(enable_ray_recovery)
        self.reset_session()

    def reset_session(self):
        """Discard temporal identity state at an episode boundary."""
        self.session_id = uuid.uuid4().hex
        self.tracks = {}
        self.next_track_key = 0
        # Preserve the simulation dataset's positive-ID convention.
        self.next_id = 1
        self.ever_confirmed = False
        self.last_new_measurements_suppressed = 0
        self.last_assignment_method = "none"
        self._last_ego_position_world = None
        self._last_ego_rotation_world_from_body = None
        self.ego_motion_updates = 0
        self.last_ego_translation_m = 0.0
        self.last_ego_rotation_deg = 0.0
        self.kalman_measurements_rejected = 0
        self.implausible_people_rejected = 0
        self.detection_only_measurements = 0
        self.sparse_measurements_rejected = 0
        self.association_measurements_rejected = 0
        self.coherent_relocalizations = 0
        self.symmetric_joint_pair_swaps = 0
        self.sparse_outputs_suppressed = 0
        return self.session_id

    def _compensate_ego_motion(self, ego_pose):
        """Re-express all tracks from the previous into the current body frame.

        Measurements and public outputs stay in ``base_link``. Internally this
        exact rigid transform removes the apparent motion shared by every
        person when the UAV translates or rotates between rendered frames.
        """
        if ego_pose is None:
            return False
        position = np.asarray(
            ego_pose.get("position_world_m"), dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("ego position must contain three finite values")
        rotation = _rotation_from_xyzw(ego_pose.get("attitude_xyzw"))
        previous_position = self._last_ego_position_world
        previous_rotation = self._last_ego_rotation_world_from_body
        self._last_ego_position_world = position.copy()
        self._last_ego_rotation_world_from_body = rotation.copy()
        if previous_position is None or previous_rotation is None:
            return False

        rotation_current_from_previous = rotation.T.dot(previous_rotation)
        translation_current = rotation.T.dot(previous_position - position)

        def point(value):
            return rotation_current_from_previous.dot(value) + translation_current

        def vector(value):
            return rotation_current_from_previous.dot(value)

        for track in self.tracks.values():
            track["center"] = point(track["center"])
            track["observed_center"] = point(track["observed_center"])
            track["velocity"] = vector(track["velocity"])
            track["velocity_history"] = deque((
                (stamp, point(history_position))
                for stamp, history_position in track["velocity_history"]
            ), maxlen=track["velocity_history"].maxlen)
            for joint in track["joints"].values():
                joint.reexpress(
                    rotation_current_from_previous, translation_current)
        self.ego_motion_updates += 1
        self.last_ego_translation_m = float(np.linalg.norm(
            previous_position - position))
        cosine = float(np.clip(
            (np.trace(rotation_current_from_previous) - 1.0) * 0.5,
            -1.0, 1.0))
        self.last_ego_rotation_deg = math.degrees(math.acos(cosine))
        return True

    @staticmethod
    def _body_geometry_score(person):
        """Return a conservative whole-body plausibility score or ``None``.

        Partial poses are deliberately left undecided. A fully measured shelf
        or ceiling pattern must not become a Human slot merely because several
        isolated keypoints each passed their local depth checks.
        """
        positions = {}
        for joint in person.get("joints", []):
            xyz = joint.get("xyz_imu_m")
            if xyz is None:
                continue
            value = np.asarray(xyz, dtype=np.float64).reshape(-1)
            if value.shape == (3,) and np.isfinite(value).all():
                positions[int(joint["id"])] = value
        evaluated = 0
        plausible = 0
        for bone in BODY_BONES:
            if bone[0] not in positions or bone[1] not in positions:
                continue
            evaluated += 1
            length = float(np.linalg.norm(
                positions[bone[0]] - positions[bone[1]]))
            low, high = BONE_LENGTH_RANGES[bone]
            plausible += int(low <= length <= high)
        if evaluated < 6:
            return None
        score = plausible / float(evaluated)
        if all(index in positions for index in CORE_JOINTS):
            shoulders = 0.5 * (positions[5] + positions[6])
            hips = 0.5 * (positions[11] + positions[12])
            torso_height = float(shoulders[2] - hips[2])
            if torso_height < 0.08 or torso_height > 1.05:
                score *= 0.35
        return float(score)

    @staticmethod
    def _share_symmetric_bones(track):
        for group in SYMMETRIC_BONE_GROUPS:
            values = [track["bone_lengths"][bone] for bone in group
                      if bone in track["bone_lengths"]]
            if not values:
                continue
            target = float(np.median(values))
            for bone in group:
                low, high = BONE_LENGTH_RANGES[bone]
                track["bone_lengths"][bone] = float(
                    np.clip(target, low, high))

    def _update_bone_model(self, track, measured_joint_ids):
        """Slowly learn person-specific lengths from trusted measurements."""
        if bool(track["template"].get("detection_only_obstacle", False)):
            return
        for bone in BODY_BONES:
            if not all(index in measured_joint_ids for index in bone):
                continue
            first, second = (track["joints"].get(index) for index in bone)
            if first is None or second is None:
                continue
            if min(first.last_score, second.last_score) < 0.42:
                continue
            if max(first.last_sigma, second.last_sigma) > 0.30:
                continue
            length = float(np.linalg.norm(first.position - second.position))
            low, high = BONE_LENGTH_RANGES[bone]
            if not low <= length <= high:
                continue
            previous = track["bone_lengths"].get(bone)
            if previous is not None and abs(length - previous) > 0.22 * previous:
                continue
            samples = track["bone_samples"].setdefault(
                bone, deque(maxlen=31))
            samples.append(length)
            median = float(np.median(np.asarray(samples, dtype=np.float64)))
            track["bone_lengths"][bone] = (
                median if previous is None else
                0.98 * previous + 0.02 * median)
        self._share_symmetric_bones(track)

    @staticmethod
    def _project_bone_constraints(track, positions):
        """Project output joints onto learned lengths without moving the root."""
        if len(track["bone_lengths"]) < 4:
            return positions, set()
        result = {joint_id: value.copy()
                  for joint_id, value in positions.items()}
        original_core = [result[index] for index in CORE_JOINTS
                         if index in result]
        refined = set()
        for _ in range(4):
            for bone in BODY_BONES:
                target = track["bone_lengths"].get(bone)
                first_id, second_id = bone
                if (target is None or first_id not in result or
                        second_id not in result):
                    continue
                delta = result[second_id] - result[first_id]
                length = float(np.linalg.norm(delta))
                if length <= 1e-7:
                    continue
                correction = delta * ((length - target) / length)
                first_state = track["joints"][first_id]
                second_state = track["joints"][second_id]
                first_trust = float(np.clip(
                    first_state.last_score /
                    (1.0 + first_state.last_sigma), 0.05, 1.0))
                second_trust = float(np.clip(
                    second_state.last_score /
                    (1.0 + second_state.last_sigma), 0.05, 1.0))
                move_first = 1.05 - first_trust
                move_second = 1.05 - second_trust
                total = move_first + move_second
                result[first_id] += correction * (move_first / total)
                result[second_id] -= correction * (move_second / total)
                refined.update(bone)
        projected_core = [result[index] for index in CORE_JOINTS
                          if index in result]
        if original_core and len(projected_core) == len(original_core):
            shift = (np.median(np.asarray(original_core), axis=0) -
                     np.median(np.asarray(projected_core), axis=0))
            for joint_id in result:
                result[joint_id] += shift
        return result, refined

    def _new_track(self, person, center, stamp, required_hits):
        track_key = self.next_track_key
        self.next_track_key += 1
        joints = {}
        for joint in person["joints"]:
            if joint.get("xyz_imu_m") is not None:
                joints[int(joint["id"])] = _JointState(joint, stamp)
        person_id = None
        if required_hits <= 1:
            person_id = self.next_id
            self.next_id += 1
            self.ever_confirmed = True
        self.tracks[track_key] = {
            "track_key": track_key,
            "person_id": person_id,
            "center": center.copy(),
            "observed_center": center.copy(),
            "velocity": np.zeros(3, dtype=np.float64),
            "velocity_history": deque(((stamp, center.copy()),), maxlen=15),
            "velocity_valid": False,
            "velocity_sigma_mps": float("inf"),
            "prediction_run_frames": 0,
            "last_stamp": stamp,
            "last_observed": stamp,
            "hits": 1,
            "consecutive_hits": 1,
            "required_hits": int(required_hits),
            "age": 1,
            "joints": joints,
            "template": dict(person),
            "bone_lengths": {},
            "bone_samples": {},
            "detection_only_hits": int(bool(
                person.get("detection_only_obstacle", False))),
            "pose_hits": int(not bool(
                person.get("detection_only_obstacle", False))),
        }
        self._update_bone_model(
            self.tracks[track_key], set(joints))

    def _predict_track(self, track, stamp):
        missing = stamp - track["last_observed"] > 0.15
        dt = max(0.0, min(0.5, stamp - track["last_stamp"]))
        decay = 1.35 if missing else 0.18
        track["center"], track["velocity"] = _decayed_step(
            track["center"], track["velocity"], dt, decay)
        track["last_stamp"] = stamp
        track["age"] += 1
        if missing:
            track["prediction_run_frames"] += 1
        for joint in track["joints"].values():
            joint.predict(stamp, missing)

    def _cost(self, track, center, stamp):
        delta = center - track["center"]
        distance = float(np.linalg.norm(delta))
        missing = max(0.0, stamp - track["last_observed"])
        speed = float(np.linalg.norm(track["velocity"]))
        gate = self.base_gate + min(0.75, 0.35 * speed + 0.35 * missing)
        if track["person_id"] is None:
            gate = max(gate, 1.35)
        angle = _bearing_angle(track["center"], center)
        angular_gate = math.radians(22.0 + 5.0 * min(1.5, missing))
        ray_continuity = (self.enable_ray_recovery and
                          track["person_id"] is not None and
                          angle <= angular_gate)
        dormant_recovery = (track["person_id"] is not None and
                            0.35 <= missing <= self.deletion_timeout)
        if distance > gate and not ray_continuity and not dormant_recovery:
            return float("inf")

        direction_penalty = 0.0
        displacement = center - track["observed_center"]
        displacement_norm = float(np.linalg.norm(displacement))
        if speed > 0.12 and displacement_norm > 0.06:
            cosine = float(np.dot(track["velocity"], displacement) /
                           (speed * displacement_norm))
            direction_penalty = 0.45 * max(0.0, 1.0 - cosine)
        if distance <= gate:
            spatial_cost = distance
        elif ray_continuity:
            radial_delta = abs(float(np.linalg.norm(center)) -
                               float(np.linalg.norm(track["center"])))
            spatial_cost = 0.48 + 2.2 * angle + 0.08 * min(3.0, radial_delta)
        else:
            radial_delta = abs(float(np.linalg.norm(center)) -
                               float(np.linalg.norm(track["center"])))
            spatial_cost = 1.10 + 0.75 * angle + 0.06 * min(4.0, radial_delta)
        provisional_penalty = 4.0 if track["person_id"] is None else 0.0
        return (spatial_cost + direction_penalty + 0.03 * missing +
                provisional_penalty)

    @staticmethod
    def _coherent_core_shift(track, person):
        """Return whether torso joints describe one coherent rigid shift."""
        measured = {
            int(joint["id"]): np.asarray(joint["xyz_imu_m"], np.float64)
            for joint in person.get("joints", [])
            if joint.get("xyz_imu_m") is not None
        }
        deltas = []
        for joint_id in CORE_JOINTS:
            state = track["joints"].get(joint_id)
            if state is not None and joint_id in measured:
                deltas.append(measured[joint_id] - state.position)
        if len(deltas) < MIN_USABLE_CORE_JOINTS:
            return False
        deltas = np.asarray(deltas, dtype=np.float64)
        median = np.median(deltas, axis=0)
        dispersion = float(np.median(np.linalg.norm(
            deltas - median[None], axis=1)))
        return dispersion <= 0.22

    @staticmethod
    def _temporally_align_symmetric_joints(track, person):
        """Correct confident left/right label flips during a body turn.

        Top-down pose occasionally exchanges a symmetric pair when the two
        limbs overlap in a side/back view.  Compare direct and crossed temporal
        assignment for each complete pair and only swap when the crossed
        assignment wins by a clear metric margin.  Missing/occluded pairs are
        left untouched.
        """
        original = list(person.get("joints", []))
        by_id = {int(joint["id"]): dict(joint) for joint in original}
        swap_count = 0
        for left_id, right_id in SYMMETRIC_JOINT_PAIRS:
            left = by_id.get(left_id)
            right = by_id.get(right_id)
            left_state = track["joints"].get(left_id)
            right_state = track["joints"].get(right_id)
            if left is None or right is None or left_state is None or right_state is None:
                continue
            left_xyz = left.get("xyz_imu_m")
            right_xyz = right.get("xyz_imu_m")
            if left_xyz is None or right_xyz is None:
                continue
            left_xyz = np.asarray(left_xyz, dtype=np.float64)
            right_xyz = np.asarray(right_xyz, dtype=np.float64)
            direct = (float(np.linalg.norm(left_xyz - left_state.position)) +
                      float(np.linalg.norm(right_xyz - right_state.position)))
            crossed = (float(np.linalg.norm(left_xyz - right_state.position)) +
                       float(np.linalg.norm(right_xyz - left_state.position)))
            if crossed + SYMMETRIC_SWAP_MARGIN_M >= direct:
                continue
            remapped_left = dict(right)
            remapped_right = dict(left)
            remapped_left.update({
                "id": left_id,
                "name": left_state.template.get("name", left.get("name")),
                "raw_joint_id": right_id,
                "temporal_symmetric_swap": True,
            })
            remapped_right.update({
                "id": right_id,
                "name": right_state.template.get("name", right.get("name")),
                "raw_joint_id": left_id,
                "temporal_symmetric_swap": True,
            })
            by_id[left_id], by_id[right_id] = remapped_left, remapped_right
            swap_count += 1
        if not swap_count:
            return person, 0
        aligned = dict(person)
        aligned["joints"] = [by_id.get(int(joint["id"]), joint)
                             for joint in original]
        aligned["temporal_symmetric_pair_swaps"] = swap_count
        return aligned, swap_count

    def _observe_track(self, track, person, center, stamp):
        person, swap_count = self._temporally_align_symmetric_joints(
            track, person)
        self.symmetric_joint_pair_swaps += swap_count
        measured_joints = {
            int(joint["id"]): joint for joint in person["joints"]
            if joint.get("xyz_imu_m") is not None
        }
        preaccepted = set()
        for joint_id, joint in measured_joints.items():
            state = track["joints"].get(joint_id)
            if state is None or state.can_observe(joint):
                preaccepted.add(joint_id)
        prerejected = set(measured_joints) - preaccepted

        relocalize = False
        if not _has_usable_body(preaccepted):
            geometry_score = person.get("body_geometry_score")
            center_distance = float(np.linalg.norm(center - track["center"]))
            relocalize = (
                _has_usable_body(measured_joints) and
                (geometry_score is None or geometry_score >= 0.70) and
                center_distance <= max(1.0, self.base_gate + 0.30) and
                self._coherent_core_shift(track, person)
            )
            if not relocalize:
                self.kalman_measurements_rejected += len(prerejected)
                self.association_measurements_rejected += 1
                return False

        dt = max(1e-3, stamp - track["last_observed"])
        maximum_innovation = (0.40 + 0.18 * min(1.5, dt) +
                              0.10 * float(np.linalg.norm(track["velocity"])))
        if relocalize:
            center = center.copy()
        else:
            center = track["center"] + _limit(
                center - track["center"], maximum_innovation)
        measured_velocity = _limit(
            (center - track["observed_center"]) / dt, self.max_speed * 1.4)
        old_velocity = track["velocity"]
        if (np.linalg.norm(old_velocity) > 0.15 and
                np.dot(measured_velocity, old_velocity) < 0.0 and dt < 0.45):
            direction = old_velocity / np.linalg.norm(old_velocity)
            measured_velocity = measured_velocity - min(
                0.0, float(np.dot(measured_velocity, direction))) * direction
        track["velocity"] = _limit(
            old_velocity * 0.70 + measured_velocity * 0.30, self.max_speed)
        track["center"] = track["center"] * 0.18 + center * 0.82
        track["observed_center"] = center.copy()
        history = track["velocity_history"]
        history.append((stamp, center.copy()))
        while history and stamp - history[0][0] > 1.0:
            history.popleft()
        robust_velocity, velocity_sigma, velocity_valid = (
            _robust_history_velocity(history, self.max_speed))
        if velocity_valid:
            track["velocity"] = _limit(
                0.25 * track["velocity"] + 0.75 * robust_velocity,
                self.max_speed)
        track["velocity_valid"] = bool(velocity_valid)
        track["velocity_sigma_mps"] = float(velocity_sigma)
        track["prediction_run_frames"] = 0
        track["last_observed"] = stamp
        track["last_stamp"] = stamp
        track["hits"] += 1
        track["consecutive_hits"] = (
            track["consecutive_hits"] + 1 if dt <= 0.35 else 1)
        if (track["person_id"] is None and
                track["consecutive_hits"] >= track["required_hits"]):
            track["person_id"] = self.next_id
            self.next_id += 1
            self.ever_confirmed = True
        track["template"] = dict(person)
        detection_only = bool(person.get("detection_only_obstacle", False))
        track["detection_only_hits"] += int(detection_only)
        track["pose_hits"] += int(not detection_only)
        measured_joint_ids = set()
        if relocalize:
            track["joints"] = {
                joint_id: _JointState(joint, stamp)
                for joint_id, joint in measured_joints.items()
            }
            measured_joint_ids.update(measured_joints)
            track["velocity"] = np.zeros(3, dtype=np.float64)
            track["velocity_history"] = deque(
                ((stamp, center.copy()),), maxlen=15)
            track["velocity_valid"] = False
            track["velocity_sigma_mps"] = float("inf")
            self.kalman_measurements_rejected += len(prerejected)
            self.coherent_relocalizations += 1
        else:
            for joint_id, joint in measured_joints.items():
                if joint_id not in track["joints"]:
                    track["joints"][joint_id] = _JointState(joint, stamp)
                    measured_joint_ids.add(joint_id)
                else:
                    accepted = track["joints"][joint_id].observe(joint, stamp)
                    if accepted:
                        measured_joint_ids.add(joint_id)
                    else:
                        self.kalman_measurements_rejected += 1
        self._update_bone_model(track, measured_joint_ids)
        return True

    def _output(self, track, stamp, predicted):
        template = dict(track["template"])
        old_joints = {int(joint["id"]): joint
                      for joint in template.get("joints", [])}
        available_positions = {
            joint_id: state.position.copy()
            for joint_id, state in track["joints"].items()
            if stamp - state.last_observed <= self.prediction_timeout
        }
        projected, refined_ids = self._project_bone_constraints(
            track, available_positions)
        joints = []
        for joint_id, name in enumerate(self.joint_names):
            state = track["joints"].get(joint_id)
            joint_age = (float("inf") if state is None else
                         max(0.0, stamp - state.last_observed))
            if state is not None and joint_age <= self.prediction_timeout:
                # A body can be observed while one occluded joint is not.
                # Preserve that joint briefly as a prediction, but never label
                # its stale state as a fresh measurement.
                joints.append(state.output(
                    stamp, predicted or joint_age > 1e-4,
                    position=projected.get(joint_id),
                    kinematic_refined=joint_id in refined_ids))
            else:
                value = dict(old_joints.get(joint_id, {}))
                value.update({"id": joint_id, "name": name,
                              "xyz_imu_m": None,
                              "xyz_base_link_m": None,
                              "predicted": False,
                              "measurement_age_ms": (
                                  None if state is None else
                                  round(joint_age * 1000.0, 3))})
                joints.append(value)
        missing = max(0.0, stamp - track["last_observed"])
        template["joints"] = joints
        template["range_gate_center_m"] = track["center"].round(6).tolist()
        template["range_m"] = round(float(np.linalg.norm(track["center"])), 4)
        template["person_id"] = int(track["person_id"])
        template["track_uid"] = "{}:{}".format(
            self.session_id, track["person_id"])
        template["track_state"] = "predicted" if predicted else "observed"
        template["predicted_track"] = bool(predicted)
        template["id_uncertain"] = bool(predicted)
        template["identity_confidence"] = round(
            math.exp(-missing / max(0.1, self.prediction_timeout)), 6)
        template["track_age_frames"] = int(track["age"])
        template["time_since_observation_ms"] = round(missing * 1000.0, 3)
        template["root_velocity_base_link_mps"] = (
            np.asarray(track["velocity"], np.float64).round(6).tolist())
        template["velocity_valid"] = bool(track["velocity_valid"])
        template["velocity_sigma_mps"] = (
            round(float(track["velocity_sigma_mps"]), 6)
            if math.isfinite(float(track["velocity_sigma_mps"])) else None)
        template["consecutive_prediction_frames"] = int(
            track["prediction_run_frames"])
        template["bone_constraint_count"] = len(track["bone_lengths"])
        template["detection_only_obstacle"] = bool(
            template.get("detection_only_obstacle", False))
        return template

    def update(self, people, stamp_ns, ego_pose=None):
        stamp = float(stamp_ns) / 1e9
        self.last_new_measurements_suppressed = 0
        self._compensate_ego_motion(ego_pose)
        for track in self.tracks.values():
            self._predict_track(track, stamp)

        measurements = []
        for person in people:
            measured_joint_ids = _usable_joint_ids(person)
            if not _has_usable_body(measured_joint_ids):
                self.sparse_measurements_rejected += 1
                continue
            center = _person_center(person)
            if center is None:
                continue
            geometry_score = self._body_geometry_score(person)
            person["body_geometry_score"] = geometry_score
            if geometry_score is not None and geometry_score < 0.45:
                self.implausible_people_rejected += 1
                continue
            if bool(person.get("detection_only_obstacle", False)):
                self.detection_only_measurements += 1
            measurements.append((person, center))

        track_ids = sorted(self.tracks)
        costs = np.full((len(track_ids), len(measurements)), np.inf,
                        dtype=np.float64)
        for row, track_id in enumerate(track_ids):
            for col, (_person, center) in enumerate(measurements):
                costs[row, col] = self._cost(
                    self.tracks[track_id], center, stamp)
        if not track_ids or not measurements:
            self.last_assignment_method = "none"
        elif len(measurements) > EXACT_ASSIGNMENT_LIMIT:
            self.last_assignment_method = "hungarian"
        else:
            self.last_assignment_method = "exact_dp"
        matched_tracks, matched_measurements = set(), set()
        remaining_costs = costs.copy()
        while True:
            pairs = _optimal_assignment(remaining_costs)
            if not pairs:
                break
            rejected_pair = False
            for row, col in pairs:
                track_id = track_ids[row]
                person, center = measurements[col]
                accepted = self._observe_track(
                    self.tracks[track_id], person, center, stamp)
                if accepted:
                    matched_tracks.add(track_id)
                    matched_measurements.add(col)
                    remaining_costs[row, :] = np.inf
                    remaining_costs[:, col] = np.inf
                else:
                    # A center-only assignment can still join the wrong body
                    # during a crossing. Remove only that edge and let the
                    # same measurement try another predicted track before it
                    # becomes a new identity.
                    remaining_costs[row, col] = np.inf
                    rejected_pair = True
            if not rejected_pair:
                break

        required_hits = self.confirmation_hits
        for col, (person, center) in enumerate(measurements):
            if col in matched_measurements:
                continue
            same_ray_track = (self.enable_ray_recovery and any(
                track["person_id"] is not None and
                stamp - track["last_observed"] <= self.deletion_timeout and
                _bearing_angle(track["center"], center) <= math.radians(24.0)
                for track in self.tracks.values()))
            if same_ray_track:
                self.last_new_measurements_suppressed += 1
                continue
            person_required_hits = required_hits
            if bool(person.get("detection_only_obstacle", False)):
                # A detector box with no usable pose is valuable as a coarse
                # alert, but must persist for one simulated second before it is
                # promoted to an articulated Human slot.
                person_required_hits = max(10, person_required_hits)
            geometry_score = person.get("body_geometry_score")
            if geometry_score is not None and geometry_score < 0.70:
                person_required_hits = max(
                    LOW_GEOMETRY_CONFIRMATION_HITS, person_required_hits)
            self._new_track(
                person, center, stamp, person_required_hits)

        expired = [
            track_id for track_id, track in self.tracks.items()
            if stamp - track["last_observed"] > (
                self.deletion_timeout if track["person_id"] is not None
                else min(0.65, self.deletion_timeout))]
        for track_id in expired:
            self.tracks.pop(track_id, None)

        output = []
        for track_key in sorted(self.tracks):
            track = self.tracks[track_key]
            missing = stamp - track["last_observed"]
            if track["person_id"] is None or missing > self.prediction_timeout:
                continue
            value = self._output(track, stamp, missing > 1e-4)
            output_joint_ids = _usable_joint_ids(value)
            if not _has_usable_body(output_joint_ids):
                self.sparse_outputs_suppressed += 1
                continue
            center = _person_center(value)
            if (center is not None and self.max_person_range > 0.0 and
                    float(np.linalg.norm(center)) > self.max_person_range):
                continue
            output.append(value)
        return sorted(output, key=lambda value: value["person_id"])

    def status(self, stamp_ns=None):
        stamp = None if stamp_ns is None else float(stamp_ns) / 1e9
        tracks = []
        for track_key in sorted(self.tracks):
            track = self.tracks[track_key]
            missing = (0.0 if stamp is None else
                       max(0.0, stamp - track["last_observed"]))
            tracks.append({
                "track_key": int(track_key),
                "person_id": track["person_id"],
                "hits": int(track["hits"]),
                "consecutive_hits": int(track["consecutive_hits"]),
                "required_hits": int(track["required_hits"]),
                "age": int(track["age"]),
                "missing_ms": round(missing * 1000.0, 3),
                "speed_mps": round(float(np.linalg.norm(
                    track["velocity"])), 4),
                "bone_constraints": len(track["bone_lengths"]),
                "detection_only_hits": int(track["detection_only_hits"]),
                "pose_hits": int(track["pose_hits"]),
            })
        return {
            "mode": "ego_compensated_kalman_kinematic",
            "session_id": self.session_id,
            "active_tracks": len(self.tracks),
            "next_person_id": self.next_id,
            "prediction_timeout_s": self.prediction_timeout,
            "deletion_timeout_s": self.deletion_timeout,
            "assignment_method": self.last_assignment_method,
            "dense_assignment_threshold": EXACT_ASSIGNMENT_LIMIT,
            "ray_recovery_enabled": self.enable_ray_recovery,
            "new_measurements_suppressed": int(
                self.last_new_measurements_suppressed),
            "kalman_measurements_rejected": int(
                self.kalman_measurements_rejected),
            "implausible_people_rejected": int(
                self.implausible_people_rejected),
            "detection_only_measurements": int(
                self.detection_only_measurements),
            "sparse_measurements_rejected": int(
                self.sparse_measurements_rejected),
            "association_measurements_rejected": int(
                self.association_measurements_rejected),
            "coherent_relocalizations": int(
                self.coherent_relocalizations),
            "symmetric_joint_pair_swaps": int(
                self.symmetric_joint_pair_swaps),
            "sparse_outputs_suppressed": int(
                self.sparse_outputs_suppressed),
            "ego_motion_compensated": (
                self._last_ego_position_world is not None),
            "ego_motion_updates": int(self.ego_motion_updates),
            "last_ego_translation_m": round(
                float(self.last_ego_translation_m), 6),
            "last_ego_rotation_deg": round(
                float(self.last_ego_rotation_deg), 6),
            "tracks": tracks,
        }
