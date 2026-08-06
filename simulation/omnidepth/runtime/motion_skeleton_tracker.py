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
from functools import lru_cache

import numpy as np


CORE_JOINTS = (5, 6, 11, 12)
EXACT_ASSIGNMENT_LIMIT = 12


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


def _decayed_step(position, velocity, dt, decay):
    """Advance without reversing while velocity exponentially approaches 0."""
    if dt <= 0.0:
        return position, velocity
    factor = math.exp(-decay * dt)
    travel = dt if decay <= 1e-9 else (1.0 - factor) / decay
    return position + velocity * travel, velocity * factor


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


class _JointState:
    def __init__(self, joint, stamp):
        self.position = np.asarray(joint["xyz_imu_m"], dtype=np.float64)
        self.observed_position = self.position.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.last_stamp = stamp
        self.last_observed = stamp
        self.template = dict(joint)

    def predict(self, stamp, missing):
        dt = max(0.0, min(0.5, stamp - self.last_stamp))
        decay = 1.35 if missing else 0.18
        self.position, self.velocity = _decayed_step(
            self.position, self.velocity, dt, decay)
        self.last_stamp = stamp

    def observe(self, joint, stamp):
        measurement = np.asarray(joint["xyz_imu_m"], dtype=np.float64)
        measurement = self.position + _limit(
            measurement - self.position, 0.55)
        dt = max(1e-3, stamp - self.last_observed)
        measured_velocity = _limit(
            (measurement - self.observed_position) / dt, 3.0)
        self.velocity = _limit(
            self.velocity * 0.72 + measured_velocity * 0.28, 2.2)
        self.position = self.position * 0.20 + measurement * 0.80
        self.observed_position = measurement
        self.last_observed = stamp
        self.last_stamp = stamp
        self.template = dict(joint)

    def output(self, stamp, predicted):
        value = dict(self.template)
        xyz = self.position.round(6).tolist()
        value["xyz_imu_m"] = xyz
        value["xyz_base_link_m"] = xyz
        value["predicted"] = bool(predicted)
        value["measurement_age_ms"] = round(
            max(0.0, stamp - self.last_observed) * 1000.0, 3)
        if predicted:
            value["source"] = "motion_prediction"
            value["score"] = round(
                float(value.get("score", 0.0)) *
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
        self.session_id = uuid.uuid4().hex
        self.tracks = {}
        self.next_track_key = 0
        # Preserve the simulation dataset's positive-ID convention.
        self.next_id = 1
        self.ever_confirmed = False
        self.last_new_measurements_suppressed = 0
        self.last_assignment_method = "none"

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
            "last_stamp": stamp,
            "last_observed": stamp,
            "hits": 1,
            "consecutive_hits": 1,
            "required_hits": int(required_hits),
            "age": 1,
            "joints": joints,
            "template": dict(person),
        }

    def _predict_track(self, track, stamp):
        missing = stamp - track["last_observed"] > 0.15
        dt = max(0.0, min(0.5, stamp - track["last_stamp"]))
        decay = 1.35 if missing else 0.18
        track["center"], track["velocity"] = _decayed_step(
            track["center"], track["velocity"], dt, decay)
        track["last_stamp"] = stamp
        track["age"] += 1
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

    def _observe_track(self, track, person, center, stamp):
        dt = max(1e-3, stamp - track["last_observed"])
        maximum_innovation = (0.40 + 0.18 * min(1.5, dt) +
                              0.10 * float(np.linalg.norm(track["velocity"])))
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
        for joint in person["joints"]:
            joint_id = int(joint["id"])
            if joint.get("xyz_imu_m") is None:
                continue
            if joint_id not in track["joints"]:
                track["joints"][joint_id] = _JointState(joint, stamp)
            else:
                track["joints"][joint_id].observe(joint, stamp)

    def _output(self, track, stamp, predicted):
        template = dict(track["template"])
        old_joints = {int(joint["id"]): joint
                      for joint in template.get("joints", [])}
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
                    stamp, predicted or joint_age > 1e-4))
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
        return template

    def update(self, people, stamp_ns):
        stamp = float(stamp_ns) / 1e9
        self.last_new_measurements_suppressed = 0
        for track in self.tracks.values():
            self._predict_track(track, stamp)

        measurements = []
        for person in people:
            center = _person_center(person)
            if center is not None:
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
        pairs = _optimal_assignment(costs)
        matched_tracks, matched_measurements = set(), set()
        for row, col in pairs:
            track_id = track_ids[row]
            person, center = measurements[col]
            self._observe_track(self.tracks[track_id], person, center, stamp)
            matched_tracks.add(track_id)
            matched_measurements.add(col)

        later_required_hits = max(7, self.confirmation_hits)
        required_hits = (later_required_hits if self.ever_confirmed
                         else self.confirmation_hits)
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
            self._new_track(person, center, stamp, required_hits)

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
            })
        return {
            "mode": "3d_motion_prediction",
            "active_tracks": len(self.tracks),
            "next_person_id": self.next_id,
            "prediction_timeout_s": self.prediction_timeout,
            "deletion_timeout_s": self.deletion_timeout,
            "assignment_method": self.last_assignment_method,
            "dense_assignment_threshold": EXACT_ASSIGNMENT_LIMIT,
            "ray_recovery_enabled": self.enable_ray_recovery,
            "new_measurements_suppressed": int(
                self.last_new_measurements_suppressed),
            "tracks": tracks,
        }
