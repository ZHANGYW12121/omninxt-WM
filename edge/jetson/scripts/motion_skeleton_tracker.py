#!/usr/bin/env python3
"""Small 3D skeleton tracker based only on position and motion continuity.

This deliberately avoids image-space IDs, appearance ReID, projected ROIs and
camera-specific tokens.  The detector/pose/stereo front end first produces one
fused 3D skeleton per visible person.  This tracker then assigns persistent
IDs by matching those measurements to constant-velocity predictions.
"""

import math
import uuid
from functools import lru_cache

import numpy as np


CORE_JOINTS = (5, 6, 11, 12)


def _person_center(person):
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
    """Advance without ever reversing; velocity exponentially approaches 0."""
    if dt <= 0.0:
        return position, velocity
    factor = math.exp(-decay * dt)
    travel = dt if decay <= 1e-9 else (1.0 - factor) / decay
    return position + velocity * travel, velocity * factor


def _optimal_assignment(costs):
    """Max-cardinality, minimum-cost assignment for the small people count."""
    costs = np.asarray(costs, dtype=np.float64)
    tracks, measurements = costs.shape
    if not tracks or not measurements:
        return []
    if measurements > 12:
        candidates = [(float(costs[row, col]), row, col)
                      for row in range(tracks)
                      for col in range(measurements)
                      if np.isfinite(costs[row, col])]
        used_rows, used_cols, result = set(), set(), []
        for cost, row, col in sorted(candidates):
            if row not in used_rows and col not in used_cols:
                used_rows.add(row)
                used_cols.add(col)
                result.append((row, col))
        return result

    @lru_cache(maxsize=None)
    def solve(row, used_mask):
        if row == tracks:
            return 0, 0.0, ()
        best = solve(row + 1, used_mask)
        for col in range(measurements):
            if used_mask & (1 << col) or not np.isfinite(costs[row, col]):
                continue
            count, total, pairs = solve(row + 1, used_mask | (1 << col))
            candidate = count + 1, total + float(costs[row, col]), \
                ((row, col),) + pairs
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
        # Sparse stereo can occasionally choose the wrong repeated texture and
        # move one joint several metres along its camera ray.  Identity has
        # already been decided at body level, so bound a single-frame joint
        # innovation instead of rendering that alias as a second skeleton.
        measurement = self.position + _limit(
            measurement - self.position, .55)
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
            value["score"] = round(float(value.get("score", 0.0)) *
                                   math.exp(-(stamp - self.last_observed) / .7),
                                   6)
        return value


class MotionSkeletonTracker:
    """Assign IDs after 3D fusion and coast occluded people to a slow stop."""

    def __init__(self, joint_names, confirmation_hits=2,
                 prediction_timeout=1.2, deletion_timeout=4.0,
                 base_gate=0.70, max_speed=2.0):
        self.joint_names = tuple(joint_names)
        self.confirmation_hits = int(confirmation_hits)
        self.prediction_timeout = float(prediction_timeout)
        self.deletion_timeout = float(deletion_timeout)
        self.base_gate = float(base_gate)
        self.max_speed = float(max_speed)
        self.session_id = uuid.uuid4().hex
        self.tracks = {}
        self.next_track_key = 0
        self.next_id = 0
        self.ever_confirmed = False
        self.last_new_measurements_suppressed = 0

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
        missing = stamp - track["last_observed"] > .15
        dt = max(0.0, min(.5, stamp - track["last_stamp"]))
        decay = 1.35 if missing else .18
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
        gate = self.base_gate + min(.75, .35 * speed + .35 * missing)
        if track["person_id"] is None:
            gate = max(gate, 1.35)
        angle = _bearing_angle(track["center"], center)
        angular_gate = math.radians(22.0 + 5.0 * min(1.5, missing))
        ray_continuity = (track["person_id"] is not None and
                          angle <= angular_gate)
        dormant_recovery = (track["person_id"] is not None and
                            missing >= .35 and
                            missing <= self.deletion_timeout)
        if distance > gate and not ray_continuity and not dormant_recovery:
            return float("inf")

        direction_penalty = 0.0
        displacement = center - track["observed_center"]
        displacement_norm = float(np.linalg.norm(displacement))
        if speed > .12 and displacement_norm > .06:
            cosine = float(np.dot(track["velocity"], displacement) /
                           (speed * displacement_norm))
            # Opposite motion is expensive during a short crossing, while a
            # real turn remains possible after the track has slowed down.
            direction_penalty = .45 * max(0.0, 1.0 - cosine)
        if distance <= gate:
            spatial_cost = distance
        elif ray_continuity:
            # Preserve identity across a bad radial-depth frame or a camera
            # boundary. Bearing comes from calibrated base_link geometry and
            # is much more stable than sparse disparity in these cases.
            radial_delta = abs(float(np.linalg.norm(center)) -
                               float(np.linalg.norm(track["center"])))
            spatial_cost = (.48 + 2.2 * angle +
                            .08 * min(3.0, radial_delta))
        else:
            # If a camera boundary produced no detections for a while, retain
            # the finite set of known identities internally.  Global
            # assignment plus motion direction chooses among dormant people;
            # the identity is not rendered after prediction_timeout.
            radial_delta = abs(float(np.linalg.norm(center)) -
                               float(np.linalg.norm(track["center"])))
            spatial_cost = (1.10 + .75 * angle +
                            .06 * min(4.0, radial_delta))
        # A provisional depth fragment must not steal observations from an
        # already published identity.  Maximum-cardinality assignment still
        # lets it take an extra measurement when a genuinely new person is
        # present, but confirmed tracks win whenever both explain the same
        # finite set of people.
        provisional_penalty = 4.0 if track["person_id"] is None else 0.0
        return (spatial_cost + direction_penalty + .03 * missing +
                provisional_penalty)

    def _observe_track(self, track, person, center, stamp):
        dt = max(1e-3, stamp - track["last_observed"])
        maximum_innovation = (.40 + .18 * min(1.5, dt) +
                              .10 * float(np.linalg.norm(track["velocity"])))
        center = track["center"] + _limit(
            center - track["center"], maximum_innovation)
        measured_velocity = _limit(
            (center - track["observed_center"]) / dt, self.max_speed * 1.4)
        old_velocity = track["velocity"]
        # A single ambiguous crossing frame must not reverse a mature track.
        if np.linalg.norm(old_velocity) > .15 and \
                np.dot(measured_velocity, old_velocity) < 0.0 and dt < .45:
            direction = old_velocity / np.linalg.norm(old_velocity)
            measured_velocity = measured_velocity - min(
                0.0, float(np.dot(measured_velocity, direction))) * direction
        track["velocity"] = _limit(
            old_velocity * .70 + measured_velocity * .30, self.max_speed)
        track["center"] = track["center"] * .18 + center * .82
        track["observed_center"] = center.copy()
        track["last_observed"] = stamp
        track["last_stamp"] = stamp
        track["hits"] += 1
        track["consecutive_hits"] = (
            track["consecutive_hits"] + 1 if dt <= .35 else 1)
        if track["person_id"] is None and \
                track["consecutive_hits"] >= track["required_hits"]:
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
            if state is not None:
                joints.append(state.output(stamp, predicted))
            else:
                value = dict(old_joints.get(joint_id, {}))
                value.update({"id": joint_id, "name": name,
                              "xyz_imu_m": None,
                              "xyz_base_link_m": None,
                              "predicted": False})
                joints.append(value)
        template["joints"] = joints
        template["person_id"] = int(track["person_id"])
        template["track_uid"] = "{}:{}".format(
            self.session_id, track["person_id"])
        template["track_state"] = "predicted" if predicted else "observed"
        template["id_uncertain"] = bool(predicted)
        template["track_age_frames"] = int(track["age"])
        template["time_since_observation_ms"] = round(
            max(0.0, stamp - track["last_observed"]) * 1000.0, 3)
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
        pairs = _optimal_assignment(costs)
        matched_tracks, matched_measurements = set(), set()
        for row, col in pairs:
            track_id = track_ids[row]
            person, center = measurements[col]
            self._observe_track(self.tracks[track_id], person, center, stamp)
            matched_tracks.add(track_id)
            matched_measurements.add(col)

        bootstrap_required_hits = self.confirmation_hits
        later_required_hits = max(7, self.confirmation_hits)
        new_track_required_hits = (
            later_required_hits if self.ever_confirmed
            else bootstrap_required_hits)
        for col, (person, center) in enumerate(measurements):
            if col not in matched_measurements:
                same_ray_track = any(
                    track["person_id"] is not None and
                    stamp - track["last_observed"] <= self.deletion_timeout and
                    _bearing_angle(track["center"], center) <=
                    math.radians(24.0)
                    for track in self.tracks.values())
                if same_ray_track:
                    self.last_new_measurements_suppressed += 1
                    continue
                self._new_track(
                    person, center, stamp, new_track_required_hits)

        expired = [
            track_id for track_id, track in self.tracks.items()
            if stamp - track["last_observed"] > (
                self.deletion_timeout if track["person_id"] is not None
                else min(.65, self.deletion_timeout))]
        for track_id in expired:
            self.tracks.pop(track_id, None)

        output = []
        for track_key in sorted(self.tracks):
            track = self.tracks[track_key]
            missing = stamp - track["last_observed"]
            if track["person_id"] is None:
                continue
            if missing <= self.prediction_timeout:
                output.append(self._output(track, stamp, missing > 1e-4))
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
            "new_measurements_suppressed": int(
                self.last_new_measurements_suppressed),
            "tracks": tracks,
        }
