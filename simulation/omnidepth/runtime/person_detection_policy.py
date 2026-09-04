"""Pure person-candidate validation shared by runtime and unit tests."""

import numpy as np


# The native image is reduced to a 416x320 rectified anchor.  A side-on adult
# at the edge of the useful range can be only 7--11 pixels wide, and a partly
# cropped/occluded body can be shorter than 28 pixels.  These are only ROI
# sanity limits; the articulated RTMPose contract below remains the semantic
# false-positive guard for shelf and ceiling proposals.
MIN_PERSON_BOX_HEIGHT_PX = 16.0
MIN_ARTICULATED_JOINT_SCORE = 0.15
MIN_ARTICULATED_STRONGEST_MEAN = 0.20


def person_box_is_usable(box):
    """Return whether one detector ROI is large enough for pose inference."""
    value = np.asarray(box, dtype=np.float64).reshape(4)
    if not np.isfinite(value).all():
        return False
    return (float(value[2] - value[0]) > 0.0 and
            float(value[3] - value[1]) >= MIN_PERSON_BOX_HEIGHT_PX)


def articulated_person_is_valid(points, scores, threshold,
                                detector_box=None):
    """Require an articulated body, not merely a detector rectangle.

    Face-only responses and spatially collapsed pose hallucinations are common
    signatures of warehouse shelving false positives.  The checks remain
    deliberately looser than a canonical standing pose so partially occluded
    and turning pedestrians remain observable.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if scores.size < 17 or points.shape[0] < 17:
        return False
    body_visible = scores[5:17] >= MIN_ARTICULATED_JOINT_SCORE
    torso_visible = scores[[5, 6, 11, 12]] >= MIN_ARTICULATED_JOINT_SCORE
    if np.count_nonzero(body_visible) < 4 or \
            np.count_nonzero(torso_visible) < 2:
        return False
    strongest = np.sort(scores[5:17])[-6:]
    if float(np.mean(strongest)) < max(
            MIN_ARTICULATED_STRONGEST_MEAN, float(threshold)):
        return False
    visible_points = points[5:17][body_visible]
    if not np.isfinite(visible_points).all():
        return False
    extent = np.ptp(visible_points, axis=0)
    if detector_box is None:
        return float(max(extent)) >= 12.0
    box = np.asarray(detector_box, dtype=np.float64).reshape(4)
    width = max(1.0, float(box[2] - box[0]))
    height = max(1.0, float(box[3] - box[1]))
    if max(float(extent[0]) / width, float(extent[1]) / height) < .12:
        return False
    margin = np.asarray((.30 * width, .30 * height), dtype=np.float64)
    if np.any(visible_points < box[:2] - margin) or \
            np.any(visible_points > box[2:] + margin):
        return False
    shoulder_ids = [index for index in (5, 6)
                    if scores[index] >= MIN_ARTICULATED_JOINT_SCORE]
    hip_ids = [index for index in (11, 12)
               if scores[index] >= MIN_ARTICULATED_JOINT_SCORE]
    if shoulder_ids and hip_ids:
        shoulder_y = float(np.mean(points[shoulder_ids, 1]))
        hip_y = float(np.mean(points[hip_ids, 1]))
        if hip_y - shoulder_y < .03 * height:
            return False
    return True
