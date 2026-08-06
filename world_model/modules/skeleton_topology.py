"""Joint topology helpers shared by the compact dataset and world model."""

from __future__ import annotations


COCO17_JOINT_COUNT = 17
COCO12_BODY_JOINT_COUNT = 12

# COCO17 indices 0..4 (nose, eyes and ears) are deliberately not recorded.
COCO12_BODY_SOURCE_INDICES = tuple(range(5, 17))
COCO12_BODY_JOINT_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
COCO12_BODY_EDGES = (
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),
)


def hip_joint_indices(num_joints: int) -> tuple[int, int] | None:
    """Return left/right hip indices for a supported stored topology."""
    count = int(num_joints)
    if count == COCO12_BODY_JOINT_COUNT:
        return 6, 7
    if count >= COCO17_JOINT_COUNT:
        return 11, 12
    return None


def skeleton_edges(num_joints: int) -> tuple[tuple[int, int], ...]:
    """Return an exact topology; silently truncating COCO17 is incorrect."""
    count = int(num_joints)
    if count == COCO12_BODY_JOINT_COUNT:
        return COCO12_BODY_EDGES
    if count == COCO17_JOINT_COUNT:
        return COCO17_EDGES
    raise ValueError(f"Unsupported skeleton topology with {count} joints")
