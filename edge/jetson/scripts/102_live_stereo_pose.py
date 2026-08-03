#!/usr/bin/env python3
"""10 Hz four-direction stereo 3D skeleton experiment.

The processing path stays in memory: fresh rectified stereo inputs are used for
2D pose and sparse joint triangulation, while the slower HITNet metric depth is
cached only for validation and fallback.  JPEG generation is isolated in an
optional display thread and never feeds perception.
"""

import argparse
import importlib.util
import json
import math
import os
import queue
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, PoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from skeleton_stream import SkeletonTcpSender, build_skeleton_packet
from trt_rtmpose import TensorRTRTMPose, TensorRTYOLOX


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
spec = importlib.util.spec_from_file_location(
    "pose3d", os.path.join(SCRIPT_DIR, "100_pose_depth_skeleton.py"))
pose3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose3d)


PAIR_NAMES = ("AB_RIGHT", "BC_REAR", "CD_LEFT", "DA_FRONT")
PAIR_COLORS = ((70, 80, 255), (90, 225, 90), (255, 145, 75), (50, 225, 245))
ANCHOR_NAMES = ("CAM_A_FRONT_RIGHT", "CAM_B_REAR_RIGHT",
                "CAM_C_REAR_LEFT", "CAM_D_FRONT_LEFT")
ANCHOR_COLORS = ((70, 80, 255), (90, 225, 90),
                 (255, 145, 75), (50, 225, 245))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--det-engine", required=True)
    parser.add_argument("--pose-engine", required=True)
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--display-hz", type=float, default=4.0)
    # YOLOX scans the 2x2 mosaic containing every direction.  RTMPose keeps
    # updating both stereo views on the intervening frames, so a 10-frame
    # detector cadence does not disable any sector and leaves enough GPU time
    # for the 10 Hz stereo-pose path alongside the 5 Hz dense HITNet path.
    parser.add_argument("--det-interval", type=int, default=10)
    parser.add_argument("--det-threshold", type=float, default=0.28)
    parser.add_argument("--person-threshold", type=float, default=0.32)
    parser.add_argument("--keypoint-threshold", type=float, default=0.28)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--min-disparity", type=float, default=0.75)
    parser.add_argument("--max-disparity", type=float, default=96.0)
    parser.add_argument("--patch-radius", type=int, default=5)
    parser.add_argument("--patch-search", type=int, default=18)
    parser.add_argument("--min-ncc", type=float, default=0.15)
    parser.add_argument("--merge-distance", type=float, default=0.9)
    parser.add_argument("--anchor-width", type=int, default=416)
    parser.add_argument("--anchor-height", type=int, default=320)
    parser.add_argument("--anchor-fov", type=float, default=120.0)
    parser.add_argument("--center-det-interval", type=int, default=10)
    parser.add_argument("--rescue-det-interval", type=int, default=3)
    parser.add_argument("--backend-host", default="",
                        help="Optional backend IP/hostname for skeleton TCP")
    parser.add_argument("--backend-port", type=int, default=9765)
    return parser.parse_args()


def decode_image(message):
    if message.encoding in ("bgr8", "rgb8"):
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step)[:, :message.width * 3]
        image = image.reshape(message.height, message.width, 3).copy()
        if message.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    if message.encoding == "mono8":
        dtype = np.dtype(np.uint8)
    elif message.encoding == "32FC1":
        dtype = np.dtype(np.float32)
    else:
        raise ValueError("Unsupported image encoding " + message.encoding)
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    values = message.step // dtype.itemsize
    image = np.frombuffer(message.data, dtype=dtype).reshape(
        message.height, values)[:, :message.width]
    return image.astype(dtype.newbyteorder("="), copy=True)


def build_anchor_maps(cameras, width, height, horizontal_fov_deg):
    """Build four optical-axis pinhole views from the calibrated Mei cameras."""
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    pixel_y, pixel_x = np.indices((height, width), dtype=np.float32)
    ray_x = (pixel_x - width * .5) / focal
    ray_y = (pixel_y - height * .5) / focal
    ray_z = np.ones_like(ray_x)
    norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    maps = []
    for camera_id in range(4):
        camera = cameras["cam{}".format(camera_id)]
        xi, fx, fy, cx, cy = (float(value)
                              for value in camera["intrinsics"])
        k1, k2, p1, p2 = (float(value)
                          for value in camera["distortion_coeffs"])
        denominator = ray_z + xi * norm
        x = ray_x / denominator
        y = ray_y / denominator
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
        distorted_x = (x * radial + 2.0 * p1 * x * y +
                       p2 * (radius2 + 2.0 * x * x))
        distorted_y = (y * radial + p1 * (radius2 + 2.0 * y * y) +
                       2.0 * p2 * x * y)
        maps.append(((fx * distorted_x + cx).astype(np.float32),
                     (fy * distorted_y + cy).astype(np.float32)))
    return maps, focal


class InputBridge:
    """Synchronize two image bundles and cache slower four-pair HITNet depth."""

    def __init__(self):
        self.required = {"anchors", "stereo"}
        self.buckets = defaultdict(dict)
        self.lock = threading.Lock()
        self.frames = queue.Queue(maxsize=1)
        self.depth = [None] * 4
        self.depth_stamp = [None] * 4
        self.subscribers = []
        self.subscribers.append(rospy.Subscriber(
            "/depth_estimation/pose_anchor_mosaic", Image,
            self._bundle_cb, callback_args="anchors", queue_size=1,
            buff_size=2 * 1024 * 1024, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber(
            "/depth_estimation/pose_stereo_mosaic", Image,
            self._bundle_cb, callback_args="stereo", queue_size=1,
            buff_size=2 * 1024 * 1024, tcp_nodelay=True))
        for index in range(4):
            self.subscribers.append(rospy.Subscriber(
                "/depth_estimation/stereo_{}/depth".format(index), Image,
                self._depth_cb, callback_args=index, queue_size=1,
                buff_size=2 * 1024 * 1024, tcp_nodelay=True))

    def _bundle_cb(self, message, kind):
        stamp = message.header.stamp.to_nsec()
        image = decode_image(message)
        images = {}
        if kind == "anchors":
            if image.shape != (640, 832):
                rospy.logwarn_throttle(
                    2.0, "Expected anchor mosaic 832x640, got %s",
                    image.shape)
                return
            for index in range(4):
                x = (index % 2) * 416
                y = (index // 2) * 320
                images[(index, "anchor")] = image[y:y + 320, x:x + 416]
        else:
            if image.shape != (960, 640):
                rospy.logwarn_throttle(
                    2.0, "Expected stereo mosaic 640x960, got %s",
                    image.shape)
                return
            for index in range(4):
                y = index * 240
                images[(index, "left")] = image[y:y + 240, :320]
                images[(index, "right")] = image[y:y + 240, 320:640]
        complete = None
        with self.lock:
            self.buckets[stamp][kind] = images
            if self.required.issubset(self.buckets[stamp]):
                bundles = self.buckets.pop(stamp)
                complete_images = {}
                complete_images.update(bundles["anchors"])
                complete_images.update(bundles["stereo"])
                complete = (stamp, complete_images)
            for old in sorted(self.buckets)[:-12]:
                self.buckets.pop(old, None)
        if complete is not None:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(complete)
            except queue.Full:
                pass

    def _depth_cb(self, message, index):
        depth = decode_image(message)
        with self.lock:
            self.depth[index] = depth
            self.depth_stamp[index] = message.header.stamp.to_nsec()

    def depth_snapshot(self):
        with self.lock:
            return ([None if value is None else value.copy()
                     for value in self.depth], list(self.depth_stamp))

def make_mosaic(images):
    return np.vstack((np.hstack(images[:2]), np.hstack(images[2:])))


def clip_quadrant_boxes(boxes, view_width=320, view_height=240):
    result = []
    for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        x0, y0, x1, y1 = box
        center_x, center_y = (x0 + x1) * .5, (y0 + y1) * .5
        col, row = int(center_x >= view_width), int(center_y >= view_height)
        qx, qy = col * view_width, row * view_height
        clipped = [max(qx, x0), max(qy, y0),
                   min(qx + view_width - 1, x1),
                   min(qy + view_height - 1, y1)]
        if clipped[2] - clipped[0] >= 18 and clipped[3] - clipped[1] >= 28:
            result.append(clipped)
    return np.asarray(result, dtype=np.float32).reshape(-1, 4)


def boxes_from_pose(keypoints, scores, view_width=320, view_height=240):
    boxes = []
    for points, confidence in zip(keypoints, scores):
        valid = confidence >= .22
        if np.count_nonzero(valid) < 3:
            continue
        xy = points[valid]
        center = np.median(xy, axis=0)
        col, row = (int(center[0] >= view_width),
                    int(center[1] >= view_height))
        qx, qy = col * view_width, row * view_height
        low, high = np.min(xy, axis=0), np.max(xy, axis=0)
        width = max(30.0, high[0] - low[0])
        height = max(50.0, high[1] - low[1])
        boxes.append([max(qx, low[0] - width * .28),
                      max(qy, low[1] - height * .28),
                      min(qx + view_width - 1, high[0] + width * .28),
                      min(qy + view_height - 1, high[1] + height * .28)])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def seed_right_boxes(left_boxes, nominal_disparity=10.0):
    """Seed right-view boxes from rectified left boxes.

    Stereo correspondence moves toward smaller x in the right image.  RTMPose
    padding and the extra horizontal margin cover the normal 4--25 px person
    disparity range; subsequent right-view keypoints update the boxes.
    """
    shifted = []
    for box in np.asarray(left_boxes, dtype=np.float32).reshape(-1, 4):
        center_x = (box[0] + box[2]) * .5
        center_y = (box[1] + box[3]) * .5
        col, row = int(center_x >= 320), int(center_y >= 240)
        qx, qy = col * 320, row * 240
        shifted.append([
            max(qx, box[0] - nominal_disparity - 6.0),
            max(qy, box[1]),
            min(qx + 319, box[2] - nominal_disparity + 6.0),
            min(qy + 239, box[3]),
        ])
    return np.asarray(shifted, dtype=np.float32).reshape(-1, 4)


def infer_view(mosaic, detector, estimator, boxes, run_detector,
               view_width=320, view_height=240):
    bgr = (mosaic if mosaic.ndim == 3 else
           cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR))
    detection_ms = 0.0
    if detector is not None and (run_detector or len(boxes) == 0):
        started = time.monotonic()
        detected, classes = detector(bgr)
        detected = np.asarray(detected, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int32).reshape(-1)
        boxes = clip_quadrant_boxes(
            detected[classes == 0], view_width, view_height)
        detection_ms = (time.monotonic() - started) * 1000.0
    started = time.monotonic()
    if len(boxes):
        keypoints, scores = estimator(bgr, bboxes=boxes)
    else:
        keypoints = np.empty((0, 17, 2), dtype=np.float32)
        scores = np.empty((0, 17), dtype=np.float32)
    pose_ms = (time.monotonic() - started) * 1000.0
    next_boxes = boxes_from_pose(
        keypoints, scores, view_width, view_height)
    # A single low-confidence pose frame must not discard the detector box.
    # The next scheduled all-sector YOLOX pass will still refresh/remove it.
    if not len(next_boxes) and len(boxes):
        next_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    people = [[] for _ in range(4)]
    for detection_id, (points, confidence) in enumerate(
            zip(keypoints, scores)):
        valid = confidence >= .22
        if np.count_nonzero(valid) < 3:
            continue
        center = np.median(points[valid], axis=0)
        sector = (int(center[1] >= view_height) * 2 +
                  int(center[0] >= view_width))
        offset = np.array([(sector % 2) * view_width,
                           (sector // 2) * view_height], dtype=np.float32)
        people[sector].append({
            "detection_id": detection_id,
            "points": points - offset,
            "scores": confidence,
        })
    return people, next_boxes, detection_ms, pose_ms


def replace_quadrant_boxes(boxes, sector, replacement,
                           view_width, view_height):
    """Replace one quadrant's detector boxes while retaining other tracks."""
    retained = []
    for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        center_x = float(box[0] + box[2]) * .5
        center_y = float(box[1] + box[3]) * .5
        box_sector = (int(center_y >= view_height) * 2 +
                      int(center_x >= view_width))
        if box_sector != sector:
            retained.append(box.tolist())
    offset = np.array([(sector % 2) * view_width,
                       (sector // 2) * view_height] * 2, dtype=np.float32)
    for box in np.asarray(replacement, dtype=np.float32).reshape(-1, 4):
        retained.append((box + offset).tolist())
    return np.asarray(retained, dtype=np.float32).reshape(-1, 4)


def deduplicate_anchor_boxes(boxes, cameras, focal, view_width, view_height):
    """Keep the most central view of one person before expensive RTMPose."""
    candidates = []
    for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        center_x = float(box[0] + box[2]) * .5
        center_y = float(box[1] + box[3]) * .5
        sector = (int(center_y >= view_height) * 2 +
                  int(center_x >= view_width))
        offset_x = (sector % 2) * view_width
        offset_y = (sector // 2) * view_height
        local_x, local_y = center_x - offset_x, center_y - offset_y
        ray_camera = np.array([
            (local_x - view_width * .5) / focal,
            (local_y - view_height * .5) / focal,
            1.0,
        ], dtype=np.float64)
        t_camera_body = np.asarray(
            cameras["cam{}".format(sector)]["T_cam_imu"],
            dtype=np.float64)
        ray_body = t_camera_body[:3, :3].T.dot(ray_camera)
        ray_body /= max(1e-9, np.linalg.norm(ray_body))
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        edge_margin = min(local_x, view_width - local_x,
                          local_y, view_height - local_y)
        centrality = edge_margin / min(view_width, view_height)
        score = centrality + .04 * math.log(max(1.0, width * height))
        candidates.append({
            "box": box, "ray": ray_body, "height": height,
            "score": score,
        })
    retained = []
    cosine_limit = math.cos(math.radians(16.0))
    for candidate in sorted(candidates, key=lambda value: -value["score"]):
        duplicate = False
        for selected in retained:
            height_ratio = candidate["height"] / selected["height"]
            same_bearing = float(np.dot(
                candidate["ray"], selected["ray"])) >= cosine_limit
            similar_scale = .55 <= height_ratio <= 1.8
            similar_elevation = abs(
                candidate["ray"][2] - selected["ray"][2]) <= .22
            if same_bearing and similar_scale and similar_elevation:
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return np.asarray([value["box"] for value in retained],
                      dtype=np.float32).reshape(-1, 4)


def detect_single_view(image, detector):
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    started = time.monotonic()
    detected, classes = detector(bgr)
    detected = np.asarray(detected, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int32).reshape(-1)
    boxes = detected[classes == 0]
    if len(boxes):
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, image.shape[1] - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, image.shape[0] - 1)
        boxes = np.asarray([
            box for box in boxes
            if box[2] - box[0] >= 18 and box[3] - box[1] >= 28
        ], dtype=np.float32).reshape(-1, 4)
    return boxes, (time.monotonic() - started) * 1000.0


def association_cost(left, right, keypoint_threshold):
    valid = ((left["scores"] >= keypoint_threshold) &
             (right["scores"] >= keypoint_threshold))
    if np.count_nonzero(valid) < 5:
        return None
    lp, rp = left["points"][valid], right["points"][valid]
    disparity = lp[:, 0] - rp[:, 0]
    vertical = np.abs(lp[:, 1] - rp[:, 1])
    median_disparity = float(np.median(disparity))
    median_vertical = float(np.median(vertical))
    if median_vertical > 12.0 or not (-3.0 < median_disparity < 105.0):
        return None
    dispersion = float(np.median(np.abs(disparity - median_disparity)))
    return median_vertical + .25 * dispersion + \
        .05 * abs(min(0.0, median_disparity))


def associate_people(left_people, right_people, keypoint_threshold):
    candidates = []
    for left_index, left in enumerate(left_people):
        for right_index, right in enumerate(right_people):
            cost = association_cost(left, right, keypoint_threshold)
            if cost is not None:
                candidates.append((cost, left_index, right_index))
    matched = {}
    used_right = set()
    for _, left_index, right_index in sorted(candidates):
        if left_index in matched or right_index in used_right:
            continue
        matched[left_index] = right_index
        used_right.add(right_index)
    return matched


def refine_epipolar_match(left, right, left_point, right_point,
                          patch_radius, search_radius, min_ncc):
    left_x, left_y = (float(value) for value in left_point)
    right_x = float(right_point[0])
    radius = patch_radius
    lx, ly = int(round(left_x)), int(round(left_y))
    if lx - radius < 0 or lx + radius >= left.shape[1] or \
            ly - radius < 0 or ly + radius >= left.shape[0]:
        return None
    template = left[ly - radius:ly + radius + 1,
                    lx - radius:lx + radius + 1]
    if float(np.std(template)) < 4.0:
        return None
    rx0 = max(radius, int(math.floor(right_x - search_radius)))
    rx1 = min(right.shape[1] - radius - 1,
              int(math.ceil(right_x + search_radius)))
    ry0 = max(radius, ly - 3)
    ry1 = min(right.shape[0] - radius - 1, ly + 3)
    if rx1 <= rx0 or ry1 < ry0:
        return None
    search = right[ry0 - radius:ry1 + radius + 1,
                   rx0 - radius:rx1 + radius + 1]
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(response)
    if not np.isfinite(maximum) or maximum < min_ncc:
        return None
    best_x = rx0 + location[0]
    best_y = ry0 + location[1]
    # One-dimensional parabolic refinement around the best horizontal score.
    row, col = location[1], location[0]
    subpixel = 0.0
    if 0 < col < response.shape[1] - 1:
        left_value = float(response[row, col - 1])
        center_value = float(response[row, col])
        right_value = float(response[row, col + 1])
        denominator = left_value - 2.0 * center_value + right_value
        if abs(denominator) > 1e-6:
            subpixel = float(np.clip(
                .5 * (left_value - right_value) / denominator, -.75, .75))
    return np.array([best_x + subpixel, float(best_y)], dtype=np.float64), \
        float(maximum)


def project_point(projection, xyz):
    homogeneous = projection.dot(np.r_[xyz, 1.0])
    return homogeneous[:2] / homogeneous[2]


def triangulate_joint(geometry, left_point, right_point,
                      min_disparity, max_disparity, min_depth, max_depth):
    disparity = float(left_point[0] - right_point[0])
    if not (min_disparity <= disparity <= max_disparity):
        return None
    homogeneous = cv2.triangulatePoints(
        geometry["projection"], geometry["projection_right"],
        np.asarray(left_point, dtype=np.float64).reshape(2, 1),
        np.asarray(right_point, dtype=np.float64).reshape(2, 1))[:, 0]
    if abs(float(homogeneous[3])) < 1e-9:
        return None
    xyz_rect = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(xyz_rect)) or \
            not (min_depth <= xyz_rect[2] <= max_depth):
        return None
    reprojection = max(
        float(np.linalg.norm(project_point(
            geometry["projection"], xyz_rect) - left_point)),
        float(np.linalg.norm(project_point(
            geometry["projection_right"], xyz_rect) - right_point)))
    if reprojection > 3.0:
        return None
    xyz_imu = geometry["rotation"].dot(xyz_rect) + geometry["translation"]
    return xyz_rect, xyz_imu, disparity, reprojection


def project_anchor_point(geometry, camera_id, point, focal, width, height):
    """Project a physical-camera center-view pixel into a rectified pair."""
    ray_camera = np.array([
        (float(point[0]) - width * .5) / focal,
        (float(point[1]) - height * .5) / focal,
        1.0,
    ], dtype=np.float64)
    if camera_id == geometry["left_camera_id"]:
        ray_virtual = geometry["virtual_rotation_left"].T.dot(ray_camera)
        ray_rectified = geometry["rectification_left"].dot(ray_virtual)
        intrinsic = geometry["projection"][:, :3]
        side = "left"
    elif camera_id == geometry["right_camera_id"]:
        ray_virtual = geometry["virtual_rotation_right"].T.dot(ray_camera)
        ray_rectified = geometry["rectification_right"].dot(ray_virtual)
        intrinsic = geometry["projection_right"][:, :3]
        side = "right"
    else:
        return None
    if ray_rectified[2] <= 1e-8:
        return None
    pixel = intrinsic.dot(ray_rectified)
    pixel = pixel[:2] / pixel[2]
    if not np.all(np.isfinite(pixel)):
        return None
    return side, pixel


def match_anchor_epipolar(left, right, anchor_side, anchor_point,
                          patch_radius, min_disparity, max_disparity,
                          min_ncc):
    """Match one anchor joint without requiring a second pose detection."""
    source = left if anchor_side == "left" else right
    target = right if anchor_side == "left" else left
    source_x, source_y = (float(value) for value in anchor_point)
    radius = patch_radius
    sx, sy = int(round(source_x)), int(round(source_y))
    if sx - radius < 0 or sx + radius >= source.shape[1] or \
            sy - radius < 0 or sy + radius >= source.shape[0]:
        return None
    template = source[sy - radius:sy + radius + 1,
                      sx - radius:sx + radius + 1]
    if float(np.std(template)) < 4.0:
        return None
    if anchor_side == "left":
        target_x0 = source_x - max_disparity
        target_x1 = source_x - min_disparity
    else:
        target_x0 = source_x + min_disparity
        target_x1 = source_x + max_disparity
    x0 = max(radius, int(math.floor(min(target_x0, target_x1))))
    x1 = min(target.shape[1] - radius - 1,
             int(math.ceil(max(target_x0, target_x1))))
    y0 = max(radius, sy - 3)
    y1 = min(target.shape[0] - radius - 1, sy + 3)
    if x1 <= x0 or y1 < y0:
        return None
    search = target[y0 - radius:y1 + radius + 1,
                    x0 - radius:x1 + radius + 1]
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(response)
    if not np.isfinite(maximum) or maximum < min_ncc:
        return None
    row, col = location[1], location[0]
    subpixel = 0.0
    if 0 < col < response.shape[1] - 1:
        value_left = float(response[row, col - 1])
        value_center = float(response[row, col])
        value_right = float(response[row, col + 1])
        denominator = value_left - 2.0 * value_center + value_right
        if abs(denominator) > 1e-6:
            subpixel = float(np.clip(
                .5 * (value_left - value_right) / denominator, -.75, .75))
    target_point = np.array(
        [x0 + col + subpixel, float(y0 + row)], dtype=np.float64)
    anchor_point = np.asarray(anchor_point, dtype=np.float64)
    if anchor_side == "left":
        return anchor_point, target_point, float(maximum)
    return target_point, anchor_point, float(maximum)


def anchor_person_is_valid(person, threshold):
    scores = np.asarray(person["scores"], dtype=np.float64)
    visible = scores >= .22
    if np.count_nonzero(visible) < 3:
        return False
    strongest = np.sort(scores)[-min(8, scores.size):]
    return float(np.mean(strongest)) >= max(.24, threshold - .05)


def guard_anchor_skeleton(person):
    """Reject isolated stereo matches that violate conservative body geometry.

    This is a sparse-joint correspondence check, not point-cloud filtering.
    Repetitive clothing/background texture can give one joint a high NCC score
    at a completely wrong disparity.  The four shoulder/hip joints provide a
    robust body reference, while deliberately loose metric bounds retain
    unusual poses and perspective foreshortening.
    """
    joints = person["joints"]
    core_ids = (5, 6, 11, 12)
    core = [np.asarray(joints[index]["xyz_imu_m"], dtype=np.float64)
            for index in core_ids
            if joints[index]["xyz_imu_m"] is not None and
            (joints[index].get("measurement_sigma_m") or 2.0) <= .75]
    if len(core) < 2:
        return 0
    center = np.median(np.asarray(core), axis=0)
    # Maximum distance from the shoulder/hip median.  These are intentionally
    # wider than normal adult proportions; they only catch gross disparity
    # aliases (for example an eye at 4 m while the torso is at 0.8 m).
    max_radius = {
        0: 1.25, 1: 1.35, 2: 1.35, 3: 1.45, 4: 1.45,
        5: 1.00, 6: 1.00, 7: 1.45, 8: 1.45, 9: 1.90, 10: 1.90,
        11: 1.00, 12: 1.00, 13: 1.55, 14: 1.55,
        15: 2.20, 16: 2.20,
    }
    rejected = 0
    for joint in joints:
        xyz = joint.get("xyz_imu_m")
        if xyz is None:
            continue
        distance = float(np.linalg.norm(
            np.asarray(xyz, dtype=np.float64) - center))
        sigma = float(joint.get("measurement_sigma_m") or 2.0)
        if distance <= max_radius[joint["id"]] and sigma <= 1.0:
            continue
        joint["rejected_source"] = joint["source"]
        joint["rejected_xyz_imu_m"] = joint["xyz_imu_m"]
        joint.update({
            "source": "invalid_kinematic_outlier",
            "depth_m": None,
            "depth_mad_m": None,
            "measurement_sigma_m": None,
            "xyz_rect_m": None,
            "xyz_imu_m": None,
        })
        rejected += 1
    return rejected


def make_anchor_person(camera_id, anchor_person, geometries,
                       left_images, right_images, depths, depth_stamps,
                       frame_stamp, anchor_focal, anchor_width,
                       anchor_height, args):
    """Lift one complete center-view pose through either adjacent stereo pair."""
    person = {
        "pair": "ANCHOR_CAM_{}".format(chr(65 + camera_id)),
        "anchor_camera": "CAM_{}".format(chr(65 + camera_id)),
        "detection_id": anchor_person["detection_id"],
        "color": "#{:02x}{:02x}{:02x}".format(
            *ANCHOR_COLORS[camera_id][::-1]),
        "joints": [],
    }
    candidates_for_camera = [
        (pair_id, geometry)
        for pair_id, geometry in enumerate(geometries)
        if camera_id in (geometry["left_camera_id"],
                         geometry["right_camera_id"])
    ]
    for joint_id, name in enumerate(pose3d.JOINTS):
        anchor_point = np.asarray(
            anchor_person["points"][joint_id], dtype=np.float64)
        confidence = float(anchor_person["scores"][joint_id])
        joint = {
            "id": joint_id, "name": name,
            "anchor_camera": person["anchor_camera"],
            "anchor_pixel": anchor_point.round(3).tolist(),
            "pixel": anchor_point.round(3).tolist(),
            "pixel_int": np.rint(anchor_point).astype(int).tolist(),
            "right_pixel": None, "score": round(confidence, 6),
            "right_score": None, "source": "invalid",
            "source_pair": None, "depth_m": None,
            "depth_mad_m": None, "measurement_sigma_m": None,
            "ncc": None, "disparity_px": None,
            "reprojection_error_px": None, "hitnet_depth_m": None,
            "hitnet_age_ms": None, "hitnet_consistent": None,
            "xyz_rect_m": None, "xyz_imu_m": None,
        }
        best = None
        fallback = None
        if confidence >= max(.20, args.keypoint_threshold - .06):
            for pair_id, geometry in candidates_for_camera:
                projection = project_anchor_point(
                    geometry, camera_id, anchor_point, anchor_focal,
                    anchor_width, anchor_height)
                if projection is None:
                    continue
                anchor_side, rectified_point = projection
                if not (-2 <= rectified_point[0] < 322 and
                        -2 <= rectified_point[1] < 242):
                    continue
                matched = match_anchor_epipolar(
                    left_images[pair_id], right_images[pair_id],
                    anchor_side, rectified_point, args.patch_radius,
                    args.min_disparity, args.max_disparity, args.min_ncc)
                if anchor_side == "left":
                    sampled = None if depths[pair_id] is None else \
                        pose3d.sample_depth(
                            depths[pair_id], rectified_point[0],
                            rectified_point[1], 2,
                            args.min_depth, args.max_depth)
                    if sampled is not None:
                        fallback = (pair_id, geometry, rectified_point,
                                    sampled)
                if matched is None:
                    continue
                left_point, right_point, ncc = matched
                triangulated = triangulate_joint(
                    geometry, left_point, right_point,
                    args.min_disparity, args.max_disparity,
                    args.min_depth, args.max_depth)
                if triangulated is None:
                    continue
                xyz_rect, xyz_body, disparity, reprojection = triangulated
                hitnet_depth = None
                hitnet_mad = None
                hitnet_age = None
                if depths[pair_id] is not None:
                    sampled = pose3d.sample_depth(
                        depths[pair_id], left_point[0], left_point[1], 2,
                        args.min_depth, args.max_depth)
                    if sampled is not None:
                        hitnet_depth, hitnet_mad, _ = sampled
                        if depth_stamps[pair_id] is not None:
                            hitnet_age = abs(
                                frame_stamp - depth_stamps[pair_id]) / 1e6
                consistent = None if hitnet_depth is None else \
                    abs(float(xyz_rect[2]) - hitnet_depth) <= max(
                        .35, .25 * float(xyz_rect[2]))
                pixel_sigma = max(.20, 1.25 * (1.0 - ncc))
                sigma_z = (xyz_rect[2] ** 2 / max(
                    1e-6, geometry["projection"][0, 0] *
                    geometry["baseline_m"])) * pixel_sigma
                quality = (ncc - .03 * reprojection -
                           .04 * min(2.0, sigma_z) +
                           (.08 if consistent is True else 0.0))
                candidate = {
                    "quality": quality, "pair_id": pair_id,
                    "geometry": geometry, "left": left_point,
                    "right": right_point, "ncc": ncc,
                    "xyz_rect": xyz_rect, "xyz_body": xyz_body,
                    "disparity": disparity, "reprojection": reprojection,
                    "sigma_z": sigma_z, "hitnet_depth": hitnet_depth,
                    "hitnet_mad": hitnet_mad, "hitnet_age": hitnet_age,
                    "consistent": consistent,
                }
                if best is None or candidate["quality"] > best["quality"]:
                    best = candidate
        if best is not None:
            geometry = best["geometry"]
            joint.update({
                "pixel": best["left"].round(3).tolist(),
                "pixel_int": np.rint(best["left"]).astype(int).tolist(),
                "right_pixel": best["right"].round(3).tolist(),
                "source": "anchor_epipolar",
                "source_pair": geometry["name"],
                "depth_m": round(float(best["xyz_rect"][2]), 6),
                "depth_mad_m": round(float(max(.01, best["sigma_z"])), 6),
                "measurement_sigma_m": round(float(np.clip(
                    best["sigma_z"], .025, 1.5)), 6),
                "ncc": round(float(best["ncc"]), 5),
                "disparity_px": round(float(best["disparity"]), 5),
                "reprojection_error_px": round(
                    float(best["reprojection"]), 5),
                "hitnet_depth_m": None if best["hitnet_depth"] is None else
                    round(float(best["hitnet_depth"]), 6),
                "hitnet_age_ms": best["hitnet_age"],
                "hitnet_consistent": best["consistent"],
                "xyz_rect_m": best["xyz_rect"].round(6).tolist(),
                "xyz_imu_m": best["xyz_body"].round(6).tolist(),
            })
        elif fallback is not None:
            pair_id, geometry, left_point, sampled = fallback
            hitnet_depth, hitnet_mad, _ = sampled
            xyz_rect, xyz_body = pose3d.lift_joint(
                geometry, left_point[0], left_point[1], hitnet_depth)
            hitnet_age = None if depth_stamps[pair_id] is None else \
                abs(frame_stamp - depth_stamps[pair_id]) / 1e6
            joint.update({
                "pixel": left_point.round(3).tolist(),
                "pixel_int": np.rint(left_point).astype(int).tolist(),
                "source": "hitnet_anchor_fallback",
                "source_pair": geometry["name"],
                "depth_m": round(float(hitnet_depth), 6),
                "depth_mad_m": round(float(hitnet_mad), 6),
                "measurement_sigma_m": round(float(max(
                    .4, min(2.0, hitnet_mad * 3.0))), 6),
                "hitnet_depth_m": round(float(hitnet_depth), 6),
                "hitnet_age_ms": hitnet_age,
                "xyz_rect_m": xyz_rect.round(6).tolist(),
                "xyz_imu_m": xyz_body.round(6).tolist(),
            })
        person["joints"].append(joint)
    person["kinematic_rejected_joints"] = guard_anchor_skeleton(person)
    person["source_pairs"] = sorted({
        joint["source_pair"] for joint in person["joints"]
        if joint.get("source_pair") is not None
    })
    return person


def make_stereo_person(sector, left_person, right_person, geometry,
                       left_image, right_image, depth, depth_stamp,
                       frame_stamp, args):
    person = {
        "pair": geometry["name"], "side": geometry["side"],
        "detection_id": left_person["detection_id"],
        "color": geometry["color"], "joints": [],
    }
    for joint_id, name in enumerate(pose3d.JOINTS):
        left_point = np.asarray(left_person["points"][joint_id], dtype=np.float64)
        left_score = float(left_person["scores"][joint_id])
        right_score = 0.0 if right_person is None else \
            float(right_person["scores"][joint_id])
        joint = {
            "id": joint_id, "name": name,
            "pixel": left_point.round(3).tolist(),
            "pixel_int": np.rint(left_point).astype(int).tolist(),
            "right_pixel": None, "score": round(left_score, 6),
            "right_score": round(right_score, 6), "source": "invalid",
            "depth_m": None, "depth_mad_m": None,
            "measurement_sigma_m": None, "ncc": None,
            "disparity_px": None, "reprojection_error_px": None,
            "hitnet_depth_m": None, "hitnet_age_ms": None,
            "hitnet_consistent": None, "xyz_rect_m": None,
            "xyz_imu_m": None,
        }
        if left_score >= args.keypoint_threshold and right_person is not None \
                and right_score >= args.keypoint_threshold:
            semantic_right = np.asarray(
                right_person["points"][joint_id], dtype=np.float64)
            refined = refine_epipolar_match(
                left_image, right_image, left_point, semantic_right,
                args.patch_radius, args.patch_search, args.min_ncc)
            match_source = "stereo_patch"
            ncc = None
            if refined is not None:
                right_point, ncc = refined
            elif abs(left_point[1] - semantic_right[1]) <= 6.0:
                right_point = semantic_right
                match_source = "stereo_pose"
            else:
                right_point = None
            if right_point is not None:
                tri = triangulate_joint(
                    geometry, left_point, right_point,
                    args.min_disparity, args.max_disparity,
                    args.min_depth, args.max_depth)
                if tri is not None:
                    xyz_rect, xyz_imu, disparity, reprojection = tri
                    pixel_sigma = .45 if ncc is None else \
                        max(.20, 1.25 * (1.0 - ncc))
                    sigma_z = (xyz_rect[2] ** 2 /
                               max(1e-6, geometry["projection"][0, 0] *
                                   geometry["baseline_m"])) * pixel_sigma
                    joint.update({
                        "right_pixel": right_point.round(3).tolist(),
                        "source": match_source,
                        "depth_m": round(float(xyz_rect[2]), 6),
                        "depth_mad_m": round(float(max(.01, sigma_z)), 6),
                        "measurement_sigma_m": round(
                            float(np.clip(sigma_z, .025, 1.5)), 6),
                        "ncc": None if ncc is None else round(ncc, 5),
                        "disparity_px": round(disparity, 5),
                        "reprojection_error_px": round(reprojection, 5),
                        "xyz_rect_m": xyz_rect.round(6).tolist(),
                        "xyz_imu_m": xyz_imu.round(6).tolist(),
                    })
        sampled = None
        if depth is not None and left_score >= args.keypoint_threshold:
            sampled = pose3d.sample_depth(
                depth, left_point[0], left_point[1], 2,
                args.min_depth, args.max_depth)
        if sampled is not None:
            hitnet_depth, hitnet_mad, _ = sampled
            hitnet_age = None if depth_stamp is None else \
                abs(frame_stamp - depth_stamp) / 1e6
            joint["hitnet_depth_m"] = round(hitnet_depth, 6)
            joint["hitnet_age_ms"] = None if hitnet_age is None else \
                round(hitnet_age, 3)
            if joint["depth_m"] is not None:
                tolerance = max(.35, .25 * joint["depth_m"])
                joint["hitnet_consistent"] = \
                    abs(joint["depth_m"] - hitnet_depth) <= tolerance
            elif hitnet_age is None or hitnet_age <= 350.0:
                xyz_rect, xyz_imu = pose3d.lift_joint(
                    geometry, left_point[0], left_point[1], hitnet_depth)
                joint.update({
                    "source": "hitnet_fallback",
                    "depth_m": round(hitnet_depth, 6),
                    "depth_mad_m": round(max(.03, hitnet_mad), 6),
                    # Dense HITNet is a validation/fallback source here, not a
                    # truth measurement.  Give it visibly lower confidence
                    # than sparse geometric triangulation so a bad dense
                    # patch cannot pull a stable skeleton onto a depth plane.
                    "measurement_sigma_m": round(
                        float(np.clip(max(.35, hitnet_mad * 3), .35, 2.0)), 6),
                    "xyz_rect_m": xyz_rect.round(6).tolist(),
                    "xyz_imu_m": xyz_imu.round(6).tolist(),
                })
        person["joints"].append(joint)
    # A wrong patch match can have an apparently small reprojection error yet
    # an almost-zero disparity, placing one limb many metres behind the rest
    # of the body.  Compare each joint with the robust disparity of this same
    # skeleton.  This is sparse skeleton validation, not dense-cloud filtering.
    reliable_disparities = [
        joint["disparity_px"] for joint in person["joints"]
        if joint["disparity_px"] is not None and
        joint["measurement_sigma_m"] is not None and
        joint["measurement_sigma_m"] <= .6]
    if len(reliable_disparities) >= 5:
        median_disparity = float(np.median(reliable_disparities))
        lower = max(args.min_disparity, median_disparity * .35)
        upper = min(args.max_disparity, median_disparity * 3.0)
        for joint in person["joints"]:
            disparity = joint["disparity_px"]
            sigma = joint["measurement_sigma_m"]
            implausible = disparity is not None and (
                disparity < lower or disparity > upper or
                (sigma is not None and sigma > .75))
            if not implausible:
                continue
            hitnet_depth = joint.get("hitnet_depth_m")
            hitnet_age = joint.get("hitnet_age_ms")
            if hitnet_depth is not None and \
                    (hitnet_age is None or hitnet_age <= 350.0):
                x, y = joint["pixel"]
                xyz_rect, xyz_imu = pose3d.lift_joint(
                    geometry, x, y, hitnet_depth)
                joint.update({
                    "source": "hitnet_guarded_fallback",
                    "depth_m": hitnet_depth,
                    "depth_mad_m": max(.05, joint.get("depth_mad_m") or .05),
                    "measurement_sigma_m": max(
                        .4, min(2.0, (joint.get("depth_mad_m") or .1) * 3)),
                    "xyz_rect_m": xyz_rect.round(6).tolist(),
                    "xyz_imu_m": xyz_imu.round(6).tolist(),
                })
            else:
                joint.update({
                    "source": "invalid_disparity_outlier",
                    "depth_m": None, "depth_mad_m": None,
                    "measurement_sigma_m": None,
                    "xyz_rect_m": None, "xyz_imu_m": None,
                })
    return person


def person_center(person):
    preferred = (5, 6, 11, 12)
    points = [person["joints"][index]["xyz_imu_m"] for index in preferred]
    points = [point for point in points if point is not None]
    if len(points) < 2:
        points = [joint["xyz_imu_m"] for joint in person["joints"]
                  if joint["xyz_imu_m"] is not None]
    return None if not points else np.median(np.asarray(points), axis=0)


def person_overlap_distance(first, second):
    """Robust same-person distance using corresponding valid 3D joints."""
    distances = []
    for joint_first, joint_second in zip(first["joints"], second["joints"]):
        if joint_first["xyz_imu_m"] is None or \
                joint_second["xyz_imu_m"] is None:
            continue
        distances.append(float(np.linalg.norm(
            np.asarray(joint_first["xyz_imu_m"], dtype=np.float64) -
            np.asarray(joint_second["xyz_imu_m"], dtype=np.float64))))
    if len(distances) < 3:
        return None
    return float(np.median(distances))


def fuse_people(raw_people, merge_distance):
    clusters = []
    for person in raw_people:
        center = person_center(person)
        if center is None:
            continue
        best, best_distance = None, float("inf")
        for cluster in clusters:
            if person["pair"] in cluster["pairs"]:
                continue
            distance = float(np.linalg.norm(center - cluster["center"]))
            overlap = [person_overlap_distance(person, member)
                       for member in cluster["members"]]
            overlap = [value for value in overlap if value is not None]
            joint_distance = min(overlap) if overlap else None
            same_person = (distance < merge_distance or
                           (joint_distance is not None and
                            joint_distance < merge_distance * .7))
            if same_person and distance < best_distance:
                best, best_distance = cluster, distance
        if best is None:
            clusters.append({"members": [person], "pairs": {person["pair"]},
                             "center": center})
        else:
            best["members"].append(person)
            best["pairs"].add(person["pair"])
            best["center"] = np.mean(
                [person_center(member) for member in best["members"]], axis=0)
    fused = []
    for cluster in clusters:
        joints = []
        for joint_id, name in enumerate(pose3d.JOINTS):
            candidates = [member["joints"][joint_id]
                          for member in cluster["members"]]
            candidates = [value for value in candidates
                          if value["xyz_imu_m"] is not None]
            if not candidates:
                joints.append({"id": joint_id, "name": name,
                               "xyz_imu_m": None, "score": 0.0,
                               "source": "invalid"})
                continue
            weights = np.array([
                max(.01, value["score"]) /
                max(.025, value.get("measurement_sigma_m") or .2)
                for value in candidates], dtype=np.float64)
            xyz = np.average(np.asarray(
                [value["xyz_imu_m"] for value in candidates]),
                axis=0, weights=weights)
            best = candidates[int(np.argmax(weights))]
            joints.append({
                "id": joint_id, "name": name,
                "xyz_imu_m": xyz.round(6).tolist(),
                "score": round(float(max(value["score"]
                                         for value in candidates)), 6),
                "source": best["source"],
                "measurement_sigma_m": best.get("measurement_sigma_m"),
                "hitnet_age_ms": best.get("hitnet_age_ms"),
                "hitnet_consistent": best.get("hitnet_consistent"),
            })
        physical_pairs = sorted({
            pair for member in cluster["members"]
            for pair in member.get("source_pairs", [])
        })
        fused.append({"source_pairs": physical_pairs,
                      "source_anchors": sorted(cluster["pairs"]),
                      "joints": joints})
    return fused


class KalmanJoint:
    def __init__(self, position, stamp_sec, sigma, score=0.0,
                 source="unknown", source_age_ms=0.0):
        self.state = np.r_[position, np.zeros(3, dtype=np.float64)]
        self.covariance = np.diag([sigma ** 2] * 3 + [.8] * 3)
        self.stamp = stamp_sec
        self.last_measurement = stamp_sec
        self.last_score = float(score)
        self.last_sigma = float(sigma)
        self.last_source = str(source)
        self.last_source_age_ms = float(source_age_ms)

    def predict(self, stamp_sec):
        dt = max(0.0, min(.5, stamp_sec - self.stamp))
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        acceleration_noise = .8
        process = np.eye(6) * 1e-5
        process[:3, :3] *= max(1e-4, dt ** 4 * acceleration_noise)
        process[3:, 3:] *= max(1e-4, dt ** 2 * acceleration_noise)
        self.state = transition.dot(self.state)
        self.covariance = transition.dot(self.covariance).dot(
            transition.T) + process
        self.stamp = stamp_sec

    def update(self, measurement, stamp_sec, sigma, max_innovation=None,
               score=0.0, source="unknown", source_age_ms=0.0):
        self.predict(stamp_sec)
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3)
        noise = np.eye(3) * sigma ** 2
        innovation = measurement - observation.dot(self.state)
        if max_innovation is not None and \
                float(np.linalg.norm(innovation)) > max_innovation:
            return False
        residual_covariance = observation.dot(self.covariance).dot(
            observation.T) + noise
        gain = self.covariance.dot(observation.T).dot(
            np.linalg.inv(residual_covariance))
        self.state += gain.dot(innovation)
        self.covariance = (np.eye(6) - gain.dot(observation)).dot(
            self.covariance)
        self.last_measurement = stamp_sec
        self.last_score = float(score)
        self.last_sigma = float(sigma)
        self.last_source = str(source)
        self.last_source_age_ms = float(source_age_ms)
        return True


class SkeletonTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, people, stamp_ns):
        stamp_sec = stamp_ns / 1e9
        measurements = []
        for person in people:
            center = person_center(person)
            if center is not None:
                measurements.append((person, center))
        associations = {}
        candidates = []
        for person_index, (_, center) in enumerate(measurements):
            for track_id, track in self.tracks.items():
                distance = float(np.linalg.norm(center - track["center_body"]))
                if distance < 1.5:
                    candidates.append((distance, person_index, track_id))
        used_tracks = set()
        for _, person_index, track_id in sorted(candidates):
            if person_index not in associations and track_id not in used_tracks:
                associations[person_index] = track_id
                used_tracks.add(track_id)
        for person_index in range(len(measurements)):
            if person_index not in associations:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "joints": {}, "center_body": measurements[person_index][1],
                    "last_seen": stamp_sec}
                associations[person_index] = track_id
        output = []
        for person_index, (person, _) in enumerate(measurements):
            track_id = associations[person_index]
            track = self.tracks[track_id]
            filtered_joints = []
            for joint in person["joints"]:
                value = dict(joint)
                xyz = joint["xyz_imu_m"]
                joint_filter = track["joints"].get(joint["id"])
                if xyz is not None:
                    body = np.asarray(xyz, dtype=np.float64)
                    sigma = float(joint.get("measurement_sigma_m") or .2)
                    source = str(joint.get("source") or "unknown")
                    source_age_ms = float(joint.get("hitnet_age_ms") or 0.0) \
                        if str(source).startswith("hitnet_") else 0.0
                    if joint_filter is None:
                        joint_filter = KalmanJoint(
                            body, stamp_sec, sigma, joint.get("score", 0.0),
                            source, source_age_ms)
                        track["joints"][joint["id"]] = joint_filter
                    else:
                        # Reject isolated HITNet fallbacks that disagree with
                        # the motion-predicted joint by more than 0.6 m.  A
                        # fresh stereo triangulation is allowed a wider gate
                        # because it is the primary geometric measurement.
                        primary_stereo = (source.startswith("stereo_") or
                                          source == "anchor_epipolar")
                        # At 10 Hz a real joint should not jump half a metre
                        # between consecutive observations.  Reset to a fresh
                        # stereo measurement beyond that bound instead of
                        # letting an old Kalman state visibly trail the body.
                        gate = .6 if source.startswith("hitnet_") else .5
                        accepted = joint_filter.update(
                            body, stamp_sec, sigma, max_innovation=gate,
                            score=joint.get("score", 0.0),
                            source=source, source_age_ms=source_age_ms)
                        if not accepted and primary_stereo:
                            joint_filter = KalmanJoint(
                                body, stamp_sec, sigma,
                                joint.get("score", 0.0),
                                source, source_age_ms)
                            track["joints"][joint["id"]] = joint_filter
                elif joint_filter is not None:
                    joint_filter.predict(stamp_sec)
                if joint_filter is not None and \
                        stamp_sec - joint_filter.last_measurement <= .35:
                    body = joint_filter.state[:3]
                    measurement_age_ms = max(
                        0.0, (stamp_sec - joint_filter.last_measurement) * 1e3 +
                        joint_filter.last_source_age_ms)
                    value["xyz_base_link_raw_m"] = xyz
                    value["xyz_base_link_m"] = body.round(6).tolist()
                    # Legacy JSON aliases kept for existing consumers. They
                    # refer to the same fixed base_link calibration origin.
                    value["xyz_imu_raw_m"] = xyz
                    value["xyz_imu_m"] = body.round(6).tolist()
                    value["predicted"] = xyz is None
                    value["measurement_age_ms"] = round(
                        measurement_age_ms, 3)
                    value["measurement_source"] = joint_filter.last_source
                    value["last_measurement_score"] = round(
                        joint_filter.last_score, 6)
                    value["last_measurement_sigma_m"] = round(
                        joint_filter.last_sigma, 6)
                else:
                    value["xyz_base_link_m"] = None
                    value["xyz_imu_m"] = None
                    value["predicted"] = False
                    value["measurement_age_ms"] = None
                filtered_joints.append(value)
            result = dict(person)
            result["person_id"] = track_id
            result["joints"] = filtered_joints
            center = person_center(result)
            if center is not None:
                track["center_body"] = center
            track["last_seen"] = stamp_sec
            output.append(result)
        self.tracks = {track_id: track for track_id, track in self.tracks.items()
                       if stamp_sec - track["last_seen"] <= 1.0}
        return output


def publish_ros(publishers, people, stamp_ns):
    stamp = rospy.Time.from_sec(stamp_ns / 1e9)
    poses = PoseArray()
    poses.header.stamp = stamp
    poses.header.frame_id = "base_link"
    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for person_index, person in enumerate(people):
        color = ((1.0, .3, .3), (.3, 1.0, .4), (.3, .6, 1.0),
                 (1.0, .85, .25))[person_index % 4]
        spheres = Marker()
        spheres.header = poses.header
        spheres.ns = "joints"
        spheres.id = person["person_id"] * 2
        spheres.type = Marker.SPHERE_LIST
        spheres.action = Marker.ADD
        spheres.scale.x = spheres.scale.y = spheres.scale.z = .055
        spheres.color.r, spheres.color.g, spheres.color.b = color
        spheres.color.a = 1.0
        spheres.lifetime = rospy.Duration(.3)
        lines = Marker()
        lines.header = poses.header
        lines.ns = "bones"
        lines.id = person["person_id"] * 2 + 1
        lines.type = Marker.LINE_LIST
        lines.action = Marker.ADD
        lines.scale.x = .028
        lines.color.r, lines.color.g, lines.color.b = color
        lines.color.a = .9
        lines.lifetime = rospy.Duration(.3)
        points = {}
        for joint in person["joints"]:
            if joint["xyz_imu_m"] is None:
                continue
            x, y, z = joint["xyz_imu_m"]
            point = Point(x=x, y=y, z=z)
            points[joint["id"]] = point
            spheres.points.append(point)
            pose = Pose()
            pose.position = point
            pose.orientation.w = 1.0
            poses.poses.append(pose)
        for first, second in pose3d.BONES:
            if first in points and second in points:
                lines.points.extend((points[first], points[second]))
        markers.markers.extend((spheres, lines))
    publishers["poses"].publish(poses)
    publishers["markers"].publish(markers)


def draw_pose_2d(view, people, color):
    for person in people:
        for first, second in pose3d.BONES:
            if person["scores"][first] >= .28 and \
                    person["scores"][second] >= .28:
                point_a = tuple(np.rint(
                    person["points"][first]).astype(int))
                point_b = tuple(np.rint(
                    person["points"][second]).astype(int))
                cv2.line(view, point_a, point_b, color, 2, cv2.LINE_AA)
        for point, score in zip(person["points"], person["scores"]):
            if score >= .28:
                cv2.circle(view, tuple(np.rint(point).astype(int)), 3,
                           color, -1, cv2.LINE_AA)


def annotate_pair(left, right, triangulated_people,
                  left_people, right_people, color):
    left_view = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_view = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    draw_pose_2d(left_view, left_people, (40, 245, 80))
    draw_pose_2d(right_view, right_people, (40, 220, 255))
    for person in triangulated_people:
        for first, second in pose3d.BONES:
            joint_a, joint_b = person["joints"][first], person["joints"][second]
            if joint_a["score"] >= .28 and joint_b["score"] >= .28:
                cv2.line(left_view, tuple(joint_a["pixel_int"]),
                         tuple(joint_b["pixel_int"]), color, 2, cv2.LINE_AA)
            if joint_a.get("right_pixel") is not None and \
                    joint_b.get("right_pixel") is not None:
                cv2.line(right_view,
                         tuple(np.rint(joint_a["right_pixel"]).astype(int)),
                         tuple(np.rint(joint_b["right_pixel"]).astype(int)),
                         color, 2, cv2.LINE_AA)
        for joint in person["joints"]:
            if joint["score"] < .28:
                continue
            point = tuple(joint["pixel_int"])
            cv2.circle(left_view, point, 3, (40, 240, 255), -1, cv2.LINE_AA)
            if joint.get("right_pixel") is not None:
                cv2.circle(right_view,
                           tuple(np.rint(joint["right_pixel"]).astype(int)),
                           3, (40, 240, 255), -1, cv2.LINE_AA)
            if joint["depth_m"] is not None:
                cv2.putText(left_view, "{}:{:.1f}m".format(
                    joint["id"], joint["depth_m"]),
                    (point[0] + 3, point[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, .28, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return np.hstack((left_view, right_view))


def annotate_anchor(image, pose_people, triangulated_people, color):
    view = image.copy() if image.ndim == 3 else \
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    draw_pose_2d(view, pose_people, (40, 245, 80))
    for person in triangulated_people:
        joints = person["joints"]
        for first, second in pose3d.BONES:
            first_joint, second_joint = joints[first], joints[second]
            if first_joint["score"] < .22 or second_joint["score"] < .22:
                continue
            first_point = tuple(np.rint(
                first_joint["anchor_pixel"]).astype(int))
            second_point = tuple(np.rint(
                second_joint["anchor_pixel"]).astype(int))
            cv2.line(view, first_point, second_point, color, 2, cv2.LINE_AA)
        for joint in joints:
            if joint["score"] < .22:
                continue
            point = tuple(np.rint(joint["anchor_pixel"]).astype(int))
            valid = joint["xyz_imu_m"] is not None
            cv2.circle(view, point, 4,
                       (40, 240, 255) if valid else (80, 80, 255),
                       -1, cv2.LINE_AA)
            if joint["depth_m"] is not None:
                cv2.putText(view, "{}:{:.1f}m".format(
                    joint["id"], joint["depth_m"]),
                    (point[0] + 3, point[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, .30, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return view


def draw_3d(people, width=640, height=960):
    panel = np.full((height, width, 3), (18, 23, 31), dtype=np.uint8)
    # Fixed ground-plane view: X/Y form the floor and +Z is exactly vertical
    # on screen.  This changes only visualization, never the base_link values.
    origin = np.array([width * .5, height * .76])
    yaw, ground_tilt, scale = -.72, .34, 105.0

    def project(value):
        x, y, z = value
        ground_x = math.cos(yaw) * x - math.sin(yaw) * y
        ground_depth = math.sin(yaw) * x + math.cos(yaw) * y
        screen_y = ground_depth * ground_tilt - z
        return tuple(np.rint(
            origin + [ground_x * scale, screen_y * scale]).astype(int))

    for value in np.arange(-3, 3.01, .5):
        cv2.line(panel, project([value, -3, 0]), project([value, 3, 0]),
                 (35, 44, 58), 1)
        cv2.line(panel, project([-3, value, 0]), project([3, value, 0]),
                 (35, 44, 58), 1)
    for endpoint, color, label in (([1, 0, 0], (60, 70, 255), "+X FRONT"),
                                   ([0, 1, 0], (60, 230, 90), "+Y LEFT"),
                                   ([0, 0, 1], (255, 120, 60), "+Z UP")):
        cv2.arrowedLine(panel, project([0, 0, 0]), project(endpoint),
                        color, 3, cv2.LINE_AA, tipLength=.12)
        cv2.putText(panel, label, project(endpoint),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    colors = ((80, 90, 255), (90, 230, 100), (255, 155, 80), (220, 100, 220))
    for index, person in enumerate(people):
        color = colors[index % len(colors)]
        for first, second in pose3d.BONES:
            a, b = person["joints"][first], person["joints"][second]
            if a["xyz_imu_m"] is not None and b["xyz_imu_m"] is not None:
                cv2.line(panel, project(a["xyz_imu_m"]),
                         project(b["xyz_imu_m"]), color, 4, cv2.LINE_AA)
        for joint in person["joints"]:
            if joint["xyz_imu_m"] is not None:
                cv2.circle(panel, project(joint["xyz_imu_m"]), 5,
                           (40, 235, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, "SPARSE STEREO 3D / BASE_LINK", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .62, (235, 240, 250), 2,
                cv2.LINE_AA)
    cv2.putText(panel, "XY GROUND / +Z UP", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (150, 205, 255), 1,
                cv2.LINE_AA)
    return panel


class WebState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = {}


HTML = """<!doctype html><html lang=zh-CN><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'><title>四向双目三维骨架</title>
<style>body{margin:0;background:#080c12;color:#eef4ff;font-family:system-ui;text-align:center}header{position:sticky;top:0;background:#101722ee;padding:9px}h1{font-size:18px;margin:0 0 4px}#s{font-size:13px;color:#b8c6d9}img{max-width:100%;height:auto}</style></head><body><header><h1>四向双目 + HITNet校验 + 三维骨架</h1><div id=s>等待数据</div></header><img id=v><script>
const v=document.querySelector('#v'),s=document.querySelector('#s');async function tick(){let t=Date.now();v.src='/frame.jpg?t='+t;try{let d=await(await fetch('/status.json?t='+t)).json();s.textContent=`输入 ${d.input_hz.toFixed(2)} Hz · 处理 ${d.processing_hz.toFixed(2)} Hz · 人体 ${d.people} · 三角化 ${d.triangulated_joints} · HITNet回退 ${d.hitnet_fallback_joints} · 三维关节 ${d.valid_3d_joints} · ${d.total_ms.toFixed(0)} ms`;}catch(e){}setTimeout(tick,150)}tick();</script></body></html>"""


class DisplayWorker:
    def __init__(self, state, display_hz):
        self.state = state
        self.period = 1.0 / max(.2, display_hz)
        self.queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, payload):
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self):
        last = 0.0
        while not rospy.is_shutdown():
            try:
                payload = self.queue.get(timeout=.5)
            except queue.Empty:
                continue
            remaining = self.period - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
            if payload.get("anchor_mode", False):
                anchor_views = [annotate_anchor(
                    payload["anchors"][index],
                    payload["anchor_people"][index],
                    payload["raw_by_sector"][index],
                    ANCHOR_COLORS[index]) for index in range(4)]
                image_panel = make_mosaic(anchor_views)
            else:
                rows = [annotate_pair(
                    payload["left"][index], payload["right"][index],
                    payload["raw_by_sector"][index],
                    payload["left_people"][index],
                    payload["right_people"][index], PAIR_COLORS[index])
                    for index in range(4)]
                image_panel = np.vstack(rows)
            panel = draw_3d(payload["people"], 640, image_panel.shape[0])
            combined = np.hstack((image_panel, panel))
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 30),
                          (5, 8, 12), -1)
            cv2.putText(combined, payload["banner"], (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .46, (80, 255, 180), 1,
                        cv2.LINE_AA)
            display = cv2.resize(combined, (960, 720),
                                 interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with self.state.lock:
                    self.state.jpeg = encoded.tobytes()
                    self.state.status = dict(payload["status"])
            last = time.monotonic()


def start_web_server(state, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            return

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                data, content_type = HTML.encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/status.json":
                with state.lock:
                    data = json.dumps(state.status, ensure_ascii=False).encode("utf-8")
                content_type = "application/json"
            elif path == "/frame.jpg":
                with state.lock:
                    data = state.jpeg
                if data is None:
                    self.send_error(503)
                    return
                content_type = "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    args = parse_args()
    rospy.init_node("omninxt_sparse_stereo_pose", anonymous=False)
    with open(os.path.join(args.config_dir, "fisheye_cams.yaml")) as stream:
        import yaml
        cameras = yaml.safe_load(stream)
    geometries = [pose3d.pair_geometry(args.config_dir, pair, cameras)
                  for pair in pose3d.PAIR_DEFS]
    anchor_focal = args.anchor_width / (2.0 * math.tan(
        math.radians(args.anchor_fov) / 2.0))
    detector = TensorRTYOLOX(
        args.det_engine, model_input_size=(416, 416),
        det_mode="multiclass", nms_thr=.45, score_thr=args.det_threshold)
    estimator = TensorRTRTMPose(
        args.pose_engine, model_input_size=(192, 256))
    bridge = InputBridge()
    publishers = {
        "poses": rospy.Publisher(
            "/omninxt_pose/joints_3d", PoseArray, queue_size=1),
        "markers": rospy.Publisher(
            "/omninxt_pose/skeleton_markers", MarkerArray, queue_size=1),
        "status": rospy.Publisher(
            "/omninxt_pose/status", String, queue_size=1),
        "skeleton": rospy.Publisher(
            "/omninxt_pose/skeleton_frame", String, queue_size=1),
    }
    tracker = SkeletonTracker()
    boxes_anchor = np.empty((0, 4), dtype=np.float32)
    boxes_left = np.empty((0, 4), dtype=np.float32)
    boxes_right = np.empty((0, 4), dtype=np.float32)
    frame_index = 0
    processing_times = deque(maxlen=30)
    input_times = deque(maxlen=30)
    web_state = WebState()
    display = None
    server = None
    backend_sender = None
    if args.backend_host:
        backend_sender = SkeletonTcpSender(
            args.backend_host, args.backend_port)
        print("Skeleton TCP backend: {}:{}".format(
            args.backend_host, args.backend_port), flush=True)
    if not args.no_web:
        display = DisplayWorker(web_state, args.display_hz)
        server = start_web_server(web_state, args.web_port)
        print("Web viewer: http://127.0.0.1:{}".format(args.web_port),
              flush=True)
    while not rospy.is_shutdown():
        try:
            stamp_ns, frame = bridge.frames.get(timeout=.5)
        except queue.Empty:
            continue
        started = time.monotonic()
        left_images = [frame[(index, "left")] for index in range(4)]
        right_images = [frame[(index, "right")] for index in range(4)]
        input_times.append(started)
        depths, depth_stamps = bridge.depth_snapshot()
        raw_people = []
        raw_by_sector = [[] for _ in range(4)]
        match_counts = [0] * 4
        anchor_views = None
        anchor_people = [[] for _ in range(4)]
        left_people = [[] for _ in range(4)]
        right_people = [[] for _ in range(4)]
        det_left_ms = det_right_ms = pose_left_ms = pose_right_ms = 0.0
        run_detector = False
        rescue_detector = False
        rescue_camera = None
        anchor_boxes_before_dedup = 0
        anchor_boxes_after_dedup = 0
        if all((index, "anchor") in frame for index in range(4)):
            anchor_views = [frame[(index, "anchor")] for index in range(4)]
            run_detector = (frame_index % max(
                1, args.center_det_interval) == 0)
            if run_detector:
                detected_boxes, detector_ms = detect_single_view(
                    make_mosaic(anchor_views), detector)
                boxes_anchor = clip_quadrant_boxes(
                    detected_boxes, args.anchor_width, args.anchor_height)
                det_left_ms += detector_ms
            if not run_detector and frame_index % max(
                    1, args.rescue_det_interval) == 0:
                rescue_camera = ((frame_index // max(
                    1, args.rescue_det_interval)) % 4)
                has_camera_track = False
                for box in np.asarray(
                        boxes_anchor, dtype=np.float32).reshape(-1, 4):
                    center_x = float(box[0] + box[2]) * .5
                    center_y = float(box[1] + box[3]) * .5
                    sector = (int(center_y >= args.anchor_height) * 2 +
                              int(center_x >= args.anchor_width))
                    has_camera_track |= sector == rescue_camera
                if not has_camera_track:
                    rescue_boxes, rescue_ms = detect_single_view(
                        anchor_views[rescue_camera], detector)
                    boxes_anchor = replace_quadrant_boxes(
                        boxes_anchor, rescue_camera, rescue_boxes,
                        args.anchor_width, args.anchor_height)
                    det_left_ms += rescue_ms
                    rescue_detector = True
            anchor_boxes_before_dedup = len(boxes_anchor)
            boxes_anchor = deduplicate_anchor_boxes(
                boxes_anchor, cameras, anchor_focal,
                args.anchor_width, args.anchor_height)
            anchor_boxes_after_dedup = len(boxes_anchor)
            anchor_people, boxes_anchor, _, pose_left_ms = infer_view(
                make_mosaic(anchor_views), None, estimator, boxes_anchor,
                False, args.anchor_width, args.anchor_height)
            for camera_id in range(4):
                for pose_person in anchor_people[camera_id]:
                    if not anchor_person_is_valid(
                            pose_person, args.person_threshold):
                        continue
                    person = make_anchor_person(
                        camera_id, pose_person, geometries,
                        left_images, right_images, depths, depth_stamps,
                        stamp_ns, anchor_focal, args.anchor_width,
                        args.anchor_height, args)
                    if person_center(person) is None:
                        continue
                    raw_people.append(person)
                    raw_by_sector[camera_id].append(person)
        else:
            # Exact-stamp raw data should normally be present. Retain the old
            # rectified-sector path as a safe fallback during startup.
            run_detector = frame_index % max(1, args.det_interval) == 0
            left_people, boxes_left, det_left_ms, pose_left_ms = infer_view(
                make_mosaic(left_images), detector, estimator, boxes_left,
                run_detector)
            if run_detector or len(boxes_right) == 0:
                boxes_right = seed_right_boxes(boxes_left)
            right_people, boxes_right, det_right_ms, pose_right_ms = infer_view(
                make_mosaic(right_images), None, estimator,
                boxes_right, False)
            for sector in range(4):
                matches = associate_people(
                    left_people[sector], right_people[sector],
                    args.keypoint_threshold)
                match_counts[sector] = len(matches)
                for left_index, left_person in enumerate(left_people[sector]):
                    if not pose3d.person_is_valid(
                            left_person["scores"], args.person_threshold):
                        continue
                    right_person = None
                    if left_index in matches:
                        right_person = right_people[sector][matches[left_index]]
                    person = make_stereo_person(
                        sector, left_person, right_person, geometries[sector],
                        left_images[sector], right_images[sector],
                        depths[sector], depth_stamps[sector], stamp_ns, args)
                    raw_people.append(person)
                    raw_by_sector[sector].append(person)
        fused = fuse_people(raw_people, args.merge_distance)
        tracked = tracker.update(fused, stamp_ns)
        publish_ros(publishers, tracked, stamp_ns)
        processing_times.append(time.monotonic())
        processing_hz = 0.0 if len(processing_times) < 2 else \
            (len(processing_times) - 1) / \
            (processing_times[-1] - processing_times[0])
        input_hz = 0.0 if len(input_times) < 2 else \
            (len(input_times) - 1) / (input_times[-1] - input_times[0])
        all_joints = [joint for person in raw_people
                      for joint in person["joints"]]
        valid_joints = [joint for person in tracked
                        for joint in person["joints"]
                        if joint["xyz_imu_m"] is not None]
        total_ms = (time.monotonic() - started) * 1000.0
        status = {
            "stamp_ns": stamp_ns,
            "input_hz": input_hz,
            "processing_hz": processing_hz,
            "people": len(tracked),
            "raw_sector_people": len(raw_people),
            "front_end": ("camera_center_anchor" if anchor_views is not None
                          else "rectified_sector_fallback"),
            "anchor_people_by_camera": [len(values)
                                         for values in anchor_people],
            "anchor_boxes_before_dedup": anchor_boxes_before_dedup,
            "anchor_boxes_after_dedup": anchor_boxes_after_dedup,
            "left_people_by_sector": [len(values) for values in left_people],
            "right_people_by_sector": [len(values) for values in right_people],
            "stereo_matches_by_sector": match_counts,
            "valid_3d_joints": len(valid_joints),
            "triangulated_joints": sum(
                (joint["source"].startswith("stereo_") or
                 joint["source"] == "anchor_epipolar")
                for joint in all_joints),
            "hitnet_fallback_joints": sum(
                joint["source"].startswith("hitnet_") for joint in all_joints),
            "hitnet_consistent_joints": sum(
                joint.get("hitnet_consistent") is True for joint in all_joints),
            "kinematic_rejected_joints": sum(
                person.get("kinematic_rejected_joints", 0)
                for person in raw_people),
            "detection_ms": det_left_ms + det_right_ms,
            "pose_ms": pose_left_ms + pose_right_ms,
            "pose_dynamic_batch": estimator.runner.dynamic,
            "pose_max_batch": estimator.runner.max_batch_size,
            "pose_last_batch_sizes": list(estimator.last_batch_sizes),
            "pose_last_person_count": estimator.last_person_count,
            "total_ms": total_ms,
            "detector_ran": run_detector,
            "rescue_detector_ran": rescue_detector,
            "rescue_camera": rescue_camera,
            "output_frame": "base_link",
            "runtime_flight_controller": False,
            "filter_mode": "body_frame_constant_velocity",
        }
        if backend_sender is not None:
            status["backend_stream"] = backend_sender.status()
        else:
            status["backend_stream"] = {"enabled": False}
        skeleton_packet = build_skeleton_packet(
            tracked, stamp_ns, frame_index, pose3d.JOINTS)
        skeleton_json = json.dumps(
            skeleton_packet, ensure_ascii=False, separators=(",", ":"))
        publishers["skeleton"].publish(String(data=skeleton_json))
        if backend_sender is not None:
            backend_sender.submit(skeleton_packet)
        publishers["status"].publish(String(
            data=json.dumps({"status": status, "people": tracked},
                            ensure_ascii=False, separators=(",", ":"))))
        if display is not None:
            banner = ("INPUT {:.2f}Hz  PROCESS {:.2f}Hz  PEOPLE {}  "
                      "TRI {}  HITNET {}  {:.0f}ms").format(
                          input_hz, processing_hz, len(tracked),
                          status["triangulated_joints"],
                          status["hitnet_fallback_joints"], total_ms)
            display.submit({
                "left": left_images, "right": right_images,
                "raw_by_sector": raw_by_sector, "people": tracked,
                "left_people": left_people, "right_people": right_people,
                "anchor_mode": anchor_views is not None,
                "anchors": anchor_views,
                "anchor_people": anchor_people,
                "status": status, "banner": banner,
            })
        if frame_index % 10 == 0:
            rospy.loginfo(
                "stereo-pose input=%.2fHz process=%.2fHz people=%d "
                "tri=%d hitnet_fallback=%d valid=%d total=%.1fms",
                input_hz, processing_hz, len(tracked),
                status["triangulated_joints"],
                status["hitnet_fallback_joints"], len(valid_joints), total_ms)
        frame_index += 1
    if server is not None:
        server.shutdown()


if __name__ == "__main__":
    main()
