#!/usr/bin/env python3
"""Cache multi-person RTMPose keypoints for Isaac crowd records.

The raw dataset is treated as read-only.  Derived pose files are written to::

    <out_root>/<record_name>/frame_000000_pose.npz

Each npz contains frame-local multi-person COCO-17 keypoints plus simple
nearest-neighbour track ids that are stable within one record when detections are
continuous.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.isaac_crowd import EpisodeInfo, build_episode_index, load_episode_index


FRAME_IMAGE_RE = re.compile(r"^frame_(\d{6})_camera\.jpg$")
cv2 = None
Body = None
draw_skeleton = None


def require_pose_dependencies() -> None:
    """Import heavy pose dependencies only when the cache command actually runs."""

    global cv2, Body, draw_skeleton
    if cv2 is not None and Body is not None and draw_skeleton is not None:
        return
    try:
        import cv2 as _cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cache_multiperson_pose.py needs opencv-python (cv2). "
            "Run it in the same environment you use for batch_pose_and_encode.py, "
            "or install opencv-python."
        ) from exc
    try:
        from rtmlib import Body as _Body
        from rtmlib import draw_skeleton as _draw_skeleton
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cache_multiperson_pose.py needs rtmlib for RTMPose inference."
        ) from exc
    cv2 = _cv2
    Body = _Body
    draw_skeleton = _draw_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Raw dataset root containing record_xxx folders. This directory is only read.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Derived pose cache root. New cache files are written here, not into the raw records.",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=None,
        help="Optional episode index from build_crowd_dataset_index.py. If omitted, scan data_root.",
    )
    parser.add_argument("--min_frames", type=int, default=30, help="Used only when --index_path is omitted.")
    parser.add_argument(
        "--exclude_termination",
        nargs="*",
        default=["stuck_timeout"],
        help="Used only when --index_path is omitted.",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Optional explicit record names to process.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Optional record limit for quick tests.")
    parser.add_argument("--max_frames_per_record", type=int, default=None, help="Optional frame limit per record.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="RTMPose ONNXRuntime device.")
    parser.add_argument(
        "--mode",
        choices=["performance", "lightweight", "balanced"],
        default="balanced",
        help="RTMPose model mode.",
    )
    parser.add_argument("--kpt_thr", type=float, default=0.3, help="Keypoint confidence threshold.")
    parser.add_argument(
        "--track_max_distance_px",
        type=float,
        default=120.0,
        help="Maximum center displacement for reusing a track id.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip frames whose pose npz already exists.")
    parser.add_argument("--save_vis", action="store_true", help="Also save visualization jpgs under out_root.")
    return parser.parse_args()


def safe_draw_skeleton(img, keypoints, scores, kpt_thr=0.3, openpose_skeleton=False):
    sig = inspect.signature(draw_skeleton)
    kwargs = {"kpt_thr": kpt_thr}
    if "openpose_skeleton" in sig.parameters:
        kwargs["openpose_skeleton"] = openpose_skeleton
    elif "to_openpose" in sig.parameters:
        kwargs["to_openpose"] = openpose_skeleton
    return draw_skeleton(img, keypoints, scores, **kwargs)


def list_record_images(frames_dir: Path) -> list[tuple[int, Path]]:
    images: list[tuple[int, Path]] = []
    for name in os.listdir(frames_dir):
        match = FRAME_IMAGE_RE.match(name)
        if match:
            images.append((int(match.group(1)), frames_dir / name))
    images.sort(key=lambda item: item[0])
    return images


def detections_to_xyc(keypoints, scores) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.ndim != 3 or scores.ndim != 2 or keypoints.shape[0] == 0:
        return np.zeros((0, 17, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if keypoints.shape[1] != 17:
        raise ValueError(f"Expected COCO-17 keypoints, got shape {keypoints.shape}")
    scores = scores[:, : keypoints.shape[1]]
    xyc = np.concatenate([keypoints[:, :, :2], scores[:, :, None]], axis=-1).astype(np.float32)
    xyc[~np.isfinite(xyc)] = 0.0
    person_scores = scores.mean(axis=1).astype(np.float32)
    return xyc, person_scores


def compute_bboxes_and_centers(xyc: np.ndarray, kpt_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_people = xyc.shape[0]
    bboxes = np.zeros((num_people, 4), dtype=np.float32)
    centers = np.zeros((num_people, 2), dtype=np.float32)
    valid = np.zeros((num_people,), dtype=np.bool_)
    for i in range(num_people):
        conf = xyc[i, :, 2]
        mask = conf >= kpt_thr
        if not mask.any():
            conf = xyc[i, :, 2]
            mask = conf > 0
        if not mask.any():
            continue
        xy = xyc[i, mask, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        bboxes[i] = np.asarray([mn[0], mn[1], mx[0], mx[1]], dtype=np.float32)
        centers[i] = (mn + mx) * 0.5
        valid[i] = True
    return bboxes, centers, valid


@dataclass
class TrackState:
    track_id: int
    center: np.ndarray


class NearestCenterTracker:
    """Tiny per-record tracker; good enough for continuous RTMPose detections."""

    def __init__(self, max_distance_px: float = 120.0) -> None:
        self.max_distance_px = float(max_distance_px)
        self.next_id = 1
        self.prev: list[TrackState] = []

    def update(self, centers: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        track_ids = np.full((len(centers),), -1, dtype=np.int32)
        candidates: list[tuple[float, int, int]] = []
        for prev_idx, prev in enumerate(self.prev):
            for det_idx, valid in enumerate(valid_mask):
                if not valid:
                    continue
                dist = float(np.linalg.norm(prev.center - centers[det_idx]))
                if math.isfinite(dist) and dist <= self.max_distance_px:
                    candidates.append((dist, prev_idx, det_idx))
        candidates.sort(key=lambda item: item[0])

        used_prev: set[int] = set()
        used_det: set[int] = set()
        for _, prev_idx, det_idx in candidates:
            if prev_idx in used_prev or det_idx in used_det:
                continue
            track_ids[det_idx] = self.prev[prev_idx].track_id
            used_prev.add(prev_idx)
            used_det.add(det_idx)

        for det_idx, valid in enumerate(valid_mask):
            if not valid:
                continue
            if track_ids[det_idx] < 0:
                track_ids[det_idx] = self.next_id
                self.next_id += 1

        self.prev = [
            TrackState(int(track_ids[i]), centers[i].copy())
            for i, valid in enumerate(valid_mask)
            if valid and track_ids[i] > 0
        ]
        return track_ids


def load_episodes(args: argparse.Namespace) -> list[EpisodeInfo]:
    if args.index_path is not None:
        episodes = load_episode_index(args.index_path)
    else:
        episodes = build_episode_index(
            args.data_root,
            min_frames=args.min_frames,
            exclude_termination=args.exclude_termination,
        )
    if args.records:
        wanted = set(args.records)
        episodes = [ep for ep in episodes if ep.record_name in wanted]
    if args.max_records is not None:
        episodes = episodes[: args.max_records]
    return episodes


def save_record_metadata(record_out: Path, record_name: str, args: argparse.Namespace, image_size: tuple[int, int] | None) -> None:
    payload = {
        "schema": "isaac_crowd_pose_cache_v1",
        "record_name": record_name,
        "keypoints": "COCO-17, [num_people, 17, x_y_confidence]",
        "track_ids": "simple nearest-center ids, stable only within this record when detections are continuous",
        "image_size_hw": list(image_size) if image_size is not None else None,
        "rtmpose_mode": args.mode,
        "rtmpose_device": args.device,
        "kpt_thr": args.kpt_thr,
        "track_max_distance_px": args.track_max_distance_px,
    }
    with (record_out / "pose_cache_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_record(ep: EpisodeInfo, pose_model: Body, args: argparse.Namespace) -> tuple[int, int]:
    frames_dir = Path(ep.record_dir) / "frames"
    record_out = args.out_root / ep.record_name
    record_out.mkdir(parents=True, exist_ok=True)
    vis_dir = record_out / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    images = list_record_images(frames_dir)
    if args.max_frames_per_record is not None:
        images = images[: args.max_frames_per_record]

    tracker = NearestCenterTracker(max_distance_px=args.track_max_distance_px)
    image_size: tuple[int, int] | None = None
    written = 0
    skipped = 0

    for frame_id, img_path in tqdm(images, desc=ep.record_name, leave=False):
        out_path = record_out / f"frame_{frame_id:06d}_pose.npz"
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Warning] Cannot read image: {img_path}")
            continue
        image_size = img.shape[:2]

        keypoints, scores = pose_model(img)
        xyc, person_scores = detections_to_xyc(keypoints, scores)
        bboxes, centers, valid_mask = compute_bboxes_and_centers(xyc, args.kpt_thr)
        track_ids = tracker.update(centers, valid_mask)

        rel_image_path = img_path.relative_to(Path(ep.record_dir)).as_posix()
        np.savez_compressed(
            out_path,
            keypoints_xyc=xyc.astype(np.float32),
            scores=xyc[:, :, 2].astype(np.float32),
            person_scores=person_scores.astype(np.float32),
            valid_mask=valid_mask.astype(np.bool_),
            track_ids=track_ids.astype(np.int32),
            bboxes_xyxy=bboxes.astype(np.float32),
            centers_xy=centers.astype(np.float32),
            frame_index=np.asarray(frame_id, dtype=np.int32),
            image_size_hw=np.asarray(image_size, dtype=np.int32),
            image_path=np.asarray(rel_image_path),
            record_name=np.asarray(ep.record_name),
        )
        written += 1

        if args.save_vis:
            vis = img.copy()
            if xyc.shape[0] > 0:
                try:
                    vis = safe_draw_skeleton(vis, xyc[:, :, :2], xyc[:, :, 2], kpt_thr=args.kpt_thr)
                except Exception as exc:  # noqa: BLE001
                    print(f"[Warning] draw_skeleton failed on {img_path.name}: {exc}")
                for det_idx, track_id in enumerate(track_ids):
                    if track_id <= 0 or not valid_mask[det_idx]:
                        continue
                    x1, y1, x2, y2 = bboxes[det_idx].astype(int).tolist()
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.putText(
                        vis,
                        f"id={track_id}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.imwrite(str(vis_dir / f"frame_{frame_id:06d}_pose.jpg"), vis)

    save_record_metadata(record_out, ep.record_name, args, image_size)
    return written, skipped


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser()
    args.out_root = args.out_root.expanduser()
    args.out_root.mkdir(parents=True, exist_ok=True)
    require_pose_dependencies()

    episodes = load_episodes(args)
    print(f"Raw dataset root: {args.data_root}")
    print(f"Pose cache root:  {args.out_root}")
    print(f"Records to process: {len(episodes)}")
    print("Loading RTMPose Body model...")
    pose_model = Body(
        to_openpose=False,
        mode=args.mode,
        backend="onnxruntime",
        device=args.device,
    )

    total_written = 0
    total_skipped = 0
    for ep in tqdm(episodes, desc="records"):
        written, skipped = process_record(ep, pose_model, args)
        total_written += written
        total_skipped += skipped
    print(f"Done. Written pose frames: {total_written}, skipped existing: {total_skipped}")


if __name__ == "__main__":
    main()
