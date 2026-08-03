import argparse
import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from rtmlib import Body, draw_skeleton
from modules.stgcn_lite import STGCNLiteEncoder


WORKSPACE_ROOT = Path(
    os.environ.get(
        "OMNINXT_WORKSPACE_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).expanduser().resolve()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames_dir",
        type=str,
        default=str(
            WORKSPACE_ROOT
            / "isaacsim/database/database/test/record_20260428_125536/frames"
        ),
        help="Folder containing frame_000000_camera.jpg images.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            WORKSPACE_ROOT / "r2dreamer/pose_outputs/record_20260428_125536"
        ),
        help="Output folder under r2dreamer.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="RTMPose inference device. Use cpu first if onnxruntime cuda reports libcudnn.so.9 missing.",
    )

    parser.add_argument(
        "--stgcn_device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="ST-GCN-lite encoding device.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["performance", "lightweight", "balanced"],
        help="RTMPose model mode.",
    )

    parser.add_argument(
        "--kpt_thr",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold for drawing.",
    )

    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save skeleton visualization images.",
    )

    parser.add_argument(
        "--out_dim",
        type=int,
        default=256,
        help="ST-GCN-lite output feature dimension.",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Number of past frames used to build one skeleton token.",
    )

    parser.add_argument(
        "--window_batch_size",
        type=int,
        default=64,
        help="Batch size for ST-GCN-lite window encoding.",
    )

    parser.add_argument(
        "--pad_mode",
        type=str,
        default="first",
        choices=["first", "current", "zero"],
        help=(
            "Padding mode for early frames. "
            "'first': repeat the first frame; "
            "'current': repeat current frame; "
            "'zero': use zero skeletons."
        ),
    )

    parser.add_argument(
        "--save_each_feature",
        action="store_true",
        help="If set, save each frame's skeleton token as a separate .npy file.",
    )

    return parser.parse_args()


def safe_draw_skeleton(img, keypoints, scores, kpt_thr=0.3, openpose_skeleton=False):
    """
    Compatible with different rtmlib versions.
    """
    sig = inspect.signature(draw_skeleton)
    kwargs = {"kpt_thr": kpt_thr}

    if "openpose_skeleton" in sig.parameters:
        kwargs["openpose_skeleton"] = openpose_skeleton
    elif "to_openpose" in sig.parameters:
        kwargs["to_openpose"] = openpose_skeleton

    return draw_skeleton(img, keypoints, scores, **kwargs)


def draw_joint_ids(img, keypoints, scores, kpt_thr=0.3):
    """
    Draw joint numbers 0~16 on image.
    """
    vis = img.copy()

    if keypoints.ndim != 3:
        return vis

    for person_id in range(keypoints.shape[0]):
        for joint_id in range(keypoints.shape[1]):
            x, y = keypoints[person_id, joint_id, :2]
            conf = scores[person_id, joint_id]

            if conf < kpt_thr:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            x, y = int(x), int(y)
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(joint_id),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return vis


def select_main_person(keypoints, scores, image_shape):
    """
    keypoints: [N, 17, 2]
    scores:    [N, 17]

    Return:
        skeleton_xyc: [17, 3]
    """
    if keypoints is None or scores is None:
        return np.zeros((17, 3), dtype=np.float32)

    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    if keypoints.ndim != 3 or scores.ndim != 2 or keypoints.shape[0] == 0:
        return np.zeros((17, 3), dtype=np.float32)

    # 优先选择平均置信度最高的人
    person_score = scores.mean(axis=1)
    best_id = int(np.argmax(person_score))

    xy = keypoints[best_id]          # [17, 2]
    conf = scores[best_id, :, None]  # [17, 1]

    skeleton_xyc = np.concatenate([xy, conf], axis=-1).astype(np.float32)

    # 清理异常值
    skeleton_xyc[~np.isfinite(skeleton_xyc)] = 0.0

    return skeleton_xyc


def build_past_windows(skeleton_seq, window_size=30, pad_mode="first"):
    """
    Build causal sliding windows.

    Args:
        skeleton_seq: [T, 17, 3]
        window_size: number of frames in each history window
        pad_mode:
            first   - repeat first frame for missing history
            current - repeat current frame for missing history
            zero    - use zero skeleton for missing history

    Returns:
        windows: [T, window_size, 17, 3]

    第 t 个窗口只包含:
        skeleton[t-window_size+1], ..., skeleton[t]
    不包含未来帧。
    """
    if skeleton_seq.ndim != 3:
        raise ValueError(f"Expected skeleton_seq shape [T, 17, 3], got {skeleton_seq.shape}")

    T, V, C = skeleton_seq.shape
    if T == 0:
        raise ValueError("Empty skeleton sequence.")

    windows = []

    first_frame = skeleton_seq[0:1]  # [1, 17, 3]

    for t in range(T):
        start = t - window_size + 1

        if start >= 0:
            window = skeleton_seq[start:t + 1]
        else:
            pad_len = -start

            if pad_mode == "first":
                pad = np.repeat(first_frame, pad_len, axis=0)
            elif pad_mode == "current":
                current_frame = skeleton_seq[t:t + 1]
                pad = np.repeat(current_frame, pad_len, axis=0)
            elif pad_mode == "zero":
                pad = np.zeros((pad_len, V, C), dtype=skeleton_seq.dtype)
            else:
                raise ValueError(f"Unknown pad_mode: {pad_mode}")

            window = np.concatenate([pad, skeleton_seq[0:t + 1]], axis=0)

        if window.shape[0] != window_size:
            raise RuntimeError(
                f"Bad window length at t={t}: expected {window_size}, got {window.shape[0]}"
            )

        windows.append(window)

    windows = np.stack(windows, axis=0).astype(np.float32)
    return windows


def encode_windows_with_stgcn(
    encoder,
    windows,
    device,
    batch_size=64,
):
    """
    Args:
        encoder: STGCNLiteEncoder
        windows: [T, W, 17, 3]
        device: torch device
        batch_size: batch size for encoding

    Returns:
        features: [T, D]

    对每个窗口:
        [W, 17, 3] -> ST-GCN-lite -> [W, D]
    只取最后一帧特征作为当前时刻 token:
        token_t = y[-1]
    """
    encoder.eval()

    all_feats = []

    with torch.no_grad():
        for i in tqdm(range(0, len(windows), batch_size), desc="Encoding windows"):
            batch = windows[i:i + batch_size]              # [B, W, 17, 3]
            x = torch.from_numpy(batch).float().to(device)

            y = encoder(x)                                # [B, W, D]
            y_last = y[:, -1, :]                          # [B, D]

            all_feats.append(y_last.detach().cpu().numpy())

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    return features


def main():
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)

    keypoints_dir = out_dir / "keypoints"
    vis_dir = out_dir / "vis"
    encoded_dir = out_dir / "encoded"
    windows_dir = out_dir / "windows"
    feature_each_dir = encoded_dir / f"features_window{args.window_size}_each"

    out_dir.mkdir(parents=True, exist_ok=True)
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.save_each_feature:
        feature_each_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(frames_dir.glob("frame_*_camera.jpg"))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No frame_*_camera.jpg found in {frames_dir}")

    print(f"Found {len(image_paths)} images.")
    print(f"Input folder:  {frames_dir}")
    print(f"Output folder: {out_dir}")

    # 读取第一张图，确定图像大小
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_paths[0]}")
    H, W = first_img.shape[:2]
    print(f"Image size: H={H}, W={W}")

    openpose_skeleton = False

    # RTMPose Body 通常输出 COCO 17 点，适合后续 ST-GCN-lite
    print("Loading RTMPose Body model...")
    pose_model = Body(
        to_openpose=openpose_skeleton,
        mode=args.mode,
        backend="onnxruntime",
        device=args.device,
    )

    all_skeletons = []
    valid_flags = []
    frame_names = []

    print("Running RTMPose on all frames...")
    for img_path in tqdm(image_paths, desc="RTMPose"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Warning] Cannot read image: {img_path}")
            skeleton_xyc = np.zeros((17, 3), dtype=np.float32)
            all_skeletons.append(skeleton_xyc)
            valid_flags.append(False)
            frame_names.append(img_path.name)
            continue

        keypoints, scores = pose_model(img)

        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)

        skeleton_xyc = select_main_person(keypoints, scores, img.shape)
        is_valid = bool(skeleton_xyc[:, 2].mean() > 0.05)

        all_skeletons.append(skeleton_xyc)
        valid_flags.append(is_valid)
        frame_names.append(img_path.name)

        stem = img_path.stem
        np.save(keypoints_dir / f"{stem}_skeleton_xyc.npy", skeleton_xyc)

        if args.save_vis:
            vis = img.copy()

            if keypoints.ndim == 3 and keypoints.shape[0] > 0:
                try:
                    vis = safe_draw_skeleton(
                        vis,
                        keypoints,
                        scores,
                        kpt_thr=args.kpt_thr,
                        openpose_skeleton=openpose_skeleton,
                    )
                except Exception as e:
                    print(f"[Warning] draw_skeleton failed on {img_path.name}: {e}")

                vis = draw_joint_ids(vis, keypoints, scores, kpt_thr=args.kpt_thr)

            cv2.imwrite(str(vis_dir / f"{stem}_pose.jpg"), vis)

    skeleton_seq = np.stack(all_skeletons, axis=0).astype(np.float32)  # [T, 17, 3]
    valid_flags = np.asarray(valid_flags, dtype=np.bool_)
    frame_names = np.asarray(frame_names)

    np.save(out_dir / "skeleton_sequence_xyc.npy", skeleton_seq)
    np.save(out_dir / "valid_flags.npy", valid_flags)
    np.save(out_dir / "frame_names.npy", frame_names)

    print("Saved skeleton sequence:")
    print(f"  {out_dir / 'skeleton_sequence_xyc.npy'}")
    print(f"  shape = {skeleton_seq.shape}")
    print(f"  valid frames = {valid_flags.sum()} / {len(valid_flags)}")

    # 构建过去窗口
    print("Building causal skeleton windows...")
    windows = build_past_windows(
        skeleton_seq=skeleton_seq,
        window_size=args.window_size,
        pad_mode=args.pad_mode,
    )  # [T, window_size, 17, 3]

    window_path = windows_dir / f"skeleton_windows_{args.window_size}_{args.pad_mode}.npy"
    np.save(window_path, windows)

    print("Saved skeleton windows:")
    print(f"  {window_path}")
    print(f"  shape = {windows.shape}")
    print(
        "  meaning = [num_frames, history_window, num_joints, x_y_conf], "
        "each window only uses past and current frames."
    )

    # ST-GCN-lite 编码
    stgcn_device = args.stgcn_device
    if stgcn_device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available for PyTorch. Use CPU for ST-GCN-lite.")
        stgcn_device = "cpu"

    print("Building ST-GCN-lite encoder...")
    encoder = STGCNLiteEncoder(
        num_joints=17,
        in_channels=3,
        hidden_channels=64,
        out_dim=args.out_dim,
        image_size=(H, W),
        dropout=0.0,
    ).to(stgcn_device)

    print(
        f"Encoding skeleton windows: window_size={args.window_size}, "
        f"batch_size={args.window_batch_size}, device={stgcn_device}"
    )

    feat = encode_windows_with_stgcn(
        encoder=encoder,
        windows=windows,
        device=stgcn_device,
        batch_size=args.window_batch_size,
    )  # [T, D]

    feature_path = encoded_dir / f"stgcn_lite_features_window{args.window_size}_{args.pad_mode}.npy"
    np.save(feature_path, feat)

    print("Saved ST-GCN-lite window features:")
    print(f"  {feature_path}")
    print(f"  shape = {feat.shape}")
    print(
        "  meaning = [num_frames, feature_dim], "
        "feature[t] is encoded from skeleton[t-window+1 : t+1]."
    )

    # 可选：每一帧 token 单独保存
    if args.save_each_feature:
        print("Saving each frame token separately...")
        for i, name in enumerate(frame_names):
            stem = Path(str(name)).stem
            np.save(feature_each_dir / f"{stem}_stgcn_token.npy", feat[i])

        print(f"Saved per-frame tokens to: {feature_each_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
