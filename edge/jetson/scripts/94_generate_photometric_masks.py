#!/usr/bin/env python3
"""Generate and validate per-camera flat-field masks for OmniDepth."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")
TARGET_GRAY = 0.7 * 255.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--blur-sigma", type=float, default=25.0)
    return parser.parse_args()


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def region_stats(image, valid):
    values = image[valid]
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "cv": round(float(np.std(values) / max(np.mean(values), 1e-6)), 6),
        "p01": round(float(np.percentile(values, 1)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "p99": round(float(np.percentile(values, 99)), 4),
    }


def make_preview(before, after, valid, label):
    left = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
    invalid = ~valid
    left[invalid] = (0.35 * left[invalid]).astype(np.uint8)
    right[invalid] = (0.35 * right[invalid]).astype(np.uint8)
    cv2.putText(left, f"{label} raw", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    cv2.putText(right, f"{label} corrected", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    return np.hstack((left, right))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    expected_root = Path("/home/neu/OmniNxt/runtime/calibration/photometric").resolve()
    if expected_root not in run_dir.parents or not run_dir.name.startswith("run_"):
        raise RuntimeError(f"Refusing unexpected run path: {run_dir}")
    if args.blur_sigma < 5 or args.blur_sigma > 100:
        raise ValueError("--blur-sigma must be between 5 and 100 pixels")

    output_dir = run_dir / "generated"
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_dir": str(run_dir),
        "target_gray": TARGET_GRAY,
        "blur_sigma": args.blur_sigma,
        "camera_mapping": {
            "CAM_A": "cam_0",
            "CAM_B": "cam_1",
            "CAM_C": "cam_2",
            "CAM_D": "cam_3",
        },
        "cameras": {},
        "warnings": [],
        "failures": [],
    }
    previews = []

    for index, camera in enumerate(CAMERAS):
        source_dir = run_dir / "raw" / camera
        flat_path = source_dir / "flat_median.png"
        dark_path = source_dir / "dark_median.png"
        if not flat_path.is_file() or not dark_path.is_file():
            summary["failures"].append(f"{camera}: missing flat or dark median")
            continue

        flat = cv2.imread(str(flat_path), cv2.IMREAD_GRAYSCALE)
        dark = cv2.imread(str(dark_path), cv2.IMREAD_GRAYSCALE)
        if flat is None or dark is None or flat.shape != (720, 1280) or dark.shape != flat.shape:
            summary["failures"].append(f"{camera}: expected matching 1280x720 images")
            continue

        response = np.maximum(flat.astype(np.float32) - dark.astype(np.float32), 0.0)
        threshold = max(6.0, float(np.percentile(response, 99)) * 0.04)
        valid = largest_component(response > threshold)
        valid = cv2.morphologyEx(
            valid.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ).astype(bool)
        valid_fraction = float(np.mean(valid))
        if valid_fraction < 0.20:
            summary["failures"].append(
                f"{camera}: illuminated valid region is only {valid_fraction:.1%}"
            )
            continue

        weighted = response.copy()
        fill_value = float(np.median(response[valid]))
        weighted[~valid] = fill_value
        smooth = cv2.GaussianBlur(
            weighted,
            (0, 0),
            sigmaX=args.blur_sigma,
            sigmaY=args.blur_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        smooth = np.clip(smooth, 8.0, 255.0)

        # The runtime computes gain = 0.7 / (mask / 255).
        mask = np.rint(smooth).astype(np.uint8)
        mask[~valid] = int(round(TARGET_GRAY))
        gain = TARGET_GRAY / np.maximum(mask.astype(np.float32), 1.0)
        corrected = np.clip(flat.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        raw_stats = region_stats(flat.astype(np.float32), valid)
        corrected_stats = region_stats(corrected.astype(np.float32), valid)
        dark_stats = region_stats(dark.astype(np.float32), valid)
        gain_stats = region_stats(gain, valid)
        improvement = 1.0 - corrected_stats["cv"] / max(raw_stats["cv"], 1e-9)

        camera_warnings = []
        if raw_stats["p99"] >= 248:
            camera_warnings.append("flat field contains saturation")
        if raw_stats["p50"] < 80:
            camera_warnings.append("flat field is too dark")
        if dark_stats["p99"] > 20:
            camera_warnings.append("dark field is not fully dark")
        if gain_stats["p99"] > 4.0:
            camera_warnings.append("required correction gain exceeds 4x")
        if corrected_stats["cv"] > 0.15:
            camera_warnings.append("corrected flat-field CV remains above 15%")
        if improvement < 0.20:
            camera_warnings.append("flat-field nonuniformity improves by less than 20%")

        mask_path = masks_dir / f"cam_{index}_vig_mask.png"
        valid_path = masks_dir / f"cam_{index}_valid_region.png"
        preview_path = previews_dir / f"{camera}_raw_vs_corrected.png"
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(valid_path), valid.astype(np.uint8) * 255)
        preview = make_preview(flat, corrected, valid, camera)
        cv2.imwrite(str(preview_path), preview)
        previews.append(cv2.resize(preview, (1280, 360)))

        summary["cameras"][camera] = {
            "mask": str(mask_path),
            "valid_fraction": round(valid_fraction, 6),
            "response_threshold": round(threshold, 4),
            "raw": raw_stats,
            "dark": dark_stats,
            "gain": gain_stats,
            "corrected": corrected_stats,
            "cv_improvement_fraction": round(improvement, 6),
            "warnings": camera_warnings,
        }
        summary["warnings"].extend(f"{camera}: {item}" for item in camera_warnings)

    if len(summary["cameras"]) != 4:
        summary["status"] = "FAIL"
    elif summary["failures"]:
        summary["status"] = "FAIL"
    elif summary["warnings"]:
        summary["status"] = "WARN"
    else:
        summary["status"] = "PASS"

    if previews:
        cv2.imwrite(str(output_dir / "all_cameras_raw_vs_corrected.png"), np.vstack(previews))
    summary_path = output_dir / "photometric_validation.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nValidation: {summary_path}")
    print(f"Preview: {output_dir / 'all_cameras_raw_vs_corrected.png'}")
    if summary["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
