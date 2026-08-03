#!/usr/bin/env python3
"""Check one still image with the Kalibr/TartanCalib AprilGrid detector."""

import argparse
import json

import cv2
import numpy as np

import aslam_cv as acv
from kalibr_camera_calibration.CameraCalibrator import TargetDetector
from kalibr_common.ConfigReader import AslamCamera, CalibrationTargetParameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--overlay", required=True)
    args = parser.parse_args()

    color = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    camera = AslamCamera(
        "pinhole",
        [500.0, 500.0, width / 2.0, height / 2.0],
        "radtan",
        [0.0, 0.0, 0.0, 0.0],
        [width, height],
    )
    target = CalibrationTargetParameters(args.target)
    target_params = target.getTargetParams()
    detector = TargetDetector(target, camera.geometry)
    success, observation = detector.detector.findTargetNoTransformation(
        acv.Time(0.0), gray
    )

    corner_ids = list(observation.getCornersIdx())
    corner_points = np.asarray(observation.getCornersImageFrame()).reshape(-1, 2)
    overlay = color.copy()
    for corner_id, (x_coord, y_coord) in zip(corner_ids, corner_points):
        point = (int(round(x_coord)), int(round(y_coord)))
        cv2.circle(overlay, point, 3, (0, 255, 0), -1, cv2.LINE_AA)
        if corner_id % 4 == 0:
            cv2.putText(
                overlay,
                str(corner_id),
                (point[0] + 4, point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(args.overlay, overlay)

    expected_corners = (
        int(target_params["tagRows"]) * int(target_params["tagCols"]) * 4
    )
    result = {
        "image": args.image,
        "kalibr_success": bool(success),
        "detected_corners": len(corner_ids),
        "expected_corners": expected_corners,
        "detection_ratio": round(len(corner_ids) / expected_corners, 4),
        "gray_mean": round(float(gray.mean()), 3),
        "gray_stddev": round(float(gray.std()), 3),
        "white_clip_percent_ge_250": round(
            float(np.count_nonzero(gray >= 250)) * 100.0 / gray.size, 3
        ),
        "black_clip_percent_le_5": round(
            float(np.count_nonzero(gray <= 5)) * 100.0 / gray.size, 3
        ),
        "laplacian_variance": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3
        ),
        "overlay": args.overlay,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
