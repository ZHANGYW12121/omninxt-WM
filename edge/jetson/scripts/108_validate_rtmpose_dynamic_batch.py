#!/usr/bin/env python3
"""Validate RTMPose TensorRT batch shapes and report synchronous latency."""

import argparse
import json
import time

import numpy as np

from trt_rtmpose import TensorRTRTMPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def boxes_for_count(count):
    boxes = []
    for index in range(count):
        col = index % 4
        row = index // 4
        x0 = 10 + col * 200
        y0 = 10 + row * 300
        boxes.append([x0, y0, x0 + 150, y0 + 260])
    return np.asarray(boxes, dtype=np.float32)


def main():
    args = parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if any(value <= 0 for value in batches) or args.iterations <= 0:
        raise ValueError("Batches and iterations must be positive")
    estimator = TensorRTRTMPose(
        args.engine, model_input_size=(192, 256))
    # Deterministic textured input exercises the complete affine/normalization
    # path without requiring a recorded person image.
    yy, xx = np.indices((640, 832), dtype=np.uint16)
    image = np.stack(((xx + yy) % 256, (2 * xx + yy) % 256,
                      (xx + 2 * yy) % 256), axis=2).astype(np.uint8)
    results = []
    for batch in batches:
        if batch > estimator.runner.max_batch_size:
            raise ValueError("Requested batch {} exceeds engine max {}".format(
                batch, estimator.runner.max_batch_size))
        boxes = boxes_for_count(batch)
        # First call includes lazy library/kernel effects and is not timed.
        keypoints, scores = estimator(image, bboxes=boxes)
        if keypoints.shape != (batch, 17, 2) or scores.shape != (batch, 17):
            raise RuntimeError("Unexpected outputs {} {}".format(
                keypoints.shape, scores.shape))
        timings = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            keypoints, scores = estimator(image, bboxes=boxes)
            timings.append((time.perf_counter() - started) * 1000.0)
        batch_calls = list(estimator.last_batch_sizes)
        individual_keypoints = []
        individual_scores = []
        for box in boxes:
            one_keypoints, one_scores = estimator(
                image, bboxes=np.asarray([box], dtype=np.float32))
            individual_keypoints.append(one_keypoints)
            individual_scores.append(one_scores)
        individual_keypoints = np.concatenate(individual_keypoints, axis=0)
        individual_scores = np.concatenate(individual_scores, axis=0)
        results.append({
            "batch": batch,
            "batch_calls": batch_calls,
            "keypoints_shape": list(keypoints.shape),
            "scores_shape": list(scores.shape),
            "max_keypoint_delta_vs_batch1_px": round(float(np.max(
                np.abs(keypoints - individual_keypoints))), 6),
            "max_score_delta_vs_batch1": round(float(np.max(
                np.abs(scores - individual_scores))), 6),
            "mean_end_to_end_ms": round(float(np.mean(timings)), 3),
            "p95_end_to_end_ms": round(float(np.percentile(timings, 95)), 3),
        })
    print(json.dumps({
        "dynamic": estimator.runner.dynamic,
        "min_input_shape": list(estimator.runner.min_input_shape),
        "opt_input_shape": list(estimator.runner.opt_input_shape),
        "max_input_shape": list(estimator.runner.max_input_shape),
        "results": results,
        "status": "RTMPOSE_BATCH_OK",
    }, indent=2))


if __name__ == "__main__":
    main()
