#!/usr/bin/env python3
"""Run the current TensorRT HITNet on image pairs shipped by official D2SLAM."""

import argparse
import html
import json
import os
import shutil
import subprocess
from datetime import datetime

import cv2
import numpy as np


ROOT = "/home/neu/OmniNxt"
SAMPLES = (
    (
        "official_quadcam_rect",
        "Official quadcam_depth_est rectified pair",
        f"{ROOT}/source/D2SLAM/quadcam_depth_est/rect_l.png",
        f"{ROOT}/source/D2SLAM/quadcam_depth_est/rect_r.png",
    ),
    (
        "official_d2frontend_stereo",
        "Official D2Frontend indoor stereo pair",
        f"{ROOT}/source/D2SLAM/d2frontend/sample_images/left.jpg",
        f"{ROOT}/source/D2SLAM/d2frontend/sample_images/right.jpg",
    ),
)


def percentile(values, q):
    if values.size == 0:
        return None
    return round(float(np.percentile(values, q)), 6)


def prepare_pair(left_path, right_path):
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise RuntimeError(f"Could not read official pair: {left_path}, {right_path}")
    # This exactly follows the historical official HitnetONNX::inference:
    # grayscale first, then cv::resize to the fixed network input size.
    left = cv2.resize(left, (320, 240), interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, (320, 240), interpolation=cv2.INTER_LINEAR)
    return left, right


def photometric_consistency(left, right, disparity):
    height, width = left.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    right_x = grid_x - disparity.astype(np.float32)
    valid = (
        np.isfinite(disparity)
        & (disparity > 0)
        & (right_x >= 0)
        & (right_x <= width - 1)
    )
    warped = cv2.remap(
        right,
        right_x,
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    error = cv2.absdiff(left, warped)
    zero_error = cv2.absdiff(left, right)
    values = error[valid].astype(np.float32)
    zero_values = zero_error[valid].astype(np.float32)
    return warped, error, valid, {
        "valid_fraction": round(float(np.mean(valid)), 8),
        "warped_mae": round(float(np.mean(values)), 6) if values.size else None,
        "warped_median": percentile(values, 50),
        "warped_p90": percentile(values, 90),
        "zero_disparity_mae_same_region": (
            round(float(np.mean(zero_values)), 6) if zero_values.size else None
        ),
        "mae_improvement_over_zero_disparity": (
            round(float(np.mean(zero_values) - np.mean(values)), 6)
            if values.size
            else None
        ),
    }


def label(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 29), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def write_html(output, metadata):
    cards = []
    for item in metadata["samples"]:
        name = item["name"]
        cards.append(
            f"""
<section>
  <h2>{html.escape(item["description"])}</h2>
  <div class="grid">
    <figure><img src="{name}_left.png"><figcaption>官方左图，按官方预处理缩放</figcaption></figure>
    <figure><img src="{name}_right.png"><figcaption>官方右图，按官方预处理缩放</figcaption></figure>
    <figure><img src="{name}_disparity_color.png"><figcaption>当前HITNet视差，固定0–96 px</figcaption></figure>
    <figure><img src="{name}_warp_error.png"><figcaption>按视差回采右图后的亮度误差</figcaption></figure>
  </div>
  <pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre>
</section>"""
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>HITNet官方样例隔离测试</title>
<style>
body{{margin:0;background:#101827;color:#e5e7eb;font-family:system-ui,sans-serif}}
main{{max-width:1400px;margin:auto;padding:20px}}
section{{background:#182235;margin:20px 0;padding:16px;border-radius:10px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
figure{{margin:0}} img{{width:100%;background:#000}}
figcaption{{background:#0f172a;padding:7px;text-align:center}}
pre{{white-space:pre-wrap}} @media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>当前HITNet在D2SLAM官方样例上的隔离测试</h1>
<p>使用当前Orin本机TensorRT engine；不经过本机鱼眼标定、点云坐标变换、
有效性过滤或光度标定。输入预处理复现官方历史HitnetONNX实现。</p>
<p><a href="overview.png">打开总览PNG</a></p>
{''.join(cards)}
</main></body></html>"""
    with open(os.path.join(output, "viewer.html"), "w", encoding="utf-8") as stream:
        stream.write(document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.realpath(
        args.output
        or f"{ROOT}/runtime/hitnet_official_samples/test_{stamp}"
    )
    runtime_root = os.path.realpath(f"{ROOT}/runtime")
    if not output.startswith(runtime_root + os.sep):
        raise ValueError("Output must be below the OmniNxt runtime directory")
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite: {output}")
    os.makedirs(output)

    prepared = []
    # Four-stream runtime requires four inputs. Repeat each independent official
    # pair once; equality of repeated outputs also checks stream determinism.
    assignments = (SAMPLES[0], SAMPLES[1], SAMPLES[0], SAMPLES[1])
    for index, sample in enumerate(assignments):
        name, _, left_path, right_path = sample
        left, right = prepare_pair(left_path, right_path)
        prefix = os.path.join(output, f"pair{index}")
        cv2.imwrite(prefix + "_left.png", left)
        cv2.imwrite(prefix + "_right.png", right)
        prepared.append(prefix)

    for name, _, left_path, right_path in SAMPLES:
        shutil.copy2(left_path, os.path.join(output, name + "_original_left" + os.path.splitext(left_path)[1]))
        shutil.copy2(right_path, os.path.join(output, name + "_original_right" + os.path.splitext(right_path)[1]))

    container_output = "/runtime/" + os.path.relpath(output, runtime_root)
    engine = (
        "/root/swarm_ws/src/D2SLAM/models/hitnet_series/"
        "hitnet_1x240x320_model_float16_quant_opt.trt"
    )
    diagnostic = "/runtime/diagnostics/pointcloud_20260730/diagnose_hitnet"
    container_prefixes = [
        container_output + f"/pair{index}" for index in range(4)
    ]
    command = (
        "set -Eeuo pipefail; "
        "export LD_LIBRARY_PATH=/root/swarm_ws/devel/lib:/usr/local/lib:"
        "${LD_LIBRARY_PATH:-}; "
        f"{diagnostic} {engine} {container_output} "
        + " ".join(container_prefixes)
    )
    subprocess.run(
        ["docker", "exec", "omninxt_omnidepth", "bash", "-lc", command],
        check=True,
    )

    metadata = {
        "engine_sha256": (
            "4197ba2c3ac0696ad9e9e3483e2eac04ee5f7c6e8b39f5883ece1da660fe7142"
        ),
        "onnx_sha256": (
            "b589c3ff5e751603874de7d7ca0e88d06db4f9db0d87378d894da9e226ce369f"
        ),
        "preprocessing": "BGR/gray -> gray -> OpenCV INTER_LINEAR resize 320x240 -> float32 / 255 -> [1,2,240,320]",
        "samples": [],
    }
    rows = []
    for sample_index, (name, description, _, _) in enumerate(SAMPLES):
        first = sample_index
        repeated = sample_index + 2
        left = cv2.imread(os.path.join(output, f"pair{first}_left.png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(os.path.join(output, f"pair{first}_right.png"), cv2.IMREAD_GRAYSCALE)
        disparity_u16 = cv2.imread(
            os.path.join(output, f"pair{first}_disparity_x256.png"),
            cv2.IMREAD_UNCHANGED,
        )
        repeated_u16 = cv2.imread(
            os.path.join(output, f"pair{repeated}_disparity_x256.png"),
            cv2.IMREAD_UNCHANGED,
        )
        disparity = disparity_u16.astype(np.float32) / 256.0
        valid_disp = disparity[np.isfinite(disparity) & (disparity > 0)]
        scaled = np.clip(disparity / 96.0 * 255.0, 0, 255).astype(np.uint8)
        disparity_color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        disparity_color[disparity <= 0] = 0
        warped, error, valid, consistency = photometric_consistency(
            left, right, disparity
        )
        error_color = cv2.applyColorMap(
            np.clip(error.astype(np.float32) * 3.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        error_color[~valid] = 0

        cv2.imwrite(os.path.join(output, name + "_left.png"), left)
        cv2.imwrite(os.path.join(output, name + "_right.png"), right)
        cv2.imwrite(os.path.join(output, name + "_disparity_x256.png"), disparity_u16)
        cv2.imwrite(os.path.join(output, name + "_disparity_color.png"), disparity_color)
        cv2.imwrite(os.path.join(output, name + "_right_warped.png"), warped)
        cv2.imwrite(os.path.join(output, name + "_warp_error.png"), error_color)

        record = {
            "name": name,
            "description": description,
            "positive_disparity_fraction": round(
                float(np.mean(disparity > 0)), 8
            ),
            "disparity_px": {
                "min": round(float(np.min(valid_disp)), 6),
                "p01": percentile(valid_disp, 1),
                "p10": percentile(valid_disp, 10),
                "p50": percentile(valid_disp, 50),
                "p90": percentile(valid_disp, 90),
                "p99": percentile(valid_disp, 99),
                "max": round(float(np.max(valid_disp)), 6),
            },
            "right_warp_consistency": consistency,
            "repeated_stream_max_abs_quantized_difference_px": round(
                float(np.max(np.abs(
                    disparity_u16.astype(np.int32)
                    - repeated_u16.astype(np.int32)
                ))) / 256.0,
                6,
            ),
        }
        metadata["samples"].append(record)
        rows.append(
            np.hstack(
                (
                    label(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), name + " LEFT"),
                    label(cv2.cvtColor(right, cv2.COLOR_GRAY2BGR), name + " RIGHT"),
                    label(disparity_color, "CURRENT HITNET 0-96 px"),
                    label(error_color, "RIGHT-WARP ERROR"),
                )
            )
        )

    cv2.imwrite(os.path.join(output, "overview.png"), np.vstack(rows))
    with open(os.path.join(output, "metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_html(output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
