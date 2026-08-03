#!/usr/bin/env python3
"""Live tracked 2D/depth/3D pose viewer served to a local browser."""

import argparse
import importlib.util
import json
import math
import os
import threading
import time
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import yaml
from rtmlib import RTMPose, YOLOX
from trt_rtmpose import TensorRTRTMPose


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
spec = importlib.util.spec_from_file_location("pose3d", os.path.join(SCRIPT_DIR, "100_pose_depth_skeleton.py"))
pose3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose3d)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--det-model", required=True)
    parser.add_argument("--pose-model", required=True)
    parser.add_argument("--pose-engine")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--det-interval", type=int, default=20)
    parser.add_argument("--person-threshold", type=float, default=0.35)
    parser.add_argument("--keypoint-threshold", type=float, default=0.30)
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=5.0)
    return parser.parse_args()


def atomic_write(path, data, mode="wb"):
    temporary = path + ".tmp"
    with open(temporary, mode) as stream:
        stream.write(data)
    os.replace(temporary, path)


def quadrant_boxes(boxes):
    result = []
    for box in boxes:
        x0, y0, x1, y1 = (float(value) for value in box)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        col, row = int(cx >= 320), int(cy >= 240)
        qx, qy = col * 320, row * 240
        clipped = [max(qx, x0), max(qy, y0), min(qx + 319, x1), min(qy + 239, y1)]
        if clipped[2] - clipped[0] >= 20 and clipped[3] - clipped[1] >= 30:
            result.append(clipped)
    return np.asarray(result, dtype=np.float32)


def update_boxes(keypoints, scores):
    boxes = []
    for points, confidence in zip(keypoints, scores):
        valid = confidence >= 0.25
        if np.count_nonzero(valid) < 5:
            continue
        xy = points[valid]
        center = np.median(xy, axis=0)
        col, row = int(center[0] >= 320), int(center[1] >= 240)
        qx, qy = col * 320, row * 240
        lo, hi = np.min(xy, axis=0), np.max(xy, axis=0)
        width, height = max(30.0, hi[0] - lo[0]), max(50.0, hi[1] - lo[1])
        boxes.append([max(qx, lo[0] - width * .25), max(qy, lo[1] - height * .25),
                      min(qx + 319, hi[0] + width * .25), min(qy + 239, hi[1] + height * .25)])
    return np.asarray(boxes, dtype=np.float32)


def make_person(geometry, local_points, scores, depth, args, detection_id):
    person = {"pair": geometry["name"], "side": geometry["side"], "detection_id": detection_id,
              "color": geometry["color"], "joints": []}
    for joint_id, name in enumerate(pose3d.JOINTS):
        u, v = (float(value) for value in local_points[joint_id])
        score = float(scores[joint_id])
        joint = {"id": joint_id, "name": name, "pixel": [round(u, 3), round(v, 3)],
                 "pixel_int": [int(round(u)), int(round(v))], "score": round(score, 6),
                 "depth_m": None, "depth_mad_m": None, "depth_samples": 0,
                 "xyz_rect_m": None, "xyz_imu_m": None}
        if score >= args.keypoint_threshold:
            sample = pose3d.sample_depth(depth, u, v, 2, args.min_depth, args.max_depth)
            if sample:
                z, mad, count = sample
                rect, imu = pose3d.lift_joint(geometry, u, v, z)
                joint.update(depth_m=round(z, 5), depth_mad_m=round(mad, 5), depth_samples=count,
                             xyz_rect_m=rect.round(5).tolist(), xyz_imu_m=imu.round(5).tolist())
        person["joints"].append(joint)
    return person


def draw_3d_panel(people, width=640, height=960):
    panel = np.full((height, width, 3), (18, 23, 31), dtype=np.uint8)
    origin = np.array([width // 2, int(height * .63)], dtype=float)
    scale = 115.0
    yaw, pitch = -.72, -.48
    colors = [(90, 100, 255), (120, 230, 90), (255, 155, 90), (220, 110, 220)]

    def project(point):
        x, y, z = point
        x1 = math.cos(yaw) * x + math.sin(yaw) * z
        z1 = -math.sin(yaw) * x + math.cos(yaw) * z
        y1 = math.cos(pitch) * y - math.sin(pitch) * z1
        return tuple(np.rint(origin + [x1 * scale, -y1 * scale]).astype(int))

    for value in np.arange(-3, 3.01, .5):
        cv2.line(panel, project([value, -3, 0]), project([value, 3, 0]), (37, 47, 62), 1)
        cv2.line(panel, project([-3, value, 0]), project([3, value, 0]), (37, 47, 62), 1)
    axes = (([1, 0, 0], (70, 70, 255), "+X FRONT"),
            ([0, 1, 0], (70, 230, 90), "+Y LEFT"),
            ([0, 0, 1], (255, 120, 70), "+Z UP"))
    for endpoint, color, label in axes:
        cv2.arrowedLine(panel, project([0, 0, 0]), project(endpoint), color, 3, cv2.LINE_AA, tipLength=.12)
        cv2.putText(panel, label, project(endpoint), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    for person_id, person in enumerate(people):
        color = colors[person_id % len(colors)]
        for a, b in pose3d.BONES:
            ja, jb = person["joints"][a], person["joints"][b]
            if ja["xyz_imu_m"] is not None and jb["xyz_imu_m"] is not None:
                cv2.line(panel, project(ja["xyz_imu_m"]), project(jb["xyz_imu_m"]), color, 4, cv2.LINE_AA)
        for joint in person["joints"]:
            if joint["xyz_imu_m"] is not None:
                cv2.circle(panel, project(joint["xyz_imu_m"]), 6, (50, 240, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, "3D SKELETON / IMU BODY FRAME", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 240, 250), 2, cv2.LINE_AA)
    return panel


HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OmniDepth 实时三维骨架</title><style>body{margin:0;background:#080c12;color:#eef4ff;font-family:system-ui;text-align:center}header{position:sticky;top:0;background:#101722ed;padding:9px;z-index:2}h1{font-size:18px;margin:0 0 4px}#s{font-size:13px;color:#b8c6d9}img{display:block;max-width:100%;height:auto;margin:auto}</style></head><body><header><h1>OmniDepth 实时二维/深度/三维人体骨架</h1><div id="s">正在等待...</div></header><img id="v"><script>const im=document.querySelector('#v'),s=document.querySelector('#s');async function tick(){const t=Date.now();im.src='latest.jpg?t='+t;try{const r=await fetch('status.json?t='+t);const d=await r.json();s.textContent=`真实新深度 ${d.depth_unique_hz.toFixed(2)} Hz · 页面处理 ${d.processing_fps.toFixed(2)} FPS · 人体 ${d.people} · 三维关节 ${d.valid_3d_joints} · 骨架推理 ${d.inference_ms.toFixed(0)} ms · 总延迟 ${d.latency_ms.toFixed(0)} ms`;}catch(e){}setTimeout(tick,100)}tick();</script></body></html>'''


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format, *args):
        return


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        return


def main():
    args = parse_args()
    root = os.path.realpath(args.input_dir)
    os.makedirs(root, exist_ok=True)
    atomic_write(os.path.join(root, "viewer.pid"), str(os.getpid()), "w")
    atomic_write(os.path.join(root, "index.html"), HTML, "w")
    handler = lambda *a, **k: QuietHandler(*a, directory=root, **k)
    server = QuietServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with open(os.path.join(args.config_dir, "fisheye_cams.yaml")) as stream:
        cameras = yaml.safe_load(stream)
    geometries = [pose3d.pair_geometry(args.config_dir, pair, cameras) for pair in pose3d.PAIR_DEFS]
    detector = YOLOX(args.det_model, model_input_size=(416, 416), backend="onnxruntime", device="cpu")
    estimator = (TensorRTRTMPose(args.pose_engine, model_input_size=(192, 256))
                 if args.pose_engine else
                 RTMPose(args.pose_model, model_input_size=(192, 256), backend="onnxruntime", device="cpu"))
    boxes = np.empty((0, 4), dtype=np.float32)
    frame_index = 0
    last_mtime = 0
    times = deque(maxlen=30)
    input_times = deque(maxlen=30)
    print("Live viewer: http://127.0.0.1:{}".format(args.port), flush=True)

    try:
        while True:
            frame_path = os.path.join(root, "latest.npz")
            try:
                mtime = os.stat(frame_path).st_mtime_ns
            except FileNotFoundError:
                time.sleep(.05)
                continue
            if mtime == last_mtime:
                time.sleep(.01)
                continue
            last_mtime = mtime
            started = time.monotonic()
            try:
                with np.load(frame_path) as data:
                    stamp_ns = int(data["stamp_ns"])
                    images = [data["left{}".format(i)].copy() for i in range(4)]
                    depths = [data["depth{}".format(i)].copy() for i in range(4)]
            except (OSError, ValueError, EOFError):
                continue
            input_times.append(time.monotonic())
            mosaic = np.vstack((np.hstack(images[:2]), np.hstack(images[2:])))
            mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
            detection_ran = (len(boxes) == 0 or frame_index % max(1, args.det_interval) == 0)
            inference_start = time.monotonic()
            if detection_ran:
                boxes = quadrant_boxes(detector(mosaic_bgr))
            if len(boxes):
                keypoints, scores = estimator(mosaic_bgr, bboxes=boxes)
            else:
                keypoints = np.empty((0, 17, 2), dtype=np.float32)
                scores = np.empty((0, 17), dtype=np.float32)
            inference_ms = (time.monotonic() - inference_start) * 1000.0
            boxes = update_boxes(keypoints, scores)

            raw_people = []
            pair_people = [[] for _ in range(4)]
            for detection_id, (points, confidence) in enumerate(zip(keypoints, scores)):
                if not pose3d.person_is_valid(confidence, args.person_threshold):
                    continue
                center = np.median(points[confidence >= .25], axis=0)
                index = int(center[1] >= 240) * 2 + int(center[0] >= 320)
                offset = np.array([(index % 2) * 320, (index // 2) * 240], dtype=np.float32)
                local = points - offset
                person = make_person(geometries[index], local, confidence, depths[index], args, detection_id)
                raw_people.append(person)
                pair_people[index].append(person)
            fused = pose3d.fuse_people(raw_people, .75)
            views = []
            for index, geometry in enumerate(geometries):
                rgb = tuple(int(geometry["color"][position:position + 2], 16) for position in (5, 3, 1))
                left_view = pose3d.annotate(images[index], pair_people[index], args.keypoint_threshold, rgb)
                depth_view = pose3d.annotate(pose3d.colorize_depth(depths[index], args.min_depth, args.max_depth), pair_people[index], args.keypoint_threshold, rgb)
                cv2.putText(left_view, geometry["name"], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
                views.append(np.hstack((left_view, depth_view)))
            left_panel = np.vstack(views)
            right_panel = draw_3d_panel(fused, 640, left_panel.shape[0])
            combined = np.hstack((left_panel, right_panel))
            times.append(time.monotonic())
            processing_fps = ((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 else 0.0
            input_fps = ((len(input_times) - 1) / (input_times[-1] - input_times[0])) if len(input_times) > 1 else 0.0
            valid_count = sum(j["xyz_imu_m"] is not None for p in fused for j in p["joints"])
            latency_ms = max(0.0, time.time() * 1000.0 - stamp_ns / 1e6)
            banner = "LIVE {:.2f} FPS | INPUT {:.2f} Hz | PEOPLE {} | 3D JOINTS {} | INFER {:.0f} ms | LATENCY {:.0f} ms".format(processing_fps, input_fps, len(fused), valid_count, inference_ms, latency_ms)
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 34), (5, 8, 12), -1)
            cv2.putText(combined, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 255, 180), 2, cv2.LINE_AA)
            display = cv2.resize(combined, (960, 720), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 86])
            if ok:
                atomic_write(os.path.join(root, "latest.jpg"), encoded.tobytes())
            depth_unique_hz = input_fps
            producer_status = os.path.join(root, "producer_status.json")
            try:
                with open(producer_status) as stream:
                    depth_unique_hz = float(json.load(stream).get("producer_hz", input_fps))
            except (OSError, ValueError, TypeError):
                pass
            result = {"stamp_ns": stamp_ns, "processing_fps": processing_fps, "input_fps": input_fps,
                      "depth_unique_hz": depth_unique_hz,
                      "people": len(fused), "valid_3d_joints": valid_count, "inference_ms": inference_ms,
                      "latency_ms": latency_ms, "detector_ran": detection_ran, "raw_people": raw_people,
                      "fused_people": fused}
            atomic_write(os.path.join(root, "status.json"), json.dumps(result, ensure_ascii=False), "w")
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            os.unlink(os.path.join(root, "viewer.pid"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
