#!/usr/bin/env python3
"""Detect COCO-17 poses and lift joints into the OmniDepth IMU/body frame."""

import argparse
import base64
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from rtmlib import Body


JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
PAIR_DEFS = (
    ("A_B_RIGHT", 0, 1, "right", "stereo_calib_0_1_240_320.yaml", "#ff5555"),
    ("B_C_REAR", 1, 2, "rear", "stereo_calib_1_2_240_320.yaml", "#55dd77"),
    ("C_D_LEFT", 2, 3, "left", "stereo_calib_2_3_240_320.yaml", "#5590ff"),
    ("D_A_FRONT", 3, 0, "front", "stereo_calib_3_0_240_320.yaml", "#ffe05a"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--mode", choices=("lightweight", "balanced", "performance"), default="lightweight")
    parser.add_argument("--det-model")
    parser.add_argument("--pose-model")
    parser.add_argument("--det-input-size", type=int, default=416)
    parser.add_argument("--person-threshold", type=float, default=0.35)
    parser.add_argument("--keypoint-threshold", type=float, default=0.30)
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--depth-radius", type=int, default=2)
    parser.add_argument("--merge-distance", type=float, default=0.75)
    return parser.parse_args()


def camera_matrix(node):
    fx, fy, cx, cy = node["intrinsics"]
    return (
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray(node["distortion_coeffs"], dtype=np.float64),
    )


def pair_geometry(config_dir, pair, cameras):
    name, left_id, right_id, side, stereo_file, color = pair
    with open(os.path.join(config_dir, stereo_file), encoding="utf-8") as stream:
        stereo = yaml.safe_load(stream)
    k0, d0 = camera_matrix(stereo["cam0"])
    k1, d1 = camera_matrix(stereo["cam1"])
    baseline = np.asarray(stereo["cam1"]["T_cn_cnm1"], dtype=np.float64)
    r1, r2, p1, p2, q, valid_roi_left, valid_roi_right = cv2.stereoRectify(
        k0, d0, k1, d1, (320, 240), baseline[:3, :3], baseline[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
    )
    t_cam_imu = np.asarray(cameras["cam{}".format(left_id)]["T_cam_imu"], dtype=np.float64)
    raw_rotation = t_cam_imu[:3, :3].T
    raw_translation = -raw_rotation.dot(t_cam_imu[:3, 3])
    angle = math.pi / 4.0
    virtual_rotation = np.array(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0],
         [-math.sin(angle), 0.0, math.cos(angle)]], dtype=np.float64,
    )
    rotation = raw_rotation.dot(virtual_rotation).dot(r1.T)
    return {
        "name": name, "left_camera": "CAM_{}".format(chr(65 + left_id)),
        "left_camera_id": left_id, "right_camera_id": right_id,
        "side": side, "color": color, "rotation": rotation,
        "translation": raw_translation, "projection": p1,
        "projection_right": p2, "reprojection": q,
        "rectification_left": r1, "rectification_right": r2,
        "virtual_rotation_left": virtual_rotation,
        "virtual_rotation_right": np.array(
            [[math.cos(-angle), 0.0, math.sin(-angle)],
             [0.0, 1.0, 0.0],
             [-math.sin(-angle), 0.0, math.cos(-angle)]],
            dtype=np.float64),
        "valid_roi_left": tuple(int(value) for value in valid_roi_left),
        "valid_roi_right": tuple(int(value) for value in valid_roi_right),
        "baseline_m": float(np.linalg.norm(baseline[:3, 3])),
        "optical_axis_imu": rotation.dot(np.array([0.0, 0.0, 1.0])),
    }


def sample_depth(depth, u, v, radius, min_depth, max_depth):
    x, y = int(round(float(u))), int(round(float(v)))
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        return None
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    values = depth[y0:y1, x0:x1].astype(np.float64).ravel()
    values = values[np.isfinite(values) & (values >= min_depth) & (values <= max_depth)]
    if values.size < max(3, (2 * radius + 1) ** 2 // 4):
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, int(values.size)


def lift_joint(geometry, u, v, depth_m):
    projection = geometry["projection"]
    fx, fy = float(projection[0, 0]), float(projection[1, 1])
    cx, cy = float(projection[0, 2]), float(projection[1, 2])
    rect = np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])
    imu = geometry["rotation"].dot(rect) + geometry["translation"]
    return rect, imu


def person_is_valid(scores, threshold):
    scores = np.asarray(scores)
    visible = scores[scores >= 0.30]
    if visible.size < 5:
        return False
    return float(np.mean(np.sort(scores)[-8:])) >= threshold


def annotate(image, people, keypoint_threshold, color_bgr):
    canvas = (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
              if image.ndim == 2 else image.copy())
    for person in people:
        joints = person["joints"]
        for a, b in BONES:
            ja, jb = joints[a], joints[b]
            if ja["score"] >= keypoint_threshold and jb["score"] >= keypoint_threshold:
                cv2.line(canvas, tuple(ja["pixel_int"]), tuple(jb["pixel_int"]), color_bgr, 2, cv2.LINE_AA)
        for joint in joints:
            if joint["score"] < keypoint_threshold:
                continue
            point = tuple(joint["pixel_int"])
            valid = joint["xyz_imu_m"] is not None
            cv2.circle(canvas, point, 4, (50, 255, 255) if valid else (80, 80, 255), -1, cv2.LINE_AA)
            depth_label = "{:.2f}m".format(joint["depth_m"]) if valid else "no depth"
            label = "{} {}".format(joint["id"], depth_label)
            origin = (point[0] + 4, point[1] - 4)
            size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            cv2.rectangle(canvas, (origin[0] - 1, origin[1] - size[1] - 2),
                          (origin[0] + size[0] + 1, origin[1] + 2), (0, 0, 0), -1)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.33, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def colorize_depth(depth, min_depth, max_depth):
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        normalized = (depth[valid] - min_depth) / (max_depth - min_depth)
        scaled[valid] = np.clip(np.rint((1.0 - normalized) * 255.0), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def center_of_person(person):
    preferred = [5, 6, 11, 12]
    points = [person["joints"][index]["xyz_imu_m"] for index in preferred]
    points = [point for point in points if point is not None]
    if len(points) < 2:
        points = [joint["xyz_imu_m"] for joint in person["joints"] if joint["xyz_imu_m"] is not None]
    return np.median(np.asarray(points), axis=0) if points else None


def fuse_people(raw_people, merge_distance):
    clusters = []
    for person in raw_people:
        center = center_of_person(person)
        if center is None:
            continue
        best = None
        best_distance = float("inf")
        for cluster in clusters:
            if person["pair"] in cluster["pairs"]:
                continue
            distance = float(np.linalg.norm(center - cluster["center"]))
            if distance < merge_distance and distance < best_distance:
                best, best_distance = cluster, distance
        if best is None:
            clusters.append({"members": [person], "pairs": {person["pair"]}, "center": center})
        else:
            best["members"].append(person)
            best["pairs"].add(person["pair"])
            best["center"] = np.mean([center_of_person(member) for member in best["members"]], axis=0)

    fused = []
    for person_id, cluster in enumerate(clusters):
        joints = []
        for joint_id, name in enumerate(JOINTS):
            candidates = [member["joints"][joint_id] for member in cluster["members"]]
            candidates = [joint for joint in candidates if joint["xyz_imu_m"] is not None]
            if not candidates:
                joints.append({"id": joint_id, "name": name, "xyz_imu_m": None, "score": 0.0})
                continue
            weights = np.array([max(1e-3, joint["score"]) / (0.02 + joint["depth_mad_m"]) for joint in candidates])
            xyz = np.average(np.array([joint["xyz_imu_m"] for joint in candidates]), axis=0, weights=weights)
            joints.append({"id": joint_id, "name": name, "xyz_imu_m": xyz.round(6).tolist(),
                           "score": round(float(max(joint["score"] for joint in candidates)), 6)})
        fused.append({"person_id": person_id, "source_pairs": sorted(cluster["pairs"]), "joints": joints})
    return fused


def image_data_uri(path):
    with open(path, "rb") as stream:
        return "data:image/png;base64," + base64.b64encode(stream.read()).decode("ascii")


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniDepth 三维人体骨架</title><style>
*{box-sizing:border-box}body{margin:0;background:#080c12;color:#e7eef8;font-family:system-ui,"Noto Sans SC",sans-serif;overflow:hidden}
#cv{position:fixed;inset:0;width:100vw;height:100vh;cursor:grab}.panel{position:fixed;z-index:2;background:#101722eF;border:1px solid #344154;border-radius:11px;padding:12px;box-shadow:0 10px 30px #0008}
#controls{left:14px;top:14px;width:340px}h1{font-size:17px;margin:0 0 5px}.small{font-size:12px;color:#aab7ca;line-height:1.5}.actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}button{background:#192536;color:#e7eef8;border:1px solid #40516a;border-radius:7px;padding:6px 9px;cursor:pointer}
#images{right:14px;top:14px;width:min(58vw,900px);max-height:55vh;overflow:auto}#images img{display:block;width:100%;margin-top:8px;border-radius:7px}.legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:4px}
@media(max-width:800px){#images{display:none}#controls{width:calc(100vw - 28px)}}
</style></head><body><canvas id="cv"></canvas><section id="controls" class="panel"><h1>OmniDepth 三维人体骨架</h1>
<div class="small">坐标：无人机 IMU/机体系，单位 m。红 X、绿 Y、蓝 Z。左键旋转，滚轮缩放，Shift/右键平移。</div>
<div class="small" id="summary"></div><div class="actions"><button id="reset">重置视角</button><button id="labels">隐藏关节名称</button><button id="raw">显示原始扇区骨架</button></div>
<div class="legend"><span><i class="dot" style="background:#ff5555"></i>AB右</span><span><i class="dot" style="background:#55dd77"></i>BC后</span><span><i class="dot" style="background:#5590ff"></i>CD左</span><span><i class="dot" style="background:#ffe05a"></i>DA前</span></div></section>
<aside id="images" class="panel"><b>同帧二维骨架与关节深度</b><img src="__OVERVIEW__"></aside><script>
const data=__DATA__,bones=__BONES__,canvas=document.querySelector('#cv'),ctx=canvas.getContext('2d');let rx=-.55,ry=-.75,zoom=1,px=0,py=0,drag=false,lx=0,ly=0,pan=false,showLabels=true,showRaw=false;
const colors=['#ff6b6b','#5ee18b','#65a0ff','#ffe36a','#d58cff','#5ee5e5'];
function resize(){const d=Math.min(devicePixelRatio,2);canvas.width=innerWidth*d;canvas.height=innerHeight*d;draw()}function project(v){const d=Math.min(devicePixelRatio,2),w=canvas.width,h=canvas.height,sy=Math.sin(ry),cy=Math.cos(ry),sx=Math.sin(rx),cx=Math.cos(rx);let x=cy*v[0]+sy*v[2],z=-sy*v[0]+cy*v[2],y=cx*v[1]-sx*z,z2=sx*v[1]+cx*z;let s=Math.min(w,h)*.23*zoom/(1+Math.max(-.7,z2*.12));return[w/2+px*d+x*s,h/2+py*d-y*s,z2,s]}
function line(a,b,color,width=3){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width*Math.min(devicePixelRatio,2);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
function skeleton(person,color,alpha=1){ctx.globalAlpha=alpha;for(const e of bones){const a=person.joints[e[0]],b=person.joints[e[1]];if(a.xyz_imu_m&&b.xyz_imu_m)line(a.xyz_imu_m,b.xyz_imu_m,color,3)}for(const j of person.joints){if(!j.xyz_imu_m)continue;const p=project(j.xyz_imu_m);ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],5*Math.min(devicePixelRatio,2),0,Math.PI*2);ctx.fill();if(showLabels){ctx.fillStyle='#fff';ctx.font=`${10*Math.min(devicePixelRatio,2)}px system-ui`;ctx.fillText(`${j.id} ${j.name}`,p[0]+6,p[1]-5)}}ctx.globalAlpha=1}
function axes(){line([0,0,0],[1,0,0],'#ff4d4d',4);line([0,0,0],[0,1,0],'#45df76',4);line([0,0,0],[0,0,1],'#518cff',4);const o=project([0,0,0]);ctx.fillStyle='#fff';ctx.font=`bold ${12*Math.min(devicePixelRatio,2)}px system-ui`;ctx.fillText('O / IMU',o[0]+7,o[1]+15);for(let g=-3;g<=3;g+=.5){line([g,-3,0],[g,3,0],'#233044',1);line([-3,g,0],[3,g,0],'#233044',1)}}
function draw(){ctx.fillStyle='#080c12';ctx.fillRect(0,0,canvas.width,canvas.height);axes();if(showRaw)for(let i=0;i<data.raw_people.length;i++)skeleton(data.raw_people[i],data.raw_people[i].color,.55);else for(let i=0;i<data.fused_people.length;i++)skeleton(data.fused_people[i],colors[i%colors.length],1)}
canvas.onpointerdown=e=>{drag=true;lx=e.clientX;ly=e.clientY;pan=e.shiftKey||e.button===2;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;let dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(pan){px+=dx;py+=dy}else{ry+=dx*.007;rx+=dy*.007}draw()};canvas.onpointerup=()=>drag=false;canvas.oncontextmenu=e=>e.preventDefault();canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));draw()};
document.querySelector('#reset').onclick=()=>{rx=-.55;ry=-.75;zoom=1;px=py=0;draw()};document.querySelector('#labels').onclick=e=>{showLabels=!showLabels;e.target.textContent=showLabels?'隐藏关节名称':'显示关节名称';draw()};document.querySelector('#raw').onclick=e=>{showRaw=!showRaw;e.target.textContent=showRaw?'显示融合骨架':'显示原始扇区骨架';draw()};
document.querySelector('#summary').textContent=`检测 ${data.raw_people.length} 个扇区人体，融合为 ${data.fused_people.length} 人；有效三维关节 ${data.valid_3d_joint_count} 个。`;addEventListener('resize',resize);resize();
</script></body></html>'''


def main():
    args = parse_args()
    input_dir = os.path.realpath(args.input_dir)
    config_dir = os.path.realpath(args.config_dir)
    with open(os.path.join(config_dir, "fisheye_cams.yaml"), encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream)
    geometries = [pair_geometry(config_dir, pair, cameras) for pair in PAIR_DEFS]
    body_kwargs = dict(mode=args.mode, backend="onnxruntime", device="cpu", to_openpose=False)
    if args.det_model or args.pose_model:
        if not (args.det_model and args.pose_model):
            raise ValueError("--det-model and --pose-model must be provided together")
        body_kwargs.update(det=args.det_model, det_input_size=(args.det_input_size, args.det_input_size),
                           pose=args.pose_model, pose_input_size=(192, 256))
    body = Body(**body_kwargs)

    raw_people = []
    annotated = []
    geometry_metadata = []
    for geometry in geometries:
        pair_dir = os.path.join(input_dir, geometry["name"])
        image = cv2.imread(os.path.join(pair_dir, "left.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(os.path.join(pair_dir, "left.png"))
        npy_path = os.path.join(pair_dir, "depth_m.npy")
        if os.path.exists(npy_path):
            depth = np.load(npy_path).astype(np.float32)
        else:
            depth_mm = cv2.imread(os.path.join(pair_dir, "depth_mm.png"), cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                raise FileNotFoundError(os.path.join(pair_dir, "depth_mm.png"))
            depth = depth_mm.astype(np.float32) / 1000.0
            depth[depth_mm == 0] = np.nan
        keypoints, scores = body(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        pair_people = []
        for detection_id in range(keypoints.shape[0]):
            if not person_is_valid(scores[detection_id], args.person_threshold):
                continue
            person = {"pair": geometry["name"], "side": geometry["side"],
                      "detection_id": detection_id, "color": geometry["color"], "joints": []}
            for joint_id, name in enumerate(JOINTS):
                u, v = (float(value) for value in keypoints[detection_id, joint_id, :2])
                score = float(scores[detection_id, joint_id])
                pixel_int = [int(round(u)), int(round(v))]
                joint = {"id": joint_id, "name": name, "pixel": [round(u, 4), round(v, 4)],
                         "pixel_int": pixel_int, "score": round(score, 6), "depth_m": None,
                         "depth_mad_m": None, "depth_samples": 0, "xyz_rect_m": None, "xyz_imu_m": None}
                if score >= args.keypoint_threshold:
                    sampled = sample_depth(depth, u, v, args.depth_radius, args.min_depth, args.max_depth)
                    if sampled is not None:
                        depth_m, mad, count = sampled
                        rect, imu = lift_joint(geometry, u, v, depth_m)
                        joint.update({"depth_m": round(depth_m, 6), "depth_mad_m": round(mad, 6),
                                      "depth_samples": count, "xyz_rect_m": rect.round(6).tolist(),
                                      "xyz_imu_m": imu.round(6).tolist()})
                person["joints"].append(joint)
            pair_people.append(person)
            raw_people.append(person)
        rgb = tuple(int(geometry["color"][index:index + 2], 16) for index in (5, 3, 1))
        image_view = annotate(image, pair_people, args.keypoint_threshold, rgb)
        depth_view = annotate(colorize_depth(depth, args.min_depth, min(args.max_depth, 5.0)),
                              pair_people, args.keypoint_threshold, rgb)
        title = "{} / {} / {} person(s)".format(geometry["name"], geometry["side"], len(pair_people))
        cv2.putText(image_view, title,
                    (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(depth_view, "DEPTH {:.1f}-{:.1f}m".format(args.min_depth, min(args.max_depth, 5.0)),
                    (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(pair_dir, "pose_on_image.png"), image_view)
        cv2.imwrite(os.path.join(pair_dir, "pose_on_depth.png"), depth_view)
        annotated.append(np.hstack((image_view, depth_view)))
        geometry_metadata.append({"pair": geometry["name"], "side": geometry["side"],
                                  "left_camera": geometry["left_camera"],
                                  "optical_axis_imu": geometry["optical_axis_imu"].round(6).tolist()})

    overview = np.vstack(annotated)
    overview_path = os.path.join(input_dir, "pose_depth_overview.png")
    cv2.imwrite(overview_path, overview)
    fused_people = fuse_people(raw_people, args.merge_distance)
    valid_count = sum(joint["xyz_imu_m"] is not None for person in fused_people for joint in person["joints"])
    result = {"frame": "imu", "units": "metres", "joint_format": "COCO-17",
              "depth_sampling": {"radius_px": args.depth_radius, "method": "finite local median",
                                  "min_depth_m": args.min_depth, "max_depth_m": args.max_depth},
              "geometry": geometry_metadata, "raw_people": raw_people, "fused_people": fused_people,
              "valid_3d_joint_count": valid_count}
    json_path = os.path.join(input_dir, "skeletons_3d.json")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    html_path = os.path.join(input_dir, "skeleton_3d_viewer.html")
    document = (HTML.replace("__OVERVIEW__", image_data_uri(overview_path))
                .replace("__DATA__", json.dumps(result, ensure_ascii=False, separators=(",", ":")))
                .replace("__BONES__", json.dumps(BONES)))
    with open(html_path, "w", encoding="utf-8") as stream:
        stream.write(document)
    print(json.dumps({"raw_people": len(raw_people), "fused_people": len(fused_people),
                      "valid_3d_joints": valid_count, "overview": overview_path,
                      "json": json_path, "html": html_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
