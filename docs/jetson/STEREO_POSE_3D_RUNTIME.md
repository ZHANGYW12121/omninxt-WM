# OmniNxt four-direction real-time 3D skeleton runtime

## Start

```bash
cd /home/neu/OmniNxt
./scripts/102_live_stereo_pose_test.sh
```

The process stays in the foreground. Press `Ctrl+C` to stop only the sparse
pose node. The OAK and OmniDepth containers remain available. No flight
controller, serial device, MAVROS process, or `/mavros/*` topic is required.

From another terminal it can also be stopped with:

```bash
cd /home/neu/OmniNxt
./scripts/103_stop_live_stereo_pose.sh
```

Open the memory-backed diagnostic viewer at:

```text
http://127.0.0.1:8766
```

The 3D panel uses a fixed ground view: the X-Y grid is the ground plane,
`+Z UP` is vertical on screen, `+X FRONT` and `+Y LEFT` lie on the ground.
This is only a display projection; published `base_link` coordinates are not
rotated or rewritten.

The formal processing loop does not write PCD, HTML, JPEG or NPZ files. JPEG
encoding is performed only by the asynchronous display thread when the viewer
is enabled. Disable the web server and all JPEG encoding with:

```bash
STEREO_POSE_WEB=false ./scripts/102_live_stereo_pose_test.sh
```

## Runtime pipeline

1. OmniDepth rectifies all four adjacent stereo pairs at 320 x 240 and
   publishes the exact-timestamp pre-HITNet left/right images at 10 Hz. For
   the Python process, the same eight images are packed as four `left|right`
   rows in `/depth_estimation/pose_stereo_mosaic` (640 x 960, mono8).
2. The same C++ node publishes four physical-camera-centred 416 x 320 mono
   anchor views with a 120-degree horizontal FOV. Adjacent anchors overlap by
   about 30 degrees, so a person on an old virtual-stereo seam is no longer
   cut into two half-person detector inputs. The remap runs only while the
   pose topic has a subscriber and does not copy the 5120 x 720 raw image into
   Python. They are packed A/B over C/D in
   `/depth_estimation/pose_anchor_mosaic` (832 x 640, mono8).
   The two mosaics share the source image timestamp. Bundling reduces twelve
   concurrent rospy image callbacks to two without removing data.
3. YOLOX-Nano scans a 2 x 2 mosaic containing all four anchor views. A full
   refresh runs every ten frames. Every third frame a rotating single-camera
   rescue pass is allowed only for an anchor with no active box. All four
   physical directions remain observable; no direction or HITNet pair is
   selectively disabled.
4. Boxes seen in adjacent overlapping anchors are compared in `base_link`
   bearing/elevation/scale space. The most central observation is retained
   before RTMPose, avoiding duplicate pose inference for one person.
5. RTMPose-S runs once on that retained, complete physical-camera view. Each
   COCO joint ray is then projected into both adjacent calibrated rectified
   stereo pairs. A local epipolar NCC search supplies the other pixel; it does
   not require a second detector or second RTMPose inference.
6. Each accepted correspondence is triangulated with that pair's calibrated
   P1/P2 matrices and transformed into the fixed drone body frame `base_link`.
   If both adjacent pairs are valid, the better NCC/reprojection/uncertainty
   candidate is selected. A conservative sparse-body consistency guard rejects
   only gross disparity aliases such as one eye several metres behind its
   torso; this is not dense point-cloud filtering.
7. Residual duplicate observations are fused in 3D by corresponding-joint and
   torso-centre distance. The ROS output therefore has one `person_id` even
   when the same person is visible in two physical cameras.
8. Four-pair HITNet continues in parallel as a lower-rate dense-depth
   validator and fallback source. It is deliberately given less filter weight
   than direct sparse triangulation.
9. A constant-velocity per-joint Kalman filter runs directly in `base_link`.
   It uses only image timestamps and visual 3D measurements. It does not use
   attitude, position, IMU, or any other flight-controller data.

The body-frame convention used by the output is:

```text
+X: front
+Y: left
+Z: up
origin: fixed calibrated body origin (historically the IMU origin in Kalibr)
```

The calibration files retain names such as `T_cam_imu` and `xyz_imu_m` for
compatibility with Kalibr and existing JSON consumers. In this runtime those
names mean the same fixed body-reference origin; they do not imply a live IMU
subscription. ROS messages are published with `frame_id=base_link`.

Because no flight attitude or position is used, coordinates are always
drone-relative. If the drone rotates or translates, a stationary person moves
in `base_link`. This is the intended representation for onboard relative
human localization and avoidance; it is not a world-stabilized track.

## ROS outputs

```text
/omninxt_pose/joints_3d          geometry_msgs/PoseArray
/omninxt_pose/skeleton_frame     std_msgs/String (fixed COCO-17/ST-GCN schema)
/omninxt_pose/skeleton_markers   visualization_msgs/MarkerArray
/omninxt_pose/status             std_msgs/String (JSON)
```

`/omninxt_pose/status` contains person IDs, COCO joint IDs/names,
`xyz_base_link_raw_m`/`xyz_base_link_m`, source (`anchor_epipolar` or HITNet
fallback), physical source anchor and adjacent stereo pair, anchor detections
before/after overlap deduplication, rescue-detector state, rejected gross
kinematic outliers, inference times, output frame, and body-frame filter mode.

COCO body-17 joint IDs are:

```text
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
```

## Implemented models

```text
YOLOX-Nano ONNX:
/home/neu/OmniNxt/models/rtmpose/yolox_nano_coco_416.onnx

YOLOX-Nano TensorRT FP16:
/home/neu/OmniNxt/models/rtmpose/tensorrt/yolox_nano_coco_416_fp16.engine

RTMPose-S TensorRT FP16:
/home/neu/OmniNxt/models/rtmpose/tensorrt/rtmpose_s_body17_256x192_fp16.engine

RTMPose-T ONNX (official body7 COCO-17 export):
/home/neu/OmniNxt/models/rtmpose/rtmpose_t_body17_256x192.onnx

RTMPose-T TensorRT FP16 dynamic batch 1/4/8:
/home/neu/OmniNxt/models/rtmpose/tensorrt/rtmpose_t_body17_256x192_fp16_dynamic_b8.engine
```

The normal launcher now selects RTMPose-T with a dynamic `min=1`, `opt=4`,
`max=8` profile. All retained people from the same anchor mosaic are affine
cropped first and submitted in one TensorRT execution when there are at most
eight. More than eight people are divided into batches of eight. This changes
only the 2D pose estimator; COCO-17 ordering, sparse stereo triangulation,
tracking, and the backend schema remain unchanged.

Start the default Tiny dynamic-batch path:

```bash
./scripts/102_live_stereo_pose_test.sh
```

Revert to the preserved Small static-batch baseline:

```bash
STEREO_POSE_MODEL=s ./scripts/102_live_stereo_pose_test.sh
```

Rebuild and validate the Tiny engine:

```bash
./scripts/107_build_rtmpose_dynamic_engine.sh t
python3 scripts/108_validate_rtmpose_dynamic_batch.py \
  --engine models/rtmpose/tensorrt/rtmpose_t_body17_256x192_fp16_dynamic_b8.engine
```

## Measured on this Orin Nano

- Rectified four-pair input: approximately 10 Hz.
- Four 416 x 320 physical-camera anchor views: approximately 10 Hz, exactly
  synchronized with the rectified pair set.
- Bundled anchor/stereo input topics: measured 10.00 Hz over approximately
  20 seconds after the callback consolidation.
- Sparse 3D skeleton output, one visible person: approximately 9.8-10.1 Hz
  in stable windows.
- Direct anchor-epipolar stereo joints in the tested view: commonly 10-16 of
  17, with temporal prediction/fallback commonly leaving 13-17 displayed.
- Four-pair HITNet dense depth under concurrent pose load: approximately
  4.0-4.6 Hz.
- YOLOX anchor-mosaic/full-resolution rescue pass: commonly about 30-40 ms.
- One RTMPose-S inference for one retained person: commonly about 23-25 ms.
- Typical total one-person frame: about 70-120 ms. Input uses a latest-frame
  queue, so occasional detector frames above 100 ms do not build latency.
- Runtime flight-controller input: disabled.

The RTMPose-S figures above are retained as the accuracy/performance baseline.
The new RTMPose-T dynamic engine was validated on this Orin Nano as follows:

```text
batch 1: 12.97 ms mean end-to-end pose adapter time
batch 4: 26.90 ms mean, one TensorRT call [4]
batch 8: 48.05 ms mean, one TensorRT call [8]
```

All tests returned `[N,17,2]` keypoints and `[N,17]` scores. Batch 4 and 8
outputs were numerically identical to running the same crops individually in
this validation. The preserved static RTMPose-S adapter measured 14.41 ms for
batch 1 in the same isolated test.

An exploratory live comparison with up to three detected people showed the
Small engine using `[1,1,1]`, about 85 ms pose time and roughly 178-215 ms total
frames. Tiny dynamic batching reduced observed total frames to roughly
119-158 ms. The exact people/poses changed between frames, so this is an
integration sanity check, not a controlled accuracy benchmark. The complete
multi-person system still measured roughly 6-8 Hz under that load and is not
yet a guaranteed 10 Hz multi-person pipeline. Dense HITNet GPU contention,
YOLOX detector frames, and CPU epipolar matching remain in the total latency.

## Quick checks

```bash
docker exec omninxt_omnidepth bash -lc '
source /opt/ros/noetic/setup.bash
rostopic hz /omninxt_pose/joints_3d
'
```

```bash
docker exec omninxt_omnidepth bash -lc '
source /opt/ros/noetic/setup.bash
rostopic echo -n 1 /omninxt_pose/status/data
'
```

```bash
docker exec omninxt_omnidepth bash -lc '
source /opt/ros/noetic/setup.bash
rostopic hz /depth_estimation/stereo_0/depth
'
```

The dense point-cloud publisher is disabled in this mode. Dense depth remains
available for joint validation, but no PCL construction or publication is
performed.
