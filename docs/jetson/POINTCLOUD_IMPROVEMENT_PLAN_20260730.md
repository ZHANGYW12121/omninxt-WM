# OmniDepth point-cloud improvement plan — 2026-07-30

## Current conclusion

The observed result has four separate causes:

1. The previous HTML renderer used an incorrect second-axis rotation formula
   and did not depth-sort points. It exaggerated the appearance of a flat
   cloud. This is now fixed.
2. The current captured scene is dominated by the floor. A single stereo depth
   image is a 2.5D surface (one depth per image pixel), so a floor-dominated
   scene naturally looks like several thin sheets when viewed edge-on.
3. The runtime uses the wrong direction for Kalibr `T_cam_imu` when merging the
   four sectors. Kalibr defines it as IMU-to-camera, while point-cloud merging
   needs camera-to-IMU. D2SLAM's own configuration explicitly states that
   OmniNxt uses `extrinsic_parameter_type: 0`, which performs this inversion.
   The current standalone loader defaults to mode 1.
4. HITNet still produces smooth but inaccurate disparity in low-texture,
   occluded, reflective, saturated, and far-range areas. Filtering can reject
   bad estimates but cannot create missing correct depth.

## Quantitative observations

- Global PCA eigenvalue ratio: `1 : 0.50 : 0.21`. The whole cloud is not one
  mathematical plane.
- Individual sector ratios along the thinnest axis are about `0.06–0.08`.
  Each sector is a thin 2.5D observation surface.
- Depending on the sector, plane residual medians are about `0.06–0.17 m`.
- Four virtual-stereo baselines are consistent at about `144.7–145.2 mm`.
- Derived `fB` is consistent at about `19.14–19.47 px·m`.
- All stereo Q matrices have the expected positive-disparity depth sign.
- Current retained disparity corresponds approximately to `0.3–4 m` in the
  captured scene; disparity has not collapsed to a single value.
- The captured reference image is mostly textured floor plus a few vertical
  surfaces, explaining much of the visual flatness.

## Phase 1 — correct geometry before evaluating the network

1. Add `extrinsic_parameter_type: 0` to the OmniDepth YAML.
2. Pass that value to `readCameraConfig` instead of using its mode-1 default.
3. Incorporate the stereo rectification rotation `R1` into the rectified-left
   camera-to-rig transform. Its present error is small, but it should not be
   omitted.
4. Publish the output in a clearly named `rig` or `imu` frame only after the
   transform direction has been verified.
5. Record one static validation scene containing the floor, two perpendicular
   walls, and an AprilGrid. Fit the same physical floor plane independently in
   all four sectors.
6. Accept the geometry only if sector floor normals agree and their plane
   offsets agree within a chosen tolerance.

The mode-0 candidate was initially diagnostic only.  It was subsequently
implemented in the live node together with the missing `R1^T` transform; see
the live-validation section below.

## Direction diagnosis from capture 20260730_123248

The sector-direction failure is now reproduced independently of disparity
quality.  With the exact transforms used by the current running process, the
four virtual-stereo optical axes are:

| Sector | Physical side | Current azimuth | Current elevation |
|---|---|---:|---:|
| 0 (A-B) | right | -124.61 deg | +29.89 deg |
| 1 (B-C) | rear | -124.61 deg | -29.67 deg |
| 2 (C-D) | left | -55.21 deg | -29.74 deg |
| 3 (D-A) | front | -55.76 deg | +29.94 deg |

Thus the current runtime mathematically has only two XY directions and tilts
the sectors upward/downward.  This matches the abnormal HTML view and cannot
be caused by HITNet disparity.

Kalibr's local `ConfigReader.py` names the corresponding API
`setExtrinsicsImuToCam()` / `getExtrinsicsImuToCam()`.  The generated
`T_cam_imu` therefore maps IMU coordinates to camera coordinates.  Point-cloud
merging needs the inverse transform.  D2SLAM's own
`d2frontend_params.cpp` also states `OmniNxt use mode 0`, where mode 0 performs
that inversion.  However, `quadcam_depth_est_trt.cpp` calls
`readCameraConfig()` without a mode, so the standalone loader's default mode 1
uses `T_cam_imu` directly.

Reprocessing the exact same colored PCD with:

1. inverse `T_cam_imu`; and
2. the missing rectified-left-to-virtual-left rotation `R1^T`

produces these optical-axis directions:

| Sector | Physical side | Corrected azimuth | Corrected elevation |
|---|---|---:|---:|
| 0 (A-B) | right | -89.87 deg | -0.10 deg |
| 1 (B-C) | rear | -179.94 deg | +0.72 deg |
| 2 (C-D) | left | +90.70 deg | +0.74 deg |
| 3 (D-A) | front | -0.43 deg | -0.74 deg |

These agree with a ROS FLU rig frame: +X front, +Y left, +Z up.  The four
omitted `R1` rotations are only 0.16--0.40 degrees, so they are a secondary
seam-accuracy issue; the reversed Kalibr transform is the dominant fault.

Diagnostic artifacts:

- `runtime/html_pointclouds/capture_20260730_123248/pointcloud_geometry_corrected_candidate.html`
- `runtime/html_pointclouds/capture_20260730_123248/xy_direction_before_after.png`
- `runtime/html_pointclouds/capture_20260730_123248/geometry_direction_diagnosis.txt`

## Live geometry correction and validation

The production node was corrected on 2026-07-30:

1. `quadcam_depth.yaml` now explicitly sets
   `extrinsic_parameter_type: 0`.
2. `quadcam_depth_est_trt.cpp` passes that mode to the camera loader.
3. The rectified-left cloud is transformed by `R1^T` before the existing
   virtual-left-to-fisheye and fisheye-to-IMU transforms.
4. Startup logs print every stereo optical axis in the IMU frame.

The rebuilt live node reported:

| Sector | Optical axis in IMU |
|---|---|
| A-B right | `[+0.002250, -0.999996, -0.001740]` |
| B-C rear | `[-0.999920, -0.001017, +0.012586]` |
| C-D left | `[-0.012165, +0.999842, +0.012983]` |
| D-A front | `[+0.999888, -0.007535, -0.012961]` |

An independently captured live cloud at
`runtime/html_pointclouds/capture_20260730_130610/` had sector median
azimuths `-89.6`, `-176.9`, `+93.1`, and `+1.3` degrees respectively.  Both
the node geometry and actual point distribution therefore agree with +X
front, +Y left, +Z up.

The corrected cloud remained stable at approximately 10.0 Hz.  No CUDA,
TensorRT, OOM, or node errors were found in the startup log.

## Filter-disabled comparison

At the user's request, `enable_disparity_filter` was set to `false` and the
live node was restarted without changing calibration, geometry, or the HITNet
engine.

- Filtered capture: 31,793 points.
- Unfiltered capture: 76,800 points.
- Each unfiltered sector retained 19,200 / 19,200 sampled pixels.
- The four directions remain correct.

The unfiltered cloud is denser and more continuous, but visibly restores
predictions from invalid overlap borders, occlusions, weak-texture regions,
and inconsistent stereo edges.  Comparison artifacts:

- `runtime/html_pointclouds/capture_20260730_131602/pointcloud_viewer.html`
- `runtime/html_pointclouds/capture_20260730_131602/xy_filtered_vs_unfiltered.png`
- filtered configuration backup:
  `runtime/backups/quadcam_depth_filtered_20260730_131417.yaml`

## Phase 2 — create a measurable depth benchmark

Place a matte planar board approximately perpendicular to each stereo pair at
measured distances:

- 0.5 m
- 1.0 m
- 2.0 m
- 3.0 m
- optionally 4.0 m

For each pair save:

- rectified left/right images;
- raw disparity;
- validity mask;
- depth image;
- individual-sector PCD.

Calculate:

- median depth bias;
- median absolute error;
- 90th/95th percentile error;
- valid-pixel ratio;
- fitted-plane RMSE;
- edge bleeding width;
- temporal standard deviation over 100 static frames.

This separates absolute-scale/calibration error from network disparity noise.

## Phase 3 — improve image-domain consistency

1. Keep fixed exposure during comparison tests.
2. Measure and generate real per-camera vignette/photometric calibration;
   never re-enable repository example masks.
3. Match camera gain/brightness between each stereo pair.
4. Reject saturated and very dark pixels before stereo inference.
5. Verify rectified vertical error across the whole image, especially edges.

## Phase 4 — improve disparity validity

The current conservative filter already applies disparity range, valid ROI,
local median deviation, texture, photometric consistency, and connected-area
filtering.

Next additions:

1. True left-right consistency. For HITNet this requires a verified reverse
   inference construction or a second model pass; it may reduce the live rate
   from 10 Hz toward 5 Hz on Orin Nano.
2. Explicit occlusion masks.
3. Edge-aware speckle/component filtering.
4. Short temporal confidence, motion-compensated when the vehicle moves.
5. A per-pixel confidence output if a replacement network supports one.

Do not increase point density until measured accuracy improves.

## Phase 5 — decide whether HITNet is usable

Compare on the exact same recorded frames:

1. raw HITNet;
2. filtered HITNet;
3. conservative StereoSGBM baseline;
4. one or more alternative stereo networks offline;
5. simulation ground-truth disparity.

If HITNet retains recognizable structure but fails the measured-distance
benchmark, fine-tune/retrain it using the exact virtual-fisheye stereo image
formation, exposure characteristics, and disparity range. The provenance and
training domain of the current ONNX file are not documented well enough to
assume that it matches this camera.

## Phase 6 — distinguish a single-frame cloud from a 3D map

One stereo frame always produces visible surfaces, not filled volume. A richer
3D environment model requires:

1. reliable VIO/odometry;
2. correct rig-to-IMU transform and time synchronization;
3. transform each cloud into a persistent world frame;
4. temporal fusion using voxel filtering/TSDF/occupancy mapping;
5. dynamic-object rejection.

Accumulation should only be enabled after single-frame depth bias and pose
accuracy are validated; otherwise it will turn current errors into thicker
ghost surfaces.

## Acceptance targets

Suggested initial targets, to be refined for the flight task:

- 0.5–2 m planar median error: below 5% of distance;
- 3 m planar median error: below 10%;
- plane RMSE at 1 m: below 5 cm;
- sector floor-normal disagreement: below 2 degrees;
- adjacent-sector plane offset disagreement: below 5 cm;
- stable static depth valid ratio: above 20%;
- live rate: at least 5 Hz for high-confidence mode, 10 Hz preferred.
