# OmniDepth point-cloud quality diagnosis — 2026-07-30

## Conclusion

The current poor point cloud is not primarily caused by the four-camera
calibration. The dominant failure mode is the combination of:

1. imperfect/dense HITNet disparity in ambiguous real-image regions; and
2. a point-cloud path that accepts almost every predicted pixel without
   confidence, occlusion, consistency, or valid-overlap rejection.

An unmeasured example vignette mask from the repository also changes the model
input substantially and should not be used as if it were calibrated for this
camera. It aggravates the result but is not the sole cause.

## Calibration evidence

- Four virtual stereo baselines: 144.732–145.195 mm.
- Baseline spread: 0.462 mm.
- Reprojection standard deviation: below 0.23 px for all four pairs.
- Real-scene rectified A-B, B-C, C-D, and D-A images show corresponding
  structures on approximately the same image rows.
- The stored nominal AprilGrid size can introduce proportional absolute-scale
  error, but cannot explain the observed unrecognizable local geometry.

Rectified-pair evidence:

`runtime/diagnostics/pointcloud_20260730/rectified_pairs/`

## HITNet evidence

Model contract:

- input: float32 `[1, 2, 240, 320]`
- output: float32 `[1, 240, 320, 1]`

The two channels are a rectified grayscale stereo pair. The output is
horizontal disparity in pixels. Depth is approximately:

`Z = f * B / disparity`

For this calibration, `f * B` is about 19.3 px·m. Therefore:

- 1 m -> about 19.3 px disparity
- 3 m -> about 6.4 px disparity
- 5 m -> about 3.9 px disparity

At long range, a one-pixel disparity error produces a large depth error:
approximately 0.46 m at 3 m and 1.28 m at 5 m.

Direct TensorRT inference on the four real rectified pairs produced positive
disparity for every one of 76,800 pixels in every pair. The maps contain
recognizable coarse scene structure, so the engine/input order is not wholly
broken, but it also assigns disparity to occlusions, low-texture surfaces,
reflections, saturated windows, and invalid border/non-overlap areas.

Raw diagnostic outputs:

`runtime/diagnostics/pointcloud_20260730/hitnet_raw/`

## Point-cloud post-processing evidence

The runtime sends every disparity map directly to `reprojectImageTo3D`.
Afterward it applies only:

- a pixel stride of 2; and
- a depth range of 0.1–10 m.

It does not apply:

- model confidence;
- left-right consistency;
- occlusion rejection;
- calibrated stereo-overlap ROI;
- photometric warp consistency;
- speckle/component filtering;
- temporal consistency.

The theoretical maximum retained point count is:

`4 * (320 / 2) * (240 / 2) = 76,800`

The captured runtime cloud contains 76,799 finite points. This proves that
almost every sampled network prediction, including unreliable predictions, is
being converted into 3D.

As a conservative comparison, OpenCV SGBM accepted only 12.7–25.9% of pixels
on the same pairs. This does not prove SGBM is more accurate, but it illustrates
how unusually permissive the current HITNet-to-cloud path is.

## Photometric-mask issue

The runtime loads four sample `uint16` vignette masks from the repository.
They were not measured on this OAK unit. On the captured real pairs, they
change mean intensities by roughly 19–28 gray levels in several images and
change the disparity distributions materially.

Disabling the masks in the standalone test changed the point cloud but did not
make it good. Therefore this is a real configuration defect and a secondary
aggravating factor, not the primary cause.

Comparison outputs:

- `runtime/diagnostics/pointcloud_20260730/pointcloud_raw_no_example_mask.pcd`
- `runtime/diagnostics/pointcloud_20260730/pointcloud_with_example_mask.pcd`

## Other observations

- `cv::reprojectImageTo3D(disparity, xyz, Q, 3)` is not requesting a 16-bit
  output. The fourth argument is `handleMissingValues`; integer `3` converts
  to `true`. The output remains floating point. This call is confusing and
  should be made explicit, but it is not the quality failure.
- A single-frame four-sector cloud is not a mapped/world-accumulated cloud.
  Some visible separation into four sectors is expected. Dense false surfaces
  and severely warped local objects are not expected.
- Publishing XYZ without image texture makes the cloud harder to interpret,
  but does not cause the geometric corruption.

## Recommended next controlled work

1. Disable the unmeasured sample vignette masks.
2. Save/publish per-pair rectified images, raw disparity, depth, and validity.
3. Restrict disparity to the calibrated common-view ROI and the desired depth
   range.
4. Add left-right and/or photometric consistency plus occlusion rejection.
5. Add speckle filtering and short temporal consistency.
6. Publish four sector-colored clouds and individual pair clouds.
7. Measure a planar AprilGrid/wall at known 0.5, 1, 2, and 3 m distances.
8. If filtered HITNet is still inaccurate, evaluate fine-tuning/retraining for
   this virtual-fisheye stereo domain or a replacement stereo network.

The priority is to make uncertainty visible and reject bad predictions before
deciding that the entire network must be replaced.
