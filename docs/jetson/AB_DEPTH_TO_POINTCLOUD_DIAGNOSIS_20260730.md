# AB depth-to-point-cloud diagnosis (2026-07-30)

## Inputs

- Point cloud capture:
  `runtime/html_pointclouds/capture_20260730_215330`
- Stereo/depth capture:
  `runtime/stereo_depth_views/capture_20260730_215315`

These captures are not the same frame. Their ROS timestamps differ by
15.350177332 seconds. The assembled image and PCD inside the point-cloud
capture are timestamp matched to 0.00014 ms.

## Finding

The HTML viewer and the stereo-to-3D coordinate transform do not flatten a
correct AB depth map. The AB HITNet output itself assigns nearly the same
optical depth to most pixels, so pinhole reprojection correctly turns those
pixels into a fronto-parallel sheet.

### AB depth capture at 21:53:15

- Positive depth range: 0.186--0.569 m
- P10 / median / P90: 0.295 / 0.322 / 0.384 m
- Approximate person ROI median: 0.322 m
- Approximate left-background ROI median: 0.344 m
- Approximate bright right-background ROI median: 0.286 m

The person is visually outlined, but the predicted metric separation between
the person and the left background is only about 22 mm. The bright background
is even predicted closer than the person. This depth map is therefore not
metricly plausible even though its colorized edges look recognizable.

The AB rectified focal length is 132.092659 px and baseline is 0.144880 m, so:

`f * B = 19.137606 px*m`

The saved depth PNG agrees with
`cv::reprojectImageTo3D(disparity, Q)` to 0.25 mm median and 0.50 mm maximum,
which is the expected uint16 millimetre quantization error.

Reprojecting this AB depth frame produces a plane whose thinnest PCA standard
deviation is 45 mm, with the plane normal aligned to the local optical axis by
0.9905.

### AB PCD capture at 21:53:30

- AB points: 19,200 (all sampled pixels retained)
- Local optical-depth P10 / median / P90:
  0.298 / 0.327 / 0.342 m
- Therefore 80% of AB points occupy only a 44 mm optical-depth interval.
- Best-fit plane thickness (smallest PCA sigma): 32.8 mm
- Plane-normal alignment with the AB optical axis: 0.999151
- 18.70% of AB samples imply `x_right = x_left - disparity < 0`, so those
  pixels have no possible correspondence in the right image.
- 1.56% of samples exceed 96 px disparity.

This proves numerically that the PCD itself contains a fronto-parallel AB
sheet. It is not an HTML projection artifact.

## Why the color depth image is misleading

The viewer uses a fixed 0.2--5.0 m TURBO scale. Almost the entire AB result is
inside the narrow 0.2--0.57 m end of that scale. Small local changes and image
boundaries remain visible as colors, which can look like a recognizable person,
but the underlying metric depth does not separate the person from the room.

## Scene-specific cause

The AB rectified pair is dominated by the holder's dark clothes and arms:

- large low-texture black regions;
- severe near-field viewpoint change and occlusion;
- approximately 0.145 m baseline at only about 0.3 m range;
- disparity close to 60--100 px for near objects;
- only part of the left image has a valid right-image correspondence.

HITNet produces a dense disparity even where no reliable match exists. In this
frame it propagates the near foreground disparity into unsupported background
regions. With validity filtering disabled, all 19,200 AB samples are converted
to XYZ.

## Ruled out

- Q/depth scale error
- disparity sign error
- point-cloud sector rigid transform
- HTML point-cloud projection
- point-cloud/image mismatch inside the point-cloud capture

## Required follow-up

For a strict same-frame comparison, capture the four debug depth topics and the
sector-colored cloud under one common ROS header timestamp. Test a static scene
without a person holding or occluding the cameras, with textured targets at
known 0.5 m, 1.0 m, 2.0 m, and 3.0 m distances.
