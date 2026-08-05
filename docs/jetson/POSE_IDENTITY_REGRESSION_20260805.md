# Skeleton identity regression — 2026-08-05

## Datasets

- `runtime/pose_test_recordings/single_walk_20260805_165055/input.bag`
  - 557 raw frames, 28.60 s, about 19.47 Hz
- `runtime/pose_test_recordings/two_people_crossing_20260805_165200/input.bag`
  - 563 raw frames, 29.05 s, about 19.38 Hz

Both bags contain the original synchronized `bgr8 5120x720` OAK mosaic on
`/oak_ffc_4p/assemble_image`. The encoded MP4 is only a preview; all regression
runs use the lossless ROS image stream.

## Root cause

The two-person baseline never produced more than two fused 3D measurements in
a frame, but the identity layer retained old predicted tracks and created new
IDs whenever sparse stereo depth jumped outside the Euclidean association
gate. This caused up to five displayed skeletons and eight public IDs from two
physical people. Repeated final timestamps from the depth node could also be
mistaken for new observations after camera input stopped.

## Implemented corrections

- Ignore repeated or stale camera-derived timestamps.
- Collapse duplicate RTMPose body skeletons before 3D lifting.
- Prefer a confirmed identity over a provisional depth fragment.
- Reassociate calibrated bearing-continuous observations when radial depth
  jumps or a person moves between cameras.
- Keep identity state internally for 4.0 s while limiting visible prediction
  to 1.2 s.
- Require seven consecutive observations before publishing a later new ID.
- Bound single-frame center and joint innovations to reject metre-scale sparse
  disparity aliases.
- Calculate 3D coordinates only for COCO body joints 5–16. The transport stays
  backward compatible at 17 rows; joints 0–4 are present with
  `coordinate_valid=0`.

## Fixed regression results

| Dataset | Published IDs | Max displayed people | Frames above truth | Median processing rate |
|---|---:|---:|---:|---:|
| Single person | `0` | 1 | 0 | 9.63 Hz |
| Two-person crossing | `0, 1` | 2 | 0 | 9.42 Hz |

The two-person fixed run processed 212 frames with two published people, four
with one, and two startup frames with zero. It never created a third internal
track. Dynamic RTMPose batching remained enabled (`max_batch=8`), with batch
sizes 1 and 2 observed in the regression.

Detailed machine-readable outputs:

- `runtime/pose_regression/fixed2_single_20260805_171020/summary.json`
- `runtime/pose_regression/fixed2_two_20260805_171315/summary.json`
- Corresponding `status.jsonl` files retain every processed sample.

## Repeat command

```bash
cd /home/neu/OmniNxt

./scripts/105_run_pose_regression.sh \
  /home/neu/OmniNxt/runtime/pose_test_recordings/single_walk_20260805_165055/input.bag \
  1 single_repeat

./scripts/105_run_pose_regression.sh \
  /home/neu/OmniNxt/runtime/pose_test_recordings/two_people_crossing_20260805_165200/input.bag \
  2 two_repeat
```

The regression uses an isolated ROS master and does not send data to a remote
backend.
