# RTMPose-T dynamic-batch validation

Date: 2026-08-02  
Device: Jetson Orin Nano, TensorRT 8.5.2, SM 8.7

## Artifacts

```text
ONNX:
/home/neu/OmniNxt/models/rtmpose/rtmpose_t_body17_256x192.onnx
SHA256 a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f

TensorRT FP16 dynamic engine:
/home/neu/OmniNxt/models/rtmpose/tensorrt/rtmpose_t_body17_256x192_fp16_dynamic_b8.engine
SHA256 c0631c0e58c3f935fc6b76d387149bb40f4a8a124bc1db4d482c1e98a758783d

Profile:
min [1,3,256,192]
opt [4,3,256,192]
max [8,3,256,192]
```

## Isolated adapter validation

The measurement includes affine crop, normalization, synchronous TensorRT,
SimCC decode, and coordinate restoration. It uses deterministic synthetic
image content and does not measure pose accuracy.

| Batch | TensorRT calls | Mean ms | p95 ms | Output |
|---:|---|---:|---:|---|
| 1 | `[1]` | 12.972 | 15.709 | `[1,17,2]`, `[1,17]` |
| 4 | `[4]` | 26.902 | 30.280 | `[4,17,2]`, `[4,17]` |
| 8 | `[8]` | 48.054 | 50.420 | `[8,17,2]`, `[8,17]` |

For batch 4 and batch 8, maximum keypoint and confidence deltas versus running
the same crops one by one were both zero in this test.

The preserved static RTMPose-S engine passed the refactored runner and measured
14.414 ms mean for batch 1.

## RTMPose-S dynamic-batch extension

Built on 2026-08-03 from the existing RTMPose-S ONNX with the same profile:

```text
Engine:
/home/neu/OmniNxt/models/rtmpose/tensorrt/rtmpose_s_body17_256x192_fp16_dynamic_b8.engine
SHA256 a7c0c88f9b851ce6c3d3c3b9a77b2975e82b1d8a31ee4a20ca3d3231befad264

Profile:
min [1,3,256,192]
opt [4,3,256,192]
max [8,3,256,192]
```

| Batch | TensorRT calls | Mean ms | p95 ms | Output |
|---:|---|---:|---:|---|
| 1 | `[1]` | 15.370 | 16.709 | `[1,17,2]`, `[1,17]` |
| 4 | `[4]` | 30.275 | 32.288 | `[4,17,2]`, `[4,17]` |
| 8 | `[8]` | 53.584 | 58.706 | `[8,17,2]`, `[8,17]` |

For all three shapes, the maximum keypoint and confidence deltas versus
running the same crops one by one were zero. Live integration with
`STEREO_POSE_MODEL=s` reported `pose_dynamic_batch=true` and
`pose_max_batch=8`.

## Live integration

- OAK assembled input remained approximately 20.0 Hz.
- Tiny engine deserialized successfully with the live four-pair HITNet node.
- Runtime status reported `pose_dynamic_batch=true`, `pose_max_batch=8`.
- One-person status reported a single `[1]` call.
- The fixed stream passed `omninxt.skeleton3d.v1` validation and contained
  exactly 17 joint rows for the detected person.
- No CUDA illegal access, TensorRT inference failure, or ROS node crash was
  observed.
- In an exploratory three-person view, Tiny reduced complete-frame latency
  relative to static Small, but the full system remained around 6-8 Hz. This
  does not establish stable 10 Hz multi-person operation.

Re-run the isolated validation with:

```bash
cd /home/neu/OmniNxt
PYTHONPATH=/home/neu/OmniNxt/source/third_party/rtmlib:/home/neu/OmniNxt/scripts \
python3 scripts/108_validate_rtmpose_dynamic_batch.py \
  --engine models/rtmpose/tensorrt/rtmpose_t_body17_256x192_fp16_dynamic_b8.engine \
  --batches 1,4,8 --iterations 5
```
