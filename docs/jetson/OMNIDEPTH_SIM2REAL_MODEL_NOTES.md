# Model provenance and portability notes

This package contains ONNX interchange models so the receiving NVIDIA GPU can
build its own TensorRT engines. The Jetson-generated `.trt/.engine` files are
intentionally excluded because serialized TensorRT plans are not portable
between different GPUs or TensorRT releases.

| File | Role | Provenance used in this project |
|---|---|---|
| `hitnet_1x240x320_model_float16_quant_opt.onnx` | Four-pair dense disparity | Model distributed in the official HKUST D2SLAM `models/hitnet_series` tree |
| `yolox_nano_coco_416.onnx` | COCO person detector | Official Megvii YOLOX Nano raw-output ONNX |
| `rtmpose_s_body17_256x192.onnx` | COCO Body-17 pose | RTMPose-S model used through Tau-J/rtmlib preprocessing and postprocessing |

The package also includes the rtmlib source and its license. D2SLAM and OmniNxt
upstream commit IDs and URLs are recorded in the main README. Before any
redistribution outside the project, re-check the upstream model and source
licenses for the intended use.

Expected TensorRT/static contracts in the current implementation:

```text
HITNet:     float32 [1,2,240,320] -> float32 [1,240,320,1]
YOLOX Nano: float32 [1,3,416,416] -> raw COCO [1,3549,85]
RTMPose-S:  float32 [1,3,256,192] -> SimCC x/y outputs handled by rtmlib
```

All three input pipelines use the exact preprocessing in the included C++ or
Python sources. Replacing a model while retaining only its input resolution is
not sufficient; output ordering, normalization and postprocessing must also
match.
