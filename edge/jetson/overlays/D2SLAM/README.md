# Jetson D2SLAM overlay

This directory contains the exact text sources and calibrated geometry used by
the Nano OmniDepth + sparse 3D skeleton runtime. Apply it over the official
D2SLAM baseline:

```text
repository: https://github.com/HKUST-Aerial-Robotics/D2SLAM
branch baseline: pr_fix_main
commit baseline: 3a4b8071f6f9c40a151c7685aa738b717ce8916a
```

From the `omnix-stack` repository root:

```bash
cp -a edge/jetson/overlays/D2SLAM/. /path/to/D2SLAM/
```

The overlay includes:

- JetPack 5.1.2 / TensorRT 8 compatibility changes;
- four-pair HITNet execution and optional point-cloud publishing;
- rectified stereo/depth topics used by the sparse skeleton path;
- four synchronized physical-camera anchor views and compact mosaics;
- CPU rectification selected for concurrent Orin Nano inference;
- the formally calibrated real-camera intrinsics/extrinsics from 2026-07-30;
- reproducible Jetson Dockerfiles.

It intentionally excludes TensorRT engines, ONNX files, bags, runtime logs,
point clouds, and optional photometric mask images. TensorRT plans must be
rebuilt on the target Jetson. The calibrated YAML is specific to OAK MXID
`19443010E16C6A2E00` and must not be presented as calibration for another rig.
