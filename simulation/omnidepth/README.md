# OmniDepth, Isaac disparity and skeleton runtime

This directory contains the current server implementation:

- Isaac `distance_to_camera` converted to synchronized disparity;
- four-camera depth feeder and patched D2SLAM configuration;
- TensorRT RTMPose detection and 3D skeleton stream;
- browser viewer on port 8766.

Run `run_isaac_sync_live.sh` after machine configuration. D2SLAM source is
reconstructed with `tools/bootstrap_d2slam.sh`; ONNX files and GPU-specific
TensorRT engines stay outside Git. `OMNINXT_MAX_PERSON_RANGE` defaults to
`0`, so the previous 5 m detection limit is disabled.
