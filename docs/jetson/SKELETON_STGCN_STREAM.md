# OmniNxt skeleton stream for an ST-GCN backend

## Purpose

The Nano sends tracked COCO-17 3D skeletons to a separate backend.  The
perception thread never waits for that backend.  A one-frame queue in a daemon
TCP sender drops stale frames during network congestion and reconnects
automatically.

The legacy `/omninxt_pose/joints_3d` `PoseArray` is retained for RViz only.  It
omits invalid joints and therefore must not be used as indexed ST-GCN input.

## Nano outputs

- `/omninxt_pose/skeleton_frame` (`std_msgs/String`): fixed local JSON packet.
- Optional newline-delimited TCP JSON stream to the backend.
- Coordinate frame: `base_link`, +X forward, +Y left, +Z up, metres.
- No flight-controller data is used.

Every person always has exactly 17 joints in this order:

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

Each joint is an array with fields:

```text
x_m, y_m, z_m, pose_score, confidence,
coordinate_valid, measured, predicted,
measurement_sigma_m, measurement_age_ms, source_code
```

Invalid coordinates are numeric zero only to keep a fixed tensor.  They are
not measurements: `coordinate_valid` and `confidence` are both zero.

Source codes:

```text
0 invalid
1 stereo geometry
2 HITNet fallback
3 temporal prediction
4 other
```

No bone-length completion is inserted into this formal stream.

## Start the backend receiver

Copy these two files to the backend while retaining the same directory:

```text
scripts/104_skeleton_backend_receiver.py
scripts/skeleton_stream.py
```

On the backend computer:

```bash
cd /path/to/OmniNxt
python3 scripts/104_skeleton_backend_receiver.py \
  --host 0.0.0.0 \
  --port 9765 \
  --window 30 \
  --max-people 4
```

Allow TCP port 9765 only on the trusted onboard LAN.  This lightweight stream
does not implement encryption or authentication.

## Start Nano transmission

Replace `192.168.1.20` with the backend computer's address:

```bash
cd /home/neu/OmniNxt
STEREO_POSE_BACKEND_HOST=192.168.1.20 \
STEREO_POSE_BACKEND_PORT=9765 \
./scripts/102_live_stereo_pose_test.sh
```

If `STEREO_POSE_BACKEND_HOST` is omitted, network transmission is disabled and
the existing real-time pose pipeline behaves normally.

## ST-GCN tensor

`StgcnWindow.push()` returns:

```text
[N, C, T, V, M]
N = 1 batch
C = 5: x, y, z, confidence, coordinate_valid
T = configured temporal window (default 30 frames = 3 seconds at 10 Hz)
V = 17 joints
M = configured person slots (default 4)
```

Person IDs remain in stable slots while present.  Missing joints and unused
person slots remain zero with a zero validity channel.  A backend ST-GCN model
must be trained/configured with `in_channels=5`; a clean 3-channel pretrained
model cannot silently consume this tensor without adapting its input layer and
training distribution.

The reference receiver marks the exact location where `stgcn_model(tensor)`
should be called.  For a production world model, keep both the absolute pelvis
position and a pelvis-relative skeleton stream if global position and body
motion are both important.

