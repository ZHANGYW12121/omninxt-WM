# Alienware skeleton receiver

This service is the formal Alienware-side consumer for the Nano sender in
`edge/jetson/scripts/102_live_stereo_pose.py`.

## Wire contract

- transport: TCP;
- framing: one UTF-8 JSON object followed by `\n`;
- default port: `9765`;
- schema: `omninxt.skeleton3d.v1`;
- frame: `base_link`, ROS FLU (`+X` forward, `+Y` left, `+Z` up);
- unit: metre;
- payload: stable person ID and exactly 17 COCO joints per person.

The canonical validator is `interfaces/skeleton3d/protocol.py`. The receiver
does not import any Nano implementation module.

## Configure and run on Alienware

```bash
cd /path/to/omninxt-WM
./tools/configure_skeleton_receiver.sh \
  --host 0.0.0.0 \
  --port 9765 \
  --max-people 20

./backend/run_skeleton_receiver.sh
```

`0.0.0.0` is required when the Nano connects over the LAN. Restrict the port
to the Nano address in the host firewall; do not expose this unauthenticated
research protocol to the public Internet.

If UFW is active, replace the placeholder with the Nano's current LAN address:

```bash
sudo ufw allow from <NANO_LAN_IP> to any port 9765 proto tcp \
  comment 'OmniNxt Nano skeleton stream'
```

For automatic startup in the current desktop login session:

```bash
./tools/install_skeleton_receiver_service.sh
systemctl --user status omninxt-skeleton-receiver.service
journalctl --user -u omninxt-skeleton-receiver.service -f
```

## Start the existing sender on Nano

Use the Alienware LAN address, not `127.0.0.1`:

```bash
cd /home/neu/OmniNxt
STEREO_POSE_BACKEND_HOST=<ALIENWARE_LAN_IP> \
STEREO_POSE_BACKEND_PORT=9765 \
./scripts/102_live_stereo_pose_test.sh
```

The Nano sender is non-blocking and retains only its newest unsent frame. A
receiver outage therefore does not stall camera/depth/pose inference.

## Output for the world model

The service atomically updates machine-local files under
`.local/run/skeleton_receiver/`:

- `latest_human_observation.npz`;
- `status.json`.

The NPZ contains:

```text
skeleton       [N,17,7]  refined absolute base_link [xyz, velocity, confidence]
skeleton_raw   [N,17,7]  received xyz/confidence retained for diagnostics
human_root     [N,10]    [root xyz, root velocity, extent xyz, confidence]
human_joints   [N,17,7]  root-relative [xyz, velocity, confidence]
human_mask     [N]
joint_mask     [N,17]    head indices 0..4 are always false; body 5..16 is used
joint_inferred_mask [N,17] true when depth was completed kinematically
human_ids      [N]
human_is_first [N]
joint_uncertainty_m      [N,17]
joint_prediction_age_ms [N,17]
```

The model-facing `skeleton` is refined causally using Nano capture timestamps.
The optimizer separates common root motion from local articulation, lowers
trust along the body-to-person viewing ray, weights measurements by source,
confidence, sigma and age, rejects isolated depth jumps, and projects the
result onto slowly learned symmetric body-bone lengths. Because the Warehouse
task contains standing pedestrians, a soft upright-walking prior keeps the
torso near +Z and leg chains predominantly toward -Z while still permitting
torso lean, knee flexion, foot lift and unrestricted whole-person motion. It
does not pin feet to a floor. Short gaps are
predicted for at most 0.35 s, including frames where the detector temporarily
omits the whole person. While the person itself remains detected, a joint that
has no usable depth is not removed after that high-confidence hold: previously
measured joints continue from their causal local-pose prediction, and
never-measured body joints are completed from learned bone lengths, the measured
opposite-side joint, and an aligned upright template. Such joints stay
visible but are marked in `joint_inferred_mask`, receive low confidence and
high uncertainty, and are never treated as new depth measurements. Stable
slots are keyed by `person_id`; a source
timestamp reset or a gap above 0.5 s resets recurrent Human state.

COCO head landmarks `nose`, `left_eye`, `right_eye`, `left_ear`, and
`right_ear` remain in the fixed 17-slot wire/tensor layout for compatibility,
but are zeroed in `skeleton`, masked out of `joint_mask`, and omitted from the
viewer and model-facing root/pose features. Their received values remain only
in `skeleton_raw` for diagnostics.

This stream supplies only the Human observation. Ego14 must come from the UAV
state source and Goal must come from the task source; neither is inferred from
the skeleton packet.

## Live 3-D viewer on Alienware

The viewer is a separate process, so browser rendering cannot block the TCP
receiver. It reads only the receiver's atomically replaced NPZ snapshot and
uses the unchanged `base_link` coordinates: X/Y are the ground plane and +Z
is up. It displays the refined `skeleton`; raw transmitted coordinates remain
available in `skeleton_raw` for offline A/B diagnostics.

Run it in a terminal:

```bash
./backend/run_skeleton_viewer.sh
```

Then open `http://127.0.0.1:8767`. The page supports mouse orbit, right-button
pan, wheel zoom, and `R` to reset the view. To start it automatically with the
desktop user session:

```bash
./tools/install_skeleton_viewer_service.sh
systemctl --user status omninxt-skeleton-viewer.service
```

The default bind address is localhost because the page has no authentication.
Override `SKELETON_VIEWER_HOST` and `SKELETON_VIEWER_PORT` in the machine-local
environment only when LAN access is intentionally required.

## Verify

```bash
ss -ltn | grep ':9765'
cat .local/run/skeleton_receiver/status.json
python3 -m unittest backend.tests.test_skeleton_receiver -v
```

## Nano pose-image and depth receiver

The independent receiver on TCP `9766` implements the formal `OPB1` stream
from `edge/jetson/scripts/109_stream_depth_to_backend.py` on `main`. It accepts
the independently timed `omninxt.pose_images.v1` and `omninxt.depth4.v1`
packets, bounds all lengths before allocation, validates block layouts,
losslessly decompresses zlib, and checks the raw CRC32.

Configure and install it without changing the skeleton service on `9765`:

```bash
./tools/configure_perception_receiver.sh \
  --host 0.0.0.0 --port 9766 --match-tolerance-ms 120
./tools/install_perception_receiver_service.sh
journalctl --user -u omninxt-perception-receiver.service -f
```

The Alienware is only a passive TCP listener. It does not need SSH access to
the Nano and must not remotely start the Nano sender. If UFW is active, allow
only the Nano address to reach this independent port:

```bash
sudo ufw allow from <NANO_LAN_IP> to any port 9766 proto tcp \
  comment 'OmniNxt Nano perception stream'
```

The Nano sender must target the Alienware LAN address (not `127.0.0.1`). For
the formal sender currently on `main`, its local operator starts it with:

```bash
cd /home/neu/OmniNxt
DEPTH_BACKEND_HOST=<ALIENWARE_LAN_IP> \
DEPTH_BACKEND_PORT=9766 \
./scripts/109_stream_depth_to_backend.sh
```

No inbound SSH connection from Alienware to Nano is part of this data path.

It atomically publishes under `.local/run/perception_receiver/`:

```text
latest_images.npz              anchor_mosaic + stereo_mosaic
latest_depth.npz               four wire depths + metric depth + valid mask
latest_matched_observation.npz nearest bounded image/depth timestamp pair
status.json                    counts, gaps, session and match diagnostics
```

Matching uses the original Nano/ROS `capture_timestamp_ns`, never TCP receive
time. The matched snapshot records `depth_time_delta_ns` and whether the match
was exact. Runtime arrays and received images remain machine-local and are not
tracked by Git.
