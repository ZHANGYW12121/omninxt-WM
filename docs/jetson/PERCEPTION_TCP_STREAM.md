# OmniNxt Nano-to-backend perception stream

## What is transmitted

A depth image is not a camera image. A ROS `32FC1` depth map stores one
floating-point distance in metres per pixel and contains no grayscale or RGB
texture suitable for YOLO or pose estimation.

Port `9766` therefore multiplexes two lossless packet kinds over one TCP
connection:

| packet kind | schema | source rate | contents |
|---|---|---:|---|
| `images` | `omninxt.pose_images.v1` | about 10 Hz | anchor mosaic and rectified stereo mosaic |
| `depth` | `omninxt.depth4.v1` | native HITNet rate, about 5 Hz | all four metric depth maps |

The image and depth rates are intentionally independent. The backend can run
2-D person/pose inference on every image packet and cache timestamped dense
depth packets for 3-D validation without reducing image inference to 5 Hz.

The image blocks are exactly the inputs used by the current Nano pose path:

```text
/depth_estimation/pose_anchor_mosaic
  mono8, 832x640
  CAM_A|CAM_B over CAM_C|CAM_D, each anchor 416x320

/depth_estimation/pose_stereo_mosaic
  mono8, 640x960
  four rows AB, BC, CD, DA; each row is left|right, each view 320x240
```

The depth block comes from:

```text
/depth_estimation/stereo_0/depth   AB / right sector
/depth_estimation/stereo_1/depth   BC / rear sector
/depth_estimation/stereo_2/depth   CD / left sector
/depth_estimation/stereo_3/depth   DA / front sector
```

No JPEG, resize, colour mapping, validity filtering, or depth cropping is
performed. Image data is raw `uint8`; the default depth format is lossless
little-endian float32 metres. Lossless zlib level 1 is applied to each packet.

## TCP framing

Each frame is:

```text
12-byte prefix + UTF-8 JSON header + binary payload
```

The prefix is network struct `!4sII`:

```text
magic        4 bytes   ASCII OPB1
header_len   uint32    big-endian
payload_len  uint32    big-endian
```

The JSON header includes:

```text
schema
kind
session_id
sequence
source_sequence
capture_timestamp_ns
send_timestamp_ns
blocks[]
compression
raw_bytes
payload_bytes
raw_crc32
```

Every entry in `blocks` describes one array with `name`, `dtype`, `shape`,
`offset`, and `nbytes`. Reconstruct each array from the decompressed payload:

```python
view = raw[block["offset"]:block["offset"] + block["nbytes"]]
array = np.frombuffer(view, dtype=np.dtype(block["dtype"]))
array = array.reshape(block["shape"])
```

Image packets contain blocks named `anchor_mosaic` and `stereo_mosaic`.
Depth packets contain one `[4,H,W]` block named `depths`, with pair order
`AB_RIGHT`, `BC_REAR`, `CD_LEFT`, `DA_FRONT`.

For default float32 depth:

```text
dtype: <f4
scale_m: 1.0
invalid values: NaN, infinity, zero, and negative source values are preserved
```

The optional `uint16_mm` mode uses `<u2`, `scale_m: 0.001`, and zero for an
invalid pixel. The backend must use packet metadata rather than hard-coding
array dimensions or data types.

## Timestamp association

Both packet kinds carry the original ROS `capture_timestamp_ns`. A backend
should keep a short ring buffer keyed by `(session_id, capture_timestamp_ns)`.
Image packets drive the 2-D detector at their full rate. When dense depth is
needed, use an exact timestamp match when available; otherwise use only a
bounded nearest timestamp and explicitly report the time delta. Do not match
using TCP receive time.

The sender keeps only the newest unsent packet of each kind. This prevents a
slow network or backend from blocking HITNet. Sequence gaps and the published
drop counters must therefore be treated as expected congestion telemetry, not
as corruption.

## Start on Nano

Start the backend listener first, then run:

```bash
cd /home/neu/OmniNxt

DEPTH_BACKEND_HOST=<ALIENWARE_IP> \
DEPTH_BACKEND_PORT=9766 \
./scripts/109_stream_depth_to_backend.sh
```

Despite the historical script filename, this now sends both pose images and
four depth maps. Defaults transmit every complete packet at its source rate.

Optional limits:

```bash
PERCEPTION_IMAGE_MAX_HZ=10 \
DEPTH_STREAM_MAX_HZ=5 \
DEPTH_BACKEND_HOST=<ALIENWARE_IP> \
./scripts/109_stream_depth_to_backend.sh
```

Check Nano status:

```bash
docker exec omninxt_omnidepth bash -lc '
source /opt/ros/noetic/setup.bash
rostopic echo /omninxt_perception_stream/status
'
```

Important fields are `connected`, `image_sent_hz`, `depth_sent_hz`,
`sent_mbps`, `dropped_pending`, and `last_error`.

## Receiver safety and validation

The receiver must:

1. read exactly 12 prefix bytes and verify `OPB1`;
2. bound header and payload lengths before allocation (64 KiB and 16 MiB are
   suitable current limits);
3. validate the schema, block offsets, shapes, and data types;
4. decompress zlib and verify `raw_bytes` plus `raw_crc32`;
5. detect session changes and sequence gaps;
6. associate data using `capture_timestamp_ns`;
7. reject stale packets according to the application's latency budget.

The protocol has no built-in authentication or encryption. Restrict port 9766
to the trusted Nano/Alienware LAN (or carry it inside WireGuard).
