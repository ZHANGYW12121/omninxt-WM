# Skeleton 3D interface

The canonical runtime contract is `SYNC_CONTRACT.yaml`. The transport
implementation is `../../simulation/omnidepth/runtime/skeleton_stream.py`.

Consumers must depend on the schema and frame identifiers, not on a server or
laptop filesystem path.

Identity metadata is additive within `omninxt.skeleton3d.v1`. `person_id`
remains a positive process-local integer for dataset compatibility. Consumers
that span perception restarts should use `track_uid` and reset temporal state
when the packet-level `session_id` changes. Frames with at most 12 valid 3D
measurements use exact max-cardinality/minimum-cost assignment; denser frames
use the O(N^3) Hungarian algorithm.

Simulation packets are episode-aware. Every synchronized image bundle carries
an increasing episode generation and the exact-render-time world pose of
`base_link`. The tracker resets on generation changes and compensates the
previous tracks into the current body frame before association. This prevents
vehicle translation, roll, pitch, or yaw from being interpreted as pedestrian
motion. Each joint then uses a covariance-carrying constant-velocity Kalman
state. Confirmed pose tracks slowly learn symmetric person-specific body-bone
lengths and project only their public output onto those lengths; projection
does not change a joint's fresh-versus-predicted temporal provenance.
Detector-only boxes require ten consecutive simulation frames before they can
be promoted to an articulated Human slot, and fully observed poses with grossly
invalid whole-body geometry are rejected. A normal articulated pose requires
two consecutive frames; a low-geometry candidate requires seven. Public
tracks always contain at least four body joints including two shoulder/hip
core joints, so a face-only or two-point response is never a Human slot.
The default public simulation output range is 6 m; the range tracker retains
0.25 m of internal exit hysteresis, and `<=0` remains an explicit diagnostic
override that disables the gate.
