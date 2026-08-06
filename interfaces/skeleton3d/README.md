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
