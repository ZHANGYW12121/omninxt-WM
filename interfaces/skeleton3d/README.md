# Skeleton 3D interface

The canonical runtime contract is `SYNC_CONTRACT.yaml`. Wire validation lives
in `protocol.py`. Producer implementations are
`../../edge/jetson/scripts/skeleton_stream.py` and
`../../simulation/omnidepth/runtime/skeleton_stream.py`; the formal Alienware
consumer is `../../backend/skeleton_receiver/server.py`.

Consumers must depend on the schema and frame identifiers, not on a server or
laptop filesystem path.

`person_id` is a non-negative tracker identifier. The Nano tracker starts at
zero; consumers reserve `-1` only for an unoccupied fixed-size model slot.
