# Pegasus dependency

Pegasus is not vendored. Both machines use commit
`bef3c57f2cfef9c6dacf3262969e01b59eeb7c7a` from the official repository and
apply `patches/0001-portable-people-runtime.patch`.

The patch contains the verified local people-asset lookup and the single
physics callback design:

- crowd control limited to `PEGASUS_PEOPLE_CONTROL_HZ` (25 Hz by default);
- state read, controller execution and AnimationGraph write occur in that order;
- the old separate 250 Hz `update_state` callback is removed;
- skeleton/collision updates are limited by simulation time (10 Hz by default).

Run `tools/apply_pegasus_patch.sh` after configuring a machine. The PX4 path
change previously present in Pegasus `configs.yaml` is intentionally excluded;
`PX4_DIR` is supplied by this repository at runtime.
