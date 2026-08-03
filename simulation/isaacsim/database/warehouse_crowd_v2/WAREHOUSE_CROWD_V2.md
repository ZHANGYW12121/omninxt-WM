# Warehouse Crowd V2

This directory implements the reproducible Warehouse crowd templates described
by `WAREHOUSE_CROWD_MAP_SERVER_REPRODUCTION.md`.

## Formal layouts

- `sparse`: a seed-determined 17--23 pedestrians. Seeds producing 17--19
  people use eight groups and seeds producing 20--23 use nine. At least two
  groups are singles; two groups use longitudinal flow and the remainder use
  transverse flow.
- `dense + transverse40`: exactly 40 pedestrians in 20 groups: twelve pairs,
  four singles, and four triples. All groups use eight-point transverse routes;
  ten start from each side. Complete formation envelopes are packed in route
  order, so neighbouring groups retain at least 1.0 m clearance even at the
  same route phase.

Both layouts keep pedestrian routes inside `x=-7.0..8.7` and
`y=-3.8..25.8`. The drone spawn is sampled in `x=-3..4`, `y=-10..-5`; its
goal uses an independent seeded random stream in `x=-3..4`, `y=27..29`.

## Launch

Sparse layout:

```bash
cd /path/to/omninxt-WM/simulation/isaacsim/database
./run_people_warehouse_omni.sh \
  --crowd-mode server_v2 \
  --crowd-layout sparse \
  --crowd-seed 1
```

Dense layout:

```bash
./run_people_warehouse_omni.sh \
  --crowd-mode server_v2 \
  --crowd-layout dense \
  --dense-profile transverse40 \
  --crowd-seed 1
```

`--crowd-count 17..23` may override a sparse population. The formal dense
profile only accepts `--crowd-count 40`. If the count is omitted it is part of
the seeded sparse map and is therefore repeatable.

Preview without Isaac:

```bash
/usr/bin/python3 preview_warehouse_crowd_layout.py \
  --crowd-layout dense \
  --dense-profile transverse40 \
  --crowd-seed 1
```

Use `--no-browser` on a remote server, or `--no-serve` to generate only the
JSON snapshot.

## Determinism reference

Seed 1 produces:

```text
drone spawn XY = (-2.0594502912, -5.7628313153)
goal XYZ        = (-0.6784015653, 28.6863600395, 1.0)
sparse          = 18 people / 8 groups
dense           = 40 people / 20 groups
```

Run the reproduction checks with:

```bash
/usr/bin/python3 -m unittest -v test_warehouse_crowd_reproduction.py
```

## Runtime and offline assets

People control and AnimationGraph writes run at 25 Hz through one physics
callback per person; skeleton collision updates run at 10 Hz. Static routes
and traffic reservations are generated before simulation rather than replanned
inside the 25 Hz loop.

The launcher exports the local character mirror at
`database/assets/people/Characters`. Verify or recreate it with:

```bash
cd /path/to/omninxt-WM
source .local/machine.env
"$ISAACSIM_ROOT/python.sh" simulation/isaacsim/database/download_people_assets.py \
  --destination "$PEGASUS_PEOPLE_ASSET_ROOT" --verify-only
"$ISAACSIM_ROOT/python.sh" simulation/isaacsim/database/download_people_assets.py \
  --destination "$PEGASUS_PEOPLE_ASSET_ROOT"
```

`warehouse_crowd_v2/__init__.py` optionally prefers a complete migration source
specified by `WAREHOUSE_CROWD_V2_SOURCE`; if it is absent or incomplete, this
checked-in implementation is used without changing the global `PYTHONPATH`.

Legacy mode remains available only for diagnosis:

```bash
WAREHOUSE_CROWD_MODE=legacy ./run_people_warehouse_omni.sh
```
