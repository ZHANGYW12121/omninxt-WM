# Isaac simulation

The tracked application lives in `database/`; Isaac Sim itself remains an
external 5.1 installation. Run:

```bash
./tools/configure_machine.sh <machine options>
./simulation/isaacsim/database/run_people_warehouse_with_map.sh
```

The launcher executes the repository copy of `database/main.py`, so later Git
pulls immediately update the simulation. Warehouse, OmniNxt USD and people
assets are read from paths in `.local/machine.env`.
