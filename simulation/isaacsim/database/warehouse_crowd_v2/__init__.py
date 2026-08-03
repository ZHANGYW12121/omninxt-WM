"""Isolated Warehouse crowd V2 package loader.

If a separately migrated implementation is present, prepend only that
``warehouse_crowd_v2`` directory to this package's search path.  Never add its
parent ``database`` directory to ``PYTHONPATH`` because that can shadow local
flight-control modules.  The checked-in implementation remains the fallback.
"""

import os
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parents[2]
_DEFAULT_SOURCE = (
    _PROJECT_ROOT
    / "warehouse_crowd_map_migration_20260724"
    / "isaacsim"
    / "database"
    / "warehouse_crowd_v2"
)
_SOURCE = Path(os.environ.get(
    "WAREHOUSE_CROWD_V2_SOURCE", str(_DEFAULT_SOURCE)
)).expanduser()
if (
    _SOURCE != _PACKAGE_DIR
    and (_SOURCE / "crowd_templates.py").is_file()
    and (_SOURCE / "person_controllers.py").is_file()
):
    __path__.insert(0, str(_SOURCE))
