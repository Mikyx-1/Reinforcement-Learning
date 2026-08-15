"""Put the repo root on sys.path for pytest.

This repo is run from a clean checkout, not installed as a package, so top-level
imports (`agents.*`, `common.*`, `training.*`, ...) need the root on the path.
The scripts under scripts/ each do this themselves; pytest collects tests/ from a
different base dir, so it needs this conftest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
