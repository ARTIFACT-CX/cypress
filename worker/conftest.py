"""
AREA: worker · TESTS · BOOTSTRAP

Top-level conftest: makes the proto-generated `workerpb` package importable
during pytest runs without going through `worker/main.py`. Mirrors the
sys.path shim main.py does in production.

Pytest collects this file before any test module is imported, so the path
is in place by the time `worker.ipc.server` (which imports workerpb) gets
pulled in.
"""

import pathlib
import sys

_PROTO_PY = pathlib.Path(__file__).resolve().parent.parent / "proto" / "dist" / "python"
if _PROTO_PY.is_dir():
    sys.path.insert(0, str(_PROTO_PY))
