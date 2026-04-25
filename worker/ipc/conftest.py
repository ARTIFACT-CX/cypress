"""
AREA: worker · IPC · TESTS · FIXTURES

Per-feature pytest fixtures. The IPC commands module keeps shared state at
module scope (_state, _write_fn, _registry) — singletons make sense in
production (one control loop per process) but tests need a clean slate
between runs. This conftest resets that state automatically.
"""

import pytest

from . import commands


@pytest.fixture(autouse=True)
def _reset_commands_state():
    # Snapshot the module-level singletons, run the test, then restore. The
    # autouse=True flag means every test in this package gets a fresh slate
    # without having to opt in.
    saved_state = dict(commands._state)
    saved_write = commands._write_fn
    saved_registry = commands._registry
    try:
        commands._state.update({"model": None, "instance": None, "device": None})
        commands._write_fn = None
        commands._registry = None
        yield
    finally:
        commands._state.clear()
        commands._state.update(saved_state)
        commands._write_fn = saved_write
        commands._registry = saved_registry
