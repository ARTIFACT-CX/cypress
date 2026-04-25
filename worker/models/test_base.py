"""
AREA: worker · MODELS · TESTS

Unit tests for the model registry and @register decorator. The registry
itself is global by design (one process, one registry), so tests
snapshot/restore it to stay isolated.
"""

import pytest

from . import base


@pytest.fixture(autouse=True)
def _isolate_registry():
    # Snapshot the global REGISTRY and restore it after each test so that
    # registering test-only model classes doesn't leak into later tests.
    saved = dict(base.REGISTRY)
    try:
        yield
    finally:
        base.REGISTRY.clear()
        base.REGISTRY.update(saved)


def test_register_populates_registry():
    @base.register
    class Dummy(base.Model):
        name = "dummy"

        async def load(self) -> None:
            return None

        async def unload(self) -> None:
            return None

    assert base.REGISTRY["dummy"] is Dummy


def test_register_rejects_empty_name():
    # Empty name would shadow whatever's at REGISTRY[""] silently — much
    # better to fail loudly at import time.
    with pytest.raises(ValueError):

        @base.register
        class Nameless(base.Model):
            async def load(self) -> None:
                return None

            async def unload(self) -> None:
                return None


def test_register_returns_class_unchanged():
    # The decorator must be transparent — `@register` shouldn't replace
    # the class with a wrapper, since users still subclass and instantiate it.
    class Pass(base.Model):
        name = "pass-through"

        async def load(self) -> None:
            return None

        async def unload(self) -> None:
            return None

    decorated = base.register(Pass)
    assert decorated is Pass
