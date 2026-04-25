"""
AREA: worker · IPC · TESTS

Unit tests for the command dispatcher. Real model loaders are swapped for
fakes — these run in milliseconds with no torch / HF dependency.
"""

import asyncio
import io
import json
from typing import Any, Callable

import pytest

from . import commands


# --- Fakes -------------------------------------------------------------------


class FakeModel:
    """Minimal Model lookalike. Loader behavior is configurable per-test
    via class attributes so individual tests can shape success/failure
    paths without subclassing."""

    name = "fake"
    load_should_raise: BaseException | None = None
    unload_should_raise: BaseException | None = None
    device_value: str | None = "cpu"

    def __init__(self, emit: Callable[[dict], None]):
        self._emit = emit
        self.loaded = False
        self.unloaded = False

    async def load(self) -> None:
        if self.load_should_raise is not None:
            raise self.load_should_raise
        self.loaded = True

    async def unload(self) -> None:
        if self.unload_should_raise is not None:
            raise self.unload_should_raise
        self.unloaded = True

    def device(self) -> str | None:
        return self.device_value


def make_fake_registry(**models: type) -> dict[str, type]:
    # The IPC layer expects a name → factory mapping. Concrete Model
    # subclasses satisfy the factory protocol via their __init__.
    return {name: cls for name, cls in models.items()}


# --- _handle_status ----------------------------------------------------------


async def test_status_returns_current_state():
    commands._state.update({"model": "moshi", "instance": object(), "device": "mps"})
    reply = await commands._handle_status({})
    assert reply == {"ok": True, "model": "moshi", "device": "mps"}


# --- _handle_load_model ------------------------------------------------------


async def test_load_model_rejects_missing_name():
    commands._registry = make_fake_registry()
    reply = await commands._handle_load_model({})
    assert "error" in reply
    assert "name" in reply["error"]


async def test_load_model_rejects_empty_name():
    commands._registry = make_fake_registry()
    reply = await commands._handle_load_model({"name": ""})
    assert "error" in reply


async def test_load_model_rejects_unknown_model():
    commands._registry = make_fake_registry(fake=FakeModel)
    reply = await commands._handle_load_model({"name": "nonexistent"})
    assert "error" in reply
    # Known-models hint should appear so the user can see what's available.
    assert "fake" in reply["error"]


async def test_load_model_success_sets_state():
    commands._registry = make_fake_registry(fake=FakeModel)
    commands._write_fn = lambda _msg: None  # emit_event no-op

    reply = await commands._handle_load_model({"name": "fake"})

    assert reply == {"ok": True, "model": "fake", "device": "cpu"}
    assert commands._state["model"] == "fake"
    assert commands._state["device"] == "cpu"
    assert commands._state["instance"] is not None


async def test_load_model_unloads_previous_first():
    # REASON: loading a new model while one is already resident must drop
    # the old one's VRAM first or we OOM on machines that can hold one
    # 7B model but not two.
    commands._registry = make_fake_registry(fake=FakeModel)
    commands._write_fn = lambda _msg: None

    await commands._handle_load_model({"name": "fake"})
    first = commands._state["instance"]

    await commands._handle_load_model({"name": "fake"})
    second = commands._state["instance"]

    assert first is not second
    assert first.unloaded is True


async def test_load_model_swallows_unload_failure():
    # An old model that can't unload cleanly shouldn't block a new load —
    # we're throwing it away anyway.
    class BadUnload(FakeModel):
        name = "bad"
        unload_should_raise = RuntimeError("oops")

    commands._registry = make_fake_registry(fake=FakeModel, bad=BadUnload)
    commands._write_fn = lambda _msg: None

    await commands._handle_load_model({"name": "bad"})
    reply = await commands._handle_load_model({"name": "fake"})

    assert reply.get("ok") is True


async def test_load_model_surfaces_loader_error_with_type():
    class Boom(FakeModel):
        name = "boom"
        load_should_raise = RuntimeError("disk full")

    commands._registry = make_fake_registry(boom=Boom)
    commands._write_fn = lambda _msg: None

    reply = await commands._handle_load_model({"name": "boom"})

    assert "error" in reply
    # The wrapper includes the exception class name — torch errors lose
    # half their meaning if you stringify them without the type.
    assert "RuntimeError" in reply["error"]
    assert "disk full" in reply["error"]
    # State must remain clean after a failed load — no half-loaded model.
    assert commands._state["model"] is None
    assert commands._state["instance"] is None


# --- _handle_unload ----------------------------------------------------------


async def test_unload_clears_state():
    commands._registry = make_fake_registry(fake=FakeModel)
    commands._write_fn = lambda _msg: None

    await commands._handle_load_model({"name": "fake"})
    reply = await commands._handle_unload({})

    assert reply == {"ok": True}
    assert commands._state["model"] is None
    assert commands._state["instance"] is None
    assert commands._state["device"] is None


async def test_unload_when_idle_is_noop():
    reply = await commands._handle_unload({})
    assert reply == {"ok": True}


async def test_unload_surfaces_error():
    class BadUnload(FakeModel):
        name = "bad"
        unload_should_raise = RuntimeError("stuck")

    commands._registry = make_fake_registry(bad=BadUnload)
    commands._write_fn = lambda _msg: None
    await commands._handle_load_model({"name": "bad"})

    reply = await commands._handle_unload({})
    assert "error" in reply
    assert "RuntimeError" in reply["error"]


# --- emit_event --------------------------------------------------------------


def test_emit_event_calls_write_fn_when_set():
    captured: list[dict] = []
    commands._write_fn = captured.append
    commands.emit_event({"event": "model_phase", "phase": "ready"})
    assert captured == [{"event": "model_phase", "phase": "ready"}]


def test_emit_event_is_silent_when_unwired():
    # No exception even when called before run_loop has wired _write_fn.
    commands._write_fn = None
    commands.emit_event({"event": "ignored"})


# --- _tag_id ----------------------------------------------------------------


def test_tag_id_echoes_when_present():
    out = commands._tag_id({"ok": True}, 42)
    assert out == {"id": 42, "ok": True}


def test_tag_id_omits_when_none():
    out = commands._tag_id({"ok": True}, None)
    assert "id" not in out


# --- run_loop dispatch -------------------------------------------------------


class _StubReader:
    """asyncio.StreamReader-shaped just enough for run_loop. Each readline
    awaits the next queued line; an empty bytes value signals EOF (which
    run_loop treats as host-closed-stdin → clean exit)."""

    def __init__(self, lines: list[bytes]):
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        for line in lines:
            self._queue.put_nowait(line)
        # Sentinel EOF so run_loop returns once the queue drains.
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


async def _drive_run_loop(lines: list[bytes], registry: dict) -> list[dict]:
    """Run run_loop with a stub stdin reader and capture every write.

    Patches _stdin_reader so we don't have to wire real OS pipes; the
    actual dispatch / parse / id-tagging logic still runs unchanged."""
    reader = _StubReader(lines)
    captured: list[dict] = []

    async def _fake_reader_factory():
        return reader

    original = commands._stdin_reader
    commands._stdin_reader = _fake_reader_factory  # type: ignore[assignment]
    try:
        await commands.run_loop(captured.append, registry)
    finally:
        commands._stdin_reader = original  # type: ignore[assignment]
    return captured


async def test_run_loop_handles_unknown_command():
    out = await _drive_run_loop(
        [json.dumps({"id": 1, "cmd": "nope"}).encode() + b"\n"],
        make_fake_registry(),
    )
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert "error" in out[0]
    assert "unknown command" in out[0]["error"]


async def test_run_loop_handles_malformed_json():
    out = await _drive_run_loop([b"{not json\n"], make_fake_registry())
    assert len(out) == 1
    assert "error" in out[0]


async def test_run_loop_dispatches_status():
    out = await _drive_run_loop(
        [json.dumps({"id": 7, "cmd": "status"}).encode() + b"\n"],
        make_fake_registry(),
    )
    assert out == [{"id": 7, "ok": True, "model": None, "device": None}]


async def test_run_loop_exits_on_shutdown():
    # The shutdown reply must be written before the loop returns so the
    # host's send() unblocks before reaping the process.
    out = await _drive_run_loop(
        [
            json.dumps({"id": 1, "cmd": "shutdown"}).encode() + b"\n",
            # This second line should never be processed — shutdown returns.
            json.dumps({"id": 2, "cmd": "status"}).encode() + b"\n",
        ],
        make_fake_registry(),
    )
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert out[0]["ok"] is True


async def test_run_loop_survives_handler_crash():
    # A handler exception must not kill the loop — host can recover by
    # sending a different command.
    class Crasher(FakeModel):
        name = "crasher"

        async def load(self):
            raise SystemError("kaboom")

    registry = make_fake_registry(crasher=Crasher)
    out = await _drive_run_loop(
        [
            json.dumps({"id": 1, "cmd": "load_model", "name": "crasher"}).encode() + b"\n",
            json.dumps({"id": 2, "cmd": "status"}).encode() + b"\n",
        ],
        registry,
    )
    assert len(out) == 2
    assert "error" in out[0]
    assert out[1]["id"] == 2
    assert out[1]["ok"] is True
