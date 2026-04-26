"""
AREA: worker · IPC · TESTS

Unit tests for the command dispatcher. Real model loaders are swapped for
fakes — these run in milliseconds with no torch / HF dependency.
"""

import asyncio
import base64
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


# --- _handle_run_wav ---------------------------------------------------------


class FakeStreamingModel(FakeModel):
    """FakeModel + a run_wav method so the streaming-handler tests exercise
    the success path without real torch / moshi."""

    name = "streaming"
    run_wav_returns: dict | None = None
    run_wav_should_raise: BaseException | None = None
    run_wav_calls: list[tuple[str, str]] = []

    def run_wav(self, input_path: str, output_path: str) -> dict:
        self.run_wav_calls.append((input_path, output_path))
        if self.run_wav_should_raise is not None:
            raise self.run_wav_should_raise
        return self.run_wav_returns or {"frames": 100, "duration_s": 1.0}


async def test_run_wav_rejects_when_no_model_loaded():
    reply = await commands._handle_run_wav({"input": "in.wav", "output": "out.wav"})
    assert "error" in reply
    assert "no model" in reply["error"]


async def test_run_wav_rejects_missing_input_path():
    commands._state["instance"] = FakeStreamingModel(emit=lambda _m: None)
    reply = await commands._handle_run_wav({"output": "out.wav"})
    assert "error" in reply
    assert "input" in reply["error"]


async def test_run_wav_rejects_missing_output_path():
    commands._state["instance"] = FakeStreamingModel(emit=lambda _m: None)
    reply = await commands._handle_run_wav({"input": "in.wav"})
    assert "error" in reply
    assert "output" in reply["error"]


async def test_run_wav_rejects_empty_paths():
    commands._state["instance"] = FakeStreamingModel(emit=lambda _m: None)
    reply = await commands._handle_run_wav({"input": "", "output": "out.wav"})
    assert "error" in reply


async def test_run_wav_rejects_when_model_lacks_capability():
    # FakeModel (without run_wav) must not be silently accepted — the
    # handler's hasattr gate is what keeps a future TTS-only model from
    # crashing inside torch.
    commands._state["model"] = "fake"
    commands._state["instance"] = FakeModel(emit=lambda _m: None)
    reply = await commands._handle_run_wav({"input": "in.wav", "output": "out.wav"})
    assert "error" in reply
    assert "run_wav" in reply["error"]


async def test_run_wav_success_returns_metadata():
    fake = FakeStreamingModel(emit=lambda _m: None)
    fake.run_wav_returns = {"frames": 24000, "duration_s": 1.0, "device": "mps"}
    commands._state["instance"] = fake

    reply = await commands._handle_run_wav({"input": "in.wav", "output": "out.wav"})

    assert reply["ok"] is True
    assert reply["frames"] == 24000
    assert reply["device"] == "mps"
    # Confirm the handler actually invoked the model with the paths it
    # received — easy to break if the kwargs get re-shuffled.
    assert fake.run_wav_calls[-1] == ("in.wav", "out.wav")


async def test_run_wav_surfaces_model_exception_with_type():
    fake = FakeStreamingModel(emit=lambda _m: None)
    fake.run_wav_should_raise = FileNotFoundError("in.wav: no such file")
    commands._state["instance"] = fake

    reply = await commands._handle_run_wav({"input": "in.wav", "output": "out.wav"})

    assert "error" in reply
    # Type prefix matters — the host UI surfaces it directly, and "permission
    # denied" reads very differently from "FileNotFoundError".
    assert "FileNotFoundError" in reply["error"]


# --- streaming handlers ------------------------------------------------------


class FakeChunk:
    """StreamChunk lookalike: the real one is a NamedTuple but the IPC
    layer only reads .audio_pcm and .text, so a tiny attrs-only object
    is enough — keeps these tests free of any models/ import."""

    def __init__(self, audio_pcm: bytes, text: str | None):
        self.audio_pcm = audio_pcm
        self.text = text


class FakeSession:
    """MoshiStream lookalike: start/feed/aclose + async iteration. Tests
    push chunks via emit_chunk so we can assert exactly what the drain
    task surfaces as audio_out events.

    Concurrency-conscious: async iteration parks on _outbox.get() until a
    chunk arrives or aclose() pushes the EOF sentinel."""

    sample_rate = 24000
    _EOF = object()

    def __init__(self):
        self.started = False
        self.closed = False
        self.fed: list[bytes] = []
        self._outbox: asyncio.Queue = asyncio.Queue()
        self.feed_should_raise: BaseException | None = None
        self.start_should_raise: BaseException | None = None

    async def start(self) -> None:
        if self.start_should_raise is not None:
            raise self.start_should_raise
        self.started = True

    async def feed(self, pcm: bytes) -> None:
        if self.feed_should_raise is not None:
            raise self.feed_should_raise
        self.fed.append(pcm)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._outbox.get()
        if item is FakeSession._EOF:
            raise StopAsyncIteration
        return item

    async def emit_chunk(self, chunk: FakeChunk) -> None:
        # Helper for tests: inject a chunk that the drain task will pick up.
        await self._outbox.put(chunk)

    async def aclose(self) -> None:
        self.closed = True
        # Unblock any pending __anext__ so the drain task exits cleanly.
        await self._outbox.put(FakeSession._EOF)


class FakeStreamingSessionModel(FakeStreamingModel):
    """FakeStreamingModel + a stream() method. Separate class so tests
    that exercise capability gating (model exists but lacks stream)
    still have a clean fake to point at."""

    name = "streaming-session"

    def __init__(self, emit):
        super().__init__(emit)
        self.session = FakeSession()
        self.stream_calls = 0

    def stream(self) -> FakeSession:
        self.stream_calls += 1
        return self.session


# start_stream


async def test_start_stream_rejects_when_no_model_loaded():
    reply = await commands._handle_start_stream({})
    assert "error" in reply
    assert "no model" in reply["error"]


async def test_start_stream_rejects_when_model_lacks_stream():
    # FakeModel has no stream() — must surface the capability gap rather
    # than crashing on AttributeError.
    commands._state["instance"] = FakeModel(emit=lambda _m: None)
    commands._state["model"] = "fake"
    reply = await commands._handle_start_stream({})
    assert "error" in reply
    assert "stream" in reply["error"]


async def test_start_stream_rejects_when_already_active():
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    first = await commands._handle_start_stream({})
    assert first["ok"] is True
    try:
        second = await commands._handle_start_stream({})
        assert "error" in second
        assert "already active" in second["error"]
    finally:
        await commands._stop_active_stream()


async def test_start_stream_returns_sample_rate_and_starts_session():
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    try:
        reply = await commands._handle_start_stream({})
        assert reply["ok"] is True
        assert reply["sample_rate"] == 24000
        assert fake.session.started is True
        assert commands._state["stream"] is fake.session
    finally:
        await commands._stop_active_stream()


async def test_start_stream_surfaces_session_start_failure():
    # If session.start() blows up (mimi cache alloc failed, say) we must
    # not leave a half-initialized session in _state.
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    fake.session.start_should_raise = RuntimeError("alloc failed")
    commands._state["instance"] = fake

    reply = await commands._handle_start_stream({})

    assert "error" in reply
    assert "RuntimeError" in reply["error"]
    assert commands._state["stream"] is None


# audio_in


async def test_audio_in_rejects_when_no_active_stream():
    reply = await commands._handle_audio_in({"pcm": base64.b64encode(b"\x00\x00").decode()})
    assert "error" in reply
    assert "no active stream" in reply["error"]


async def test_audio_in_rejects_missing_pcm_field():
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    await commands._handle_start_stream({})
    try:
        reply = await commands._handle_audio_in({})
        assert "error" in reply
        assert "pcm" in reply["error"]
    finally:
        await commands._stop_active_stream()


async def test_audio_in_rejects_invalid_base64():
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    await commands._handle_start_stream({})
    try:
        reply = await commands._handle_audio_in({"pcm": "not!!base64!!"})
        assert "error" in reply
        assert "base64" in reply["error"]
    finally:
        await commands._stop_active_stream()


async def test_audio_in_decodes_and_feeds_session():
    raw = b"\x01\x00\x02\x00\x03\x00"
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    await commands._handle_start_stream({})
    try:
        reply = await commands._handle_audio_in({"pcm": base64.b64encode(raw).decode()})
        assert reply == {"ok": True}
        assert fake.session.fed == [raw]
    finally:
        await commands._stop_active_stream()


# stop_stream


async def test_stop_stream_idempotent_when_inactive():
    # Defensive call from the host (e.g. on disconnect) must not error.
    reply = await commands._handle_stop_stream({})
    assert reply == {"ok": True}


async def test_stop_stream_closes_active_session():
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    await commands._handle_start_stream({})

    reply = await commands._handle_stop_stream({})

    assert reply == {"ok": True}
    assert fake.session.closed is True
    assert commands._state["stream"] is None
    assert commands._state["stream_drain"] is None


# drain task → audio_out events


async def test_drain_task_emits_audio_out_events_with_base64():
    captured: list[dict] = []
    commands._write_fn = captured.append
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    await commands._handle_start_stream({})
    try:
        await fake.session.emit_chunk(FakeChunk(audio_pcm=b"\xaa\xbb", text="hi"))
        # Yield to let the drain task pick it up. Loop briefly because
        # task scheduling order isn't guaranteed in a single sleep(0).
        for _ in range(20):
            await asyncio.sleep(0)
            if captured:
                break
    finally:
        await commands._stop_active_stream()

    assert len(captured) >= 1
    evt = captured[0]
    assert evt["event"] == "audio_out"
    assert base64.b64decode(evt["pcm"]) == b"\xaa\xbb"
    assert evt["text"] == "hi"


async def test_drain_task_passes_none_text_through():
    # Pad-token frames have text=None; the host renders text only when
    # present. The event still fires (audio is on every frame).
    captured: list[dict] = []
    commands._write_fn = captured.append
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    await commands._handle_start_stream({})
    try:
        await fake.session.emit_chunk(FakeChunk(audio_pcm=b"\x00\x00", text=None))
        for _ in range(20):
            await asyncio.sleep(0)
            if captured:
                break
    finally:
        await commands._stop_active_stream()

    assert captured[0]["text"] is None


# Cleanup interactions with load_model / unload


async def test_unload_stops_active_stream():
    # Critical: stream holds references to the model's mimi/lm_gen. If we
    # unloaded without closing the session, the next chunk it processed
    # would crash on a torn-down model.
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    commands._state["model"] = "streaming-session"
    await commands._handle_start_stream({})

    await commands._handle_unload({})

    assert fake.session.closed is True
    assert commands._state["stream"] is None
    assert commands._state["instance"] is None


async def test_load_model_stops_active_stream_before_swap():
    # Same hazard as unload: a streamed-against model getting swapped out
    # must not leave a session iterating into a freed mimi.
    fake = FakeStreamingSessionModel(emit=lambda _m: None)
    commands._state["instance"] = fake
    commands._state["model"] = "streaming-session"
    await commands._handle_start_stream({})

    commands._registry = make_fake_registry(fake=FakeModel)
    reply = await commands._handle_load_model({"name": "fake"})

    assert reply["ok"] is True
    assert fake.session.closed is True
    assert commands._state["stream"] is None


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


# --- _handle_download_model -------------------------------------------------


class _FakeHubInstaller:
    """Stands in for huggingface_hub. Records calls and returns paths
    inside the test's tmp dir so the download command can `os.path.getsize`
    them without touching the network."""

    def __init__(self, tmp_path, file_size: int = 100):
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, str]] = []
        self.file_size = file_size

    def hf_hub_download(self, repo: str, filename: str, **_kw):
        # Mirror HF's contract: returns the on-disk path of the file
        # after fetching it. We just write a stub so getsize works.
        self.calls.append((repo, filename))
        path = self.tmp_path / repo.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * self.file_size)
        return str(path)

    class _Sibling:
        def __init__(self, name: str, size: int):
            self.rfilename = name
            self.size = size

    def model_info(self, repo: str, **_kw):
        # Return a stub object whose .siblings list mirrors HF's shape.
        sibs = [self._Sibling("a", self.file_size), self._Sibling("b", self.file_size)]
        return type("Info", (), {"siblings": sibs})()


def _install_fake_hub(monkeypatch, fake: _FakeHubInstaller):
    # huggingface_hub is imported inside the download task, not at the top
    # of commands.py — so we have to inject a fake module before the task
    # runs. sys.modules wins over real disk imports.
    import sys, types

    mod = types.ModuleType("huggingface_hub")
    mod.hf_hub_download = fake.hf_hub_download
    mod.HfApi = lambda: types.SimpleNamespace(model_info=fake.model_info)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


async def test_download_model_rejects_missing_name():
    reply = await commands._handle_download_model({"repo": "x", "files": ["a"]})
    assert "error" in reply


async def test_download_model_rejects_missing_repo():
    reply = await commands._handle_download_model({"name": "moshi", "files": ["a"]})
    assert "error" in reply


async def test_download_model_rejects_empty_files():
    reply = await commands._handle_download_model(
        {"name": "moshi", "repo": "x", "files": []}
    )
    assert "error" in reply


async def test_download_model_emits_progress_and_done(tmp_path, monkeypatch):
    # Capture every event the command emits so we can assert the shape +
    # ordering of progress / done events.
    events: list[dict] = []
    commands._write_fn = lambda msg: events.append(msg)
    fake = _FakeHubInstaller(tmp_path, file_size=100)
    _install_fake_hub(monkeypatch, fake)

    reply = await commands._handle_download_model(
        {"name": "moshi", "repo": "kyutai/x", "files": ["a", "b"]}
    )
    assert reply["ok"] is True

    # Wait for the spawned task to finish so all events are in.
    task = commands._state["download"]
    await task

    # Find the events we care about.
    progress = [e for e in events if e["event"] == "download_progress"]
    done = [e for e in events if e["event"] == "download_done"]

    # At least one starting + one per-file progress + one done.
    assert any(p["phase"] == "starting" for p in progress)
    assert any(p["file"] == "a" for p in progress)
    assert any(p["file"] == "b" for p in progress)
    assert len(done) == 1
    assert done[0]["totalBytes"] == 200  # two 100-byte files
    assert len(done[0]["files"]) == 2
    assert fake.calls == [("kyutai/x", "a"), ("kyutai/x", "b")]


async def test_download_model_rejects_concurrent_runs():
    # The "already in progress" check is structural — it just looks at
    # whether _state["download"] is a non-done task. Inject a not-yet-done
    # future directly so we don't have to race a real download to observe
    # the rejection.
    pending: asyncio.Future = asyncio.get_event_loop().create_future()
    commands._state["download"] = pending
    try:
        reply = await commands._handle_download_model(
            {"name": "moshi", "repo": "kyutai/x", "files": ["a"]}
        )
        assert "error" in reply
        assert "already" in reply["error"]
    finally:
        pending.set_result(None)
        commands._state["download"] = None


async def test_download_model_emits_error_on_hf_failure(tmp_path, monkeypatch):
    events: list[dict] = []
    commands._write_fn = lambda msg: events.append(msg)
    fake = _FakeHubInstaller(tmp_path)
    fake.hf_hub_download = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("network down")
    )
    _install_fake_hub(monkeypatch, fake)

    await commands._handle_download_model(
        {"name": "moshi", "repo": "kyutai/x", "files": ["a"]}
    )
    await commands._state["download"]

    errs = [e for e in events if e["event"] == "download_error"]
    assert len(errs) == 1
    assert "network down" in errs[0]["error"]


async def test_cancel_download_is_noop_when_idle():
    commands._state["download"] = None
    reply = await commands._handle_cancel_download({})
    assert reply == {"ok": True, "active": False}
