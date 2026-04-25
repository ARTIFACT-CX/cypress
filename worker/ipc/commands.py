"""
AREA: worker · IPC · COMMANDS

Reads JSON-line commands from stdin and writes JSON-line replies via the
`write` callback supplied by main.py. One command = one reply, always.

Command shape:
    {"id": <opaque>, "cmd": "<name>", ...params}

Reply shape:
    {"id": <same>, "ok": true, ...result}
    {"id": <same>, "error": "<message>"}

The `id` field is optional but the host typically sets it to correlate
async requests. We echo it back verbatim so the host can match replies to
outstanding requests.

Cross-feature boundary: this module never imports from `models` or any
other feature directly. Instead it accepts a `ModelRegistry` at startup
(see ports.py) and looks models up through that. The composition root
(main.py) is the only place that knows about both ipc and models.

SWAP: handler table. Adding a new command is one entry here plus one
function. If the handler table grows large we'll split per-category
modules (model commands, audio commands, tool commands) and compose them.
"""

import asyncio
import json
import sys
from typing import Any, Awaitable, Callable

from .ports import ModelRegistry

# SETUP: shared worker state. The model instance is kept so load_model
# can cleanly unload the previous one before swapping. device is surfaced
# to /status so the UI can show "Running on MPS" etc.
_state: dict[str, Any] = {
    "model": None,  # Name of the currently loaded model, or None.
    "instance": None,  # Live Model subclass instance, or None.
    "device": None,  # Device string ("mps", "cuda", "cpu") or None.
}

# SETUP: dependencies wired in by run_loop. Stashing them as module
# globals (rather than threading them through every handler signature)
# matches the singleton nature of the worker — there is exactly one
# control loop per process, so per-call injection would be ceremony.
_write_fn: "WriteFn | None" = None
_registry: ModelRegistry | None = None


Handler = Callable[[dict], Awaitable[dict]]
WriteFn = Callable[[dict], None]


def emit_event(msg: dict) -> None:
    # Events are worker→host one-way messages with no id. The host
    # readLoop routes them into a Manager event handler rather than any
    # waiter channel. Keeping them out-of-band means they never collide
    # with pending request/reply pairs. Public so other features (audio,
    # eventually session) can push events without depending on commands'
    # internals.
    if _write_fn is not None:
        _write_fn(msg)


async def _handle_status(_msg: dict) -> dict:
    # Mirror of what the host's UI needs to show: are we idle, or do we
    # have a model loaded, and on what device?
    return {
        "ok": True,
        "model": _state["model"],
        "device": _state["device"],
    }


async def _handle_load_model(msg: dict) -> dict:
    # STEP 1: validate the target name.
    name = msg.get("name")
    if not isinstance(name, str) or not name:
        return {"error": "missing or invalid 'name'"}

    assert _registry is not None, "registry not wired; call run_loop first"
    cls = _registry.get(name)
    if cls is None:
        known = ", ".join(sorted(_registry.keys())) or "(none registered)"
        return {"error": f"unknown model {name!r}; known: {known}"}

    # STEP 2: unload any previously loaded model so we don't double up
    # on VRAM. Swallow errors from the old one — we're throwing it away
    # anyway, and a failed unload shouldn't block a new load.
    prev = _state["instance"]
    if prev is not None:
        try:
            await prev.unload()
        except Exception:
            pass
        _state["model"] = None
        _state["instance"] = None
        _state["device"] = None

    # STEP 3: instantiate and load. The loader handles its own phase
    # events via emit_event so the host can relay them to the UI.
    instance = cls(emit_event)
    try:
        await instance.load()
    except Exception as e:
        # Include the repr — torch / HF errors often have type info in
        # the class name that's lost by str(e) alone.
        return {"error": f"{type(e).__name__}: {e}"}

    _state["model"] = name
    _state["instance"] = instance
    _state["device"] = instance.device() if hasattr(instance, "device") else None
    return {"ok": True, "model": name, "device": _state["device"]}


async def _handle_unload(_msg: dict) -> dict:
    prev = _state["instance"]
    if prev is not None:
        try:
            await prev.unload()
        except Exception as e:
            return {"error": f"unload failed: {type(e).__name__}: {e}"}
    _state["model"] = None
    _state["instance"] = None
    _state["device"] = None
    return {"ok": True}


async def _handle_shutdown(_msg: dict) -> dict:
    # The reply is sent before the loop exits (see run_loop below).
    return {"ok": True, "bye": True}


_HANDLERS: dict[str, Handler] = {
    "status": _handle_status,
    "load_model": _handle_load_model,
    "unload": _handle_unload,
    "shutdown": _handle_shutdown,
}


async def _stdin_reader() -> asyncio.StreamReader:
    # REASON: asyncio needs stdin wrapped as a StreamReader before we can
    # await readline(). Without this wrapper, reading stdin blocks the
    # entire event loop and nothing else (audio socket, timers) can run.
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    return reader


async def run_loop(write: WriteFn, registry: ModelRegistry) -> None:
    # Stash the wired dependencies. run_loop is the single entry point,
    # so doing it once here covers every downstream caller.
    global _write_fn, _registry
    _write_fn = write
    _registry = registry

    reader = await _stdin_reader()

    while True:
        # STEP 1: read one newline-delimited command from the host. An
        # empty read means the host closed stdin — treat as shutdown.
        line = await reader.readline()
        if not line:
            return

        # STEP 2: parse JSON. Malformed input is reported but does not
        # terminate the loop — the host may send more valid commands next.
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            write({"error": f"invalid json: {e}"})
            continue

        cmd = msg.get("cmd")
        msg_id = msg.get("id")

        # STEP 3: dispatch.
        handler = _HANDLERS.get(cmd)
        if handler is None:
            write(_tag_id({"error": f"unknown command: {cmd!r}"}, msg_id))
            continue

        # STEP 4: run the handler. Catch broadly so a buggy handler never
        # takes down the worker; the host sees the error and can decide
        # whether to retry, send a new command, or restart the worker.
        try:
            reply = await handler(msg)
        except Exception as e:
            reply = {"error": f"handler crashed: {e!r}"}

        write(_tag_id(reply, msg_id))

        # STEP 5: shutdown is special — its reply goes out *before* we
        # return, so the host gets its ack and can proceed to join the
        # process cleanly.
        if cmd == "shutdown":
            return


def _tag_id(reply: dict, msg_id) -> dict:
    # Echo the caller's correlation id back if they sent one. Keeps the
    # reply shape deterministic for the host's request/reply matcher.
    if msg_id is not None:
        reply = {"id": msg_id, **reply}
    return reply
