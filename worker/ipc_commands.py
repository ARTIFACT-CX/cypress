"""
AREA: worker · COMMANDS

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

SWAP: handler table. Adding a new command is one entry here plus one
function. If the handler table grows large we'll split per-category files
(model commands, audio commands, tool commands) and compose them here.
"""

import asyncio
import json
import sys
from typing import Any, Awaitable, Callable

# SETUP: shared worker state. Currently trivial — just tracks whether a
# model is loaded. Will grow once models actually load.
_state: dict[str, Any] = {
    "model": None,  # Name of the currently loaded model, or None.
}


Handler = Callable[[dict], Awaitable[dict]]
WriteFn = Callable[[dict], None]


async def _handle_status(_msg: dict) -> dict:
    # Mirror of what the host's UI needs to show: are we idle, or do we
    # have a model loaded?
    return {"ok": True, "model": _state["model"]}


async def _handle_load_model(msg: dict) -> dict:
    # STEP: parse and validate the target model name.
    name = msg.get("name")
    if not isinstance(name, str) or not name:
        return {"error": "missing or invalid 'name'"}

    # TODO: look up the model in models.REGISTRY, instantiate, call load().
    # For the scaffold we just record intent and return not-implemented so
    # the host + UI wiring can be tested end-to-end before any real weights
    # get loaded.
    return {"error": f"model '{name}' loader not yet implemented"}


async def _handle_unload(_msg: dict) -> dict:
    # TODO: real unload once loaders exist.
    _state["model"] = None
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
    # WHY: asyncio needs stdin wrapped as a StreamReader before we can
    # await readline(). Without this wrapper, reading stdin blocks the
    # entire event loop and nothing else (audio socket, timers) can run.
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    return reader


async def run_loop(write: WriteFn) -> None:
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
