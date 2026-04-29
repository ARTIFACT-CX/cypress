"""
AREA: worker · IPC · GRPC

gRPC servicer that fronts the command dispatch table in commands.py.
One bidi RPC (Worker.Session) per Manager↔worker connection carries
every command, reply, and event multiplexed via the proto oneofs.

The shape of the dispatcher hasn't moved — handlers still take a dict
and return a dict, and emit_event() still pushes dicts. This module is
purely the I/O shell: it parses ClientMsg → (cmd, dict), runs the
handler, and converts the dict reply back into a typed ServerMsg. The
same conversion runs for unsolicited events.

SWAP: transport. Whether the underlying socket is a unix domain socket
(local subprocess) or TCP+TLS (remote worker) is decided by main.py
when it calls `serve()` with a `--listen` target. Everything below this
file is transport-agnostic.
"""

import asyncio
from typing import Any

import grpc

from . import commands
from .ports import ModelRegistry
from workerpb import worker_pb2 as pb
from workerpb import worker_pb2_grpc as pb_grpc


# --- Wire conversion ---------------------------------------------------------
#
# These functions are the only place protobuf types are mentioned outside
# the generated stubs. Keeping the conversion contained means the rest of
# the worker stays the same plain-dict shape that test_commands.py drives.


def _client_to_dict(msg: pb.ClientMsg) -> tuple[str | None, dict]:
    """Decode a ClientMsg's oneof payload into the (cmd_name, params) tuple
    the handler table expects. Unknown / unset oneof returns (None, {})."""
    which = msg.WhichOneof("payload")
    if which is None:
        return (None, {})
    payload = getattr(msg, which)
    if which in ("status", "unload", "shutdown", "start_stream", "stop_stream", "cancel_download"):
        return (which, {})
    if which == "load_model":
        return ("load_model", {"name": payload.name})
    if which == "run_wav":
        return ("run_wav", {"input": payload.input, "output": payload.output})
    if which == "audio_in":
        return ("audio_in", {"pcm": payload.pcm})
    if which == "download_model":
        params: dict = {
            "name": payload.name,
            "repo": payload.repo,
            "files": list(payload.files),
        }
        if payload.HasField("revision"):
            params["revision"] = payload.revision
        return ("download_model", params)
    return (which, {})


def _reply_to_proto(msg_id: int, cmd: str | None, reply: dict) -> pb.Reply:
    """Convert a handler's dict reply into a typed Reply proto. Errors
    short-circuit into the string error field; the rest pick the
    cmd-appropriate OkReply variant."""
    if "error" in reply:
        return pb.Reply(id=msg_id, error=str(reply["error"]))
    if cmd == "status":
        kw: dict[str, Any] = {}
        if reply.get("model") is not None:
            kw["model"] = reply["model"]
        if reply.get("device") is not None:
            kw["device"] = reply["device"]
        return pb.Reply(id=msg_id, status=pb.StatusOk(**kw))
    if cmd == "load_model":
        kw = {"model": reply.get("model", "")}
        if reply.get("device") is not None:
            kw["device"] = reply["device"]
        return pb.Reply(id=msg_id, load_model=pb.LoadModelOk(**kw))
    if cmd == "start_stream":
        # sample_rate is optional in the dict; default 0 means "unknown,"
        # which the Go side interprets as "fall back to model default."
        return pb.Reply(
            id=msg_id,
            start_stream=pb.StartStreamOk(sample_rate=int(reply.get("sample_rate") or 0)),
        )
    if cmd == "download_model":
        return pb.Reply(
            id=msg_id,
            download_started=pb.DownloadStartedOk(started=bool(reply.get("started", True))),
        )
    if cmd == "cancel_download":
        return pb.Reply(
            id=msg_id,
            cancel_download=pb.CancelDownloadOk(active=bool(reply.get("active", False))),
        )
    # unload / shutdown / stop_stream / audio_in / run_wav: no typed
    # payload, just an ack.
    return pb.Reply(id=msg_id, ok=pb.OkEmpty())


def _event_to_proto(msg: dict) -> pb.Event | None:
    """Convert an emit_event() dict into a typed Event proto. Returns
    None for unrecognized event names (logged upstream rather than sent)."""
    event = msg.get("event")
    if event == "model_phase":
        kw: dict[str, Any] = {"phase": msg.get("phase", "")}
        if msg.get("device") is not None:
            kw["device"] = msg["device"]
        return pb.Event(model_phase=pb.ModelPhase(**kw))
    if event == "audio_out":
        return pb.Event(
            audio_out=pb.AudioOut(
                pcm=msg.get("pcm", b"") or b"",
                text=msg.get("text") or "",
            )
        )
    if event == "stream_error":
        return pb.Event(stream_error=pb.StreamError(error=msg.get("error", "")))
    if event == "download_progress":
        return pb.Event(
            download_progress=pb.DownloadProgress(
                name=msg.get("name", ""),
                phase=msg.get("phase", ""),
                downloaded=int(msg.get("downloaded", 0)),
                total=int(msg.get("total", 0)),
                file=msg.get("file", ""),
                file_index=int(msg.get("fileIndex", 0)),
                file_count=int(msg.get("fileCount", 0)),
            )
        )
    if event == "download_done":
        kw = {
            "name": msg.get("name", ""),
            "repo": msg.get("repo", ""),
            "files": list(msg.get("files", [])),
            "total_bytes": int(msg.get("totalBytes", 0)),
        }
        if msg.get("revision") is not None:
            kw["revision"] = msg["revision"]
        return pb.Event(download_done=pb.DownloadDone(**kw))
    if event == "download_error":
        return pb.Event(
            download_error=pb.DownloadError(
                name=msg.get("name", ""),
                error=msg.get("error", ""),
            )
        )
    return None


# --- Servicer ----------------------------------------------------------------


class WorkerServicer(pb_grpc.WorkerServicer):
    """Implements the Worker service. Holds the model registry plus a
    shutdown event the composition root waits on so it can stop the
    gRPC server cleanly when a client sends a shutdown command."""

    def __init__(self, registry: ModelRegistry, shutdown_event: asyncio.Event):
        self._registry = registry
        self._shutdown = shutdown_event

    async def Session(self, request_iterator, context):
        # Per-session outbox merges replies and events into the single
        # response stream. asyncio.Queue's None sentinel signals the
        # generator to terminate (used after a shutdown reply lands).
        outbox: asyncio.Queue[pb.ServerMsg | None] = asyncio.Queue()

        # Wire emit_event into our outbox. emit_event runs synchronously
        # from handler code, so put_nowait is fine — the queue is
        # unbounded by design (events are small protobufs and the
        # consumer is the same coroutine that yields them).
        def _on_event(msg: dict) -> None:
            evt = _event_to_proto(msg)
            if evt is not None:
                outbox.put_nowait(pb.ServerMsg(event=evt))

        commands.configure(_on_event, self._registry)

        # First message after stream open per the proto contract — the
        # Go host won't dispatch any commands until it sees this.
        yield pb.ServerMsg(handshake=pb.Handshake(ready=True))

        async def _consume():
            # One ClientMsg at a time keeps reply ordering tied to send
            # ordering. Handlers that schedule background work (download,
            # stream drain) return promptly so this never bottlenecks.
            async for client_msg in request_iterator:
                cmd, params = _client_to_dict(client_msg)
                handler = commands._HANDLERS.get(cmd) if cmd else None
                if handler is None:
                    reply = {"error": f"unknown command: {cmd!r}"}
                else:
                    try:
                        reply = await handler(params)
                    except Exception as e:
                        reply = {"error": f"handler crashed: {e!r}"}
                outbox.put_nowait(
                    pb.ServerMsg(reply=_reply_to_proto(client_msg.id, cmd, reply))
                )
                if cmd == "shutdown":
                    # Reply is queued; signal the outbox generator to
                    # exit and the composition root to stop the server.
                    outbox.put_nowait(None)
                    self._shutdown.set()
                    return

        consumer = asyncio.create_task(_consume(), name="grpc-session-consumer")
        try:
            while True:
                item = await outbox.get()
                if item is None:
                    return
                yield item
        finally:
            consumer.cancel()
            # Tear down any in-flight stream so a reconnect sees a clean
            # slate. download tasks can keep running across reconnects
            # (they're tied to the worker process, not the session) so
            # we deliberately leave _state["download"] alone.
            await commands._stop_active_stream()


async def serve(listen: str, registry: ModelRegistry) -> None:
    """Boot the gRPC server, bind the listen target, and block until a
    client requests shutdown. `listen` is one of:

      unix:<path>          local subprocess (file perms = auth)
      tcp://host:port      remote worker (TLS / auth wired later)

    Returns once the server has fully stopped."""
    shutdown_event = asyncio.Event()
    server = grpc.aio.server()
    pb_grpc.add_WorkerServicer_to_server(
        WorkerServicer(registry, shutdown_event), server
    )

    # REASON: gRPC accepts unix paths in the form `unix:<path>` but TCP
    # targets as bare host:port. Translate our scheme prefix here so the
    # CLI surface stays uniform across local and remote.
    if listen.startswith("unix:"):
        bind = listen
    elif listen.startswith("tcp://"):
        bind = listen[len("tcp://") :]
    else:
        raise ValueError(f"unsupported listen target {listen!r}; use unix:<path> or tcp://host:port")
    server.add_insecure_port(bind)

    await server.start()
    try:
        # Wait for either a shutdown command from the client or an
        # external signal (e.g. SIGTERM from the host) — whichever comes
        # first ends the loop.
        await shutdown_event.wait()
    finally:
        # Brief grace period so the in-flight shutdown reply gets
        # serialized onto the wire before we close the listener.
        await server.stop(grace=1.0)
