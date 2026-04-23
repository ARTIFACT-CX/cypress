"""
AREA: worker · AUDIO

Unix-domain socket server that carries binary audio frames between the Go
orchestration server and the Python worker. Data format is raw PCM 16-bit
mono @ 24kHz — matching Moshi's native sample rate.

The scaffold intentionally does no real audio processing; it just accepts a
connection, drains incoming bytes, and logs. Real routing (frames → model
→ frames back) lands when the first model implementation arrives.

SWAP: transport. Unix sockets are local-only and fast. If we ever need the
worker to run on a different machine (remote inference pool), swap this for
a TCP or gRPC channel behind the same start_server/connection shape.
"""

import asyncio
import os
import sys


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    # STEP 1: log that the host connected. During dev this is the main
    # signal that the audio channel is wired up end-to-end.
    peer = writer.get_extra_info("peername", "audio-peer")
    print(f"[worker.audio] client connected: {peer}", file=sys.stderr, flush=True)

    # STEP 2: drain the channel. We read whatever comes in and discard it
    # until the host closes the connection. TODO: forward frames into the
    # model pipeline and write generated frames back on `writer`.
    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            # TODO: feed chunk into the audio pipeline. For now we just
            # acknowledge by counting bytes seen.
            _ = chunk
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        print("[worker.audio] client disconnected", file=sys.stderr, flush=True)


async def start_server(path: str) -> asyncio.AbstractServer:
    # SAFETY: remove any stale socket file left behind by a previous crash.
    # Without this, asyncio.start_unix_server fails with EADDRINUSE and the
    # worker can't come up.
    if os.path.exists(path):
        os.unlink(path)

    server = await asyncio.start_unix_server(_handle_connection, path=path)
    return server
