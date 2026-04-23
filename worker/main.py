"""
AREA: worker · ENTRY

Cypress Python inference worker. Launched as a subprocess by the Go
orchestration server. Speaks two channels:

  - stdin / stdout : JSON-line control protocol (see ipc_commands.py)
  - unix socket    : binary audio frames, path announced in handshake

On startup it opens the audio socket, prints a ready handshake on stdout,
and then runs the control loop until the host closes stdin or sends
`{"cmd":"shutdown"}`. Any uncaught error is reported on stdout as
`{"fatal": "..."}` so the host can surface it rather than silently exiting.
"""

import asyncio
import json
import os
import sys
import traceback

import audio
import ipc_commands


# SETUP: the unix socket path is derived from our pid so multiple workers
# can coexist during dev (e.g. one orphaned from a prior crash plus a new
# one the user just launched). The host reads this path from the handshake.
SOCKET_DIR = "/tmp"


def _socket_path() -> str:
    return os.path.join(SOCKET_DIR, f"cypress-{os.getpid()}.sock")


def _write(msg: dict) -> None:
    # WHY: stdout is line-buffered by default when attached to a pipe in
    # Python 3.11+, but flush explicitly so the host sees each reply
    # immediately. Any delay here shows up as apparent hangs on the Go side.
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


async def _run() -> None:
    # STEP 1: open the audio socket. Must be ready before we send the
    # handshake because the host may connect as soon as it reads the path.
    sock_path = _socket_path()
    audio_server = await audio.start_server(sock_path)

    # STEP 2: announce ready. Once this line is on stdout the host knows
    # where to connect for audio and can start sending control commands.
    _write({"ready": True, "audio_socket": sock_path})

    # STEP 3: run the control loop. Returns when stdin EOFs or a shutdown
    # command is received.
    try:
        await ipc_commands.run_loop(_write)
    finally:
        # STEP 4: clean up the audio socket even on error paths so we don't
        # leave stale sockets in /tmp for the next run.
        audio_server.close()
        await audio_server.wait_closed()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass


def main() -> None:
    try:
        asyncio.run(_run())
    except Exception:
        # WHY: surface fatal startup errors through the same stdout channel
        # the host is already reading. If we just raised, the host would see
        # an unexplained non-zero exit.
        _write({"fatal": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
