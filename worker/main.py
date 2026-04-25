"""
AREA: worker · ENTRY · COMPOSITION-ROOT

Cypress Python inference worker. Launched as a subprocess by the Go
orchestration server. Speaks two channels:

  - stdin / stdout : JSON-line control protocol (see ipc/commands.py)
  - unix socket    : binary audio frames, path announced in handshake

This file is the composition root: it imports each feature, wires them
together, and starts the run loop. Cross-feature wiring (models →
ipc.run_loop) lives only here so the dependency graph between features
stays grep-able to one place.

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

# WHY: drop scheduling priority before any heavy work so the loader and
# (later) the streaming generation don't starve the rest of the system.
# Loading a 7B model into MPS saturates memory bandwidth and CPU cores
# enough to make the whole desktop feel laggy at default priority. nice
# +10 leaves us well below interactive apps in the macOS / Linux
# scheduler. Wrapped in try/except because os.nice is a noop on Windows
# and we don't want startup to hard-fail there.
try:
    os.nice(10)
except (OSError, AttributeError):
    pass

# WHY: cap thread pools that PyTorch / OpenMP read at import time. Apple
# Silicon ships 8–10 cores; letting torch claim all of them during a
# load is what causes other apps to stutter. Set before any torch
# import — these env vars are read once at OMP/MKL init and ignored
# afterward. set_num_threads inside Python is a runtime escape hatch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

# WHY: optional GPU allocator cap. PyTorch's defaults overcommit
# unified memory enough to push the desktop into swap during a load,
# but capping low enough to keep the system fully responsive doesn't
# leave room for the 7B model on smaller machines. Whether the
# trade-off is worth it depends on the machine + model, so this is
# opt-in rather than a default. Set CYPRESS_MPS_HIGH_RATIO to enable
# (e.g. 1.0 for "no overcommit", lower for stricter caps). PyTorch
# enforces LOW <= HIGH and its default LOW is 1.4, so we pin LOW
# alongside HIGH whenever the user opts in.
_mps_high = os.environ.get("CYPRESS_MPS_HIGH_RATIO")
if _mps_high:
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", _mps_high)

import audio
import ipc

# WHY: importing `models` triggers each concrete model's @register
# decorator and populates models.REGISTRY. We do that here, in the
# composition root, then hand the populated registry to ipc — keeping
# ipc unaware of the models feature.
import models


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
    # command is received. We pass the model registry as an explicit
    # dependency so ipc never imports from the models feature.
    try:
        await ipc.run_loop(_write, models.REGISTRY)
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
