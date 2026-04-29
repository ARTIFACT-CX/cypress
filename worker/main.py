"""
AREA: worker · ENTRY · COMPOSITION-ROOT

Cypress Python inference worker. Launched as a subprocess by the Go
orchestration server (local) or run standalone on a remote box.

Speaks one channel: a gRPC bidi RPC (`Worker.Session`) over either a
unix domain socket (local subprocess, file perms = auth) or TCP
(remote worker, TLS + auth wired in a follow-up). The same dispatcher
serves both — only the listener changes.

This file is the composition root: it imports each feature, wires them
together, starts the gRPC server, and waits for a shutdown command.
Cross-feature wiring (models → ipc) lives only here so the dependency
graph between features stays grep-able to one place.
"""

import argparse
import asyncio
import os
import pathlib
import sys
import traceback

# REASON: the proto-generated `workerpb` package lives at the repo root
# (proto/dist/python/workerpb), shared by both Go and Python sides. Each
# per-family venv runs `python -m worker.main` from this file's parent,
# so we add the generated dir to sys.path here in the composition root.
# Doing it via sys.path (rather than installing as an editable package
# in every per-family pyproject) keeps the worker venvs lean.
_PROTO_PY = pathlib.Path(__file__).resolve().parent.parent / "proto" / "dist" / "python"
if _PROTO_PY.is_dir():
    sys.path.insert(0, str(_PROTO_PY))

# REASON: drop scheduling priority before any heavy work so the loader and
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

# REASON: cap thread pools that PyTorch / OpenMP read at import time. Apple
# Silicon ships 8–10 cores; letting torch claim all of them during a
# load is what causes other apps to stutter. Set before any torch
# import — these env vars are read once at OMP/MKL init and ignored
# afterward. set_num_threads inside Python is a runtime escape hatch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

# REASON: optional GPU allocator cap. PyTorch's defaults overcommit
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

import ipc

# REASON: importing `models` triggers each concrete model's @register
# decorator and populates models.REGISTRY. We do that here, in the
# composition root, then hand the populated registry to ipc — keeping
# ipc unaware of the models feature.
import models


def _default_listen() -> str:
    # SETUP: derive a per-pid unix socket path so multiple workers can
    # coexist during dev (e.g. one orphaned from a prior crash plus a
    # new one the user just launched). The Go host's spawnWorker mints
    # this same path before exec so it knows where to dial.
    return f"unix:/tmp/cypress-{os.getpid()}.sock"


async def _run(listen: str) -> None:
    # Cleanup the unix socket on exit so we don't leave stale files in
    # /tmp for the next run. TCP listeners self-clean.
    sock_path: str | None = None
    if listen.startswith("unix:"):
        sock_path = listen[len("unix:") :]
        # Stale socket from a prior crashed run blocks bind; clear it.
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

    try:
        await ipc.serve(listen, models.REGISTRY)
    finally:
        if sock_path is not None:
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(prog="cypress-worker")
    parser.add_argument(
        "--listen",
        default=None,
        help=(
            "gRPC listen target. Forms: 'unix:<path>' (local subprocess, "
            "default) or 'tcp://host:port' (remote worker)."
        ),
    )
    args = parser.parse_args()
    listen = args.listen or _default_listen()

    try:
        asyncio.run(_run(listen))
    except KeyboardInterrupt:
        # Ctrl-C from a manual run is not an error; exit cleanly.
        pass
    except Exception:
        # REASON: surface fatal startup errors on stderr so the host's
        # forwarded stderr shows them. The handshake path can no longer
        # carry a fatal message once we've failed to even bind the
        # gRPC listener — exit non-zero and let the host report the
        # spawn failure.
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
