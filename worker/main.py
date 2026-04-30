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

# REASON: silence the "FD from fork parent still in poll list" log line
# grpc-c-core emits when huggingface_hub forks download workers after
# our gRPC server has initialized. Benign — the child fixes its FDs —
# but surfaces through the Go stderr forwarder as if it were an error.
# Must be set before any gRPC import (read once at c-core init).
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

# REASON: kill the two HF features that fork their own worker pools.
# Telemetry is a one-call analytics ping; hf_transfer is the Rust
# parallel-download path. Single-process downloads are plenty fast for
# moshi's ~5 GB and remove the fork-after-init trigger entirely.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import ipc

# REASON: family selection happens here, in the composition root.
# Each family lives in its own venv with conflicting deps, so we only
# import the one this process is running — see models.load_family.
import models

# REASON: platform_info is a pure-stdlib helper (no model deps); safe
# to import unconditionally before load_family runs.
import platform_info as worker_platform_info


def _default_listen() -> str:
    # SETUP: derive a per-pid unix socket path so multiple workers can
    # coexist during dev (e.g. one orphaned from a prior crash plus a
    # new one the user just launched). The Go host's spawnWorker mints
    # this same path before exec so it knows where to dial.
    return f"unix:/tmp/cypress-{os.getpid()}.sock"


async def _run(
    listen: str,
    token: str | None,
    tls: tuple[bytes, bytes] | None,
    platform_info: dict | None = None,
) -> None:
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
        await ipc.serve(
            listen,
            models.REGISTRY,
            token=token,
            tls=tls,
            platform_info=platform_info,
        )
    finally:
        if sock_path is not None:
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass


def _read_tls(args: argparse.Namespace) -> tuple[bytes, bytes] | None:
    """Materialize --tls cert key into (cert_pem, key_pem) bytes. Wraps
    the underlying FileNotFoundError so an operator typo points at which
    flag was wrong instead of just a bare path."""
    if not args.tls:
        return None
    cert_path, key_path = args.tls
    try:
        cert_pem = pathlib.Path(cert_path).read_bytes()
        key_pem = pathlib.Path(key_path).read_bytes()
    except OSError as e:
        raise SystemExit(f"--tls: cannot read {e.filename!r}: {e.strerror}") from e
    return cert_pem, key_pem


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
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Bearer token clients must present. Falls back to CYPRESS_TOKEN "
            "env var. Required for any tcp:// listener."
        ),
    )
    parser.add_argument(
        "--family",
        default=None,
        help=(
            "Model family to serve (e.g. 'moshi', 'personaplex'). Falls "
            "back to CYPRESS_FAMILY env var. Required: workers are "
            "family-scoped because per-family venvs hold conflicting "
            "deps and only one family's stack is importable at a time."
        ),
    )
    parser.add_argument(
        "--tls",
        nargs=2,
        metavar=("CERT", "KEY"),
        default=None,
        help=(
            "PEM-encoded server certificate + private key paths. Required "
            "for tcp:// to a non-loopback host so the bearer token never "
            "rides cleartext."
        ),
    )
    args = parser.parse_args()
    listen = args.listen or _default_listen()
    token = args.token or os.environ.get("CYPRESS_TOKEN") or None
    tls = _read_tls(args)
    family = args.family or os.environ.get("CYPRESS_FAMILY") or None
    if not family:
        raise SystemExit(
            "--family is required (or set CYPRESS_FAMILY). "
            "Pick the family this venv was built for, e.g. 'moshi'."
        )
    # SETUP: trigger the family's @register decorators before we serve.
    # Failing here (rather than at first load command) makes a misbuilt
    # image fail loud at startup.
    try:
        models.load_family(family)
    except ImportError as e:
        raise SystemExit(
            f"--family {family!r}: cannot import models.{family} — is the "
            f"{family} venv built? ({e})"
        ) from e

    # SETUP: snapshot platform + cache state once, after load_family
    # so the backend probe sees whatever the family installed in this
    # venv. Cheap (a few sys-syscall stats); doesn't grow with cache size.
    info = worker_platform_info.gather(family)

    try:
        asyncio.run(_run(listen, token, tls, info))
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
