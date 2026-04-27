"""
AREA: worker · TESTS · INTEGRATION

End-to-end test of the worker's stdin/stdout protocol against a real Python
subprocess. Marked `integration` so default `pytest` runs stay fast and
dependency-light. Run explicitly with:

    uv run --group dev pytest -m integration

Mirrors the Go-side integration test in server/inference/integration_test.go
but stays inside Python so we don't need the Go toolchain to validate the
worker side of the protocol independently.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest


WORKER_DIR = Path(__file__).resolve().parent


async def _read_json_line(stream: asyncio.StreamReader) -> dict:
    line = await stream.readline()
    if not line:
        raise EOFError("worker stdout closed before reply")
    return json.loads(line.decode())


@pytest.mark.integration
async def test_handshake_and_status_round_trip():
    # STEP 1: launch the worker the same way the Go server does — point
    # Python at the per-family venv (moshi here) with cwd=worker/ so the
    # shared scaffold (audio/, ipc/, models/) imports cleanly.
    family_python = WORKER_DIR / "models" / "moshi" / ".venv" / "bin" / "python"
    if not family_python.exists():
        pytest.skip(f"moshi family venv missing at {family_python} — run `uv sync` there")
    proc = await asyncio.create_subprocess_exec(
        str(family_python), "main.py",
        cwd=str(WORKER_DIR),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,  # forward tracebacks so failures are diagnosable
        env=os.environ.copy(),
    )
    try:
        # STEP 2: read the handshake. Generous timeout because uv cold
        # start + Python imports run a few seconds on first invocation.
        hs = await asyncio.wait_for(_read_json_line(proc.stdout), timeout=30)
        assert hs.get("ready") is True, f"unexpected handshake: {hs}"
        assert isinstance(hs.get("audio_socket"), str)

        # STEP 3: round-trip a status command. Confirms the dispatcher,
        # id-tagging, and stdout flushing all work end-to-end.
        proc.stdin.write(json.dumps({"id": 1, "cmd": "status"}).encode() + b"\n")
        await proc.stdin.drain()

        reply = await asyncio.wait_for(_read_json_line(proc.stdout), timeout=5)
        assert reply["id"] == 1
        assert reply["ok"] is True
        assert reply["model"] is None  # no model loaded

        # STEP 4: graceful shutdown.
        proc.stdin.write(json.dumps({"id": 2, "cmd": "shutdown"}).encode() + b"\n")
        await proc.stdin.drain()
        await asyncio.wait_for(_read_json_line(proc.stdout), timeout=5)
    finally:
        if proc.returncode is None:
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
