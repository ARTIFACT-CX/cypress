"""
AREA: worker · AUDIO · TESTS

Unit tests for the unix-domain socket server. These exercise the real
socket plumbing (start, accept, drain, close) but are still fast — UDS
on /tmp is sub-millisecond and there's no model in the loop.
"""

import asyncio
import os
import tempfile

import pytest

from . import socket as audio_socket


def _socket_path() -> str:
    # A fresh path per test so parallel runs (pytest-xdist, future) don't
    # collide. tempfile.mkstemp returns a path we then delete so
    # start_server can create the socket fresh.
    fd, path = tempfile.mkstemp(prefix="cypress-test-", suffix=".sock")
    os.close(fd)
    os.unlink(path)
    return path


async def test_start_server_creates_socket_file():
    path = _socket_path()
    server = await audio_socket.start_server(path)
    try:
        assert os.path.exists(path)
    finally:
        server.close()
        await server.wait_closed()
        if os.path.exists(path):
            os.unlink(path)


async def test_start_server_clears_stale_socket():
    # SAFETY: a previous worker crash can leave the UDS file behind, and
    # asyncio.start_unix_server fails with EADDRINUSE if it exists. The
    # cleanup in start_server is the fix; this test guards it.
    path = _socket_path()
    # Pre-create a stale file at the target path.
    with open(path, "w") as f:
        f.write("stale")

    server = await audio_socket.start_server(path)
    try:
        # Should have removed the stale file and bound a real socket.
        assert os.path.exists(path)
    finally:
        server.close()
        await server.wait_closed()
        if os.path.exists(path):
            os.unlink(path)


async def test_drains_client_data_until_disconnect():
    # End-to-end: connect, send some bytes, close. The handler should
    # consume them and exit cleanly without raising.
    path = _socket_path()
    server = await audio_socket.start_server(path)
    try:
        reader, writer = await asyncio.open_unix_connection(path)
        writer.write(b"hello world" * 100)
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        # Give the server's handler a moment to notice the close. Without
        # a brief await the test can race ahead and shut down the server
        # before the handler logs disconnect (cosmetic, but pytest-asyncio
        # complains about pending tasks at teardown).
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()
        if os.path.exists(path):
            os.unlink(path)
