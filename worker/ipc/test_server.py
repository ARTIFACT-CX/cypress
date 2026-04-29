"""
AREA: worker · IPC · TESTS

Tests for the gRPC servicer's auth + listener-gating policy. The Session
RPC itself is exercised end-to-end in worker/test_integration.py; here
we cover the gates that protect it: who can reach it (AuthInterceptor)
and what binds we'll accept (_parse_listen + serve preconditions).
"""

import pytest

from . import server


# --- _parse_listen -----------------------------------------------------------
#
# Mirrors the Go side's parseRemoteURL. Both ends must agree on what
# counts as loopback or the auth/TLS gate becomes asymmetric.


def test_parse_listen_unix():
    bind, is_tcp, is_loopback = server._parse_listen("unix:/tmp/x.sock")
    assert bind == "unix:/tmp/x.sock"
    assert is_tcp is False
    assert is_loopback is False  # not meaningful for unix; gate ignores it


@pytest.mark.parametrize("listen", [
    "tcp://localhost:7843",
    "tcp://127.0.0.1:7843",
    "tcp://[::1]:7843",
])
def test_parse_listen_tcp_loopback(listen):
    bind, is_tcp, is_loopback = server._parse_listen(listen)
    assert is_tcp is True
    assert is_loopback is True
    # bind must drop the scheme; gRPC wants bare host:port for TCP.
    assert "tcp://" not in bind


def test_parse_listen_tcp_remote():
    _, is_tcp, is_loopback = server._parse_listen("tcp://worker.example.com:7843")
    assert is_tcp is True
    assert is_loopback is False


@pytest.mark.parametrize("bad", [
    "worker.example.com:7843",   # missing scheme
    "tcp://worker.example.com",  # missing port
    "http://worker.example.com:7843",  # unsupported scheme
])
def test_parse_listen_rejects_bad(bad):
    with pytest.raises(ValueError):
        server._parse_listen(bad)


# --- serve preconditions -----------------------------------------------------
#
# REASON: the most important policy in this file. A misconfigured remote
# (TCP open without TLS, or without a token) must fail at startup, not
# silently accept connections. We exercise serve() far enough to hit
# the preconditions but never call .start() — those raises happen first.


async def test_serve_rejects_remote_tcp_without_tls():
    with pytest.raises(ValueError, match="requires --tls"):
        await server.serve("tcp://worker.example.com:7843", registry=None, token="t")


async def test_serve_rejects_tcp_without_token():
    with pytest.raises(ValueError, match="--token"):
        await server.serve("tcp://127.0.0.1:7843", registry=None, token=None)


# --- AuthInterceptor ---------------------------------------------------------


class _FakeCallDetails:
    """Minimal stand-in for grpc.HandlerCallDetails — only the field
    AuthInterceptor reads. Building real call details requires a live
    server; this is enough for unit-testing the policy."""

    def __init__(self, metadata):
        self.invocation_metadata = metadata
        self.method = "/cypress.worker.v1.Worker/Session"


def _continuation_marker():
    """Returns an async continuation plus a sentinel the test asserts
    was reached. The interceptor must `await continuation(details)`;
    the sentinel makes the accept-path observable in a unit test."""
    sentinel = object()

    async def _cont(_details):
        return sentinel

    return _cont, sentinel


async def test_auth_interceptor_accepts_correct_bearer():
    cont, sentinel = _continuation_marker()
    interceptor = server.AuthInterceptor("s3cret")
    details = _FakeCallDetails([("authorization", "Bearer s3cret")])
    assert await interceptor.intercept_service(cont, details) is sentinel


@pytest.mark.parametrize("metadata", [
    [],                                          # no auth header
    [("authorization", "Bearer wrong")],         # wrong token
    [("authorization", "s3cret")],               # missing Bearer prefix
    [("x-other", "Bearer s3cret")],              # right value, wrong key
])
async def test_auth_interceptor_rejects(metadata):
    cont, sentinel = _continuation_marker()
    interceptor = server.AuthInterceptor("s3cret")
    details = _FakeCallDetails(metadata)
    result = await interceptor.intercept_service(cont, details)
    # Rejection returns the unauth handler, not the continuation's
    # sentinel. The handler aborts when invoked by gRPC's machinery;
    # the unit-level guarantee is "we did not invoke continuation".
    assert result is not sentinel
    assert result is server._UNAUTH_HANDLER
