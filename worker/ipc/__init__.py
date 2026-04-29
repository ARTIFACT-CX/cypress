"""
AREA: worker · IPC

Feature: gRPC bidi control + audio stream with the Go host. Owns the
dispatch table for inbound commands and the event-emit callback used by
handlers (and side channels like model phase events) to push messages
back. Concrete handlers live in commands.py; the gRPC servicer that
fronts them lives in server.py.

Public surface:
- serve(listen, registry): the gRPC server entry point
- emit_event(msg): out-of-band (no-id) push from anywhere in the worker

Anything else here is an implementation detail.
"""

from .commands import emit_event
from .server import serve

__all__ = ["serve", "emit_event"]
