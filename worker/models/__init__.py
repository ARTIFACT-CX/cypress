"""
AREA: worker · MODELS

Package of voice-model implementations. Each concrete model lives in its
own file (moshi.py, moshi_mlx.py, personaplex.py, …) and registers
itself via the `register` decorator in base.py. The command handler
looks models up by name in REGISTRY.

Backend autoselection: the `"moshi"` name is an alias that resolves to
the right backend for the current platform. MLX on Apple Silicon (q8 by
default for usable real-time perf), PyTorch elsewhere. Explicit
selectors `"moshi-mlx"` and `"moshi-torch"` are also exposed so a power
user can force a backend on either platform — useful for A/B testing.
A v0.2 settings UI (#21) will surface the override; for now,
CYPRESS_MOSHI_BACKEND=mlx|torch overrides the autoselect from the
environment.
"""

import os
import platform

from .base import Model, REGISTRY, register

# REASON: importing concrete model modules here triggers their @register
# decorators so REGISTRY is fully populated by the time ipc_commands
# looks anything up. Both backends are imported unconditionally — they
# only do lazy imports of their heavy deps inside load(), so importing
# the modules themselves is cheap and works on any platform. The
# autoselect below decides which one answers `"moshi"`.
from . import moshi  # noqa: F401
from . import moshi_mlx  # noqa: F401


def _default_moshi_backend() -> str:
    """Pick the registered class name `"moshi"` should alias to.

    Order: explicit env override → platform default → torch fallback.
    Apple Silicon picks MLX because torch-MPS at bf16 runs roughly an
    order of magnitude slower than MLX-q8 in practice (one frame per
    minute vs real-time on the same hardware)."""
    override = os.environ.get("CYPRESS_MOSHI_BACKEND", "").strip().lower()
    if override == "mlx":
        return "moshi-mlx"
    if override == "torch":
        return "moshi-torch"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "moshi-mlx"
    return "moshi-torch"


# SETUP: register the autoselect alias. We point REGISTRY["moshi"] at
# the same class as the chosen backend rather than registering twice —
# both names refer to one class, so a swap of CYPRESS_MOSHI_BACKEND on
# restart picks up the new default cleanly.
_default = _default_moshi_backend()
if _default in REGISTRY:
    REGISTRY["moshi"] = REGISTRY[_default]


__all__ = ["Model", "REGISTRY", "register"]
