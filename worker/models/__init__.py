"""
AREA: worker · MODELS

Package of voice-model implementations. Each model family lives in its
own subpackage (moshi/, personaplex/, …) and registers its concrete
classes via the `register` decorator in base.py. The command handler
looks models up by name in REGISTRY.
"""

from .base import Model, REGISTRY, register

# REASON: importing each model subpackage triggers its registrations and
# any per-family backend autoselect (e.g. moshi/__init__.py picks MLX vs
# torch on the current host).
from . import moshi  # noqa: F401
from . import personaplex  # noqa: F401


__all__ = ["Model", "REGISTRY", "register"]
