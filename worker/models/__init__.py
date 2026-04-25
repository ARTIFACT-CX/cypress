"""
AREA: worker · MODELS

Package of voice-model implementations. Each concrete model lives in its
own file (moshi.py, personaplex.py) and registers itself via the `register`
decorator in base.py. The command handler looks models up by name in
REGISTRY.
"""

from .base import Model, REGISTRY, register

# WHY: importing concrete model modules here triggers their @register
# decorators so REGISTRY is fully populated by the time ipc_commands
# looks anything up. Keep this list in sync with files in this package.
from . import moshi  # noqa: F401

__all__ = ["Model", "REGISTRY", "register"]
