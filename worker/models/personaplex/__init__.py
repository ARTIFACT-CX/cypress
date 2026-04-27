"""
AREA: worker · MODELS · PERSONAPLEX

NVIDIA PersonaPlex backend. One concrete loader for now (torch.py);
an MLX/INT4 path for Apple Silicon is the open work tracked in #3.

Importing this subpackage registers the class in the top-level
REGISTRY so the host can look it up by name. The catalog marks
PersonaPlex Available:false until the loader is verified end-to-end,
so this registration is currently for completeness rather than active
use — the host won't be asked to load `personaplex` from the UI yet.
"""

from . import torch  # noqa: F401
