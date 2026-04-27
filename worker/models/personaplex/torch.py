"""
AREA: worker · MODELS · PERSONAPLEX

Loader for NVIDIA PersonaPlex 7B via the `moshi` package — but the
NVIDIA *fork*, not kyutai's. Both packages share the import name; the
multi-venv infra (#22) is what lets us keep them apart by giving each
family its own `.venv`.

This is currently a STUB — `load()` raises so any accidental request
fails loud. The catalog keeps PersonaPlex Available:false until the
inference path on Apple Silicon is sorted (#3); the most likely route
is an INT4 quant (MLX or torch.ao) since the bf16 checkpoint is
~16.7GB, which won't fit alongside Mimi + KV cache on a 16GB M1 Pro.

When the loader is wired:
  - Mimi : audio codec / tokenizer (same filename as kyutai moshi)
  - PersonaPlex LM : NVIDIA's persona-conditioned variant
  - Text tokenizer : SentencePiece, shared with moshi
"""

from ..base import Model, register


# SETUP: HF repo. Single bf16 build for now — the q4 community quant
# (Codes4Fun/personaplex-7b-v1-q4_k-GGUF) strips the audio components
# and isn't loadable through the moshi runtime.
DEFAULT_REPO = "nvidia/personaplex-7b-v1"


@register
class PersonaPlex(Model):
    name = "personaplex"

    async def load(self) -> None:
        # REASON: fail loud rather than silently no-op so any caller that
        # bypasses the catalog Available:false guard surfaces immediately.
        # The catalog and HTTP layer normally prevent this from being hit.
        raise NotImplementedError(
            "PersonaPlex loader is not yet implemented (see #3). "
            "Inference path on Apple Silicon (likely INT4/MLX) is open work."
        )

    async def unload(self) -> None:
        # No-op: load() never succeeds, so there's nothing to release.
        return None
