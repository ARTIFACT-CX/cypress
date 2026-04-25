"""
AREA: worker · MODELS · MOSHI

Loader for Kyutai's Moshi duplex voice model via the official `moshi`
PyTorch package. Three things land on the device:

  - Mimi   : audio codec / tokenizer
  - Moshi LM : the language model driving both text and codec tokens
  - Text tokenizer : SentencePiece, used for the "inner monologue" stream

Weights come from HuggingFace on first load and are cached under
~/.cache/huggingface/hub by the hub client, so subsequent launches skip
the download.

SWAP: checkpoint source. We go through `CheckpointInfo.from_hf_repo` which
downloads on demand. A future packaged build can hand in a local path via
CYPRESS_MOSHI_REPO (any hub repo) or by wiring `from_local_path` here.

TODO: moshi-mlx backend for Apple Silicon (see project board). Same Model
interface, different loader module, chosen per-platform.
"""

import asyncio
import os
from typing import Any, Callable, Optional

from .base import Model, register


# SETUP: which HF repo we pull from. Defaults to the bf16 moshiko variant
# (male voice, ~14GB). CYPRESS_MOSHI_REPO lets the user point at an
# alternative — e.g. a q8 variant for tighter memory, or a local mirror.
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"


def _detect_device() -> str:
    # Import here, not at module top, so the worker starts even if torch
    # is broken — we only need torch once a model is actually requested.
    import torch

    # CYPRESS_DEVICE is the explicit escape hatch. Useful when MPS is
    # technically available but misbehaving for a given op (Apple ships
    # periodic MPS regressions), or when benchmarking on CPU.
    override = os.environ.get("CYPRESS_DEVICE")
    if override:
        return override

    # Auto-detect priority: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU.
    # On Linux workstations CUDA is always preferred when present; on
    # Macs MPS is the only GPU path.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


EmitFn = Callable[[dict], None]


@register
class Moshi(Model):
    # Must match the `name` string the UI sends in load_model commands.
    name = "moshi"

    def __init__(self, emit: EmitFn):
        # emit is a worker→host one-way channel (no reply expected). Used
        # for phase events during load so the UI can show "downloading
        # mimi…", "loading to mps…" instead of a silent 60s spinner.
        self._emit = emit
        self._device: Optional[str] = None
        self._mimi: Any = None
        self._lm: Any = None
        self._text_tokenizer: Any = None

    async def load(self) -> None:
        # STEP 1: decide where the model will live. Recorded now so the
        # host can surface it in the status panel even before weights
        # land (download phase happens on CPU regardless of target).
        self._device = _detect_device()
        repo = os.environ.get("CYPRESS_MOSHI_REPO", DEFAULT_REPO)
        self._emit(
            {
                "event": "model_phase",
                "phase": "resolving",
                "device": self._device,
                "repo": repo,
            }
        )

        # STEP 2: run the blocking HF/torch work in a thread so the
        # asyncio event loop stays responsive. Without this the worker
        # couldn't react to a shutdown command mid-download — it would
        # sit blocked on the network read.
        def _do_load():
            import torch
            from moshi.models import loaders

            # WHY: belt-and-suspenders thread cap. main.py sets the env
            # vars before torch import (which is the *primary* lever),
            # but set_num_threads catches anything that ignored them.
            # Keeping the worker on a fraction of cores leaves the OS
            # responsive while a 7B model is shuffling into MPS.
            try:
                torch.set_num_threads(4)
            except RuntimeError:
                # Already initialized; non-fatal.
                pass

            # STEP 2a: resolve the checkpoint descriptor. This hits HF to
            # look up the current revision but does not yet download the
            # big weight files — those happen inside get_mimi/get_moshi.
            checkpoint = loaders.CheckpointInfo.from_hf_repo(repo)

            # STEP 2b: Mimi (audio codec). First thing we need; it's also
            # the smallest (~100MB) so this phase completes quickly and
            # gives the user visible progress that *something* is moving.
            self._emit({"event": "model_phase", "phase": "downloading_mimi"})
            mimi = checkpoint.get_mimi(device=self._device)

            # STEP 2c: Moshi LM (~7B). This is the slow one — several GB
            # to pull on first run, then several seconds to move to MPS/
            # CUDA. Subsequent loads hit the HF cache and skip straight
            # to the device transfer.
            self._emit({"event": "model_phase", "phase": "downloading_lm"})
            lm = checkpoint.get_moshi(device=self._device)

            # STEP 2d: text tokenizer (SentencePiece, ~1MB). Needed to
            # decode the LM's "inner monologue" token stream that we'll
            # later use to drive tool calls.
            self._emit({"event": "model_phase", "phase": "loading_tokenizer"})
            text_tokenizer = checkpoint.get_text_tokenizer()

            return mimi, lm, text_tokenizer

        self._mimi, self._lm, self._text_tokenizer = await asyncio.to_thread(_do_load)

        self._emit({"event": "model_phase", "phase": "ready", "device": self._device})

    async def unload(self) -> None:
        # Drop references first so Python's GC can reclaim the tensors.
        self._mimi = None
        self._lm = None
        self._text_tokenizer = None

        # WHY: MPS and CUDA both hold onto a caching allocator that
        # doesn't release until explicitly asked. Without these calls, a
        # model unload wouldn't actually free VRAM until the worker
        # process exits — which would make "swap to a different model"
        # fail with out-of-memory even on machines that technically have
        # enough space for one model at a time.
        import torch

        if self._device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()

    def device(self) -> Optional[str]:
        return self._device
