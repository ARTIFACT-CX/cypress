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
from .moshi_stream import MoshiStream, _StreamComponents


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
    # `moshi-torch` is the explicit selector for the PyTorch backend. On
    # platforms where torch is the autoselect default (anything but
    # Apple Silicon today), __init__.py also aliases `"moshi"` to this
    # class. Keeping the explicit name lets users force-select the torch
    # path even on Apple Silicon, which is useful for A/B comparison.
    name = "moshi-torch"

    def __init__(self, emit: EmitFn):
        # emit is a worker→host one-way channel (no reply expected). Used
        # for phase events during load so the UI can show "downloading
        # mimi…", "loading to mps…" instead of a silent 60s spinner.
        self._emit = emit
        self._device: Optional[str] = None
        self._checkpoint: Any = None  # moshi loaders.CheckpointInfo, kept for run_wav
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

            # REASON: belt-and-suspenders thread cap. main.py sets the env
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
            # The descriptor is also kept on self because run_wav needs
            # its `lm_gen_config` and `model_type` to build an
            # InferenceState that matches how the model was trained.
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

            return checkpoint, mimi, lm, text_tokenizer

        (
            self._checkpoint,
            self._mimi,
            self._lm,
            self._text_tokenizer,
        ) = await asyncio.to_thread(_do_load)

        self._emit({"event": "model_phase", "phase": "ready", "device": self._device})

    def run_wav(self, input_path: str, output_path: str) -> dict:
        """Offline self-test: run a wav file through the loaded model and
        write the response to disk. Synchronous and blocking — callers
        should dispatch via asyncio.to_thread so the worker's control loop
        stays responsive (a long input clip can take many seconds even
        on GPU, and we still want the host to be able to send `shutdown`).

        Returns a metadata dict (frames, duration, elapsed) that the IPC
        handler merges into its reply.

        Wraps moshi.run_inference.InferenceState — the upstream library's
        own self-test harness — rather than reimplementing the streaming
        loop here. Reusing it means we track upstream behavior changes
        (model-type quirks, EOS handling) for free.
        """
        if self._mimi is None or self._lm is None or self._checkpoint is None:
            # Defensive: handler should already gate on `instance is not
            # None`, but a half-loaded model would have these unset.
            raise RuntimeError("model not loaded")

        # Imports here, not at module top, mirror the rest of this file:
        # keep the worker startable even when torch / sphn / moshi-runtime
        # have issues, surfacing the failure only when generation is
        # actually attempted.
        import time

        import sphn
        import torch
        from moshi.run_inference import InferenceState, seed_all

        # SETUP: deterministic seeding. Matches upstream's run_inference
        # so two consecutive run_wav calls with the same input produce
        # the same output — useful for diffing against a golden file.
        seed_all(4242)

        # STEP 1: read input wav, resample to mimi's native rate, ship to
        # the model's device. sphn returns (channels, samples) float32 in
        # [-1, 1]; we keep just channel 0 because moshi is mono.
        in_pcms, _ = sphn.read(input_path, sample_rate=self._mimi.sample_rate)
        in_pcms_t = torch.from_numpy(in_pcms).to(device=self._device)
        batch_size = 1
        in_pcms_t = in_pcms_t[None, 0:1].expand(batch_size, -1, -1)

        # STEP 2: build the InferenceState and run. cfg_coef=1.0 disables
        # classifier-free guidance — moshiko (the default checkpoint)
        # isn't a CFG-conditioned model, so any other value would just
        # waste compute. lm_gen_config carries sampling temperatures and
        # any model-type-specific knobs the checkpoint shipped with.
        with torch.no_grad():
            state = InferenceState(
                self._checkpoint,
                self._mimi,
                self._text_tokenizer,
                self._lm,
                batch_size,
                cfg_coef=1.0,
                device=self._device,
                **self._checkpoint.lm_gen_config,
            )
            t0 = time.time()
            out_items = state.run(in_pcms_t)
            elapsed = time.time() - t0

        # STEP 3: handle empty output. Some model_types return [] when
        # `dep_q == 0` (no audio decoder head). Surface that explicitly
        # rather than letting the caller find an empty file.
        if not out_items:
            return {
                "frames": 0,
                "duration_s": 0.0,
                "elapsed_s": float(elapsed),
                "device": self._device,
                "note": "model produced no audio output",
            }

        # STEP 4: write the (single, since batch_size=1) output wav. sphn
        # writes float32 PCM; the .numpy() copy is unavoidable because
        # sphn doesn't accept torch tensors directly.
        _, out_pcm = out_items[0]
        sample_rate = self._mimi.sample_rate
        sphn.write_wav(output_path, out_pcm[0].numpy(), sample_rate=sample_rate)

        return {
            "frames": int(out_pcm.shape[1]),
            "duration_s": float(out_pcm.shape[1]) / float(sample_rate),
            "elapsed_s": float(elapsed),
            "device": self._device,
            "sample_rate": int(sample_rate),
        }

    def stream(self) -> MoshiStream:
        """Open a realtime streaming session against this loaded model.
        Caller must `await session.start()` before feeding audio. Raises
        if no model is loaded.

        Only one active session at a time — the underlying mimi and
        lm_gen are stateful, so a second concurrent session would
        corrupt both. The audio pipeline enforces this at its level
        (one connection at a time in v0.1).
        """
        if self._mimi is None or self._lm is None or self._checkpoint is None:
            raise RuntimeError("model not loaded")

        # STEP 1: build a fresh LMGen for this session. Reusing the
        # upstream condition-tensor helper keeps us aligned with how
        # different model_types (moshi, hibiki, stt) want to be primed —
        # rebuilding here means we don't have to special-case any of
        # them in our own code.
        from moshi.models import LMGen
        from moshi.run_inference import get_condition_tensors

        condition_tensors = get_condition_tensors(
            self._checkpoint.model_type, self._lm, batch_size=1, cfg_coef=1.0
        )
        lm_gen = LMGen(
            self._lm,
            cfg_coef=1.0,
            condition_tensors=condition_tensors,
            **self._checkpoint.lm_gen_config,
        )

        # STEP 2: bundle everything the streaming session needs into the
        # internal components struct. Computed properties (frame_size,
        # sample_rate) are pulled off mimi here so the session itself
        # doesn't have to know about mimi's API.
        components = _StreamComponents(
            mimi=self._mimi,
            lm_gen=lm_gen,
            text_tokenizer=self._text_tokenizer,
            device=self._device or "cpu",
            frame_size=int(self._mimi.sample_rate / self._mimi.frame_rate),
            sample_rate=int(self._mimi.sample_rate),
        )
        return MoshiStream(components)

    async def unload(self) -> None:
        # Drop references first so Python's GC can reclaim the tensors.
        self._checkpoint = None
        self._mimi = None
        self._lm = None
        self._text_tokenizer = None

        # REASON: MPS and CUDA both hold onto a caching allocator that
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
