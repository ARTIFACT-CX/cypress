"""
AREA: worker · MODELS · MOSHI · MLX

Apple-Silicon-native loader for Kyutai's Moshi via the official
`moshi-mlx` package. Distinct from moshi.py (PyTorch) because MLX has
its own model classes, weight format, and a Rust-implemented mimi
tokenizer (rustymimi); the two backends share little code below the
streaming-session interface.

Why this exists alongside moshi.py: bf16 Moshi on torch-MPS runs at
roughly one frame per minute on Apple Silicon, which is unusable for
streaming. MLX targets Metal natively and the q8 variant runs in
real-time on the same hardware. On other platforms (Linux/CUDA) MLX
isn't available, so the torch backend is still the right choice there.

Selection: __init__.py registers this as `"moshi"` on Apple Silicon and
the torch backend as `"moshi"` elsewhere. Explicit `"moshi-mlx"` /
`"moshi-torch"` names are also exposed for testing both backends on
the same machine. v0.2 will surface this in a settings UI (#21).

SWAP: checkpoint repo via CYPRESS_MOSHI_REPO. Default is
`kyutai/moshika-mlx-q8` — female voice, int8 quantized, ~4GB, the
speed/quality knee on most Apple Silicon. Other valid repos:
`kyutai/moshika-mlx-bf16` (full quality, ~14GB, slower),
`kyutai/moshika-mlx-q4` (smaller and faster, more quality cost), or
the moshiko-* equivalents for the male voice.
"""

import asyncio
import os
from typing import Any, Callable, Optional

from ..base import Model, register
from .mlx_stream import MoshiMlxStream, _StreamComponents


DEFAULT_REPO = "kyutai/moshika-mlx-q8"

# REASON: max_steps allocates the LM's KV cache up front. Streaming
# sessions don't have a known length; ~10K steps = ~13 minutes of
# real-time audio at 80ms per step, which covers any v0.1 conversation.
# A future change can grow this on demand or split sessions, but for now
# the over-allocation is cheap (a few MB) compared to the model itself.
DEFAULT_MAX_STEPS = 10_000


EmitFn = Callable[[dict], None]


@register
class MoshiMlx(Model):
    # `moshi-mlx` lets a user select this backend explicitly even on
    # platforms where it's not the default; the autoselect logic in
    # __init__.py also maps `"moshi"` → this class on Apple Silicon.
    name = "moshi-mlx"

    def __init__(self, emit: EmitFn):
        self._emit = emit
        # MLX targets Apple Metal; we report "mlx" rather than "metal"
        # because that's the package name the user sees and it matches
        # what we'd want to expose in a settings UI.
        self._device: Optional[str] = "mlx"
        self._lm_config: Any = None
        self._model: Any = None
        self._mimi: Any = None  # rustymimi.Tokenizer
        self._text_tokenizer: Any = None
        self._generated_codebooks: int = 0
        self._other_codebooks: int = 0
        self._condition_tensor: Any = None  # None for unconditioned checkpoints

    async def load(self) -> None:
        # STEP 1: surface the resolution phase before any blocking work
        # so the UI can show progress immediately, even on a cold cache.
        repo = os.environ.get("CYPRESS_MOSHI_REPO", DEFAULT_REPO)
        self._emit(
            {
                "event": "model_phase",
                "phase": "resolving",
                "device": self._device,
                "repo": repo,
            }
        )

        # STEP 2: blocking work in a thread so the asyncio loop stays
        # responsive (shutdown commands, status pings) during the multi-
        # gigabyte download + load.
        def _do_load():
            from huggingface_hub import hf_hub_download
            import mlx.core as mx
            import mlx.nn as nn
            import rustymimi
            import sentencepiece
            from moshi_mlx import models

            # File names are fixed across the published moshiko-mlx-*
            # repos (q4 / q8 / bf16). The quantization variant is
            # encoded in the moshi weights filename suffix; mimi and
            # tokenizer files are identical across variants. No JSON
            # config — the config is built from code (config_v0_1)
            # because these checkpoints predate kyutai's per-repo
            # config.json convention.
            mimi_name = "tokenizer-e351c8d8-checkpoint125.safetensors"
            tokenizer_name = "tokenizer_spm_32k_3.model"
            if "q4" in repo:
                moshi_name = "model.q4.safetensors"
            elif "q8" in repo:
                moshi_name = "model.q8.safetensors"
            else:
                moshi_name = "model.safetensors"

            # STEP 2a: download (cache-friendly). HF hub silently uses
            # ~/.cache/huggingface/hub on subsequent runs, so most loads
            # reach this code with nothing to download.
            self._emit({"event": "model_phase", "phase": "loading_lm"})
            moshi_weights = hf_hub_download(repo, moshi_name)
            self._emit({"event": "model_phase", "phase": "loading_mimi"})
            mimi_weights = hf_hub_download(repo, mimi_name)
            self._emit({"event": "model_phase", "phase": "loading_tokenizer"})
            tokenizer_path = hf_hub_download(repo, tokenizer_name)

            # STEP 2b: build the LM. config_v0_1 matches what the
            # moshiko-mlx-* checkpoints were trained against (8
            # generated codebooks, 8 other codebooks, etc.). Set bf16
            # base dtype, then quantize if the filename signals it.
            # Group sizes follow upstream local.py defaults — these are
            # what the published weights were quantized at, so any
            # other value would mismatch and produce garbage.
            lm_config = models.config_v0_1()
            model = models.Lm(lm_config)
            model.set_dtype(mx.bfloat16)
            if moshi_weights.endswith(".q4.safetensors"):
                nn.quantize(model, bits=4, group_size=32)
            elif moshi_weights.endswith(".q8.safetensors"):
                nn.quantize(model, bits=8, group_size=64)
            model.load_weights(moshi_weights, strict=True)

            # STEP 2c: condition tensor. moshiko has no conditioner —
            # the attribute is checked defensively in case a future
            # repo we point CYPRESS_MOSHI_REPO at does.
            ct = None
            if getattr(model, "condition_provider", None) is not None:
                ct = model.condition_provider.condition_tensor(
                    "description", "very_good"
                )

            # STEP 2d: warmup. MLX compiles kernels on first use;
            # doing it here means the streaming session's first frame
            # doesn't eat a multi-second JIT hit. Cheap (sub-second
            # on q8). Pass `ct` if the model takes one — moshiko's
            # warmup ignores the arg.
            try:
                model.warmup(ct)
            except TypeError:
                # Some upstream versions don't accept the condition
                # arg; fall back to plain warmup.
                model.warmup()

            # STEP 2e: rustymimi tokenizer (Rust streaming codec).
            # Sized for max(generated, other) so it covers both
            # encode and decode paths. encode_step / decode_step are
            # stateful and streaming.
            generated_codebooks = lm_config.generated_codebooks
            other_codebooks = lm_config.other_codebooks
            mimi_codebooks = max(generated_codebooks, other_codebooks)
            audio_tokenizer = rustymimi.Tokenizer(
                mimi_weights, num_codebooks=mimi_codebooks
            )

            # STEP 2f: text tokenizer (SentencePiece). Used to render
            # the inner-monologue token stream into readable text.
            text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

            return (
                lm_config,
                model,
                audio_tokenizer,
                text_tokenizer,
                generated_codebooks,
                other_codebooks,
                ct,
            )

        (
            self._lm_config,
            self._model,
            self._mimi,
            self._text_tokenizer,
            self._generated_codebooks,
            self._other_codebooks,
            self._condition_tensor,
        ) = await asyncio.to_thread(_do_load)

        self._emit({"event": "model_phase", "phase": "ready", "device": self._device})

    def stream(self) -> MoshiMlxStream:
        """Open a realtime streaming session against this loaded model.
        Caller awaits session.start() before feeding audio.

        One session at a time — `LmGen` and the Tokenizer are stateful.
        We rebuild a fresh `LmGen` per session so its KV cache starts
        empty; the Tokenizer survives across sessions and gets its
        `reset()` called from the session's aclose."""
        if self._model is None:
            raise RuntimeError("model not loaded")

        from moshi_mlx import models, utils

        # STEP 1: fresh LmGen per session. max_steps allocates the KV
        # cache; we use a generous default rather than tracking session
        # length. Samplers default to upstream's run_inference.py choices.
        gen = models.LmGen(
            model=self._model,
            max_steps=DEFAULT_MAX_STEPS,
            text_sampler=utils.Sampler(),
            audio_sampler=utils.Sampler(),
            cfg_coef=1.0,
            check=False,
        )

        # STEP 2: bundle into the session's components struct. Frame size
        # and sample rate are hard-coded to the values mimi expects (1920
        # samples = 80ms at 24kHz); they aren't surfaced through the
        # rustymimi API so we have to know them here.
        components = _StreamComponents(
            mimi=self._mimi,
            lm_gen=gen,
            text_tokenizer=self._text_tokenizer,
            condition_tensor=self._condition_tensor,
            generated_codebooks=self._generated_codebooks,
            other_codebooks=self._other_codebooks,
            frame_size=1920,
            sample_rate=24000,
        )
        return MoshiMlxStream(components)

    async def unload(self) -> None:
        # Drop refs first so MLX's allocator + Python's GC can reclaim.
        self._lm_config = None
        self._model = None
        self._mimi = None
        self._text_tokenizer = None
        self._condition_tensor = None

        # REASON: MLX's Metal allocator caches buffers similar to torch's
        # CUDA caching allocator. Without this, an unload + reload of a
        # different model would fail with out-of-memory on machines that
        # technically have enough room for one model at a time. Wrapped
        # in try because clear_cache moved between MLX versions.
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            pass

    def device(self) -> Optional[str]:
        return self._device
