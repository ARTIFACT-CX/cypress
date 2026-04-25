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
`kyutai/moshiko-mlx-q8` — int8 quantized, ~4GB, the speed/quality knee
on most Apple Silicon. Other valid repos: `kyutai/moshiko-mlx-bf16`
(full quality, ~14GB, slower) and `kyutai/moshiko-mlx-q4` (smaller and
faster, more quality cost).
"""

import asyncio
import json
import os
from typing import Any, Callable, Optional

from .base import Model, register
from .moshi_mlx_stream import MoshiMlxStream, _StreamComponents


DEFAULT_REPO = "kyutai/moshiko-mlx-q8"

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

            # STEP 2a: pull config first so we know which weight files
            # this checkpoint variant uses (mimi name, tokenizer name,
            # and the moshi weights filename, which encodes the
            # quantization scheme: model.q8.safetensors etc.).
            config_path = hf_hub_download(repo, "config.json")
            with open(config_path) as f:
                lm_config_dict = json.load(f)
            mimi_name = lm_config_dict["mimi_name"]
            tokenizer_name = lm_config_dict["tokenizer_name"]
            moshi_name = lm_config_dict.get("moshi_name", "model.safetensors")

            # STEP 2b: download (cache-friendly). HF hub silently uses
            # ~/.cache/huggingface/hub on subsequent runs, so most loads
            # reach this code with nothing to download.
            self._emit({"event": "model_phase", "phase": "loading_lm"})
            moshi_weights = hf_hub_download(repo, moshi_name)
            self._emit({"event": "model_phase", "phase": "loading_mimi"})
            mimi_weights = hf_hub_download(repo, mimi_name)
            self._emit({"event": "model_phase", "phase": "loading_tokenizer"})
            tokenizer_path = hf_hub_download(repo, tokenizer_name)

            # STEP 2c: build the LM. Set bf16 base dtype, then quantize
            # if the weight filename signals a quantized variant. Group
            # sizes follow the upstream `local.py` defaults — these are
            # what the published weights were quantized at, so any other
            # value here would mismatch and produce garbage.
            lm_config = models.LmConfig.from_config_dict(lm_config_dict)
            model = models.Lm(lm_config)
            model.set_dtype(mx.bfloat16)
            if moshi_weights.endswith(".q4.safetensors"):
                nn.quantize(model, bits=4, group_size=32)
            elif moshi_weights.endswith(".q8.safetensors"):
                nn.quantize(model, bits=8, group_size=64)
            model.load_weights(moshi_weights, strict=True)

            # STEP 2d: condition tensor. Some Moshi variants are trained
            # with a conditioning scheme ("very_good" description, etc.);
            # others have no conditioner and we pass None through to step.
            ct = None
            if model.condition_provider is not None:
                ct = model.condition_provider.condition_tensor(
                    "description", "very_good"
                )

            # STEP 2e: warmup. MLX compiles kernels on first use; doing
            # it here means the streaming session's first frame doesn't
            # eat a multi-second JIT hit. Cheap (sub-second on q8).
            model.warmup(ct)

            # STEP 2f: rustymimi tokenizer (Rust streaming codec). We
            # build it sized for `max(generated, other)` codebooks so it
            # covers both encode and decode paths. encode_step /
            # decode_step are streaming and stateful.
            generated_codebooks = lm_config.generated_codebooks
            other_codebooks = lm_config.other_codebooks
            mimi_codebooks = max(generated_codebooks, other_codebooks)
            audio_tokenizer = rustymimi.Tokenizer(
                mimi_weights, num_codebooks=mimi_codebooks
            )

            # STEP 2g: text tokenizer (SentencePiece). Used to render the
            # inner-monologue token stream into readable text.
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
