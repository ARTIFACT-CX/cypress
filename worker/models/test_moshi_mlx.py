"""
AREA: worker · MODELS · MOSHI · MLX · TESTS

Unit tests for the MLX Moshi backend's lifecycle and registration. We
don't load real weights here — those tests live in test_integration.py
behind the `integration` marker. What we cover:

- Registration under the explicit `"moshi-mlx"` name.
- Initial state is unloaded (device reported, no model object).
- The streaming session adapts to fakes correctly (so the IPC tests
  can drive moshi-mlx without real weights — see test_commands.py's
  FakeStreamingSessionModel pattern).
"""

from __future__ import annotations

import asyncio

import pytest

from . import moshi_mlx
from .moshi_mlx_stream import MoshiMlxStream, _StreamComponents, StreamChunk


def test_moshi_mlx_registered_under_explicit_name():
    from . import REGISTRY

    assert "moshi-mlx" in REGISTRY
    assert REGISTRY["moshi-mlx"] is moshi_mlx.MoshiMlx


def test_moshi_mlx_initial_device_is_mlx():
    # The mlx backend reports its device as "mlx" before load runs so the
    # UI status panel doesn't show a stale or empty device string.
    instance = moshi_mlx.MoshiMlx(emit=lambda _msg: None)
    assert instance.device() == "mlx"


def test_moshi_mlx_stream_raises_when_not_loaded():
    # stream() pre-flights on _model so the IPC handler can surface a
    # useful error before allocating queues / spawning the worker task.
    instance = moshi_mlx.MoshiMlx(emit=lambda _msg: None)
    with pytest.raises(RuntimeError, match="not loaded"):
        instance.stream()


# --- Stream session lifecycle (fake components) ----------------------------


class _FakeMimi:
    """Fake rustymimi.Tokenizer. encode_step / decode_step return shape-
    correct numpy arrays so the real numpy/mx code in MoshiMlxStream._step
    can run without spinning up MLX or Metal."""

    def __init__(self):
        self.reset_calls = 0
        self.encode_calls = 0
        self.decode_calls = 0

    def encode_step(self, _pcm):
        import numpy as np

        self.encode_calls += 1
        # Shape: (batch=1, codebooks=8, frames=1) of int32 token ids.
        return np.zeros((1, 8, 1), dtype=np.int32)

    def decode_step(self, _tokens):
        import numpy as np

        self.decode_calls += 1
        # 1920 float32 samples in [-1, 1].
        return np.zeros((1, 1, 1920), dtype=np.float32)

    def reset(self):
        self.reset_calls += 1


class _FakeLmGen:
    """Fake LmGen. step() returns a deterministic text token id; the
    backing audio tokens come from last_audio_tokens()."""

    def __init__(self, text_id: int = 5, emit_audio: bool = True):
        self._text_id = text_id
        self._emit_audio = emit_audio
        self.steps = 0

    def step(self, _other_audio_tokens, _ct):
        import mlx.core as mx

        self.steps += 1
        return mx.array([self._text_id])

    def last_audio_tokens(self):
        if not self._emit_audio:
            return None
        import mlx.core as mx

        # Shape: (batch=1, codebooks=8). The stream code adds a trailing
        # dimension before passing to decode_step.
        return mx.zeros((1, 8), dtype=mx.int32)


class _FakeTextTokenizer:
    def id_to_piece(self, token_id: int) -> str:
        return f"\u2581tok{token_id}"


def _build_components(mimi=None, lm_gen=None, text_tokenizer=None) -> _StreamComponents:
    return _StreamComponents(
        mimi=mimi or _FakeMimi(),
        lm_gen=lm_gen or _FakeLmGen(),
        text_tokenizer=text_tokenizer or _FakeTextTokenizer(),
        condition_tensor=None,
        generated_codebooks=8,
        other_codebooks=8,
        frame_size=1920,
        sample_rate=24000,
    )


@pytest.mark.skipif(
    pytest.importorskip("mlx.core", reason="mlx not available") is None,
    reason="mlx required for stream _step",
)
async def test_stream_emits_chunks_and_resets_mimi_on_aclose():
    # Drive the streaming worker through fakes for one frame and verify:
    # (1) we get a StreamChunk back with both audio and text;
    # (2) aclose calls mimi.reset so a follow-up session starts clean.
    mimi = _FakeMimi()
    session = MoshiMlxStream(_build_components(mimi=mimi))
    await session.start()

    # One frame of silence, sized to the worker's expected frame_bytes
    # (1920 samples * 2 bytes int16).
    await session.feed(b"\x00\x00" * 1920)

    chunk = await asyncio.wait_for(session.__anext__(), timeout=5.0)
    assert isinstance(chunk, StreamChunk)
    assert chunk.text == " tok5"  # ▁tok5 with U+2581 → space
    assert len(chunk.audio_pcm) == 1920 * 2  # int16 LE bytes

    await session.aclose()
    assert mimi.reset_calls == 1


async def test_stream_feed_before_start_raises():
    session = MoshiMlxStream(_build_components())
    with pytest.raises(RuntimeError, match="not started"):
        await session.feed(b"")


async def test_stream_aclose_is_idempotent():
    session = MoshiMlxStream(_build_components())
    await session.start()
    await session.aclose()
    # Second aclose is a no-op; must not raise or double-call reset.
    await session.aclose()
