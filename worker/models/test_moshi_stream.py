"""
AREA: worker · MODELS · MOSHI · STREAM · TESTS

Unit tests for MoshiStream. mimi/lm_gen/tokenizer are faked; torch is
real (the session uses torch tensors internally) but no model weights
are loaded — every fake returns small zero-filled tensors.
"""

import asyncio

import pytest
import torch

from .moshi_stream import (
    MoshiStream,
    StreamChunk,
    _StreamComponents,
)


# --- Fakes -------------------------------------------------------------------


class FakeMimi:
    """Minimal mimi-shaped object. encode emits a fixed shape; decode
    returns silence."""

    sample_rate = 24000
    frame_rate = 12.5  # 1920 samples / frame at 24kHz

    def __init__(self) -> None:
        self.streaming_forever_calls = 0
        self.reset_streaming_calls = 0

    def streaming_forever(self, batch_size: int) -> None:
        self.streaming_forever_calls += 1

    def reset_streaming(self) -> None:
        self.reset_streaming_calls += 1

    def encode(self, _chunk: torch.Tensor) -> torch.Tensor:
        # Real mimi returns shape (batch, num_codebooks, num_positions);
        # K=1 keeps the inner loop in _step trivial.
        return torch.zeros(1, 8, 1, dtype=torch.long)

    def decode(self, _tokens: torch.Tensor) -> torch.Tensor:
        # PCM-shape (batch, channels, frame_size) of silence. The exact
        # values don't matter for these tests; only the shape and the
        # int16 round-trip do.
        return torch.zeros(1, 1, 1920, dtype=torch.float32)


class FakeLMGen:
    """Configurable lm_gen. emit_after_steps lets a test simulate the
    real model's first-frame delay; text_id_pattern lets a test pin
    which steps emit a real text token vs padding (id=0)."""

    def __init__(
        self,
        emit_after_steps: int = 0,
        text_id_pattern: list[int] | None = None,
    ):
        self._emit_after = emit_after_steps
        self._steps = 0
        # Default pattern: emit a real (non-pad/special) token every step.
        self._text_pattern = text_id_pattern or [100]

    def streaming_forever(self, batch_size: int) -> None:
        pass

    def reset_streaming(self) -> None:
        pass

    def step(self, _codes: torch.Tensor) -> torch.Tensor | None:
        self._steps += 1
        if self._steps <= self._emit_after:
            return None
        text_id = self._text_pattern[(self._steps - 1) % len(self._text_pattern)]
        # Shape (batch=1, dep_q+1, 1). Channel 0 = text token, rest =
        # codec tokens. Values for codec channels don't matter here
        # because FakeMimi.decode returns zeros regardless.
        codec = [42] * 8
        return torch.tensor([[[text_id]] + [[c] for c in codec]], dtype=torch.long)


class FakeTokenizer:
    """SentencePiece-shaped — id_to_piece returns a token string with the
    ▁ word-start marker so tests can verify space conversion."""

    def id_to_piece(self, idx: int) -> str:
        return f"\u2581tok{idx}"


def make_components(
    lm_gen: FakeLMGen | None = None,
) -> _StreamComponents:
    return _StreamComponents(
        mimi=FakeMimi(),
        lm_gen=lm_gen or FakeLMGen(),
        text_tokenizer=FakeTokenizer(),
        device="cpu",
        frame_size=1920,
        sample_rate=24000,
    )


# Bytes for one full PCM frame of silence (int16 = 2 bytes/sample).
_FULL_FRAME = b"\x00\x00" * 1920


# --- Lifecycle ---------------------------------------------------------------


async def test_start_initializes_streaming_state():
    components = make_components()
    session = MoshiStream(components)
    await session.start()
    try:
        # streaming_forever then reset_streaming on both mimi and lm_gen.
        assert components.mimi.streaming_forever_calls == 1
        assert components.mimi.reset_streaming_calls == 1
    finally:
        await session.aclose()


async def test_start_is_idempotent():
    session = MoshiStream(make_components())
    await session.start()
    await session.start()  # second call must be a no-op, not an error
    await session.aclose()


async def test_feed_before_start_raises():
    session = MoshiStream(make_components())
    with pytest.raises(RuntimeError, match="not started"):
        await session.feed(_FULL_FRAME)


async def test_aclose_is_idempotent():
    session = MoshiStream(make_components())
    await session.start()
    await session.aclose()
    await session.aclose()  # no error


async def test_aclose_unblocks_iteration():
    # Consumer parked on __anext__ must be released by aclose, not hang.
    session = MoshiStream(make_components())
    await session.start()

    async def consumer():
        chunks = []
        async for chunk in session:
            chunks.append(chunk)
        return chunks

    consumer_task = asyncio.create_task(consumer())
    # Give the consumer a tick to park on output_queue.get().
    await asyncio.sleep(0.01)
    await session.aclose()

    chunks = await asyncio.wait_for(consumer_task, 1.0)
    # No frames fed → no chunks produced.
    assert chunks == []


# --- Buffering ---------------------------------------------------------------


async def test_partial_frame_does_not_emit():
    # Pushing < frame_size of bytes must accumulate silently; the worker
    # only sees full frames. We assert by trying to read with a short
    # timeout — should expire because nothing flowed.
    session = MoshiStream(make_components())
    await session.start()
    try:
        half_frame = b"\x00\x00" * 960  # half of mimi's 1920
        await session.feed(half_frame)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(session.__anext__(), 0.1)
    finally:
        await session.aclose()


async def test_split_frame_reassembles():
    # Feeding a frame's worth in two halves must produce the same output
    # as feeding it whole. Proves the buffer logic isn't dropping bytes.
    session = MoshiStream(make_components())
    await session.start()
    try:
        await session.feed(b"\x00\x00" * 960)
        await session.feed(b"\x00\x00" * 960)
        chunk = await asyncio.wait_for(session.__anext__(), 1.0)
        assert isinstance(chunk, StreamChunk)
    finally:
        await session.aclose()


async def test_oversized_feed_emits_multiple_frames():
    # Pushing 2.5 frames in one call should produce 2 chunks; the
    # remainder stays in the buffer.
    session = MoshiStream(make_components())
    await session.start()
    try:
        await session.feed(_FULL_FRAME * 2 + b"\x00\x00" * 960)
        c1 = await asyncio.wait_for(session.__anext__(), 1.0)
        c2 = await asyncio.wait_for(session.__anext__(), 1.0)
        assert isinstance(c1, StreamChunk)
        assert isinstance(c2, StreamChunk)
        # Third should not be ready — the half-frame remainder is buffered.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(session.__anext__(), 0.1)
    finally:
        await session.aclose()


# --- Chunk shape -------------------------------------------------------------


async def test_chunk_audio_is_int16_bytes():
    # FakeMimi.decode returns 1920 zero samples → 3840 bytes int16 LE.
    # Pin both type and length so a regression to numpy arrays or to a
    # different dtype gets caught.
    session = MoshiStream(make_components())
    await session.start()
    try:
        await session.feed(_FULL_FRAME)
        chunk = await asyncio.wait_for(session.__anext__(), 1.0)
        assert isinstance(chunk.audio_pcm, bytes)
        assert len(chunk.audio_pcm) == 3840
    finally:
        await session.aclose()


async def test_chunk_text_strips_sentencepiece_marker():
    # Token id 100 → "▁tok100" → " tok100" (▁ becomes a leading space).
    session = MoshiStream(make_components(lm_gen=FakeLMGen(text_id_pattern=[100])))
    await session.start()
    try:
        await session.feed(_FULL_FRAME)
        chunk = await asyncio.wait_for(session.__anext__(), 1.0)
        assert chunk.text == " tok100"
    finally:
        await session.aclose()


async def test_chunk_text_is_none_for_pad_tokens():
    # Token ids 0 (pad) and 3 (other special) must not surface as text —
    # they're filler the LM emits between real tokens.
    session = MoshiStream(make_components(lm_gen=FakeLMGen(text_id_pattern=[0])))
    await session.start()
    try:
        await session.feed(_FULL_FRAME)
        chunk = await asyncio.wait_for(session.__anext__(), 1.0)
        assert chunk.text is None
    finally:
        await session.aclose()


async def test_lmgen_returning_none_yields_no_chunk():
    # Real models emit None for the first few steps (delay). Session
    # must just skip those — not crash, not emit empty chunks.
    session = MoshiStream(make_components(lm_gen=FakeLMGen(emit_after_steps=2)))
    await session.start()
    try:
        # First two frames produce no chunks, third does.
        await session.feed(_FULL_FRAME * 2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(session.__anext__(), 0.1)
        await session.feed(_FULL_FRAME)
        chunk = await asyncio.wait_for(session.__anext__(), 1.0)
        assert isinstance(chunk, StreamChunk)
    finally:
        await session.aclose()
