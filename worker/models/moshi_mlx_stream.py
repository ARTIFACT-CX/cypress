"""
AREA: worker · MODELS · MOSHI · MLX · STREAM

Realtime streaming session for the MLX Moshi backend. Same external
shape as moshi_stream.py (start / feed / async iteration / aclose) so
the IPC layer can drive either backend without caring which one's
loaded.

Core differences from the torch version:
  - No `streaming_forever` context manager — `LmGen` is created fresh
    per session and `rustymimi.Tokenizer` exposes streaming `encode_step`
    / `decode_step` calls without needing entry/exit.
  - Audio tokenizer is Rust-backed (rustymimi). Encode returns codec
    tokens directly; we don't poll a queue like the upstream `local.py`
    example does because we already have an asyncio queue around it.
  - text and audio outputs are demuxed at the `LmGen` level: `step()`
    returns the text token, `last_audio_tokens()` retrieves the audio
    tokens (or None during the model's warmup-delay window).

Concurrency: one session per loaded model — the `Tokenizer` and `LmGen`
are stateful. The model owns those, hands fresh ones to each session
via _StreamComponents, and `aclose` discards them.

SWAP: this is the in-process interface between the audio transport and
the MLX model. Same shape as MoshiStream so a future remote-inference
backend can implement either against gRPC streaming.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional


class StreamChunk(NamedTuple):
    """One model-step output. audio_pcm is exactly mimi.frame_size samples
    of int16 LE mono PCM at 24kHz; empty bytes when the model emitted
    only text on this step (rare). text is set when the model emitted a
    real (non-padding) inner-monologue token."""

    audio_pcm: bytes
    text: Optional[str]


# Sentinel objects on the internal queues. Distinct singletons so a real
# zero-byte PCM payload can't ever be mistaken for "stream closed."
_INPUT_EOF = object()
_OUTPUT_EOF = object()


@dataclass
class _StreamComponents:
    """Everything MoshiMlxStream needs to run a step. Built fresh by
    MoshiMlx.stream() per session; tests construct one with fakes."""

    mimi: Any  # rustymimi.Tokenizer
    lm_gen: Any  # moshi_mlx.models.LmGen
    text_tokenizer: Any  # sentencepiece.SentencePieceProcessor
    condition_tensor: Any  # ConditionTensor or None
    generated_codebooks: int
    other_codebooks: int
    frame_size: int  # 1920 = 80ms at 24kHz
    sample_rate: int  # 24000


# Same depth as the torch backend so the two have matching headroom and
# the WS layer can stay unaware of which is loaded. See moshi_stream.py
# for the rationale on the depth choice.
_INPUT_QUEUE_DEPTH = 8
_OUTPUT_QUEUE_DEPTH = 32


class MoshiMlxStream:
    """Async push/pull session against a loaded MoshiMlx model. Mirrors
    MoshiStream's lifecycle:

        session = moshi_mlx.stream()
        await session.start()
        await session.feed(pcm_bytes)
        async for chunk in session:
            ...
        await session.aclose()
    """

    def __init__(self, components: _StreamComponents):
        self._c = components
        self._input_queue: asyncio.Queue = asyncio.Queue(_INPUT_QUEUE_DEPTH)
        self._output_queue: asyncio.Queue = asyncio.Queue(_OUTPUT_QUEUE_DEPTH)
        self._worker_task: Optional[asyncio.Task] = None
        self._pcm_buffer = bytearray()
        self._frame_bytes = components.frame_size * 2  # int16 = 2 bytes/sample
        self._started = False
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return self._c.sample_rate

    async def start(self) -> None:
        """Spawn the generation worker. Idempotent. No streaming-context
        setup needed (unlike the torch backend) — `LmGen` and `Tokenizer`
        are already in streaming-ready state from construction."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("session already closed")
        self._worker_task = asyncio.create_task(self._run(), name="moshi-mlx-stream")
        self._started = True

    async def feed(self, pcm: bytes) -> None:
        """Push int16 LE mono PCM. Reframes to mimi.frame_size internally;
        any input size is accepted."""
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        if self._closed:
            raise RuntimeError("session closed")
        self._pcm_buffer.extend(pcm)
        while len(self._pcm_buffer) >= self._frame_bytes:
            frame = bytes(self._pcm_buffer[: self._frame_bytes])
            del self._pcm_buffer[: self._frame_bytes]
            await self._input_queue.put(frame)

    def __aiter__(self) -> "MoshiMlxStream":
        return self

    async def __anext__(self) -> StreamChunk:
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        item = await self._output_queue.get()
        if item is _OUTPUT_EOF:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        """Stop the worker. Idempotent and safe from any task. Resets the
        Rust mimi tokenizer so a follow-up session starts on a clean
        state machine."""
        if self._closed:
            return
        self._closed = True
        try:
            self._input_queue.put_nowait(_INPUT_EOF)
        except asyncio.QueueFull:
            pass
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

        # SAFETY: rustymimi.Tokenizer is stateful — without a reset, the
        # next session's first encode_step would start from this session's
        # last position. LmGen is throwaway (one per session) so it
        # doesn't need a reset; only the tokenizer survives across
        # sessions because it lives on the model.
        try:
            self._c.mimi.reset()
        except Exception:
            pass

    async def _run(self) -> None:
        """Generation worker. Per frame: encode PCM via rustymimi → step
        LmGen for text token → fetch audio tokens → decode → emit."""
        try:
            while True:
                frame = await self._input_queue.get()
                if frame is _INPUT_EOF:
                    break
                chunks = await asyncio.to_thread(self._step, frame)
                for chunk in chunks:
                    await self._output_queue.put(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            await self._output_queue.put(_OUTPUT_EOF)

    def _step(self, frame_pcm: bytes) -> "list[StreamChunk]":
        """One frame through encode → LM step → decode. Imports lazy to
        keep the worker importable when mlx isn't available."""
        import mlx.core as mx
        import numpy as np

        # int16 LE bytes → float32 in [-1, 1]. rustymimi expects shape
        # (batch=1, channels=1, samples) for encode_step.
        samples = np.frombuffer(frame_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        pcm_for_mimi = samples[None, None, :]

        # STEP 1: encode PCM → codec tokens for the LM. The transpose +
        # slice mirrors upstream run_inference.py — mimi may produce
        # extra codebooks beyond what the LM consumes; we trim to
        # other_codebooks.
        other_audio_tokens = self._c.mimi.encode_step(pcm_for_mimi)
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)
        other_audio_tokens = other_audio_tokens[:, :, : self._c.other_codebooks]

        # STEP 2: step the LM. Returns a (batch=1,) array of text tokens.
        # condition_tensor is None for unconditioned models (default).
        text_token = self._c.lm_gen.step(other_audio_tokens[0], self._c.condition_tensor)
        text_id = int(text_token[0].item())

        text: Optional[str] = None
        # Token ids 0 (pad) and 3 (other special) are filler — drop, same
        # filtering the upstream server uses.
        if text_id not in (0, 3):
            piece = self._c.text_tokenizer.id_to_piece(text_id)
            # SentencePiece prefixes word-starts with U+2581 (▁); turn
            # back into a regular leading space.
            text = piece.replace("\u2581", " ")

        out: "list[StreamChunk]" = []
        # STEP 3: fetch the audio tokens for this step and decode them.
        # last_audio_tokens() returns None during the model's warmup
        # delay window (first ~25 frames) — emit a text-only chunk if we
        # have text but no audio so the consumer still sees inner-
        # monologue progress.
        audio_tokens = self._c.lm_gen.last_audio_tokens()
        if audio_tokens is not None and self._c.generated_codebooks > 0:
            audio_tokens_np = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = self._c.mimi.decode_step(audio_tokens_np)
            out_np = np.asarray(out_pcm).reshape(-1)
            # float [-1, 1] → int16 LE bytes; clip first so a slightly
            # out-of-range sample doesn't wrap to a loud pop.
            out_int16 = (np.clip(out_np, -1.0, 1.0) * 32767.0).astype(np.int16)
            out.append(StreamChunk(audio_pcm=out_int16.tobytes(), text=text))
        elif text is not None:
            # Text without audio: emit so the inner-monologue still
            # appears in the wire stream during warmup.
            out.append(StreamChunk(audio_pcm=b"", text=text))

        return out
