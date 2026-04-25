"""
AREA: worker · MODELS · MOSHI · STREAM

Realtime streaming session against a loaded Moshi model. Hot path between
the audio transport (future: WS+UDS) and the model. Caller pushes raw PCM
in via .feed(); pulls StreamChunk objects (audio + optional text token)
out via async iteration.

Buffering: the caller can push arbitrary-sized PCM chunks; this session
accumulates into mimi's native frame_size (1920 samples = 80ms at 24kHz)
before running each model step. That lets the WS layer ship whatever
frame size is convenient (typically 20-40ms) without per-call alignment.

Concurrency: one session per loaded model — mimi and lm_gen are stateful
so a second concurrent session would corrupt both. Construct via
Moshi.stream(), drive from two coroutines (one feeding, one draining).

SWAP: this is the in-process interface between the audio transport and
the model. A future remote-inference backend implements the same shape
against gRPC streaming.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional


class StreamChunk(NamedTuple):
    """One model-step output. audio_pcm is exactly mimi.frame_size samples
    of int16 LE mono PCM at the model's native sample rate (24kHz for
    moshiko). text is set when the model emitted a real (non-padding)
    inner-monologue token this step; typically present on a fraction of
    frames since text is much sparser than audio."""

    audio_pcm: bytes
    text: Optional[str]


# Sentinel objects on the internal queues. Distinct singletons so a real
# zero-byte PCM payload can't ever be mistaken for "stream closed."
_INPUT_EOF = object()
_OUTPUT_EOF = object()


@dataclass
class _StreamComponents:
    """Everything MoshiStream needs to run a step. Built by Moshi.stream()
    from its loaded weights; tests construct one directly with fakes."""

    mimi: Any
    lm_gen: Any
    text_tokenizer: Any
    device: str
    frame_size: int  # mimi samples per frame, e.g. 1920
    sample_rate: int


# REASON: cap on input frames buffered before backpressuring the producer.
# Each frame is ~80ms; 8 frames = ~0.6s of headroom. Higher values mask
# bursty WS delivery but stretch latency under overload; lower values
# surface congestion sooner. Tune once we have real traces.
_INPUT_QUEUE_DEPTH = 8

# Output queue can be deeper because the consumer (audio writer) is
# typically faster than the producer (model step). We just don't want
# unbounded growth if the consumer stalls entirely.
_OUTPUT_QUEUE_DEPTH = 32


class MoshiStream:
    """Async push/pull session against a loaded Moshi model. Lifecycle:

        session = moshi.stream()
        await session.start()           # warmup, ready
        await session.feed(pcm_bytes)   # any size; session reframes
        async for chunk in session:     # StreamChunk(audio_pcm, text)
            ...
        await session.aclose()          # frees worker task, queues
    """

    def __init__(self, components: _StreamComponents):
        self._c = components
        # Bounded queues give us natural backpressure. The worker drains
        # the input queue; the consumer drains the output queue.
        self._input_queue: asyncio.Queue = asyncio.Queue(_INPUT_QUEUE_DEPTH)
        self._output_queue: asyncio.Queue = asyncio.Queue(_OUTPUT_QUEUE_DEPTH)
        self._worker_task: Optional[asyncio.Task] = None
        # int16 LE byte buffer accumulating until we have a whole frame.
        # Bytearray (not bytes) so extend / del are O(1) at the front.
        self._pcm_buffer = bytearray()
        self._frame_bytes = components.frame_size * 2  # int16 = 2 bytes/sample
        self._started = False
        self._closed = False

    @property
    def sample_rate(self) -> int:
        """Output PCM sample rate in Hz. Surfaced so the IPC layer (and
        eventually the audio transport) can tell consumers the playback
        rate without having to peek at mimi internals."""
        return self._c.sample_rate

    async def start(self) -> None:
        """Initialize streaming state and spawn the worker task. Must be
        awaited before feed() / iteration. Idempotent."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("session already closed")
        # STEP 1: prime the streaming caches and reset any stale state.
        # mimi and lm_gen are stateful — running them fresh against new
        # audio without reset would carry over the previous session's
        # context.
        await asyncio.to_thread(self._init_streaming)
        # STEP 2: spawn the generation worker. Its lifetime is tied to
        # this session — aclose() cancels it.
        self._worker_task = asyncio.create_task(self._run(), name="moshi-stream")
        self._started = True

    def _init_streaming(self) -> None:
        # mimi/lm_gen.streaming_forever(N) allocates streaming caches
        # sized for batch=N; reset_streaming clears any prior state in
        # those caches. Order matters: configure capacity, then reset.
        self._c.mimi.streaming_forever(1)
        self._c.lm_gen.streaming_forever(1)
        self._c.mimi.reset_streaming()
        self._c.lm_gen.reset_streaming()

    async def feed(self, pcm: bytes) -> None:
        """Push int16 LE mono PCM into the session. Any size accepted —
        the session reframes to mimi.frame_size internally. Awaits if
        the input queue is full (backpressure)."""
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        if self._closed:
            raise RuntimeError("session closed")
        self._pcm_buffer.extend(pcm)
        # Flush every complete frame to the worker. The remainder stays
        # in the buffer for the next feed().
        while len(self._pcm_buffer) >= self._frame_bytes:
            frame = bytes(self._pcm_buffer[: self._frame_bytes])
            del self._pcm_buffer[: self._frame_bytes]
            await self._input_queue.put(frame)

    def __aiter__(self) -> "MoshiStream":
        return self

    async def __anext__(self) -> StreamChunk:
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        item = await self._output_queue.get()
        if item is _OUTPUT_EOF:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        """Stop the worker, drain queues. Idempotent and safe from any
        task. Always wait for the previous session to close before
        starting a new one — they share mimi/lm_gen state."""
        if self._closed:
            return
        self._closed = True
        # Tell the worker to stop. put_nowait avoids deadlock when the
        # input queue is full — cancel() below picks up that case.
        try:
            self._input_queue.put_nowait(_INPUT_EOF)
        except asyncio.QueueFull:
            pass
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                # Worker exceptions during teardown aren't actionable —
                # we're throwing the session away anyway.
                pass

    async def _run(self) -> None:
        """Generation worker. Runs until it sees _INPUT_EOF or is
        cancelled. Per frame: encode PCM → step LM → decode emitted
        tokens → emit chunks. CPU/GPU work goes through to_thread so
        the asyncio loop stays responsive (shutdown commands, the
        consumer draining outputs, etc.)."""
        try:
            while True:
                frame = await self._input_queue.get()
                if frame is _INPUT_EOF:
                    break
                chunks = await asyncio.to_thread(self._step, frame)
                for chunk in chunks:
                    await self._output_queue.put(chunk)
        except asyncio.CancelledError:
            # Cooperative shutdown via aclose(). Don't re-raise — the
            # finally clause below puts the EOF sentinel so the consumer
            # exits cleanly, and we don't want CancelledError to escape
            # past the task boundary.
            pass
        finally:
            # Always send EOF so a consumer parked on __anext__ exits
            # rather than hanging forever.
            await self._output_queue.put(_OUTPUT_EOF)

    def _step(self, frame_pcm: bytes) -> "list[StreamChunk]":
        """Run one frame through the model. Returns 0 or more chunks —
        models with a delay (lm_gen.lm_model.delays > 0) emit None for
        the first few steps before output starts flowing, and even after
        that lm_gen returns None on intermediate code positions."""
        # Imports lazy mirror moshi.py: keep the worker importable when
        # torch is misbehaving, surfacing the failure only when streaming
        # is actually attempted.
        import numpy as np
        import torch

        # SAFETY: inference_mode disables autograd graph construction *and*
        # version-counter tracking. Without it, every lm_gen.step() retains
        # the full forward graph; over hundreds of streamed frames that's
        # GBs of dangling tensors, and on Mac unified memory the OS swaps
        # itself to death (this crashed a machine before the wrapper went
        # in). run_wav has the equivalent torch.no_grad() — keep both paths
        # gradient-free.
        with torch.inference_mode():
            # int16 LE bytes → float32 in [-1, 1], shape (frame_size,)
            samples = np.frombuffer(frame_pcm, dtype=np.int16).astype(np.float32)
            samples /= 32768.0
            chunk_t = torch.from_numpy(samples).to(self._c.device)[None, None]

            codes = self._c.mimi.encode(chunk_t)

            out: "list[StreamChunk]" = []
            # codes.shape[-1] is the number of code positions per audio frame —
            # typically 1, but the loop generalizes for safety.
            for c in range(codes.shape[-1]):
                tokens = self._c.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                # tokens shape: (batch=1, dep_q+1, 1). Channel 0 = text token;
                # channels 1.. = codec tokens that decode into PCM.
                text_id = int(tokens[0, 0, 0].item())
                text: Optional[str] = None
                # Token ids 0 (pad) and 3 (other special) are filler — not
                # part of the spoken inner monologue, so we drop them. This
                # matches upstream's filtering in moshi.server.
                if text_id not in (0, 3):
                    piece = self._c.text_tokenizer.id_to_piece(text_id)
                    # SentencePiece prefixes word starts with U+2581 (▁); turn
                    # back into a regular leading space so consumers can just
                    # concatenate pieces into a readable string.
                    text = piece.replace("▁", " ")

                audio_t = self._c.mimi.decode(tokens[:, 1:]).cpu()  # (1, 1, N)
                audio_np = audio_t[0, 0].numpy()
                # float [-1, 1] → int16 LE bytes for the wire. Clip first so a
                # mildly out-of-range sample doesn't wrap around to a loud pop.
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                out.append(StreamChunk(audio_pcm=audio_int16.tobytes(), text=text))

            return out
