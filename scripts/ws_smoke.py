#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "websockets>=12",
#   "soundfile>=0.12",
#   "numpy>=1.26",
# ]
# ///
"""
AREA: scripts · WS · SMOKE

End-to-end smoke test for the Cypress audio pipeline. Connects to the
Go server's /ws endpoint, streams a wav file through at real-time pace,
and writes whatever Moshi sends back to disk + prints any inner-monologue
text tokens.

Use this to verify the whole path works (UI WS upgrade → Pipeline →
inference.Stream → IPC → MoshiStream → mimi/lm_gen → back) before the
real mic-capture UI lands. Run with:

    ./scripts/ws_smoke.py path/to/input.wav path/to/output.wav

PEP 723 inline metadata above means uv resolves deps automatically — no
venv to set up. Requires `uv` on PATH.

Pacing: real-time by default (one 80ms frame per 80ms wall clock) so
Moshi sees the same input rhythm a real mic would. The model is a
duplex streaming model; flooding it with all the input at once would
trigger backpressure and distort the timing of its responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import soundfile as sf
import websockets

# SETUP: must match the worker's mimi config. 1920 samples = 80ms at
# 24kHz, which is one mimi frame. The server reframes anyway, but
# sending in mimi-frame chunks keeps the WS traffic + log output
# legible (one binary frame in == one binary frame back).
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes/sample
FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE  # 0.08s

DEFAULT_URL = "ws://127.0.0.1:7842/ws"


def load_pcm(path: str) -> bytes:
    """Read a wav file and return int16 LE mono PCM at SAMPLE_RATE.

    soundfile returns float32 in [-1, 1] regardless of source dtype, so
    the conversion to int16 is uniform. Stereo gets averaged down to
    mono — moshi only takes one channel anyway and a hard-error on
    stereo input would be more annoying than helpful for testing."""
    audio, sr = sf.read(path, always_2d=True)
    if sr != SAMPLE_RATE:
        # Cheap linear resample — fine for a smoke test, not for real
        # audio quality. Tell the user so they don't draw conclusions
        # about Moshi's quality from a badly-resampled input.
        print(
            f"[smoke] note: input is {sr}Hz; doing a quick linear resample "
            f"to {SAMPLE_RATE}Hz. Quality will suffer — convert with ffmpeg "
            f"for a better test: ffmpeg -i in.wav -ar {SAMPLE_RATE} -ac 1 out.wav",
            file=sys.stderr,
        )
        ratio = SAMPLE_RATE / sr
        new_len = int(audio.shape[0] * ratio)
        # Resample each channel separately, then take channel 0.
        idx = np.linspace(0, audio.shape[0] - 1, new_len)
        resampled = np.empty((new_len, audio.shape[1]), dtype=audio.dtype)
        for c in range(audio.shape[1]):
            resampled[:, c] = np.interp(idx, np.arange(audio.shape[0]), audio[:, c])
        audio = resampled

    # Mono mixdown.
    mono = audio.mean(axis=1)
    # float [-1, 1] → int16 LE. Clip first so an out-of-range sample
    # doesn't wrap to a loud pop on the wire.
    int16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    return int16.tobytes()


def chunked(data: bytes, size: int):
    # Generator over fixed-size chunks. Trailing partial chunk is left
    # for the caller — the server's reframer handles it but we'd rather
    # not send a half-frame at end-of-input that the worker buffers
    # without ever emitting.
    for i in range(0, len(data) - size + 1, size):
        yield data[i : i + size]


async def run(url: str, input_path: str, output_path: str, pace: float) -> int:
    pcm_in = load_pcm(input_path)
    total_frames = len(pcm_in) // FRAME_BYTES
    print(
        f"[smoke] loaded {len(pcm_in) / FRAME_BYTES * FRAME_DURATION:.1f}s "
        f"of audio ({total_frames} frames)",
        file=sys.stderr,
    )

    out_chunks: list[bytes] = []
    sample_rate_out = SAMPLE_RATE  # overwritten by the open envelope

    async with websockets.connect(url, max_size=1 << 22) as ws:
        # STEP 1: handshake — first server message is the open envelope.
        # Anything else (or no message) means something's wrong server-
        # side and we should bail with a clear hint.
        first = await asyncio.wait_for(ws.recv(), timeout=10.0)
        if isinstance(first, bytes):
            print("[smoke] server sent binary before open envelope; aborting")
            return 2
        env = json.loads(first)
        if env.get("type") == "error":
            print(f"[smoke] server rejected session: {env.get('message')}")
            return 3
        if env.get("type") != "open":
            print(f"[smoke] unexpected first frame: {env}")
            return 4
        sample_rate_out = int(env.get("sample_rate", SAMPLE_RATE))
        print(f"[smoke] session open; output sample_rate={sample_rate_out}", file=sys.stderr)

        # STEP 2: spawn a reader task that drains everything the server
        # sends — binary PCM into out_chunks, text envelopes to stdout.
        # Lives separately from the writer so audio_out events keep
        # flowing while we're paced-sleeping between mic frames.
        async def reader():
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        out_chunks.append(msg)
                    else:
                        env = json.loads(msg)
                        kind = env.get("type")
                        if kind == "text":
                            # Inner-monologue token; print without
                            # newlines so concatenation reads naturally.
                            print(env.get("data", ""), end="", flush=True)
                        elif kind == "error":
                            print(f"\n[smoke] error from server: {env.get('message')}")
                        else:
                            print(f"\n[smoke] unknown envelope: {env}")
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(reader())

        # STEP 3: stream input at real-time pace. We track an absolute
        # deadline rather than `sleep(FRAME_DURATION)` per iteration so
        # any slow send doesn't permanently shift the schedule into the
        # future.
        start = time.monotonic()
        for i, frame in enumerate(chunked(pcm_in, FRAME_BYTES)):
            await ws.send(frame)
            target = start + (i + 1) * FRAME_DURATION / pace
            now = time.monotonic()
            if target > now:
                await asyncio.sleep(target - now)

        # STEP 4: give the model a moment to flush its last few frames
        # of response before we close. Moshi has internal delay, so
        # cutting off at end-of-input would truncate the answer.
        print("\n[smoke] input finished; waiting 2s for tail audio…", file=sys.stderr)
        await asyncio.sleep(2.0)
        await ws.close()

        # Drain reader so any frames in flight at close-time land.
        try:
            await asyncio.wait_for(reader_task, timeout=2.0)
        except asyncio.TimeoutError:
            reader_task.cancel()

    # STEP 5: write the response wav. Same dtype/rate the server sent
    # us; soundfile accepts int16 directly.
    pcm_out = b"".join(out_chunks)
    if not pcm_out:
        print("[smoke] no audio came back — model may have produced silence", file=sys.stderr)
        return 0
    samples = np.frombuffer(pcm_out, dtype=np.int16)
    sf.write(output_path, samples, sample_rate_out)
    print(
        f"[smoke] wrote {len(samples) / sample_rate_out:.1f}s to {output_path}",
        file=sys.stderr,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push a wav file through the Cypress WS audio path and capture the response."
    )
    parser.add_argument("input", help="input .wav (any rate; mono preferred)")
    parser.add_argument("output", help="output .wav for the model's response")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"WebSocket URL (default {DEFAULT_URL})",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="playback speed multiplier (1.0 = real-time; >1 sends faster, "
        "useful only for stress-testing backpressure)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.url, args.input, args.output, args.pace)))


if __name__ == "__main__":
    main()
