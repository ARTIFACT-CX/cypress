// AREA: inference · STREAM
// Realtime streaming session against the Python worker. Pairs a single
// open IPC stream (start_stream / audio_in / stop_stream) with the
// worker→host audio_out / stream_error events that flow alongside,
// hiding the gRPC wire format behind plain []byte / string.
//
// Concurrency: at most one Stream per Manager, matching the worker's
// own one-session-per-process limit (mimi/lm_gen are stateful). Manager
// enforces this; callers see a clean error if they double-open.
//
// SWAP: this is the seam audio.Pipeline talks to. The same shape is
// used for remote (TCP+TLS) inference because the gRPC schema doesn't
// care which transport backs it.

package inference

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// streamOutputBuffer caps the channel from the worker readLoop into the
// consumer. Matches the worker's output_queue depth so the two ends
// have the same headroom. Drop-on-full (rather than block) is critical:
// readLoop dispatches every reply on the same goroutine, so a blocked
// audio_out send would also stall stop_stream's reply and deadlock.
const streamOutputBuffer = 32

// StreamOutput is one model-step output in audio's wire shape. PCM is
// int16 LE mono at the session's sample rate (24kHz for Moshi). Text is
// non-empty only on frames where the LM emitted a real inner-monologue
// token; most frames have it empty.
type StreamOutput struct {
	PCM  []byte
	Text string
}

// Stream is a live duplex session. Construct via Manager.StartStream;
// drive via Feed/Outputs/Close. Safe to call Feed from one goroutine
// and read Outputs from another — that's the expected usage pattern,
// matching how the WS handler runs reader+writer goroutines per conn.
type Stream struct {
	mgr        *Manager
	sampleRate int

	// outputs receives audio_out chunks demuxed from the worker readLoop.
	// Buffered (see streamOutputBuffer); closed by Close once the worker
	// has acked stop_stream so consumers reading via "for range" exit.
	outputs chan StreamOutput

	// SAFETY: closeMu serializes Close so it's idempotent and so the
	// channel close happens exactly once even if WS reader and writer
	// race to call it on connection drop.
	closeMu sync.Mutex
	closed  bool
}

// SampleRate is the PCM rate (Hz) the worker reported on session open.
// The WS handler surfaces this to the UI so playback uses the right rate
// without baking 24000 in everywhere.
func (s *Stream) SampleRate() int { return s.sampleRate }

// Outputs is the chunk channel. Closed when the session ends (either
// Close or worker exit). Range-friendly:
//
//	for chunk := range stream.Outputs() { ... }
func (s *Stream) Outputs() <-chan StreamOutput { return s.outputs }

// Feed pushes one PCM chunk toward the worker. Any size accepted; the
// worker reframes internally. Blocks if the worker's input queue is
// full (~0.6s headroom by default) — that's the backpressure signal
// the WS reader should propagate by pausing its own reads.
func (s *Stream) Feed(ctx context.Context, pcm []byte) error {
	s.closeMu.Lock()
	closed := s.closed
	s.closeMu.Unlock()
	if closed {
		return errors.New("stream closed")
	}
	// PCM rides as native bytes on the gRPC wire — no base64 tax. The
	// adapter in wireconv.go pulls "pcm" out of the extra map and packs
	// it into AudioInReq.pcm directly.
	_, err := s.mgr.workerSend(ctx, "audio_in", map[string]any{"pcm": pcm})
	return err
}

// Close ends the session: tells the worker to stop, unregisters from
// the Manager so a new StartStream can succeed, and closes the outputs
// channel. Idempotent and safe from any goroutine.
func (s *Stream) Close(ctx context.Context) error {
	s.closeMu.Lock()
	if s.closed {
		s.closeMu.Unlock()
		return nil
	}
	s.closed = true
	s.closeMu.Unlock()

	// Best-effort stop. Even if the worker errors (or is already gone),
	// we still need to unhook from the Manager and close outputs so the
	// consumer exits cleanly.
	_, sendErr := s.mgr.workerSend(ctx, "stop_stream", nil)

	s.mgr.detachStream(s)
	close(s.outputs)
	return sendErr
}

// receive is called from the Manager's event handler when an audio_out
// or stream_error event arrives. Non-blocking on outputs — if the
// consumer is behind we drop the frame rather than stall the readLoop.
// PERF: dropping is the right call for audio; staleness is worse than
// gaps to a listener, and the readLoop dispatches all replies including
// stop_stream's, so blocking here would deadlock graceful shutdown.
func (s *Stream) receive(out StreamOutput) {
	s.closeMu.Lock()
	closed := s.closed
	s.closeMu.Unlock()
	if closed {
		return
	}
	select {
	case s.outputs <- out:
	default:
		// Drop. Could surface a counter here later for telemetry; not
		// worth a feedback loop in v0.1.
	}
}

// --- Manager additions -------------------------------------------------------

// StartStream opens a streaming session against the loaded model.
// Returns an error if no model is loaded or the model doesn't support
// streaming. If a previous session is still tearing down (its Close is
// in flight, e.g. the user clicked End-then-Start fast enough that the
// worker's aclose hasn't finished), we preempt it here rather than
// rejecting — a fresh user gesture should always win over a session
// the client already abandoned. The returned Stream owns the IPC
// channel until its Close runs.
func (m *Manager) StartStream(ctx context.Context) (*Stream, error) {
	m.mu.Lock()
	if m.state != StateServing {
		state := m.state
		m.mu.Unlock()
		return nil, fmt.Errorf("cannot stream: state=%s (need %s)", state, StateServing)
	}
	prev := m.activeStream
	w := m.worker
	m.mu.Unlock()

	// REASON: preempt rather than reject. The previous session's Close
	// blocks on a stop_stream IPC roundtrip (worker's aclose can take a
	// beat to wind up the model thread). Without this, a fast End→Start
	// from the UI lands in StartStream while activeStream is still set,
	// and the user gets "stream already active" for what should be a
	// natural reconnect. Close is idempotent + safe from any goroutine.
	if prev != nil {
		_ = prev.Close(ctx)
	}

	if w == nil {
		return nil, errors.New("no worker")
	}

	reply, err := w.send(ctx, "start_stream", nil)
	if err != nil {
		return nil, err
	}

	// Sample rate is optional in the reply (older fakes omit it). Fall
	// back to Moshi's native rate so the WS handler always has a number
	// to surface to the UI.
	sampleRate := 24000
	if sr, ok := reply["sample_rate"].(float64); ok {
		sampleRate = int(sr)
	}

	s := &Stream{
		mgr:        m,
		sampleRate: sampleRate,
		outputs:    make(chan StreamOutput, streamOutputBuffer),
	}

	m.mu.Lock()
	// Re-check under lock: a concurrent StartStream could have raced
	// past the first check while we were waiting on send. Whoever gets
	// the slot wins; the loser tears their own session down.
	if m.activeStream != nil {
		m.mu.Unlock()
		_, _ = w.send(ctx, "stop_stream", nil)
		return nil, errors.New("stream slot taken by concurrent caller")
	}
	m.activeStream = s
	m.mu.Unlock()
	return s, nil
}

// detachStream clears the active-stream slot. Called from Stream.Close;
// kept on the Manager so the lock that guards activeStream is here too.
func (m *Manager) detachStream(s *Stream) {
	m.mu.Lock()
	if m.activeStream == s {
		m.activeStream = nil
	}
	m.mu.Unlock()
}

// workerSend is a thin pass-through that takes the lock to read m.worker.
// Public-ish (lowercase but used from stream.go) so Stream doesn't have
// to touch m.mu directly.
func (m *Manager) workerSend(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error) {
	m.mu.Lock()
	w := m.worker
	m.mu.Unlock()
	if w == nil {
		return nil, errors.New("worker not running")
	}
	return w.send(ctx, cmd, extra)
}

// dispatchStreamEvent is called from Manager.handleEvent for audio_out
// and stream_error events. Decoupling from handleEvent keeps the event
// switch readable and keeps base64-decoding logic out of manager.go.
func (m *Manager) dispatchStreamEvent(msg map[string]any) {
	m.mu.Lock()
	s := m.activeStream
	m.mu.Unlock()
	if s == nil {
		// Late event after Close — common during graceful shutdown
		// where stop_stream's drain is still in flight. Drop silently.
		return
	}

	event, _ := msg["event"].(string)
	switch event {
	case "audio_out":
		// pcm comes through as native []byte from the gRPC adapter
		// (see wireconv.eventToMap) — no decode step. Missing/wrong
		// type is a programming error in the wire layer; we surface
		// a zero-PCM chunk rather than panicking so the consumer still
		// sees the frame's text if any.
		pcm, _ := msg["pcm"].([]byte)
		text, _ := msg["text"].(string)
		s.receive(StreamOutput{PCM: pcm, Text: text})
	case "stream_error":
		// Surfaced as a text-only chunk so the consumer sees something
		// in the same channel it's already reading. The WS handler can
		// decide to translate this into a JSON error frame for the UI.
		errStr, _ := msg["error"].(string)
		s.receive(StreamOutput{Text: "[stream_error] " + errStr})
	}
}
