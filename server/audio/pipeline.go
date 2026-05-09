// AREA: audio · PIPELINE
// Glues the UI-facing WebSocket transport to the inference layer. One
// WS connection = one streaming session = one Pipeline.Run. The pipeline
// itself is intentionally thin — it owns the per-connection lifecycle
// (open session, run reader+writer, close on first error/disconnect)
// but neither encodes audio frames nor knows about the model.
//
// Why a separate file from ws.go: ws.go owns transport details (upgrade,
// frame types, close codes). pipeline.go owns the *what* — open a
// session, copy bytes both ways, shut down cleanly. The WS handler
// could be replaced with a different transport (WebRTC, gRPC streaming)
// and this file would barely change.

package audio

import (
	"context"
	"errors"
	"io"
	"log"
	"sync"
)

// Pipeline routes audio between a transport (today: WebSocket) and the
// inference layer.
//
// SWAP: Pipeline is the single join point between transports and
// models. Keep it thin — no model logic, no codec logic, just plumbing.
type Pipeline struct {
	// SAFETY: depend on the InferenceClient port, never the concrete
	// inference.Manager type. Cross-feature dependencies must go
	// through an interface declared in this package (see ports.go).
	inference InferenceClient
}

// NewPipeline wires a new pipeline to the given inference client. The
// pipeline does not start anything itself — Run is invoked per WS
// connection by the WS handler.
func NewPipeline(client InferenceClient) *Pipeline {
	return &Pipeline{inference: client}
}

// Transport is the per-connection interface ws.go presents to Run. Any
// realtime-audio transport with binary in/out + a "send text event" call
// can plug in. Reader-style API (ReadFrame) rather than a callback so
// Run can drive both directions from one goroutine each, which is the
// natural shape for backpressure.
type Transport interface {
	// ReadFrame returns the next mic frame from the client. Returns
	// io.EOF on clean close. Blocks until a frame arrives or ctx
	// expires.
	ReadFrame(ctx context.Context) ([]byte, error)

	// WriteFrame sends one audio frame to the client. Treated as best-
	// effort: if the client is slow, individual frames may be dropped
	// at the transport level, but a write error here means the
	// connection is gone and Run must exit.
	WriteFrame(ctx context.Context, pcm []byte) error

	// SendOpen tells the client the session is up and what sample rate
	// to play back at. Sent once, before the first WriteFrame.
	SendOpen(ctx context.Context, sampleRate int) error

	// SendText surfaces an inner-monologue token. Empty/missing on most
	// frames; the WS handler decides whether to coalesce.
	SendText(ctx context.Context, text string) error

	// SendError surfaces a fatal error to the client before close. The
	// transport may map this onto a WS close frame with code+reason.
	SendError(ctx context.Context, message string) error
}

// Run drives one WS connection: opens a streaming session, fans bytes
// both directions, exits on first error/disconnect. Returns nil on
// clean client disconnect; non-nil if something went wrong worth
// logging at the call site.
func (p *Pipeline) Run(ctx context.Context, t Transport) error {
	// STEP 1: open the inference session. Done before any reads/writes
	// so a "no model loaded" error surfaces as a clean SendError + close
	// rather than the client having to infer it from a silent stream.
	session, err := p.inference.StartStream(ctx)
	if err != nil {
		// Best-effort error frame; client may already have hung up.
		_ = t.SendError(ctx, err.Error())
		return err
	}
	// STEP 2: announce the session up so the client knows the rate to
	// schedule playback at. Failure here means the client died between
	// connect and now — bail before allocating goroutines.
	if err := t.SendOpen(ctx, session.SampleRate()); err != nil {
		_ = session.Close(context.Background())
		return err
	}

	// STEP 3: spawn reader+writer. We use a child context so either
	// goroutine's exit cancels the other — no half-alive sessions.
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	var readErr, writeErr error
	go func() {
		defer wg.Done()
		readErr = pumpRead(runCtx, t, session)
		cancel() // wake the writer
	}()
	go func() {
		defer wg.Done()
		writeErr = pumpWrite(runCtx, t, session)
		cancel() // wake the reader
	}()

	wg.Wait()
	// Always close the session, even on a clean reader EOF — the worker
	// must release its mimi/lm_gen state for the next connection.
	_ = session.Close(context.Background())

	// Clean disconnect (reader hit EOF, writer's channel closed) is
	// success; surface only "real" errors to the caller for logging.
	return firstNonClose(readErr, writeErr)
}

// pumpRead drains the WS reader into session.Feed. Backpressure is
// natural: Feed blocks when the worker's input queue is full, which in
// turn parks ReadFrame and stalls the WS reader's TCP buffer.
func pumpRead(ctx context.Context, t Transport, session StreamSession) error {
	for {
		frame, err := t.ReadFrame(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, context.Canceled) {
				return nil
			}
			return err
		}
		if len(frame) == 0 {
			continue
		}
		if err := session.Feed(ctx, frame); err != nil {
			return err
		}
	}
}

// pumpWrite drains the session's output channel onto the WS writer.
// One PCM frame per WriteFrame call; text tokens piggyback as a
// separate SendText so the client can buffer audio independently of
// the transcript display.
func pumpWrite(ctx context.Context, t Transport, session StreamSession) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		case chunk, ok := <-session.Outputs():
			if !ok {
				// Session closed — clean exit.
				return nil
			}
			if len(chunk.PCM) > 0 {
				if err := t.WriteFrame(ctx, chunk.PCM); err != nil {
					return err
				}
			}
			if chunk.Text != "" {
				// Text errors are non-fatal: a failed text send shouldn't
				// kill an otherwise-healthy audio stream. Log so it's
				// visible during debugging.
				if err := t.SendText(ctx, chunk.Text); err != nil {
					log.Printf("audio: send text failed: %v", err)
				}
			}
			if chunk.Error != "" {
				// REASON: a worker-side stream_error means generation
				// can't continue. Forward as a {type:"error",...}
				// envelope; voiceSession.ts already promotes that to
				// the session's error state, which the existing
				// caption renders. Returning here ends the pump cleanly
				// — the worker has stopped emitting, so there's no
				// further audio to drain.
				_ = t.SendError(ctx, chunk.Error)
				return nil
			}
		}
	}
}

// firstNonClose returns whichever of the two pump errors is worth
// surfacing. Clean closes (nil, EOF, ctx-cancel) collapse to nil so the
// caller can distinguish "client disconnected" from "something broke."
func firstNonClose(a, b error) error {
	for _, e := range []error{a, b} {
		if e != nil && !errors.Is(e, io.EOF) && !errors.Is(e, context.Canceled) {
			return e
		}
	}
	return nil
}
