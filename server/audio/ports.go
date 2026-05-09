// AREA: audio · PORTS
// Outbound ports for the audio feature. The pipeline talks to other
// features (today: inference) only through interfaces declared here, so
// this file is the entire surface area of audio's dependencies on the
// rest of the server. Concrete implementations are injected at startup
// from main.go.
//
// SWAP: any of these interfaces can be replaced with a fake for tests
// or with a different backend (e.g. a remote inference client) without
// touching the pipeline.

package audio

import "context"

// InferenceClient is the part of the inference subsystem that audio
// needs to know about. Kept narrow on purpose — a wider interface would
// re-couple audio to inference internals.
//
// The concrete implementation is *inference.Manager, wired in main.go.
// Keeping it as an interface here means audio can be tested with a
// fake and the inference feature can grow new methods without audio
// caring.
type InferenceClient interface {
	// StartStream opens a duplex streaming session against the loaded
	// model. Returns an error if no model is loaded or another session
	// is already active.
	StartStream(ctx context.Context) (StreamSession, error)
}

// StreamSession is one live duplex session. The WS handler runs a
// reader goroutine pumping mic frames into Feed and a writer goroutine
// draining Outputs back to the client. Close ends both.
type StreamSession interface {
	// Feed pushes one PCM chunk (int16 LE mono at SampleRate) toward
	// the model. Any size accepted; the inference layer reframes.
	Feed(ctx context.Context, pcm []byte) error

	// Outputs is the model's audio + text chunk channel. Closes when
	// the session ends so a `for range` loop in the WS writer exits.
	Outputs() <-chan StreamOutput

	// SampleRate is the PCM rate (Hz) for both Feed and Outputs PCM.
	// The WS handler surfaces this to the UI on session open so
	// playback uses the right rate.
	SampleRate() int

	// Close ends the session. Idempotent; safe to call from any
	// goroutine. WS reader and writer both call it on exit so
	// whichever side notices the disconnect first wins.
	Close(ctx context.Context) error
}

// StreamOutput is one model-step output. Mirrors the inference shape so
// the WS handler can send it on without translation.
type StreamOutput struct {
	// PCM is int16 LE mono at SampleRate. Always non-empty on a real
	// audio_out frame.
	PCM []byte
	// Text is the inner-monologue token if the model emitted one this
	// step; empty on the (majority of) frames that have only audio.
	Text string
	// Error carries a worker-side stream_error (e.g. CUDA crash, mimi
	// shape mismatch). Non-empty signals "generation can't continue"
	// — the WS handler maps this to a {type:"error",...} envelope so
	// the UI shows a banner instead of treating it as transcript text.
	Error string
}
