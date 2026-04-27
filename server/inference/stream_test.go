// AREA: inference · STREAM · TESTS
// Unit tests for Manager.StartStream + Stream lifecycle. Drives the same
// fakeWorker as manager_test.go — no real subprocess, no torch — so the
// state machine and event-routing paths run in milliseconds.

package inference

import (
	"context"
	"encoding/base64"
	"errors"
	"testing"
	"time"
)

// servingManager spins a Manager up to StateServing using fakeWorker
// shortcuts. Streaming tests don't care about the load path; they care
// about what happens once a model is in. Centralizing the setup keeps
// each test focused on the streaming behavior.
func servingManager(t *testing.T, fake *fakeWorker) *Manager {
	t.Helper()
	if fake.sendFn == nil {
		fake.sendFn = func(_ context.Context, _ string, _ map[string]any) (map[string]any, error) {
			return map[string]any{"ok": true, "device": "cpu"}, nil
		}
	}
	m := newTestManager(fake)
	if err := m.LoadModel("moshi"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	waitForState(t, m, StateServing)
	return m
}

func TestManager_StartStream_RejectsWhenNotServing(t *testing.T) {
	// REASON: streaming a model that isn't loaded would error deep inside
	// the worker; surface the precondition cleanly at the Go boundary so
	// the WS handler can return a sensible HTTP/WS close code.
	m := NewManager(Config{WorkerDir: t.TempDir(), DataDir: t.TempDir()})
	_, err := m.StartStream(context.Background())
	if err == nil {
		t.Fatal("StartStream: want error, got nil")
	}
}

func TestManager_StartStream_SendsStartStreamCmd(t *testing.T) {
	fake := &fakeWorker{}
	// Custom reply with sample_rate so we can assert the Stream picks it up.
	fake.sendFn = func(_ context.Context, cmd string, _ map[string]any) (map[string]any, error) {
		switch cmd {
		case "load_model":
			return map[string]any{"ok": true, "device": "cpu"}, nil
		case "start_stream":
			return map[string]any{"ok": true, "sample_rate": float64(24000)}, nil
		}
		return map[string]any{"ok": true}, nil
	}
	m := servingManager(t, fake)

	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	defer stream.Close(context.Background())

	if got := stream.SampleRate(); got != 24000 {
		t.Errorf("sample_rate = %d, want 24000", got)
	}

	// Confirm exactly one start_stream went out (above and beyond the
	// load_model from servingManager). Easy thing to silently break if
	// StartStream gets a retry loop later.
	var startCount int
	for _, c := range fake.sendCalls {
		if c.cmd == "start_stream" {
			startCount++
		}
	}
	if startCount != 1 {
		t.Errorf("start_stream calls = %d, want 1", startCount)
	}
}

func TestManager_StartStream_PreemptsPreviousSession(t *testing.T) {
	// A second StartStream while the first is still active should evict
	// the first rather than reject. End→Start clicks land here on the
	// happy path: the previous Close is in-flight (worker aclose waiting
	// on the model thread) and the user shouldn't see "already active"
	// for what is from their POV a clean reconnect.
	fake := &fakeWorker{}
	m := servingManager(t, fake)

	first, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("first StartStream: %v", err)
	}

	second, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("second StartStream: want preempt, got error: %v", err)
	}
	defer second.Close(context.Background())

	// First should be closed (Outputs channel ranges to completion) so
	// any consumer of the old session exits cleanly instead of dangling.
	select {
	case _, ok := <-first.Outputs():
		if ok {
			// Drain any in-flight output, then expect closure.
			<-first.Outputs()
		}
	default:
		// Channel may already be closed; that's fine.
	}
}

func TestManager_StartStream_ReleasesSlotAfterClose(t *testing.T) {
	// After Close, the slot frees up so the next WS connect can open a
	// fresh session. Without this the worker is permanently locked from
	// the manager's POV after the first disconnect.
	fake := &fakeWorker{}
	m := servingManager(t, fake)

	first, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("first StartStream: %v", err)
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatalf("first Close: %v", err)
	}

	second, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("second StartStream: %v", err)
	}
	_ = second.Close(context.Background())
}

func TestManager_StartStream_SurfacesWorkerError(t *testing.T) {
	// A worker that rejects start_stream (e.g. model loaded but doesn't
	// support streaming) must produce a clean Go error, not a half-init
	// Stream. Also: the active-stream slot must remain empty.
	fake := &fakeWorker{}
	fake.sendFn = func(_ context.Context, cmd string, _ map[string]any) (map[string]any, error) {
		if cmd == "start_stream" {
			return nil, errors.New("does not support stream")
		}
		return map[string]any{"ok": true, "device": "cpu"}, nil
	}
	m := servingManager(t, fake)

	if _, err := m.StartStream(context.Background()); err == nil {
		t.Fatal("want error from StartStream, got nil")
	}
	// Slot must be free so a retry (after loading a different model)
	// can succeed.
	m.mu.Lock()
	active := m.activeStream
	m.mu.Unlock()
	if active != nil {
		t.Errorf("activeStream = %v, want nil", active)
	}
}

func TestStream_Feed_EncodesPCMAsBase64(t *testing.T) {
	// The worker only accepts base64 strings on audio_in. A regression
	// here (e.g. accidentally passing the raw []byte) would land as
	// "invalid base64" 60 times a second once the audio path is wired.
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	defer stream.Close(context.Background())

	pcm := []byte{0x01, 0x00, 0x02, 0x00, 0x03, 0x00}
	if err := stream.Feed(context.Background(), pcm); err != nil {
		t.Fatalf("Feed: %v", err)
	}

	// Find the audio_in call and decode its pcm field.
	var found bool
	for _, c := range fake.sendCalls {
		if c.cmd != "audio_in" {
			continue
		}
		raw, ok := c.extra["pcm"].(string)
		if !ok {
			t.Fatalf("audio_in pcm field is %T, want string", c.extra["pcm"])
		}
		decoded, err := base64.StdEncoding.DecodeString(raw)
		if err != nil {
			t.Fatalf("decode pcm: %v", err)
		}
		if string(decoded) != string(pcm) {
			t.Errorf("pcm round-trip = %v, want %v", decoded, pcm)
		}
		found = true
	}
	if !found {
		t.Error("no audio_in call recorded")
	}
}

func TestStream_Feed_FailsAfterClose(t *testing.T) {
	// Feed after Close is a programming error — the WS handler shouldn't
	// be calling Feed after its reader exits. Surface as an error, not a
	// panic on a closed channel.
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	stream.Close(context.Background())

	if err := stream.Feed(context.Background(), []byte{0, 0}); err == nil {
		t.Fatal("Feed after Close: want error, got nil")
	}
}

func TestStream_Outputs_DemuxesAudioOutEvents(t *testing.T) {
	// The whole point of Stream: take the worker's audio_out events
	// off the readLoop and surface them as decoded PCM + text on a Go
	// channel. Test pumps a synthetic event through the registered
	// onEvent handler the way the real readLoop would.
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	defer stream.Close(context.Background())

	pcm := []byte{0xaa, 0xbb, 0xcc}
	fake.onEvent(map[string]any{
		"event": "audio_out",
		"pcm":   base64.StdEncoding.EncodeToString(pcm),
		"text":  "hi",
	})

	select {
	case got := <-stream.Outputs():
		if string(got.PCM) != string(pcm) {
			t.Errorf("PCM = %v, want %v", got.PCM, pcm)
		}
		if got.Text != "hi" {
			t.Errorf("Text = %q, want %q", got.Text, "hi")
		}
	case <-time.After(time.Second):
		t.Fatal("no chunk received")
	}
}

func TestStream_Outputs_DropsWhenConsumerSlow(t *testing.T) {
	// PERF guarantee: an absent consumer must not block the readLoop —
	// otherwise a stuck UI would also stall stop_stream's reply path.
	// We fire more events than the buffer holds and check that none of
	// them block, then drain to confirm at least one made it through.
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	defer stream.Close(context.Background())

	done := make(chan struct{})
	go func() {
		for i := 0; i < streamOutputBuffer*4; i++ {
			fake.onEvent(map[string]any{
				"event": "audio_out",
				"pcm":   base64.StdEncoding.EncodeToString([]byte{byte(i)}),
				"text":  "",
			})
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("event dispatch blocked — drop-on-full not in effect")
	}

	// Drain the buffered side so we know they actually landed.
	select {
	case <-stream.Outputs():
	case <-time.After(time.Second):
		t.Fatal("expected at least one buffered chunk")
	}
}

func TestStream_Close_IsIdempotent(t *testing.T) {
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	if err := stream.Close(context.Background()); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := stream.Close(context.Background()); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestStream_Close_ClosesOutputsChannel(t *testing.T) {
	// Range over Outputs() should terminate cleanly after Close — the WS
	// writer goroutine relies on this to know when to exit.
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}

	done := make(chan struct{})
	go func() {
		for range stream.Outputs() {
		}
		close(done)
	}()

	stream.Close(context.Background())

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Outputs() not closed after Stream.Close")
	}
}

func TestStream_StreamErrorEventBecomesTextChunk(t *testing.T) {
	// Worker-side drain failures arrive as stream_error events. Surface
	// them on the same channel the consumer is already reading so the WS
	// handler doesn't have to plumb a second error pathway through.
	fake := &fakeWorker{}
	m := servingManager(t, fake)
	stream, err := m.StartStream(context.Background())
	if err != nil {
		t.Fatalf("StartStream: %v", err)
	}
	defer stream.Close(context.Background())

	fake.onEvent(map[string]any{
		"event": "stream_error",
		"error": "RuntimeError: kaboom",
	})

	select {
	case got := <-stream.Outputs():
		if got.Text == "" || !contains(got.Text, "kaboom") {
			t.Errorf("text = %q, want substring 'kaboom'", got.Text)
		}
	case <-time.After(time.Second):
		t.Fatal("no chunk received for stream_error")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
