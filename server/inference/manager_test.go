// AREA: inference · TESTS
// Unit tests for the Manager state machine. The real subprocess spawner is
// swapped out for a fakeWorker so these run in milliseconds and don't need
// uv / Python on the test host.

package inference

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/workers"
)

// fakeWorker stands in for the real *worker in tests. Each behavior the
// Manager cares about (send result, stop, event registration) is
// configurable via fields so individual tests can shape its responses.
type fakeWorker struct {
	mu sync.Mutex

	// sendFn lets a test pick exactly what send returns. Defaults to a
	// generic ok reply.
	sendFn func(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error)

	// stopErr is what stop returns. Default: nil.
	stopErr error

	// stopFn, if set, runs in stop() instead of just returning stopErr.
	// Used by tests that need to observe a stop call (e.g. the
	// idle-worker shutdown path in maybeRemoveFamilyEnv).
	stopFn func(ctx context.Context) error

	// onEvent captures whatever the Manager registered, so tests can fire
	// synthetic events at the Manager's handler.
	onEvent func(map[string]any)

	// sendCalls records every send for ordering / arg assertions.
	sendCalls []sendCall
}

type sendCall struct {
	cmd   string
	extra map[string]any
}

func (f *fakeWorker) Send(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error) {
	f.mu.Lock()
	f.sendCalls = append(f.sendCalls, sendCall{cmd: cmd, extra: extra})
	fn := f.sendFn
	f.mu.Unlock()
	if fn != nil {
		return fn(ctx, cmd, extra)
	}
	return map[string]any{"ok": true}, nil
}

func (f *fakeWorker) Stop(ctx context.Context) error {
	f.mu.Lock()
	fn := f.stopFn
	f.mu.Unlock()
	if fn != nil {
		return fn(ctx)
	}
	return f.stopErr
}

func (f *fakeWorker) SetOnEvent(fn func(map[string]any)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.onEvent = fn
}

// waitForState polls Status until it matches `want` or the deadline elapses.
// Used because LoadModel is fire-and-forget — doLoad runs on a goroutine so
// the test must wait for it to land in the terminal state.
func waitForState(t *testing.T, m *Manager, want State) Snapshot {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		snap := m.Status()
		if snap.State == want {
			return snap
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for state=%q, last status: %+v", want, m.Status())
	return Snapshot{}
}

// newTestManager wires a Manager to a fakeWorker via a one-shot spawner.
// Returned alongside the fake so tests can assert on send calls / fire
// events.
func newTestManager(fake *fakeWorker) *Manager {
	return newManagerWithSpawn(func(_ context.Context, _ string) (workers.Handle, error) {
		return fake, nil
	})
}

func TestManager_NewManager_StartsIdle(t *testing.T) {
	m := NewManager(Config{WorkerDir: t.TempDir(), DataDir: t.TempDir()})
	if got := m.Status().State; got != StateIdle {
		t.Fatalf("state = %q, want %q", got, StateIdle)
	}
}

func TestManager_LoadModel_Success_TransitionsToServing(t *testing.T) {
	fake := &fakeWorker{
		sendFn: func(_ context.Context, _ string, _ map[string]any) (map[string]any, error) {
			return map[string]any{"ok": true, "device": "mps"}, nil
		},
	}
	m := newTestManager(fake)

	if err := m.LoadModel("moshi"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	snap := waitForState(t, m, StateServing)
	if snap.Model != "moshi" {
		t.Errorf("model = %q, want %q", snap.Model, "moshi")
	}
	if snap.Device != "mps" {
		t.Errorf("device = %q, want %q", snap.Device, "mps")
	}
	if snap.Error != "" {
		t.Errorf("error = %q, want empty", snap.Error)
	}
}

func TestManager_LoadModel_FlipsStateSynchronously(t *testing.T) {
	// REASON: the synchronous state-flip is what prevents the UI's first
	// /status poll from racing in before doLoad and seeing stale Idle.
	// Block send forever to keep the goroutine in StateLoading; we only
	// care that the flip happened before LoadModel returns.
	block := make(chan struct{})
	defer close(block)
	fake := &fakeWorker{
		sendFn: func(ctx context.Context, _ string, _ map[string]any) (map[string]any, error) {
			select {
			case <-block:
			case <-ctx.Done():
			}
			return nil, ctx.Err()
		},
	}
	m := newTestManager(fake)

	if err := m.LoadModel("moshi"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Right after LoadModel returns we must already be off Idle. No sleep,
	// no polling — that's the whole point of the synchronous flip.
	got := m.Status().State
	if got != StateStarting && got != StateLoading {
		t.Fatalf("state = %q, want Starting or Loading immediately after LoadModel", got)
	}
}

func TestManager_LoadModel_RejectsWhenBusy(t *testing.T) {
	block := make(chan struct{})
	defer close(block)
	fake := &fakeWorker{
		sendFn: func(ctx context.Context, _ string, _ map[string]any) (map[string]any, error) {
			<-block
			return map[string]any{"ok": true}, nil
		},
	}
	m := newTestManager(fake)

	if err := m.LoadModel("moshi"); err != nil {
		t.Fatalf("first LoadModel: %v", err)
	}
	// Second call while first is still in flight must error.
	if err := m.LoadModel("personaplex"); err == nil {
		t.Fatalf("second LoadModel: want busy error, got nil")
	}
}

func TestManager_LoadModel_SpawnFailure_ReturnsToIdle(t *testing.T) {
	m := newManagerWithSpawn(func(_ context.Context, _ string) (workers.Handle, error) {
		return nil, errors.New("uv not found")
	})

	if err := m.LoadModel("moshi"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	snap := waitForState(t, m, StateIdle)
	if snap.Error == "" {
		t.Errorf("error empty, want spawn failure surfaced")
	}
}

func TestManager_LoadModel_LoaderFailure_StaysReady(t *testing.T) {
	// REASON: a failed load shouldn't kill the worker — the user can
	// retry with a different model. State must end at Ready, not Idle.
	fake := &fakeWorker{
		sendFn: func(_ context.Context, _ string, _ map[string]any) (map[string]any, error) {
			return nil, errors.New("unknown model")
		},
	}
	m := newTestManager(fake)

	// Use a real catalog name; the fakeWorker simulates the loader
	// rejecting it. Pre-flight catalog check would short-circuit on
	// an unknown name before we got to the worker call.
	if err := m.LoadModel("moshi"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	snap := waitForState(t, m, StateReady)
	if snap.Error == "" {
		t.Errorf("error empty, want loader failure surfaced")
	}
}

func TestManager_LoadModel_ClearsStaleError(t *testing.T) {
	// First load fails, leaving lastError set. Second load must clear it
	// before reporting any new outcome — otherwise the UI keeps showing
	// the old error during the retry.
	fail := true
	fake := &fakeWorker{
		sendFn: func(_ context.Context, _ string, _ map[string]any) (map[string]any, error) {
			if fail {
				return nil, errors.New("first attempt failed")
			}
			return map[string]any{"ok": true, "device": "cpu"}, nil
		},
	}
	m := newTestManager(fake)

	_ = m.LoadModel("moshi")
	waitForState(t, m, StateReady)
	if m.Status().Error == "" {
		t.Fatalf("precondition: lastError should be set after first failure")
	}

	fail = false
	_ = m.LoadModel("moshi")
	snap := waitForState(t, m, StateServing)
	if snap.Error != "" {
		t.Errorf("error = %q after retry success, want cleared", snap.Error)
	}
}

func TestManager_HandleEvent_UpdatesPhase(t *testing.T) {
	// Drive a load, intercept the registered event handler, fire a synthetic
	// phase event, and observe phase + device on Status.
	loaded := make(chan struct{})
	fake := &fakeWorker{
		sendFn: func(ctx context.Context, _ string, _ map[string]any) (map[string]any, error) {
			<-loaded
			return map[string]any{"ok": true}, nil
		},
	}
	m := newTestManager(fake)

	_ = m.LoadModel("moshi")
	// Wait for Manager to register its handler on the fake.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		fake.mu.Lock()
		fn := fake.onEvent
		fake.mu.Unlock()
		if fn != nil {
			fn(map[string]any{"event": "model_phase", "phase": "downloading_lm", "device": "mps"})
			break
		}
		time.Sleep(time.Millisecond)
	}

	snap := m.Status()
	if snap.Phase != "downloading_lm" {
		t.Errorf("phase = %q, want %q", snap.Phase, "downloading_lm")
	}
	if snap.Device != "mps" {
		t.Errorf("device = %q, want %q (set by phase event)", snap.Device, "mps")
	}

	// Let the load finish to terminal state so the goroutine doesn't leak
	// past the test boundary.
	close(loaded)
	waitForState(t, m, StateServing)
	// Phase must clear once load_model returns regardless of outcome.
	if got := m.Status().Phase; got != "" {
		t.Errorf("phase = %q after load complete, want empty", got)
	}
}

func TestManager_Shutdown_ResetsState(t *testing.T) {
	fake := &fakeWorker{
		sendFn: func(_ context.Context, _ string, _ map[string]any) (map[string]any, error) {
			return map[string]any{"ok": true, "device": "cpu"}, nil
		},
	}
	m := newTestManager(fake)

	_ = m.LoadModel("moshi")
	waitForState(t, m, StateServing)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	m.Shutdown(ctx)

	snap := m.Status()
	if snap.State != StateIdle {
		t.Errorf("state = %q, want %q", snap.State, StateIdle)
	}
	if snap.Model != "" || snap.Device != "" || snap.Phase != "" {
		t.Errorf("residual fields after shutdown: %+v", snap)
	}
}
