// AREA: inference · TESTS · REACHABILITY
// Manager's remote-reachability tracking. The Go HTTP server "running"
// state means the laptop process is up; it doesn't mean the configured
// remote worker is reachable. Three things to cover:
//   - Status payload includes transport + remote block when remote.
//   - Reachability flips between probe / dial sites.
//   - Periodic re-probe brings the flag back true after the worker
//     comes online without any user action.

package inference

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/workers"
)

func TestStatus_LocalMode_NoRemoteBlock(t *testing.T) {
	m := newTestManager(&fakeWorker{})
	snap := m.Status()
	if snap.Transport != "local" {
		t.Errorf("transport = %q, want local", snap.Transport)
	}
	if snap.Remote != nil {
		t.Errorf("Remote = %+v, want nil in local mode", snap.Remote)
	}
}

func TestStatus_RemoteMode_PopulatesRemoteBlock(t *testing.T) {
	m := newTestManager(&fakeWorker{})
	m.remote = &workers.RemoteEndpoint{
		URL:   "grpcs://example.test:7843",
		Token: "t",
	}
	// Set the reachability state directly — we're testing the Status
	// projection, not the dial path.
	m.markRemoteReachable()

	snap := m.Status()
	if snap.Transport != "remote" {
		t.Errorf("transport = %q, want remote", snap.Transport)
	}
	if snap.Remote == nil {
		t.Fatal("Remote is nil; want populated block")
	}
	if snap.Remote.URL != "grpcs://example.test:7843" {
		t.Errorf("URL = %q", snap.Remote.URL)
	}
	if !snap.Remote.Reachable {
		t.Errorf("Reachable = false, want true")
	}
	if snap.Remote.LastError != "" {
		t.Errorf("LastError = %q, want empty after reachable", snap.Remote.LastError)
	}
	if snap.Remote.LastChecked.IsZero() {
		t.Errorf("LastChecked is zero")
	}
}

func TestReachable_FailedDial_MarksUnreachableWithError(t *testing.T) {
	dialErr := errors.New("connection refused")
	m := newTestManager(&fakeWorker{})
	m.remote = &workers.RemoteEndpoint{URL: "grpcs://example.test:7843", Token: "t"}

	m.markRemoteUnreachable(dialErr)

	snap := m.Status()
	if snap.Remote == nil {
		t.Fatal("expected remote block")
	}
	if snap.Remote.Reachable {
		t.Errorf("Reachable = true, want false")
	}
	if snap.Remote.LastError != "connection refused" {
		t.Errorf("LastError = %q", snap.Remote.LastError)
	}
}

func TestReachable_TransitionsViaProbeSiteAlongsidePlatform(t *testing.T) {
	// Both probeRemotePlatform success and failure should drive the
	// reachability flag — without that, /status lies during the gap
	// between the eager probe and the periodic re-probe ticker.
	probeErr := errors.New("dial refused")
	spawn := func(_ context.Context, _ string) (workers.Handle, error) {
		return nil, probeErr
	}
	m := newManagerWithSpawn(spawn)
	m.remote = &workers.RemoteEndpoint{URL: "grpcs://example.test:7843", Token: "t"}
	m.platformReady = make(chan struct{})
	m.platformReadyOnce = onceReset()

	m.probeRemotePlatform()

	snap := m.Status()
	if snap.Remote.Reachable {
		t.Fatal("Reachable should be false after failed probe")
	}
	if snap.Remote.LastError == "" {
		t.Fatal("LastError should carry the dial failure")
	}
}

func TestRemoteHealthLoop_RecoversWhenWorkerComesOnline(t *testing.T) {
	// REASON: the whole point of the periodic re-probe — user starts
	// the app, sees the banner, fixes their tunnel, and the banner
	// clears without restarting. Models the "coming back online" by
	// flipping the spawn closure mid-test.
	withFastReachability(t)

	var attempt atomic.Int32
	good := newPlatformFake(workers.Platform{
		OS: "linux", Arch: "amd64",
		AvailableBackends: []string{"torch"},
	})
	spawn := func(_ context.Context, _ string) (workers.Handle, error) {
		// First N attempts fail (worker not up yet). After that, succeed.
		if attempt.Add(1) < 3 {
			return nil, errors.New("dial refused")
		}
		return good, nil
	}
	m := newManagerWithSpawn(spawn)
	m.remote = &workers.RemoteEndpoint{URL: "grpcs://example.test:7843", Token: "t"}
	m.platformReady = make(chan struct{})
	m.platformReadyOnce = onceReset()

	// Eager probe fires once and fails.
	m.probeRemotePlatform()
	if snap := m.Status(); snap.Remote.Reachable {
		t.Fatal("expected unreachable after failed eager probe")
	}

	// Start the loop and wait for it to recover.
	go m.remoteHealthLoop()
	t.Cleanup(func() {
		if m.shutdownCancel != nil {
			m.shutdownCancel()
		}
	})

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if m.Status().Remote.Reachable {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	snap := m.Status()
	if !snap.Remote.Reachable {
		t.Fatalf("loop never recovered; attempts=%d, last status=%+v",
			attempt.Load(), snap.Remote)
	}
	if snap.Remote.LastError != "" {
		t.Errorf("LastError = %q after recovery, want empty", snap.Remote.LastError)
	}
}

func TestRemoteHealthLoop_StopsProbingWhileReachable(t *testing.T) {
	// SAFETY: the loop should be a no-op when reachable=true. Otherwise
	// every tick burns a TLS handshake against the worker for no
	// information gain.
	withFastReachability(t)

	var attempt atomic.Int32
	spawn := func(_ context.Context, _ string) (workers.Handle, error) {
		attempt.Add(1)
		return newPlatformFake(workers.Platform{OS: "linux", Arch: "amd64"}), nil
	}
	m := newManagerWithSpawn(spawn)
	m.remote = &workers.RemoteEndpoint{URL: "grpcs://example.test:7843", Token: "t"}
	m.platformReady = make(chan struct{})
	m.platformReadyOnce = onceReset()
	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())
	m.shutdownCtx = shutdownCtx
	m.shutdownCancel = shutdownCancel
	t.Cleanup(shutdownCancel)

	m.probeRemotePlatform() // succeeds, flips reachable=true. attempt = 1.
	go m.remoteHealthLoop()

	// Give the loop several ticks. With reachable=true throughout it
	// must never spawn again.
	time.Sleep(50 * time.Millisecond)
	if got := attempt.Load(); got != 1 {
		t.Errorf("spawn attempts = %d, want 1 (loop must skip while reachable)", got)
	}
}

// withFastReachability shrinks the periodic re-probe interval so tests
// don't sleep on a 30s ticker. Restored by t.Cleanup.
func withFastReachability(t *testing.T) {
	t.Helper()
	orig := remoteHealthInterval
	remoteHealthInterval = 5 * time.Millisecond
	t.Cleanup(func() { remoteHealthInterval = orig })
}
