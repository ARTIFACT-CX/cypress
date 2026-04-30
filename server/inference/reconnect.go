// AREA: inference · RECONNECT
// Auto-recovery for worker transport drops. The Manager spawns one
// watchWorker goroutine per live worker handle; it parks on the
// handle's Done channel and runs the recovery path when the transport
// dies. Local subprocess crashes surface as StateIdle (something's
// structurally wrong, don't try to paper over it). Remote drops are
// retried with bounded exponential backoff because transient blips
// are routine on residential / pod networks.
//
// Active streams are torn down on every drop. The model's stream
// session is server-side state that can't be resumed across a fresh
// connection, so we don't pretend resume worked when it didn't —
// the user reloads and reopens the session.

package inference

import (
	"context"
	"errors"
	"log"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/workers"
)

// reconnectInitialBackoff / reconnectMaxBackoff bound the redial
// cadence after a remote drop. Starts tight (so a one-frame blip
// recovers near-instantly) and caps so we don't burn CPU on a long
// outage. Total budget is reconnectTotalBudget — past that we give
// up and surface to the user.
// REASON: vars rather than consts so reconnect_test.go can shrink
// them for fast-running unit tests. Production callers don't mutate.
var (
	reconnectInitialBackoff = 250 * time.Millisecond
	reconnectMaxBackoff     = 4 * time.Second
	reconnectTotalBudget    = 30 * time.Second
	// reconnectDialTimeout is the per-attempt dial deadline. Capped
	// generously because handshakeTimeout in workers/ already gates
	// individual handshakes; this is a defense-in-depth bound so
	// hung dial state can't pin the watcher past total budget.
	reconnectDialTimeout = 10 * time.Second
)

// watchWorker parks on the handle's Done channel and runs the recovery
// path when it closes. One per spawned worker; exits cleanly when the
// handle is replaced (Stop, or a previous reconnect already happened).
//
// Concurrency: takes m.mu only for short snapshot/mutation windows; the
// network redial happens lock-free so /status keeps responding while we
// retry. Active-stream teardown also happens outside the lock so we
// don't deadlock against any caller currently in workerSend.
func (m *Manager) watchWorker(w workers.Handle, family string) {
	<-w.Done()

	// SETUP: snapshot under lock and decide whether this drop concerns
	// us. If m.worker has already been replaced (Stop, or a parallel
	// recovery completed), the close we observed is stale — bail.
	m.mu.Lock()
	if m.worker != w {
		m.mu.Unlock()
		return
	}
	priorState := m.state
	activeStream := m.activeStream
	// Clear worker-shaped state so callers see "no worker" while we
	// recover. For local crashes we'll flip to Idle below; for remote
	// drops we use StateStarting during the redial so /status shows a
	// transient "reconnecting" rather than bouncing through Idle.
	m.worker = nil
	m.workerFamily = ""
	m.activeStream = nil
	m.model = ""
	m.device = ""
	m.phase = ""
	if m.remote != nil {
		m.state = StateStarting
	} else {
		m.state = StateIdle
	}
	m.mu.Unlock()

	// STEP 1: tear down the user-visible stream if one was active.
	// activeStream.Close would try to talk to the now-dead worker via
	// workerSend (and its workerSend takes m.mu), so we close the
	// outputs directly here. detachStream is a no-op since we already
	// cleared activeStream; consumers see the channel close and exit.
	if activeStream != nil {
		failStream(activeStream)
	}

	// STEP 2: branch on transport. Local subprocess crash means
	// something's wrong with the deployment (Python died, OOM,
	// missing dep) — reconnecting wouldn't help, surface and stop.
	if m.remote == nil {
		m.mu.Lock()
		// state is already Idle from the snapshot block above.
		m.lastError = "worker subprocess exited unexpectedly"
		m.mu.Unlock()
		log.Printf("worker: subprocess exited unexpectedly (family=%s)", family)
		return
	}

	// STEP 3: remote drop — retry with backoff. priorState shapes the
	// post-reconnect message: a serving session means the user lost
	// audio context; a ready session means they didn't notice.
	// REASON: a transport drop means the remote is no longer reachable
	// for the duration of the redial budget. Mark unreachable now so
	// /status reflects the disconnect during the backoff window —
	// without this the UI keeps showing "reachable" for up to 30s
	// while we silently retry. markRemoteReachable below flips it
	// back if redial succeeds.
	if m.remote != nil {
		m.markRemoteUnreachable(errors.New("transport dropped; reconnecting"))
	}
	log.Printf("remote worker disconnected (family=%s); attempting reconnect", family)
	fresh, err := m.redialWithBackoff(family)
	if err != nil {
		m.mu.Lock()
		m.state = StateIdle
		m.lastError = "remote worker disconnected: " + err.Error()
		m.mu.Unlock()
		if m.remote != nil {
			m.markRemoteUnreachable(err)
		}
		log.Printf("remote worker reconnect failed: %v", err)
		return
	}
	if m.remote != nil {
		m.markRemoteReachable()
	}

	// STEP 4: install the fresh handle, but first re-check that we
	// still own the recovery slot. SAFETY: redialWithBackoff ran
	// lock-free; a concurrent Shutdown could have cleared state and
	// expects authority. If we're no longer in StateStarting, someone
	// else (Shutdown, a parallel recovery on an already-replaced
	// handle) intervened — drop the freshly dialed worker so we
	// don't resurrect after Shutdown.
	fresh.SetOnEvent(m.handleEvent)
	m.mu.Lock()
	if m.state != StateStarting || m.worker != nil {
		m.mu.Unlock()
		log.Printf("worker reconnect raced with state change; discarding fresh handle")
		stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = fresh.Stop(stopCtx)
		cancel()
		return
	}
	m.worker = fresh
	m.workerFamily = family
	m.state = StateReady
	if priorState == StateServing || priorState == StateLoading {
		m.lastError = "remote worker reconnected; reload model to resume"
	}
	// REASON: a reconnect is a fresh handshake, so the worker may have
	// picked up new cache entries (or lost some) since we last looked.
	// Refresh the platform/downloaded snapshot to keep variant selection
	// honest.
	m.recordHandshakePlatformLocked(fresh.Platform())
	m.mu.Unlock()

	// Install a watcher on the new handle so the next drop gets the
	// same treatment. Separate goroutine — tail-recursive in spirit.
	go m.watchWorker(fresh, family)
}

// redialWithBackoff retries m.spawn until one succeeds or the total
// budget elapses. Backoff is capped at reconnectMaxBackoff so the
// worst case is steady polling, not exponential blowup.
func (m *Manager) redialWithBackoff(family string) (workers.Handle, error) {
	backoff := reconnectInitialBackoff
	deadline := time.Now().Add(reconnectTotalBudget)
	var lastErr error
	for {
		ctx, cancel := context.WithTimeout(context.Background(), reconnectDialTimeout)
		w, err := m.spawn(ctx, family)
		cancel()
		if err == nil {
			return w, nil
		}
		lastErr = err
		log.Printf("worker reconnect attempt failed: %v (next retry in %s)", err, backoff)

		if time.Now().Add(backoff).After(deadline) {
			break
		}
		time.Sleep(backoff)
		backoff *= 2
		if backoff > reconnectMaxBackoff {
			backoff = reconnectMaxBackoff
		}
	}
	if lastErr == nil {
		lastErr = errors.New("redial budget exhausted")
	}
	return nil, lastErr
}

// failStream tears down a stream whose worker has died. We bypass
// Stream.Close because Close would try to send stop_stream to a
// vanished worker (taking m.mu in the process); here we only need
// to mark it closed and unblock any range-over-Outputs consumer.
func failStream(s *Stream) {
	s.closeMu.Lock()
	defer s.closeMu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	close(s.outputs)
}
