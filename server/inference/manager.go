// AREA: inference · SUBPROCESS
// Owns the Python inference worker lifecycle. Responsible for launching,
// health-checking, and tearing down the Python process that loads models and
// does the actual generation. The Go side never imports PyTorch — this is the
// only bridge.
//
// SWAP: the worker backend. Today this shells out to a Python subprocess; a
// future implementation could target a remote gRPC inference server (cloud
// tier). Consumers should depend on the Manager interface, not on subprocess
// details.

package inference

import (
	"context"
	"errors"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// loadTimeout caps how long a single model load can run. First-run HF
// downloads of the LM weights are several GB on a residential link, so
// budget generously — but bound it so a wedged loader can't pin the
// worker forever and silently swallow further requests.
const loadTimeout = 15 * time.Minute

// State is the public lifecycle of the inference subsystem. The UI polls
// this to render "loading…", enable/disable buttons, etc.
type State string

const (
	StateIdle     State = "idle"     // no worker process, no model loaded
	StateStarting State = "starting" // subprocess launched, waiting for handshake
	StateReady    State = "ready"    // worker alive, no model loaded
	StateLoading  State = "loading"  // worker alive, loading a model
	StateServing  State = "serving"  // worker alive, model loaded
)

// Manager is the Go-side handle to the Python inference worker.
type Manager struct {
	// SAFETY: mu guards all mutable state. Exported methods must take mu
	// (or delegate). We *do* release mu across the spawn call since that
	// blocks on uv cold start — other callers during that window get a
	// "busy" error from the state check.
	mu     sync.Mutex
	state  State
	worker *worker
	model  string
	device string // populated on successful load; "mps" / "cuda" / "cpu"
	phase  string // current loader phase ("downloading_mimi", etc.), cleared on ready
	// lastError stores the most recent load failure so the UI can surface it
	// even when it polled in after the failing HTTP request already returned.
	// Cleared when a new load starts so stale errors don't linger.
	lastError string
}

// Snapshot is the Manager's external view — what the HTTP /status endpoint
// returns. Grouped here so callers don't accidentally read half-updated
// fields without the lock.
type Snapshot struct {
	State  State  `json:"state"`
	Model  string `json:"model"`
	Device string `json:"device"`
	Phase  string `json:"phase"`
	// Error is the last load failure, if any. Present so the UI can show
	// the error even if its triggering HTTP request already timed out.
	Error string `json:"error,omitempty"`
}

// NewManager builds a Manager in the idle state. Starting the worker is
// deferred until the UI asks for a model — spawning Python at server boot
// would add seconds of latency before the UI could even connect.
func NewManager() *Manager {
	return &Manager{state: StateIdle}
}

// LoadModel kicks off a model load. Returns synchronously as soon as the
// manager has advanced to a non-idle state — the long-running work (uv
// cold start, HF download, weight transfer) runs on a background
// goroutine and posts results back via state mutations. This matters for
// the UI: by the time our HTTP handler returns 202, /status will already
// report state=starting/loading instead of the stale previous state. If
// the flip were inside the goroutine, the UI's first poll could race in
// before the goroutine ran and see the *previous* state, prematurely
// concluding the load was done.
//
// Errors returned here are pre-flight only (busy). Errors from spawn,
// handshake, or the worker-side loader land in m.lastError and surface
// via /status.
func (m *Manager) LoadModel(name string) error {
	// STEP 1: synchronously advance state under the lock so a /status
	// call observed any time after this returns sees "starting" or
	// "loading", never a stale "idle"/"serving".
	m.mu.Lock()
	if m.state == StateStarting || m.state == StateLoading {
		m.mu.Unlock()
		return errors.New("busy: another load in progress")
	}
	// Clear any stale error from a previous failed attempt so the UI
	// doesn't continue to show it once the user retries.
	m.lastError = ""
	if m.worker == nil {
		m.state = StateStarting
	} else {
		m.state = StateLoading
	}
	m.mu.Unlock()

	// STEP 2: hand off the slow work with our own internal context. We
	// don't reuse the HTTP request's context because the request returns
	// in milliseconds; doLoad needs to outlive it for the full download.
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), loadTimeout)
		defer cancel()
		m.doLoad(ctx, name)
	}()
	return nil
}

// doLoad is the actually-blocking half of LoadModel. Runs on a goroutine
// owned by the manager; never call directly. All exit paths must leave
// the manager in a sensible terminal state (Idle / Ready / Serving) so
// the UI's /status poll converges.
func (m *Manager) doLoad(ctx context.Context, name string) {
	// STEP A: spawn the worker if we don't have one. uv cold start can
	// take several seconds on first run; we hold no lock during it so
	// /status keeps responding.
	m.mu.Lock()
	needSpawn := m.worker == nil
	m.mu.Unlock()

	if needSpawn {
		w, err := spawnWorker(ctx, workerDir())
		m.mu.Lock()
		if err != nil {
			m.state = StateIdle
			m.lastError = err.Error()
			m.mu.Unlock()
			log.Printf("worker spawn failed: %v", err)
			return
		}
		w.onEvent = m.handleEvent
		m.worker = w
		m.state = StateLoading
		m.mu.Unlock()
	}

	// STEP B: ask the worker to load the model. The "name" field matches
	// the worker's ipc_commands.load_model handler.
	m.mu.Lock()
	w := m.worker
	m.mu.Unlock()

	reply, err := w.send(ctx, "load_model", map[string]any{"name": name})

	m.mu.Lock()
	defer m.mu.Unlock()
	// Phase is only meaningful while load_model is in flight; clear it
	// regardless of outcome so the UI doesn't show "downloading…" forever.
	m.phase = ""

	if err != nil {
		// WHY: on loader failure stay in Ready, not Idle — the worker is
		// still up and willing to try a different model. Killing it on
		// every failed load would make recovery needlessly expensive.
		m.state = StateReady
		m.lastError = err.Error()
		log.Printf("load_model %q failed: %v", name, err)
		return
	}
	m.model = name
	if dev, ok := reply["device"].(string); ok {
		m.device = dev
	}
	m.state = StateServing
}

// handleEvent is the worker.onEvent callback. Fires for every unsolicited
// worker→host message; today that's just phase updates from model loaders.
// Keep this fast — it runs on the worker's read-loop goroutine.
func (m *Manager) handleEvent(msg map[string]any) {
	event, _ := msg["event"].(string)
	switch event {
	case "model_phase":
		phase, _ := msg["phase"].(string)
		m.mu.Lock()
		m.phase = phase
		// The "resolving" phase includes the device up front so the UI
		// can start showing "Loading on MPS…" before any weights have
		// actually landed there.
		if dev, ok := msg["device"].(string); ok {
			m.device = dev
		}
		m.mu.Unlock()
	}
}

// Status returns a point-in-time snapshot of the inference subsystem.
// Grouped into one struct (rather than separate accessors per field) so
// callers can't interleave reads and see inconsistent combinations.
func (m *Manager) Status() Snapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	return Snapshot{
		State:  m.state,
		Model:  m.model,
		Device: m.device,
		Phase:  m.phase,
		Error:  m.lastError,
	}
}

// Shutdown terminates the worker subprocess if one is running. Safe to call
// even when idle. Honors ctx deadline for the kill timeout.
func (m *Manager) Shutdown(ctx context.Context) {
	m.mu.Lock()
	w := m.worker
	m.worker = nil
	m.model = ""
	m.device = ""
	m.phase = ""
	m.state = StateIdle
	m.mu.Unlock()

	if w != nil {
		_ = w.stop(ctx)
	}
}

// workerDir resolves the worker/ directory. In dev the server runs with
// cwd = server/, so the worker is a sibling. CYPRESS_WORKER_DIR overrides
// this for tests and packaged builds where the layout differs.
func workerDir() string {
	if env := os.Getenv("CYPRESS_WORKER_DIR"); env != "" {
		return env
	}
	abs, _ := filepath.Abs("../worker")
	return abs
}
