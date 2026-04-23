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
	"os"
	"path/filepath"
	"sync"
)

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
}

// NewManager builds a Manager in the idle state. Starting the worker is
// deferred until the UI asks for a model — spawning Python at server boot
// would add seconds of latency before the UI could even connect.
func NewManager() *Manager {
	return &Manager{state: StateIdle}
}

// LoadModel lazily boots the worker (if idle), then asks it to load `name`.
// Errors from spawn, handshake, and the worker-side loader all bubble up
// verbatim so the UI can display a meaningful message instead of a generic
// "something went wrong".
func (m *Manager) LoadModel(ctx context.Context, name string) error {
	// STEP 1: take the lock, check we're in a state that accepts a new
	// load, and optimistically advance. Releasing before the slow spawn
	// means a second LoadModel arriving during boot gets a clean "busy"
	// rather than silently queuing.
	m.mu.Lock()
	if m.state == StateStarting || m.state == StateLoading {
		m.mu.Unlock()
		return errors.New("busy: another load in progress")
	}

	if m.worker == nil {
		m.state = StateStarting
		m.mu.Unlock()

		// STEP 2: spawn outside the lock. This can take several seconds
		// on first uv run (resolving + installing deps into the venv).
		w, err := spawnWorker(ctx, workerDir())

		m.mu.Lock()
		if err != nil {
			m.state = StateIdle
			m.mu.Unlock()
			return err
		}
		m.worker = w
	}

	m.state = StateLoading
	w := m.worker
	m.mu.Unlock()

	// STEP 3: ask the worker to load the model. The "name" field matches
	// the worker's ipc_commands.load_model handler.
	_, err := w.send(ctx, "load_model", map[string]any{"name": name})

	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		// WHY: on loader failure we stay in Ready, not Idle — the worker
		// is still up and happy to try a different model. Killing it on
		// every failed load would make recovery needlessly expensive.
		m.state = StateReady
		return err
	}
	m.model = name
	m.state = StateServing
	return nil
}

// Status returns the current lifecycle state. Cheap and lock-protected so
// the HTTP /status endpoint can poll it freely.
func (m *Manager) Status() State {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state
}

// Model returns the name of the currently loaded model, or "" if none.
func (m *Manager) Model() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.model
}

// Shutdown terminates the worker subprocess if one is running. Safe to call
// even when idle. Honors ctx deadline for the kill timeout.
func (m *Manager) Shutdown(ctx context.Context) {
	m.mu.Lock()
	w := m.worker
	m.worker = nil
	m.model = ""
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
