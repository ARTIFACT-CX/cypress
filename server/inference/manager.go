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
	"fmt"
	"log"
	"sync"
	"time"
)

// loadTimeout caps how long a single model load can run. First-run HF
// downloads of the LM weights are several GB on a residential link, so
// budget generously — but bound it so a wedged loader can't pin the
// worker forever and silently swallow further requests.
const loadTimeout = 15 * time.Minute

// downloadTimeout caps a single download. Same rationale as loadTimeout
// but a bit more generous since download alone (without subsequent
// load) is the whole budget here.
const downloadTimeout = 20 * time.Minute

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

// workerHandle is the part of the worker subprocess Manager actually uses.
// Declared as an interface so unit tests can swap in a fake without
// spawning a real Python process. The concrete *worker satisfies it.
//
// SWAP: this is the seam between Manager (business logic) and the
// outbound subprocess adapter. A future remote-inference backend
// implements the same interface against gRPC instead of stdin/stdout.
type workerHandle interface {
	send(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error)
	stop(ctx context.Context) error
	setOnEvent(fn func(map[string]any))
}

// spawnFn is the constructor for a workerHandle. NewManager defaults this
// to a closure over the configured workerDir; tests inject a fake. The
// family argument selects which per-family Python venv
// (worker/models/<family>/.venv) the subprocess is launched from.
type spawnFn func(ctx context.Context, family string) (workerHandle, error)

// Manager is the Go-side handle to the Python inference worker.
type Manager struct {
	// SAFETY: mu guards all mutable state. Exported methods must take mu
	// (or delegate). We *do* release mu across the spawn call since that
	// blocks on uv cold start — other callers during that window get a
	// "busy" error from the state check.
	mu     sync.Mutex
	state  State
	worker workerHandle
	// workerFamily is the model family the running worker subprocess was
	// spawned for (matches catalogEntry.Family). Empty when no worker is
	// up. Used to refuse load/download requests that would require
	// swapping the Python venv mid-session — the host shuts the worker
	// down first instead.
	workerFamily string
	model        string
	device       string // populated on successful load; "mps" / "cuda" / "cpu"
	phase        string // current loader phase ("downloading_mimi", etc.), cleared on ready
	// lastError stores the most recent load failure so the UI can surface it
	// even when it polled in after the failing HTTP request already returned.
	// Cleared when a new load starts so stale errors don't linger.
	lastError string

	// spawn is the worker constructor. Defaults to the real subprocess
	// spawner; tests override via newManagerWithSpawn.
	spawn spawnFn

	// activeStream is the at-most-one streaming session against this
	// worker. nil when idle. Guarded by mu — see stream.go for the
	// lifecycle hooks (StartStream / detachStream).
	activeStream *Stream

	// manifest is the on-disk record of installed models. Owned by the
	// manager so download completion can update it under mu without an
	// extra lock. nil-safe: a nil manifest just means we can't persist
	// installs (used in unit tests that don't care about disk state).
	manifest *Manifest

	// inflightDownloads tracks live download progress per model name,
	// keyed for quick lookup when the worker emits download_progress
	// events. Cleared on download_done / download_error / cancellation.
	inflightDownloads map[string]*DownloadProgress

	// familySetupMu serializes per-family `uv sync` runs so two
	// concurrent downloads against the same family don't race on the
	// venv directory. Lazily populated; lookup is guarded by mu but
	// the Lock/Unlock on the inner mutex happens without mu held.
	familySetupMu map[string]*sync.Mutex

	// syncFamily creates / refreshes the Python venv for a family.
	// Defaults to running `uv sync`; tests inject a fake to avoid
	// shelling out. SWAP seam for a future packaged build that ships
	// a pre-baked venv and wants this to be a no-op.
	syncFamily func(ctx context.Context, family string) error

	// workerDir is the resolved path to the worker/ scaffold root. Used
	// to locate per-family venvs (workerDir/models/<family>/.venv) and
	// for the spawned process's cwd. Injected via Config so tests can
	// point at a tmpdir and the composition root can resolve a packaged-
	// app path via os.Executable() without this package knowing how.
	workerDir string
}

// Config is the composition-root contract for building a Manager.
// Both fields are required in production; tests construct Managers via
// the package-internal helpers below and don't go through Config.
type Config struct {
	// WorkerDir points at the worker/ scaffold (Python entrypoint +
	// models/<family>/.venv subtrees). main.go resolves this from
	// os.Executable() so a packaged app finds the bundled worker tree.
	WorkerDir string
	// DataDir points at Cypress's metadata root (manifest, future
	// settings). main.go resolves this to ~/.cypress in production.
	DataDir string
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
//
// The manifest is opened eagerly because it's tiny (a JSON file) and
// the catalog endpoint needs it the moment the UI opens. A failure to
// open the manifest is logged but non-fatal — we degrade to a memory-
// only manifest so the rest of the server still runs.
func NewManager(cfg Config) *Manager {
	mf, err := NewManifest(cfg.DataDir)
	if err != nil {
		log.Printf("manifest open failed (continuing without persistence): %v", err)
		mf = nil
	}
	workerDir := cfg.WorkerDir
	return &Manager{
		state:     StateIdle,
		workerDir: workerDir,
		// REASON: close over workerDir so the spawn/syncFamily defaults
		// remain context.Context-only at the call sites — Manager
		// internals don't need to thread the path through every call.
		spawn: func(ctx context.Context, family string) (workerHandle, error) {
			return spawnWorker(ctx, workerDir, family)
		},
		syncFamily: func(ctx context.Context, family string) error {
			return defaultSyncFamily(ctx, workerDir, family)
		},
		manifest:          mf,
		inflightDownloads: map[string]*DownloadProgress{},
		familySetupMu:     map[string]*sync.Mutex{},
	}
}

// newManagerWithSpawn is the test constructor. Lets a unit test inject a
// fake spawner without going through subprocess machinery. syncFamily
// defaults to a no-op so tests don't need a real uv on PATH. Tests that
// exercise venv lifecycle paths set m.workerDir directly afterwards.
func newManagerWithSpawn(spawn spawnFn) *Manager {
	return &Manager{
		state:             StateIdle,
		spawn:             spawn,
		syncFamily:        func(_ context.Context, _ string) error { return nil },
		inflightDownloads: map[string]*DownloadProgress{},
		familySetupMu:     map[string]*sync.Mutex{},
	}
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
	// STEP 0: pre-flight the catalog so we know which family to spawn
	// (and which venv that selects). Unknown / unavailable models fail
	// here rather than after we've spawned a worker for nothing.
	entry := catalogEntryByName(name)
	if entry == nil {
		return fmt.Errorf("unknown model %q", name)
	}
	if entry.Family == "" {
		return fmt.Errorf("model %q has no family configured", name)
	}

	// STEP 1: synchronously advance state under the lock so a /status
	// call observed any time after this returns sees "starting" or
	// "loading", never a stale "idle"/"serving".
	m.mu.Lock()
	if m.state == StateStarting || m.state == StateLoading {
		m.mu.Unlock()
		return errors.New("busy: another load in progress")
	}
	// REASON: the worker subprocess imports a single family's Python
	// stack; PersonaPlex's forked `moshi` and kyutai's `moshi` cannot
	// coexist in one process. Refuse cross-family loads while a worker
	// is up and force the user to unload first — Shutdown drops the
	// subprocess so the next load spawns the right venv.
	if m.worker != nil && m.workerFamily != "" && m.workerFamily != entry.Family {
		m.mu.Unlock()
		return fmt.Errorf("worker is running %q models; unload before loading %q (family %q)",
			m.workerFamily, name, entry.Family)
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
		m.doLoad(ctx, name, entry.Family)
	}()
	return nil
}

// doLoad is the actually-blocking half of LoadModel. Runs on a goroutine
// owned by the manager; never call directly. All exit paths must leave
// the manager in a sensible terminal state (Idle / Ready / Serving) so
// the UI's /status poll converges.
func (m *Manager) doLoad(ctx context.Context, name string, family string) {
	// STEP A: spawn the worker if we don't have one. uv cold start can
	// take several seconds on first run; we hold no lock during it so
	// /status keeps responding.
	m.mu.Lock()
	needSpawn := m.worker == nil
	m.mu.Unlock()

	if needSpawn {
		// STEP A0: ensure the family venv exists. Normally the user
		// downloaded the model first (which sync'd the venv), so this
		// is a fast os.Stat. The slow path is rare — manual venv
		// removal or a fresh checkout where weights survived but
		// .venv didn't.
		m.mu.Lock()
		m.phase = "preparing_env"
		m.mu.Unlock()
		if err := m.ensureFamilyEnv(ctx, family); err != nil {
			m.mu.Lock()
			m.state = StateIdle
			m.phase = ""
			m.lastError = err.Error()
			m.mu.Unlock()
			log.Printf("ensure family env failed: %v", err)
			return
		}

		w, err := m.spawn(ctx, family)
		m.mu.Lock()
		if err != nil {
			m.state = StateIdle
			m.phase = ""
			m.lastError = err.Error()
			m.mu.Unlock()
			log.Printf("worker spawn failed: %v", err)
			return
		}
		w.setOnEvent(m.handleEvent)
		m.worker = w
		m.workerFamily = family
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
		// REASON: on loader failure stay in Ready, not Idle — the worker is
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
	case "audio_out", "stream_error":
		// Stream events go to the active session's channel. Routing
		// lives in stream.go to keep base64 / chunk-shape concerns out
		// of the lifecycle state machine.
		m.dispatchStreamEvent(msg)
	case "download_progress", "download_done", "download_error":
		// Download lifecycle events update the inflightDownloads map
		// and (on completion) write the manifest entry. Routing lives
		// in downloads.go to keep that state machine separate from
		// the load state machine.
		m.handleDownloadEvent(event, msg)
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
	m.workerFamily = ""
	m.model = ""
	m.device = ""
	m.phase = ""
	m.state = StateIdle
	m.mu.Unlock()

	if w != nil {
		_ = w.stop(ctx)
	}
}

