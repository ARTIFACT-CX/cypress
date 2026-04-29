// AREA: inference · MANAGER
// Owns the Python inference worker lifecycle and the load state machine.
// Coordinates the workers/ adapter (subprocess + gRPC), the downloads/
// service (HF pulls), and the models/ manifest. Anything model-flavored
// that crosses those packages comes back through here so the load
// state and download state stay coherent.
//
// SWAP: the worker backend lives behind workers.Handle. Today the spawn
// closure shells out to a Python subprocess via workers.SpawnLocal; a
// future remote-inference backend slots in by injecting a different
// spawn closure that dials a gRPC endpoint.

package inference

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/downloads"
	"github.com/ARTIFACT-CX/cypress/server/models"
	"github.com/ARTIFACT-CX/cypress/server/workers"
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

// SpawnFn is the constructor for a workers.Handle. NewManager defaults
// this to a closure over workers.SpawnLocal; tests inject a fake. The
// family argument selects which per-family Python venv
// (worker/models/<family>/.venv) the subprocess is launched from.
type SpawnFn func(ctx context.Context, family string) (workers.Handle, error)

// Manager is the Go-side handle to the Python inference worker.
type Manager struct {
	// SAFETY: mu guards all mutable state. Exported methods must take mu
	// (or delegate). We *do* release mu across the spawn call since that
	// blocks on uv cold start — other callers during that window get a
	// "busy" error from the state check.
	mu     sync.Mutex
	state  State
	worker workers.Handle
	// workerFamily is the model family the running worker subprocess was
	// spawned for (matches models.Entry.Family). Empty when no worker is
	// up. Used to refuse load/download requests that would require
	// swapping the Python venv mid-session — the host shuts the worker
	// down first instead.
	workerFamily string
	model        string
	device       string // populated on successful load; "mps" / "cuda" / "cpu"
	phase        string // current loader phase ("downloading_lm", etc.), cleared on ready
	// lastError stores the most recent load failure so the UI can surface it
	// even when it polled in after the failing HTTP request already returned.
	// Cleared when a new load starts so stale errors don't linger.
	lastError string

	// spawn is the worker constructor. Defaults to the real subprocess
	// spawner; tests override via newManagerWithSpawn.
	spawn SpawnFn

	// activeStream is the at-most-one streaming session against this
	// worker. nil when idle. Guarded by mu — see stream.go for the
	// lifecycle hooks (StartStream / detachStream).
	activeStream *Stream

	// envSetup owns per-family venv preparation/removal. Composition
	// root injects it; tests use a no-op via newManagerWithSpawn.
	envSetup *workers.EnvSetup

	// manifest is the on-disk record of installed models. Read here for
	// the family-removal policy (any sibling installed?). Owned by the
	// composition root and shared with the downloads service.
	manifest *models.Manifest

	// downloads is the download/cancel/delete service. Manager satisfies
	// downloads.WorkerProvider so the service can spawn/reuse this
	// Manager's worker for download IPC.
	downloads *downloads.Service
}

// Config is the composition-root contract for building a Manager.
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
	mf, err := models.NewManifest(cfg.DataDir)
	if err != nil {
		log.Printf("manifest open failed (continuing without persistence): %v", err)
		mf = nil
	}
	envSetup := workers.NewEnvSetup(cfg.WorkerDir, nil)
	m := &Manager{
		state:    StateIdle,
		envSetup: envSetup,
		manifest: mf,
		// REASON: close over workerDir so the spawn default remains
		// context.Context-only at the call sites — Manager internals
		// don't thread the path through every call.
		spawn: func(ctx context.Context, family string) (workers.Handle, error) {
			return workers.SpawnLocal(ctx, cfg.WorkerDir, family)
		},
	}
	m.downloads = downloads.New(m, envSetup, mf)
	return m
}

// newManagerWithSpawn is the test constructor. Lets a unit test inject
// a fake spawner without going through subprocess machinery. envSetup
// is nil (Service skips Ensure) and manifest is nil unless the test
// supplies one via setManifest.
func newManagerWithSpawn(spawn SpawnFn) *Manager {
	m := &Manager{
		state: StateIdle,
		spawn: spawn,
	}
	m.downloads = downloads.New(m, nil, nil)
	return m
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
	entry := models.EntryByName(name)
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
		if m.envSetup != nil {
			if err := m.envSetup.Ensure(ctx, family); err != nil {
				m.mu.Lock()
				m.state = StateIdle
				m.phase = ""
				m.lastError = err.Error()
				m.mu.Unlock()
				log.Printf("ensure family env failed: %v", err)
				return
			}
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
		w.SetOnEvent(m.handleEvent)
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

	reply, err := w.Send(ctx, "load_model", map[string]any{"name": name})

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
// worker→host message; today that's phase updates, audio frames, and
// download lifecycle events. Keep this fast — it runs on the worker's
// read-loop goroutine.
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
		// lives in stream.go to keep chunk-shape concerns out of the
		// lifecycle state machine.
		m.dispatchStreamEvent(msg)
	case "download_progress", "download_done", "download_error":
		// Download lifecycle is owned by the downloads.Service; route
		// through so its inflight map and manifest writes stay current.
		m.downloads.HandleEvent(event, msg)
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
		_ = w.Stop(ctx)
	}
}

// --- downloads.WorkerProvider --------------------------------------------------

// Worker satisfies downloads.WorkerProvider. Returns the current worker
// and its family or (nil, "") when no worker is up.
func (m *Manager) Worker() (workers.Handle, string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.worker, m.workerFamily
}

// SpawnWorker satisfies downloads.WorkerProvider. Reuses an existing
// worker if one is up for the requested family; otherwise spawns one
// and installs the Manager's event handler. Cross-family requests are
// rejected so a download can't kick out a load mid-session.
func (m *Manager) SpawnWorker(ctx context.Context, family string) (workers.Handle, error) {
	m.mu.Lock()
	if m.worker != nil {
		if m.workerFamily != "" && m.workerFamily != family {
			f := m.workerFamily
			m.mu.Unlock()
			return nil, fmt.Errorf("worker is running %q models; unload before spawning %q", f, family)
		}
		w := m.worker
		m.mu.Unlock()
		return w, nil
	}
	m.mu.Unlock()

	// Spawn outside the lock — uv cold start can take seconds.
	spawned, err := m.spawn(ctx, family)
	if err != nil {
		return nil, err
	}
	spawned.SetOnEvent(m.handleEvent)

	m.mu.Lock()
	defer m.mu.Unlock()
	// Race: another caller may have spawned in parallel. Keep the
	// first one; tear ours down asynchronously so this caller doesn't
	// block on Stop.
	if m.worker != nil {
		go func() {
			stopCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
			defer c()
			_ = spawned.Stop(stopCtx)
		}()
		return m.worker, nil
	}
	m.worker = spawned
	m.workerFamily = family
	if m.state == StateIdle {
		m.state = StateReady
	}
	return spawned, nil
}

// --- download / delete passthroughs --------------------------------------------

// DownloadModel kicks off a download via the downloads service. Thin
// passthrough kept on Manager so the HTTP handler has one entry point
// for model lifecycle commands.
func (m *Manager) DownloadModel(name string) error { return m.downloads.Start(name) }

// CancelDownload aborts the in-flight pull for name.
func (m *Manager) CancelDownload(name string) error { return m.downloads.Cancel(name) }

// ModelInfos is the catalog merged with inflight progress. Single
// entry point for the /models route.
func (m *Manager) ModelInfos() []models.ModelInfo { return m.downloads.ModelInfos() }

// DeleteModel removes an installed model from disk and the manifest.
// Refuses if the model is currently loaded or downloading. After a
// successful delete, drops the per-family venv if no sibling install
// or download depends on it (and the worker isn't busy with it).
func (m *Manager) DeleteModel(name string) error {
	m.mu.Lock()
	if m.state == StateServing && m.model == name {
		m.mu.Unlock()
		return errors.New("model is currently loaded; unload first")
	}
	m.mu.Unlock()
	if m.downloads.IsInflight(name) {
		return errors.New("download in progress for this model")
	}

	cat := models.EntryByName(name)
	if err := m.downloads.DeleteFiles(name); err != nil {
		return err
	}
	if cat != nil && cat.Family != "" {
		m.maybeRemoveFamily(cat.Family)
	}
	return nil
}

// maybeRemoveFamily drops the family's .venv if no installed models or
// inflight downloads from that family remain and the worker isn't busy
// loading/serving it. Called from DeleteModel after the manifest entry
// is gone. Conservative — does nothing on the busy path.
//
// REASON: tying venv lifetime to "any model from this family is
// installed" matches the user's mental model — they downloaded a
// model and now they're deleting it; the supporting Python env should
// go too. Otherwise hundreds of MB linger on disk indefinitely with
// no UI affordance to clean up.
func (m *Manager) maybeRemoveFamily(family string) {
	if family == "" {
		return
	}
	if m.familyHasInstalled(family) {
		return
	}
	if m.downloads.FamilyHasInflight(family) {
		return
	}

	// REASON: the worker stays spawned after a download (so the next
	// load/download is fast). With the last model in this family now
	// gone, that idle worker has nothing left to do — but its python
	// process is still holding open files in .venv, so we have to
	// stop it before removing. Refuse only when it's actively busy
	// (loading/serving), since killing then would interrupt the user.
	var workerToStop workers.Handle
	m.mu.Lock()
	if m.workerFamily == family && m.worker != nil {
		switch m.state {
		case StateLoading, StateServing:
			m.mu.Unlock()
			return
		case StateStarting, StateReady, StateIdle:
			workerToStop = m.worker
			m.worker = nil
			m.workerFamily = ""
			m.state = StateIdle
		}
	}
	m.mu.Unlock()

	if workerToStop != nil {
		stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = workerToStop.Stop(stopCtx)
		cancel()
	}

	if m.envSetup != nil {
		_ = m.envSetup.Remove(family)
	}
}

// familyHasInstalled reports whether any manifest entry shares the
// given family. Cheap: manifest.All copies the map under its lock.
func (m *Manager) familyHasInstalled(family string) bool {
	if m.manifest == nil {
		return false
	}
	for _, entry := range m.manifest.All() {
		if cat := models.EntryByName(entry.Name); cat != nil && cat.Family == family {
			return true
		}
	}
	return false
}
