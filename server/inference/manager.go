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

	// remote is the configured remote endpoint, if any. Stored so the
	// auto-reconnect path knows whether a transport drop should trigger
	// a redial (remote: yes; local subprocess crash: no, surface and
	// transition to idle).
	remote *workers.RemoteEndpoint

	// workerPlatform is the snapshot the worker stamped onto its last
	// gRPC Handshake (OS, arch, available backends, downloaded repos).
	// Used to pick model variants — without this, an Apple-Silicon
	// laptop dialing a Linux GPU worker would pick MLX weights the
	// worker can't load. Refreshed every time we install a new worker
	// handle.
	//
	// Local mode keeps this empty; Platform() falls back to
	// runtime.GOOS/GOARCH there since the laptop *is* the worker.
	workerPlatform workers.Platform
	// platformReady fires once the first remote handshake has populated
	// workerPlatform. Until then /models returns an empty + loading=true
	// response so the UI shows a spinner instead of the host's catalog.
	// Closed exactly once.
	platformReady     chan struct{}
	platformReadyOnce sync.Once
	// downloadedRepos tracks which HF repos the worker reports as fully
	// cached. Seeded from the handshake snapshot, kept current via
	// download_done events for the rest of the session. Guarded by mu.
	downloadedRepos map[string]bool

	// Remote reachability state. The Go HTTP server can come up fine
	// (port bound, /status answering) while the configured remote
	// worker is unreachable — typo in the URL, expired SLURM
	// allocation, dead SSH tunnel. Without surfacing this the UI shows
	// "Started" with an empty catalog and no clue why. Captured from
	// every dial attempt (eager probe, LoadModel spawn, redial); a
	// background loop re-probes while not reachable so the banner
	// auto-clears once the user fixes the underlying issue.
	//
	// All three are guarded by mu. Only meaningful in remote mode;
	// local mode leaves them at zero values.
	remoteReachable   bool
	remoteLastError   string
	remoteLastChecked time.Time

	// shutdownCtx aborts the periodic remote-health probe loop on
	// Manager.Shutdown so it doesn't outlive the process. Built
	// alongside the Manager; cancel runs in Shutdown.
	shutdownCtx    context.Context
	shutdownCancel context.CancelFunc
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
	// Remote, when non-nil, points the Manager at an external worker
	// (RunPod / BYO GPU box) instead of spawning a local subprocess.
	// The local WorkerDir / per-family venvs are unused in that mode —
	// the remote container brings its own family. Composition-root
	// builds this from CYPRESS_REMOTE_URL / _TOKEN env vars.
	Remote *workers.RemoteEndpoint
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
	// Transport identifies the worker backend: "local" (subprocess) or
	// "remote" (gRPC over TCP+TLS or SSH-tunneled loopback). Lets the
	// UI label "Local subprocess" vs "Remote (<url>)" in the
	// server-details popup without having to peek at env vars.
	Transport string `json:"transport"`
	// Remote is populated only when Transport == "remote". Lets the UI
	// surface a banner when the worker is unreachable instead of
	// leaving the catalog silently empty.
	Remote *RemoteStatus `json:"remote,omitempty"`
}

// RemoteStatus reports the health of the configured remote worker
// connection. URL is the dial target (token redacted by definition —
// it's never in here). Reachable flips false when a dial fails and
// back true when the next handshake (eager probe, LoadModel, periodic
// re-probe) succeeds; LastError is the most recent dial failure
// message, cleared on a reachable transition.
type RemoteStatus struct {
	URL         string    `json:"url"`
	Reachable   bool      `json:"reachable"`
	LastError   string    `json:"lastError,omitempty"`
	LastChecked time.Time `json:"lastChecked"`
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
	// REASON: when a remote endpoint is configured we don't need the
	// local family-venv plumbing — the remote container ships its own.
	// Pass a nil envSetup so Ensure/Remove become no-ops on this side.
	var envSetup *workers.EnvSetup
	if cfg.Remote == nil {
		envSetup = workers.NewEnvSetup(cfg.WorkerDir, nil)
	}
	// REASON: pick the spawn closure based on transport. Local mode
	// dispatches per family (each family has its own venv). Remote
	// mode ignores the family arg — the deployed container is one
	// fixed family, configured at deploy time, not per request.
	var spawn SpawnFn
	if cfg.Remote != nil {
		spawn = func(ctx context.Context, _ string) (workers.Handle, error) {
			return workers.DialRemote(ctx, cfg.Remote)
		}
	} else {
		spawn = func(ctx context.Context, family string) (workers.Handle, error) {
			return workers.SpawnLocal(ctx, cfg.WorkerDir, family)
		}
	}
	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())
	m := &Manager{
		state:           StateIdle,
		envSetup:        envSetup,
		manifest:        mf,
		spawn:           spawn,
		remote:          cfg.Remote,
		platformReady:   make(chan struct{}),
		downloadedRepos: map[string]bool{},
		shutdownCtx:     shutdownCtx,
		shutdownCancel:  shutdownCancel,
	}
	m.downloads = downloads.New(m, envSetup, mf)

	// REASON: in local mode the worker is the laptop, so the platform
	// is known synchronously from runtime. Mark ready immediately so
	// /models doesn't sit in the loading state forever.
	if cfg.Remote == nil {
		m.markPlatformReady()
	} else {
		// Eager probe: dial the remote worker once at startup, capture
		// the handshake fields, then disconnect. Subsequent LoadModel
		// calls open their own handles. This is what makes the catalog
		// UI render the right variants on first paint instead of after
		// the user clicks Load. Runs on a goroutine so NewManager stays
		// non-blocking.
		go m.probeRemotePlatform()
		// Periodic re-probe whenever reachability is false. Lets the
		// UI banner clear automatically when the user fixes a tunnel /
		// starts a SLURM job / pastes the right token without having
		// to restart the app.
		go m.remoteHealthLoop()
	}
	return m
}

// newManagerWithSpawn is the test constructor. Lets a unit test inject
// a fake spawner without going through subprocess machinery. envSetup
// is nil (Service skips Ensure) and manifest is nil unless the test
// supplies one via setManifest.
func newManagerWithSpawn(spawn SpawnFn) *Manager {
	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())
	m := &Manager{
		state:           StateIdle,
		spawn:           spawn,
		platformReady:   make(chan struct{}),
		downloadedRepos: map[string]bool{},
		shutdownCtx:     shutdownCtx,
		shutdownCancel:  shutdownCancel,
	}
	m.downloads = downloads.New(m, nil, nil)
	// Tests default to local mode (no remote endpoint set).
	m.markPlatformReady()
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
	// Platform-independent — we only need the family field, which is
	// the same across variants.
	family := models.FamilyOf(name)
	if family == "" {
		return fmt.Errorf("unknown model %q", name)
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
	if m.worker != nil && m.workerFamily != "" && m.workerFamily != family {
		m.mu.Unlock()
		return fmt.Errorf("worker is running %q models; unload before loading %q (family %q)",
			m.workerFamily, name, family)
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
		m.doLoad(ctx, name, family)
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
			// Spawn failure on a remote worker is the same signal as a
			// failed eager probe — the box isn't reachable. Marking it
			// here keeps /status fresh between periodic re-probes.
			if m.remote != nil {
				m.markRemoteUnreachable(err)
			}
			return
		}
		w.SetOnEvent(m.handleEvent)
		m.worker = w
		m.workerFamily = family
		m.state = StateLoading
		// Refresh platform snapshot from this handle's handshake. In
		// local mode the runtime values stay authoritative (recorded
		// values are unused for variant selection); in remote mode this
		// catches any cache changes the worker picked up since the
		// eager probe.
		m.recordHandshakePlatformLocked(w.Platform())
		m.markPlatformReady()
		m.mu.Unlock()
		if m.remote != nil {
			m.markRemoteReachable()
		}
		go m.watchWorker(w, family)
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
		// REASON: keep the worker's downloaded-repos snapshot fresh as
		// installs land. Without this, /models would still show the
		// model as "Download" after a successful pull until the next
		// handshake — exactly the regression described in the bug.
		if event == "download_done" {
			if repo, ok := msg["repo"].(string); ok && repo != "" {
				m.mu.Lock()
				m.downloadedRepos[repo] = true
				m.mu.Unlock()
			}
		}
	}
}

// Status returns a point-in-time snapshot of the inference subsystem.
// Grouped into one struct (rather than separate accessors per field) so
// callers can't interleave reads and see inconsistent combinations.
func (m *Manager) Status() Snapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	snap := Snapshot{
		State:  m.state,
		Model:  m.model,
		Device: m.device,
		Phase:  m.phase,
		Error:  m.lastError,
	}
	if m.remote != nil {
		snap.Transport = "remote"
		snap.Remote = &RemoteStatus{
			URL:         m.remote.URL,
			Reachable:   m.remoteReachable,
			LastError:   m.remoteLastError,
			LastChecked: m.remoteLastChecked,
		}
	} else {
		snap.Transport = "local"
	}
	return snap
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
	cancel := m.shutdownCancel
	m.mu.Unlock()

	// SAFETY: cancel the shutdown context BEFORE calling Stop on the
	// worker — the periodic remote-health probe runs on this context
	// and holds gRPC connections; if we leave it running across
	// Shutdown it'll dial against a stopped state machine.
	if cancel != nil {
		cancel()
	}
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
		if m.remote != nil {
			m.markRemoteUnreachable(err)
		}
		return nil, err
	}
	if m.remote != nil {
		m.markRemoteReachable()
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
	m.recordHandshakePlatformLocked(spawned.Platform())
	m.markPlatformReady()
	go m.watchWorker(spawned, family)
	return spawned, nil
}

// --- download / delete passthroughs --------------------------------------------

// DownloadModel kicks off a download via the downloads service. Thin
// passthrough kept on Manager so the HTTP handler has one entry point
// for model lifecycle commands. The repo is selected for the worker's
// platform — for remote workers the laptop's runtime is the wrong
// answer (would request MLX weights for a Linux GPU box).
func (m *Manager) DownloadModel(name string) error {
	os, arch, ready := m.Platform()
	if !ready {
		// REASON: catalog UI shouldn't surface a Download button before
		// the platform is known, but a CLI / curl could still hit this
		// — refuse rather than guess wrong.
		return errors.New("worker platform not yet known; try again in a moment")
	}
	return m.downloads.Start(name, os, arch)
}

// CancelDownload aborts the in-flight pull for name.
func (m *Manager) CancelDownload(name string) error { return m.downloads.Cancel(name) }

// ModelInfos is the catalog merged with inflight progress for the
// worker's platform. Returns nil when the platform isn't known yet
// (eager remote probe still in flight) so the handler can surface a
// loading state to the UI.
func (m *Manager) ModelInfos() []models.ModelInfo {
	os, arch, ready := m.Platform()
	if !ready {
		return nil
	}
	return m.downloads.ModelInfos(os, arch, m.DownloadedRepos())
}

// PlatformReadyForResponse exposes whether the catalog is currently
// renderable. Used by the /models handler to set a loading flag in
// the JSON response so the UI can show a spinner instead of an empty
// list.
func (m *Manager) PlatformReadyForResponse() bool {
	_, _, ready := m.Platform()
	return ready
}

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

	family := models.FamilyOf(name)
	if err := m.downloads.DeleteFiles(name); err != nil {
		return err
	}
	if family != "" {
		m.maybeRemoveFamily(family)
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
		if models.FamilyOf(entry.Name) == family {
			return true
		}
	}
	return false
}

// --- Platform discovery -------------------------------------------------------
//
// The Manager exposes the platform that variant selection should target.
// Local mode: laptop's runtime values. Remote mode: whatever the worker
// reported on its last Handshake. The /models handler reads this to
// pick repo + files matched to where the model will actually run.

// Platform returns the (os, arch) tuple variant selection should target,
// plus whether the worker has been probed yet. Callers that need to
// render the catalog should consult `ready` first — empty values
// before the eager probe completes mean "platform unknown, defer."
func (m *Manager) Platform() (os, arch string, ready bool) {
	// Fast path: closed channel means probe completed (or local mode).
	select {
	case <-m.platformReady:
	default:
		return "", "", false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.remote == nil {
		host, harch := models.HostPlatform()
		return host, harch, true
	}
	return m.workerPlatform.OS, m.workerPlatform.Arch, true
}

// PlatformReady returns a channel that closes when the platform tuple
// is available — local mode: at NewManager return; remote mode: after
// the eager probe handshake. Useful for tests that want to wait
// deterministically rather than poll.
func (m *Manager) PlatformReady() <-chan struct{} { return m.platformReady }

// DownloadedRepos returns a copy of the worker-reported set of
// fully-cached HF repos. Used by ModelInfos to avoid the wrong-side
// IsRepoCached probe (laptop's cache is empty when the worker is
// remote). Returns nil before the platform is ready so the caller can
// fall through to a local probe in local mode.
func (m *Manager) DownloadedRepos() map[string]bool {
	select {
	case <-m.platformReady:
	default:
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.remote == nil {
		// Local mode: the worker IS the laptop, so the on-disk probe
		// in info.go is authoritative. Returning nil opts back into it.
		return nil
	}
	out := make(map[string]bool, len(m.downloadedRepos))
	for k, v := range m.downloadedRepos {
		out[k] = v
	}
	return out
}

// markPlatformReady closes platformReady exactly once. Idempotent so
// either the eager probe path or a concurrent watcher refresh can call
// it without a panic on double-close.
func (m *Manager) markPlatformReady() {
	m.platformReadyOnce.Do(func() { close(m.platformReady) })
}

// recordHandshakePlatform stores the platform snapshot from a freshly
// installed worker handle. Called by every site that installs a new
// `m.worker`: doLoad, SpawnWorker, watchWorker (post-redial), and the
// eager probe. Must be called with m.mu held.
func (m *Manager) recordHandshakePlatformLocked(p workers.Platform) {
	m.workerPlatform = p
	// Reset and reseed the downloaded-repos set from the handshake.
	// REASON: the worker is the source of truth on every reconnect; if
	// its cache changed (volume swap, manual delete) we want to honor
	// the new reality, not stale state from a prior session.
	m.downloadedRepos = make(map[string]bool, len(p.DownloadedRepos))
	for _, repo := range p.DownloadedRepos {
		m.downloadedRepos[repo] = true
	}
}

// probeRemotePlatform dials the remote worker once at startup, captures
// the handshake fields, and disconnects. Runs on a goroutine kicked
// off by NewManager — failures here aren't fatal (the user might not
// have started the remote worker yet); markPlatformReady still fires
// so /models can return its empty/loading response definitively rather
// than spinning forever. The next LoadModel will dial again and
// refresh the snapshot.
func (m *Manager) probeRemotePlatform() {
	// REASON: scope the dial deadline tightly. The eager probe is
	// best-effort; if the remote isn't reachable yet we'd rather mark
	// ready quickly so the UI can render *something*.
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Family arg is ignored by the remote spawn closure — see NewManager.
	w, err := m.spawn(ctx, "")
	if err != nil {
		log.Printf("remote platform probe failed: %v (catalog will populate after first load)", err)
		m.markRemoteUnreachable(err)
		m.markPlatformReady()
		return
	}
	plat := w.Platform()

	m.mu.Lock()
	m.recordHandshakePlatformLocked(plat)
	m.mu.Unlock()
	m.markRemoteReachable()
	m.markPlatformReady()

	// SAFETY: Disconnect, not Stop. The probe's job is "open a session,
	// read the handshake, hang up." Stop sends a shutdown RPC that
	// kills the Python process — the next LoadModel / Download dial
	// would then get a connection-reset because there's no listener
	// left. Disconnect closes only the gRPC channel; the worker stays
	// up to serve subsequent dials.
	_ = w.Disconnect()
}

// --- Remote reachability ----------------------------------------------------
//
// The Go HTTP server's "running" state means the laptop-side process
// is up; it doesn't mean the remote worker is reachable. These helpers
// + the periodic loop below let /status carry the remote-side health
// separately so the UI can show a banner instead of an empty catalog.

// remoteHealthInterval governs how often the periodic loop re-probes
// while reachability is false. var (not const) so tests can shrink it
// without slowing the suite. 30s balances "fix-then-see" responsiveness
// against connection churn while the user fiddles with their tunnel.
var remoteHealthInterval = 30 * time.Second

// markRemoteReachable records that a recent dial against the remote
// worker succeeded. Idempotent; the LastChecked timestamp updates on
// every success so the UI can show "checked 4s ago."
func (m *Manager) markRemoteReachable() {
	now := time.Now().UTC()
	m.mu.Lock()
	m.remoteReachable = true
	m.remoteLastError = ""
	m.remoteLastChecked = now
	m.mu.Unlock()
}

// markRemoteUnreachable records the most recent dial failure. Repeated
// failures with the same error overwrite each other (no error
// accumulation) so the UI shows the latest, not the first.
func (m *Manager) markRemoteUnreachable(err error) {
	now := time.Now().UTC()
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	m.mu.Lock()
	m.remoteReachable = false
	m.remoteLastError = msg
	m.remoteLastChecked = now
	m.mu.Unlock()
}

// remoteHealthLoop re-probes the remote worker on a slow interval
// while reachability is false. Stops re-probing when reachability
// flips true — subsequent user actions (LoadModel, Download) keep the
// flag fresh, and a transport drop will mark unreachable again to
// re-arm the loop.
//
// SAFETY: shutdownCtx is the kill switch; Manager.Shutdown cancels it
// to bring this goroutine down before the test/process exits. Without
// that, tests would leak a goroutine per Manager construction.
func (m *Manager) remoteHealthLoop() {
	t := time.NewTicker(remoteHealthInterval)
	defer t.Stop()
	for {
		select {
		case <-m.shutdownCtx.Done():
			return
		case <-t.C:
			// Skip the probe when reachability is already true — saves
			// a TLS handshake every interval for no information gain.
			// Other dial sites refresh the flag on success/failure so
			// staleness is bounded by user activity in the happy case.
			m.mu.Lock()
			reachable := m.remoteReachable
			m.mu.Unlock()
			if reachable {
				continue
			}
			m.runHealthProbe()
		}
	}
}

// runHealthProbe is one health-check tick: dial, capture handshake,
// disconnect. Same shape as probeRemotePlatform but doesn't touch the
// platformReady gate (already closed by the eager probe). Refreshes
// the platform snapshot on success so a worker that came back with a
// new HF cache state (e.g. an out-of-band download) reflects in the
// catalog without waiting for a LoadModel.
func (m *Manager) runHealthProbe() {
	ctx, cancel := context.WithTimeout(m.shutdownCtx, 15*time.Second)
	defer cancel()
	w, err := m.spawn(ctx, "")
	if err != nil {
		m.markRemoteUnreachable(err)
		return
	}
	plat := w.Platform()
	m.mu.Lock()
	m.recordHandshakePlatformLocked(plat)
	m.mu.Unlock()
	m.markRemoteReachable()
	_ = w.Disconnect()
}
