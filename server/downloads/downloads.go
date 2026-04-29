// AREA: downloads · SERVICE
// Download/cancel/delete state machine for the model picker. The Python
// worker actually does the HF pull; this package owns the Go-side
// inflight bookkeeping, the manifest write on completion, and the cache
// teardown on delete. Keeps the load state machine (inference.Manager)
// out of the download lifecycle so the two evolve independently.
//
// Wire shape:
//
//	UI → POST /models/{name}/download
//	  → Service.Start(name)
//	  → ensure family venv (workers.EnvSetup)
//	  → ensure worker spawned (WorkerProvider)
//	  → IPC: download_model {name, repo, files}
//	  → worker emits download_progress events
//	  → Service.HandleEvent updates inflight
//	  → worker emits download_done
//	  → Service records ManifestEntry, clears inflight
//
// Cancellation is best-effort: the worker accepts cancel_download and
// surfaces a download_error("cancelled") on the way out.

package downloads

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/models"
	"github.com/ARTIFACT-CX/cypress/server/workers"
)

// downloadTimeout caps a single download. Generous because residential
// links can take many minutes for multi-GB pulls, but bounded so a
// wedged transfer can't pin the worker forever.
const downloadTimeout = 20 * time.Minute

// WorkerProvider is the Service's view of the worker lifecycle. The
// inference Manager satisfies it — Service doesn't import inference.
//
// SWAP: any orchestrator that wants to drive downloads through a
// different worker pool implements this same surface.
type WorkerProvider interface {
	// Worker returns the running worker handle and its family, or
	// (nil, "") if no worker is up.
	Worker() (workers.Handle, string)
	// SpawnWorker ensures a worker is running for the given family,
	// installing the orchestrator's event handler on it. Errors if a
	// worker is already up for a different family — the caller should
	// have refused the cross-family request before reaching here, but
	// this is a defensive backstop.
	SpawnWorker(ctx context.Context, family string) (workers.Handle, error)
}

// Service owns the download lifecycle. Construct via New; methods are
// safe for concurrent use.
type Service struct {
	provider WorkerProvider
	envSetup *workers.EnvSetup
	manifest *models.Manifest

	// SAFETY: mu guards inflight. Held only briefly — IPC roundtrips
	// happen with mu released so the UI's /models poll stays responsive.
	mu       sync.Mutex
	inflight map[string]*models.DownloadProgress
}

// New wires a Service. envSetup may be nil to skip venv preparation
// (test convenience); manifest may be nil to skip persistence.
func New(provider WorkerProvider, envSetup *workers.EnvSetup, manifest *models.Manifest) *Service {
	return &Service{
		provider: provider,
		envSetup: envSetup,
		manifest: manifest,
		inflight: map[string]*models.DownloadProgress{},
	}
}

// Start kicks off (or rejects) a model download. Returns synchronously
// after the inflight slot is reserved; the IPC and worker setup run on
// a background goroutine so the HTTP handler returns fast.
func (s *Service) Start(name string) error {
	entry := models.EntryByName(name)
	if entry == nil {
		return fmt.Errorf("unknown model %q", name)
	}
	if !entry.Available {
		return fmt.Errorf("model %q is not yet available", name)
	}
	if entry.Repo == "" || len(entry.Files) == 0 {
		return fmt.Errorf("model %q has no download metadata", name)
	}
	if entry.Family == "" {
		return fmt.Errorf("model %q has no family configured", name)
	}

	// REASON: same single-family worker constraint as LoadModel — refuse
	// a download that would target a different family's venv than the
	// running worker. The user can unload first; .incomplete blobs the
	// previous family wrote stay on disk so they can resume later.
	if _, runningFamily := s.provider.Worker(); runningFamily != "" && runningFamily != entry.Family {
		return fmt.Errorf("worker is running %q models; unload before downloading %q (family %q)",
			runningFamily, name, entry.Family)
	}

	// STEP 1: reserve the inflight slot so a /models call right after
	// this returns shows the download as active.
	s.mu.Lock()
	if _, exists := s.inflight[name]; exists {
		s.mu.Unlock()
		return errors.New("download already in progress")
	}
	s.inflight[name] = &models.DownloadProgress{
		Phase:     "starting",
		FileCount: len(entry.Files),
	}
	s.mu.Unlock()

	// STEP 2: hand off the slow work. Any failure releases the inflight
	// slot via an error-shaped progress entry so the UI can render the
	// failure and the user can retry.
	ctx, cancel := context.WithTimeout(context.Background(), downloadTimeout)
	go func() {
		defer cancel()
		if err := s.run(ctx, name, entry); err != nil {
			s.mu.Lock()
			s.inflight[name] = &models.DownloadProgress{
				Phase: "error",
				Error: err.Error(),
			}
			s.mu.Unlock()
			log.Printf("download %q failed: %v", name, err)
		}
	}()
	return nil
}

// run is the blocking half. Returns an error if anything before the
// worker accepts the IPC fails; once the worker has the command,
// completion is reported via events (HandleEvent).
func (s *Service) run(ctx context.Context, name string, entry *models.Entry) error {
	// STEP 0: ensure the family venv exists. First-time downloads pay
	// `uv sync` here — surface as a "preparing_env" phase so the UI
	// progress bar doesn't silently freeze for the 30-60s wheel pull.
	s.mu.Lock()
	if cur := s.inflight[name]; cur != nil {
		cur.Phase = "preparing_env"
	}
	s.mu.Unlock()
	if s.envSetup != nil {
		if err := s.envSetup.Ensure(ctx, entry.Family); err != nil {
			return fmt.Errorf("prepare family env: %w", err)
		}
	}

	// STEP 1: spawn or reuse the worker.
	w, err := s.provider.SpawnWorker(ctx, entry.Family)
	if err != nil {
		return fmt.Errorf("worker spawn failed: %w", err)
	}

	// STEP 2: send the IPC and wait for the synchronous "started" reply.
	// The worker returns immediately (it spawns its own asyncio task);
	// progress + completion arrive as events.
	if _, err := w.Send(ctx, "download_model", map[string]any{
		"name":  name,
		"repo":  entry.Repo,
		"files": entry.Files,
	}); err != nil {
		return fmt.Errorf("worker rejected download: %w", err)
	}
	return nil
}

// HandleEvent processes a single download_* event from the worker.
// Caller (orchestrator's event router) routes here based on event type.
// Holds mu only briefly to keep event-loop latency low.
func (s *Service) HandleEvent(event string, msg map[string]any) {
	name := workers.StringField(msg, "name")
	if name == "" {
		return
	}
	switch event {
	case "download_progress":
		p := &models.DownloadProgress{
			Phase:      workers.StringField(msg, "phase"),
			File:       workers.StringField(msg, "file"),
			FileIndex:  workers.IntField(msg, "fileIndex"),
			FileCount:  workers.IntField(msg, "fileCount"),
			Downloaded: workers.Int64Field(msg, "downloaded"),
			Total:      workers.Int64Field(msg, "total"),
		}
		s.mu.Lock()
		s.inflight[name] = p
		s.mu.Unlock()
	case "download_done":
		// Persist the install before clearing the inflight slot so a
		// /models call right after sees Installed=true even before the
		// next progress poll. Manifest write failures are logged but
		// not surfaced — the cache files exist on disk regardless.
		entry := models.ManifestEntry{
			Name:        name,
			Repo:        workers.StringField(msg, "repo"),
			Revision:    workers.StringField(msg, "revision"),
			Files:       workers.StringSliceField(msg, "files"),
			SizeBytes:   workers.Int64Field(msg, "totalBytes"),
			InstalledAt: time.Now().UTC(),
		}
		if s.manifest != nil {
			if err := s.manifest.Put(entry); err != nil {
				log.Printf("manifest write failed for %q: %v", name, err)
			}
		}
		s.mu.Lock()
		delete(s.inflight, name)
		s.mu.Unlock()
	case "download_error":
		errMsg := workers.StringField(msg, "error")
		s.mu.Lock()
		// REASON: cancellation surfaces as an error event from the worker
		// but the user already knows it happened — clear the slot so the
		// row reverts to "Download" instead of showing a stuck error.
		if errMsg == "cancelled" {
			delete(s.inflight, name)
		} else {
			s.inflight[name] = &models.DownloadProgress{
				Phase: "error",
				Error: errMsg,
			}
		}
		s.mu.Unlock()
	}
}

// Cancel asks the worker to abort the in-flight pull. The worker emits
// a download_error("cancelled") on its way out, which HandleEvent
// translates into a slot-clear. .incomplete blobs stay on disk so a
// retry resumes via HF's Range-aware fetch.
func (s *Service) Cancel(name string) error {
	s.mu.Lock()
	_, inflight := s.inflight[name]
	s.mu.Unlock()
	if !inflight {
		return errors.New("no download in progress for this model")
	}
	w, _ := s.provider.Worker()
	if w == nil {
		// Inflight slot exists but no worker — shouldn't happen, but
		// drop the slot so the UI recovers either way.
		s.mu.Lock()
		delete(s.inflight, name)
		s.mu.Unlock()
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := w.Send(ctx, "cancel_download", map[string]any{"name": name}); err != nil {
		return fmt.Errorf("worker cancel rejected: %w", err)
	}
	return nil
}

// DeleteFiles wipes the HF cache directory and manifest entry for name.
// Pure mechanical step — the caller (orchestrator) is responsible for
// the policy checks (model not loaded, not downloading) and for any
// downstream cleanup like dropping the family venv.
func (s *Service) DeleteFiles(name string) error {
	var entry *models.ManifestEntry
	if s.manifest != nil {
		entry = s.manifest.Get(name)
	}
	cat := models.EntryByName(name)

	// STEP 1: nuke the HF cache directory for this repo. Removing the
	// snapshot tree alone would leave orphaned blobs; HF's loader is
	// smart enough to re-fetch on demand if we over-delete, so we go
	// for the whole models--<repo> tree.
	repo := ""
	if entry != nil {
		repo = entry.Repo
	} else if cat != nil {
		repo = cat.Repo
	}
	if repo != "" {
		if root := models.HubCacheDir(); root != "" {
			if err := os.RemoveAll(models.RepoCacheDir(root, repo)); err != nil {
				return fmt.Errorf("remove cache dir: %w", err)
			}
		}
	}

	// STEP 2: drop the manifest entry. Failure here leaves the cache
	// already gone — surface the error so the caller knows the on-disk
	// truth (cache cleared, manifest stale). On a retry, Delete is
	// idempotent so we converge.
	if s.manifest != nil {
		if err := s.manifest.Delete(name); err != nil {
			return fmt.Errorf("manifest delete: %w", err)
		}
	}
	return nil
}

// IsInflight reports whether a download is currently active for name.
// Cheap snapshot — the orchestrator uses this for pre-flight checks.
func (s *Service) IsInflight(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.inflight[name]
	return ok
}

// FamilyHasInflight reports whether any inflight download targets the
// given family. Used by the orchestrator to decide whether dropping
// the family venv would crash a sibling download in flight.
func (s *Service) FamilyHasInflight(family string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for name := range s.inflight {
		if cat := models.EntryByName(name); cat != nil && cat.Family == family {
			return true
		}
	}
	return false
}

// Inflight returns a snapshot of the inflight map. Used by tests and
// by ModelInfos to merge progress into the catalog response.
func (s *Service) Inflight() map[string]*models.DownloadProgress {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]*models.DownloadProgress, len(s.inflight))
	for k, v := range s.inflight {
		cp := *v
		out[k] = &cp
	}
	return out
}

// ModelInfos returns the catalog merged with current inflight progress.
// Single entry point for the /models route so the handler doesn't
// reach into Service internals.
func (s *Service) ModelInfos() []models.ModelInfo {
	return models.ModelInfos(s.Inflight())
}
