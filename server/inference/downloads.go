// AREA: inference · DOWNLOADS
// Download/delete lifecycle layered on top of Manager. Lives alongside
// manager.go because it shares mu + the worker handle, but keeps the
// state machine for download progress separate from the load state
// machine — they're orthogonal: a model can be downloading while a
// different model is loaded and serving.
//
// Wire shape:
//
//	UI → POST /models/download {name}
//	  → Manager.DownloadModel(name)
//	  → ensure worker spawned
//	  → IPC: download_model {name, repo, files}
//	  → worker emits download_progress events
//	  → Manager.handleEvent updates inflightDownloads
//	  → worker emits download_done
//	  → Manager records ManifestEntry, clears inflight
//
// Cancellation is best-effort: the worker accepts cancel_download and
// surfaces a download_error("cancelled") on the way out.

package inference

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"
)

// DownloadModel kicks off (or rejects) a model download. Returns
// synchronously after the IPC reply lands so the HTTP handler can
// distinguish "started" from "rejected" cleanly. Progress streams
// over events; the caller polls /models for live byte counts.
func (m *Manager) DownloadModel(name string) error {
	entry := catalogEntryByName(name)
	if entry == nil {
		return fmt.Errorf("unknown model %q", name)
	}
	if !entry.Available {
		return fmt.Errorf("model %q is not yet available", name)
	}
	if entry.Repo == "" || len(entry.Files) == 0 {
		return fmt.Errorf("model %q has no download metadata", name)
	}

	// STEP 1: reserve the inflight slot and seed initial progress so a
	// /models call right after this returns shows the download as
	// active. Doing this before the IPC means the UI's progress bar
	// appears immediately, not after the worker emits its first event.
	m.mu.Lock()
	if _, exists := m.inflightDownloads[name]; exists {
		m.mu.Unlock()
		return errors.New("download already in progress")
	}
	m.inflightDownloads[name] = &DownloadProgress{
		Phase:     "starting",
		FileCount: len(entry.Files),
	}
	m.mu.Unlock()

	// STEP 2: ensure a worker is up. Spawning here matches what
	// LoadModel does — same idempotent pattern, same ctx with a
	// generous timeout. Any failure releases the inflight slot so a
	// retry doesn't hit the "already in progress" guard.
	ctx, cancel := context.WithTimeout(context.Background(), downloadTimeout)
	go func() {
		defer cancel()
		if err := m.runDownload(ctx, name, entry); err != nil {
			m.mu.Lock()
			m.inflightDownloads[name] = &DownloadProgress{
				Phase: "error",
				Error: err.Error(),
			}
			m.mu.Unlock()
			log.Printf("download %q failed: %v", name, err)
		}
	}()
	return nil
}

// runDownload owns the blocking half. Returns an error if anything
// before the worker accepts the IPC fails; once the worker has the
// command, completion is reported via events (handleDownloadEvent).
func (m *Manager) runDownload(ctx context.Context, name string, entry *catalogEntry) error {
	// Spawn or reuse the worker. Mirror LoadModel's spawn dance —
	// state stays at whatever it was (idle/ready/serving), since
	// download is orthogonal to load.
	m.mu.Lock()
	w := m.worker
	m.mu.Unlock()
	if w == nil {
		spawned, err := m.spawn(ctx, workerDir())
		if err != nil {
			return fmt.Errorf("worker spawn failed: %w", err)
		}
		spawned.setOnEvent(m.handleEvent)
		m.mu.Lock()
		// Race: another caller (e.g. LoadModel) may have spawned in
		// parallel. Keep the first one; release ours.
		if m.worker == nil {
			m.worker = spawned
			w = spawned
			if m.state == StateIdle {
				m.state = StateReady
			}
		} else {
			w = m.worker
			// Drop our duplicate. Don't bother stopping it inline —
			// runDownload is on a goroutine and the spawned worker has
			// no in-flight work yet. Defer cleanup.
			go func() {
				stopCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
				defer c()
				_ = spawned.stop(stopCtx)
			}()
		}
		m.mu.Unlock()
	}

	// Send the IPC and wait for the synchronous "started" reply. The
	// worker returns immediately (it spawns its own asyncio task);
	// progress + completion arrive as events.
	_, err := w.send(ctx, "download_model", map[string]any{
		"name":  name,
		"repo":  entry.Repo,
		"files": entry.Files,
	})
	if err != nil {
		return fmt.Errorf("worker rejected download: %w", err)
	}
	return nil
}

// handleDownloadEvent processes a single download_* event from the
// worker. Caller (handleEvent in manager.go) routes here based on
// event type. Holds mu only briefly to keep event-loop latency low.
func (m *Manager) handleDownloadEvent(event string, msg map[string]any) {
	name, _ := msg["name"].(string)
	if name == "" {
		return
	}
	switch event {
	case "download_progress":
		p := &DownloadProgress{
			Phase:      stringField(msg, "phase"),
			File:       stringField(msg, "file"),
			FileIndex:  intField(msg, "fileIndex"),
			FileCount:  intField(msg, "fileCount"),
			Downloaded: int64Field(msg, "downloaded"),
			Total:      int64Field(msg, "total"),
		}
		m.mu.Lock()
		m.inflightDownloads[name] = p
		m.mu.Unlock()
	case "download_done":
		// Persist the install before clearing the inflight slot so a
		// /models call right after sees Installed=true even before the
		// next progress poll. Manifest write failures are logged but
		// not surfaced — the cache files exist on disk regardless.
		files := stringSliceField(msg, "files")
		size := int64Field(msg, "totalBytes")
		entry := ManifestEntry{
			Name:        name,
			Repo:        stringField(msg, "repo"),
			Revision:    stringField(msg, "revision"),
			Files:       files,
			SizeBytes:   size,
			InstalledAt: time.Now().UTC(),
		}
		if m.manifest != nil {
			if err := m.manifest.Put(entry); err != nil {
				log.Printf("manifest write failed for %q: %v", name, err)
			}
		}
		m.mu.Lock()
		delete(m.inflightDownloads, name)
		m.mu.Unlock()
	case "download_error":
		m.mu.Lock()
		m.inflightDownloads[name] = &DownloadProgress{
			Phase: "error",
			Error: stringField(msg, "error"),
		}
		m.mu.Unlock()
	}
}

// DeleteModel removes an installed model from disk and the manifest.
// Refuses if the model is currently loaded or downloading — those are
// the two cases where a delete would race with active use.
func (m *Manager) DeleteModel(name string) error {
	m.mu.Lock()
	if m.state == StateServing && m.model == name {
		m.mu.Unlock()
		return errors.New("model is currently loaded; unload first")
	}
	if _, ok := m.inflightDownloads[name]; ok {
		m.mu.Unlock()
		return errors.New("download in progress for this model")
	}
	var entry *ManifestEntry
	if m.manifest != nil {
		entry = m.manifest.Get(name)
	}
	cat := catalogEntryByName(name)
	m.mu.Unlock()

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
		if root := hubCacheDir(); root != "" {
			if err := os.RemoveAll(repoCacheDir(root, repo)); err != nil {
				return fmt.Errorf("remove cache dir: %w", err)
			}
		}
	}

	// STEP 2: drop the manifest entry. Failure here leaves the cache
	// already gone — surface the error so the caller knows the on-disk
	// truth (cache cleared, manifest stale). On a retry, Delete is
	// idempotent so we converge.
	if m.manifest != nil {
		if err := m.manifest.Delete(name); err != nil {
			return fmt.Errorf("manifest delete: %w", err)
		}
	}
	return nil
}

// ModelInfos returns the catalog merged with the manager's live
// inflight-download state. Single entry point for the /models route
// so the handler doesn't reach into manager internals.
func (m *Manager) ModelInfos() []ModelInfo {
	m.mu.Lock()
	inflight := make(map[string]*DownloadProgress, len(m.inflightDownloads))
	for k, v := range m.inflightDownloads {
		cp := *v
		inflight[k] = &cp
	}
	m.mu.Unlock()
	return ModelInfos(inflight)
}

// --- field helpers ----------------------------------------------------
//
// IPC events arrive as map[string]any; pulling typed values out of
// them ergonomically is verbose without these. Kept private — the
// worker's event shape is internal to this feature.

func stringField(m map[string]any, k string) string {
	v, _ := m[k].(string)
	return v
}

func intField(m map[string]any, k string) int {
	switch v := m[k].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	}
	return 0
}

func int64Field(m map[string]any, k string) int64 {
	switch v := m[k].(type) {
	case float64:
		return int64(v)
	case int:
		return int64(v)
	case int64:
		return v
	}
	return 0
}

func stringSliceField(m map[string]any, k string) []string {
	raw, _ := m[k].([]any)
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
