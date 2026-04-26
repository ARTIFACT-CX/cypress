// AREA: inference · DOWNLOADS · TEST
// Unit tests for the download/delete state machine. Worker is faked
// (see fakeWorker in manager_test.go); HF cache is faked via tmpdir.

package inference

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// newTestDownloadManager wires a Manager with a fakeWorker AND a real
// (tmpdir) manifest, since download tests check both inflight tracking
// and persistence.
func newTestDownloadManager(t *testing.T, fake *fakeWorker) *Manager {
	t.Helper()
	t.Setenv("CYPRESS_DATA_DIR", t.TempDir())
	mf, err := NewManifest()
	if err != nil {
		t.Fatal(err)
	}
	m := newManagerWithSpawn(func(_ context.Context, _ string) (workerHandle, error) {
		return fake, nil
	})
	m.manifest = mf
	return m
}

// testInflight is a same-package accessor for inflightDownloads — the
// public API exposes only ModelInfos() (catalog + inflight merged) so
// tests reach in directly to assert raw state transitions.
func testInflight(m *Manager) map[string]*DownloadProgress {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]*DownloadProgress, len(m.inflightDownloads))
	for k, v := range m.inflightDownloads {
		cp := *v
		out[k] = &cp
	}
	return out
}

// waitFor polls fn until it returns true or the deadline elapses.
func waitFor(t *testing.T, name string, fn func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if fn() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", name)
}

func TestManager_DownloadModel_HappyPath(t *testing.T) {
	// Worker accepts download_model immediately; we then fire synthetic
	// progress + done events to drive the state machine through to a
	// manifest write.
	var mu sync.Mutex
	var sendCalls []string
	fake := &fakeWorker{
		sendFn: func(_ context.Context, cmd string, _ map[string]any) (map[string]any, error) {
			mu.Lock()
			sendCalls = append(sendCalls, cmd)
			mu.Unlock()
			return map[string]any{"ok": true, "started": true}, nil
		},
	}
	m := newTestDownloadManager(t, fake)

	if err := m.DownloadModel("moshi"); err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}

	// Wait for the manager to register its event handler on the fake.
	waitFor(t, "onEvent registered", func() bool {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		return fake.onEvent != nil
	})

	// Inflight starts populated.
	waitFor(t, "inflight populated", func() bool {
		_, ok := testInflight(m)["moshi"]
		return ok
	})

	// Progress event lands on InflightDownloads.
	fake.mu.Lock()
	emit := fake.onEvent
	fake.mu.Unlock()
	emit(map[string]any{
		"event":      "download_progress",
		"name":       "moshi",
		"phase":      "downloading",
		"file":       "model.safetensors",
		"fileIndex":  float64(0),
		"fileCount":  float64(3),
		"downloaded": float64(1024),
		"total":      float64(4096),
	})
	p := testInflight(m)["moshi"]
	if p == nil || p.Downloaded != 1024 || p.Total != 4096 || p.Phase != "downloading" {
		t.Fatalf("progress not propagated: %+v", p)
	}

	// Completion clears inflight + writes the manifest.
	emit(map[string]any{
		"event":      "download_done",
		"name":       "moshi",
		"repo":       "kyutai/moshiko-mlx-q8",
		"files":      []any{"/cache/a", "/cache/b"},
		"totalBytes": float64(4096),
	})
	waitFor(t, "inflight cleared", func() bool {
		_, ok := testInflight(m)["moshi"]
		return !ok
	})
	if !m.manifest.Has("moshi") {
		t.Fatal("manifest should record the install on download_done")
	}
	entry := m.manifest.Get("moshi")
	if entry.Repo != "kyutai/moshiko-mlx-q8" || entry.SizeBytes != 4096 {
		t.Fatalf("manifest entry mismatch: %+v", entry)
	}
	if len(entry.Files) != 2 {
		t.Fatalf("expected 2 files, got %v", entry.Files)
	}
}

func TestManager_DownloadModel_RejectsConcurrent(t *testing.T) {
	block := make(chan struct{})
	fake := &fakeWorker{
		sendFn: func(ctx context.Context, _ string, _ map[string]any) (map[string]any, error) {
			<-block
			return map[string]any{"ok": true}, nil
		},
	}
	defer close(block)
	m := newTestDownloadManager(t, fake)

	if err := m.DownloadModel("moshi"); err != nil {
		t.Fatalf("first DownloadModel: %v", err)
	}
	if err := m.DownloadModel("moshi"); err == nil {
		t.Fatal("second DownloadModel should error while first is in flight")
	}
}

func TestManager_DownloadModel_UnknownRejected(t *testing.T) {
	m := newTestDownloadManager(t, &fakeWorker{})
	if err := m.DownloadModel("not-a-real-model"); err == nil {
		t.Fatal("unknown model should be rejected")
	}
}

func TestManager_DownloadModel_ErrorEventClearsInflight(t *testing.T) {
	fake := &fakeWorker{}
	m := newTestDownloadManager(t, fake)
	if err := m.DownloadModel("moshi"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "onEvent ready", func() bool {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		return fake.onEvent != nil
	})
	fake.mu.Lock()
	emit := fake.onEvent
	fake.mu.Unlock()
	emit(map[string]any{
		"event": "download_error",
		"name":  "moshi",
		"error": "network down",
	})
	p := testInflight(m)["moshi"]
	if p == nil || p.Phase != "error" || p.Error == "" {
		t.Fatalf("error event not propagated: %+v", p)
	}
	if m.manifest.Has("moshi") {
		t.Fatal("manifest must not record entry on error")
	}
}

func TestManager_DeleteModel_RemovesCacheAndManifest(t *testing.T) {
	// Stand up a fake HF cache layout so DeleteModel has something to
	// remove. We point HUGGINGFACE_HUB_CACHE at a tmpdir and pre-create
	// the moshi cache subtree.
	cacheRoot := t.TempDir()
	t.Setenv("HUGGINGFACE_HUB_CACHE", cacheRoot)

	entry := defaultMoshiEntry()
	repoDir := filepath.Join(
		cacheRoot,
		"models--"+strings.ReplaceAll(entry.Repo, "/", "--"),
	)
	if err := os.MkdirAll(filepath.Join(repoDir, "snapshots", "abc"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repoDir, "snapshots", "abc", "f"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := newTestDownloadManager(t, &fakeWorker{})
	if err := m.manifest.Put(ManifestEntry{
		Name: "moshi",
		Repo: entry.Repo,
	}); err != nil {
		t.Fatal(err)
	}

	if err := m.DeleteModel("moshi"); err != nil {
		t.Fatalf("DeleteModel: %v", err)
	}
	if _, err := os.Stat(repoDir); !os.IsNotExist(err) {
		t.Fatalf("cache dir should be gone, stat err = %v", err)
	}
	if m.manifest.Has("moshi") {
		t.Fatal("manifest entry should be removed")
	}
}

func TestManager_DeleteModel_RefusesWhenLoaded(t *testing.T) {
	m := newTestDownloadManager(t, &fakeWorker{})
	m.mu.Lock()
	m.state = StateServing
	m.model = "moshi"
	m.mu.Unlock()
	if err := m.DeleteModel("moshi"); err == nil {
		t.Fatal("delete should refuse while model is loaded")
	}
}

func TestManager_DeleteModel_RefusesWhenDownloading(t *testing.T) {
	m := newTestDownloadManager(t, &fakeWorker{})
	m.mu.Lock()
	m.inflightDownloads["moshi"] = &DownloadProgress{Phase: "downloading"}
	m.mu.Unlock()
	if err := m.DeleteModel("moshi"); err == nil {
		t.Fatal("delete should refuse while download is in flight")
	}
}
