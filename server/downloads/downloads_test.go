// AREA: downloads · TESTS
// Unit tests for the download/cancel/delete state machine. Worker is
// faked via a tiny stub satisfying workers.Handle; HF cache is faked
// via tmpdir + HUGGINGFACE_HUB_CACHE.

package downloads

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/models"
	"github.com/ARTIFACT-CX/cypress/server/workers"
)

// fakeWorker is a workers.Handle stand-in. Each behavior the Service
// cares about (send result, event registration) is configurable so
// individual tests can shape responses.
type fakeWorker struct {
	mu sync.Mutex

	sendFn  func(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error)
	stopErr error
	onEvent func(map[string]any)

	sendCalls []string
}

func (f *fakeWorker) Send(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error) {
	f.mu.Lock()
	f.sendCalls = append(f.sendCalls, cmd)
	fn := f.sendFn
	f.mu.Unlock()
	if fn != nil {
		return fn(ctx, cmd, extra)
	}
	return map[string]any{"ok": true}, nil
}

func (f *fakeWorker) Stop(_ context.Context) error { return f.stopErr }

func (f *fakeWorker) SetOnEvent(fn func(map[string]any)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.onEvent = fn
}

// fakeProvider satisfies WorkerProvider with the given fake worker as
// the always-current handle. SpawnWorker installs the manager-style
// onEvent handler on the fake (a no-op handler by default; override
// via setEventHandler).
type fakeProvider struct {
	fake         *fakeWorker
	family       string
	eventHandler func(map[string]any)
}

func (p *fakeProvider) Worker() (workers.Handle, string) {
	if p.fake == nil {
		return nil, ""
	}
	return p.fake, p.family
}

func (p *fakeProvider) SpawnWorker(_ context.Context, family string) (workers.Handle, error) {
	p.family = family
	if p.eventHandler != nil {
		p.fake.SetOnEvent(p.eventHandler)
	}
	return p.fake, nil
}

func newTestService(t *testing.T, fake *fakeWorker) (*Service, *fakeProvider) {
	t.Helper()
	mf, err := models.NewManifest(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	provider := &fakeProvider{fake: fake}
	s := New(provider, nil, mf)
	// Wire the Service's HandleEvent the way the orchestrator would.
	provider.eventHandler = func(m map[string]any) {
		event, _ := m["event"].(string)
		s.HandleEvent(event, m)
	}
	return s, provider
}

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

func TestStart_HappyPath(t *testing.T) {
	fake := &fakeWorker{
		sendFn: func(_ context.Context, _ string, _ map[string]any) (map[string]any, error) {
			return map[string]any{"ok": true, "started": true}, nil
		},
	}
	s, _ := newTestService(t, fake)

	if err := s.Start("moshi"); err != nil {
		t.Fatalf("Start: %v", err)
	}

	waitFor(t, "onEvent registered", func() bool {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		return fake.onEvent != nil
	})
	waitFor(t, "inflight populated", func() bool {
		return s.IsInflight("moshi")
	})

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
	p := s.Inflight()["moshi"]
	if p == nil || p.Downloaded != 1024 || p.Total != 4096 || p.Phase != "downloading" {
		t.Fatalf("progress not propagated: %+v", p)
	}

	emit(map[string]any{
		"event":      "download_done",
		"name":       "moshi",
		"repo":       "kyutai/moshiko-mlx-q8",
		"files":      []any{"/cache/a", "/cache/b"},
		"totalBytes": float64(4096),
	})
	waitFor(t, "inflight cleared", func() bool {
		return !s.IsInflight("moshi")
	})
	if !s.manifest.Has("moshi") {
		t.Fatal("manifest should record install on download_done")
	}
}

func TestStart_RejectsConcurrent(t *testing.T) {
	block := make(chan struct{})
	defer close(block)
	fake := &fakeWorker{
		sendFn: func(ctx context.Context, _ string, _ map[string]any) (map[string]any, error) {
			<-block
			return map[string]any{"ok": true}, nil
		},
	}
	s, _ := newTestService(t, fake)

	if err := s.Start("moshi"); err != nil {
		t.Fatalf("first Start: %v", err)
	}
	if err := s.Start("moshi"); err == nil {
		t.Fatal("second Start should error while first is in flight")
	}
}

func TestStart_UnknownRejected(t *testing.T) {
	s, _ := newTestService(t, &fakeWorker{})
	if err := s.Start("not-a-real-model"); err == nil {
		t.Fatal("unknown model should be rejected")
	}
}

func TestStart_RejectsCrossFamily(t *testing.T) {
	fake := &fakeWorker{}
	s, provider := newTestService(t, fake)
	provider.family = "personaplex" // worker already running another family
	if err := s.Start("moshi"); err == nil {
		t.Fatal("should refuse cross-family while worker is up")
	}
}

func TestHandleEvent_ErrorClearsManifest(t *testing.T) {
	fake := &fakeWorker{}
	s, _ := newTestService(t, fake)
	if err := s.Start("moshi"); err != nil {
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
	p := s.Inflight()["moshi"]
	if p == nil || p.Phase != "error" || p.Error == "" {
		t.Fatalf("error event not propagated: %+v", p)
	}
	if s.manifest.Has("moshi") {
		t.Fatal("manifest must not record entry on error")
	}
}

func TestDeleteFiles_RemovesCacheAndManifest(t *testing.T) {
	cacheRoot := t.TempDir()
	t.Setenv("HUGGINGFACE_HUB_CACHE", cacheRoot)

	entry := models.DefaultMoshiEntry()
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

	s, _ := newTestService(t, &fakeWorker{})
	if err := s.manifest.Put(models.ManifestEntry{Name: "moshi", Repo: entry.Repo}); err != nil {
		t.Fatal(err)
	}

	if err := s.DeleteFiles("moshi"); err != nil {
		t.Fatalf("DeleteFiles: %v", err)
	}
	if _, err := os.Stat(repoDir); !os.IsNotExist(err) {
		t.Fatalf("cache dir should be gone, stat err = %v", err)
	}
	if s.manifest.Has("moshi") {
		t.Fatal("manifest entry should be removed")
	}
}

func TestFamilyHasInflight(t *testing.T) {
	s, _ := newTestService(t, &fakeWorker{})
	if s.FamilyHasInflight("moshi") {
		t.Fatal("empty service should report no inflight")
	}
	s.mu.Lock()
	s.inflight["moshi"] = &models.DownloadProgress{Phase: "downloading"}
	s.mu.Unlock()
	if !s.FamilyHasInflight("moshi") {
		t.Fatal("FamilyHasInflight should detect inflight in family")
	}
	if s.FamilyHasInflight("personaplex") {
		t.Fatal("FamilyHasInflight should not match unrelated family")
	}
}
