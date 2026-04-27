// AREA: inference · FAMILY-ENV · TEST
// Unit tests for the per-family venv lifecycle. The `uv sync` call is
// faked via Manager.syncFamily; the venv directory is a tmpdir we
// create/inspect ourselves. No real Python on PATH required.

package inference

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// setupFamilyTree wires a worker tree under a tmpdir and pre-creates
// worker/models/<family>/pyproject.toml so the real defaultSyncFamily
// path could find it (we still inject a fake sync in most tests).
// Returns (workerDir, venvPath); caller assigns m.workerDir = workerDir
// on the Manager under test so per-family code looks at this tree.
func setupFamilyTree(t *testing.T, family string) (string, string) {
	t.Helper()
	root := t.TempDir()
	famDir := filepath.Join(root, "models", family)
	if err := os.MkdirAll(famDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(famDir, "pyproject.toml"), []byte("[project]\nname=\"x\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root, filepath.Join(famDir, ".venv")
}

func TestEnsureFamilyEnv_SkipsSyncWhenVenvExists(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	var calls int32
	m := newManagerWithSpawn(nil)
	m.workerDir = root
	m.syncFamily = func(_ context.Context, _ string) error {
		atomic.AddInt32(&calls, 1)
		return nil
	}

	if err := m.ensureFamilyEnv(context.Background(), "moshi"); err != nil {
		t.Fatalf("ensureFamilyEnv: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 0 {
		t.Errorf("syncFamily calls = %d, want 0 (venv already present)", got)
	}
}

func TestEnsureFamilyEnv_RunsSyncWhenVenvMissing(t *testing.T) {
	root, _ := setupFamilyTree(t, "moshi")
	var calls int32
	m := newManagerWithSpawn(nil)
	m.workerDir = root
	m.syncFamily = func(_ context.Context, family string) error {
		if family != "moshi" {
			t.Errorf("family = %q, want moshi", family)
		}
		atomic.AddInt32(&calls, 1)
		return nil
	}

	if err := m.ensureFamilyEnv(context.Background(), "moshi"); err != nil {
		t.Fatalf("ensureFamilyEnv: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("syncFamily calls = %d, want 1", got)
	}
}

func TestEnsureFamilyEnv_SerializesConcurrentCallsForSameFamily(t *testing.T) {
	// Two goroutines hitting ensureFamilyEnv at the same time must not
	// both shell out — the per-family mutex serializes them and the
	// second one observes the first's freshly-created venv on the
	// re-check path. We model that by having syncFamily create the
	// venv on its only invocation.
	root, venvPath := setupFamilyTree(t, "moshi")
	var calls int32
	m := newManagerWithSpawn(nil)
	m.workerDir = root
	m.syncFamily = func(_ context.Context, _ string) error {
		atomic.AddInt32(&calls, 1)
		// Simulate uv sync's actual side effect (creating the venv
		// dir) plus a small delay so the second caller is forced to
		// wait on the mutex.
		time.Sleep(20 * time.Millisecond)
		return os.MkdirAll(venvPath, 0o755)
	}

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.ensureFamilyEnv(context.Background(), "moshi"); err != nil {
				t.Errorf("ensureFamilyEnv: %v", err)
			}
		}()
	}
	wg.Wait()
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("syncFamily calls = %d, want 1 (mutex should dedupe)", got)
	}
}

func TestMaybeRemoveFamilyEnv_RemovesWhenLastModelDeleted(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	m := newTestDownloadManager(t, &fakeWorker{})
	m.workerDir = root

	m.maybeRemoveFamilyEnv("moshi")

	if _, err := os.Stat(venvPath); !os.IsNotExist(err) {
		t.Errorf("venv should be removed, stat err = %v", err)
	}
}

func TestMaybeRemoveFamilyEnv_KeepsWhenWorkerBusy(t *testing.T) {
	// Loading or Serving means the worker has Python in flight against
	// .venv files; pulling them would crash mid-call.
	for _, state := range []State{StateLoading, StateServing} {
		t.Run(string(state), func(t *testing.T) {
			root, venvPath := setupFamilyTree(t, "moshi")
			if err := os.MkdirAll(venvPath, 0o755); err != nil {
				t.Fatal(err)
			}
			fake := &fakeWorker{}
			m := newTestDownloadManager(t, fake)
			m.workerDir = root
			m.mu.Lock()
			m.worker = fake
			m.workerFamily = "moshi"
			m.state = state
			m.mu.Unlock()

			m.maybeRemoveFamilyEnv("moshi")

			if _, err := os.Stat(venvPath); err != nil {
				t.Errorf("venv should survive %s, stat err = %v", state, err)
			}
		})
	}
}

func TestMaybeRemoveFamilyEnv_StopsIdleWorkerThenRemoves(t *testing.T) {
	// After a download leaves the worker idle (StateReady) and the
	// last family model is then deleted, the cleanup must stop the
	// worker so its open file handles release before we drop .venv.
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	stopped := make(chan struct{}, 1)
	fake := &fakeWorker{stopFn: func(_ context.Context) error {
		stopped <- struct{}{}
		return nil
	}}
	m := newTestDownloadManager(t, fake)
	m.workerDir = root
	m.mu.Lock()
	m.worker = fake
	m.workerFamily = "moshi"
	m.state = StateReady
	m.mu.Unlock()

	m.maybeRemoveFamilyEnv("moshi")

	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("idle worker should have been stopped")
	}
	if _, err := os.Stat(venvPath); !os.IsNotExist(err) {
		t.Errorf("venv should be removed after stopping idle worker, stat err = %v", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker != nil || m.workerFamily != "" || m.state != StateIdle {
		t.Errorf("manager state not reset: worker=%v family=%q state=%s",
			m.worker, m.workerFamily, m.state)
	}
}

func TestMaybeRemoveFamilyEnv_KeepsWhenSiblingInstalled(t *testing.T) {
	// Catalog has only "moshi" today, but the manifest may persist
	// arbitrary names from prior installs (e.g. an explicit moshi-mlx
	// download). The test seeds a sibling that maps to the same family
	// via the catalog so the family-set check trips.
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	m := newTestDownloadManager(t, &fakeWorker{})
	m.workerDir = root
	if err := m.manifest.Put(ManifestEntry{Name: "moshi", Repo: "kyutai/moshiko"}); err != nil {
		t.Fatal(err)
	}

	m.maybeRemoveFamilyEnv("moshi")

	if _, err := os.Stat(venvPath); err != nil {
		t.Errorf("venv should survive while a sibling model is installed, stat err = %v", err)
	}
}

func TestMaybeRemoveFamilyEnv_KeepsWhenSiblingDownloading(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	m := newTestDownloadManager(t, &fakeWorker{})
	m.workerDir = root
	m.mu.Lock()
	m.inflightDownloads["moshi"] = &DownloadProgress{Phase: "downloading"}
	m.mu.Unlock()

	m.maybeRemoveFamilyEnv("moshi")

	if _, err := os.Stat(venvPath); err != nil {
		t.Errorf("venv should survive while a sibling download is in flight, stat err = %v", err)
	}
}
