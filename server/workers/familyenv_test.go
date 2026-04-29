// AREA: workers · FAMILY-ENV · TEST
// Unit tests for the per-family venv lifecycle. The `uv sync` call is
// faked via a SyncFunc; the venv directory is a tmpdir we
// create/inspect ourselves. No real Python on PATH required.

package workers

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
// worker/models/<family>/pyproject.toml so the real DefaultSync path
// could find it (we still inject a fake sync in most tests). Returns
// (workerDir, venvPath); caller passes workerDir to NewEnvSetup.
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

func TestEnsure_SkipsSyncWhenVenvExists(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	var calls int32
	e := NewEnvSetup(root, func(_ context.Context, _ string) error {
		atomic.AddInt32(&calls, 1)
		return nil
	})

	if err := e.Ensure(context.Background(), "moshi"); err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 0 {
		t.Errorf("sync calls = %d, want 0 (venv already present)", got)
	}
}

func TestEnsure_RunsSyncWhenVenvMissing(t *testing.T) {
	root, _ := setupFamilyTree(t, "moshi")
	var calls int32
	e := NewEnvSetup(root, func(_ context.Context, family string) error {
		if family != "moshi" {
			t.Errorf("family = %q, want moshi", family)
		}
		atomic.AddInt32(&calls, 1)
		return nil
	})

	if err := e.Ensure(context.Background(), "moshi"); err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("sync calls = %d, want 1", got)
	}
}

func TestEnsure_SerializesConcurrentCallsForSameFamily(t *testing.T) {
	// Two goroutines hitting Ensure at the same time must not both
	// shell out — the per-family mutex serializes them and the second
	// one observes the first's freshly-created venv on the re-check
	// path. We model that by having sync create the venv on its only
	// invocation.
	root, venvPath := setupFamilyTree(t, "moshi")
	var calls int32
	e := NewEnvSetup(root, func(_ context.Context, _ string) error {
		atomic.AddInt32(&calls, 1)
		// Simulate uv sync's actual side effect (creating the venv
		// dir) plus a small delay so the second caller is forced to
		// wait on the mutex.
		time.Sleep(20 * time.Millisecond)
		return os.MkdirAll(venvPath, 0o755)
	})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := e.Ensure(context.Background(), "moshi"); err != nil {
				t.Errorf("Ensure: %v", err)
			}
		}()
	}
	wg.Wait()
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("sync calls = %d, want 1 (mutex should dedupe)", got)
	}
}

func TestRemove_DeletesVenv(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	e := NewEnvSetup(root, func(context.Context, string) error { return nil })

	if err := e.Remove("moshi"); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := os.Stat(venvPath); !os.IsNotExist(err) {
		t.Errorf("venv should be removed, stat err = %v", err)
	}
}

func TestRemove_NoopWhenMissing(t *testing.T) {
	root, _ := setupFamilyTree(t, "moshi")
	e := NewEnvSetup(root, func(context.Context, string) error { return nil })
	if err := e.Remove("moshi"); err != nil {
		t.Errorf("Remove on missing venv = %v, want nil", err)
	}
}
