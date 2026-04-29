// AREA: inference · FAMILY-POLICY · TEST
// Tests the Manager's policy for removing the per-family venv after a
// model delete. The mechanical bits (file removal, cache scrub, sync)
// are tested in workers/ and downloads/; here we cover the cross-package
// orchestration: when does Manager actually call envSetup.Remove?

package inference

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ARTIFACT-CX/cypress/server/downloads"
	"github.com/ARTIFACT-CX/cypress/server/models"
	"github.com/ARTIFACT-CX/cypress/server/workers"
)

// setupFamilyTree mirrors workers/setupFamilyTree but reachable from
// the inference package — they're intentional dupes (small, internal,
// per-feature) to keep the family layout known by the test.
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

// newFamilyTestManager wires a Manager with a fake worker, a real
// (tmpdir) manifest, and an EnvSetup pointed at a tmpdir family tree.
// The fakeWorker is the only spawn target; downloads.Service shares
// the manifest so manifest-driven sibling checks fire.
func newFamilyTestManager(t *testing.T, fake *fakeWorker, root string) *Manager {
	t.Helper()
	mf, err := models.NewManifest(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	envSetup := workers.NewEnvSetup(root, func(context.Context, string) error { return nil })
	m := &Manager{
		state:    StateIdle,
		envSetup: envSetup,
		manifest: mf,
		spawn: func(_ context.Context, _ string) (workers.Handle, error) {
			return fake, nil
		},
	}
	m.downloads = downloads.New(m, envSetup, mf)
	return m
}

func TestDeleteModel_RemovesFamilyVenvWhenLast(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	m := newFamilyTestManager(t, &fakeWorker{}, root)
	// Seed manifest with the model we're about to delete so DeleteFiles
	// has something to drop. Catalog repo isn't needed for this path.
	if err := m.manifest.Put(models.ManifestEntry{Name: "moshi", Repo: "kyutai/moshiko"}); err != nil {
		t.Fatal(err)
	}

	if err := m.DeleteModel("moshi"); err != nil {
		t.Fatalf("DeleteModel: %v", err)
	}

	if _, err := os.Stat(venvPath); !os.IsNotExist(err) {
		t.Errorf("venv should be removed, stat err = %v", err)
	}
}

func TestDeleteModel_KeepsVenvWhenWorkerBusy(t *testing.T) {
	for _, state := range []State{StateLoading, StateServing} {
		t.Run(string(state), func(t *testing.T) {
			root, venvPath := setupFamilyTree(t, "moshi")
			if err := os.MkdirAll(venvPath, 0o755); err != nil {
				t.Fatal(err)
			}
			fake := &fakeWorker{}
			m := newFamilyTestManager(t, fake, root)
			if err := m.manifest.Put(models.ManifestEntry{Name: "moshi"}); err != nil {
				t.Fatal(err)
			}
			m.mu.Lock()
			m.worker = fake
			m.workerFamily = "moshi"
			m.state = state
			// REASON: when state is Serving, DeleteModel("moshi") would
			// short-circuit on "currently loaded". Use a different model
			// name on Serving so we exercise the family-policy branch.
			if state == StateServing {
				m.model = "other"
			}
			m.mu.Unlock()

			if err := m.DeleteModel("moshi"); err != nil {
				t.Fatalf("DeleteModel: %v", err)
			}
			if _, err := os.Stat(venvPath); err != nil {
				t.Errorf("venv should survive %s, stat err = %v", state, err)
			}
		})
	}
}

func TestDeleteModel_StopsIdleWorkerThenRemovesVenv(t *testing.T) {
	root, venvPath := setupFamilyTree(t, "moshi")
	if err := os.MkdirAll(venvPath, 0o755); err != nil {
		t.Fatal(err)
	}
	stopped := make(chan struct{}, 1)
	fake := &fakeWorker{stopFn: func(context.Context) error {
		stopped <- struct{}{}
		return nil
	}}
	m := newFamilyTestManager(t, fake, root)
	if err := m.manifest.Put(models.ManifestEntry{Name: "moshi"}); err != nil {
		t.Fatal(err)
	}
	m.mu.Lock()
	m.worker = fake
	m.workerFamily = "moshi"
	m.state = StateReady
	m.mu.Unlock()

	if err := m.DeleteModel("moshi"); err != nil {
		t.Fatalf("DeleteModel: %v", err)
	}

	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("idle worker should have been stopped")
	}
	if _, err := os.Stat(venvPath); !os.IsNotExist(err) {
		t.Errorf("venv should be removed, stat err = %v", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.worker != nil || m.workerFamily != "" || m.state != StateIdle {
		t.Errorf("manager not reset: worker=%v family=%q state=%s",
			m.worker, m.workerFamily, m.state)
	}
}

