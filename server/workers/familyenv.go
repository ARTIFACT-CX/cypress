// AREA: workers · FAMILY-ENV
// Lifecycle for the per-family Python venvs under
// worker/models/<family>/.venv. Created lazily on first download for a
// family, removed when the orchestration layer decides no installed
// model from the family remains. Keeps disk hygiene tied to user
// actions in the model picker — nothing accumulates that the user
// can't see and remove.
//
// Design points:
//   - `uv sync` is shelled out (no Python in-process). On a cold cache
//     this can take ~30-60s for the moshi family (torch wheel is the
//     bottleneck). The download flow surfaces this as a "preparing_env"
//     phase on the existing inflight progress so the UI doesn't go
//     silent.
//   - Per-family setup mutex serializes concurrent `uv sync` runs
//     against the same family. Different families can sync in parallel.
//   - This package only knows how to ensure / remove the venv. The
//     "should we remove it?" decision (no installed sibling, no inflight
//     download, worker not busy) lives in the orchestration layer
//     (inference.Manager) since that's where worker state lives.

package workers

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

// SyncFunc creates / refreshes the Python venv for a family. Defaults
// to running `uv sync`; tests inject a fake to avoid shelling out.
// SWAP seam for a future packaged build that ships pre-baked venvs
// and wants this to be a no-op.
type SyncFunc func(ctx context.Context, family string) error

// EnvSetup owns the per-family venv lifecycle for one workerDir. One
// instance per Manager — keeps the per-family mutex map scoped to the
// process that's spawning workers against it.
type EnvSetup struct {
	workerDir string
	sync      SyncFunc

	mu    sync.Mutex
	famMu map[string]*sync.Mutex
}

// NewEnvSetup wires an EnvSetup for the given worker scaffold root.
// If syncFn is nil, defaults to DefaultSync (real `uv sync`); tests
// pass a fake that creates the .venv directory directly.
func NewEnvSetup(workerDir string, syncFn SyncFunc) *EnvSetup {
	if syncFn == nil {
		syncFn = func(ctx context.Context, family string) error {
			return DefaultSync(ctx, workerDir, family)
		}
	}
	return &EnvSetup{
		workerDir: workerDir,
		sync:      syncFn,
		famMu:     map[string]*sync.Mutex{},
	}
}

// Ensure guarantees that worker/models/<family>/.venv exists before a
// worker is spawned for that family. Cheap when the venv is already
// in place (just an os.Stat). The first call for a family pays the
// full uv sync cost.
func (e *EnvSetup) Ensure(ctx context.Context, family string) error {
	if family == "" {
		return errors.New("EnvSetup.Ensure: family is required")
	}
	venvPath := e.venvPath(family)

	// STEP 1: fast path. The venv directory exists — assume it's
	// usable. If a partial sync left it broken the spawn will fail
	// with a clear import error and the user can manually nuke it.
	if _, err := os.Stat(venvPath); err == nil {
		return nil
	}

	// STEP 2: acquire the per-family setup lock. Another goroutine
	// (e.g. a second download for the same family fired off by the
	// UI) may already be running uv sync; we'd rather wait for it
	// than start a duplicate that races on the same files.
	famMu := e.familyMutex(family)
	famMu.Lock()
	defer famMu.Unlock()

	// STEP 3: re-check after acquiring. The goroutine that held the
	// mutex before us may have just finished a successful sync.
	if _, err := os.Stat(venvPath); err == nil {
		return nil
	}

	// STEP 4: do the actual sync. If sync claims success but the venv
	// is still missing, the next step (SpawnLocal) will surface a
	// clear "family venv missing" error — no need for a duplicate
	// stat here.
	return e.sync(ctx, family)
}

// Remove drops the family's .venv. Acquires the per-family mutex so
// it can't race with a freshly-started Ensure. Safe to call when the
// venv doesn't exist.
//
// REASON: the orchestration layer is responsible for the policy
// decision (is anyone still using this family?) and for stopping
// any worker that has the venv's files open. By the time Remove is
// called those preconditions hold; we just unlink.
func (e *EnvSetup) Remove(family string) error {
	if family == "" {
		return errors.New("EnvSetup.Remove: family is required")
	}
	famMu := e.familyMutex(family)
	famMu.Lock()
	defer famMu.Unlock()

	venvPath := e.venvPath(family)
	if _, err := os.Stat(venvPath); err != nil {
		return nil // already gone
	}
	if err := os.RemoveAll(venvPath); err != nil {
		log.Printf("remove family venv %q failed: %v", family, err)
		return err
	}
	return nil
}

// familyMutex returns (or lazily creates) the per-family setup lock.
func (e *EnvSetup) familyMutex(family string) *sync.Mutex {
	e.mu.Lock()
	defer e.mu.Unlock()
	mu, ok := e.famMu[family]
	if !ok {
		mu = &sync.Mutex{}
		e.famMu[family] = mu
	}
	return mu
}

func (e *EnvSetup) venvPath(family string) string {
	return filepath.Join(e.workerDir, "models", family, ".venv")
}

// DefaultSync runs `uv sync` in worker/models/<family>/. Used as
// EnvSetup.sync when NewEnvSetup is called with a nil syncFn; tests
// inject a no-op directly.
func DefaultSync(ctx context.Context, workerDir, family string) error {
	uvPath, err := exec.LookPath("uv")
	if err != nil {
		return fmt.Errorf("uv not found on PATH (install from https://docs.astral.sh/uv/): %w", err)
	}
	famDir := filepath.Join(workerDir, "models", family)
	if _, err := os.Stat(filepath.Join(famDir, "pyproject.toml")); err != nil {
		return fmt.Errorf("family %q has no pyproject.toml at %s: %w", family, famDir, err)
	}
	// REASON: forward stdout/stderr to our stderr so a long sync prints
	// progress to the server log instead of disappearing. Users running
	// `cypress server` in a terminal then see the wheel download bars.
	cmd := exec.CommandContext(ctx, uvPath, "sync")
	cmd.Dir = famDir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("uv sync %q: %w", family, err)
	}
	return nil
}
