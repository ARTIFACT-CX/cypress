// AREA: inference · FAMILY-ENV
// Lifecycle for the per-family Python venvs under
// worker/models/<family>/.venv. Created lazily on first download for a
// family, removed when the last installed model in the family is
// deleted. Keeps disk hygiene tied to user actions in the model
// picker — nothing accumulates that the user can't see and remove.
//
// Design points:
//   - `uv sync` is shelled out (no Python in-process). On a cold cache
//     this can take ~30-60s for the moshi family (torch wheel is the
//     bottleneck). The download flow surfaces this as a "preparing_env"
//     phase on the existing inflight progress so the UI doesn't go
//     silent.
//   - Per-family setup mutex (Manager.familySetupMu) serializes
//     concurrent `uv sync` runs against the same family. Different
//     families can sync in parallel.
//   - Removal is conservative: only when no installed models from the
//     family remain in the manifest, no inflight downloads target it,
//     and the worker isn't currently running on it. A failed remove is
//     logged but not surfaced — the user can re-trigger by downloading
//     and deleting again.

package inference

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// defaultSyncFamily runs `uv sync` in worker/models/<family>/. Used as
// Manager.syncFamily by NewManager; tests inject a no-op. workerDir is
// the resolved worker scaffold root (composition root passes it in).
func defaultSyncFamily(ctx context.Context, workerDir, family string) error {
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

// ensureFamilyEnv guarantees that worker/models/<family>/.venv exists
// before a worker is spawned for that family. Cheap when the venv is
// already in place (just an os.Stat). The first call for a family pays
// the full uv sync cost.
func (m *Manager) ensureFamilyEnv(ctx context.Context, family string) error {
	if family == "" {
		return errors.New("ensureFamilyEnv: family is required")
	}
	venvPath := filepath.Join(m.workerDir, "models", family, ".venv")

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
	m.mu.Lock()
	if m.familySetupMu == nil {
		m.familySetupMu = map[string]*sync.Mutex{}
	}
	famMu, ok := m.familySetupMu[family]
	if !ok {
		famMu = &sync.Mutex{}
		m.familySetupMu[family] = famMu
	}
	m.mu.Unlock()

	famMu.Lock()
	defer famMu.Unlock()

	// STEP 3: re-check after acquiring. The goroutine that held the
	// mutex before us may have just finished a successful sync.
	if _, err := os.Stat(venvPath); err == nil {
		return nil
	}

	// STEP 4: do the actual sync. syncFamily is injected so unit
	// tests can drive this without uv on PATH. If sync claims success
	// but the venv is still missing, the next step (spawnWorker) will
	// surface a clear "family venv missing" error — no need for a
	// duplicate stat here.
	return m.syncFamily(ctx, family)
}

// maybeRemoveFamilyEnv drops the family's .venv if no installed models
// from that family remain and the worker isn't currently running on
// it. Called from DeleteModel after the manifest entry is gone. Safe
// to call when the venv doesn't exist.
//
// REASON: tying venv lifetime to "any model from this family is
// installed" matches the user's mental model — they downloaded a
// model and now they're deleting it; the supporting Python env should
// go too. Otherwise hundreds of MB linger on disk indefinitely with
// no UI affordance to clean up.
func (m *Manager) maybeRemoveFamilyEnv(family string) {
	if family == "" {
		return
	}
	m.mu.Lock()
	// Refuse if any other installed model belongs to this family.
	if m.manifest != nil {
		for _, entry := range m.manifest.All() {
			if cat := catalogEntryByName(entry.Name); cat != nil && cat.Family == family {
				m.mu.Unlock()
				return
			}
		}
	}
	// Refuse if any inflight download targets this family — dropping
	// the venv would crash the download mid-flight.
	for name := range m.inflightDownloads {
		if cat := catalogEntryByName(name); cat != nil && cat.Family == family {
			m.mu.Unlock()
			return
		}
	}

	// REASON: the worker stays spawned after a download (so the next
	// load/download is fast). With the last model in this family now
	// gone, that idle worker has nothing left to do — but its python
	// process is still holding open files in .venv, so we have to
	// stop it before removing. Refuse only when it's actively busy
	// (loading/serving), since killing then would interrupt the user.
	var workerToStop workerHandle
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
		_ = workerToStop.stop(stopCtx)
		cancel()
	}

	// Acquire the per-family setup mutex too, so a remove can't race
	// with a freshly-started download's uv sync.
	m.mu.Lock()
	famMu, ok := m.familySetupMu[family]
	m.mu.Unlock()
	if ok {
		famMu.Lock()
		defer famMu.Unlock()
	}

	venvPath := filepath.Join(m.workerDir, "models", family, ".venv")
	if _, err := os.Stat(venvPath); err != nil {
		return // already gone
	}
	if err := os.RemoveAll(venvPath); err != nil {
		log.Printf("remove family venv %q failed: %v", family, err)
	}
}
