// AREA: workers · SPAWN · LOCAL
// Local-subprocess flavor of a worker. Launches the per-family Python
// venv at worker/models/<family>/.venv with `--listen unix:<path>`,
// dials the resulting gRPC server, and waits for the handshake.
// Returns a *Grpc that satisfies Handle so callers don't care which
// transport backs it.
//
// Lifecycle:
//
//	SpawnLocal → handshake → Send* → Stop
//
// The caller (inference.Manager) serializes state transitions; this
// file is concerned only with one live subprocess + its gRPC channel.

package workers

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
)

// SETUP: how long we're willing to wait for the worker's
// Handshake{ready=true} after `exec.Start`. uv cold starts + Python
// imports can be slow on first run, so budget generously rather than
// surface a false failure.
const handshakeTimeout = 20 * time.Second

// socketPollInterval bounds how aggressively we poll for the worker's
// unix socket to appear after exec. 25ms is fast enough that an
// already-warm Python startup feels instant and slow enough that we
// don't burn CPU for the whole 20s budget if something's wrong.
const socketPollInterval = 25 * time.Millisecond

// SpawnLocal launches a worker subprocess against the per-family Python
// venv and dials it over a unix socket. workerDir is the resolved
// worker/ scaffold root; family selects which venv. Any failure before
// handshake kills the process and surfaces a descriptive error —
// callers should not use the returned worker if err != nil.
func SpawnLocal(ctx context.Context, workerDir, family string) (*Grpc, error) {
	if family == "" {
		return nil, errors.New("SpawnLocal: family is required")
	}
	if workerDir == "" {
		return nil, errors.New("SpawnLocal: workerDir is required")
	}

	// STEP 1: resolve the family venv's python. Direct interpreter path
	// avoids `uv run`'s resolve overhead on every spawn (and avoids the
	// requirement for a pyproject in the cwd).
	absWorkerDir, err := filepath.Abs(workerDir)
	if err != nil {
		return nil, fmt.Errorf("resolve worker dir: %w", err)
	}
	if _, err := os.Stat(absWorkerDir); err != nil {
		return nil, fmt.Errorf("worker dir %q: %w", absWorkerDir, err)
	}
	pythonPath := filepath.Join(absWorkerDir, "models", family, ".venv", "bin", "python")
	if _, err := os.Stat(pythonPath); err != nil {
		return nil, fmt.Errorf("family venv missing for %q (run `uv sync` in worker/models/%s): %w",
			family, family, err)
	}

	// STEP 2: mint the socket path. We pick it (rather than letting
	// Python derive a per-pid default) so we know exactly where to dial
	// without parsing handshake metadata. /tmp is fine on macOS;
	// DataDir-rooted paths are a follow-up if Linux /tmp cleaners bite.
	sockPath := filepath.Join(os.TempDir(), fmt.Sprintf("cypress-worker-%d-%d.sock", os.Getpid(), time.Now().UnixNano()))
	// REASON: stale socket from a prior crash blocks bind on the Python
	// side. Best-effort unlink before exec.
	_ = os.Remove(sockPath)

	// STEP 3: build the command. Stderr is forwarded to our stderr so
	// Python tracebacks land in the server log unmodified.
	cmd := exec.Command(pythonPath, "main.py", "--listen", "unix:"+sockPath)
	cmd.Dir = absWorkerDir
	cmd.Env = os.Environ()
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start worker: %w", err)
	}

	// STEP 4: wait for the socket to appear, then dial + open Session.
	// Wrap kill so any error path before we have a *Grpc can clean up
	// the subprocess.
	killOnFail := func() {
		if cmd.Process != nil {
			_ = cmd.Process.Signal(syscall.SIGKILL)
		}
		_ = os.Remove(sockPath)
	}

	if err := waitForSocket(ctx, sockPath, handshakeTimeout); err != nil {
		killOnFail()
		return nil, err
	}

	w, err := dialGRPC(ctx, "unix:"+sockPath, cmd, sockPath)
	if err != nil {
		killOnFail()
		return nil, err
	}
	return w, nil
}

// waitForSocket polls until the unix socket file exists or the budget
// runs out. We can't dial blindly because grpc.NewClient is lazy about
// errors and a connect to a nonexistent socket fails on the first RPC,
// not at dial time — so polling here gives us a clean error message
// ("worker handshake timeout") instead of an opaque later failure.
func waitForSocket(ctx context.Context, path string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	tick := time.NewTicker(socketPollInterval)
	defer tick.Stop()
	for {
		if _, err := os.Stat(path); err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("worker socket %s did not appear within %s", path, timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
		}
	}
}
