//go:build integration

// AREA: inference · TESTS · INTEGRATION
// Runs against a real Python worker subprocess. Gated by the `integration`
// build tag so default `go test ./...` stays fast and dependency-free; run
// explicitly with:
//
//	go test -tags=integration ./inference/...
//
// Requires `uv` on PATH and the worker/ directory's deps installed.

package inference

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// findWorkerDir walks up from the test's CWD looking for a sibling `worker`
// directory. Lets the integration test run from server/ or server/inference/
// without hardcoding paths.
func findWorkerDir(t *testing.T) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	dir := cwd
	for i := 0; i < 5; i++ {
		candidate := filepath.Join(dir, "worker")
		if _, err := os.Stat(filepath.Join(candidate, "main.py")); err == nil {
			return candidate
		}
		dir = filepath.Dir(dir)
	}
	t.Fatalf("could not locate worker/ from %s", cwd)
	return ""
}

// TestIntegration_Spawn_HandshakeAndStatus verifies the end-to-end gRPC
// session against a real Python process spoken over a Unix-domain socket.
// Doesn't load a model — that would download GBs and take minutes. Just
// proves spawn, handshake, one round-trip, and clean shutdown all work.
func TestIntegration_Spawn_HandshakeAndStatus(t *testing.T) {
	dir := findWorkerDir(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	w, err := spawnWorker(ctx, dir, "moshi")
	if err != nil {
		t.Fatalf("spawnWorker: %v", err)
	}
	defer func() {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer stopCancel()
		_ = w.stop(stopCtx)
	}()

	reply, err := w.send(ctx, "status", nil)
	if err != nil {
		t.Fatalf("send status: %v", err)
	}
	if reply["model"] != nil {
		t.Errorf("model = %v, want nil (no model loaded yet)", reply["model"])
	}
}
