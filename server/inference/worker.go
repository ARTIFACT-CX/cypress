// AREA: inference · WORKER · IPC
// Owns the Python inference worker subprocess. Speaks the JSON-line control
// protocol on stdin/stdout, multiplexing many in-flight requests onto one
// pipe by correlating replies to requests via a monotonically increasing id.
//
// Lifecycle:
//
//	spawnWorker → ready handshake → send* → stop
//
// The caller (Manager) serializes state transitions; this file is concerned
// only with one live subprocess and the bytes flowing in and out of it.

package inference

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// SETUP: how long we're willing to wait for the worker's `{"ready": true}`
// line after `exec.Start`. uv cold starts + Python imports can be slow on
// first run, so budget generously rather than surface a false failure.
const handshakeTimeout = 20 * time.Second

// reply wraps a decoded JSON line from the worker plus any transport-level
// error (e.g. the worker exited before answering).
type reply struct {
	raw map[string]any
	err error
}

// worker is the Go-side handle to one live Python subprocess. Not safe to
// construct directly — go through spawnWorker so the handshake is verified.
type worker struct {
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	audioSock string

	// nextID hands out correlation ids. Python echoes these back so we can
	// route replies to the goroutine that issued the request.
	nextID atomic.Uint64

	// SAFETY: mu guards waiters. The read loop and every Send caller touch
	// it, so contention is possible but the critical sections are tiny.
	mu      sync.Mutex
	waiters map[uint64]chan reply

	// onEvent handles unsolicited worker→host messages (lines with an
	// "event" field but no "id"). Phase updates during model load flow
	// through here. Set by the Manager after spawn.
	onEvent func(map[string]any)

	// done closes when the read loop exits (worker stdout hit EOF). Used to
	// unblock Send callers with a clear error instead of hanging forever.
	done chan struct{}
}

// spawnWorker launches the worker with the per-family Python venv at
// worker/models/<family>/.venv (cwd stays at worker/ so `import audio`,
// `import ipc`, `import models` resolve to the shared scaffold). Waits
// for the ready handshake and returns a live handle. Any failure before
// handshake kills the process and surfaces a descriptive error — callers
// should not use the returned worker if err != nil.
func spawnWorker(ctx context.Context, family string) (*worker, error) {
	if family == "" {
		return nil, errors.New("spawnWorker: family is required")
	}

	// STEP 1: resolve worker scaffold dir + the family venv's python.
	// We point Python directly at the venv interpreter rather than
	// going through `uv run` so we don't pay uv's resolve overhead on
	// every spawn (and so we don't need a pyproject in the cwd).
	absWorkerDir, err := filepath.Abs(workerRootDir())
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

	// STEP 2: build the command. Stderr is forwarded to our stderr so
	// Python tracebacks land in the server log unmodified — the host sees
	// exactly what the worker printed, which is invaluable when debugging
	// import or device errors.
	cmd := exec.Command(pythonPath, "main.py")
	cmd.Dir = absWorkerDir
	cmd.Env = os.Environ()
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start worker: %w", err)
	}

	w := &worker{
		cmd:     cmd,
		stdin:   stdin,
		waiters: make(map[uint64]chan reply),
		done:    make(chan struct{}),
	}

	// STEP 3: read the handshake line. We scan on the caller's goroutine
	// but with a timeout so a hung worker doesn't block LoadModel forever.
	scanner := bufio.NewScanner(stdout)

	type hsResult struct {
		msg map[string]any
		err error
	}
	hsCh := make(chan hsResult, 1)
	go func() {
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				hsCh <- hsResult{err: fmt.Errorf("read handshake: %w", err)}
			} else {
				hsCh <- hsResult{err: errors.New("worker exited before handshake")}
			}
			return
		}
		var msg map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &msg); err != nil {
			hsCh <- hsResult{err: fmt.Errorf("handshake not valid json: %w (line=%q)", err, scanner.Bytes())}
			return
		}
		hsCh <- hsResult{msg: msg}
	}()

	hsCtx, cancel := context.WithTimeout(ctx, handshakeTimeout)
	defer cancel()

	var hs map[string]any
	select {
	case <-hsCtx.Done():
		_ = w.kill()
		return nil, fmt.Errorf("worker handshake timeout after %s", handshakeTimeout)
	case r := <-hsCh:
		if r.err != nil {
			_ = w.kill()
			return nil, r.err
		}
		hs = r.msg
	}

	// STEP 4: validate the handshake shape. The worker reports fatal
	// startup errors on this same channel as `{"fatal": "..."}`, so surface
	// those verbatim rather than as a generic "unexpected handshake".
	if fatal, ok := hs["fatal"].(string); ok {
		_ = w.kill()
		return nil, fmt.Errorf("worker fatal on startup: %s", fatal)
	}
	if ready, _ := hs["ready"].(bool); !ready {
		_ = w.kill()
		return nil, fmt.Errorf("unexpected handshake: %v", hs)
	}
	if sock, ok := hs["audio_socket"].(string); ok {
		w.audioSock = sock
	}

	// STEP 5: hand the scanner off to a long-lived reader that demuxes all
	// subsequent lines onto per-request waiter channels.
	go w.readLoop(scanner)

	return w, nil
}

// readLoop runs until worker stdout closes. Every line is expected to be a
// JSON object; lines with an "id" are routed to the matching Send caller,
// anything else is logged (unsolicited worker chatter).
func (w *worker) readLoop(scanner *bufio.Scanner) {
	defer close(w.done)

	for scanner.Scan() {
		var msg map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &msg); err != nil {
			log.Printf("worker: malformed line: %s", scanner.Bytes())
			continue
		}

		// id comes back as a float64 because Go's json decoder has no
		// concept of "number that was an integer". We cast on the way out.
		idF, ok := msg["id"].(float64)
		if !ok {
			// No id = unsolicited event from the worker (phase updates
			// during model load, future telemetry, etc.). Route them to
			// the Manager's handler if one is set; otherwise log so we
			// don't silently drop useful signal.
			if _, isEvent := msg["event"]; isEvent && w.onEvent != nil {
				w.onEvent(msg)
			} else {
				log.Printf("worker: unsolicited: %v", msg)
			}
			continue
		}
		id := uint64(idF)

		w.mu.Lock()
		ch, found := w.waiters[id]
		if found {
			delete(w.waiters, id)
		}
		w.mu.Unlock()

		if found {
			ch <- reply{raw: msg}
		} else {
			log.Printf("worker: reply with unknown id=%d: %v", id, msg)
		}
	}

	// Drain any waiters still parked. Without this they'd block forever on
	// their reply channels when the worker has actually exited.
	w.mu.Lock()
	for id, ch := range w.waiters {
		ch <- reply{err: errors.New("worker exited")}
		delete(w.waiters, id)
	}
	w.mu.Unlock()
}

// send dispatches one command and blocks for its reply. Many goroutines may
// call this in parallel; each gets its own correlation id and waiter channel
// so replies don't cross.
func (w *worker) send(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error) {
	id := w.nextID.Add(1)
	payload := map[string]any{"id": id, "cmd": cmd}
	for k, v := range extra {
		payload[k] = v
	}
	line, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode command: %w", err)
	}
	line = append(line, '\n')

	ch := make(chan reply, 1)
	w.mu.Lock()
	w.waiters[id] = ch
	w.mu.Unlock()

	// Clean up the waiter entry on any non-reply exit path so we don't leak.
	drop := func() {
		w.mu.Lock()
		delete(w.waiters, id)
		w.mu.Unlock()
	}

	if _, err := w.stdin.Write(line); err != nil {
		drop()
		return nil, fmt.Errorf("write command: %w", err)
	}

	select {
	case <-ctx.Done():
		drop()
		return nil, ctx.Err()
	case <-w.done:
		return nil, errors.New("worker exited before reply")
	case r := <-ch:
		if r.err != nil {
			return nil, r.err
		}
		// The worker uses "error" as a sentinel for handler-level failures
		// (invalid args, unknown model, etc.). Surface these as Go errors
		// so the HTTP layer can return them to the UI.
		if errStr, ok := r.raw["error"].(string); ok {
			return nil, errors.New(errStr)
		}
		return r.raw, nil
	}
}

// stop asks the worker to exit cleanly, then waits for the process. If the
// caller's ctx expires we escalate to SIGKILL so server shutdown never
// hangs on an unresponsive worker.
func (w *worker) stop(ctx context.Context) error {
	// Best-effort graceful shutdown. Errors here are ignored — we're about
	// to close stdin and reap the process either way.
	_, _ = w.send(ctx, "shutdown", nil)
	_ = w.stdin.Close()

	exited := make(chan error, 1)
	go func() { exited <- w.cmd.Wait() }()

	select {
	case <-ctx.Done():
		_ = w.kill()
		<-exited
		return ctx.Err()
	case err := <-exited:
		return err
	}
}

// setOnEvent is part of the workerHandle interface (see manager.go). The
// Manager calls this immediately after spawn to register its event handler;
// keeping it a method (rather than a public field) lets fakes implement it
// without exposing concurrent-write surface on the real worker struct.
func (w *worker) setOnEvent(fn func(map[string]any)) { w.onEvent = fn }

func (w *worker) kill() error {
	if w.cmd == nil || w.cmd.Process == nil {
		return nil
	}
	return w.cmd.Process.Signal(syscall.SIGKILL)
}
