// AREA: inference · WORKER · GRPC
// gRPC client implementation of workerHandle. Speaks the Worker.Session
// bidi RPC defined in proto/cypress/worker/v1/worker.proto: every command
// the Manager calls becomes a typed ClientMsg, every reply / event flowing
// the other way is a typed ServerMsg. Wire conversion is contained in
// wireconv.go so the rest of the file is plumbing.
//
// Two transports use this same code path:
//   - Local subprocess: dial unix://<path>, see worker.go.
//   - Remote worker: dial dns:///host:port + TLS + bearer auth (TODO).

package inference

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/ARTIFACT-CX/cypress/proto/dist/go/workerpb"
)

// reply wraps a decoded ServerMsg.Reply plus any transport-level error
// (e.g. the worker exited before answering).
type reply struct {
	raw map[string]any
	err error
}

// grpcWorker is the live gRPC handle to one worker. Local and remote
// flavors construct it the same way after their respective dials.
type grpcWorker struct {
	conn   *grpc.ClientConn
	stream grpc.BidiStreamingClient[pb.ClientMsg, pb.ServerMsg]
	cancel context.CancelFunc

	// cmd / sockPath are populated only for the local-subprocess flavor;
	// nil for remote workers. stop() consults them to reap the process
	// and clean up the unix socket.
	cmd      *exec.Cmd
	sockPath string

	// nextID hands out correlation ids. Python echoes these back so we
	// can route replies to the goroutine that issued the request.
	nextID atomic.Uint64

	// SAFETY: mu guards waiters. The recv loop and every send caller
	// touch it, so contention is possible but the critical sections are tiny.
	mu      sync.Mutex
	waiters map[uint64]chan reply

	// sendCh serializes outbound writes. gRPC requires that stream.Send
	// be called from at most one goroutine; we feed it from here.
	sendCh chan *pb.ClientMsg

	// onEvent handles unsolicited worker→host messages. Set by the
	// Manager after spawn.
	onEvent func(map[string]any)

	// done closes when the recv loop exits. Used to unblock send callers
	// with a clear error rather than hanging forever.
	done chan struct{}
}

// dialGRPC opens the Session bidi stream against `target` and waits
// for the Handshake. The cmd / sockPath args are optional: pass them
// for the local-subprocess flavor so stop() can reap and clean up;
// pass nil/"" for remote workers.
func dialGRPC(ctx context.Context, target string, cmd *exec.Cmd, sockPath string) (*grpcWorker, error) {
	// REASON: insecure on a unix socket is fine — file perms are the
	// auth. Remote (TCP) callers will swap this for credentials.NewTLS
	// + a per-RPC bearer creds when that path lands.
	conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("grpc dial %s: %w", target, err)
	}

	client := pb.NewWorkerClient(conn)

	// REASON: derive a long-lived context from Background, not the
	// caller's ctx. The caller's ctx times out after handshakeTimeout
	// (or whatever the LoadModel deadline is) but the bidi stream
	// must outlive it — otherwise every long-running download would
	// kill its own session.
	streamCtx, cancel := context.WithCancel(context.Background())
	stream, err := client.Session(streamCtx)
	if err != nil {
		cancel()
		_ = conn.Close()
		return nil, fmt.Errorf("open session: %w", err)
	}

	// STEP: read the handshake on the caller's ctx so a hung worker
	// doesn't pin LoadModel forever.
	hsCh := make(chan error, 1)
	var first *pb.ServerMsg
	go func() {
		msg, recvErr := stream.Recv()
		first = msg
		hsCh <- recvErr
	}()

	hsCtx, hsCancel := context.WithTimeout(ctx, handshakeTimeout)
	defer hsCancel()

	select {
	case <-hsCtx.Done():
		cancel()
		_ = conn.Close()
		return nil, fmt.Errorf("worker handshake timeout after %s", handshakeTimeout)
	case err := <-hsCh:
		if err != nil {
			cancel()
			_ = conn.Close()
			return nil, fmt.Errorf("read handshake: %w", err)
		}
	}

	hs := first.GetHandshake()
	if hs == nil {
		cancel()
		_ = conn.Close()
		return nil, fmt.Errorf("first server message was %T, want Handshake", first.GetPayload())
	}
	if !hs.GetReady() {
		fatal := hs.GetFatal()
		cancel()
		_ = conn.Close()
		if fatal != "" {
			return nil, fmt.Errorf("worker fatal on startup: %s", fatal)
		}
		return nil, errors.New("worker handshake reported not ready")
	}

	w := &grpcWorker{
		conn:     conn,
		stream:   stream,
		cancel:   cancel,
		cmd:      cmd,
		sockPath: sockPath,
		waiters:  make(map[uint64]chan reply),
		sendCh:   make(chan *pb.ClientMsg, 16),
		done:     make(chan struct{}),
	}

	// STEP: hand the stream off to the read + write loops. Send is
	// owned by sendLoop; Recv by recvLoop. Both exit when the stream
	// ends (client cancel, server close, or transport error).
	go w.sendLoop()
	go w.recvLoop()
	return w, nil
}

// sendLoop owns stream.Send. Pulls ClientMsg values off sendCh and
// writes them to the wire in arrival order. Exits when sendCh is
// closed (stop) or Send returns an error (transport dead).
func (w *grpcWorker) sendLoop() {
	for msg := range w.sendCh {
		if err := w.stream.Send(msg); err != nil {
			// Failed sends drain to the waiter via recvLoop's exit
			// path — recv will hit io.EOF / Canceled when the stream
			// dies, surface "worker exited", and unparker will pick
			// it up. No need to also surface here.
			return
		}
	}
	// Closing send half tells the server we're done so it can exit
	// the request iterator cleanly. recvLoop hits EOF after this.
	_ = w.stream.CloseSend()
}

// recvLoop owns stream.Recv. Each ServerMsg routes to either a
// waiter (Reply) or the registered onEvent callback (Event); the
// initial Handshake was consumed by dialGRPC.
func (w *grpcWorker) recvLoop() {
	defer close(w.done)
	for {
		msg, err := w.stream.Recv()
		if err != nil {
			// EOF / cancellation are normal stop paths. Anything else
			// gets logged so an unexpected transport failure isn't
			// silent.
			if err != io.EOF && !errors.Is(err, context.Canceled) {
				log.Printf("worker: recv: %v", err)
			}
			break
		}

		switch p := msg.GetPayload().(type) {
		case *pb.ServerMsg_Reply:
			r := p.Reply
			id := r.GetId()
			w.mu.Lock()
			ch, found := w.waiters[id]
			if found {
				delete(w.waiters, id)
			}
			w.mu.Unlock()
			if !found {
				log.Printf("worker: reply with unknown id=%d", id)
				continue
			}
			if errStr := r.GetError(); errStr != "" {
				ch <- reply{err: errors.New(errStr)}
			} else {
				ch <- reply{raw: replyToMap(r)}
			}

		case *pb.ServerMsg_Event:
			if w.onEvent == nil {
				continue
			}
			if dict := eventToMap(p.Event); dict != nil {
				w.onEvent(dict)
			}

		case *pb.ServerMsg_Handshake:
			// Spurious second handshake — ignore. The contract is
			// "first message"; anything after that is unexpected.
			log.Printf("worker: ignoring late handshake")
		}
	}

	// Drain any waiters still parked. Without this they'd block forever
	// on their reply channels when the worker has actually exited.
	w.mu.Lock()
	for id, ch := range w.waiters {
		ch <- reply{err: errors.New("worker exited before reply")}
		delete(w.waiters, id)
	}
	w.mu.Unlock()
}

// send dispatches one command and blocks for its reply. Many goroutines
// may call this in parallel; each gets its own correlation id and
// waiter channel so replies don't cross.
func (w *grpcWorker) send(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error) {
	id := w.nextID.Add(1)
	msg, err := buildClientMsg(id, cmd, extra)
	if err != nil {
		return nil, err
	}

	ch := make(chan reply, 1)
	w.mu.Lock()
	w.waiters[id] = ch
	w.mu.Unlock()
	drop := func() {
		w.mu.Lock()
		delete(w.waiters, id)
		w.mu.Unlock()
	}

	// SAFETY: select on done so a dead worker doesn't block this
	// caller indefinitely on a full sendCh.
	select {
	case w.sendCh <- msg:
	case <-ctx.Done():
		drop()
		return nil, ctx.Err()
	case <-w.done:
		drop()
		return nil, errors.New("worker exited before send")
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
		return r.raw, nil
	}
}

// stop asks the worker to exit cleanly, closes the gRPC channel, then
// (for local subprocesses) waits for the process. Honors ctx deadline
// so server shutdown never hangs on an unresponsive worker.
func (w *grpcWorker) stop(ctx context.Context) error {
	// Best-effort graceful shutdown over the wire. Errors here are
	// ignored — we're about to tear the channel down either way.
	_, _ = w.send(ctx, "shutdown", nil)

	// Close send half to let the server's request iterator exit, then
	// cancel the stream context so recvLoop unblocks.
	close(w.sendCh)
	w.cancel()
	_ = w.conn.Close()

	// Remote-only: no subprocess to reap; we're done.
	if w.cmd == nil {
		return nil
	}

	exited := make(chan error, 1)
	go func() { exited <- w.cmd.Wait() }()

	select {
	case <-ctx.Done():
		_ = w.cmd.Process.Signal(syscall.SIGKILL)
		<-exited
		_ = removeIfPresent(w.sockPath)
		return ctx.Err()
	case err := <-exited:
		_ = removeIfPresent(w.sockPath)
		return err
	}
}

// setOnEvent is part of the workerHandle interface (see manager.go).
// The Manager calls this immediately after spawn to register its
// event handler.
func (w *grpcWorker) setOnEvent(fn func(map[string]any)) { w.onEvent = fn }

// removeIfPresent unlinks the socket file if it still exists. The
// kernel cleans up the socket binding when the Python process exits;
// the on-disk inode does not. Best effort — leftover sockets are a
// warning, not a failure.
func removeIfPresent(path string) error {
	if path == "" {
		return nil
	}
	err := syscallUnlink(path)
	if err != nil && !errors.Is(err, syscall.ENOENT) {
		return err
	}
	return nil
}

// syscallUnlink wraps the unlink syscall to keep the import surface in
// this file small. (os.Remove would also work; this is a stylistic nit
// to avoid pulling "os" purely for cleanup.)
func syscallUnlink(path string) error {
	// PERF: zero-allocation unlink. Not material at this call rate but
	// keeps the syscall path obvious.
	return syscall.Unlink(path)
}

// suppress unused-import warning during early scaffolding when stop
// might not yet route through ctx-deadline branches in tests. Once
// integration coverage exercises stop, this no-op goes away.
var _ = time.Second
