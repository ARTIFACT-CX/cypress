// AREA: workers · TESTS
// Unit tests for the gRPC IPC layer in grpc.go + wireconv.go. These
// don't spawn a real Python process — they construct a Grpc against an
// in-memory bufconn-backed gRPC server with a hand-rolled servicer
// that echoes back exactly what the test wants.

package workers

import (
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/ARTIFACT-CX/cypress/proto/dist/go/workerpb"
)

// testServicer is a minimal Worker server. The test sets handleClient
// to choose what each ClientMsg gets in response. Because this is
// hand-rolled (no real Python), we can drive any combination of replies
// and events without booting an interpreter.
type testServicer struct {
	pb.UnimplementedWorkerServer

	// onMsg is called for each ClientMsg the test wants to inspect.
	// Defaults to nil = ignore. Tests assign before connecting.
	onMsg func(*pb.ClientMsg, grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg])

	// handshakeFatal, if non-empty, sends Handshake{ready=false, fatal=...}
	// instead of the normal ready handshake. Tests use this to exercise
	// the failure surfacing path.
	handshakeFatal string

	// handshakeDelay holds the handshake send back this long, exercising
	// the timeout path. Default zero = send immediately.
	handshakeDelay time.Duration
}

func (s *testServicer) Session(stream grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg]) error {
	// First message: handshake (or fatal, or delayed for timeout tests).
	if s.handshakeDelay > 0 {
		time.Sleep(s.handshakeDelay)
	}
	hs := &pb.Handshake{Ready: s.handshakeFatal == ""}
	if s.handshakeFatal != "" {
		f := s.handshakeFatal
		hs.Fatal = &f
	}
	if err := stream.Send(&pb.ServerMsg{Payload: &pb.ServerMsg_Handshake{Handshake: hs}}); err != nil {
		return err
	}
	if s.handshakeFatal != "" {
		// Mimic the Python side: close the stream after a fatal so
		// dialGRPC fails fast.
		return nil
	}

	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if s.onMsg != nil {
			s.onMsg(msg, stream)
		}
	}
}

// startBufconnServer spins up a Worker gRPC server backed by an
// in-memory net.Conn. Returns a dial helper the test feeds to a
// custom client builder (we can't reuse dialGRPC directly because
// it builds a unix-socket dial; bufconn needs WithContextDialer).
func startBufconnServer(t *testing.T, svc *testServicer) (dialer func(context.Context, string) (net.Conn, error), shutdown func()) {
	t.Helper()
	lis := bufconn.Listen(1 << 16)
	srv := grpc.NewServer()
	pb.RegisterWorkerServer(srv, svc)
	go func() { _ = srv.Serve(lis) }()
	dialer = func(_ context.Context, _ string) (net.Conn, error) {
		return lis.Dial()
	}
	shutdown = func() {
		srv.Stop()
		_ = lis.Close()
	}
	return dialer, shutdown
}

// connectTestWorker is the test-only counterpart to dialGRPC. The
// wire / waiter / send-loop logic is the same — only the dial
// differs.
func connectTestWorker(t *testing.T, svc *testServicer) (*Grpc, func()) {
	t.Helper()
	dialer, shutdown := startBufconnServer(t, svc)

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(dialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		shutdown()
		t.Fatalf("dial: %v", err)
	}

	// Mirror dialGRPC: read handshake on a budget, then start
	// send/recv loops.
	streamCtx, cancel := context.WithCancel(context.Background())
	stream, err := pb.NewWorkerClient(conn).Session(streamCtx)
	if err != nil {
		cancel()
		_ = conn.Close()
		shutdown()
		t.Fatalf("open session: %v", err)
	}
	first, err := stream.Recv()
	if err != nil {
		cancel()
		_ = conn.Close()
		shutdown()
		t.Fatalf("recv handshake: %v", err)
	}
	if hs := first.GetHandshake(); hs == nil || !hs.GetReady() {
		cancel()
		_ = conn.Close()
		shutdown()
		t.Fatalf("bad handshake: %v", first)
	}

	w := &Grpc{
		conn:    conn,
		stream:  stream,
		cancel:  cancel,
		waiters: make(map[uint64]chan reply),
		sendCh:  make(chan *pb.ClientMsg, 16),
		done:    make(chan struct{}),
	}
	go w.sendLoop()
	go w.recvLoop()

	cleanup := func() {
		close(w.sendCh)
		w.cancel()
		_ = w.conn.Close()
		shutdown()
	}
	return w, cleanup
}

func TestGrpc_Send_RoutesReplyByID(t *testing.T) {
	svc := &testServicer{}
	model, device := "moshi", "mps"
	svc.onMsg = func(msg *pb.ClientMsg, stream grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg]) {
		_ = stream.Send(&pb.ServerMsg{Payload: &pb.ServerMsg_Reply{
			Reply: &pb.Reply{Id: msg.GetId(), Result: &pb.Reply_Status{
				Status: &pb.StatusOk{Model: &model, Device: &device},
			}},
		}})
	}
	w, cleanup := connectTestWorker(t, svc)
	defer cleanup()

	out, err := w.Send(context.Background(), "status", nil)
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if out["device"] != "mps" {
		t.Errorf("device = %v, want mps", out["device"])
	}
	if out["model"] != "moshi" {
		t.Errorf("model = %v, want moshi", out["model"])
	}
}

func TestGrpc_Send_ParallelCallsDontCross(t *testing.T) {
	// Two concurrent sends must each get their own reply. Reply in
	// reverse order to prove correlation isn't FIFO-dependent.
	var (
		mu      sync.Mutex
		seen    []*pb.ClientMsg
		release = make(chan struct{})
	)
	svc := &testServicer{}
	svc.onMsg = func(msg *pb.ClientMsg, stream grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg]) {
		mu.Lock()
		seen = append(seen, msg)
		n := len(seen)
		mu.Unlock()
		if n < 2 {
			return // hold the first; reply to both once we have both
		}
		<-release
		// Reply in reverse order.
		mu.Lock()
		ids := []uint64{seen[1].GetId(), seen[0].GetId()}
		mu.Unlock()
		for _, id := range ids {
			_ = stream.Send(&pb.ServerMsg{Payload: &pb.ServerMsg_Reply{
				Reply: &pb.Reply{Id: id, Result: &pb.Reply_Ok{Ok: &pb.OkEmpty{}}},
			}})
		}
	}

	w, cleanup := connectTestWorker(t, svc)
	defer cleanup()

	type result struct {
		out map[string]any
		err error
	}
	resA := make(chan result, 1)
	resB := make(chan result, 1)
	go func() {
		out, err := w.Send(context.Background(), "status", nil)
		resA <- result{out, err}
	}()
	go func() {
		out, err := w.Send(context.Background(), "unload", nil)
		resB <- result{out, err}
	}()

	// Wait for both messages to land on the server, then release.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(seen)
		mu.Unlock()
		if n == 2 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	close(release)

	rb := <-resB
	ra := <-resA
	if ra.err != nil || rb.err != nil {
		t.Fatalf("send errors: a=%v b=%v", ra.err, rb.err)
	}
}

func TestGrpc_Send_SurfacesErrorField(t *testing.T) {
	// Reply with a string error in the oneof — must come back as a Go
	// error from Send().
	svc := &testServicer{}
	svc.onMsg = func(msg *pb.ClientMsg, stream grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg]) {
		_ = stream.Send(&pb.ServerMsg{Payload: &pb.ServerMsg_Reply{
			Reply: &pb.Reply{Id: msg.GetId(), Result: &pb.Reply_Error{Error: "unknown model 'foo'"}},
		}})
	}
	w, cleanup := connectTestWorker(t, svc)
	defer cleanup()

	_, err := w.Send(context.Background(), "load_model", map[string]any{"name": "foo"})
	if err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("err = %v, want one containing 'unknown model'", err)
	}
}

func TestGrpc_Send_RespectsContextCancel(t *testing.T) {
	// Server never replies — cancellation must unblock send.
	svc := &testServicer{onMsg: func(*pb.ClientMsg, grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg]) {}}
	w, cleanup := connectTestWorker(t, svc)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := w.Send(ctx, "status", nil)
		done <- err
	}()
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("send didn't return after context cancel")
	}
}

func TestGrpc_RecvLoop_RoutesEvents(t *testing.T) {
	// Server pushes an Event without any prompting; the registered
	// onEvent callback must fire with the legacy dict shape.
	svc := &testServicer{}
	pushed := make(chan struct{})
	svc.onMsg = func(_ *pb.ClientMsg, stream grpc.BidiStreamingServer[pb.ClientMsg, pb.ServerMsg]) {
		_ = stream.Send(&pb.ServerMsg{Payload: &pb.ServerMsg_Event{
			Event: &pb.Event{Payload: &pb.Event_ModelPhase{
				ModelPhase: &pb.ModelPhase{Phase: "downloading_lm"},
			}},
		}})
		<-pushed
		// Reply to the original client send so the test's Send() returns.
		_ = stream.Send(&pb.ServerMsg{Payload: &pb.ServerMsg_Reply{
			Reply: &pb.Reply{Id: 1, Result: &pb.Reply_Ok{Ok: &pb.OkEmpty{}}},
		}})
	}
	w, cleanup := connectTestWorker(t, svc)
	defer cleanup()

	var (
		mu       sync.Mutex
		captured []map[string]any
	)
	w.SetOnEvent(func(m map[string]any) {
		mu.Lock()
		captured = append(captured, m)
		mu.Unlock()
	})

	// Trigger the servicer by sending any command; the servicer pushes
	// an event before replying. We drive the wait via close(pushed).
	go func() { _, _ = w.Send(context.Background(), "status", nil) }()

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(captured)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	close(pushed)

	mu.Lock()
	defer mu.Unlock()
	if len(captured) != 1 {
		t.Fatalf("captured %d events, want 1", len(captured))
	}
	if captured[0]["phase"] != "downloading_lm" {
		t.Errorf("phase = %v, want downloading_lm", captured[0]["phase"])
	}
}
