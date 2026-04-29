// AREA: workers · HANDLE
// The interface every worker transport satisfies. Decouples the
// orchestration code (inference.Manager, downloads.Service) from
// whether the worker is a local subprocess (Grpc over Unix socket)
// or a remote endpoint (Grpc over TCP+TLS, future). Tests inject
// in-memory fakes against the same shape.
//
// SWAP: this is the seam between the host-side state machines and
// the outbound subprocess / network adapter. Anything that takes a
// Handle should be transport-agnostic.

package workers

import "context"

// Handle is the live worker handle. send/stop are the two RPCs every
// caller needs; setOnEvent registers a callback for unsolicited
// worker→host messages (model phase, audio out, download progress).
//
// The map[string]any contract on send/setOnEvent is a deliberate
// lowest-common-denominator: the typed gRPC schema lives behind
// wireconv, but lifting it through the interface would force every
// orchestration callsite onto typed replies in one go. Future cleanup.
type Handle interface {
	Send(ctx context.Context, cmd string, extra map[string]any) (map[string]any, error)
	Stop(ctx context.Context) error
	SetOnEvent(fn func(map[string]any))
}
