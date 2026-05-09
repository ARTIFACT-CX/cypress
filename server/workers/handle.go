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
	// Done returns a channel that closes when the underlying transport
	// terminates — local subprocess exit, remote stream EOF, or
	// transport-level error. Owners (inference.Manager) watch this to
	// drive auto-reconnect. Stop closes the channel as part of normal
	// teardown, so watchers must check whether the drop was expected
	// before reacting.
	Done() <-chan struct{}
	// Platform returns the worker's reported platform snapshot, captured
	// from the gRPC Handshake at session open. Empty fields mean "the
	// worker didn't report" (older worker, fake test handle); callers
	// should fall back to the host's runtime in that case.
	Platform() Platform
	// Disconnect closes the gRPC channel without telling the worker to
	// shut down. Used by the eager platform probe — the host reads the
	// handshake, then drops the session so subsequent LoadModel /
	// Download dials get a fresh stream against the same worker
	// process. Stop, by contrast, sends a `shutdown` RPC that kills
	// the Python process; using it from the probe leaves nothing for
	// the next dial to connect to. For local subprocess workers this
	// also leaves the process running, so callers must follow up with
	// Stop later if they want it reaped.
	Disconnect() error
}

// Platform is the snapshot the worker stamps onto every Handshake. The
// Go host uses these fields to pick model variants — without them, an
// Apple-Silicon laptop dialing a Linux GPU worker would tell it to
// download MLX weights it can't load. OS / Arch follow Go's runtime
// naming ("linux"/"darwin", "amd64"/"arm64") so the host can compare
// strings without a translation table. GPUName / GPUMemoryGB are
// diagnostic — surfaced in the server-details popup so operators can
// confirm RunPod / SLURM gave them the box they actually requested.
// Both empty when the worker can't tell.
type Platform struct {
	OS                string
	Arch              string
	AvailableBackends []string
	DownloadedRepos   []string
	GPUName           string
	GPUMemoryGB       uint32
}
