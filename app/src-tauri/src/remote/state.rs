// AREA: desktop · REMOTE · STATE
// In-memory pieces of an active remote connection: the SSH tunnel child,
// the (Phase 2) worker child, and the bearer token we generated for this
// session. Everything else (profile list, active id) lives in
// profiles.rs's on-disk file.
//
// SAFETY: every field is "Some only while connected." Disconnect MUST
// clear them and reap children — leaving an orphan ssh child with a
// bound -L port would block the next connect attempt with EADDRINUSE.

use std::sync::Arc;

use tokio::process::Child;
use tokio::sync::Mutex;

/// Per-process remote-connection state. Wrapped in an Arc<Mutex<…>> so
/// commands and the cleanup path can both reach it.
#[derive(Default)]
pub struct RemoteState {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
pub struct Inner {
    /// `ssh -N -L …` child. Dropping this kills the tunnel.
    pub tunnel: Option<Child>,
    /// (Phase 2) `ssh user@host "python main.py"` child. Dropping this
    /// signals SIGHUP to the python process via the SSH session.
    pub worker: Option<Child>,
    /// The bearer token we generated for this connect — set as
    /// CYPRESS_REMOTE_TOKEN on the Go server's env, and (Phase 2) passed
    /// to the python worker via --token. None when local mode.
    pub token: Option<String>,
    /// The local port the tunnel is forwarding from. We hand this to the
    /// Go server as the URL's port. Today always 7843 to match the
    /// hard-coded default; kept as a field so a future "two remotes
    /// stacked at once" feature has somewhere to plug in.
    pub local_port: Option<u16>,
}

impl RemoteState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shared(&self) -> Arc<Mutex<Inner>> {
        self.inner.clone()
    }
}
