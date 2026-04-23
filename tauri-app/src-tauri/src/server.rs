// AREA: desktop · IPC · SERVER
// Manages the Go orchestration server subprocess from the Tauri Rust shell.
// In dev we spawn `go run .` against the sibling `server/` directory; in a
// release build this will be replaced by a bundled sidecar binary.
//
// SWAP: the spawn target. Today `go run .` (fast iteration, no build step);
// later a prebuilt binary resolved via tauri's sidecar API. The public
// command surface (start/stop/status) stays the same either way.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};

// SETUP: port must match server/main.go's `listenAddr`. If you change it
// there, change it here. Kept as a plain const (rather than config) because
// it's a hard-coded local dev address — no reason to surface it.
const SERVER_HOST: &str = "127.0.0.1";
const SERVER_PORT: u16 = 7842;

// How long we wait for the server to become reachable after spawning.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);
// Poll interval while waiting for the port to come up.
const STARTUP_POLL: Duration = Duration::from_millis(250);
// Grace period between SIGTERM and force-kill when stopping.
const STOP_GRACE: Duration = Duration::from_secs(3);

// Name of the Tauri event we emit whenever status changes. React listens for
// this instead of polling.
const STATUS_EVENT: &str = "server-status";

/// Lifecycle of the Go server subprocess as seen by the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum ServerStatus {
    Idle,
    Starting,
    Running,
    Stopping,
    Error { message: String },
}

/// Shared Tauri state. The Mutex guards both the child handle and the
/// currently-advertised status so they can't drift out of sync.
pub struct ServerState {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    child: Option<Child>,
    status: ServerStatus,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                child: None,
                status: ServerStatus::Idle,
            })),
        }
    }
}

// STEP: resolve the absolute path of the Go server directory at compile time.
// CARGO_MANIFEST_DIR points at `tauri-app/src-tauri`; the server sits at
// `../../server` relative to that. Resolving once at build time means the
// dev-mode path can't drift based on where the binary is launched from.
fn server_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("server")
}

// Emit a status change to the frontend and cache it on state. All transitions
// must go through this so UI and state stay consistent.
async fn set_status(app: &AppHandle, inner: &Mutex<Inner>, status: ServerStatus) {
    let mut guard = inner.lock().await;
    guard.status = status.clone();
    let _ = app.emit(STATUS_EVENT, status);
}

/// Ask Tauri to start the Go server. Idempotent: if it's already starting or
/// running, returns immediately. Otherwise spawns `go run .` in the server
/// directory, waits until the health port is reachable, and emits status
/// transitions along the way.
#[tauri::command]
pub async fn start_server(app: AppHandle, state: State<'_, ServerState>) -> Result<(), String> {
    // STEP 1: fast-path check. Grab the lock only long enough to inspect and
    // flip to Starting — we don't want to hold it across the spawn.
    {
        let mut guard = state.inner.lock().await;
        match guard.status {
            ServerStatus::Running | ServerStatus::Starting => return Ok(()),
            _ => {}
        }
        guard.status = ServerStatus::Starting;
    }
    let _ = app.emit(STATUS_EVENT, ServerStatus::Starting);

    // STEP 2: spawn `go run .` with stdout/stderr piped so we can log it.
    // We create a new process group (on unix) so that when we later kill the
    // process, the go toolchain's child binary gets cleaned up too — without
    // this, `go run` leaks its grandchild.
    let mut cmd = Command::new("go");
    cmd.current_dir(server_dir())
        .arg("run")
        .arg(".")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    #[cfg(unix)]
    {
        // SAFETY: setting a new process group is a direct syscall; no Rust
        // invariants are touched. The child will be killable via its group id.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    let mut child = cmd.spawn().map_err(|e| {
        let msg = format!("failed to spawn go server: {e}");
        // STEP 2a: if spawn fails, roll status back to Error so UI unsticks.
        let inner = state.inner.clone();
        let app2 = app.clone();
        let emsg = msg.clone();
        tauri::async_runtime::spawn(async move {
            set_status(&app2, &inner, ServerStatus::Error { message: emsg }).await;
        });
        msg
    })?;

    // STEP 3: pipe subprocess stdout/stderr to our logs. Tauri's stdout in
    // dev lands in the terminal running `tauri dev`, so this is where the Go
    // server's logs show up during development.
    if let Some(out) = child.stdout.take() {
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("[go-server] {line}");
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[go-server] {line}");
            }
        });
    }

    // STEP 4: stash the handle so stop_server can find it later.
    {
        let mut guard = state.inner.lock().await;
        guard.child = Some(child);
    }

    // STEP 5: poll the health port until it accepts a TCP connection. If we
    // time out, the server failed to start (bad Go install, port conflict,
    // compile error in the Go source). In any of those cases we kill the
    // child and surface an error to the UI.
    let wait = timeout(STARTUP_TIMEOUT, wait_for_port()).await;

    match wait {
        Ok(()) => {
            set_status(&app, &state.inner, ServerStatus::Running).await;
            // STEP 6: watch for unexpected exit. If the child dies while we
            // think it's running, flip state to Error so the UI can recover.
            let inner = state.inner.clone();
            let app2 = app.clone();
            tauri::async_runtime::spawn(async move {
                // Take the child handle out so we can own its lifetime here.
                let child_opt = {
                    let mut g = inner.lock().await;
                    g.child.take()
                };
                let Some(mut child) = child_opt else { return };
                let exit = child.wait().await;
                let mut g = inner.lock().await;
                // WHY: only emit Error if we're still in Running. If stop_server
                // flipped us to Stopping already, the exit is expected.
                if matches!(g.status, ServerStatus::Running) {
                    let msg = match exit {
                        Ok(s) => format!("server exited unexpectedly: {s}"),
                        Err(e) => format!("server wait failed: {e}"),
                    };
                    g.status = ServerStatus::Error {
                        message: msg.clone(),
                    };
                    let _ = app2.emit(STATUS_EVENT, ServerStatus::Error { message: msg });
                } else {
                    g.status = ServerStatus::Idle;
                    let _ = app2.emit(STATUS_EVENT, ServerStatus::Idle);
                }
            });
            Ok(())
        }
        Err(_) => {
            // Timed out. Kill whatever we spawned.
            let mut guard = state.inner.lock().await;
            if let Some(mut c) = guard.child.take() {
                let _ = c.kill().await;
            }
            guard.status = ServerStatus::Error {
                message: "server did not become reachable before timeout".into(),
            };
            let msg = "server did not become reachable before timeout".to_string();
            let _ = app.emit(
                STATUS_EVENT,
                ServerStatus::Error {
                    message: msg.clone(),
                },
            );
            Err(msg)
        }
    }
}

/// Stop the Go server. Sends a termination signal and waits briefly for the
/// process to exit cleanly; force-kills if it doesn't.
#[tauri::command]
pub async fn stop_server(app: AppHandle, state: State<'_, ServerState>) -> Result<(), String> {
    // STEP 1: transition to Stopping and pull out the child handle.
    let child_opt = {
        let mut guard = state.inner.lock().await;
        if matches!(guard.status, ServerStatus::Idle) {
            return Ok(());
        }
        guard.status = ServerStatus::Stopping;
        guard.child.take()
    };
    let _ = app.emit(STATUS_EVENT, ServerStatus::Stopping);

    // STEP 2: there may be no child if the startup watcher already reaped it.
    // Either way, flip back to Idle.
    if let Some(mut child) = child_opt {
        // SAFETY: `id()` can return None if the process already exited. We
        // only signal if we still have a pid.
        #[cfg(unix)]
        if let Some(pid) = child.id() {
            // Signal the entire process group we created in start_server.
            // Negative pid = process group on Unix.
            // SAFETY: killpg is a direct syscall; caller just needs a valid pid.
            unsafe {
                libc::killpg(pid as i32, libc::SIGTERM);
            }
        }

        // STEP 3: wait up to STOP_GRACE for a clean exit; otherwise force-kill.
        match timeout(STOP_GRACE, child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        }
    }

    let mut guard = state.inner.lock().await;
    guard.status = ServerStatus::Idle;
    let _ = app.emit(STATUS_EVENT, ServerStatus::Idle);
    Ok(())
}

/// Read-only: returns the current advertised status. The UI also listens for
/// the `server-status` event, so this is mostly used for initial hydration.
#[tauri::command]
pub async fn server_status(state: State<'_, ServerState>) -> Result<ServerStatus, String> {
    Ok(state.inner.lock().await.status.clone())
}

// PERF: opens a short-lived TCP connection to the health port. We intentionally
// don't issue an HTTP GET here — reaching accept() is a sufficient liveness
// signal for v0.1 and avoids pulling in an HTTP client dependency.
async fn wait_for_port() {
    // Outer loop keeps polling until the connection succeeds.
    loop {
        if TcpStream::connect((SERVER_HOST, SERVER_PORT)).await.is_ok() {
            return;
        }
        sleep(STARTUP_POLL).await;
    }
}

