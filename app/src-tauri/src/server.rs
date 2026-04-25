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
    // child is moved into the exit-watcher task once the server reaches
    // Running, so stop_server can't rely on it. Everything about *killing*
    // the server goes through pgid instead; child is here only to let us
    // wait()/kill() during startup or after spawn failure.
    child: Option<Child>,
    // pgid is the process group id we created via setsid at spawn time.
    // Stays Some as long as the group is alive so stop_server and the
    // window-close handler can signal it regardless of who owns Child.
    pgid: Option<i32>,
    status: ServerStatus,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                child: None,
                pgid: None,
                status: ServerStatus::Idle,
            })),
        }
    }

}

// STEP: resolve the absolute path of the Go server directory at compile time.
// CARGO_MANIFEST_DIR points at `app/src-tauri`; the server sits at
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

    // STEP 1a: defensive port cleanup. Hot reloads of `tauri dev`, an app
    // crash, or a Cargo rebuild that SIGKILLs the parent before the close
    // handler runs can leave a previous Go server still bound to 7842.
    // Without this the next start fails immediately with "address already
    // in use". We force-kill anything holding the port and give the OS a
    // moment to release the socket before spawning. Safe to run when the
    // port is free — lsof simply returns nothing.
    free_port_blocking(SERVER_PORT);
    sleep(Duration::from_millis(200)).await;

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

    // STEP 4: stash the handle and the process group id. We record pgid
    // separately from `child` because the exit watcher (spawned below once
    // startup succeeds) moves `child` out of state — without pgid being
    // independent, stop_server would have nothing left to signal.
    //
    // Because we called setsid() in pre_exec, the direct child's pid is
    // also its process group id, so `child.id()` doubles as the pgid.
    {
        let mut guard = state.inner.lock().await;
        #[cfg(unix)]
        {
            guard.pgid = child.id().map(|p| p as i32);
        }
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
                // The process is gone — clear pgid so neither stop_server
                // nor the close handler try to signal a stale group id
                // that the OS may have already recycled for someone else.
                g.pgid = None;
                // REASON: only emit Error if we're still in Running. If stop_server
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
            // Timed out. Kill the whole process group — same reasoning as
            // stop_server: child.kill() alone would only hit `go run`.
            let mut guard = state.inner.lock().await;
            let pgid = guard.pgid.take();
            let child_opt = guard.child.take();
            #[cfg(unix)]
            if let Some(pgid) = pgid {
                unsafe {
                    libc::killpg(pgid, libc::SIGKILL);
                }
            }
            if let Some(mut c) = child_opt {
                let _ = c.wait().await;
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

/// Stop the Go server. Sends SIGTERM to the server's process group, waits
/// briefly for clean exit, then SIGKILLs the group if the graceful path
/// didn't finish in time. Always signals via pgid so we kill `go run` *and*
/// its compiled-binary grandchild as one unit.
#[tauri::command]
pub async fn stop_server(app: AppHandle, state: State<'_, ServerState>) -> Result<(), String> {
    // STEP 1: transition to Stopping and snapshot pgid + any still-owned
    // child handle. We take pgid too so a concurrent window-close handler
    // doesn't try to signal a group we're already reaping.
    let (pgid, child_opt) = {
        let mut guard = state.inner.lock().await;
        if matches!(guard.status, ServerStatus::Idle) {
            return Ok(());
        }
        guard.status = ServerStatus::Stopping;
        (guard.pgid.take(), guard.child.take())
    };
    let _ = app.emit(STATUS_EVENT, ServerStatus::Stopping);

    // STEP 2: SIGTERM the whole process group. This is what actually shuts
    // down the listening server — `child.kill()` would only hit `go run`
    // and leave the compiled server binary running on port 7842.
    #[cfg(unix)]
    if let Some(pgid) = pgid {
        // SAFETY: killpg is a direct libc syscall; pgid is the group we
        // created via setsid at spawn time.
        unsafe {
            libc::killpg(pgid, libc::SIGTERM);
        }
    }

    // STEP 3: wait for graceful exit. If we still own the Child handle we
    // can await it directly; otherwise the watcher task has it and will
    // reap on its own — we just poll the port instead.
    let graceful = if let Some(mut child) = child_opt {
        timeout(STOP_GRACE, child.wait()).await.is_ok()
    } else {
        // REASON: no Child handle means the exit watcher owns it. We can't
        // await cross-task, so poll the port instead — once it's free the
        // server has released its listener, which is all we actually care
        // about for the "port in use" failure mode.
        timeout(STOP_GRACE, wait_for_port_free()).await.is_ok()
    };

    // STEP 4: if graceful exit didn't complete, SIGKILL the whole group.
    // Using killpg (not child.kill) ensures the compiled server binary
    // dies even when `go run` has already reaped its child.
    #[cfg(unix)]
    if !graceful {
        if let Some(pgid) = pgid {
            unsafe {
                libc::killpg(pgid, libc::SIGKILL);
            }
        }
    }

    let mut guard = state.inner.lock().await;
    guard.status = ServerStatus::Idle;
    let _ = app.emit(STATUS_EVENT, ServerStatus::Idle);
    Ok(())
}

/// Best-effort synchronous shutdown used by the window-close handler. Sends
/// SIGTERM then SIGKILL to the server's process group without awaiting —
/// the app is exiting and we just need the port released before Go
/// restarts. Returns Some(pgid) if a group was signalled, None otherwise.
pub fn signal_shutdown_sync(state: &ServerState) -> Option<i32> {
    // try_lock: if someone else has the lock we skip. The app is closing
    // anyway; blocking here would deadlock if the holder is awaiting on
    // something.
    let pgid = state.inner.try_lock().ok().and_then(|mut g| {
        let p = g.pgid.take();
        if p.is_some() {
            g.status = ServerStatus::Stopping;
        }
        p
    })?;

    #[cfg(unix)]
    unsafe {
        // SIGTERM first so the Go server runs its graceful shutdown (which
        // also tells the Python worker to exit). SIGKILL follows as a
        // backstop in case the graceful path hangs.
        libc::killpg(pgid, libc::SIGTERM);
    }
    Some(pgid)
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

// Force-release a TCP port by SIGKILLing any process holding it. Shells
// out to `lsof -ti tcp:<port>` (macOS / linux) which prints one pid per
// line of holders, then signals each. This is the brute-force backstop
// for stale go-server processes leaked by previous dev sessions; the
// graceful shutdown path in stop_server / signal_shutdown_sync handles
// the common case.
//
// SWAP: replace with a platform-specific implementation if we ever need
// Windows support — `netstat -ano | findstr :<port>` then `taskkill /F`.
fn free_port_blocking(port: u16) {
    let out = match std::process::Command::new("lsof")
        .args(["-ti", &format!("tcp:{port}")])
        .output()
    {
        Ok(o) => o,
        // lsof missing or unreachable. Nothing we can do here; the spawn
        // below will surface a clear EADDRINUSE if the port is actually
        // held.
        Err(_) => return,
    };
    if !out.status.success() {
        return;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    for pid_str in stdout.split_whitespace() {
        if let Ok(pid) = pid_str.parse::<i32>() {
            // SAFETY: kill is a direct libc syscall; pid came from lsof
            // and is at worst stale (kill returns ESRCH, harmless).
            #[cfg(unix)]
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
            eprintln!("[server] freed port {port} (killed pid {pid})");
        }
    }
}

// Inverse of wait_for_port: returns once the port is NOT accepting
// connections, i.e. the listener has released it. Used during stop_server
// when we no longer own the Child handle and can't await its exit.
async fn wait_for_port_free() {
    loop {
        if TcpStream::connect((SERVER_HOST, SERVER_PORT)).await.is_err() {
            return;
        }
        sleep(STARTUP_POLL).await;
    }
}

