// AREA: desktop · ENTRY
// Tauri app entry. Registers IPC commands and the long-lived state used by
// those commands. Individual subsystems live in sibling modules (currently
// `server` for the Go child and `remote` for SSH-based remote workers);
// as we add more (audio capture, model controls), each gets its own
// module and a single `.manage()` / `.invoke_handler` registration here.

mod remote;
mod server;

use tauri::{Manager, WindowEvent};

use remote::{
    RemoteState, delete_remote_profile, list_remote_profiles, save_remote_profile,
    set_active_profile,
};
use server::{ServerState, server_status, signal_shutdown_sync, start_server, stop_server};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // SETUP: per-process state for the Go server subprocess handle.
        .manage(ServerState::new())
        // SETUP: shared state for the active SSH tunnel + worker child
        // when a remote profile is selected. server.rs reads this at
        // start_server time to pick the URL + token to hand the Go child.
        .manage(RemoteState::new())
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            server_status,
            list_remote_profiles,
            save_remote_profile,
            delete_remote_profile,
            set_active_profile,
        ])
        // STEP: clean up the Go server on window close. Without this, quitting
        // the app (Cmd-Q, closing the window) leaves a detached `go run .`
        // process still bound to port 7842, which makes the *next* launch
        // fail with "address already in use".
        //
        // We signal synchronously (SIGTERM to the process group) so the Go
        // server's own shutdown path runs — that's what tears down the
        // Python worker and releases the listener cleanly. The app exits
        // immediately after; any further reaping is handled by the OS.
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<ServerState>();
                let remote = window.state::<RemoteState>();
                // STEP 1: SIGTERM the go server's process group so it can
                // run its graceful shutdown (which also tells the python
                // worker to exit).
                let pgid = signal_shutdown_sync(&state);
                // STEP 2: give the graceful path a beat to release the
                // listener, then SIGKILL the group as a backstop. Without
                // this, a slow shutdown (worker still draining, etc.) can
                // leave the port bound after the rust process is reaped —
                // the next launch then fails with EADDRINUSE. Blocking the
                // close handler briefly is fine; the user expects a small
                // pause on quit.
                if let Some(pgid) = pgid {
                    std::thread::sleep(std::time::Duration::from_millis(400));
                    #[cfg(unix)]
                    unsafe {
                        libc::killpg(pgid, libc::SIGKILL);
                    }
                }
                // STEP 3: best-effort kill of any active remote SSH
                // children. Same orphan-port risk: a leftover ssh -L
                // child would hold the local 7843 binding and block the
                // next launch's tunnel from coming up.
                remote::sync_kill_children(&remote);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
