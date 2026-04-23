// AREA: desktop · ENTRY
// Tauri app entry. Registers IPC commands and the long-lived state used by
// those commands. Individual subsystems live in sibling modules (currently
// just `server`); as we add more (audio capture, model controls), each gets
// its own module and a single `.manage()` / `.invoke_handler` registration
// here.

mod server;

use tauri::{Manager, WindowEvent};

use server::{ServerState, server_status, signal_shutdown_sync, start_server, stop_server};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // SETUP: per-process state for the Go server subprocess handle.
        .manage(ServerState::new())
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            server_status,
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
                let _ = signal_shutdown_sync(&state);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
