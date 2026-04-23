// AREA: desktop · ENTRY
// Tauri app entry. Registers IPC commands and the long-lived state used by
// those commands. Individual subsystems live in sibling modules (currently
// just `server`); as we add more (audio capture, model controls), each gets
// its own module and a single `.manage()` / `.invoke_handler` registration
// here.

mod server;

use server::{ServerState, server_status, start_server, stop_server};

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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
