// AREA: desktop · REMOTE
// Public surface of the remote-worker subsystem: profile CRUD commands,
// connect/disconnect orchestration, and the shared state types other
// Tauri modules consume (server.rs reads `RemoteState` to decide how to
// spawn the Go child).
//
// The flow when the user clicks Start Server with a remote profile active:
//
//   1. ssh::generate_token()
//   2. ssh::spawn_tunnel + ssh::wait_for_tunnel  → local port is live
//   3. (caller in server.rs) spawn Go server with CYPRESS_REMOTE_URL +
//      CYPRESS_REMOTE_TOKEN env from RemoteState
//   4. (Phase 2) ssh::spawn_worker on the same profile + token
//
// Disconnect reverses 4 → 1, killing children in the opposite order so
// the Go server stops dialing before the tunnel goes away (otherwise
// reachability flips false twice).

pub mod profiles;
pub mod ssh;
pub mod state;

use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Manager};

pub use state::RemoteState;

use profiles::RemoteProfile;

/// How long we wait for the remote python worker to bind its gRPC
/// port after we kicked it off. Generous — a fresh venv on a cold
/// node imports torch + moshi, which can take ~20s; subsequent loads
/// land in 1-2.
const WORKER_READY_TIMEOUT_SECS: u64 = 60;

/// Wrapper returned by list_profiles — bundles the profile array and
/// the active id in one round-trip so the UI doesn't have to fetch
/// twice. `active_id == None` is the "Local subprocess" mode.
#[derive(Debug, Serialize)]
pub struct ProfilesView {
    pub profiles: Vec<RemoteProfile>,
    pub active_id: Option<String>,
}

/// Resolve the platform's app-data dir via Tauri's path resolver. We
/// thread the AppHandle in (rather than caching) because Tauri only
/// finalizes the path after the app is fully built, and our state
/// constructor runs before that.
fn data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|e| format!("could not resolve app data dir: {e}"))
}

#[tauri::command]
pub async fn list_remote_profiles(app: AppHandle) -> Result<ProfilesView, String> {
    let dir = data_dir(&app)?;
    let file = profiles::load(&dir).map_err(|e| format!("load profiles: {e}"))?;
    Ok(ProfilesView {
        profiles: file.profiles,
        active_id: file.active_id,
    })
}

#[tauri::command]
pub async fn save_remote_profile(
    app: AppHandle,
    mut profile: RemoteProfile,
) -> Result<RemoteProfile, String> {
    let dir = data_dir(&app)?;
    let mut file = profiles::load(&dir).map_err(|e| format!("load profiles: {e}"))?;

    // REASON: empty id from the UI means "this is a new profile" —
    // mint a fresh uuid. Any existing id is honored as-is so edits
    // preserve the active-pointer + manifest references.
    if profile.id.is_empty() {
        profile.id = profiles::new_id();
    }

    if let Some(existing) = file.profiles.iter_mut().find(|p| p.id == profile.id) {
        *existing = profile.clone();
    } else {
        file.profiles.push(profile.clone());
    }

    profiles::save(&dir, &file).map_err(|e| format!("save profiles: {e}"))?;
    Ok(profile)
}

#[tauri::command]
pub async fn delete_remote_profile(app: AppHandle, id: String) -> Result<(), String> {
    let dir = data_dir(&app)?;
    let mut file = profiles::load(&dir).map_err(|e| format!("load profiles: {e}"))?;

    let before = file.profiles.len();
    file.profiles.retain(|p| p.id != id);
    if file.profiles.len() == before {
        return Err(format!("no profile with id {id}"));
    }

    // SAFETY: clearing the active pointer if it referenced the deleted
    // profile prevents a dangling id surviving across launches. The user
    // falls back to local mode after the next read.
    if file.active_id.as_ref() == Some(&id) {
        file.active_id = None;
    }

    profiles::save(&dir, &file).map_err(|e| format!("save profiles: {e}"))
}

/// Set the active profile pointer. `None` (or null from JS) means
/// "Local subprocess." Validates the id exists; an unknown id errors
/// rather than silently no-oping so a typo from the UI surfaces.
#[tauri::command]
pub async fn set_active_profile(app: AppHandle, id: Option<String>) -> Result<(), String> {
    let dir = data_dir(&app)?;
    let mut file = profiles::load(&dir).map_err(|e| format!("load profiles: {e}"))?;

    if let Some(ref want) = id {
        if !file.profiles.iter().any(|p| &p.id == want) {
            return Err(format!("no profile with id {want}"));
        }
    }

    file.active_id = id;
    profiles::save(&dir, &file).map_err(|e| format!("save profiles: {e}"))
}

/// Returns the currently-active profile, or None for local mode. Used
/// by server.rs at start_server time to decide whether to bring up the
/// SSH tunnel before spawning Go.
pub async fn active_profile(app: &AppHandle) -> Result<Option<RemoteProfile>, String> {
    let dir = data_dir(app)?;
    let file = profiles::load(&dir).map_err(|e| format!("load profiles: {e}"))?;
    let Some(id) = file.active_id else {
        return Ok(None);
    };
    Ok(file.profiles.into_iter().find(|p| p.id == id))
}

/// Bring up the SSH tunnel for the supplied profile and stash the child
/// + token in RemoteState. Returns the (local_port, token) the caller
/// uses to point the Go server at this remote.
///
/// Caller must have stopped any previous remote connection first —
/// this function bails if RemoteState already has a tunnel installed,
/// rather than silently kicking out the old one.
pub async fn connect(
    app: &AppHandle,
    profile: &RemoteProfile,
    state: &RemoteState,
) -> Result<(u16, String), String> {
    // REASON: bind shared() to a local before locking — the Arc has to
    // outlive the guard, otherwise the temporary is dropped at the end
    // of the statement and the lock guard refers to freed memory.
    let shared = state.shared();
    let mut guard = shared.lock().await;
    if guard.tunnel.is_some() {
        return Err("a remote connection is already active; disconnect first".into());
    }

    // STEP 1: token + tunnel. We use the same port locally as the
    // remote binds (7843 — matches the hard-coded default elsewhere).
    let token = ssh::generate_token();
    let local_port: u16 = profile.port; // 1:1 mapping for now

    let mut tunnel = ssh::spawn_tunnel(profile, local_port).await?;
    if let Err(e) = ssh::wait_for_tunnel(&mut tunnel, local_port).await {
        // SAFETY: kill_on_drop covers the child if we let it fall out
        // of scope, but explicit start_kill makes the cleanup happen
        // immediately rather than at the next yield point — we'd rather
        // surface the failure with no orphan than reap on drop and have
        // the next connect see a still-bound port.
        let _ = tunnel.start_kill();
        return Err(e);
    }

    // STEP 2 (Phase 2): launch the worker process on the remote box
    // via a second SSH session and wait for it to bind. If the worker
    // fails to come up, tear down the tunnel too so we don't leave
    // an orphan local listener pointing at nothing.
    let worker = match ssh::spawn_worker(app, profile, &token).await {
        Ok(w) => w,
        Err(e) => {
            let _ = tunnel.start_kill();
            return Err(e);
        }
    };
    if let Err(e) = ssh::wait_for_worker(local_port, WORKER_READY_TIMEOUT_SECS).await {
        // worker is in `worker_holder` — drop kills it. Tunnel handled
        // explicitly so the port releases immediately, otherwise a
        // tight retry loop would hit EADDRINUSE.
        drop(worker);
        let _ = tunnel.start_kill();
        return Err(format!("remote worker: {e}"));
    }

    guard.tunnel = Some(tunnel);
    guard.worker = Some(worker);
    guard.token = Some(token.clone());
    guard.local_port = Some(local_port);

    Ok((local_port, token))
}

/// Tear down whatever's in RemoteState. Idempotent — empty state means
/// we're in local mode and there's nothing to do.
///
/// Takes the AppHandle + active profile to issue an explicit SSH-kill
/// of the remote worker process. We can't rely on SIGHUP from closing
/// the worker SSH child reaching the python process — moshi's CUDA
/// context holding behavior, combined with the way bash -lc forks,
/// leaves orphans that survive the SSH client teardown. See #47.
pub async fn disconnect(app: &AppHandle, state: &RemoteState) {
    let shared = state.shared();

    // STEP 1: explicit remote kill. Issue this BEFORE killing the
    // tunnel so the SSH path is still up; tearing the tunnel first
    // would force this to dial a fresh control channel through the
    // jump host, which is slower and may fail if the user's
    // allocation just expired.
    //
    // We re-look-up the profile rather than caching it on RemoteState
    // — a profile edit between connect and disconnect should still
    // dispatch the kill to the right host. Failures here don't block
    // the rest of the teardown.
    if let Ok(Some(profile)) = active_profile(app).await {
        // Only relevant if there's actually a worker; skip the SSH
        // round-trip when state is empty.
        let needs_kill = {
            let guard = shared.lock().await;
            guard.worker.is_some()
        };
        if needs_kill {
            if let Err(e) = ssh::kill_remote_workers(&profile).await {
                eprintln!("[remote] kill_remote_workers: {e} (continuing teardown)");
            }
        }
    }

    let mut guard = shared.lock().await;
    // STEP 2: reap the local SSH children. Worker first so its closing
    // doesn't fight with the still-up tunnel; then the tunnel.
    if let Some(mut w) = guard.worker.take() {
        let _ = w.start_kill();
        let _ = w.wait().await;
    }
    if let Some(mut t) = guard.tunnel.take() {
        let _ = t.start_kill();
        let _ = t.wait().await;
    }
    guard.token = None;
    guard.local_port = None;
}

/// Synchronous best-effort SIGKILL of the SSH children. Used by the
/// window-close handler — it runs in a sync context and the app is
/// exiting, so we can't (and don't need to) do the polite shutdown
/// dance. We pull the inner under try_lock to avoid deadlocking if
/// another task happens to be holding the lock; missing the kill is
/// strictly better than hanging the close.
pub fn sync_kill_children(state: &RemoteState) {
    let shared = state.shared();
    let Ok(mut guard) = shared.try_lock() else {
        return;
    };
    if let Some(child) = guard.worker.as_mut() {
        if let Some(pid) = child.id() {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
    }
    if let Some(child) = guard.tunnel.as_mut() {
        if let Some(pid) = child.id() {
            #[cfg(unix)]
            unsafe {
                // SAFETY: tunnel was spawned with setsid so pid == pgid;
                // killpg sweeps any ProxyCommand / ssh-agent grandchildren
                // along with the parent ssh.
                libc::killpg(pid as i32, libc::SIGKILL);
            }
        }
    }
}
