// AREA: desktop · REMOTE · PROFILES
// Persistent list of remote-worker profiles plus the id of the one that's
// currently "active" (or None for local subprocess mode). Stored as JSON
// under the platform's app-data dir (e.g. ~/Library/Application
// Support/com.cypress.app/profiles.json on macOS) so the user keeps their
// remotes between launches without having to retype them.
//
// SAFETY: file is written with 0600 perms because the SSH key path lives
// here. The token doesn't — that's regenerated per-connect and never
// touches disk.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// One saved remote-worker profile. Field shapes mirror what the SSH
/// command line wants — host/user/jump_host/jump_user map directly into
/// `ssh -J jump_user@jump_host user@host`. Defaults applied at
/// construction so a partially-filled profile from the UI still produces
/// a runnable command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteProfile {
    /// Stable id. Survives rename so the active-profile pointer doesn't
    /// dangle when the user edits the display name.
    pub id: String,
    pub name: String,
    pub host: String,
    pub user: String,
    /// Optional ProxyJump host (HPC clusters need this — direct SSH to
    /// compute nodes is blocked, you tunnel through the login node).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jump_host: Option<String>,
    /// Defaults to `user` if not set; HPC users sometimes have different
    /// usernames on the login vs compute nodes (rare but real).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jump_user: Option<String>,
    /// Worker's gRPC port on the remote box. We forward localhost to it
    /// over SSH; the worker itself binds 127.0.0.1:<port> and only sees
    /// loopback.
    #[serde(default = "default_port")]
    pub port: u16,
    /// Path to a private key on the laptop. Empty ⇒ ssh's default search
    /// (~/.ssh/id_ed25519, id_rsa, etc.) handles it. We pass
    /// `IdentitiesOnly=yes` when set so a noisy ~/.ssh/config doesn't
    /// silently substitute a different key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_path: Option<String>,
    /// Absolute path to the worker checkout on the remote box. Phase 2
    /// uses this to launch the worker process.
    pub worker_dir: String,
    /// Model family this profile serves. Today only "moshi"; future
    /// profiles for personaplex / others land here.
    pub family: String,
}

fn default_port() -> u16 {
    7843
}

/// On-disk shape. We keep the active-profile pointer in the same file as
/// the profiles themselves so a single fs::write is atomic for both —
/// no chance of saving a profile and crashing before the active id lands.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProfilesFile {
    #[serde(default)]
    pub profiles: Vec<RemoteProfile>,
    /// None = active profile is "Local subprocess" (no SSH).
    #[serde(default)]
    pub active_id: Option<String>,
}

/// Compute the JSON path inside the app-data dir, creating any missing
/// parent directories. Caller passes the dir resolved by Tauri so this
/// module stays free of any tauri:: imports.
pub fn profiles_path(app_data_dir: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(app_data_dir)?;
    Ok(app_data_dir.join("profiles.json"))
}

/// Load profiles + active id. A missing file is normal first-run state —
/// we return Default rather than erroring so the UI just shows an empty
/// list. Malformed JSON DOES error so the user notices a corrupted state
/// rather than silently losing all their remotes.
pub fn load(app_data_dir: &Path) -> std::io::Result<ProfilesFile> {
    let path = profiles_path(app_data_dir)?;
    if !path.exists() {
        return Ok(ProfilesFile::default());
    }
    let body = fs::read_to_string(&path)?;
    serde_json::from_str(&body)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Persist profiles + active id. Writes through a temp file + rename for
/// atomicity (so a kill mid-write can't leave a half-truncated JSON).
pub fn save(app_data_dir: &Path, file: &ProfilesFile) -> std::io::Result<()> {
    let path = profiles_path(app_data_dir)?;
    let tmp = path.with_extension("json.tmp");

    let body = serde_json::to_vec_pretty(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&body)?;
        f.sync_all()?;
    }

    // SAFETY: tighten perms to user-only before swapping in. SSH key paths
    // live in here; world-readable would defeat the point. We do this on
    // the temp file (not the final path) so a crash mid-tighten can't
    // leave a 0600 file with stale contents.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))?;
    }

    fs::rename(&tmp, &path)?;
    Ok(())
}

/// Generate a fresh stable id. Called when the UI POSTs a new profile
/// without one (edits keep the existing id).
pub fn new_id() -> String {
    Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Build a unique scratch dir under the OS temp root. We avoid the
    /// `tempfile` crate to keep the dep tree lean — leftover dirs are
    /// harmless on a dev machine and CI runners are wiped between jobs.
    fn scratch(tag: &str) -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!("cypress-profiles-{}-{}", tag, new_id()));
        p
    }

    fn fixture() -> RemoteProfile {
        RemoteProfile {
            id: new_id(),
            name: "Phoenix H100".into(),
            host: "atl1-1-01-006-24-0.pace.gatech.edu".into(),
            user: "sho81".into(),
            jump_host: Some("login-phoenix.pace.gatech.edu".into()),
            jump_user: None,
            port: 7843,
            key_path: None,
            worker_dir: "~/scratch/Cypress/worker".into(),
            family: "moshi".into(),
        }
    }

    #[test]
    fn roundtrip_through_disk() {
        let dir = scratch("roundtrip");
        let p = fixture();
        let id = p.id.clone();
        let original = ProfilesFile {
            profiles: vec![p],
            active_id: Some(id),
        };
        save(&dir, &original).unwrap();
        let loaded = load(&dir).unwrap();
        assert_eq!(loaded.profiles.len(), 1);
        assert_eq!(loaded.profiles[0].name, "Phoenix H100");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_is_empty_default() {
        let dir = scratch("empty");
        let loaded = load(&dir).unwrap();
        assert!(loaded.profiles.is_empty());
        assert!(loaded.active_id.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_json_errors() {
        let dir = scratch("malformed");
        let path = profiles_path(&dir).unwrap();
        fs::write(&path, "{not json").unwrap();
        assert!(load(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
