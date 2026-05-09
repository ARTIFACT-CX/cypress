// AREA: desktop · REMOTE · SSH
// Spawns and supervises the system `ssh` client for two purposes:
//
//   1. tunnel  — `ssh -N -L <local>:127.0.0.1:<remote> [-J jump] user@host`
//                Forwards a local TCP port to the worker on the remote box.
//                No remote command; stays open until killed.
//
//   2. worker  — (Phase 2) `ssh [-J jump] user@host "cd <dir> && python ..."`
//                Runs the worker process on the remote and stays attached so
//                we capture stderr. Killing this child kills the python
//                process via SIGHUP propagation.
//
// SWAP: we shell out to the system `ssh` binary because it inherits the
// user's ~/.ssh/config (handy on HPC where ProxyJump + agent forwarding
// are pre-configured) and adds zero deps. If we ever want in-process
// control (e.g. so the app survives without an `ssh` on $PATH on
// Windows), russh is the swap target — same Child trait, just construct
// the channels in-process instead of fork+exec.

use std::process::Stdio;
use std::time::Duration;

use rand::RngCore;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

use super::profiles::RemoteProfile;

/// Tauri event the worker stderr panel listens on. Each line emitted as
/// a separate event so the frontend can ringbuffer without parsing
/// boundaries itself.
pub const WORKER_LOG_EVENT: &str = "remote-worker-log";

/// Cap on how many lines the worker stderr stream forwards. After this
/// we silently drop — the typical shape is "thousands of normal lines
/// then a traceback at the end"; we'd rather surface the first error
/// than tail-chase an infinite log. Re-bumped on each new connect.
const MAX_FORWARDED_LINES: usize = 2000;

/// How long we wait for the SSH tunnel's local listener to come up after
/// spawning. Generous because the first hop (login node auth, agent
/// forwarding negotiation, 2FA on some HPC clusters) can take a few
/// seconds even with ControlMaster.
const TUNNEL_READY_TIMEOUT: Duration = Duration::from_secs(20);
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Bearer token shared between the laptop's Go server and the remote
/// worker. Generated fresh per-connect, never written to disk.
pub fn generate_token() -> String {
    // 32 random bytes hex-encoded = 64 hex chars. Matches what users
    // would otherwise generate via `openssl rand -hex 32`.
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Build the base `ssh` command with shared options applied (BatchMode,
/// connect timeout, StrictHostKeyChecking=accept-new, ProxyJump if set,
/// IdentityFile if set). Both spawn paths use this so the auth surface
/// stays identical between tunnel and worker.
fn base_ssh(profile: &RemoteProfile) -> Command {
    let mut cmd = Command::new("ssh");
    cmd
        // BatchMode: refuse to prompt for passwords / passphrases. We
        // run from a Tauri child with no terminal — a hidden prompt
        // would just hang. The user has to set up keys (we'll surface
        // the failure message clearly).
        .arg("-o").arg("BatchMode=yes")
        // ConnectTimeout caps the TCP+handshake wait so a wrong host
        // fails fast instead of dragging us through the whole 75s
        // default. Auth + tunnel setup happens after, with its own
        // budget upstream.
        .arg("-o").arg("ConnectTimeout=10")
        // accept-new auto-accepts unknown hosts on first contact (and
        // pins them) without prompting. Better than StrictHostKeyChecking=no
        // (silent MITM tolerance) but doesn't require pre-seeding
        // known_hosts.
        .arg("-o").arg("StrictHostKeyChecking=accept-new")
        // ServerAliveInterval keeps NAT mappings warm and surfaces a
        // dead tunnel within ~30s instead of letting it hang silently.
        .arg("-o").arg("ServerAliveInterval=15")
        .arg("-o").arg("ServerAliveCountMax=2");

    if let Some(jump) = &profile.jump_host {
        let user = profile.jump_user.as_ref().unwrap_or(&profile.user);
        cmd.arg("-J").arg(format!("{}@{}", user, jump));
    }

    if let Some(key) = &profile.key_path {
        // IdentitiesOnly=yes prevents ssh from trying every key in the
        // user's agent / config when this profile pins one explicitly.
        // Without it, a noisy agent can rate-limit us out before
        // reaching the right key.
        cmd.arg("-i").arg(key);
        cmd.arg("-o").arg("IdentitiesOnly=yes");
    }

    cmd
}

/// Spawn the port-forwarding tunnel. The returned Child stays alive
/// until killed; dropping it (kill_on_drop) releases the local port.
/// Caller owns the lifecycle.
///
/// `local_port` is what the laptop's Go server will dial; `profile.port`
/// is what the worker binds on the remote side.
pub async fn spawn_tunnel(profile: &RemoteProfile, local_port: u16) -> Result<Child, String> {
    let mut cmd = base_ssh(profile);
    cmd
        // -N: do not run a remote command, just hold the connection
        // open for forwarding. Without this we'd land in a remote
        // shell and inherit its quirks.
        .arg("-N")
        // -L local:remote: forward laptop's <local_port> → remote's
        // 127.0.0.1:<profile.port>. We bind localhost (the ssh default)
        // so other apps on the laptop can't reach our worker by accident.
        .arg("-L")
        .arg(format!("{}:127.0.0.1:{}", local_port, profile.port))
        .arg(format!("{}@{}", profile.user, profile.host))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // SAFETY: setsid so killing the parent ssh also tears down any
    // ssh-agent / ProxyCommand children. Same pattern as the go-server
    // spawn — pgid == pid because we're at the head of a new session.
    #[cfg(unix)]
    {
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    cmd.spawn()
        .map_err(|e| format!("failed to spawn ssh tunnel: {e} (is `ssh` on $PATH?)"))
}

/// Block until either (a) `127.0.0.1:local_port` accepts a TCP connect
/// or (b) the tunnel ssh child exits, or (c) timeout. (b) means the
/// tunnel failed to establish (auth error, host unreachable) — we read
/// stderr to surface the real reason.
pub async fn wait_for_tunnel(child: &mut Child, local_port: u16) -> Result<(), String> {
    let probe = async {
        loop {
            if TcpStream::connect(("127.0.0.1", local_port)).await.is_ok() {
                return Ok::<(), String>(());
            }
            // If the ssh process is gone, no point polling further.
            if let Ok(Some(status)) = child.try_wait() {
                let stderr = drain_stderr(child).await;
                return Err(format!(
                    "ssh exited before tunnel was ready (status={status}): {stderr}"
                ));
            }
            sleep(POLL_INTERVAL).await;
        }
    };

    match timeout(TUNNEL_READY_TIMEOUT, probe).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => {
            let stderr = drain_stderr(child).await;
            Err(format!(
                "ssh tunnel did not become reachable within {}s: {stderr}",
                TUNNEL_READY_TIMEOUT.as_secs()
            ))
        }
    }
}

/// Spawn the remote worker process via a second SSH session. Stays
/// attached to the worker so SIGHUP propagates when the laptop tears
/// the SSH child down. stderr streams back into the app via the
/// WORKER_LOG_EVENT — one Tauri event per line — so the user can see
/// the actual exception when something like the Python 3.14 / torch
/// .compile case from #41 fires.
///
/// `token` is the same token the laptop's Go server uses; we hand it
/// to the worker via --token so both ends authenticate against the
/// same secret without it ever touching disk.
pub async fn spawn_worker(
    app: &AppHandle,
    profile: &RemoteProfile,
    token: &str,
) -> Result<Child, String> {
    let mut cmd = base_ssh(profile);
    cmd.arg(format!("{}@{}", profile.user, profile.host));

    // REASON: build the remote shell command in two layers so we can
    // keep path safety AND let the remote home expand:
    //   1. The cd target uses "$HOME/..." (double-quoted) so a profile
    //      saved as "~/scratch/Cypress/worker" expands against the
    //      remote user's home, not the laptop's. Single-quoting the
    //      whole path would kill the expansion (the original Phoenix
    //      "No such file or directory" failure mode).
    //   2. The whole bash body is then single-quoted as one ssh arg
    //      so spaces and meta chars in the body don't get re-parsed
    //      by the remote login shell.
    //
    // bash -lc forces a login shell so the user's .bash_profile /
    // .bashrc-set PATH (uv install dir on Phoenix), HF_HOME, and
    // UV_CACHE_DIR take effect — without -l the remote command runs
    // with a stripped env and uv-installed python isn't on PATH.
    let worker_dir = expand_tilde(&profile.worker_dir);
    let bash_body = format!(
        "cd \"{worker_dir}\" && \
         exec models/{family}/.venv/bin/python -u main.py \
           --listen tcp://127.0.0.1:{port} \
           --family {family} \
           --token {token}",
        worker_dir = worker_dir,
        // family + token are restricted enough today (alphanumerics
        // for family, hex for token) that quoting them inside double
        // quotes would only catch deliberate misuse. Keep them bare
        // so the command stays readable in logs.
        family = profile.family,
        port = profile.port,
        token = token,
    );
    cmd.arg(format!("bash -lc {}", shell_escape(&bash_body)));

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    #[cfg(unix)]
    {
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn ssh worker: {e}"))?;

    // STEP: forward stdout + stderr lines to the frontend. Both go to
    // the same event — the worker prints model phases / debug info on
    // stderr today, but we don't want to lose anything if a future
    // change emits to stdout instead.
    if let Some(stdout) = child.stdout.take() {
        forward_lines(app.clone(), stdout);
    }
    if let Some(stderr) = child.stderr.take() {
        forward_lines(app.clone(), stderr);
    }

    Ok(child)
}

/// Open a fresh SSH session and explicitly kill any python workers
/// matching `main.py.*moshi` on the remote node. Used at disconnect
/// time because relying on SIGHUP propagation through the worker SSH
/// child is unreliable — the python process detaches enough that
/// closing the SSH client doesn't take it down. Without this, every
/// Stop/Start cycle stacks another orphan worker on the remote side
/// and the kernel routes new connections to whichever it likes,
/// producing a token-mismatch / Unauthenticated error that's nearly
/// impossible to diagnose from the laptop logs.
///
/// Best-effort: any failure (SSH down, pkill not present) just logs
/// and returns — the app is shutting down a session, not a critical
/// path.
pub async fn kill_remote_workers(profile: &RemoteProfile) -> Result<(), String> {
    let mut cmd = base_ssh(profile);
    cmd.arg(format!("{}@{}", profile.user, profile.host));
    // SAFETY: -9 because plain pkill (SIGTERM) sometimes doesn't take —
    // the moshi loader holds onto a CUDA context that doesn't
    // shut down on SIGTERM in a reasonable budget. SIGKILL is fine
    // because we don't care about graceful shutdown of a session
    // we're explicitly tearing down.
    //
    // pkill exits 1 when no processes match; treat that as success
    // (idempotent disconnect).
    cmd.arg("pkill -9 -f 'main.py.*moshi' || true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("kill remote workers: {e}"))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("kill remote workers exit {}: {}", output.status, err));
    }
    Ok(())
}

/// Wait for the worker's gRPC port to actually accept a connection
/// through the existing tunnel. Bounded by `timeout_secs` because moshi
/// + torch import on a fresh venv can take 10–20s on a cold node.
pub async fn wait_for_worker(local_port: u16, timeout_secs: u64) -> Result<(), String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if TcpStream::connect(("127.0.0.1", local_port)).await.is_ok() {
            // REASON: TCP connect succeeds the moment the gRPC server
            // binds, which can be a beat before the python dispatcher
            // is actually wired up. The Go server's eager probe (#36)
            // tolerates this with a 15s handshake budget; we just need
            // the listener up before returning.
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            return Err(format!(
                "worker did not become reachable within {timeout_secs}s",
            ));
        }
        sleep(POLL_INTERVAL).await;
    }
}

/// Spawn a task that pumps lines from the given async reader into Tauri
/// events. Stops at MAX_FORWARDED_LINES or EOF, whichever comes first.
fn forward_lines<R>(app: AppHandle, reader: R)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tauri::async_runtime::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        let mut count = 0usize;
        while let Ok(Some(line)) = lines.next_line().await {
            // Mirror to the dev terminal too so a missing webview
            // listener doesn't lose log lines during early bring-up.
            eprintln!("[remote-worker] {line}");
            let _ = app.emit(WORKER_LOG_EVENT, line);
            count += 1;
            if count >= MAX_FORWARDED_LINES {
                let _ = app.emit(
                    WORKER_LOG_EVENT,
                    format!("[truncated after {MAX_FORWARDED_LINES} lines]"),
                );
                break;
            }
        }
    });
}

/// Single-quote escape for splicing into a remote sh command. We don't
/// pull in shell-words because the rules here are fixed (POSIX sh),
/// and the user-supplied strings are all paths / identifiers — no
/// embedded newlines to worry about.
fn shell_escape(s: &str) -> String {
    let escaped = s.replace('\'', "'\\''");
    format!("'{escaped}'")
}

/// Translate a leading `~/` (or bare `~`) to `$HOME/` so the path can
/// sit inside double quotes and still expand to the remote user's
/// home. Any other tilde form (e.g. `~user`) is left alone — those
/// don't have a portable expansion when the laptop and remote may
/// disagree about which user the path belongs to.
fn expand_tilde(p: &str) -> String {
    if p == "~" {
        "$HOME".to_string()
    } else if let Some(rest) = p.strip_prefix("~/") {
        format!("$HOME/{rest}")
    } else {
        p.to_string()
    }
}

/// Best-effort drain of whatever ssh has written to stderr. Used only
/// when the tunnel has failed and we're building the error message —
/// non-blocking, no waiting for EOF.
async fn drain_stderr(child: &mut Child) -> String {
    use tokio::io::AsyncReadExt;
    let Some(mut stderr) = child.stderr.take() else {
        return String::new();
    };
    let mut buf = Vec::new();
    // SAFETY: small read window — this fires after the process is
    // already dead, so EOF will come quickly. Not waiting forever in
    // case stderr is somehow held open by a forked grandchild.
    let _ = timeout(Duration::from_millis(500), stderr.read_to_end(&mut buf)).await;
    String::from_utf8_lossy(&buf).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_64_hex_chars() {
        let t = generate_token();
        assert_eq!(t.len(), 64);
        assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn token_is_unique_across_calls() {
        // SAFETY: not a strict uniqueness guarantee — collisions are
        // statistically negligible at 256 bits. This guards against the
        // one-time mistake of accidentally returning a constant.
        assert_ne!(generate_token(), generate_token());
    }

    #[test]
    fn tilde_expansion_swaps_to_home() {
        // REASON: regression for the Phoenix "cd ~/scratch: No such
        // file or directory" failure. ~/foo must become $HOME/foo so
        // it survives double-quoting at the remote shell.
        assert_eq!(expand_tilde("~/scratch/Cypress"), "$HOME/scratch/Cypress");
        assert_eq!(expand_tilde("~"), "$HOME");
        assert_eq!(expand_tilde("/abs/path"), "/abs/path");
        assert_eq!(expand_tilde("relative/path"), "relative/path");
        // ~user not supported — the remote shell handles whatever it
        // wants with that.
        assert_eq!(expand_tilde("~other/foo"), "~other/foo");
    }
}
