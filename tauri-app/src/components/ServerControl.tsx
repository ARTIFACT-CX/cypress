// AREA: ui · SERVER-CONTROL
// Small floating control for starting/stopping the Go orchestration server
// subprocess and surfacing its live status. State is owned by the Rust side
// (see src-tauri/src/server.rs); we hydrate once via `server_status` and then
// listen for `server-status` events for every transition.

import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// SWAP: keep this shape in sync with ServerStatus in server.rs. If we add
// more lifecycle states there, mirror them here so the dot/label logic stays
// exhaustive.
type ServerStatus =
  | { state: "idle" }
  | { state: "starting" }
  | { state: "running" }
  | { state: "stopping" }
  | { state: "error"; message: string };

// Static per-state presentation. Keeping this declarative means adding a new
// state is one entry in this map + one arm in the button logic below.
const STATUS_META: Record<
  ServerStatus["state"],
  { label: string; dotClass: string }
> = {
  idle: { label: "Server idle", dotClass: "bg-muted-foreground" },
  starting: { label: "Starting…", dotClass: "bg-yellow-400 animate-pulse" },
  running: { label: "Running", dotClass: "bg-green-500" },
  stopping: { label: "Stopping…", dotClass: "bg-yellow-400 animate-pulse" },
  error: { label: "Error", dotClass: "bg-red-500" },
};

export function ServerControl() {
  const [status, setStatus] = useState<ServerStatus>({ state: "idle" });
  const [busy, setBusy] = useState(false);

  // STEP 1: hydrate initial state + subscribe to live updates. The Rust side
  // emits `server-status` on every transition; we treat events as the source
  // of truth and fall back to the invoke only for the initial render.
  useEffect(() => {
    let mounted = true;
    invoke<ServerStatus>("server_status")
      .then((s) => mounted && setStatus(s))
      .catch(() => {});
    const unlistenPromise = listen<ServerStatus>("server-status", (e) => {
      if (mounted) setStatus(e.payload);
    });
    return () => {
      mounted = false;
      unlistenPromise.then((un) => un());
    };
  }, []);

  // STEP 2: button action. While a command is in flight we disable the button
  // so the user can't queue start/stop/start in quick succession.
  const onClick = useCallback(async () => {
    setBusy(true);
    try {
      if (status.state === "running" || status.state === "starting") {
        await invoke("stop_server");
      } else {
        await invoke("start_server");
      }
    } catch {
      // The Rust side already emitted an Error status event; no extra UI.
    } finally {
      setBusy(false);
    }
  }, [status.state]);

  const meta = STATUS_META[status.state];
  const isRunLike =
    status.state === "running" || status.state === "starting";
  const buttonLabel = isRunLike ? "Stop Server" : "Start Server";
  const disabled =
    busy || status.state === "starting" || status.state === "stopping";

  return (
    <div className="fixed bottom-4 right-4 flex items-center gap-3 rounded-md border bg-card/80 px-3 py-2 text-xs backdrop-blur">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${meta.dotClass}`} />
        <span className="text-foreground">{meta.label}</span>
      </div>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className="rounded border border-border bg-secondary px-2 py-1 text-secondary-foreground hover:bg-accent disabled:opacity-50"
      >
        {buttonLabel}
      </button>
    </div>
  );
}
