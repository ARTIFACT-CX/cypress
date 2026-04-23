// AREA: ui · MODEL-PICKER
// Lets the user pick and load a model. Clicking a model button fires a
// single `/model/load` POST to the Go server, which in turn lazily spawns
// the Python worker if needed. All lifecycle state (spawning worker,
// loading weights, load error) is surfaced here so the user sees one
// clear status instead of having to watch a console.

import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";

// SWAP: model catalog. Extend this list (Kokoro, Orpheus, PersonaPlex
// variants, etc.) as loaders are implemented worker-side. The `name`
// is the exact string the worker's ipc_commands.load_model receives.
const MODELS = [
  { name: "moshi", label: "Moshi", hint: "3.5B · duplex · lighter" },
  { name: "personaplex", label: "PersonaPlex", hint: "7B · duplex + persona" },
] as const;

// SETUP: server base URL. Matches server/main.go listenAddr. Hard-coded
// for v0.1 because the UI and server always co-locate on the same machine.
const SERVER_URL = "http://127.0.0.1:7842";

type ServerState = "idle" | "starting" | "running" | "stopping" | "error";

// InferenceState mirrors server/inference/manager.go's State enum. Kept as
// a string union on this side so TS can exhaustively check the UI labels.
type InferenceState = "idle" | "starting" | "ready" | "loading" | "serving";

type Status = {
  state: InferenceState;
  model: string;
};

export function ModelPicker() {
  // STEP 1: track whether the Go server is up. The model buttons are
  // useless when it isn't, so we disable them rather than letting fetch
  // fail with a cryptic "connection refused" in devtools.
  const [serverState, setServerState] = useState<ServerState>("idle");

  // STEP 2: track the inference subsystem's own state. Populated by polling
  // /status after the server comes up, and after each load attempt.
  const [status, setStatus] = useState<Status>({ state: "idle", model: "" });
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // STEP 3: subscribe to the Rust-emitted server-status events so we know
  // when to start polling /status. No point fetching while the server is
  // booting or down.
  useEffect(() => {
    let mounted = true;
    const unlisten = listen<{ state: ServerState }>("server-status", (e) => {
      if (mounted) setServerState(e.payload.state);
    });
    return () => {
      mounted = false;
      unlisten.then((un) => un());
    };
  }, []);

  // STEP 4: refresh /status whenever the server comes online. One poll is
  // enough — after that, every load action refreshes status locally from
  // its own response, so we don't need continuous polling.
  useEffect(() => {
    if (serverState !== "running") return;
    let cancelled = false;
    fetch(`${SERVER_URL}/status`)
      .then((r) => r.json())
      .then((s: Status) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [serverState]);

  const loadModel = useCallback(async (name: string) => {
    setPending(name);
    setError(null);
    try {
      const res = await fetch(`${SERVER_URL}/model/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        // WHY: surface the server's error verbatim. uv-not-found, handshake
        // timeout, and worker-side "model not implemented" all flow through
        // here, and each one is more useful raw than wrapped.
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      // Optimistic update. A subsequent /status poll would confirm, but
      // the response already tells us everything we need.
      setStatus({ state: "serving", model: name });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // Refresh status so we don't leave the UI showing a stale "loading".
      fetch(`${SERVER_URL}/status`)
        .then((r) => r.json())
        .then((s: Status) => setStatus(s))
        .catch(() => {});
    } finally {
      setPending(null);
    }
  }, []);

  const serverUp = serverState === "running";
  const busy = pending !== null || status.state === "loading" || status.state === "starting";

  return (
    <div className="fixed bottom-4 left-4 flex max-w-xs flex-col gap-2 rounded-md border bg-card/80 p-3 text-xs backdrop-blur">
      <div className="font-medium text-foreground">Model</div>
      <div className="flex flex-col gap-1.5">
        {MODELS.map((m) => {
          const isActive = status.model === m.name && status.state === "serving";
          const isPending = pending === m.name;
          return (
            <button
              key={m.name}
              type="button"
              onClick={() => loadModel(m.name)}
              disabled={!serverUp || busy}
              className={`flex flex-col items-start rounded border px-2 py-1.5 text-left transition-colors disabled:opacity-50 ${
                isActive
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border bg-secondary text-secondary-foreground hover:bg-accent"
              }`}
            >
              <span className="font-medium">
                {m.label}
                {isPending && " · loading…"}
                {isActive && " · active"}
              </span>
              <span className="text-muted-foreground">{m.hint}</span>
            </button>
          );
        })}
      </div>
      {!serverUp && (
        <div className="text-muted-foreground">Start the server first.</div>
      )}
      {error && (
        <div className="rounded border border-red-500/30 bg-red-500/10 px-2 py-1 text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
