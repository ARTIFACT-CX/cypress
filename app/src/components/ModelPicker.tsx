// AREA: ui · MODEL-PICKER
// Lets the user pick and load a model. Clicking a model button fires a
// single `/model/load` POST to the Go server, which in turn lazily spawns
// the Python worker if needed. All lifecycle state (spawning worker,
// loading weights, load error) is surfaced here so the user sees one
// clear status instead of having to watch a console.

import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useToast } from "./Toast";

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
  device: string;
  // Phase is transitional — only populated while the worker is mid-load
  // (downloading_mimi, downloading_lm, loading_tokenizer, etc.).
  phase: string;
  // Error is the last load failure, if any. Surfaced by the server so we
  // catch failures that happened after the fire-and-forget POST returned.
  error?: string;
};

// Human-readable mapping for phase strings emitted by the worker. Unknown
// phases fall back to the raw string, so new worker-side phases show up
// in the UI without requiring a frontend change.
const PHASE_LABELS: Record<string, string> = {
  resolving: "Resolving checkpoint…",
  downloading_mimi: "Downloading audio codec…",
  downloading_lm:
    "Downloading language model… (first run only — can take several minutes)",
  loading_tokenizer: "Loading text tokenizer…",
  ready: "Finishing up…",
};

const EMPTY_STATUS: Status = { state: "idle", model: "", device: "", phase: "" };

export function ModelPicker() {
  // STEP 1: track whether the Go server is up. The model buttons are
  // useless when it isn't, so we disable them rather than letting fetch
  // fail with a cryptic "connection refused" in devtools.
  const [serverState, setServerState] = useState<ServerState>("idle");

  // STEP 2: track the inference subsystem's own state. Populated by polling
  // /status while a load is in flight, and once on server-ready.
  const [status, setStatus] = useState<Status>(EMPTY_STATUS);
  // submitting is set the moment the user clicks, while the POST is in
  // flight. Once the server returns 202 we move the value to `pending`,
  // which gates the /status polling. Splitting these two prevents a
  // race: if we drove polling off `pending` from the click moment, the
  // first /status request could land in parallel with /model/load and
  // return the previous "serving" snapshot before LoadModel had a chance
  // to flip state — the UI would then immediately clear pending and
  // appear unchanged.
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // SAFETY: ref tracks "any load in flight" (submitting or pending) so a
  // burst of rapid clicks before React commits any state still sees
  // there's a load in flight. setState is async; this ref isn't.
  const pendingRef = useRef<string | null>(null);
  const toast = useToast();
  useEffect(() => {
    pendingRef.current = pending ?? submitting;
  }, [pending, submitting]);

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

  // STEP 4: refresh /status once when the server comes online so we pick
  // up any model that was left loaded from a previous session. When the
  // server leaves "running" we also wipe local status back to idle — the
  // worker is gone, so any "Moshi active" badge would otherwise linger
  // from stale React state.
  useEffect(() => {
    if (serverState !== "running") {
      setStatus(EMPTY_STATUS);
      setSubmitting(null);
      setPending(null);
      setError(null);
      return;
    }
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

  // STEP 5: while a load is in flight, poll /status so the UI can show
  // the worker's current phase (downloading mimi, downloading lm, etc.)
  // instead of a silent spinner. The server's /model/load is fire-and-
  // forget, so this poll is also how we learn that the load finished —
  // when state transitions to "serving" (success) or "ready" with an
  // error field (failure), we clear `pending` and surface accordingly.
  useEffect(() => {
    if (pending === null) return;
    let cancelled = false;
    const tick = () => {
      fetch(`${SERVER_URL}/status`)
        .then((r) => r.json())
        .then((s: Status) => {
          if (cancelled) return;
          setStatus(s);
          // Terminal-state detection: an error from any state is final
          // (worker rejected the load, or the worker crashed). Otherwise
          // we only consider the load complete when the *expected* model
          // is actually serving — comparing s.model to pending guards
          // against a transient "serving" snapshot for a previous model
          // that the polling effect might catch before the worker has
          // unloaded it.
          if (s.error) {
            setError(s.error);
            setPending(null);
          } else if (s.state === "serving" && s.model === pending) {
            setPending(null);
          }
        })
        .catch(() => {});
    };
    // Kick off immediately so the UI updates on the first phase event
    // rather than waiting a full interval, then poll steadily. 500ms is
    // fast enough to feel live without being wasteful for an all-local
    // HTTP call.
    tick();
    const handle = setInterval(tick, 500);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
    // pending is the trigger; including it here means a successful load
    // (which clears pending) tears the interval down on the next render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  const loadModel = useCallback(async (name: string) => {
    // STEP 1: short-circuit if a load is already in flight (covers both
    // the in-flight POST and an active polling cycle).
    if (pendingRef.current !== null) return;
    // STEP 2: refuse if the server isn't up yet. We surface this as a
    // toast rather than silently no-op'ing so the user knows *why*
    // nothing happened. The button is intentionally not `disabled` in
    // this case — a click should still get feedback.
    if (serverState !== "running") {
      toast.show("Start the server first to load a model.", {
        variant: "warn",
      });
      return;
    }
    pendingRef.current = name;
    // submitting drives the *visual* "loading…" state immediately so the
    // user sees feedback while the POST is in flight. We deliberately
    // don't set `pending` yet — `pending` gates the /status polling and
    // we need that to start *after* the server has acknowledged the
    // load (see the race comment near useState above).
    setSubmitting(name);
    setError(null);
    console.log(`[model] click → loading "${name}"`);
    try {
      const res = await fetch(`${SERVER_URL}/model/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      console.log(`[model] server accepted load (${res.status})`);
      // STEP 3: server has synchronously flipped state to starting/
      // loading. Now it's safe to start the /status poll — the next
      // tick is guaranteed to read the new state.
      setSubmitting(null);
      setPending(name);
    } catch (e) {
      console.error(`[model] load request failed:`, e);
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(null);
      pendingRef.current = null;
    }
  }, [serverState, toast]);

  // DEV: log lifecycle transitions so we can trace click → download →
  // loaded in the browser console without needing the server logs side-
  // by-side. Watches state and phase; both are debounced by React's
  // dedupe so we only log on real changes.
  const prevPhase = useRef<string>("");
  const prevState = useRef<InferenceState>("idle");
  useEffect(() => {
    if (status.phase && status.phase !== prevPhase.current) {
      console.log(`[model] phase: ${status.phase}`);
      prevPhase.current = status.phase;
    }
    if (status.state !== prevState.current) {
      if (status.state === "serving") {
        console.log(
          `[model] loaded "${status.model}" on ${status.device || "?"}`,
        );
      } else if (status.state === "loading") {
        console.log(`[model] downloading/loading "${status.model}"…`);
      }
      prevState.current = status.state;
    }
    if (status.error) {
      console.error(`[model] error: ${status.error}`);
    }
  }, [status]);

  const serverUp = serverState === "running";
  const busy =
    submitting !== null ||
    pending !== null ||
    status.state === "loading" ||
    status.state === "starting";
  // REASON: there's a small window between clicking a model and the worker
  // emitting its first phase event where status.phase is empty. Without a
  // fallback the section would look frozen — show a generic "Preparing…"
  // until the real phase string arrives.
  const phaseLabel = status.phase
    ? PHASE_LABELS[status.phase] ?? status.phase
    : busy
      ? "Preparing…"
      : null;

  return (
    <div className="fixed bottom-4 left-4 flex max-w-xs flex-col gap-2 rounded-md border bg-card/80 p-3 text-xs backdrop-blur">
      <div className="font-medium text-foreground">Model</div>
      {/* REASON: while a load is in flight the whole section is locked —
          aria-busy + pointer-events-none on the inner list keeps clicks
          from queueing on top of an in-flight request, which previously
          caused noticeable lag from piled-up POSTs. The pending button
          stays at full opacity so the user can see *which* model is
          loading at a glance. */}
      <div
        aria-busy={busy}
        className={`flex flex-col gap-1.5 ${
          busy ? "pointer-events-none" : ""
        }`}
      >
        {MODELS.map((m) => {
          const isActive =
            status.model === m.name && status.state === "serving";
          const isPending = pending === m.name || submitting === m.name;
          return (
            <button
              key={m.name}
              type="button"
              onClick={() => loadModel(m.name)}
              // REASON: only `busy` truly disables the button. When the server
              // isn't up we want the click to still fire so loadModel can
              // surface a toast ("Start the server first") — silently
              // disabling leaves the user wondering why nothing happens.
              disabled={busy}
              className={`group/model flex flex-col items-start rounded border px-2 py-1.5 text-left transition-all ${
                isActive
                  ? "border-primary bg-primary/10 text-foreground"
                  : // REASON: hover lifts the border to primary + nudges the row
                    // right one pixel and adds a soft ring so the user gets a
                    // clear "this is clickable" cue. Cursor-pointer is
                    // explicit because Tauri/macOS won't add it on buttons by
                    // default in some webview versions.
                    "cursor-pointer border-border bg-secondary text-secondary-foreground hover:translate-x-0.5 hover:border-primary/60 hover:bg-accent hover:shadow-sm"
              } ${
                busy && !isPending
                  ? "opacity-40"
                  : !serverUp
                    ? "opacity-50"
                    : ""
              }`}
            >
              <span className="flex items-center gap-1.5 font-medium">
                {m.label}
                {isPending && (
                  <span className="text-muted-foreground">· loading…</span>
                )}
                {isActive && (
                  // Green dot indicates the model is live and serving. Kept
                  // as a tiny visual cue instead of a text suffix so the row
                  // stays compact at a glance.
                  <span
                    aria-label="active"
                    className="h-1.5 w-1.5 rounded-full bg-green-500"
                  />
                )}
              </span>
              <span className="text-muted-foreground">{m.hint}</span>
            </button>
          );
        })}
      </div>
      {!serverUp && (
        <div className="text-muted-foreground">Start the server first.</div>
      )}
      {phaseLabel && (
        <div className="text-muted-foreground italic">{phaseLabel}</div>
      )}
      {error && (
        <div className="rounded border border-red-500/30 bg-red-500/10 px-2 py-1 text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
