// AREA: ui · MODEL-PICKER
// Lets the user pick and load a model. Clicking a model button calls
// the server store's `loadModel` action, which fires `/model/load`
// and flips the store's `pendingModel` so the global poller picks up
// the load progress automatically.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  selectIsRunning,
  type InferenceState,
  useServerStore,
} from "../store/serverStore";
import { useToast } from "./Toast";

// SWAP: model catalog. Extend this list (Kokoro, Orpheus, PersonaPlex
// variants, etc.) as loaders are implemented worker-side. The `name`
// is the exact string the worker's ipc_commands.load_model receives.
const MODELS = [
  { name: "moshi", label: "Moshi", hint: "3.5B · duplex · lighter" },
  { name: "personaplex", label: "PersonaPlex", hint: "7B · duplex + persona" },
] as const;

// Human-readable mapping for phase strings emitted by the worker.
// Phases are loading_* rather than downloading_* because HF's hub
// client transparently uses its on-disk cache after the first run —
// the worker can't easily tell "fetched bytes" from "loaded from
// cache". The language-model phase still calls out the first-run
// cost so a fresh install isn't mistaken for a hung load.
const PHASE_LABELS: Record<string, string> = {
  resolving: "Resolving checkpoint…",
  loading_mimi: "Loading audio codec…",
  loading_lm: "Loading language model… (first run downloads several GB)",
  loading_tokenizer: "Loading text tokenizer…",
  ready: "Finishing up…",
};

export function ModelPicker() {
  const serverUp = useServerStore(selectIsRunning);
  const status = useServerStore((s) => s.inference);
  const pending = useServerStore((s) => s.pendingModel);
  const loadModel = useServerStore((s) => s.loadModel);

  // Local: surfaced load error from the POST itself (vs a worker-side
  // error which lands in status.error). Cleared when a new load starts.
  const [error, setError] = useState<string | null>(null);
  // submitting tracks the in-flight POST. Once the server returns 202
  // the store flips pendingModel; we keep this distinction so the
  // button shows "loading…" immediately on click rather than waiting
  // for the round-trip.
  const [submitting, setSubmitting] = useState<string | null>(null);
  const toast = useToast();

  const onLoad = useCallback(
    async (name: string) => {
      // Short-circuit if a load is already in flight.
      if (submitting !== null || pending !== null) return;
      // Refuse if the server isn't up — surface as a toast rather than
      // silently no-op'ing so the user knows *why* nothing happened.
      if (!serverUp) {
        toast.show("Start the server first to load a model.", {
          variant: "warn",
        });
        return;
      }
      setError(null);
      setSubmitting(name);
      console.log(`[model] click → loading "${name}"`);
      try {
        await loadModel(name);
        console.log("[model] server accepted load");
      } catch (e) {
        console.error("[model] load request failed:", e);
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(null);
      }
    },
    [submitting, pending, serverUp, loadModel, toast],
  );

  // DEV: log lifecycle transitions so we can trace click → download →
  // loaded in the browser console without needing the server logs
  // side-by-side.
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

  const busy =
    submitting !== null ||
    pending !== null ||
    status.state === "loading" ||
    status.state === "starting";
  // REASON: small window between click and the worker emitting its
  // first phase event where status.phase is empty. Without a fallback
  // the section would look frozen — show a generic "Preparing…".
  const phaseLabel = status.phase
    ? PHASE_LABELS[status.phase] ?? status.phase
    : busy
      ? "Preparing…"
      : null;

  return (
    <div className="fixed bottom-4 left-4 flex max-w-xs flex-col gap-2 rounded-md border bg-card/80 p-3 text-xs backdrop-blur">
      <div className="font-medium text-foreground">Model</div>
      {/* REASON: while a load is in flight the whole section is locked
          — aria-busy + pointer-events-none on the inner list keeps
          clicks from queueing on top of an in-flight request. The
          pending button stays at full opacity so the user can see
          *which* model is loading at a glance. */}
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
              onClick={() => onLoad(m.name)}
              // REASON: only `busy` truly disables the button. When the
              // server isn't up we want the click to still fire so
              // onLoad can surface a toast — silently disabling leaves
              // the user wondering why nothing happens.
              disabled={busy}
              className={`group/model flex flex-col items-start rounded border px-2 py-1.5 text-left transition-all ${
                isActive
                  ? "border-primary bg-primary/10 text-foreground"
                  : "cursor-pointer border-border bg-secondary text-secondary-foreground hover:translate-x-0.5 hover:border-primary/60 hover:bg-accent hover:shadow-sm"
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
