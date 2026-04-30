// AREA: ui · SERVER-CONTROL
// Small floating control for starting/stopping the Go orchestration
// server subprocess and surfacing its live status. Reads state from
// the server store (which owns the Tauri event listener + the
// /status poller); this component is now pure presentation.
//
// On hover, expands into a details panel showing the inference
// subsystem's snapshot (device, loaded model, phase).

import { useCallback, useState } from "react";
import {
  type InferenceSnapshot,
  type ServerStatus,
  useServerStore,
} from "../store/serverStore";

// Human-readable device labels. Unknown devices fall through to the
// raw string so a new backend shows up in the UI without a frontend
// change.
const DEVICE_LABEL: Record<string, string> = {
  mps: "Apple Silicon (MPS)",
  cuda: "NVIDIA (CUDA)",
  cpu: "CPU",
};

// Static per-state presentation. Adding a new state is one entry in
// this map + one arm in the button logic below.
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

function inferenceLabel(s: InferenceSnapshot): string {
  switch (s.state) {
    case "idle":
      return "No worker";
    case "starting":
      return "Starting worker…";
    case "ready":
      return "Ready";
    case "loading":
      return "Loading model…";
    case "serving":
      return "Serving";
  }
}

export function ServerControl() {
  const status = useServerStore((s) => s.status);
  const snapshot = useServerStore((s) => s.inference);
  const startServer = useServerStore((s) => s.startServer);
  const stopServer = useServerStore((s) => s.stopServer);

  // Local-only: tracks the in-flight invoke so we can show "Stopping…"
  // visuals before the Rust side has emitted its status transition.
  const [busy, setBusy] = useState(false);

  const onClick = useCallback(async () => {
    setBusy(true);
    try {
      if (status.state === "running" || status.state === "starting") {
        await stopServer();
      } else {
        await startServer();
      }
    } catch {
      // The Rust side already emitted an Error status event; no extra UI.
    } finally {
      setBusy(false);
    }
  }, [status.state, startServer, stopServer]);

  const meta = STATUS_META[status.state];
  const isRunLike =
    status.state === "running" || status.state === "starting";
  const buttonLabel = isRunLike ? "Stop Server" : "Start Server";
  const disabled =
    busy || status.state === "starting" || status.state === "stopping";

  const deviceLabel = snapshot.device
    ? DEVICE_LABEL[snapshot.device] ?? snapshot.device
    : "—";

  // Transport label: "Local subprocess" vs "Remote (<url>)". Helps when
  // the user toggles env vars and forgets which mode the server picked
  // up — without this they'd have to read stderr to figure out.
  const transportLabel =
    snapshot.transport === "remote"
      ? `Remote (${snapshot.remote?.url ?? "—"})`
      : "Local subprocess";

  // Reachability badge for the popup row. A separate signal from
  // "Server: Running" because the Go server can be up while the
  // remote worker is down.
  const remoteReachable = snapshot.remote?.reachable;
  const reachLabel =
    snapshot.transport === "remote"
      ? remoteReachable
        ? "Reachable"
        : "Unreachable"
      : null;

  return (
    <div className="group fixed bottom-4 right-4 flex flex-col items-end gap-2">
      {/* Hover panel — slides up from the control. Only meaningful while
          the server is running, since otherwise there's nothing to show. */}
      <div
        className="pointer-events-none flex w-64 translate-y-1 flex-col gap-1 rounded-md border bg-card/90 p-3 text-xs opacity-0 shadow-md backdrop-blur transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100"
      >
        <div className="mb-1 font-medium text-foreground">Server details</div>
        <Row label="Server" value={meta.label} />
        <Row label="Transport" value={transportLabel} />
        {reachLabel && (
          <Row
            label="Worker"
            value={reachLabel}
            valueClass={
              remoteReachable ? "text-foreground" : "text-red-500"
            }
          />
        )}
        <Row label="Device" value={deviceLabel} />
        <Row label="Model" value={snapshot.model || "—"} />
        <Row label="Inference" value={inferenceLabel(snapshot)} />
        {snapshot.phase && <Row label="Phase" value={snapshot.phase} />}
      </div>

      <div className="flex items-center gap-3 rounded-md border bg-card/80 px-3 py-2 text-xs backdrop-blur">
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
    </div>
  );
}

function Row({
  label,
  value,
  valueClass,
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className={`truncate ${valueClass ?? "text-foreground"}`}>
        {value}
      </span>
    </div>
  );
}
