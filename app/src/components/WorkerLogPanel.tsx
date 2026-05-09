// AREA: ui · WORKER-LOG-PANEL
// Tail of remote worker stderr lines, mounted next to the server
// control. Visible only while a remote profile is active and the
// server is up — local mode has no remote stderr to forward and an
// empty tail would just be noise.
//
// Reads from useWorkerLogStore which is filled by the bootstrap
// listener subscribed to the Tauri remote-worker-log event.

import { useEffect, useRef } from "react";
import { Terminal, X } from "lucide-react";
import { useServerStore } from "../store/serverStore";
import { useWorkerLogStore } from "../store/workerLogStore";

export function WorkerLogPanel() {
  const lines = useWorkerLogStore((s) => s.lines);
  const open = useWorkerLogStore((s) => s.open);
  const setOpen = useWorkerLogStore((s) => s.setOpen);
  const transport = useServerStore((s) => s.inference.transport);
  const serverState = useServerStore((s) => s.status.state);

  // STEP: auto-open the panel when the worker writes its first line.
  // Before that we don't know there's anything worth showing — moshi
  // is quiet on a healthy startup, but anything else (warning,
  // traceback) we want surfaced immediately. If the user has manually
  // closed it, respect that until next connect (the bootstrap clears
  // the buffer on idle, which resets the "first line" condition).
  const openedOnce = useRef(false);
  useEffect(() => {
    if (lines.length === 0) {
      openedOnce.current = false;
      return;
    }
    if (!openedOnce.current) {
      openedOnce.current = true;
      setOpen(true);
    }
  }, [lines, setOpen]);

  // Auto-scroll to the tail on every new line.
  const tailRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [lines]);

  // Surface the panel only when there's something to show — no remote,
  // no connection, no panel.
  const visible =
    transport === "remote" && serverState === "running" && lines.length > 0;
  if (!visible || !open) return null;

  return (
    <div className="pointer-events-auto flex w-96 flex-col rounded-md border bg-card/90 text-xs shadow-md backdrop-blur">
      <div className="flex items-center justify-between border-b border-border/60 px-3 py-1.5">
        <span className="flex items-center gap-1.5 font-medium text-foreground">
          <Terminal className="h-3 w-3" />
          Remote worker log
        </span>
        <button
          type="button"
          onClick={() => setOpen(false)}
          aria-label="Hide worker log"
          className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <X className="h-3 w-3" />
        </button>
      </div>
      <div className="max-h-48 overflow-y-auto px-3 py-2 font-mono text-[10px] leading-snug text-muted-foreground">
        {lines.map((l, i) => (
          <div key={i} className="whitespace-pre-wrap break-all">
            {l}
          </div>
        ))}
        <div ref={tailRef} />
      </div>
    </div>
  );
}
