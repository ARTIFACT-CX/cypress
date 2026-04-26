// AREA: ui · MODEL-PICKER
// Lets the user browse, download, load, and delete models. Each row
// has its own action depending on disk state: "Download" for missing,
// "Load" for downloaded-but-not-loaded, "Active" for the live model.
// A trash icon next to installed rows runs the delete flow.
//
// Download progress streams via /models polling (fast cadence while a
// download is active) — see store/bootstrap.ts.

import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Download, Loader2, MoreHorizontal, Play, Trash2 } from "lucide-react";
import {
  selectIsRunning,
  type DownloadProgress,
  type ModelInfo,
  useServerStore,
} from "../store/serverStore";
import { useToast } from "./Toast";

// Human-readable mapping for phase strings emitted by the worker
// during *load* (post-download). Phases are loading_* rather than
// downloading_* because by the time we hit load, all bytes are local.
const PHASE_LABELS: Record<string, string> = {
  resolving: "Resolving checkpoint…",
  loading_mimi: "Loading audio codec…",
  loading_lm: "Loading language model…",
  loading_tokenizer: "Loading text tokenizer…",
  ready: "Finishing up…",
};

// Format bytes as "12.3 GB" / "456 MB" / "789 KB". Tabular sums are
// rare here so we keep it lightweight rather than pulling in numeral.js.
function formatBytes(n: number): string {
  if (n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

// Render the per-row download bar. Lives as its own component so the
// model-row JSX stays scannable. Indeterminate (animated) when total
// is unknown; determinate fill when we have HF's metadata.
function DownloadBar({ d }: { d: DownloadProgress }) {
  const known = d.total > 0;
  const pct = known
    ? Math.min(100, Math.max(0, (d.downloaded / d.total) * 100))
    : 0;
  const label =
    d.phase === "starting"
      ? "Preparing…"
      : known
        ? `${formatBytes(d.downloaded)} / ${formatBytes(d.total)}`
        : `Downloading ${d.fileIndex + 1}/${d.fileCount}…`;
  return (
    <div className="mt-1 w-full">
      <div className="mb-0.5 flex items-center justify-between text-[10px] tabular-nums text-muted-foreground">
        <span className="truncate">{d.file || "preparing"}</span>
        <span>{label}</span>
      </div>
      <div className="relative h-1 w-full overflow-hidden rounded bg-secondary">
        {known ? (
          <div
            className="absolute inset-y-0 left-0 bg-primary transition-[width] duration-300"
            style={{ width: `${pct}%` }}
          />
        ) : (
          // SWAP: indeterminate stripe. Tailwind's animate-pulse gives a
          // subtle "working" cue without us shipping a custom keyframe.
          <div className="absolute inset-y-0 left-0 w-1/3 animate-pulse bg-primary/60" />
        )}
      </div>
    </div>
  );
}

// Tiny popover menu anchored to the "more actions" button. Rolled
// inline rather than pulling in a Radix dropdown — we have one menu,
// one item, and click-outside is a few lines. Closes on outside click,
// Escape, or item activation.
function RowMenu({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      ref={ref}
      role="menu"
      // REASON: explicit hsl(var(--popover)) instead of `bg-popover`.
      // The project's Tailwind v4 setup doesn't have an @theme block,
      // so semantic color tokens like `bg-popover` resolve to nothing.
      // Arbitrary-value form pulls the live CSS var (set by the theme
      // JS) and gives us a fully opaque fill so the menu doesn't bleed
      // the row text underneath.
      style={{ backgroundColor: "hsl(var(--popover))" }}
      className="absolute right-0 top-full z-10 mt-1 min-w-[7rem] overflow-hidden rounded border border-border text-popover-foreground shadow-lg"
    >
      {children}
    </div>
  );
}

// One row in the picker. Pulled out so the action-button branching
// (download / load / active / delete) reads as a single unit.
function ModelRow({
  m,
  isActive,
  isPending,
  busy,
  serverUp,
  onLoad,
  onDownload,
  onDelete,
}: {
  m: ModelInfo;
  isActive: boolean;
  isPending: boolean;
  busy: boolean;
  serverUp: boolean;
  onLoad: (n: string) => void;
  onDownload: (n: string) => void;
  onDelete: (n: string) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const downloading = !!m.download && m.download.phase !== "error";
  const downloadError = m.download?.phase === "error" ? m.download.error : null;

  // What the primary button does depends on disk state. Order:
  //   downloading → spinner+label, no click
  //   active (currently serving) → "Active" badge, no click
  //   missing weights → Download
  //   weights present → Load
  let primary: React.ReactNode;
  if (downloading) {
    primary = (
      <span className="flex items-center gap-1 text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>Downloading…</span>
      </span>
    );
  } else if (isActive) {
    primary = (
      <span className="flex items-center gap-1 text-green-500">
        <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
        Active
      </span>
    );
  } else if (!m.downloaded) {
    primary = (
      <button
        type="button"
        onClick={() => onDownload(m.name)}
        disabled={!m.available || busy}
        className="flex items-center gap-1 rounded border border-border bg-secondary px-2 py-0.5 text-[10px] hover:bg-accent disabled:opacity-50"
      >
        <Download className="h-3 w-3" />
        Download {m.sizeGb}
      </button>
    );
  } else {
    primary = (
      <button
        type="button"
        onClick={() => onLoad(m.name)}
        disabled={!m.available || busy}
        className="flex items-center gap-1 rounded border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] text-foreground hover:bg-primary/20 disabled:opacity-50"
      >
        {isPending ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Play className="h-3 w-3" />
        )}
        {isPending ? "Loading…" : "Load"}
      </button>
    );
  }

  // The overflow menu shows up whenever there's at least one applicable
  // action. Today that's just Delete; future items (open cache dir,
  // copy repo URL…) plug in the same way.
  const canDelete = m.downloaded && !isActive && !downloading;
  const hasMenu = canDelete;

  return (
    <div
      className={`flex flex-col rounded border px-2 py-1.5 transition-colors ${
        isActive
          ? "border-primary/40 bg-primary/5"
          : m.available
            ? "border-border bg-secondary/60"
            : "border-border bg-secondary/30"
      } ${!m.available ? "opacity-60" : !serverUp ? "opacity-60" : ""}`}
    >
      <div className="flex items-start gap-2">
        <div className="flex flex-1 flex-col">
          <span className="flex items-center gap-1.5 font-medium">
            {m.label}
            {m.downloaded && (
              <Check
                className="h-3 w-3 text-green-500"
                aria-label="downloaded"
              />
            )}
            {!m.available && (
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                soon
              </span>
            )}
          </span>
          <span className="text-muted-foreground">{m.hint}</span>
          <span className="text-[10px] text-muted-foreground">
            {m.requirements}
          </span>
        </div>
        <div className="flex flex-shrink-0 items-start gap-1">
          {primary}
          {hasMenu && (
            <div className="relative">
              <button
                type="button"
                onClick={() => setMenuOpen((v) => !v)}
                disabled={busy}
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                aria-label={`Actions for ${m.label}`}
                className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50"
              >
                <MoreHorizontal className="h-3.5 w-3.5" />
              </button>
              <RowMenu open={menuOpen} onClose={() => setMenuOpen(false)}>
                {canDelete && (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setMenuOpen(false);
                      onDelete(m.name);
                    }}
                    className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-[11px] text-red-400 hover:bg-accent"
                  >
                    <Trash2 className="h-3 w-3" />
                    Delete
                  </button>
                )}
              </RowMenu>
            </div>
          )}
        </div>
      </div>
      {m.download && <DownloadBar d={m.download} />}
      {downloadError && (
        <div className="mt-1 text-[10px] text-red-400">
          Download failed: {downloadError}
        </div>
      )}
    </div>
  );
}

export function ModelPicker() {
  const serverUp = useServerStore(selectIsRunning);
  const status = useServerStore((s) => s.inference);
  const pending = useServerStore((s) => s.pendingModel);
  const loadModel = useServerStore((s) => s.loadModel);
  const downloadModel = useServerStore((s) => s.downloadModel);
  const deleteModel = useServerStore((s) => s.deleteModel);
  const models = useServerStore((s) => s.models);

  // Local: surfaced action error. Cleared when the next action starts.
  const [error, setError] = useState<string | null>(null);
  // submitting tracks the in-flight POST. We share one slot across
  // load/download/delete since the picker only ever runs one at a time.
  const [submitting, setSubmitting] = useState<string | null>(null);
  const toast = useToast();

  // requireServer wraps actions that need the Go server up; surfaces
  // a toast rather than silently no-op'ing so the user knows why.
  const requireServer = useCallback(() => {
    if (!serverUp) {
      toast.show("Start the server first.", { variant: "warn" });
      return false;
    }
    return true;
  }, [serverUp, toast]);

  const onLoad = useCallback(
    async (name: string) => {
      if (submitting !== null || pending !== null) return;
      if (!requireServer()) return;
      setError(null);
      setSubmitting(name);
      try {
        await loadModel(name);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(null);
      }
    },
    [submitting, pending, requireServer, loadModel],
  );

  const onDownload = useCallback(
    async (name: string) => {
      if (submitting !== null) return;
      if (!requireServer()) return;
      setError(null);
      setSubmitting(name);
      try {
        await downloadModel(name);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(null);
      }
    },
    [submitting, requireServer, downloadModel],
  );

  // Delete confirmation is handled by the row's overflow menu —
  // opening the menu and clicking the explicit Delete item is itself
  // a deliberate two-step, so no extra JS confirm needed here.
  const onDelete = useCallback(
    async (name: string) => {
      if (submitting !== null) return;
      if (!requireServer()) return;
      setError(null);
      setSubmitting(name);
      try {
        await deleteModel(name);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(null);
      }
    },
    [submitting, requireServer, deleteModel],
  );

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
    : busy && status.state === "loading"
      ? "Preparing…"
      : null;

  return (
    <div className="fixed bottom-4 left-4 flex w-80 max-w-[90vw] flex-col gap-2 rounded-md border bg-card/80 p-3 text-xs backdrop-blur">
      <div className="font-medium text-foreground">Models</div>
      <div className="flex flex-col gap-1.5">
        {models.length === 0 && serverUp && (
          // REASON: catalog hasn't landed yet — first /models fetch is
          // in flight. Render a quiet placeholder rather than nothing
          // so the section doesn't visibly snap in.
          <div className="text-muted-foreground italic">Loading catalog…</div>
        )}
        {models.map((m) => {
          const isActive =
            status.model === m.name && status.state === "serving";
          const isPending = pending === m.name || submitting === m.name;
          return (
            <ModelRow
              key={m.name}
              m={m}
              isActive={isActive}
              isPending={isPending}
              busy={busy}
              serverUp={serverUp}
              onLoad={onLoad}
              onDownload={onDownload}
              onDelete={onDelete}
            />
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
