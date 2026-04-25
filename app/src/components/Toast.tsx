// AREA: ui · TOAST
// Tiny global toast system. A provider holds the active toast queue; any
// component calls `useToast().show(...)` to enqueue one. Toasts auto-dismiss
// after a timeout and stack at the bottom-center of the viewport.
//
// SWAP: if we ever need richer toasts (actions, undo, progress), swap the
// implementation behind useToast for sonner or radix-toast — the public
// hook surface (show/dismiss) is intentionally minimal so callers don't
// have to change.

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

type ToastVariant = "info" | "warn" | "error" | "success";

type Toast = {
  id: number;
  message: string;
  variant: ToastVariant;
};

type ToastContextValue = {
  show: (message: string, opts?: { variant?: ToastVariant; duration?: number }) => void;
  dismiss: (id: number) => void;
};

const ToastContext = createContext<ToastContextValue | null>(null);

// Default visible time before auto-dismiss. Long enough to read a sentence,
// short enough that stale toasts don't pile up if the user is mashing
// buttons. Override per-call via the duration option.
const DEFAULT_DURATION_MS = 3500;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Monotonic id source. Refs avoid re-renders just to bump the counter.
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const show = useCallback<ToastContextValue["show"]>(
    (message, opts) => {
      const id = nextId.current++;
      const variant = opts?.variant ?? "info";
      const duration = opts?.duration ?? DEFAULT_DURATION_MS;
      setToasts((prev) => [...prev, { id, message, variant }]);
      // Auto-dismiss. We don't bother cancelling on unmount because the
      // provider lives at the app root and never unmounts in practice.
      window.setTimeout(() => dismiss(id), duration);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ show, dismiss }), [show, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    // Fail loudly at dev time — silent no-op toasts are a nightmare to
    // diagnose. Wrap the app in <ToastProvider> at the root.
    throw new Error("useToast must be used within <ToastProvider>");
  }
  return ctx;
}

// Variant → tailwind classes. Keeping this as a plain map (rather than CVA)
// because the toast surface is small and we don't need composition yet.
const VARIANT_STYLES: Record<ToastVariant, string> = {
  info: "border-border bg-card/90 text-foreground",
  // REASON: warn used to be amber, but in practice most "warnings" here are
  // soft prompts ("start the server first") that don't warrant a loud
  // color. Muted grey reads as a neutral hint without alarming the user.
  warn: "border-border bg-card/90 text-muted-foreground",
  error: "border-red-500/40 bg-red-500/10 text-red-200",
  success: "border-green-500/40 bg-green-500/10 text-green-200",
};

// SETUP: how many toasts are visually stacked at once. Older ones beyond
// this still exist (and auto-dismiss on their timer) but are hidden so
// the stack doesn't visually grow forever.
const VISIBLE_STACK = 3;

// Per-depth styles. depth 0 = newest (front, fully visible); higher
// depths sit behind it, slightly offset up and scaled down — sonner-style.
// On hover of the stack we spread the cards apart along Y so each is
// readable. The vertical offset assumes toasts are ~3rem tall; close
// enough since messages are kept short.
const STACK_STYLES: Record<number, string> = {
  0: "translate-y-0 scale-100 opacity-100 z-30",
  1: "-translate-y-1 scale-[0.96] opacity-80 z-20 group-hover/toasts:-translate-y-[3.25rem] group-hover/toasts:scale-100 group-hover/toasts:opacity-100",
  2: "-translate-y-2 scale-[0.92] opacity-60 z-10 group-hover/toasts:-translate-y-[6.5rem] group-hover/toasts:scale-100 group-hover/toasts:opacity-100",
};

function ToastViewport({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}) {
  // Newest first so depth = index. We slice to VISIBLE_STACK so older
  // ones drop off the visual stack rather than smearing further behind.
  const visible = [...toasts].reverse().slice(0, VISIBLE_STACK);

  // Fixed at bottom-center, above any other floating UI. pointer-events-none
  // on the wrapper means the toasts don't block clicks on the empty area
  // around them; each toast re-enables pointer events for itself so the X
  // button stays clickable.
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-4">
      <ol className="group/toasts relative flex h-12 w-full max-w-sm justify-center">
        {visible.map((t, depth) => (
          <li
            key={t.id}
            role="status"
            className={`pointer-events-auto absolute bottom-0 w-full origin-bottom transition-all duration-200 ${STACK_STYLES[depth]}`}
          >
            <div
              className={`flex items-start gap-3 rounded-md border px-3 py-2 text-xs shadow-lg backdrop-blur ${VARIANT_STYLES[t.variant]}`}
            >
              <span className="flex-1 leading-relaxed">{t.message}</span>
              <button
                type="button"
                onClick={() => onDismiss(t.id)}
                aria-label="Dismiss"
                className="text-muted-foreground hover:text-foreground"
              >
                ×
              </button>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
