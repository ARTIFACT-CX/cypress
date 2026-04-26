// AREA: ui · VOICE · BUTTON
// Tap-to-converse button + live transcript, fixed at the bottom of the
// window. Reads voice state from the global store; the imperative
// VoiceSession lives behind that store.
//
// Visibility rules:
//   - The button only appears once the Tauri-managed server is running
//     AND the inference manager reports a model serving with no error.
//     Both gates come from the server store.
//   - The transcript stays hidden until the user starts a conversation
//     for the first time, then sticks around so they can read what was
//     said after they tap to end.
//
// Click = toggle. The model is duplex, so "live" means both directions
// are open; tap again to end the conversation.

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Loader2, Mic, MicOff } from "lucide-react";
import { selectIsModelReady, useServerStore } from "../store/serverStore";
import { useVoiceStore, type Turn } from "../store/voiceStore";
import { cn } from "../lib/utils";

// Format wall-clock as HH:MM (local). Used as the small subscript on
// each turn bubble — precision is to-the-minute since "5:42pm" is
// what a human cares about, not seconds.
function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

// Format an active user-turn's running duration as "0:05" so the
// bubble shows a live counter while the user is still speaking.
function formatDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

// Single turn rendered as a chat bubble. User turns align right with
// no text content (we don't ASR the user); agent turns align left
// and stream text live with a soft cursor while open.
function TurnBubble({ turn, now }: { turn: Turn; now: number }) {
  const isUser = turn.role === "user";
  const isOpen = turn.endedAt === null;
  const elapsed = (turn.endedAt ?? now) - turn.startedAt;

  // User bubbles get a duration label (live counter while open,
  // final length once closed). Agent bubbles show their accumulated
  // text plus a blinking cursor while the model is still emitting.
  const body = isUser ? (
    <span className="flex items-center gap-1.5">
      <span className="font-medium">You</span>
      <span className="text-muted-foreground">·</span>
      <span className="text-muted-foreground tabular-nums">
        {isOpen ? "Speaking…" : `Spoke ${formatDuration(elapsed)}`}
      </span>
    </span>
  ) : (
    <span className="whitespace-pre-wrap break-words">
      <span className="mr-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
        Cypress
      </span>
      {turn.text || (isOpen ? "…" : "(no response)")}
      {isOpen && (
        <span className="ml-0.5 inline-block w-[1ch] animate-pulse">▌</span>
      )}
    </span>
  );

  return (
    <div className={cn("flex flex-col", isUser ? "items-end" : "items-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-md border px-2 py-1 text-xs leading-relaxed shadow-sm",
          isUser
            ? "border-sky-300/40 bg-sky-500/10 text-foreground"
            : "border-border bg-secondary/80 text-foreground",
        )}
      >
        {body}
      </div>
      <div className="mt-0.5 px-1 text-[10px] text-muted-foreground tabular-nums">
        {formatTime(turn.startedAt)}
      </div>
    </div>
  );
}

export function VoiceButton() {
  // Voice state — single-field selectors so this component only
  // re-renders when the relevant slice changes.
  const state = useVoiceStore((s) => s.state);
  const error = useVoiceStore((s) => s.error);
  const turns = useVoiceStore((s) => s.turns);
  const activeTurn = useVoiceStore((s) => s.activeTurn);
  const toggle = useVoiceStore((s) => s.toggle);

  // Visibility gate. modelReady covers both "Tauri server running"
  // and "Go-side model loaded with no error" — see selectIsModelReady.
  const modelReady = useServerStore(selectIsModelReady);

  // STEP 1: track whether the user has started a conversation at least
  // once. Drives transcript visibility — we don't want an empty box
  // floating on first launch, but we do want the transcript to persist
  // after the user taps to end so they can read the last reply.
  const [hasStarted, setHasStarted] = useState(false);
  useEffect(() => {
    if (state === "connecting" || state === "live") setHasStarted(true);
  }, [state]);

  // Local UI state for the transcript panel. Minimizing collapses the
  // panel to just its header bar so the user can keep an eye on the
  // logo / app surface without the scrollable bubble list in the way.
  const [minimized, setMinimized] = useState(false);

  const live = state === "live";
  const busy = state === "connecting" || state === "closing";

  // Tick a "now" value once a second while a turn is active so the
  // user-turn duration counter ("Speaking… 0:07") advances without
  // every level-meter update forcing a re-render. The interval only
  // runs while there's an open turn, so an idle UI is at rest.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!activeTurn) return;
    setNow(Date.now());
    const handle = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(handle);
  }, [activeTurn]);

  // Combine closed turns + active turn for rendering. activeTurn is
  // the only thing that mutates frame-to-frame (text appends, mic
  // updates), so keeping it separate in the store avoided rebuilding
  // the closed-turn array on every keystroke.
  const visibleTurns = useMemo(
    () => (activeTurn ? [...turns, activeTurn] : turns),
    [turns, activeTurn],
  );

  // Auto-scroll the transcript to the newest turn. We use a tail
  // sentinel + scrollIntoView so this works regardless of how the
  // panel grows (variable bubble heights). v0.1 always auto-scrolls;
  // a future enhancement could hold scroll if the user has manually
  // moved away from the bottom.
  const tailRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (minimized) return;
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [visibleTurns, activeTurn?.text, minimized]);

  if (!modelReady) return null;

  const label = live
    ? "Tap to end"
    : busy
      ? state === "connecting"
        ? "Connecting…"
        : "Closing…"
      : "Tap to talk";

  return (
    <>
      {/* Transcript — pinned to the top-right of the window, separate
          from the bottom-center button stack. Only mounts after the
          user has started a conversation at least once so first-launch
          is uncluttered. The panel has its own header bar with a
          minimize toggle; when minimized only the header remains so
          the bubble list gets out of the way without losing the
          user's context. */}
      {hasStarted && (
        <div className="pointer-events-auto fixed right-4 top-12 flex w-[24rem] flex-col rounded-md border bg-card/80 text-xs shadow-sm backdrop-blur">
          <div className="flex items-center justify-between border-b border-border/60 px-3 py-1.5">
            <span className="font-medium text-foreground">Transcript</span>
            <button
              type="button"
              onClick={() => setMinimized((m) => !m)}
              aria-label={minimized ? "Expand transcript" : "Minimize transcript"}
              aria-pressed={minimized}
              className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              {minimized ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronUp className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
          {!minimized && (
            <div className="flex h-72 flex-col gap-2 overflow-y-auto px-3 py-3">
              {visibleTurns.length === 0 ? (
                <span className="text-muted-foreground">
                  {live ? "Listening…" : "No transcript yet."}
                </span>
              ) : (
                <>
                  {visibleTurns.map((t) => (
                    <TurnBubble key={t.id} turn={t} now={now} />
                  ))}
                  <div ref={tailRef} />
                </>
              )}
            </div>
          )}
        </div>
      )}

      <div className="pointer-events-none fixed inset-x-0 bottom-6 flex flex-col items-center gap-3">
      {/* Status / error caption. min-h reserves a row so the button
          doesn't jump when the caption appears/disappears. */}
      <div className="min-h-[1rem] text-center text-[10px] text-muted-foreground">
        {error ? <span className="text-red-400">{error}</span> : label}
      </div>

      {/* Button. The audio-reactive visual lives on the logo (see
          useChromaticAberration in App), so this stays a calm static
          control — no competing pulse here. */}
      <div className="pointer-events-auto relative flex h-14 w-14 items-center justify-center">
        <button
          type="button"
          onClick={toggle}
          disabled={busy}
          aria-pressed={live}
          aria-label={label}
          className={cn(
            "relative flex h-12 w-12 items-center justify-center rounded-full border shadow-md transition-colors",
            live
              ? "border-sky-300/50 bg-sky-500/20 text-sky-100 hover:bg-sky-500/30"
              : "border-border bg-card text-foreground hover:bg-accent",
            busy && "cursor-wait opacity-70",
          )}
        >
          {busy ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : live ? (
            <MicOff className="h-5 w-5" />
          ) : (
            <Mic className="h-5 w-5" />
          )}
        </button>
      </div>
      </div>
    </>
  );
}
