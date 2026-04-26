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

import { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Loader2 } from "lucide-react";
import { selectIsModelReady, useServerStore } from "../store/serverStore";
import { useVoiceStore } from "../store/voiceStore";
import { cn } from "../lib/utils";

export function VoiceButton() {
  // Voice state — single-field selectors so this component only
  // re-renders when the relevant slice changes.
  const state = useVoiceStore((s) => s.state);
  const error = useVoiceStore((s) => s.error);
  const transcript = useVoiceStore((s) => s.transcript);
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

  const live = state === "live";
  const busy = state === "connecting" || state === "closing";

  // Auto-scroll transcript to the newest token. Append-only stream, no
  // observer needed.
  const tailRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [transcript]);

  if (!modelReady) return null;

  const label = live
    ? "Tap to end"
    : busy
      ? state === "connecting"
        ? "Connecting…"
        : "Closing…"
      : "Tap to talk";

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-6 flex flex-col items-center gap-3">
      {/* Transcript — only after the user has actually started once. */}
      {hasStarted && (
        <div className="pointer-events-auto h-20 w-72 overflow-y-auto rounded-md border bg-card/80 px-3 py-2 text-xs leading-relaxed text-foreground shadow-sm backdrop-blur">
          {transcript.length === 0 ? (
            <span className="text-muted-foreground">
              {live ? "Listening…" : "No transcript yet."}
            </span>
          ) : (
            <>
              {transcript.join("")}
              <div ref={tailRef} />
            </>
          )}
        </div>
      )}

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
  );
}
