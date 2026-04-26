// AREA: ui · STORE · VOICE
// Reactive state for the in-app voice session. Wraps the imperative
// VoiceSession class (lib/voiceSession.ts) and surfaces its state +
// per-frame audio levels through a zustand store so any component
// can read them without prop-drilling.
//
// The VoiceSession instance itself is held in a module-level let
// rather than store state — it has no meaningful equality and is
// constructed lazily on first start. The reactive parts of the
// session live in the store.
//
// Audio levels are smoothed here (fast attack, slow decay) before
// being committed to the store, so the chromatic-aberration filter
// + any future level meter pulses gently instead of strobing.

import { create } from "zustand";
import { VoiceSession, type SessionState } from "../lib/voiceSession";

// Bounded transcript buffer. Inner-monologue tokens stream fast;
// without a cap a long session would balloon the React tree. 400
// entries ≈ a few minutes at Moshi's typical token rate.
const MAX_TRANSCRIPT = 400;

type VoiceStore = {
  state: SessionState;
  error: string | null;
  transcript: string[];
  micLevel: number;
  playbackLevel: number;

  // Actions. Called from buttons / effects; lifecycle of the
  // underlying VoiceSession is managed inside.
  start: () => Promise<void>;
  stop: () => Promise<void>;
  toggle: () => Promise<void>;
};

// SAFETY: module-level singleton. The session owns a WS, a mic, and
// an AudioContext — having more than one would race all three. React
// StrictMode will mount/unmount twice in dev but `ensureSession()` is
// idempotent (returns the existing instance) so that's fine.
let session: VoiceSession | null = null;

// Smoothing tail for level meters. Raw frames arrive every ~85ms and
// look strobey; an exponential decay turns them into a gentle pulse.
let micTail = 0;
let playTail = 0;
const smooth = (prev: number, level: number) =>
  level > prev ? level : prev * 0.85 + level * 0.15;

export const useVoiceStore = create<VoiceStore>((set, get) => {
  const ensureSession = (): VoiceSession => {
    if (session) return session;
    session = new VoiceSession({
      onState: (next, reason) => {
        set({ state: next });
        if (next === "error") set({ error: reason ?? "unknown error" });
        else if (next === "live") set({ error: null });
      },
      onText: (chunk) => {
        const prev = get().transcript;
        const next =
          prev.length + 1 > MAX_TRANSCRIPT
            ? [...prev.slice(prev.length - MAX_TRANSCRIPT + 1), chunk]
            : [...prev, chunk];
        set({ transcript: next });
      },
      onMicLevel: (level) => {
        micTail = smooth(micTail, level);
        set({ micLevel: micTail });
      },
      onPlaybackLevel: (level) => {
        playTail = smooth(playTail, level);
        set({ playbackLevel: playTail });
      },
    });
    return session;
  };

  return {
    state: "idle",
    error: null,
    transcript: [],
    micLevel: 0,
    playbackLevel: 0,

    start: async () => {
      set({ error: null, transcript: [] });
      try {
        await ensureSession().start();
      } catch {
        // setState already fired error via onState callback.
      }
    },
    stop: async () => {
      await session?.stop("user stopped");
    },
    toggle: async () => {
      const s = session?.getState() ?? "idle";
      if (s === "live") {
        await session!.stop("user stopped");
      } else if (s === "idle" || s === "error") {
        // Re-enter via start() so error/transcript wipes happen.
        await get().start();
      }
    },
  };
});

// REASON: dev hot-reload safety. Without this, an HMR replacement
// leaves the previous module's VoiceSession instance running with a
// dangling WS + mic permission. Vite calls dispose hooks before
// swapping the module, so we tear down here.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    void session?.stop("hmr");
    session = null;
  });
}
