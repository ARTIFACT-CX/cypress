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

// Bounded turn history. Long sessions still need a cap so the React
// tree doesn't balloon. 200 turns ≈ tens of minutes of back-and-forth.
const MAX_TURNS = 200;

// Turn-boundary heuristics. The model emits text tokens for the agent
// path and we infer user turns from mic level (no ASR is happening).
//
//   - USER_GATE: smoothed mic level above this counts as the user
//     speaking. The level is already smoothed in onMicLevel below;
//     0.03 sits comfortably above ambient noise but below quiet talk.
//   - TURN_GAP_MS: a same-role event arriving within this window of
//     the previous one continues the current turn; outside it, we
//     open a new turn. Same value for both sides — feels natural at
//     conversation pace.
//   - TURN_FLIP_DEBOUNCE_MS: while in turn X, a single brief event
//     from role Y inside this window is ignored. Prevents one stray
//     mic frame during agent playback from fragmenting the agent's
//     turn into a "user → agent → user" chain.
const USER_GATE = 0.03;
const TURN_GAP_MS = 1500;
const TURN_FLIP_DEBOUNCE_MS = 250;

export type TurnRole = "user" | "agent";

export type Turn = {
  id: string;
  role: TurnRole;
  // For agent turns this accumulates the streamed text tokens.
  // For user turns it stays empty — Moshi doesn't ASR the user input
  // in this pipeline, so we render user bubbles as activity rather
  // than text.
  text: string;
  startedAt: number;
  endedAt: number | null;
};

type VoiceStore = {
  state: SessionState;
  error: string | null;
  // Closed turns history. activeTurn (below) is shown alongside this
  // in the UI; keeping them split avoids re-allocating the whole
  // array on every token append.
  turns: Turn[];
  activeTurn: Turn | null;
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

// Last-event timestamps (ms) used to decide whether an incoming
// mic-active or text event extends the current turn or opens a new
// one. Module-level instead of in the store so updating them doesn't
// trigger a React re-render — they're plumbing, not view state.
let lastUserActivityAt = 0;
let lastAgentTextAt = 0;
let turnSeq = 0;
const newTurnId = () => `t${++turnSeq}`;

export const useVoiceStore = create<VoiceStore>((set, get) => {
  // Helpers operating on the live store. Kept inside the create
  // closure so they can reach set/get without exposing them.

  // Close out the active turn (if any), pushing it onto the history
  // with an endedAt timestamp. Caps history at MAX_TURNS by dropping
  // from the front so memory usage is bounded over a long session.
  const closeActive = (now: number) => {
    const { activeTurn, turns } = get();
    if (!activeTurn) return;
    const closed: Turn = { ...activeTurn, endedAt: now };
    const next = turns.length + 1 > MAX_TURNS
      ? [...turns.slice(turns.length - MAX_TURNS + 1), closed]
      : [...turns, closed];
    set({ activeTurn: null, turns: next });
  };

  // Open a fresh turn for the given role. Caller is responsible for
  // closing any pre-existing active turn first (closeActive).
  const openTurn = (role: TurnRole, now: number) => {
    set({
      activeTurn: {
        id: newTurnId(),
        role,
        text: "",
        startedAt: now,
        endedAt: null,
      },
    });
  };

  // Replace the whole activeTurn object so React notices the change.
  // Mutating in place would skip re-renders since zustand uses
  // reference equality on the slice.
  const appendTextToActive = (chunk: string) => {
    const { activeTurn } = get();
    if (!activeTurn) return;
    set({ activeTurn: { ...activeTurn, text: activeTurn.text + chunk } });
  };

  // Decide what to do when a new event arrives for `role`. If we're
  // already in a same-role turn within the gap window, extend it.
  // Otherwise close the current turn and open a new one.
  const ensureRole = (role: TurnRole, now: number): boolean => {
    const { activeTurn } = get();
    if (activeTurn && activeTurn.role === role) {
      // Same role; reuse the open turn. Caller appends as needed.
      return false;
    }
    if (activeTurn && activeTurn.endedAt === null) {
      // Different role — but only flip if the *current* turn's last
      // activity is old enough that a transient blip from the
      // incoming role shouldn't fragment it. (Incoming role is `role`,
      // so current turn role is the opposite.)
      const currentTurnLastActivity =
        role === "user" ? lastAgentTextAt : lastUserActivityAt;
      if (now - currentTurnLastActivity < TURN_FLIP_DEBOUNCE_MS) {
        return false;
      }
      closeActive(now);
    }
    openTurn(role, now);
    return true;
  };

  const ensureSession = (): VoiceSession => {
    if (session) return session;
    session = new VoiceSession({
      onState: (next, reason) => {
        set({ state: next });
        if (next === "error") set({ error: reason ?? "unknown error" });
        else if (next === "live") set({ error: null });
        // STEP: close the in-flight turn whenever the session winds
        // down. Without this the last bubble would render forever as
        // "live" (no endedAt) until the user starts a new session.
        if (next === "idle" || next === "closing" || next === "error") {
          closeActive(Date.now());
        }
      },
      onText: (chunk) => {
        const now = Date.now();
        // If we'd been quiet (or in a user turn) and this is the
        // first agent token, ensureRole opens a new agent turn. Same
        // role within the gap window just appends.
        if (
          !get().activeTurn ||
          get().activeTurn?.role !== "agent" ||
          now - lastAgentTextAt > TURN_GAP_MS
        ) {
          ensureRole("agent", now);
        }
        appendTextToActive(chunk);
        lastAgentTextAt = now;
      },
      onMicLevel: (level) => {
        micTail = smooth(micTail, level);
        set({ micLevel: micTail });
        // Treat the smoothed level crossing the gate as user activity.
        // We use the raw incoming `level` (not the smoothed tail) so
        // the turn opens promptly on the first loud frame, not after
        // smoothing has caught up.
        if (level < USER_GATE) return;
        const now = Date.now();
        if (
          !get().activeTurn ||
          get().activeTurn?.role !== "user" ||
          now - lastUserActivityAt > TURN_GAP_MS
        ) {
          ensureRole("user", now);
        }
        lastUserActivityAt = now;
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
    turns: [],
    activeTurn: null,
    micLevel: 0,
    playbackLevel: 0,

    start: async () => {
      // Wipe transcript history when explicitly starting a fresh
      // conversation. Stopping leaves history visible so the user
      // can read what was said after they tap to end.
      set({ error: null, turns: [], activeTurn: null });
      lastUserActivityAt = 0;
      lastAgentTextAt = 0;
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
