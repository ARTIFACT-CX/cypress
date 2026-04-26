// AREA: ui · STORE · BOOTSTRAP
// Wires the server store to its external sources of truth: the Tauri
// `server-status` event stream and the Go server's /status HTTP
// endpoint. Idempotent — calling init() twice is a no-op, so React
// StrictMode's double-effect doesn't double-subscribe.
//
// Lives outside the store module itself because zustand stores are
// best kept side-effect-free (so they can be imported in tests
// without spinning up listeners). This file is imported once from
// main.tsx where the side effects are wanted.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  SERVER_URL,
  useServerStore,
  type InferenceSnapshot,
  type ServerStatus,
} from "./serverStore";

// Polling cadences. Fast while a model is loading or a load is
// pending so the user sees live phase updates; slower while idle.
const POLL_FAST_MS = 500;
const POLL_SLOW_MS = 2000;

let started = false;
let unlistenServer: UnlistenFn | null = null;
let pollHandle: ReturnType<typeof setTimeout> | null = null;

function pollDelay(): number | null {
  const { status, inference, pendingModel } = useServerStore.getState();
  if (status.state !== "running") return null;
  if (
    pendingModel !== null ||
    inference.state === "loading" ||
    inference.state === "starting"
  ) {
    return POLL_FAST_MS;
  }
  return POLL_SLOW_MS;
}

async function pollOnce() {
  try {
    const res = await fetch(`${SERVER_URL}/status`);
    const snap = (await res.json()) as InferenceSnapshot;
    useServerStore.getState().setInference(snap);
  } catch {
    // Server might be mid-restart; just try again on next tick.
  }
}

function schedulePoll() {
  if (pollHandle !== null) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
  const delay = pollDelay();
  if (delay === null) return;
  pollHandle = setTimeout(async () => {
    await pollOnce();
    schedulePoll();
  }, delay);
}

export function initServerStore() {
  if (started) return;
  started = true;

  // STEP 1: hydrate initial server lifecycle once. The Rust side has
  // a current value even before our event listener is attached.
  invoke<ServerStatus>("server_status")
    .then((s) => {
      useServerStore.getState().setStatus(s);
      // Kick a poll immediately if already running so the inference
      // snapshot lands on the first paint instead of waiting an
      // interval.
      if (s.state === "running") {
        void pollOnce();
      }
      schedulePoll();
    })
    .catch(() => {});

  // STEP 2: subscribe to live transitions. Each event re-evaluates
  // poll cadence so we go fast/slow without a separate watcher.
  void listen<ServerStatus>("server-status", (e) => {
    useServerStore.getState().setStatus(e.payload);
    if (e.payload.state === "running") {
      void pollOnce();
    }
    schedulePoll();
  }).then((un) => {
    unlistenServer = un;
  });

  // STEP 3: re-evaluate poll cadence when pendingModel or inference
  // state changes — those drive the fast/slow switch. Subscribing to
  // the whole store is fine here; the schedule check is cheap.
  useServerStore.subscribe(schedulePoll);
}

// Test/HMR teardown. Not used in normal app lifetime — the listeners
// live as long as the webview does — but lets dev hot-reload start
// clean.
export function disposeServerStore() {
  started = false;
  if (pollHandle !== null) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
  if (unlistenServer) {
    unlistenServer();
    unlistenServer = null;
  }
}

if (import.meta.hot) {
  import.meta.hot.dispose(disposeServerStore);
}
