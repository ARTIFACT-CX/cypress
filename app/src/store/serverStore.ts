// AREA: ui · STORE · SERVER
// Single source of truth for the orchestration server's state — both
// the Tauri-managed lifecycle (`server-status` event from src-tauri)
// and the Go-side inference snapshot (HTTP /status).
//
// Before this existed, three components (ServerControl, ModelPicker,
// VoiceButton) each subscribed to `server-status` and ran their own
// /status poll at different cadences. Consolidating into one store
// removes the duplicate listeners and the stagger between pollers.
//
// The store self-runs an adaptive poller while the server is up:
//   - 500ms cadence while a model is loading/starting (UI wants live
//     phase updates) or a load is pending from the user side.
//   - 2000ms cadence otherwise (just keeping the snapshot fresh).
//   - paused entirely when the Tauri server isn't running.
//
// Bootstrap (Tauri event listener + initial poll kick-off) is started
// by `bootstrap.ts`; this module just exposes the state and actions.

import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

// SETUP: Go server base URL. Matches listenAddr in server/main.go.
export const SERVER_URL = "http://127.0.0.1:7842";

// SWAP: keep in sync with ServerStatus in src-tauri/src/server.rs.
export type ServerStatus =
  | { state: "idle" }
  | { state: "starting" }
  | { state: "running" }
  | { state: "stopping" }
  | { state: "error"; message: string };

// Mirror of server/inference/manager.go's State enum.
export type InferenceState =
  | "idle"
  | "starting"
  | "ready"
  | "loading"
  | "serving";

// Mirror of server/inference/manager.go's Snapshot struct.
export type InferenceSnapshot = {
  state: InferenceState;
  model: string;
  device: string;
  phase: string;
  error?: string;
  // "local" (subprocess) or "remote" (gRPC over TCP+TLS / SSH tunnel).
  // Lets the popup label the transport without peeking at env vars.
  transport: "local" | "remote";
  // Present only when transport == "remote". Drives the unreachable
  // banner — when reachable=false, lastError carries the dial error.
  remote?: RemoteStatus;
};

export type RemoteStatus = {
  url: string;
  reachable: boolean;
  lastError?: string;
  lastChecked: string; // RFC3339
};

const EMPTY_SNAPSHOT: InferenceSnapshot = {
  state: "idle",
  model: "",
  device: "",
  phase: "",
  transport: "local",
};

// Per-download progress, mirrored from server/inference/catalog.go's
// DownloadProgress. Total stays 0 if HF didn't return file sizes (rare
// but possible) — the UI renders an indeterminate state in that case.
export type DownloadProgress = {
  phase: "starting" | "downloading" | "error";
  file: string;
  fileIndex: number;
  fileCount: number;
  downloaded: number;
  total: number;
  error?: string;
};

// Catalog row mirrored from server/inference/catalog.go's ModelInfo.
// Refreshed on a slow timer alongside the inference snapshot so the
// "downloaded" badge updates after a successful load without forcing
// a manual reload.
export type ModelInfo = {
  name: string;
  label: string;
  hint: string;
  backend: string;
  repo: string;
  files: string[];
  sizeGb: string;
  requirements: string;
  available: boolean;
  downloaded: boolean;
  download?: DownloadProgress;
};

type ServerStore = {
  status: ServerStatus;
  inference: InferenceSnapshot;
  // Name of a model the user has just clicked to load, before the
  // server flips its own state. Lets the polling loop know to stay in
  // fast-cadence mode until the requested model is actually serving,
  // and lets ModelPicker show "loading…" without local state.
  pendingModel: string | null;
  // Static catalog + dynamic download status. Populated once on first
  // /models fetch and refreshed when the inference state changes
  // (e.g. a load completing flips a model's downloaded flag).
  models: ModelInfo[];
  // True while the server's eager remote-platform probe is in flight
  // (remote-worker mode only). UI renders a spinner over the catalog
  // instead of an empty grid. Always false in local mode.
  catalogLoading: boolean;

  // --- internal setters used by bootstrap + actions -------------------
  setStatus: (s: ServerStatus) => void;
  setInference: (s: InferenceSnapshot) => void;
  setPendingModel: (name: string | null) => void;
  setModels: (m: ModelInfo[]) => void;
  setCatalog: (c: { models: ModelInfo[]; loading: boolean }) => void;

  // --- actions --------------------------------------------------------
  startServer: () => Promise<void>;
  stopServer: () => Promise<void>;
  loadModel: (name: string) => Promise<void>;
  downloadModel: (name: string) => Promise<void>;
  cancelDownload: (name: string) => Promise<void>;
  deleteModel: (name: string) => Promise<void>;
};

export const useServerStore = create<ServerStore>((set, get) => ({
  status: { state: "idle" },
  inference: EMPTY_SNAPSHOT,
  pendingModel: null,
  models: [],
  catalogLoading: false,

  setStatus: (status) => {
    set({ status });
    // When the server leaves "running" the inference snapshot is stale
    // by definition — wipe it so a stopped-server UI doesn't keep
    // showing "Moshi serving" from before. Same idea handled in the
    // old ModelPicker via a separate effect.
    if (status.state !== "running") {
      set({
        inference: EMPTY_SNAPSHOT,
        pendingModel: null,
        models: [],
        catalogLoading: false,
      });
    }
  },
  setInference: (inference) => {
    // Auto-clear pendingModel once the server confirms the requested
    // model is serving (or any error surfaced). Keeps the store
    // self-consistent without the caller having to remember.
    const { pendingModel } = get();
    if (pendingModel !== null) {
      if (inference.error) {
        set({ pendingModel: null });
      } else if (
        inference.state === "serving" &&
        inference.model === pendingModel
      ) {
        set({ pendingModel: null });
      }
    }
    set({ inference });
  },
  setPendingModel: (pendingModel) => set({ pendingModel }),
  setModels: (models) => set({ models }),
  setCatalog: ({ models, loading }) => set({ models, catalogLoading: loading }),

  startServer: async () => {
    await invoke("start_server");
  },
  stopServer: async () => {
    await invoke("stop_server");
  },
  loadModel: async (name) => {
    // REASON: pendingModel goes up *before* the POST so the polling
    // loop kicks into fast cadence immediately. Cleared on terminal
    // state via setInference (above) or on POST failure (below).
    set({ pendingModel: name });
    try {
      const res = await fetch(`${SERVER_URL}/model/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
    } catch (e) {
      set({ pendingModel: null });
      throw e;
    }
  },
  downloadModel: async (name) => {
    // Fire-and-forget at the HTTP level — the 202 lands quickly and
    // progress streams through /models polling. Optimistically seed a
    // local "starting" entry so the UI flips to "downloading…"
    // immediately, before the bootstrap poller catches up.
    const { models } = get();
    set({
      models: models.map((m) =>
        m.name === name
          ? {
              ...m,
              download: {
                phase: "starting",
                file: "",
                fileIndex: 0,
                fileCount: m.files.length || 1,
                downloaded: 0,
                total: 0,
              },
            }
          : m,
      ),
    });
    const res = await fetch(
      `${SERVER_URL}/models/${encodeURIComponent(name)}/download`,
      { method: "POST" },
    );
    if (!res.ok) {
      // Roll the optimistic state back so the user sees the failure
      // landing on this row, not lingering as a hung download.
      const body = await res.json().catch(() => ({}));
      const after = get().models.map((m) =>
        m.name === name ? { ...m, download: undefined } : m,
      );
      set({ models: after });
      throw new Error(body.error || `HTTP ${res.status}`);
    }
  },
  cancelDownload: async (name) => {
    const res = await fetch(
      `${SERVER_URL}/models/${encodeURIComponent(name)}/download`,
      { method: "DELETE" },
    );
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    // Optimistic clear so the row reverts to "Download" before the next
    // /models poll lands. The server-side slot is cleared via the
    // download_error("cancelled") event the worker emits on its way out.
    const after = get().models.map((m) =>
      m.name === name ? { ...m, download: undefined } : m,
    );
    set({ models: after });
  },
  deleteModel: async (name) => {
    const res = await fetch(
      `${SERVER_URL}/models/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    // Optimistic update: flip downloaded false locally so the UI
    // reflects the delete before the next /models poll lands.
    const after = get().models.map((m) =>
      m.name === name ? { ...m, downloaded: false } : m,
    );
    set({ models: after });
  },
}));

// --- selectors --------------------------------------------------------
//
// Single-field selectors keep components subscribed to the smallest
// slice they care about — no re-render storms when an unrelated field
// changes. Components that need multiple fields should call the store
// multiple times rather than building tuples (which break referential
// equality every render in zustand v5).

export const selectIsRunning = (s: ServerStore) =>
  s.status.state === "running";

export const selectIsModelReady = (s: ServerStore) => {
  if (s.status.state !== "running") return false;
  const inf = s.inference;
  return inf.state === "serving" && !!inf.model && !inf.error;
};
