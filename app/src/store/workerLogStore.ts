// AREA: ui · STORE · WORKER-LOG
// Ringbuffered tail of stderr lines forwarded by the Tauri shell from
// the remote worker SSH child. Mirrors WORKER_LOG_EVENT in
// app/src-tauri/src/remote/ssh.rs — one line per event.
//
// Bounded so a chatty worker can't blow up the React tree. Cleared on
// every server-state transition out of "running" so the tail you're
// staring at is always from the current connection.

import { create } from "zustand";

const MAX_LINES = 300;

type WorkerLogStore = {
  lines: string[];
  // Visibility toggle — separate from the lines themselves so toggling
  // the panel doesn't trash the tail.
  open: boolean;

  push: (line: string) => void;
  clear: () => void;
  toggle: () => void;
  setOpen: (open: boolean) => void;
};

export const useWorkerLogStore = create<WorkerLogStore>((set, get) => ({
  lines: [],
  open: false,

  push: (line) =>
    set((s) => {
      const next = s.lines.length >= MAX_LINES
        ? [...s.lines.slice(-MAX_LINES + 1), line]
        : [...s.lines, line];
      return { lines: next };
    }),

  clear: () => set({ lines: [] }),

  toggle: () => set({ open: !get().open }),
  setOpen: (open) => set({ open }),
}));
