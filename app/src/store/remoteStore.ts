// AREA: ui · STORE · REMOTE
// Reactive state for the remote-worker profiles managed by the Tauri
// shell. Mirrors the Rust commands in app/src-tauri/src/remote/mod.rs:
//
//   list_remote_profiles   → load on app boot + after every CRUD
//   save_remote_profile    → upsert; returns the canonical profile
//   delete_remote_profile  → remove by id
//   set_active_profile     → null = local subprocess; uuid = remote
//
// Active profile drives serverStore.startServer's behavior: with a
// remote selected, the Rust side brings up the SSH tunnel before
// spawning the Go child and threads CYPRESS_REMOTE_URL/_TOKEN onto its
// env. There's no "Connect" command — Start Server *is* connect when a
// remote is active.

import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";

// SWAP: keep in sync with RemoteProfile in src-tauri/src/remote/profiles.rs.
export type RemoteProfile = {
  id: string;
  name: string;
  host: string;
  user: string;
  jump_host?: string;
  jump_user?: string;
  port: number;
  key_path?: string;
  worker_dir: string;
  family: string;
};

export type ProfilesView = {
  profiles: RemoteProfile[];
  active_id: string | null;
};

type RemoteStore = {
  profiles: RemoteProfile[];
  // null = "Local subprocess" mode; otherwise the id of the active
  // remote profile.
  activeId: string | null;
  // Loading state for the initial list fetch + every CRUD round-trip.
  // Used to disable the Save button while a write is in flight.
  loading: boolean;
  error: string | null;

  refresh: () => Promise<void>;
  save: (p: Partial<RemoteProfile> & { id?: string }) => Promise<RemoteProfile>;
  remove: (id: string) => Promise<void>;
  setActive: (id: string | null) => Promise<void>;
};

// Defaults applied when the user is creating a new profile from scratch.
// Keeps the form prefilled with sensible values so the typical case is
// just "name + host + user + worker_dir".
export const newProfileDefaults: () => Partial<RemoteProfile> = () => ({
  id: "",
  name: "",
  host: "",
  user: "",
  jump_host: "",
  jump_user: "",
  port: 7843,
  key_path: "",
  worker_dir: "",
  family: "moshi",
});

export const useRemoteStore = create<RemoteStore>((set, get) => ({
  profiles: [],
  activeId: null,
  loading: false,
  error: null,

  refresh: async () => {
    set({ loading: true, error: null });
    try {
      const view = (await invoke("list_remote_profiles")) as ProfilesView;
      set({
        profiles: view.profiles,
        activeId: view.active_id ?? null,
        loading: false,
      });
    } catch (e) {
      set({ loading: false, error: String(e) });
    }
  },

  save: async (p) => {
    set({ loading: true, error: null });
    try {
      // REASON: trim empty optional fields. The UI binds them as ""
      // because <input> can't represent "absent"; the Rust side wants
      // them omitted entirely so serde's #[skip_serializing_if]
      // matches. Empty string would still serialize as "" which the
      // SSH builder would happily try to use as a host/key path.
      const cleaned: RemoteProfile = {
        id: p.id ?? "",
        name: (p.name ?? "").trim(),
        host: (p.host ?? "").trim(),
        user: (p.user ?? "").trim(),
        port: p.port ?? 7843,
        worker_dir: (p.worker_dir ?? "").trim(),
        family: p.family ?? "moshi",
        ...(p.jump_host && p.jump_host.trim()
          ? { jump_host: p.jump_host.trim() }
          : {}),
        ...(p.jump_user && p.jump_user.trim()
          ? { jump_user: p.jump_user.trim() }
          : {}),
        ...(p.key_path && p.key_path.trim()
          ? { key_path: p.key_path.trim() }
          : {}),
      };
      const saved = (await invoke("save_remote_profile", {
        profile: cleaned,
      })) as RemoteProfile;
      // Re-fetch rather than splice locally — the source of truth is
      // the Rust file, and a refresh() costs us one extra IPC for a
      // guaranteed-consistent list.
      await get().refresh();
      return saved;
    } catch (e) {
      set({ loading: false, error: String(e) });
      throw e;
    }
  },

  remove: async (id) => {
    set({ loading: true, error: null });
    try {
      await invoke("delete_remote_profile", { id });
      await get().refresh();
    } catch (e) {
      set({ loading: false, error: String(e) });
      throw e;
    }
  },

  setActive: async (id) => {
    set({ loading: true, error: null });
    try {
      await invoke("set_active_profile", { id });
      // Optimistic — server file already updated; refresh is fine but
      // the round-trip is what we'd surface to the user as "applying."
      set({ activeId: id, loading: false });
    } catch (e) {
      set({ loading: false, error: String(e) });
      throw e;
    }
  },
}));
