// AREA: ui · REMOTES-PANEL
// Collapsible block in the server-control area. Lists saved remote
// profiles plus the "Local subprocess" entry, lets the user pick one as
// active, and offers add/edit/delete affordances. Picking a profile
// just flips the active id — actually bringing up the SSH tunnel
// happens when the user clicks "Start Server" (the existing button).
//
// Editing happens in a small inline form below the list. We deliberately
// don't use a separate modal: it'd require a focus-trap + dialog
// component, and this is a low-traffic settings surface where inline
// editing reads cleanly.

import { useEffect, useState } from "react";
import { Pencil, Plus, Trash2 } from "lucide-react";
import {
  newProfileDefaults,
  useRemoteStore,
  type RemoteProfile,
} from "../store/remoteStore";
import { cn } from "../lib/utils";

type EditState =
  | { mode: "none" }
  | { mode: "new"; draft: Partial<RemoteProfile> }
  | { mode: "edit"; id: string; draft: Partial<RemoteProfile> };

export function RemotesPanel() {
  const profiles = useRemoteStore((s) => s.profiles);
  const activeId = useRemoteStore((s) => s.activeId);
  const loading = useRemoteStore((s) => s.loading);
  const error = useRemoteStore((s) => s.error);
  const setActive = useRemoteStore((s) => s.setActive);
  const save = useRemoteStore((s) => s.save);
  const remove = useRemoteStore((s) => s.remove);

  const [edit, setEdit] = useState<EditState>({ mode: "none" });

  return (
    <div className="flex flex-col gap-2 rounded-md border bg-card/80 p-3 text-xs backdrop-blur">
      <div className="flex items-center justify-between">
        <span className="font-medium text-foreground">Worker location</span>
        <button
          type="button"
          onClick={() => setEdit({ mode: "new", draft: newProfileDefaults() })}
          disabled={loading || edit.mode !== "none"}
          className="flex items-center gap-1 rounded border border-border bg-secondary px-2 py-0.5 text-secondary-foreground hover:bg-accent disabled:opacity-50"
        >
          <Plus className="h-3 w-3" />
          Add remote
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-400/40 bg-red-500/10 p-1.5 text-red-400">
          {error}
        </div>
      )}

      {/* Local entry — always present, can't be deleted, can be active. */}
      <ProfileRow
        label="Local subprocess"
        sub="Run the worker on this machine"
        active={activeId === null}
        onPick={() => setActive(null)}
        disabled={loading}
      />

      {profiles.map((p) => (
        <ProfileRow
          key={p.id}
          label={p.name}
          sub={`${p.user}@${p.host}${p.jump_host ? ` via ${p.jump_host}` : ""}`}
          active={activeId === p.id}
          onPick={() => setActive(p.id)}
          onEdit={() => setEdit({ mode: "edit", id: p.id, draft: { ...p } })}
          onDelete={() => {
            if (confirm(`Delete profile "${p.name}"?`)) {
              void remove(p.id);
            }
          }}
          disabled={loading}
        />
      ))}

      {edit.mode !== "none" && (
        <ProfileForm
          draft={edit.draft}
          onCancel={() => setEdit({ mode: "none" })}
          onSubmit={async (d) => {
            await save({
              ...d,
              id: edit.mode === "edit" ? edit.id : "",
            });
            setEdit({ mode: "none" });
          }}
          loading={loading}
        />
      )}
    </div>
  );
}

function ProfileRow({
  label,
  sub,
  active,
  onPick,
  onEdit,
  onDelete,
  disabled,
}: {
  label: string;
  sub: string;
  active: boolean;
  onPick: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  disabled?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between rounded border px-2 py-1.5",
        active
          ? "border-sky-300/50 bg-sky-500/10"
          : "border-border bg-background/40 hover:bg-accent/40",
      )}
    >
      <button
        type="button"
        onClick={onPick}
        disabled={disabled}
        className="flex flex-1 flex-col items-start text-left disabled:opacity-50"
      >
        <span className="flex items-center gap-1.5 font-medium text-foreground">
          <span
            className={cn(
              "inline-block h-1.5 w-1.5 rounded-full",
              active ? "bg-sky-300" : "bg-muted-foreground/40",
            )}
          />
          {label}
        </span>
        <span className="text-[10px] text-muted-foreground">{sub}</span>
      </button>
      <div className="flex gap-1">
        {onEdit && (
          <button
            type="button"
            onClick={onEdit}
            disabled={disabled}
            aria-label={`Edit ${label}`}
            className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50"
          >
            <Pencil className="h-3 w-3" />
          </button>
        )}
        {onDelete && (
          <button
            type="button"
            onClick={onDelete}
            disabled={disabled}
            aria-label={`Delete ${label}`}
            className="rounded p-1 text-muted-foreground hover:bg-red-500/20 hover:text-red-400 disabled:opacity-50"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        )}
      </div>
    </div>
  );
}

function ProfileForm({
  draft,
  onCancel,
  onSubmit,
  loading,
}: {
  draft: Partial<RemoteProfile>;
  onCancel: () => void;
  onSubmit: (d: Partial<RemoteProfile>) => Promise<void>;
  loading: boolean;
}) {
  const [d, setD] = useState<Partial<RemoteProfile>>(draft);
  // REASON: re-sync when draft changes (switching from "new" to
  // "edit" without unmounting the form between).
  useEffect(() => setD(draft), [draft]);

  const valid =
    !!d.name?.trim() &&
    !!d.host?.trim() &&
    !!d.user?.trim() &&
    !!d.worker_dir?.trim();

  return (
    <form
      className="flex flex-col gap-2 rounded border border-border bg-background/60 p-2"
      onSubmit={(e) => {
        e.preventDefault();
        if (valid) void onSubmit(d);
      }}
    >
      <Field label="Name" value={d.name} onChange={(v) => setD({ ...d, name: v })} placeholder="Phoenix H100" />
      <Field label="Host" value={d.host} onChange={(v) => setD({ ...d, host: v })} placeholder="node_name.pace.gatech.edu" />
      <Field label="User" value={d.user} onChange={(v) => setD({ ...d, user: v })} placeholder="username" />
      <Field label="Jump host (optional)" value={d.jump_host} onChange={(v) => setD({ ...d, jump_host: v })} placeholder="login-phoenix.pace.gatech.edu" />
      <Field label="Worker dir" value={d.worker_dir} onChange={(v) => setD({ ...d, worker_dir: v })} placeholder="~/scratch/Cypress/worker" />
      <Field
        label="SSH key path (optional)"
        value={d.key_path}
        onChange={(v) => setD({ ...d, key_path: v })}
        placeholder="~/.ssh/id_ed25519 (leave blank for ssh's default search)"
      />
      <Field
        label="Port"
        value={String(d.port ?? 7843)}
        onChange={(v) => setD({ ...d, port: Number(v) || 7843 })}
        placeholder="7843"
      />

      <div className="flex justify-end gap-2 pt-1">
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-border bg-background px-2 py-1 text-foreground hover:bg-accent"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!valid || loading}
          className="rounded border border-sky-400/50 bg-sky-500/30 px-2 py-1 text-sky-100 hover:bg-sky-500/40 disabled:opacity-50"
        >
          Save
        </button>
      </div>
    </form>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value?: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <input
        type="text"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="rounded border border-border bg-background/80 px-2 py-1 text-foreground placeholder:text-muted-foreground/60"
      />
    </label>
  );
}
