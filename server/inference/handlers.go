// AREA: inference · HANDLERS · INBOUND-ADAPTER
// HTTP routes for the inference feature. The UI hits these directly via
// fetch() from localhost — no Rust bridge needed since the traffic is
// intra-machine and same-origin from Tauri's perspective.
//
// This is inference's *inbound adapter* — the way external callers (the
// UI) drive the inference business logic in manager.go. Lives in the
// inference feature because every route here is about model lifecycle.

package inference

import (
	"encoding/json"
	"log"
	"net/http"
)

// RegisterRoutes attaches the inference HTTP routes onto the given mux.
// Caller (main.go) supplies the mux; we don't own it. Keeps the wiring
// in one place at the composition root.
func RegisterRoutes(mux *http.ServeMux, mgr *Manager) {
	mux.HandleFunc("/status", func(w http.ResponseWriter, _ *http.Request) {
		// Snapshot carries {state, model, device, phase, error} in one
		// atomic read — see Manager.Status.
		writeJSON(w, http.StatusOK, mgr.Status())
	})

	mux.HandleFunc("/models", func(w http.ResponseWriter, _ *http.Request) {
		// Static catalog + a per-call HF cache probe + any inflight
		// downloads. Cheap (a few stat()s + a map snapshot) so we
		// recompute per request rather than caching. The UI polls
		// this for live download progress.
		writeJSON(w, http.StatusOK, map[string]any{"models": mgr.ModelInfos()})
	})

	// REASON: scope download under the model's own path so the URL
	// shape mirrors the action and avoids a {name} pattern colliding
	// with a sibling literal "/models/download" (Go 1.22 ServeMux
	// rejects that overlap at registration time).
	mux.HandleFunc("POST /models/{name}/download", func(w http.ResponseWriter, r *http.Request) {
		name := r.PathValue("name")
		if name == "" {
			http.Error(w, "missing name", http.StatusBadRequest)
			return
		}
		if err := mgr.DownloadModel(name); err != nil {
			log.Printf("download_model %q rejected: %v", name, err)
			writeJSON(w, http.StatusConflict, map[string]any{"error": err.Error()})
			return
		}
		// 202: started, progress will land in /models entries.
		writeJSON(w, http.StatusAccepted, map[string]any{"ok": true, "name": name})
	})

	mux.HandleFunc("DELETE /models/{name}", func(w http.ResponseWriter, r *http.Request) {
		name := r.PathValue("name")
		if name == "" {
			http.Error(w, "missing name", http.StatusBadRequest)
			return
		}
		if err := mgr.DeleteModel(name); err != nil {
			log.Printf("delete_model %q rejected: %v", name, err)
			writeJSON(w, http.StatusConflict, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "name": name})
	})

	mux.HandleFunc("/model/load", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid json body", http.StatusBadRequest)
			return
		}
		if body.Name == "" {
			http.Error(w, "missing 'name'", http.StatusBadRequest)
			return
		}
		// REASON: LoadModel returns as soon as the manager has flipped to
		// starting/loading; the actual blocking work (HF download, weight
		// transfer) runs on a goroutine the manager owns. The UI drives
		// completion via /status polling. Errors here are pre-flight only
		// (e.g. another load already in progress); loader failures
		// surface through Snapshot.Error.
		if err := mgr.LoadModel(body.Name); err != nil {
			log.Printf("load_model %q rejected: %v", body.Name, err)
			writeJSON(w, http.StatusConflict, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusAccepted, map[string]any{
			"ok":    true,
			"model": body.Name,
		})
	})
}

// writeJSON keeps Content-Type correct and the status code explicit at
// the call site. Local to this file because no other inbound adapter in
// this feature exists yet.
func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
