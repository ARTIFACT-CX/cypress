// AREA: models · INFO
// The wire shape (ModelInfo) the UI consumes from GET /models, plus
// the merge that combines the static catalog with cache state and
// any inflight download progress. Lives in models/ rather than
// downloads/ because ModelInfo is read-only data — every consumer
// (HTTP handler, future settings UI) gets the same struct.

package models

// ModelInfo is the per-model row returned by GET /models. Field names
// are the exact JSON shape the UI consumes.
type ModelInfo struct {
	Name         string   `json:"name"`         // identifier passed to /model/load
	Label        string   `json:"label"`        // display name
	Hint         string   `json:"hint"`         // short tagline (parameters · capability)
	Backend      string   `json:"backend"`      // "mlx", "torch", "—"
	Repo         string   `json:"repo"`         // HF repo (e.g. "kyutai/moshiko-mlx-q8")
	Files        []string `json:"files"`        // weight filenames within the repo
	SizeGB       string   `json:"sizeGb"`       // approximate disk + RAM footprint
	Requirements string   `json:"requirements"` // human-readable ram/vram + device hints
	Available    bool     `json:"available"`    // false = visible but not yet implemented
	// Downloaded means "weights are fully on disk" — true when the HF
	// cache has all blobs and there's no .incomplete file. The manifest
	// is internal bookkeeping; we don't surface it as a separate flag.
	Downloaded bool `json:"downloaded"`
	// Download is non-nil while a download is in flight. Lets the UI
	// render a progress bar without polling a separate endpoint.
	Download *DownloadProgress `json:"download,omitempty"`
}

// DownloadProgress mirrors the worker's download_progress event,
// surfaced to the UI through GET /models. Bytes are best-effort —
// HF's metadata API can omit sizes on private repos, in which case
// Total stays 0 and the UI renders an indeterminate spinner. Lives
// in models/ alongside ModelInfo so both the producer (downloads)
// and the consumer (the catalog merge below) share the same type.
type DownloadProgress struct {
	Phase      string `json:"phase"`      // "starting" | "downloading" | "error"
	File       string `json:"file"`       // current file being pulled
	FileIndex  int    `json:"fileIndex"`  // 0-based
	FileCount  int    `json:"fileCount"`  // total files in the install
	Downloaded int64  `json:"downloaded"` // cumulative bytes
	Total      int64  `json:"total"`      // estimated total bytes (0 if unknown)
	Error      string `json:"error,omitempty"`
}

// ModelInfos returns the catalog with download status filled in.
// Computed fresh on each call — the cache directory could change
// between calls if the user kicks off a load. inflight maps model
// name → live download progress and is merged in for any matching
// entry; pass nil if no downloads are tracked.
func ModelInfos(inflight map[string]*DownloadProgress) []ModelInfo {
	root := HubCacheDir()
	cat := Catalog()
	out := make([]ModelInfo, 0, len(cat))
	for _, e := range cat {
		info := ModelInfo{
			Name:         e.Name,
			Label:        e.Label,
			Hint:         e.Hint,
			Backend:      e.Backend,
			Repo:         e.Repo,
			Files:        append([]string(nil), e.Files...),
			SizeGB:       e.SizeGB,
			Requirements: e.Requirements,
			Available:    e.Available,
			Downloaded:   IsRepoCached(root, e.Repo),
		}
		if inflight != nil {
			if p, ok := inflight[e.Name]; ok && p != nil {
				cp := *p
				info.Download = &cp
			}
		}
		out = append(out, info)
	}
	return out
}
