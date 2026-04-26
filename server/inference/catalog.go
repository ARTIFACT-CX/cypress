// AREA: inference · CATALOG
// Static metadata for the set of models Cypress knows how to load,
// plus a filesystem probe of the Hugging Face hub cache so the UI can
// show "downloaded" vs "needs download" before the user clicks load.
//
// The catalog mirrors what the Python worker actually loads — the
// repo names here have to match worker/models/*.py defaults. Keeping
// it Go-side means the picker UI can render this list without first
// spinning up the Python worker (which costs several seconds and only
// runs on demand). The trade-off is that the two sides need to stay
// in sync; there's exactly one knob today (CYPRESS_MOSHI_REPO) and we
// document the linkage in comments.
//
// Download status is "best effort" — we look for the canonical HF hub
// cache layout (~/.cache/huggingface/hub/models--<org>--<name>/) and
// report `downloaded: true` if any snapshot directory under that path
// is non-empty. We don't verify individual file hashes; HF's own
// loaders will do that at load time and trigger a refetch if needed.

package inference

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

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
// Total stays 0 and the UI renders an indeterminate spinner.
type DownloadProgress struct {
	Phase      string `json:"phase"`      // "starting" | "downloading" | "error"
	File       string `json:"file"`       // current file being pulled
	FileIndex  int    `json:"fileIndex"`  // 0-based
	FileCount  int    `json:"fileCount"`  // total files in the install
	Downloaded int64  `json:"downloaded"` // cumulative bytes
	Total      int64  `json:"total"`      // estimated total bytes (0 if unknown)
	Error      string `json:"error,omitempty"`
}

// catalogEntry is the static description; the dynamic Downloaded flag
// is computed per request. Kept private so callers can't accidentally
// mutate the catalog.
type catalogEntry struct {
	Name         string
	Label        string
	Hint         string
	Backend      string
	Repo         string
	Files        []string // weight file names within the repo
	SizeGB       string
	Requirements string
	Available    bool
}

// REASON: backend selection mirrors worker/models/__init__.py's
// _default_moshi_backend. We can't import that logic, so we duplicate
// the platform check; if the worker logic ever grows more complex
// we'll need an IPC query instead.
// SETUP: shared file names. mimi + tokenizer files are identical across
// all moshiko-* repos (mlx-q4 / mlx-q8 / mlx-bf16 / pytorch-bf16); only
// the LM weights filename varies by quantization. Kept as constants so
// any future repo addition reuses these without re-encoding the
// filename strings.
const (
	moshiMimiFile      = "tokenizer-e351c8d8-checkpoint125.safetensors"
	moshiTokenizerFile = "tokenizer_spm_32k_3.model"
	moshiTorchLMFile   = "model.safetensors"
	moshiMlxQ8LMFile   = "model.q8.safetensors"
)

func defaultMoshiEntry() catalogEntry {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return catalogEntry{
			Name:         "moshi",
			Label:        "Moshi",
			Hint:         "3.5B · duplex · lighter",
			Backend:      "mlx",
			Repo:         "kyutai/moshiko-mlx-q8",
			Files:        []string{moshiMlxQ8LMFile, moshiMimiFile, moshiTokenizerFile},
			SizeGB:       "~4 GB",
			Requirements: "~6 GB unified memory · Apple Silicon",
			Available:    true,
		}
	}
	return catalogEntry{
		Name:         "moshi",
		Label:        "Moshi",
		Hint:         "3.5B · duplex · lighter",
		Backend:      "torch",
		Repo:         "kyutai/moshiko-pytorch-bf16",
		Files:        []string{moshiTorchLMFile, moshiMimiFile, moshiTokenizerFile},
		SizeGB:       "~14 GB",
		Requirements: "~14 GB VRAM · CUDA preferred",
		Available:    true,
	}
}

// catalog returns the static set; Available reflects whether the
// worker has a registered loader. PersonaPlex is listed so users see
// what's coming but can't click it yet — the moment its loader lands
// in the worker we just flip Available.
func catalog() []catalogEntry {
	return []catalogEntry{
		defaultMoshiEntry(),
		{
			Name:         "personaplex",
			Label:        "PersonaPlex",
			Hint:         "7B · duplex + persona",
			Backend:      "torch",
			Repo:         "", // not yet published
			SizeGB:       "~14 GB",
			Requirements: "~16 GB VRAM · NVIDIA",
			Available:    false,
		},
	}
}

// catalogEntryByName returns the static entry for a given model name,
// or nil. Used by Manager when starting a download — it needs to know
// which repo + files to ask the worker to pull.
func catalogEntryByName(name string) *catalogEntry {
	for _, e := range catalog() {
		if e.Name == name {
			cp := e
			return &cp
		}
	}
	return nil
}

// ModelInfos returns the catalog with download status filled in.
// Computed fresh on each call — the cache directory could change
// between calls if the user kicks off a load. inflight maps model
// name → live download progress and is merged in for any matching
// entry; pass nil if no downloads are tracked.
func ModelInfos(inflight map[string]*DownloadProgress) []ModelInfo {
	root := hubCacheDir()
	out := make([]ModelInfo, 0, len(catalog()))
	for _, e := range catalog() {
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
			Downloaded:   isRepoCached(root, e.Repo),
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

// repoCacheDir returns the on-disk directory HF uses for a given
// repo. Shared by isRepoCached and DeleteModel so they stay in sync
// on the layout assumption.
func repoCacheDir(root, repo string) string {
	return filepath.Join(root, "models--"+strings.ReplaceAll(repo, "/", "--"))
}

// hubCacheDir resolves the HF hub cache root. Honors HF_HOME and
// HUGGINGFACE_HUB_CACHE so a user with a custom cache path doesn't
// see false negatives.
func hubCacheDir() string {
	if v := os.Getenv("HUGGINGFACE_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// isRepoCached probes HF's on-disk cache for a complete download of
// the given repo. The cache layout is:
//
//	<root>/models--<org>--<name>/blobs/<sha>          (final blob)
//	<root>/models--<org>--<name>/blobs/<sha>.incomplete (in-progress)
//	<root>/models--<org>--<name>/snapshots/<sha>/<files>
//
// REASON: a partial/interrupted download leaves `.incomplete` blobs
// behind plus dangling symlinks under snapshots. Treating any snapshot
// entry as "downloaded" reported true for half-finished pulls and the
// UI then offered Load instead of resuming the Download. We now treat
// any `.incomplete` blob as "not ready", which is the simplest reliable
// signal — HF tears it down on `download_done` (rename to final blob).
func isRepoCached(root, repo string) bool {
	if root == "" || repo == "" {
		return false
	}
	repoDir := repoCacheDir(root, repo)
	if blobs, err := os.ReadDir(filepath.Join(repoDir, "blobs")); err == nil {
		for _, b := range blobs {
			if strings.HasSuffix(b.Name(), ".incomplete") {
				return false
			}
		}
	}
	snapshots := filepath.Join(repoDir, "snapshots")
	entries, err := os.ReadDir(snapshots)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		inner, err := os.ReadDir(filepath.Join(snapshots, e.Name()))
		if err != nil {
			continue
		}
		if len(inner) > 0 {
			return true
		}
	}
	return false
}
