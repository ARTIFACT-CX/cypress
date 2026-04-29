// AREA: models · CACHE
// Filesystem probe of the Hugging Face hub cache. Lets the UI show
// "downloaded" vs "needs download" before the user clicks load,
// without booting the Python worker. The probe is best-effort — we
// look for the canonical HF hub cache layout and report `downloaded:
// true` if any snapshot directory is non-empty and no `.incomplete`
// blobs are dangling. We don't verify individual file hashes; HF's
// own loaders will do that at load time and trigger a refetch if
// needed.

package models

import (
	"os"
	"path/filepath"
	"strings"
)

// HubCacheDir resolves the HF hub cache root. Honors HF_HOME and
// HUGGINGFACE_HUB_CACHE so a user with a custom cache path doesn't
// see false negatives.
func HubCacheDir() string {
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

// RepoCacheDir returns the on-disk directory HF uses for a given
// repo. Shared by IsRepoCached and downloads' delete path so they
// stay in sync on the layout assumption.
func RepoCacheDir(root, repo string) string {
	return filepath.Join(root, "models--"+strings.ReplaceAll(repo, "/", "--"))
}

// IsRepoCached probes HF's on-disk cache for a complete download of
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
func IsRepoCached(root, repo string) bool {
	if root == "" || repo == "" {
		return false
	}
	repoDir := RepoCacheDir(root, repo)
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
