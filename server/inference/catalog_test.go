// AREA: inference · CATALOG · TEST
// Unit tests for the HF cache probe. Construct a fake hub layout in a
// tmpdir and verify the probe correctly distinguishes downloaded vs
// missing repos. The catalog itself is tested implicitly via ModelInfos.

package inference

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// makeFakeRepo lays out the minimum structure HF's hub cache uses for
// a "downloaded" repo: <root>/models--<org>--<name>/snapshots/<sha>/<file>.
// Returns the cache root.
func makeFakeRepo(t *testing.T, repo string, withFile bool) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(
		root,
		"models--"+strings.ReplaceAll(repo, "/", "--"),
		"snapshots",
		"abc123",
	)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if withFile {
		f := filepath.Join(dir, "model.safetensors")
		if err := os.WriteFile(f, []byte("not-real-weights"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestIsRepoCached_PresentSnapshotWithFile(t *testing.T) {
	root := makeFakeRepo(t, "kyutai/moshiko-mlx-q8", true)
	if !isRepoCached(root, "kyutai/moshiko-mlx-q8") {
		t.Fatal("expected downloaded repo to be detected")
	}
}

func TestIsRepoCached_EmptySnapshotReportsMissing(t *testing.T) {
	// Empty snapshot dir (no files inside) should not count — HF
	// occasionally creates the dir before populating it.
	root := makeFakeRepo(t, "kyutai/moshiko-mlx-q8", false)
	if isRepoCached(root, "kyutai/moshiko-mlx-q8") {
		t.Fatal("expected empty snapshot to be reported as not downloaded")
	}
}

func TestIsRepoCached_AbsentRepo(t *testing.T) {
	root := t.TempDir()
	if isRepoCached(root, "kyutai/moshiko-mlx-q8") {
		t.Fatal("expected absent repo to be reported as not downloaded")
	}
}

func TestIsRepoCached_EmptyArgsAreFalse(t *testing.T) {
	if isRepoCached("", "kyutai/moshiko-mlx-q8") {
		t.Fatal("empty root should be false")
	}
	if isRepoCached(t.TempDir(), "") {
		t.Fatal("empty repo should be false")
	}
}

func TestModelInfos_PopulatesAvailableAndDownloaded(t *testing.T) {
	// Point the hub probe at a tmpdir with the default Moshi repo
	// pre-cached so we can assert the dynamic fields land correctly.
	// Picks the right repo for the current platform via the same
	// helper the catalog uses.
	entry := defaultMoshiEntry()
	root := makeFakeRepo(t, entry.Repo, true)
	t.Setenv("HUGGINGFACE_HUB_CACHE", root)

	infos := ModelInfos(nil)
	var moshi *ModelInfo
	for i := range infos {
		if infos[i].Name == "moshi" {
			moshi = &infos[i]
		}
	}
	if moshi == nil {
		t.Fatal("expected catalog to include moshi entry")
	}
	if !moshi.Available {
		t.Fatal("moshi should be available")
	}
	if !moshi.Downloaded {
		t.Fatal("moshi should be reported downloaded after fake cache layout")
	}
	if moshi.Requirements == "" {
		t.Fatal("moshi entry must surface a requirements string")
	}
}
