// AREA: inference · MANIFEST · TEST
// Round-trip and corruption tolerance for the on-disk manifest.

package inference

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func newTestManifest(t *testing.T) *Manifest {
	t.Helper()
	t.Setenv("CYPRESS_DATA_DIR", t.TempDir())
	m, err := NewManifest()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestManifest_PutGetHasAll(t *testing.T) {
	m := newTestManifest(t)
	if m.Has("moshi") {
		t.Fatal("fresh manifest should not report any installs")
	}
	if got := m.Get("moshi"); got != nil {
		t.Fatal("Get on absent name should return nil")
	}

	entry := ManifestEntry{
		Name:        "moshi",
		Repo:        "kyutai/moshiko-mlx-q8",
		Files:       []string{"/cache/model.q8.safetensors"},
		SizeBytes:   1234,
		InstalledAt: time.Now().UTC(),
	}
	if err := m.Put(entry); err != nil {
		t.Fatal(err)
	}
	if !m.Has("moshi") {
		t.Fatal("Has should be true after Put")
	}
	got := m.Get("moshi")
	if got == nil || got.Repo != entry.Repo || got.SizeBytes != entry.SizeBytes {
		t.Fatalf("Get mismatch: got %+v", got)
	}

	// Mutating the returned copy must not affect the manifest.
	got.Repo = "tampered"
	again := m.Get("moshi")
	if again.Repo != entry.Repo {
		t.Fatal("Get should return a defensive copy")
	}

	all := m.All()
	if len(all) != 1 || all["moshi"].Repo != entry.Repo {
		t.Fatalf("All mismatch: %+v", all)
	}
}

func TestManifest_PersistsAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CYPRESS_DATA_DIR", dir)

	m1, err := NewManifest()
	if err != nil {
		t.Fatal(err)
	}
	if err := m1.Put(ManifestEntry{Name: "moshi", Repo: "x"}); err != nil {
		t.Fatal(err)
	}

	m2, err := NewManifest()
	if err != nil {
		t.Fatal(err)
	}
	if !m2.Has("moshi") {
		t.Fatal("entry should survive reopen")
	}
}

func TestManifest_DeleteIsIdempotent(t *testing.T) {
	m := newTestManifest(t)
	if err := m.Delete("nope"); err != nil {
		t.Fatalf("delete on absent name should be no-op: %v", err)
	}
	if err := m.Put(ManifestEntry{Name: "moshi"}); err != nil {
		t.Fatal(err)
	}
	if err := m.Delete("moshi"); err != nil {
		t.Fatal(err)
	}
	if m.Has("moshi") {
		t.Fatal("delete should remove entry")
	}
	if err := m.Delete("moshi"); err != nil {
		t.Fatal("second delete should still be no-op")
	}
}

func TestManifest_CorruptFileFallsBackToEmpty(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CYPRESS_DATA_DIR", dir)
	if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := NewManifest()
	if err != nil {
		t.Fatalf("corrupt file should not error: %v", err)
	}
	if len(m.All()) != 0 {
		t.Fatal("corrupt manifest should yield empty entries")
	}
}

func TestManifest_UnknownVersionDiscarded(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CYPRESS_DATA_DIR", dir)
	body, _ := json.Marshal(map[string]any{
		"version": 999,
		"entries": map[string]any{
			"moshi": map[string]any{"name": "moshi"},
		},
	})
	if err := os.WriteFile(filepath.Join(dir, "models.json"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := NewManifest()
	if err != nil {
		t.Fatal(err)
	}
	if m.Has("moshi") {
		t.Fatal("unknown schema version should be treated as empty")
	}
}
