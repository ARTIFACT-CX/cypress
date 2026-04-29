// AREA: models · MANIFEST
// Persistent record of which models the user has installed. We treat
// this as the canonical "is the model available" answer — separate
// from the HF cache probe in cache.go, which only tells us whether
// some files happen to live in HF's cache directory. The manifest
// records intent: "the user installed this model on this date with
// these files at this revision."
//
// File location: provided by the caller (composition root passes
// ~/.cypress in prod, a tmpdir in tests). Schema-versioned so future
// field additions don't silently corrupt old installs — an unrecognized
// version reads as empty and we start fresh, which is the safest
// fallback for a v0.1 cache file.
//
// SAFETY: every write goes through atomic rename (write tmp, rename
// over). A crash mid-save loses the in-flight change, never the file.

package models

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// manifestVersion bumps when the schema changes incompatibly. v1 is
// the initial schema below; v2 (etc.) gets a migration step in load.
const manifestVersion = 1

// ManifestEntry records one installed model. JSON tags pin the wire
// format — renaming a Go field shouldn't silently break old caches.
type ManifestEntry struct {
	Name        string    `json:"name"`
	Repo        string    `json:"repo"`
	Revision    string    `json:"revision,omitempty"`
	Files       []string  `json:"files"`       // absolute paths returned by HF
	SizeBytes   int64     `json:"sizeBytes"`   // sum of file sizes at install time
	InstalledAt time.Time `json:"installedAt"` // wall-clock; informational
}

// manifestFile is the on-disk shape — wraps entries in a versioned
// envelope so we can evolve the schema without trampling old data.
type manifestFile struct {
	Version int                       `json:"version"`
	Entries map[string]*ManifestEntry `json:"entries"`
}

// Manifest is the in-memory handle. All methods are safe for
// concurrent use; mu guards entries + the on-disk write.
type Manifest struct {
	mu      sync.Mutex
	path    string
	entries map[string]*ManifestEntry
}

// NewManifest opens (or creates) the manifest at <dataDir>/models.json.
// Caller (composition root) decides the path — prod passes ~/.cypress,
// tests pass a tmpdir. A missing file is treated as "no installs yet"
// — not an error — so a fresh user lands in a clean state without a
// setup step.
func NewManifest(dataDir string) (*Manifest, error) {
	if dataDir == "" {
		return nil, errors.New("NewManifest: dataDir is required")
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, err
	}
	m := &Manifest{
		path:    filepath.Join(dataDir, "models.json"),
		entries: map[string]*ManifestEntry{},
	}
	if err := m.load(); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	return m, nil
}

// load reads and decodes the manifest. Unknown versions are treated
// as "discard and start fresh" — preferable to a partial parse that
// would silently lose entries.
func (m *Manifest) load() error {
	data, err := os.ReadFile(m.path)
	if err != nil {
		return err
	}
	var f manifestFile
	if err := json.Unmarshal(data, &f); err != nil {
		// Corrupted manifest: same fallback as a missing file. The
		// alternative — refusing to start — would be worse for users
		// with no easy way to fix the file by hand.
		m.entries = map[string]*ManifestEntry{}
		return nil
	}
	if f.Version != manifestVersion {
		m.entries = map[string]*ManifestEntry{}
		return nil
	}
	if f.Entries == nil {
		f.Entries = map[string]*ManifestEntry{}
	}
	m.entries = f.Entries
	return nil
}

// save serializes + atomically writes the manifest. Caller holds m.mu.
func (m *Manifest) saveLocked() error {
	f := manifestFile{Version: manifestVersion, Entries: m.entries}
	data, err := json.MarshalIndent(&f, "", "  ")
	if err != nil {
		return err
	}
	tmp := m.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, m.path)
}

// Get returns a copy of the entry for name, or nil if not installed.
// Returning a copy keeps callers from mutating the in-memory map by
// accident — they have to go through Put to persist a change.
func (m *Manifest) Get(name string) *ManifestEntry {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.entries[name]
	if !ok {
		return nil
	}
	cp := *e
	return &cp
}

// Has is the cheap predicate the catalog probe uses to fill
// ModelInfo.Installed. Avoids the per-call copy that Get does.
func (m *Manifest) Has(name string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.entries[name]
	return ok
}

// Put inserts or replaces an entry and persists. Failure to write
// rolls back the in-memory change so a subsequent Has() returns the
// pre-Put truth — keeps memory and disk in sync even on flaky FS.
func (m *Manifest) Put(e ManifestEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	prev, hadPrev := m.entries[e.Name]
	cp := e
	m.entries[e.Name] = &cp
	if err := m.saveLocked(); err != nil {
		if hadPrev {
			m.entries[e.Name] = prev
		} else {
			delete(m.entries, e.Name)
		}
		return err
	}
	return nil
}

// Delete removes an entry. Idempotent: deleting a non-existent name
// is a no-op (no error) so the HTTP handler can call this defensively.
func (m *Manifest) Delete(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	prev, ok := m.entries[name]
	if !ok {
		return nil
	}
	delete(m.entries, name)
	if err := m.saveLocked(); err != nil {
		m.entries[name] = prev
		return err
	}
	return nil
}

// All returns a snapshot of every installed entry. Used by /models to
// merge manifest state into the catalog response in one read.
func (m *Manifest) All() map[string]ManifestEntry {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]ManifestEntry, len(m.entries))
	for k, v := range m.entries {
		out[k] = *v
	}
	return out
}
