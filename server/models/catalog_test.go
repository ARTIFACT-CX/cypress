// AREA: models · TESTS · CATALOG
// Coverage for the platform-aware catalog. Two cases that matter most:
//   - darwin/arm64 → MLX-q8 variant.
//   - everything else (linux/amd64 covers RunPod / Phoenix / most cloud
//     GPUs) → torch bf16 variant.
// Plus EntryFor + ModelInfos pass-through assertions so the laptop's
// runtime never leaks into a remote-target lookup.

package models

import "testing"

func TestDefaultMoshiEntry_AppleSilicon_PicksMLX(t *testing.T) {
	e := DefaultMoshiEntry("darwin", "arm64")
	if e.Backend != "mlx" {
		t.Errorf("backend = %q, want %q", e.Backend, "mlx")
	}
	if e.Repo != "kyutai/moshika-mlx-q8" {
		t.Errorf("repo = %q, want kyutai/moshika-mlx-q8", e.Repo)
	}
}

func TestDefaultMoshiEntry_LinuxAmd64_PicksTorch(t *testing.T) {
	// REASON: this is the case the bug actually hit — Apple-Silicon
	// laptop dialing a Linux GPU box. Before the fix, the laptop's
	// runtime resolved to MLX and the worker couldn't load it.
	e := DefaultMoshiEntry("linux", "amd64")
	if e.Backend != "torch" {
		t.Errorf("backend = %q, want %q", e.Backend, "torch")
	}
	if e.Repo != "kyutai/moshika-pytorch-bf16" {
		t.Errorf("repo = %q, want kyutai/moshika-pytorch-bf16", e.Repo)
	}
}

func TestDefaultMoshiEntry_LinuxArm64_PicksTorch(t *testing.T) {
	// REASON: arm64 alone (without darwin) shouldn't trigger the MLX
	// branch. AWS Graviton, raspberry-pi-style hosts, etc.
	e := DefaultMoshiEntry("linux", "arm64")
	if e.Backend != "torch" {
		t.Errorf("backend = %q, want %q", e.Backend, "torch")
	}
}

func TestDefaultMoshiEntry_DarwinAmd64_PicksTorch(t *testing.T) {
	// Intel Mac — no MLX availability, fall through to torch.
	e := DefaultMoshiEntry("darwin", "amd64")
	if e.Backend != "torch" {
		t.Errorf("backend = %q, want %q", e.Backend, "torch")
	}
}

func TestEntryFor_RoutesByPlatform(t *testing.T) {
	mac := EntryFor("moshi", "darwin", "arm64")
	linux := EntryFor("moshi", "linux", "amd64")
	if mac == nil || linux == nil {
		t.Fatal("EntryFor returned nil for known model")
	}
	if mac.Repo == linux.Repo {
		t.Errorf("expected different repos by platform; both = %q", mac.Repo)
	}
}

func TestEntryFor_UnknownReturnsNil(t *testing.T) {
	if e := EntryFor("nonsense", "linux", "amd64"); e != nil {
		t.Errorf("expected nil for unknown model; got %+v", e)
	}
}

func TestFamilyOf_PlatformIndependent(t *testing.T) {
	// Family is the same across variants, so FamilyOf should not depend
	// on the host's runtime — but as a smoke check that it returns the
	// expected value for the only family wired up today.
	if got := FamilyOf("moshi"); got != "moshi" {
		t.Errorf("FamilyOf(moshi) = %q, want moshi", got)
	}
	if got := FamilyOf("personaplex"); got != "personaplex" {
		t.Errorf("FamilyOf(personaplex) = %q, want personaplex", got)
	}
	if got := FamilyOf("nope"); got != "" {
		t.Errorf("FamilyOf(unknown) = %q, want empty", got)
	}
}

func TestModelInfos_DownloadedSetWins_OverLocalProbe(t *testing.T) {
	// REASON: when an explicit downloaded set is passed (remote-worker
	// case), ModelInfos should NOT consult the laptop's HF cache —
	// otherwise the catalog flips back to "Download" right after a
	// successful remote pull. This is the second half of the bug.
	t.Setenv("HUGGINGFACE_HUB_CACHE", t.TempDir()) // empty cache locally
	downloaded := map[string]bool{
		"kyutai/moshika-pytorch-bf16": true,
	}
	infos := ModelInfos("linux", "amd64", downloaded, nil)
	var moshi *ModelInfo
	for i := range infos {
		if infos[i].Name == "moshi" {
			moshi = &infos[i]
		}
	}
	if moshi == nil {
		t.Fatal("expected moshi in catalog")
	}
	if !moshi.Downloaded {
		t.Errorf("expected Downloaded=true from explicit set; local cache was empty")
	}
}

func TestModelInfos_NilDownloadedSet_FallsBackToLocalProbe(t *testing.T) {
	// Local-subprocess mode: caller passes nil so info.go consults
	// IsRepoCached on disk. Empty cache → not downloaded.
	t.Setenv("HUGGINGFACE_HUB_CACHE", t.TempDir())
	infos := ModelInfos("linux", "amd64", nil, nil)
	for _, info := range infos {
		if info.Name == "moshi" && info.Downloaded {
			t.Errorf("expected Downloaded=false with empty local cache and nil set")
		}
	}
}
