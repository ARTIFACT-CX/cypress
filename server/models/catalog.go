// AREA: models · CATALOG
// Static metadata for the set of models Cypress knows how to load. The
// catalog mirrors what the Python worker actually loads — the repo
// names here have to match worker/models/*.py defaults. Keeping it
// Go-side means the picker UI can render this list without first
// spinning up the Python worker (which costs several seconds and only
// runs on demand). The trade-off is that the two sides need to stay
// in sync; there's exactly one knob today (CYPRESS_MOSHI_REPO) and we
// document the linkage in comments.
//
// All variant-selection functions take an explicit (os, arch) tuple
// so the caller can choose between the laptop's platform (local
// worker mode) and the remote worker's reported platform (remote
// mode). Without that split, an Apple-Silicon laptop dialing a Linux
// GPU worker would tell it to download MLX weights it can't load.

package models

import "runtime"

// Entry is the static description of a model the catalog knows about.
// Exported so other packages (downloads, inference) can read the repo
// + family without having to round-trip through ModelInfo.
type Entry struct {
	Name         string
	Label        string
	Hint         string
	Backend      string
	Repo         string
	Files        []string // weight file names within the repo
	SizeGB       string
	Requirements string
	Available    bool
	// Family selects which per-family Python venv the worker is spawned
	// from (worker/models/<family>/.venv). Two model names that share a
	// family (e.g. `moshi` / `moshi-mlx` / `moshi-torch`) reuse the same
	// subprocess; switching to a different family requires a worker
	// restart so we don't try to import two conflicting Python stacks at
	// once (PersonaPlex's forked `moshi` package vs kyutai's).
	Family string
}

// HostPlatform returns the laptop's (GOOS, GOARCH). Use this only when
// the caller is genuinely running against a local worker, OR for
// family-only lookups where the repo/files fields are immaterial.
// Platform-sensitive paths (downloads, /models) should pass the
// worker's reported tuple instead.
func HostPlatform() (os, arch string) {
	return runtime.GOOS, runtime.GOARCH
}

// REASON: backend selection mirrors worker/models/__init__.py's
// _default_moshi_backend. We can't import that logic, so we duplicate
// the platform check; if the worker logic ever grows more complex
// we'll need an IPC query instead.
// SETUP: shared file names. mimi + tokenizer files are identical across
// every moshi checkpoint (moshika / moshiko, mlx / pytorch, every
// quant) — only the LM weights filename varies. Kept as constants so
// any future repo addition reuses these without re-encoding the
// filename strings.
const (
	moshiMimiFile      = "tokenizer-e351c8d8-checkpoint125.safetensors"
	moshiTokenizerFile = "tokenizer_spm_32k_3.model"
	moshiTorchLMFile   = "model.safetensors"
	moshiMlxQ8LMFile   = "model.q8.safetensors"
)

// DefaultMoshiEntry returns the Moshi entry for the given target
// platform. Apple Silicon → MLX-q8 (only place where the bf16-MPS path
// is too slow to be usable in practice); everything else → torch-bf16.
// Moshi is 7B parameters; the moshika voice (female) is the default —
// CYPRESS_MOSHI_REPO can swap to moshiko-* for the male voice.
func DefaultMoshiEntry(os, arch string) Entry {
	if os == "darwin" && arch == "arm64" {
		return Entry{
			Name:         "moshi",
			Label:        "Moshi",
			Hint:         "7B · duplex",
			Backend:      "mlx",
			Repo:         "kyutai/moshika-mlx-q8",
			Files:        []string{moshiMlxQ8LMFile, moshiMimiFile, moshiTokenizerFile},
			SizeGB:       "~4 GB",
			Requirements: "~6 GB unified memory · Apple Silicon",
			Available:    true,
			Family:       "moshi",
		}
	}
	return Entry{
		Name:         "moshi",
		Label:        "Moshi",
		Hint:         "7B · duplex · female voice",
		Backend:      "torch",
		Repo:         "kyutai/moshika-pytorch-bf16",
		Files:        []string{moshiTorchLMFile, moshiMimiFile, moshiTokenizerFile},
		SizeGB:       "~14 GB",
		Requirements: "~14 GB VRAM · CUDA preferred",
		Available:    true,
		Family:       "moshi",
	}
}

// Catalog returns the static set for a target platform. Available
// reflects whether the worker has a registered loader. PersonaPlex is
// listed so users see what's coming but can't click it yet — the
// moment its loader lands in the worker we just flip Available.
func Catalog(os, arch string) []Entry {
	return []Entry{
		DefaultMoshiEntry(os, arch),
		{
			Name:    "personaplex",
			Label:   "PersonaPlex",
			Hint:    "7B · duplex + persona",
			Backend: "torch",
			Repo:    "nvidia/personaplex-7b-v1",
			// REASON: PersonaPlex reuses moshi's mimi codec + SPM tokenizer
			// filenames verbatim (NVIDIA forked the runtime), so we share
			// the constants. The LM weight is a single bf16 safetensors at
			// ~16.7 GB.
			Files:        []string{"model.safetensors", moshiMimiFile, moshiTokenizerFile},
			SizeGB:       "~17 GB",
			Requirements: "~20 GB VRAM · NVIDIA Ampere/Hopper",
			// Available stays false until #3 ships an inference path that
			// works on the platforms we target (M1 Pro 16 GB needs INT4
			// quant; the bf16 checkpoint is too large for unified memory).
			Available: false,
			Family:    "personaplex",
		},
	}
}

// EntryFor returns the static entry for a given model name on the
// target platform, or nil. Used by callers that need the platform-
// specific repo / files (downloads.Service when starting a pull,
// inference.Manager when picking the venv to spawn).
func EntryFor(name, os, arch string) *Entry {
	for _, e := range Catalog(os, arch) {
		if e.Name == name {
			cp := e
			return &cp
		}
	}
	return nil
}

// FamilyOf returns the per-family venv key for a model name, or "" if
// unknown. Family is platform-independent — every variant of moshi
// belongs to the "moshi" family — so callers that only care about
// family don't need a platform tuple.
func FamilyOf(name string) string {
	host, arch := HostPlatform()
	if e := EntryFor(name, host, arch); e != nil {
		return e.Family
	}
	return ""
}
