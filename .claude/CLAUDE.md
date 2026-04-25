# Cypress — AI Collaboration Guide

## Project

Cypress is a local-first desktop app for duplex voice AI with agentic capabilities — "Ollama/LM Studio but for voice agents." Open-source core, potential premium/hosted tier later.

## Architecture

```
Tauri app (React/TypeScript + Rust shell)
    ↕ WebSocket (audio streams, localhost)
Go server (orchestration, model management, API)
    ↕ subprocess / gRPC
Python inference workers (PyTorch, model loading, generation)
    ↕
Local models on disk
```

Models: **Moshi 3.5B** (default, lighter) and **PersonaPlex 7B** (NVIDIA, duplex + persona conditioning). Future: Kokoro, Orpheus TTS.

## Structure

**Scope:** this applies to **`server/` (Go) and `worker/` (Python)**. The React app (`app/`) is exempt for now — it stays component-organized until it grows enough to need the discipline.

Within `server/` and `worker/`, code is organized **by feature** (vertical slice), not by technical layer. A feature owns one capability end-to-end: its business logic, the adapters that connect it inward (HTTP handlers, IPC commands), and the adapters that connect it outward (clients for external deps), plus the interfaces those adapters satisfy.

```
server/
  inference/                ← feature
    manager.go              business logic
    worker.go               outbound adapter (Python subprocess)
    handlers.go             inbound adapter (HTTP routes)
  audio/                    ← feature
    pipeline.go
    transport.go
  shared/                   infrastructure only — no domain
    log/
    config/
```

**Rules:**

- **Features don't import from other features.** Cross-feature needs go through a public interface, or get coordinated by a dedicated orchestration feature.
- **`shared/` is infrastructure only** — logger, config, DB connection, event bus. The moment something there knows about a domain concept, it moves into a feature.
- **No `utils/` folder.** Shared packages are named for what they do (`shared/log`, not `shared/utils`).
- **Prefer duplication over premature abstraction.** Two features doing similar things stay separate until they've proven they should be identical.
- **Inside a feature, business logic depends on interfaces.** Concrete adapters are injected at startup (ports & adapters / hexagonal). Adapter interfaces are natural `// SWAP:` seams — no separate tagging needed.
- **Split a feature when it grows unwieldy.** Smaller features compose better than one big one.

The Tauri/Go/Python service split stays as it is — feature slicing is the rule **inside** `server/` and `worker/`, not across services. Both the Go server and the Python worker follow this layout; concrete file names differ (Python uses packages of `__init__.py` modules where Go uses `package` directories), but the feature-as-vertical-slice idea is the same.

### AI codegen scope

- A feature is the natural prompt boundary. Generated changes should stay inside one feature when possible.
- Before adding code: does it belong in an existing feature, warrant a new one, or is it genuinely shared infrastructure? Default to "existing feature" unless there's a real reason otherwise.
- Don't expand `shared/` without justification — it's the fastest way for this structure to rot.

## Coding Style

### Comments

Write **generous inline comments** that explain _what the code is doing and why_ — not just mechanics, but the reasoning. Code should read top-to-bottom like a narrated walkthrough. A future maintainer should be able to understand the intent without reading the git history.

**Do:**

- Explain the _process_ at the top of each function / module (2-3 line header).
- Call out non-obvious decisions and tradeoffs inline.
- Tag logical sections within long functions so they're easy to scan.

**Don't:**

- Restate trivial mechanics (`// increment i`).
- Reference specific issues, tickets, or PRs ("fix for #42") — those rot. Explain the _reason_ instead.

### Tags

**File-level `AREA` header.** Every source file starts with a one-line area tag and a short purpose sentence. This lets us grep features across the codebase (e.g. `rg "AREA: audio"`) without tagging every comment.

```go
// AREA: audio · TRANSPORT · WS
// Reads mic frames off the WebSocket and fans them into the model pipeline.
```

Format: `AREA: <domain> · <subsystem> [· <sub-subsystem>]`. Categories we'll use: `audio`, `models`, `transport`, `tools`, `ui`, `ipc`, `config`, `build`.

**Inline tags** describe what a line/block _does_, not which feature it belongs to:

- `// SETUP:` — initialization / configuration
- `// STEP n:` — numbered steps within a process
- `// REASON:` — explains a non-obvious decision
- `// SWAP:` — marks a module / interface designed to be replaced later (reader hint: don't inline it, don't tightly couple)
- `// TODO:` — deferred work (include a clear next step)
- `// PERF:` — performance-critical section
- `// SAFETY:` — concurrency, locking, or data-integrity invariant

### Modularity

- Each module should have a **single, stated purpose** (top-of-file comment).
- Any part of the system we might swap later (models, transports, tools, UI components) lives behind a clear interface — same seam as feature adapter interfaces (see Structure). Mark with `// SWAP:`.
- Prefer small files with obvious names over large catch-all modules.

### Conciseness

- No ceremony or boilerplate. If there's a simpler way, use it.
- No defensive validation on internal calls — validate only at system boundaries (user input, external APIs, subprocess IPC).
- No speculative abstractions for "maybe later."
- Trust the type system / compiler; don't write redundant runtime checks.

### Examples over prose

When a comment would be longer than the code, the code should probably be simpler.

## Testing

**Scope:** applies to **`server/` (Go) and `worker/` (Python)**. The React app (`app/`) is exempt for now.

Test-driven development is the default. New non-trivial code lands with tests. Trivial glue (a handler that decodes JSON and calls one method, a struct field accessor) doesn't need a test — don't pad coverage with tests of the framework.

### Unit vs integration

- **Unit tests** cover function-level logic: branches, error paths, state transitions, parsing, framing, dispatch. Heavy dependencies (real subprocess, real model load, real network) are faked. Fast — a unit suite should run in seconds.
- **Integration tests** cover a feature's full flow with real adapters: HTTP → manager → real Python subprocess → reply, IPC framing over a real pipe, audio path over a real socket. Slower; run before commit, not on every edit.

### Layout

Tests live alongside the code they test, inside the feature folder. This is the Go convention (required for access to unexported symbols) and a fine Python convention too — keeps the feature self-contained.

```
server/inference/
  manager.go
  manager_test.go           ← unit
  worker.go
  worker_test.go            ← unit
  integration_test.go       ← integration (build tag: //go:build integration)
  testdata/                 ← fixtures, golden files

worker/models/
  moshi.py
  test_moshi.py             ← unit
  test_integration.py       ← integration (pytest marker: @pytest.mark.integration)
  testdata/
```

- File naming: Go `<name>_test.go`, Python `test_<name>.py` — language-idiomatic, picked up by `go test` / `pytest` automatically.
- Test naming: `unit_under_test_scenario_expectation`. Go: `TestManager_LoadModel_RejectsWhenBusy`. Python: `test_load_model_rejects_when_busy`. Greppable.
- Integration tests are gated so a default `go test ./...` / `pytest` runs only the fast unit suite. Run integration explicitly: `go test -tags=integration ./...`, `pytest -m integration`.

### When to write them

- **New business logic / state machine / parser:** test-first. Red → green → refactor.
- **New adapter glue:** test-alongside. Write the code and its unit test in the same change.
- **Bug fix:** add the failing test first, then the fix. The test stays as a regression guard.

### When to run them

- **After editing a file inside a feature:** run that feature's unit tests. `go test ./inference/...` or `pytest worker/models/`. Feature-scoped, fast.
- **Before committing a feature change:** run the full unit suite + the relevant integration tests.
- **Before committing a cross-feature or infrastructure change:** run everything, both suites.
- A failing test blocks the commit. Don't skip, don't `t.Skip()`, don't comment it out — fix it or fix the code.

### Fakes and fixtures

- Per-feature fakes live in the feature's test files (or a `fakes_test.go` / `conftest.py` if shared across that feature's tests). Don't build a global mocking framework.
- Byte-level fixtures (IPC frames, audio chunks, golden JSON replies) live in `testdata/` rather than being rebuilt inline — keeps tests readable and lets us diff against known-good bytes.

## Process

### Ask clarifying questions

When a task description is ambiguous — architecturally, behaviorally, or in scope — **stop and ask** before implementing. Specifically pause when:

- There are multiple plausible interfaces or data flows.
- The task touches a subsystem that hasn't been designed yet.
- The change would commit us to a direction we'd want to reconsider (e.g., picking a protocol, adding a dependency, choosing a file layout).

One well-scoped question now saves a rewrite later.

### Small, reversible steps

Prefer a sequence of small, working changes over one big sweep. Each step should leave the app in a runnable state.

### Don't generate docs unprompted

Don't create README, design docs, or summary markdown files unless explicitly asked. Work from conversation context.

### Track progress on GitHub issues

When work maps to an existing GitHub issue, post a progress comment on that issue as commits land — list the relevant commit SHAs and a short note on what's done vs still pending. Don't close the issue until it's actually finished end-to-end. Use `gh issue comment <n>` for the updates and `gh issue view <n>` first to confirm scope before commenting.
