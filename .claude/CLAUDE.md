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

## Coding Style

### Comments

Write **generous inline comments** that explain *what the code is doing and why* — not just mechanics, but the reasoning. Code should read top-to-bottom like a narrated walkthrough. A future maintainer should be able to understand the intent without reading the git history.

**Do:**
- Explain the *process* at the top of each function / module (2-3 line header).
- Call out non-obvious decisions and tradeoffs inline.
- Tag logical sections within long functions so they're easy to scan.

**Don't:**
- Restate trivial mechanics (`// increment i`).
- Reference specific issues, tickets, or PRs ("fix for #42") — those rot. Explain the *reason* instead.

### Tags

**File-level `AREA` header.** Every source file starts with a one-line area tag and a short purpose sentence. This lets us grep features across the codebase (e.g. `rg "AREA: audio"`) without tagging every comment.

```go
// AREA: audio · TRANSPORT · WS
// Reads mic frames off the WebSocket and fans them into the model pipeline.
```

Format: `AREA: <domain> · <subsystem> [· <sub-subsystem>]`. Categories we'll use: `audio`, `models`, `transport`, `tools`, `ui`, `ipc`, `config`, `build`.

**Inline tags** describe what a line/block *does*, not which feature it belongs to:

- `// SETUP:` — initialization / configuration
- `// STEP n:` — numbered steps within a process
- `// WHY:` — explains a non-obvious decision
- `// SWAP:` — marks a module / interface designed to be replaced later (reader hint: don't inline it, don't tightly couple)
- `// TODO:` — deferred work (include a clear next step)
- `// PERF:` — performance-critical section
- `// SAFETY:` — concurrency, locking, or data-integrity invariant

### Modularity

- Each module should have a **single, stated purpose** (top-of-file comment).
- Any part of the system we might swap later (models, transports, tools, UI components) lives behind a clear interface. Mark the seam with `// SWAP:`.
- Prefer small files with obvious names over large catch-all modules.

### Conciseness

- No ceremony or boilerplate. If there's a simpler way, use it.
- No defensive validation on internal calls — validate only at system boundaries (user input, external APIs, subprocess IPC).
- No speculative abstractions for "maybe later."
- Trust the type system / compiler; don't write redundant runtime checks.

### Examples over prose

When a comment would be longer than the code, the code should probably be simpler.

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
