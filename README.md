# Vexis

Vexis is a deterministic, stateful intelligence architecture with two cooperating layers:

- a deterministic core for evidence, uncertainty, memory, learning, and resolution
- an expressive interface layer for conversation, presence, and presentation

The repository is organized around code direction and implementation structure. It does not include private notes, local machine state, or bundled model weights.

## Repository Direction

### Core goals

- keep belief formation evidence-bound and revisable
- separate reasoning rigor from personality and rendering
- support persistent state, memory, learning, and follow-up questioning
- support a desktop controller and an external presence client

### Major areas

- [`core`](./core): deterministic reasoning, learning, routing, state, inquiry, resolution
- [`memory`](./memory): memory storage helpers
- [`ingest`](./ingest): text, PDF, image, and knowledge intake
- [`interface`](./interface): Qt controller and bridge-side interface code
- [`speech`](./speech): text-to-speech support
- [`tests`](./tests): regression coverage for core behavior and integration paths
- [`unreal/VexisPresence`](./unreal/VexisPresence): Unreal-based presence prototype

## Local Model Placement

The local LLM is expected to run through `llama.cpp`.

Default paths:

- server executable: [`bin/llama.cpp/llama-server.exe`](./bin/llama.cpp/llama-server.exe)
- model file: [`models/text/qwen3_8b/Qwen3-8B-Q4_K_M.gguf`](./models/text/qwen3_8b/Qwen3-8B-Q4_K_M.gguf)

If you want different locations, set environment variables before launch:

- `VEXIS_LLAMA_SERVER_PATH`
- `VEXIS_LLM_MODEL_PATH`

The repo does not ship model weights or the `llama.cpp` binary.

## Quick Start

### 1. Create the Python environment

Use Python 3.12 and install the runtime packages you need for the current path, for example:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -U pip
```

### 2. Put the local model in place

Place:

- `llama-server.exe` in `bin/llama.cpp/`
- the GGUF model in `models/text/qwen3_8b/`

### 3. Run the controller

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_vexis_controller_foreground.ps1
```

## Unreal Presence

The Unreal presence project is included in:

- [`unreal/VexisPresence/VexisPresence.uproject`](./unreal/VexisPresence/VexisPresence.uproject)

The helper scripts resolve the repo-relative project path automatically. If Unreal Engine is not installed in a common location, set:

- `VEXIS_UNREAL_EDITOR`

Then launch:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_vexis_unreal.ps1
```

Additional Unreal notes are in [`docs/unreal_presence_setup.md`](./docs/unreal_presence_setup.md).

## Notes

- The repo intentionally ignores local runtime state, logs, caches, and generated Unreal output.
- Third-party content packs are not bundled in this repository snapshot.
