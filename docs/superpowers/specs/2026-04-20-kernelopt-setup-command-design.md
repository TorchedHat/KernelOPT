# kernelopt setup command — Design Spec

**Goal:** Add `kernelopt setup` (interactive wizard) and `kernelopt doctor` (read-only status check) to the CLI so a user can go from `pip install kernelopt` to a fully working environment without any external scripts.

**Date:** 2026-04-20

---

## Problem

`setup_environment.py` exists as a repo-level bootstrap script but is not shipped with the pip package — users who install via `pip install kernelopt` have no guided path to configure Vertex AI credentials, verify GPU/NCU, or persist config. They must manually set environment variables every session.

## Solution

Two new Typer subcommands:

- `kernelopt setup` — one-time interactive wizard run after install
- `kernelopt doctor` — read-only health check, scriptable

A new `kernelopt/config.py` module persists config to `~/.config/kernelopt/config.toml`. The existing `run` command gains a fallback to read from this file.

---

## Architecture

### New files

**`kernelopt/kernelopt/config.py`**
Config file I/O. Responsible for reading and writing `~/.config/kernelopt/config.toml`. No side effects beyond file I/O.

```python
def config_path() -> Path: ...
def load_config() -> dict: ...
def save_config(project_id: str, region: str) -> Path: ...
```

**`kernelopt/kernelopt/functions/setup.py`**
Pure check functions. Each returns `(bool, str)` — success flag + human-readable message. No prompts, no file I/O, no side effects.

```python
def check_python_version() -> tuple[bool, str]: ...
def check_gpu() -> tuple[bool, str]: ...
def check_ncu() -> tuple[bool, str]: ...
def check_auth() -> tuple[bool, str]: ...
def check_vertex_connection(project_id: str, region: str) -> tuple[bool, str]: ...
```

### Modified files

**`kernelopt/kernelopt/cli.py`**
- Add `setup` command (interactive wizard)
- Add `doctor` command (read-only checks)
- Modify `run` command: if `project_id` / `region` not set via flag or env var, fall back to `load_config()`

**`kernelopt/pyproject.toml`**
- Change `anthropic>=0.80` → `anthropic[vertex]>=0.80`
- Add `google-auth-oauthlib`

### New test file

**`tests_suite/unit/functions/test_setup.py`**
Unit tests for all functions in `kernelopt/functions/setup.py`, using `unittest.mock` to mock subprocess calls.

---

## Config File

**Location:** `~/.config/kernelopt/config.toml`

**Format:**
```toml
[vertex]
project_id = "my-gcp-project"
region = "us-east5"
```

---

## Config Precedence

For `project_id` and `region` in the `run` command:

```
CLI flag  >  env var  >  ~/.config/kernelopt/config.toml  >  "" (fails with clear error)
```

| Setting | CLI flag | Env var |
|---|---|---|
| project_id | `--project-id` | `ANTHROPIC_VERTEX_PROJECT_ID` (already wired) |
| region | `--region` | `VERTEX_REGION` (must be added to cli.py in this task) |

---

## `kernelopt setup` Flow

1. Print welcome header
2. Load existing config (if present) — use as defaults for prompts
3. Prompt: `Google Cloud project ID [<current or none>]:`
4. Prompt: `Vertex AI region [us-east5]:`
5. Run OAuth via `google-auth-oauthlib` — opens browser, caches ADC credentials via Google's standard path
6. Write `~/.config/kernelopt/config.toml`
7. Run all checks with Rich spinners: GPU, NCU, Vertex AI connection
8. Print summary table
9. Print: `Ready! Run: kernelopt run <kernel.py>`

---

## `kernelopt doctor` Output

Read-only. Reads config + env vars, runs all checks, prints Rich table:

```
Check                Status   Detail
─────────────────────────────────────────────────────
Config file            ✓      ~/.config/kernelopt/config.toml
project_id             ✓      my-gcp-project
region                 ✓      us-east5
Auth credentials       ✓      application_default_credentials.json
GPU                    ✓      NVIDIA A100 (compute 8.0)
NCU binary             ✗      not found — install Nsight Compute
Vertex AI              ✓      connected (us-east5)
```

Exit code 0 if all checks pass, exit code 1 if any fail — scriptable for CI.

---

## Dependencies

| Package | Why |
|---|---|
| `anthropic[vertex]>=0.80` | Replaces `anthropic>=0.80`; pulls in `google-auth`, `google-cloud-aiplatform` |
| `google-auth-oauthlib` | Interactive OAuth flow in `kernelopt setup` |

---

## Out of Scope

- `setup_environment.py` is unchanged — it remains a repo-level bootstrap for contributors
- No changes to `kernelopt profile`, `kernelopt plan`, `kernelopt show`, `kernelopt kernels`
- No API key auth path (Vertex AI only)
