# kernelopt setup + doctor Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `kernelopt setup` (interactive wizard) and `kernelopt doctor` (read-only health check) so a user can go from `pip install kernelopt` to a fully working environment without any external scripts.

**Architecture:** `kernelopt/config.py` persists config to `~/.config/kernelopt/config.toml`; pure check functions in `kernelopt/functions/setup.py` return `(bool, str)` tuples; two new Typer commands compose them; the existing `run` command gains a config-file fallback for `project_id` and `region`.

**Tech Stack:** Python 3.10+, Typer, Rich, tomllib (stdlib 3.11+) / tomli (3.10 fallback), google-auth, anthropic[vertex], google-auth-oauthlib

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `kernelopt/pyproject.toml` | Modify | Add new deps |
| `kernelopt/kernelopt/config.py` | Create | Config file I/O (`~/.config/kernelopt/config.toml`) |
| `kernelopt/kernelopt/functions/setup.py` | Create | Pure environment check functions returning `(bool, str)` |
| `kernelopt/kernelopt/cli.py` | Modify | Add `setup` and `doctor` commands; update `run` fallback |
| `tests_suite/unit/test_config.py` | Create | Tests for config.py |
| `tests_suite/unit/functions/test_setup.py` | Create | Tests for check functions |
| `tests_suite/unit/test_cli_doctor.py` | Create | Tests for `doctor` command |
| `tests_suite/unit/test_cli_setup.py` | Create | Tests for `setup` command |
| `tests_suite/unit/test_cli_run_config.py` | Create | Tests for `run` config fallback |

---

### Task 1: Update pyproject.toml dependencies

**Files:**
- Modify: `kernelopt/pyproject.toml`

- [ ] **Step 1: Edit dependencies in pyproject.toml**

Replace the `dependencies` list in `kernelopt/pyproject.toml` with:

```toml
[project]
name = "kernelopt"
version = "0.1.0"
description = "Multi-agent GPU kernel optimizer powered by LangGraph and NCU"
license = { text = "MIT" }
requires-python = ">=3.10"

dependencies = [
    "typer[all]>=0.12",
    "langgraph>=0.2",
    "langchain-anthropic>=1.0",
    "anthropic[vertex]>=0.80",
    "pydantic>=2.0",
    "rich>=13.0",
    "torch>=2.0",
    "google-auth-oauthlib>=1.0",
    "tomli>=2.0; python_version < '3.11'",
]
```

(Changed: `anthropic>=0.80` → `anthropic[vertex]>=0.80`; Added: `google-auth-oauthlib>=1.0`, `tomli>=2.0; python_version < '3.11'`)

- [ ] **Step 2: Reinstall the package**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt && pip install -e ".[triton]"
```

Expected: pip output ending in `Successfully installed` or `Requirement already satisfied`.

- [ ] **Step 3: Verify new deps are importable**

```bash
python -c "import google.auth; import google_auth_oauthlib; print('deps ok')"
```

Expected: `deps ok`

- [ ] **Step 4: Commit**

```bash
git add kernelopt/pyproject.toml
git commit -m "feat: update deps for kernelopt setup command (anthropic[vertex], google-auth-oauthlib, tomli)"
```

---

### Task 2: Create kernelopt/config.py (TDD)

**Files:**
- Create: `kernelopt/kernelopt/config.py`
- Create: `tests_suite/unit/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests_suite/unit/test_config.py`:

```python
"""Tests for kernelopt.config — config file I/O."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigPath:
    def test_returns_path_ending_in_config_toml(self):
        from kernelopt.config import config_path
        p = config_path()
        assert p.name == "config.toml"
        assert p.parent.name == "kernelopt"
        assert ".config" in str(p)


class TestLoadConfig:
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        from kernelopt.config import load_config
        missing = tmp_path / "config.toml"
        with patch("kernelopt.config.config_path", return_value=missing):
            result = load_config()
        assert result == {}

    def test_returns_parsed_toml(self, tmp_path):
        from kernelopt.config import load_config
        cfg = tmp_path / "config.toml"
        cfg.write_text('[vertex]\nproject_id = "my-project"\nregion = "us-east5"\n')
        with patch("kernelopt.config.config_path", return_value=cfg):
            result = load_config()
        assert result["vertex"]["project_id"] == "my-project"
        assert result["vertex"]["region"] == "us-east5"

    def test_returns_empty_dict_on_malformed_toml(self, tmp_path):
        from kernelopt.config import load_config
        cfg = tmp_path / "config.toml"
        cfg.write_text("not valid = = toml ===")
        with patch("kernelopt.config.config_path", return_value=cfg):
            result = load_config()
        assert result == {}


class TestSaveConfig:
    def test_writes_project_id_and_region(self, tmp_path):
        from kernelopt.config import save_config
        target = tmp_path / "kernelopt" / "config.toml"
        with patch("kernelopt.config.config_path", return_value=target):
            path = save_config("my-project", "us-east5")
        assert path == target
        content = target.read_text()
        assert "my-project" in content
        assert "us-east5" in content

    def test_creates_parent_directories(self, tmp_path):
        from kernelopt.config import save_config
        target = tmp_path / "deep" / "nested" / "config.toml"
        with patch("kernelopt.config.config_path", return_value=target):
            save_config("proj", "us-east5")
        assert target.exists()

    def test_roundtrip_load_after_save(self, tmp_path):
        from kernelopt.config import load_config, save_config
        target = tmp_path / "config.toml"
        with patch("kernelopt.config.config_path", return_value=target):
            save_config("roundtrip-project", "europe-west4")
            result = load_config()
        assert result["vertex"]["project_id"] == "roundtrip-project"
        assert result["vertex"]["region"] == "europe-west4"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
python -m pytest tests_suite/unit/test_config.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.config'`

- [ ] **Step 3: Create kernelopt/kernelopt/config.py**

```python
"""Config file I/O for kernelopt — reads/writes ~/.config/kernelopt/config.toml."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


def config_path() -> Path:
    return Path.home() / ".config" / "kernelopt" / "config.toml"


def load_config() -> dict:
    path = config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def save_config(project_id: str, region: str) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[vertex]\n"
        f'project_id = "{project_id}"\n'
        f'region = "{region}"\n'
    )
    path.write_text(content, encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests_suite/unit/test_config.py -v
```

Expected: 6 tests, all PASSED.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/config.py tests_suite/unit/test_config.py
git commit -m "feat: add config.py for ~/.config/kernelopt/config.toml I/O"
```

---

### Task 3: Create kernelopt/functions/setup.py (TDD)

**Files:**
- Create: `kernelopt/kernelopt/functions/setup.py`
- Create: `tests_suite/unit/functions/test_setup.py`

- [ ] **Step 1: Write failing tests**

Create `tests_suite/unit/functions/test_setup.py`:

```python
"""Tests for kernelopt.functions.setup — pure environment check functions."""
from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCheckPythonVersion:
    def test_passes_for_current_python(self):
        from kernelopt.functions.setup import check_python_version
        ok, msg = check_python_version()
        assert ok  # tests always run on Python 3.10+
        assert f"{sys.version_info[0]}.{sys.version_info[1]}" in msg

    def test_message_starts_with_python(self):
        from kernelopt.functions.setup import check_python_version
        _, msg = check_python_version()
        assert msg.startswith("Python ")


class TestCheckGpu:
    def test_passes_for_capable_gpu(self):
        from kernelopt.functions.setup import check_gpu
        mock_result = MagicMock()
        mock_result.stdout = "8.0, NVIDIA A100-SXM4-80GB\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_gpu()
        assert ok
        assert "A100" in msg
        assert "8.0" in msg

    def test_fails_when_nvidia_smi_not_found(self):
        from kernelopt.functions.setup import check_gpu
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ok, msg = check_gpu()
        assert not ok
        assert "nvidia-smi" in msg

    def test_fails_for_compute_capability_below_6(self):
        from kernelopt.functions.setup import check_gpu
        mock_result = MagicMock()
        mock_result.stdout = "5.2, NVIDIA GTX 960\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_gpu()
        assert not ok
        assert "5.2" in msg

    def test_fails_on_timeout(self):
        from kernelopt.functions.setup import check_gpu
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            ok, msg = check_gpu()
        assert not ok

    def test_fails_when_stdout_empty(self):
        from kernelopt.functions.setup import check_gpu
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_gpu()
        assert not ok


class TestCheckNcu:
    def test_passes_when_ncu_available(self):
        from kernelopt.functions.setup import check_ncu
        mock_result = MagicMock()
        mock_result.stdout = "NVIDIA Nsight Compute, Version 2023.2.0\nBuild ...\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, msg = check_ncu()
        assert ok

    def test_fails_when_ncu_not_found(self):
        from kernelopt.functions.setup import check_ncu
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ok, msg = check_ncu()
        assert not ok
        assert "not found" in msg

    def test_fails_on_subprocess_error(self):
        from kernelopt.functions.setup import check_ncu
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ncu")):
            ok, msg = check_ncu()
        assert not ok


class TestCheckAuth:
    def test_passes_when_credentials_present(self):
        from kernelopt.functions.setup import check_auth
        mock_creds = MagicMock()
        with patch("google.auth.default", return_value=(mock_creds, "my-project")):
            ok, msg = check_auth()
        assert ok
        assert "my-project" in msg

    def test_passes_with_none_project_in_credentials(self):
        from kernelopt.functions.setup import check_auth
        mock_creds = MagicMock()
        with patch("google.auth.default", return_value=(mock_creds, None)):
            ok, msg = check_auth()
        assert ok

    def test_fails_when_no_credentials(self):
        from kernelopt.functions.setup import check_auth
        with patch("google.auth.default", side_effect=Exception("could not find credentials")):
            ok, msg = check_auth()
        assert not ok
        assert "gcloud" in msg or "credentials" in msg.lower()


class TestCheckVertexConnection:
    def test_passes_when_vertex_connects(self):
        from kernelopt.functions.setup import check_vertex_connection
        with patch("anthropic.AnthropicVertex"):
            ok, msg = check_vertex_connection("my-project", "us-east5")
        assert ok
        assert "us-east5" in msg

    def test_fails_on_permission_denied(self):
        from kernelopt.functions.setup import check_vertex_connection
        with patch("anthropic.AnthropicVertex", side_effect=Exception("403 permission denied")):
            ok, msg = check_vertex_connection("my-project", "us-east5")
        assert not ok
        assert "permission" in msg

    def test_fails_on_project_not_found(self):
        from kernelopt.functions.setup import check_vertex_connection
        with patch("anthropic.AnthropicVertex", side_effect=Exception("404 project not found")):
            ok, msg = check_vertex_connection("bad-project", "us-east5")
        assert not ok
        assert "not found" in msg

    def test_fails_when_import_error(self):
        from kernelopt.functions.setup import check_vertex_connection
        with patch("anthropic.AnthropicVertex", side_effect=ImportError):
            ok, msg = check_vertex_connection("my-project", "us-east5")
        assert not ok
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests_suite/unit/functions/test_setup.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.functions.setup'`

- [ ] **Step 3: Create kernelopt/kernelopt/functions/setup.py**

```python
"""Pure environment check functions — each returns (ok: bool, message: str)."""
from __future__ import annotations

import subprocess
import sys


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v[0], v[1]) >= (3, 10)
    ver = f"{v[0]}.{v[1]}.{v[2]}"
    return ok, f"Python {ver}" + ("" if ok else " (need >= 3.10)")


def check_gpu() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        line = result.stdout.strip()
        if not line:
            return False, "no GPU detected by nvidia-smi"
        parts = line.split(",", 1)
        cap = float(parts[0].strip())
        name = parts[1].strip() if len(parts) > 1 else "Unknown GPU"
        if cap >= 6.0:
            return True, f"{name} (compute {cap})"
        return False, f"{name} has compute {cap} (need >= 6.0)"
    except FileNotFoundError:
        return False, "nvidia-smi not found — install NVIDIA drivers"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"
    except subprocess.CalledProcessError:
        return False, "nvidia-smi failed — check driver installation"
    except (ValueError, IndexError):
        return False, "could not parse GPU info from nvidia-smi"


def check_ncu() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True, check=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Version" in line:
                return True, f"NCU {line.split('Version')[-1].strip()}"
        return True, "NCU installed"
    except FileNotFoundError:
        return False, "not found — install from https://developer.nvidia.com/nsight-compute"
    except subprocess.CalledProcessError:
        return False, "ncu failed to run"
    except subprocess.TimeoutExpired:
        return False, "ncu timed out"


def check_auth() -> tuple[bool, str]:
    try:
        import google.auth
        _creds, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return True, f"application default credentials (project: {project or 'unknown'})"
    except Exception:
        return False, "no credentials — run: gcloud auth application-default login"


def check_vertex_connection(project_id: str, region: str) -> tuple[bool, str]:
    try:
        from anthropic import AnthropicVertex
        AnthropicVertex(project_id=project_id, region=region)
        return True, f"connected ({region})"
    except ImportError:
        return False, "anthropic[vertex] not installed"
    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "permission" in msg.lower():
            return False, "permission denied (need Vertex AI User role)"
        if "404" in msg:
            return False, f"project '{project_id}' or region '{region}' not found"
        return False, f"connection failed: {msg[:80]}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests_suite/unit/functions/test_setup.py -v
```

Expected: 15 tests, all PASSED.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/functions/setup.py tests_suite/unit/functions/test_setup.py
git commit -m "feat: add functions/setup.py with pure environment check functions"
```

---

### Task 4: Add `kernelopt doctor` command (TDD)

**Files:**
- Modify: `kernelopt/kernelopt/cli.py` — add `doctor` command before `_print_profile` (after the `kernels` command, around line 554)
- Create: `tests_suite/unit/test_cli_doctor.py`

- [ ] **Step 1: Write failing tests**

Create `tests_suite/unit/test_cli_doctor.py`:

```python
"""Tests for `kernelopt doctor` command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner(mix_stderr=False)


class TestDoctorCommand:
    def test_exits_0_when_all_checks_pass(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "proj"\nregion = "us-east5"\n')
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100 (compute 8.0)")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU 2023.2.0")),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "application default credentials")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected (us-east5)")),
        ):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

    def test_exits_1_when_gpu_check_fails(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "proj"\nregion = "us-east5"\n')
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_gpu", return_value=(False, "nvidia-smi not found")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU 2023.2.0")),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1

    def test_shows_project_id_in_output(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "my-gcp-project"\nregion = "us-east5"\n')
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            result = runner.invoke(app, ["doctor"])
        assert "my-gcp-project" in result.output

    def test_skips_vertex_check_when_project_id_missing(self, tmp_path):
        cfg_path = tmp_path / "missing.toml"  # does not exist
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_vertex_connection") as mock_vertex,
        ):
            result = runner.invoke(app, ["doctor"])
        mock_vertex.assert_not_called()
        assert "skipped" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests_suite/unit/test_cli_doctor.py -v 2>&1 | head -20
```

Expected: `Error: No such command 'doctor'`

- [ ] **Step 3: Add `doctor` command to cli.py**

In `kernelopt/kernelopt/cli.py`, add this block immediately before the `def _print_profile` function (after the `kernels` command, around line 554):

```python
@app.command()
def doctor():
    """Check environment health — config, credentials, GPU, NCU, Vertex AI."""
    from rich.table import Table

    from kernelopt.config import config_path, load_config
    from kernelopt.functions.setup import (
        check_auth,
        check_gpu,
        check_ncu,
        check_python_version,
        check_vertex_connection,
    )

    cfg = load_config().get("vertex", {})
    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or cfg.get("project_id", "")
    region = os.environ.get("VERTEX_REGION") or cfg.get("region", "")

    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Check", style="dim", min_width=22)
    t.add_column("Status", justify="center", min_width=6)
    t.add_column("Detail")

    all_ok = True

    def _row(label: str, ok: bool, detail: str) -> None:
        nonlocal all_ok
        if not ok:
            all_ok = False
        t.add_row(label, "[green]✓[/]" if ok else "[red]✗[/]", detail)

    cp = config_path()
    _row("Config file", cp.exists(), str(cp) if cp.exists() else "not found — run: kernelopt setup")
    _row("project_id", bool(project_id), project_id or "not set — run: kernelopt setup")
    _row("region", bool(region), region or "not set — run: kernelopt setup")

    ok, msg = check_python_version()
    _row("Python", ok, msg)

    ok, msg = check_auth()
    _row("Auth credentials", ok, msg)

    ok, msg = check_gpu()
    _row("GPU", ok, msg)

    ok, msg = check_ncu()
    _row("NCU binary", ok, msg)

    if project_id and region:
        ok, msg = check_vertex_connection(project_id, region)
        _row("Vertex AI", ok, msg)
    else:
        _row("Vertex AI", False, "skipped — project_id or region not set")

    console.print(t)
    raise typer.Exit(0 if all_ok else 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests_suite/unit/test_cli_doctor.py -v
```

Expected: 4 tests, all PASSED.

- [ ] **Step 5: Manual smoke test**

```bash
kernelopt doctor
echo "Exit code: $?"
```

Expected: A table printed to terminal. Exit code `1` (config not yet set up in this env).

- [ ] **Step 6: Commit**

```bash
git add kernelopt/kernelopt/cli.py tests_suite/unit/test_cli_doctor.py
git commit -m "feat: add kernelopt doctor command"
```

---

### Task 5: Add `kernelopt setup` command (TDD)

**Files:**
- Modify: `kernelopt/kernelopt/cli.py` — add `setup` command immediately before the `doctor` command added in Task 4
- Create: `tests_suite/unit/test_cli_setup.py`

- [ ] **Step 1: Write failing tests**

Create `tests_suite/unit/test_cli_setup.py`:

```python
"""Tests for `kernelopt setup` command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner(mix_stderr=False)


class TestSetupCommand:
    def test_saves_config_with_provided_inputs(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            result = runner.invoke(
                app, ["setup"],
                input="my-project\nus-east5\n",  # answers to project_id prompt, then region prompt
            )
        assert result.exit_code == 0, result.output
        assert cfg_path.exists()
        content = cfg_path.read_text()
        assert "my-project" in content
        assert "us-east5" in content

    def test_shows_ready_message_on_success(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            result = runner.invoke(app, ["setup"], input="proj\nus-east5\n")
        assert "Ready" in result.output

    def test_uses_existing_config_as_defaults(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "existing-proj"\nregion = "us-central1"\n')
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            # User hits enter twice to accept both defaults
            result = runner.invoke(app, ["setup"], input="\n\n")
        assert result.exit_code == 0, result.output
        content = cfg_path.read_text()
        assert "existing-proj" in content

    def test_shows_auth_instructions_when_no_gcloud_and_no_credentials(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(False, "no credentials")),
            patch("shutil.which", return_value=None),  # gcloud not on PATH
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
        ):
            result = runner.invoke(app, ["setup"], input="proj\nus-east5\n")
        assert "gcloud" in result.output or "GOOGLE_APPLICATION_CREDENTIALS" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests_suite/unit/test_cli_setup.py -v 2>&1 | head -20
```

Expected: `Error: No such command 'setup'`

- [ ] **Step 3: Add `setup` command to cli.py**

In `kernelopt/kernelopt/cli.py`, add this block immediately **before** the `doctor` command added in Task 4:

```python
@app.command()
def setup():
    """Interactive wizard — configure Vertex AI credentials and verify environment."""
    import shutil
    import subprocess as _sp

    from rich.table import Table

    from kernelopt.config import config_path, load_config, save_config
    from kernelopt.functions.setup import (
        check_auth,
        check_gpu,
        check_ncu,
        check_vertex_connection,
    )

    console.print("\n[bold green]kernelopt setup[/]\n")
    console.print("Configures Vertex AI and verifies your environment.\n")

    existing = load_config().get("vertex", {})
    default_project = (
        os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or existing.get("project_id") or ""
    )
    default_region = (
        os.environ.get("VERTEX_REGION") or existing.get("region") or "us-east5"
    )

    if default_project:
        project_id = typer.prompt("Google Cloud project ID", default=default_project)
    else:
        project_id = typer.prompt("Google Cloud project ID")

    region = typer.prompt("Vertex AI region", default=default_region)

    path = save_config(project_id, region)
    console.print(f"\n[green]✓[/] Config saved → [cyan]{path}[/]\n")

    console.print("[bold]Checking authentication...[/]")
    auth_ok, auth_msg = check_auth()
    if auth_ok:
        console.print(f"[green]✓[/] {auth_msg}")
    else:
        console.print(f"[yellow]⚠[/] {auth_msg}")
        if shutil.which("gcloud"):
            console.print("\nLaunching: gcloud auth application-default login")
            try:
                _sp.run(["gcloud", "auth", "application-default", "login"], check=True)
                auth_ok, auth_msg = check_auth()
                if auth_ok:
                    console.print("[green]✓[/] Authenticated")
                else:
                    console.print("[red]✗[/] Auth failed after gcloud login")
            except _sp.CalledProcessError:
                console.print("[red]✗[/] gcloud auth failed")
        else:
            console.print("\nOptions to set up credentials:")
            console.print("  • Install gcloud: https://cloud.google.com/sdk/docs/install")
            console.print("    Then run: gcloud auth application-default login")
            console.print("  • Or set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")

    console.print("\n[bold]Verifying environment...[/]\n")

    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Check", style="dim", min_width=22)
    t.add_column("Status", justify="center", min_width=6)
    t.add_column("Detail")

    def _row(label: str, ok: bool, detail: str) -> None:
        t.add_row(label, "[green]✓[/]" if ok else "[red]✗[/]", detail)

    ok, msg = check_gpu()
    _row("GPU", ok, msg)

    ok, msg = check_ncu()
    _row("NCU binary", ok, msg)

    if auth_ok:
        ok, msg = check_vertex_connection(project_id, region)
        _row("Vertex AI", ok, msg)

    console.print(t)
    console.print(f"\n[bold green]Ready![/] Run: [cyan]kernelopt run <kernel.py>[/]")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests_suite/unit/test_cli_setup.py -v
```

Expected: 4 tests, all PASSED.

- [ ] **Step 5: Manual smoke test**

```bash
kernelopt setup
```

Expected: Prompts for project ID and region, then runs checks and prints a summary table.

- [ ] **Step 6: Commit**

```bash
git add kernelopt/kernelopt/cli.py tests_suite/unit/test_cli_setup.py
git commit -m "feat: add kernelopt setup interactive wizard"
```

---

### Task 6: Update `run` command — config fallback + VERTEX_REGION env var

**Files:**
- Modify: `kernelopt/kernelopt/cli.py`
- Create: `tests_suite/unit/test_cli_run_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests_suite/unit/test_cli_run_config.py`:

```python
"""Tests for run command config-file fallback for project_id and region."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner(mix_stderr=False)


def _make_mock_pipeline(calls: list):
    def _pipeline(state):
        calls.append({"project_id": state.config.project_id, "region": state.config.region})
        final = MagicMock()
        final.baseline_profiling_context = {}
        final.best_latency_ns = float("nan")
        return final
    return _pipeline


class TestRunConfigFallback:
    def test_reads_project_id_from_config_file(self, tmp_path):
        kernel = tmp_path / "kernel.py"
        kernel.write_text("# dummy kernel")
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "config-project"\nregion = "us-east5"\n')
        calls: list = []
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.langgraph_pipeline.run_langgraph_pipeline", side_effect=_make_mock_pipeline(calls)),
            patch("kernelopt.functions.gpu_monitoring.select_gpu", return_value=0),
            patch("kernelopt.functions.gpu_monitoring.validate_gpu_for_profiling", return_value={}),
            patch("kernelopt.agents.profiler.get_latency_ns", return_value=float("nan")),
        ):
            result = runner.invoke(app, ["run", str(kernel)])
        assert calls, f"Pipeline not called. Output:\n{result.output}"
        assert calls[0]["project_id"] == "config-project"

    def test_cli_flag_overrides_config_file(self, tmp_path):
        kernel = tmp_path / "kernel.py"
        kernel.write_text("# dummy kernel")
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "config-project"\nregion = "us-east5"\n')
        calls: list = []
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.langgraph_pipeline.run_langgraph_pipeline", side_effect=_make_mock_pipeline(calls)),
            patch("kernelopt.functions.gpu_monitoring.select_gpu", return_value=0),
            patch("kernelopt.functions.gpu_monitoring.validate_gpu_for_profiling", return_value={}),
            patch("kernelopt.agents.profiler.get_latency_ns", return_value=float("nan")),
        ):
            result = runner.invoke(app, ["run", str(kernel), "--project-id", "cli-project"])
        assert calls, f"Pipeline not called. Output:\n{result.output}"
        assert calls[0]["project_id"] == "cli-project"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests_suite/unit/test_cli_run_config.py -v 2>&1 | head -30
```

Expected: Tests FAIL — `run` doesn't yet read from the config file, so `project_id` stays empty.

- [ ] **Step 3: Add `envvar="VERTEX_REGION"` to the `--region` option in `run`**

In `kernelopt/kernelopt/cli.py`, find this inside the `run` command parameter list (around line 154):

```python
    region: str = typer.Option(
        "us-east5", "--region", help="Vertex AI region"
    ),
```

Replace with:

```python
    region: str = typer.Option(
        "us-east5", "--region", envvar="VERTEX_REGION", help="Vertex AI region"
    ),
```

- [ ] **Step 4: Add config file fallback at the top of the `run` function body**

In `kernelopt/kernelopt/cli.py`, at the very start of the `run` function body — before the `ckpt: dict | None = None` line (around line 200) — add:

```python
    from kernelopt.config import load_config as _load_config
    _file_cfg = _load_config().get("vertex", {})
    if not project_id:
        project_id = _file_cfg.get("project_id") or ""
    if (
        ctx.get_parameter_source("region") not in (
            ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT
        )
        and _file_cfg.get("region")
    ):
        region = _file_cfg["region"]
```

Note: `ParameterSource` is already imported at line 11 of `cli.py` — no new import needed.

- [ ] **Step 5: Run new tests to verify they pass**

```bash
python -m pytest tests_suite/unit/test_cli_run_config.py -v
```

Expected: 2 tests, all PASSED.

- [ ] **Step 6: Run full unit suite to check for regressions**

```bash
python -m pytest tests_suite/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: All previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add kernelopt/kernelopt/cli.py tests_suite/unit/test_cli_run_config.py
git commit -m "feat: run command falls back to config file for project_id/region; add VERTEX_REGION envvar"
```

---

## Self-Review

**Spec coverage:**
- ✓ `config.py` with `load_config`, `save_config`, `config_path` → Task 2
- ✓ `functions/setup.py` with all 5 check functions → Task 3
- ✓ `kernelopt doctor` with Rich table + exit 1 on failure → Task 4
- ✓ `kernelopt setup` interactive wizard with gcloud fallback → Task 5
- ✓ `run` config fallback + `VERTEX_REGION` envvar → Task 6
- ✓ `anthropic[vertex]>=0.80`, `google-auth-oauthlib`, `tomli` deps → Task 1

**Placeholder scan:** None. All steps contain concrete code.

**Type consistency:**
- `check_*` functions → `tuple[bool, str]` — consistent across Tasks 3, 4, 5
- `load_config()` → `dict`, consumed as `.get("vertex", {})` — consistent across Tasks 4, 5, 6
- `save_config(project_id: str, region: str) -> Path` — defined Task 2, used Task 5
- `config_path() -> Path` — defined Task 2, used in Task 4 doctor command
