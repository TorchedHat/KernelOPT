"""Tests for `kernelopt setup` command.

The setup wizard is a multi-step interactive flow:
1. Choose LLM provider (numbered menu)
2. Provider-specific config (project_id/region for Vertex, URL for local)
3. Auth check (provider-specific)
4. Environment verification (GPU, NCU)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner()


class TestSetupCommand:
    def test_saves_vertex_config(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            # Input: choose Vertex (1), then project_id, then region
            result = runner.invoke(app, ["setup"], input="1\nmy-project\nus-east5\n")
        assert result.exit_code == 0, result.output
        assert cfg_path.exists()
        content = cfg_path.read_text()
        assert "my-project" in content

    def test_shows_ready_message(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            result = runner.invoke(app, ["setup"], input="1\nproj\nus-east5\n")
        assert "Ready" in result.output

    def test_uses_existing_vertex_defaults(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            '[provider]\nname = "vertex"\n\n'
            '[vertex]\nproject_id = "existing-proj"\nregion = "us-central1"\n'
        )
        with (
            patch.dict("os.environ", {"ANTHROPIC_VERTEX_PROJECT_ID": "", "VERTEX_REGION": ""}, clear=False),
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_vertex_connection", return_value=(True, "connected")),
        ):
            # Hit enter 3 times: accept default provider (1), default project, default region
            result = runner.invoke(app, ["setup"], input="\n\n\n")
        assert result.exit_code == 0, result.output
        content = cfg_path.read_text()
        assert "existing-proj" in content

    def test_shows_auth_instructions_when_no_gcloud(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_auth", return_value=(False, "no credentials")),
            patch("shutil.which", return_value=None),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
        ):
            result = runner.invoke(app, ["setup"], input="1\nproj\nus-east5\n")
        assert "gcloud" in result.output or "GOOGLE_APPLICATION_CREDENTIALS" in result.output

    def test_invalid_provider_choice(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with patch("kernelopt.config.config_path", return_value=cfg_path):
            result = runner.invoke(app, ["setup"], input="99\n")
        assert result.exit_code == 1
        assert "Invalid" in result.output

    def test_local_provider_config(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        with (
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_local_connection", return_value=(True, "connected")),
        ):
            # Choose local (6), then URL, then model name
            result = runner.invoke(
                app, ["setup"],
                input="6\nhttp://localhost:8000/v1\nllama-3\n",
            )
        assert result.exit_code == 0, result.output
        assert cfg_path.exists()
