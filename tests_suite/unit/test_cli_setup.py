"""Tests for `kernelopt setup` command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner()


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
            patch.dict("os.environ", {"ANTHROPIC_VERTEX_PROJECT_ID": "", "VERTEX_REGION": ""}, clear=False),
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
