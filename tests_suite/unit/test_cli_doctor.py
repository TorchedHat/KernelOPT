"""Tests for `kernelopt doctor` command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner()

# Environment variables that the doctor command checks — cleared in all tests
# so env-level config does not bleed into test assertions.
_CLEAR_ENV = {"ANTHROPIC_VERTEX_PROJECT_ID": "", "VERTEX_REGION": ""}


class TestDoctorCommand:
    def test_exits_0_when_all_checks_pass(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[vertex]\nproject_id = "proj"\nregion = "us-east5"\n')
        with (
            patch.dict("os.environ", _CLEAR_ENV),
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
            patch.dict("os.environ", _CLEAR_ENV),
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
            patch.dict("os.environ", _CLEAR_ENV),
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
            patch.dict("os.environ", _CLEAR_ENV),
            patch("kernelopt.config.config_path", return_value=cfg_path),
            patch("kernelopt.functions.setup.check_gpu", return_value=(True, "A100")),
            patch("kernelopt.functions.setup.check_ncu", return_value=(True, "NCU")),
            patch("kernelopt.functions.setup.check_auth", return_value=(True, "ok")),
            patch("kernelopt.functions.setup.check_vertex_connection") as mock_vertex,
        ):
            result = runner.invoke(app, ["doctor"])
        mock_vertex.assert_not_called()
        assert "skipped" in result.output
