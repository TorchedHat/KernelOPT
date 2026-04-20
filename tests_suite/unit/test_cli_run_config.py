"""Tests for run command config-file fallback for project_id and region."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from kernelopt.cli import app

runner = CliRunner()


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
            patch.dict("os.environ", {"ANTHROPIC_VERTEX_PROJECT_ID": "", "VERTEX_REGION": ""}, clear=False),
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
            patch.dict("os.environ", {"ANTHROPIC_VERTEX_PROJECT_ID": "", "VERTEX_REGION": ""}, clear=False),
        ):
            result = runner.invoke(app, ["run", str(kernel), "--project-id", "cli-project"])
        assert calls, f"Pipeline not called. Output:\n{result.output}"
        assert calls[0]["project_id"] == "cli-project"
