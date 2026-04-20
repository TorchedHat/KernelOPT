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
