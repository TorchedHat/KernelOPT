import subprocess
from unittest.mock import MagicMock, patch

import pytest

from kernelopt.functions.compiler import compile_cuda, _format_nvcc_errors


def test_format_nvcc_errors_keeps_error_lines():
    stderr = (
        "/tmp/tmpXYZ/kernel.cu(42): error: identifier 'foo' is undefined\n"
        "1 error detected in the compilation of 'kernel.cu'.\n"
    )
    result = _format_nvcc_errors(stderr)
    assert "kernel.cu(42): error: identifier 'foo' is undefined" in result


def test_format_nvcc_errors_strips_tmp_prefix():
    stderr = "/tmp/tmpABC123/softmax.cu(10): error: expected ';'\n"
    result = _format_nvcc_errors(stderr)
    assert "/tmp/tmpABC123" not in result
    assert "softmax.cu(10): error: expected ';'" in result


def test_format_nvcc_errors_keeps_warning_lines():
    stderr = "/tmp/x/k.cu(5): warning: unused variable 'x'\n"
    result = _format_nvcc_errors(stderr)
    assert "k.cu(5): warning: unused variable 'x'" in result


def test_format_nvcc_errors_empty_stderr():
    assert _format_nvcc_errors("") == ""


def test_compile_cuda_success_returns_binary_path_and_none():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        binary, err = compile_cuda("/tmp/kernel.cu")
    assert binary == "/tmp/kernel"
    assert err is None
    cmd = mock_run.call_args[0][0]
    assert "nvcc" in cmd
    assert "/tmp/kernel.cu" in cmd
    assert "/tmp/kernel" in cmd


def test_compile_cuda_failure_returns_binary_path_and_error():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "/tmp/tmpXYZ/kernel.cu(5): error: expected ';'\n"
    with patch("subprocess.run", return_value=mock_result):
        binary, err = compile_cuda("/tmp/kernel.cu")
    assert binary == "/tmp/kernel"
    assert err is not None
    assert "kernel.cu(5): error: expected ';'" in err


def test_compile_cuda_binary_path_is_always_predictable():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        binary, _ = compile_cuda("/some/dir/my_kernel.cu")
    assert binary == "/some/dir/my_kernel"


def test_compile_cuda_invokes_nvcc_with_O2():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        compile_cuda("/tmp/k.cu")
    cmd = mock_run.call_args[0][0]
    assert "-O2" in cmd


def test_compile_cuda_timeout_returns_error():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["nvcc"], timeout=60)):
        binary, err = compile_cuda("/tmp/k.cu")
    assert binary == "/tmp/k"
    assert err is not None
    assert "timeout" in err.lower() or "timed out" in err.lower()
