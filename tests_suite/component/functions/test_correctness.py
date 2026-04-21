"""Component tests for kernelopt.functions.correctness."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# ---------------------------------------------------------------------------
# Task 5: _has_builtin_self_test
# ---------------------------------------------------------------------------

from kernelopt.functions.correctness import _has_builtin_self_test

WITH_CHECK_FN = """\
#include <cuda_runtime.h>
static int check_correctness(int N) {
    printf("Correctness: PASS\\n");
    return 1;
}
int main() { check_correctness(4096); }
"""

WITH_PASS_PRINTF = """\
#include <stdio.h>
int main() {
    int ok = run_test();
    if (ok) printf("PASS\\n");
    else printf("FAIL\\n");
}
"""

WITHOUT_SELF_TEST = """\
#include <cuda_runtime.h>
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

KERNEL_ONLY = """\
__global__ void softmax(float* x, int N) {
}
"""


def test_has_builtin_self_test_true_when_main_present():
    # WITH_CHECK_FN has int main() — detection is by main(), not the function name
    assert _has_builtin_self_test(WITH_CHECK_FN) is True


def test_has_builtin_self_test_true_for_source_with_main_and_pass_printf():
    # WITH_PASS_PRINTF has int main() — detection is by main(), not the printf strings
    assert _has_builtin_self_test(WITH_PASS_PRINTF) is True


def test_has_builtin_self_test_returns_false_for_kernel_without_main():
    assert _has_builtin_self_test(WITHOUT_SELF_TEST) is False


def test_has_builtin_self_test_returns_false_for_kernel_only():
    assert _has_builtin_self_test(KERNEL_ONLY) is False


def test_has_builtin_self_test_returns_false_for_empty_string():
    assert _has_builtin_self_test("") is False


# ---------------------------------------------------------------------------
# Task 6: _check_cuda_kernel with binary_path
# ---------------------------------------------------------------------------

from kernelopt.functions.correctness import _check_cuda_kernel


def test_check_cuda_kernel_with_binary_path_skips_nvcc():
    """When binary_path is provided, nvcc should not be called."""
    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="ok\n", stderr=""))

    with patch("subprocess.run", mock_run), \
         patch("pathlib.Path.exists", return_value=True):
        passed, msg = _check_cuda_kernel(
            original_path="/tmp/k.cu",
            optimized_source="__global__ void k(){}",
            binary_path="/tmp/k",
        )

    calls = mock_run.call_args_list
    assert len(calls) == 1
    cmd = calls[0][0][0]
    assert "nvcc" not in cmd[0]


def test_check_cuda_kernel_passes_on_exit_zero():
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="ok\n", stderr="")), \
         patch("pathlib.Path.exists", return_value=True):
        passed, msg = _check_cuda_kernel("/tmp/k.cu", "__global__ void k(){}", binary_path="/tmp/k")
    assert passed is True


def test_check_cuda_kernel_fails_on_nonzero_exit():
    with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="CUDA error")), \
         patch("pathlib.Path.exists", return_value=True):
        passed, msg = _check_cuda_kernel("/tmp/k.cu", "__global__ void k(){}", binary_path="/tmp/k")
    assert passed is False
    assert "1" in msg


# ---------------------------------------------------------------------------
# Task 7: _check_cuda_with_llm_reference
# ---------------------------------------------------------------------------

from kernelopt.functions.correctness import _check_cuda_with_llm_reference

SIMPLE_CUDA_SOURCE = """\
#include <cuda_runtime.h>
__global__ void scale(float* x, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""


def _make_mock_llm_with_spec(harness_code, skip=False, skip_reason=None):
    from kernelopt.agents.schemas import CudaCorrectnessSpec
    spec = CudaCorrectnessSpec(test_harness_code=harness_code, skip=skip, skip_reason=skip_reason)
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = spec
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


def test_check_cuda_with_llm_reference_pass():
    mock_llm = _make_mock_llm_with_spec("#include<stdio.h>\nint main(){printf(\"Correctness: PASS\\n\");return 0;}")
    compile_ok = MagicMock(returncode=0, stdout="", stderr="")
    run_ok = MagicMock(returncode=0, stdout="Correctness: PASS\n", stderr="")
    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm), \
         patch("subprocess.run", side_effect=[compile_ok, run_ok]), \
         patch("pathlib.Path.unlink"):
        passed, msg = _check_cuda_with_llm_reference(
            cu_source=SIMPLE_CUDA_SOURCE, kernel_path="/tmp/scale.cu",
            api_key=None, model="unused", project_id=None, region=None,
        )
    assert passed is True


def test_check_cuda_with_llm_reference_fail():
    mock_llm = _make_mock_llm_with_spec("#include<stdio.h>\nint main(){printf(\"Correctness: FAIL\\n\");return 1;}")
    compile_ok = MagicMock(returncode=0, stdout="", stderr="")
    run_fail = MagicMock(returncode=1, stdout="Correctness: FAIL\n", stderr="")
    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm), \
         patch("subprocess.run", side_effect=[compile_ok, run_fail]), \
         patch("pathlib.Path.unlink"):
        passed, msg = _check_cuda_with_llm_reference(
            cu_source=SIMPLE_CUDA_SOURCE, kernel_path="/tmp/scale.cu",
            api_key=None, model="unused", project_id=None, region=None,
        )
    assert passed is False


def test_check_cuda_with_llm_reference_passes_on_exit_zero_without_pass_string():
    """Exit code 0 is sufficient — no 'PASS' string required."""
    mock_llm = _make_mock_llm_with_spec(
        "#include<stdio.h>\nint main(){printf(\"All done.\\n\");return 0;}"
    )
    compile_ok = MagicMock(returncode=0, stdout="", stderr="")
    run_ok = MagicMock(returncode=0, stdout="All done.\n", stderr="")
    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm), \
         patch("subprocess.run", side_effect=[compile_ok, run_ok]), \
         patch("pathlib.Path.unlink"):
        passed, msg = _check_cuda_with_llm_reference(
            cu_source=SIMPLE_CUDA_SOURCE, kernel_path="/tmp/scale.cu",
            api_key=None, model="unused", project_id=None, region=None,
        )
    assert passed is True, f"Expected pass but got: {msg}"


def test_check_cuda_with_llm_reference_skip():
    mock_llm = _make_mock_llm_with_spec("", skip=True, skip_reason="custom RNG")
    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm):
        passed, msg = _check_cuda_with_llm_reference(
            cu_source=SIMPLE_CUDA_SOURCE, kernel_path="/tmp/scale.cu",
            api_key=None, model="unused", project_id=None, region=None,
        )
    assert passed is False
    assert "custom RNG" in msg or "skip" in msg.lower()


# ---------------------------------------------------------------------------
# Task 8: check_correctness dispatcher
# ---------------------------------------------------------------------------

from kernelopt.functions.correctness import check_correctness

SOURCE_WITH_SELF_TEST = WITH_CHECK_FN
SOURCE_WITHOUT_SELF_TEST = WITHOUT_SELF_TEST


def test_check_correctness_cu_with_self_test_uses_subprocess_path():
    with patch("kernelopt.functions.correctness._check_cuda_kernel", return_value=(True, "ok")) as mock_ck, \
         patch("kernelopt.functions.correctness._check_cuda_with_llm_reference") as mock_llm:
        passed, msg = check_correctness(
            "/tmp/k.cu", SOURCE_WITH_SELF_TEST, "cuda", use_agentic=True, binary_path="/tmp/k",
        )
    mock_ck.assert_called_once()
    mock_llm.assert_not_called()
    assert passed is True


def test_check_correctness_cu_without_self_test_and_agentic_uses_llm():
    with patch("kernelopt.functions.correctness._check_cuda_kernel") as mock_ck, \
         patch("kernelopt.functions.correctness._check_cuda_with_llm_reference", return_value=(True, "ok")) as mock_llm:
        passed, msg = check_correctness(
            "/tmp/k.cu", SOURCE_WITHOUT_SELF_TEST, "cuda", use_agentic=True, binary_path="/tmp/k",
        )
    mock_llm.assert_called_once()
    mock_ck.assert_not_called()
    assert passed is True


def test_check_correctness_cu_without_self_test_non_agentic_uses_subprocess():
    with patch("kernelopt.functions.correctness._check_cuda_kernel", return_value=(True, "ok")) as mock_ck, \
         patch("kernelopt.functions.correctness._check_cuda_with_llm_reference") as mock_llm:
        passed, msg = check_correctness(
            "/tmp/k.cu", SOURCE_WITHOUT_SELF_TEST, "cuda", use_agentic=False, binary_path="/tmp/k",
        )
    mock_ck.assert_called_once()
    mock_llm.assert_not_called()


def test_check_correctness_python_kernel_uses_agentic_path():
    with patch("kernelopt.agents.correctness.run_correctness_agent", return_value=(True, "ok")) as mock_agent:
        passed, msg = check_correctness(
            "/tmp/softmax.py", "@triton.jit\ndef k(): pass", "triton", use_agentic=True,
        )
    mock_agent.assert_called_once()
    assert passed is True


# ---------------------------------------------------------------------------
# TestCudaAgenticCorrectness: matmul_cuda.cu end-to-end (mocked LLM)
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "kernels"
MATMUL_CU = FIXTURE_DIR / "matmul_cuda.cu"


class TestCudaAgenticCorrectness:
    """Exercise the _check_cuda_with_llm_reference path on a real kernel-only fixture."""

    def _make_passing_harness_llm(self):
        """Mock LLM that returns a harness which exits 0."""
        from kernelopt.agents.schemas import CudaCorrectnessSpec
        spec = CudaCorrectnessSpec(
            test_harness_code="#include<stdio.h>\nint main(){return 0;}",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = spec
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        return mock_llm

    def _make_failing_harness_llm(self):
        """Mock LLM that returns a harness which exits 1 with diagnostic output."""
        from kernelopt.agents.schemas import CudaCorrectnessSpec
        spec = CudaCorrectnessSpec(
            test_harness_code=(
                "#include<stdio.h>\nint main(){"
                "printf(\"index=0 expected=1.000000 got=2.000000 maxdiff=1.000000\\n\");"
                "return 1;}"
            ),
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = spec
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        return mock_llm

    def test_fixture_has_no_main(self):
        """matmul_cuda.cu must not contain int main() — otherwise it bypasses the agentic path."""
        source = MATMUL_CU.read_text()
        assert _has_builtin_self_test(source) is False

    def test_correct_kernel_passes_with_mocked_llm(self):
        """A correct kernel + LLM harness that exits 0 → (True, ...)."""
        source = MATMUL_CU.read_text()
        compile_ok = MagicMock(returncode=0, stdout="", stderr="")
        run_ok = MagicMock(returncode=0, stdout="", stderr="")

        with patch("kernelopt.functions.correctness.build_llm", return_value=self._make_passing_harness_llm()), \
             patch("subprocess.run", side_effect=[compile_ok, run_ok]), \
             patch("pathlib.Path.unlink"):
            passed, msg = check_correctness(
                str(MATMUL_CU), source, "cuda",
                use_agentic=True, model="unused", project_id=None, region=None,
            )

        assert passed is True, f"Expected pass, got: {msg}"

    def test_broken_kernel_fails_with_mocked_llm(self):
        """LLM harness that exits 1 (numerical mismatch) → (False, ...)."""
        source = MATMUL_CU.read_text()
        compile_ok = MagicMock(returncode=0, stdout="", stderr="")
        run_fail = MagicMock(
            returncode=1,
            stdout="index=0 expected=1.000000 got=2.000000 maxdiff=1.000000\n",
            stderr="",
        )

        with patch("kernelopt.functions.correctness.build_llm", return_value=self._make_failing_harness_llm()), \
             patch("subprocess.run", side_effect=[compile_ok, run_fail]), \
             patch("pathlib.Path.unlink"):
            passed, msg = check_correctness(
                str(MATMUL_CU), source, "cuda",
                use_agentic=True, model="unused", project_id=None, region=None,
            )

        assert passed is False
        assert "exit 1" in msg or "failed" in msg.lower()
