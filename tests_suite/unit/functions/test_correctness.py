"""
Unit tests for correctness checking functions.

Tests cover:
- Correctness validation subprocess execution
- Kernel function extraction
- Framework-specific correctness checks
- Error handling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestCorrectness:
    """Test suite for kernel correctness checking."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_check_correctness_subprocess_valid(self, test_kernel_path, skip_if_no_cuda):
        """Test correctness check with valid kernel."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        kernel_source = test_kernel_path.read_text()
        passed, msg = check_correctness_subprocess(
            str(test_kernel_path), kernel_source
        )

        # Same kernel should pass correctness
        assert passed is True
        assert msg != ""


class TestSelector:
    """Test suite for plan selection logic."""

    def test_select_best_candidate_by_latency(self):
        """Test selecting best candidate by latency."""
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(
                plan_letter="A",
                direction="opt_a",
                latency_ns=100.0,
                speedup=1.5,
            ),
            PlanCandidate(
                plan_letter="B",
                direction="opt_b",
                latency_ns=80.0,
                speedup=1.8,
            ),
            PlanCandidate(
                plan_letter="C",
                direction="opt_c",
                latency_ns=120.0,
                speedup=1.2,
            ),
        ]

        best = min(candidates, key=lambda c: c.latency_ns)
        assert best.plan_letter == "B"
        assert best.latency_ns == 80.0

    def test_select_best_candidate_correctness(self):
        """Test that only correct candidates are considered."""
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(
                plan_letter="A",
                direction="opt_a",
                latency_ns=80.0,
                correctness_passed=False,
            ),
            PlanCandidate(
                plan_letter="B",
                direction="opt_b",
                latency_ns=100.0,
                correctness_passed=True,
            ),
        ]

        valid = [c for c in candidates if c.correctness_passed]
        assert len(valid) == 1
        assert valid[0].plan_letter == "B"

    def test_select_candidates_by_speedup(self):
        """Test selecting candidates sorted by speedup."""
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_letter="A", speedup=1.2, correctness_passed=True),
            PlanCandidate(plan_letter="B", speedup=1.8, correctness_passed=True),
            PlanCandidate(plan_letter="C", speedup=1.1, correctness_passed=True),
        ]

        sorted_candidates = sorted(
            [c for c in candidates if c.correctness_passed],
            key=lambda c: c.speedup,
            reverse=True
        )

        assert sorted_candidates[0].plan_letter == "B"
        assert sorted_candidates[0].speedup == 1.8


class TestCorrectnessAgainstPytorch:
    """Test check_correctness_against_pytorch with standalone kernel files."""

    BASELINE_STANDALONE = """\
import torch
import triton
import triton.language as tl

@triton.jit
def triton_add_bias_relu(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex
    tmp0 = tl.load(in_out_ptr0 + x1, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.maximum(0, tmp2)
    tl.store(in_out_ptr0 + x1, tmp3, xmask)

def kernel_function(*inputs):
    in_out_ptr0 = inputs[0]
    in_ptr0 = inputs[1]
    xnumel = in_out_ptr0.numel()
    grid = (triton.cdiv(xnumel, 1024),)
    triton_add_bias_relu[grid](in_out_ptr0, in_ptr0, xnumel, XBLOCK=1024)
    return in_out_ptr0

def get_inputs():
    in_out_ptr0 = torch.randn((32, 64), device='cuda:0', dtype=torch.float32)
    in_ptr0 = torch.randn((64,), device='cuda:0', dtype=torch.float32)
    return [in_out_ptr0, in_ptr0]

if __name__ == "__main__":
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
"""

    OPTIMIZED_CORRECT = """\
import torch
import triton
import triton.language as tl

@triton.jit
def triton_add_bias_relu(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex
    tmp0 = tl.load(in_out_ptr0 + x1, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.maximum(0, tmp2)
    tl.store(in_out_ptr0 + x1, tmp3, xmask)

def kernel_function(*inputs):
    in_out_ptr0 = inputs[0]
    in_ptr0 = inputs[1]
    xnumel = in_out_ptr0.numel()
    grid = (triton.cdiv(xnumel, 1024),)
    triton_add_bias_relu[grid](in_out_ptr0, in_ptr0, xnumel, XBLOCK=1024)
    return in_out_ptr0

def get_inputs():
    in_out_ptr0 = torch.randn((32, 64), device='cuda:0', dtype=torch.float32)
    in_ptr0 = torch.randn((64,), device='cuda:0', dtype=torch.float32)
    return [in_out_ptr0, in_ptr0]

if __name__ == "__main__":
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
"""

    OPTIMIZED_WRONG = """\
import torch
import triton
import triton.language as tl

@triton.jit
def triton_add_bias_relu(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex
    # Bug: zeroes out the output instead of computing correctly
    tl.store(in_out_ptr0 + x1, tl.full([XBLOCK], 0.0, tl.float32), xmask)

def kernel_function(*inputs):
    in_out_ptr0 = inputs[0]
    in_ptr0 = inputs[1]
    xnumel = in_out_ptr0.numel()
    grid = (triton.cdiv(xnumel, 1024),)
    triton_add_bias_relu[grid](in_out_ptr0, in_ptr0, xnumel, XBLOCK=1024)
    return in_out_ptr0

def get_inputs():
    in_out_ptr0 = torch.randn((32, 64), device='cuda:0', dtype=torch.float32)
    in_ptr0 = torch.randn((64,), device='cuda:0', dtype=torch.float32)
    return [in_out_ptr0, in_ptr0]

if __name__ == "__main__":
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
"""

    @pytest.mark.gpu
    def test_same_kernel_passes(self, tmp_path):
        """Identical baseline and optimized should pass correctness."""
        ref_path = tmp_path / "baseline_standalone.py"
        ref_path.write_text(self.BASELINE_STANDALONE)

        from kernelopt.functions.correctness import check_correctness_against_pytorch
        passed, msg = check_correctness_against_pytorch(
            original_pytorch_path=str(ref_path),
            optimized_source=self.OPTIMIZED_CORRECT,
            optimized_kernel_path=str(tmp_path / "optimized.py"),
        )
        assert passed is True, f"Expected pass: {msg}"

    @pytest.mark.gpu
    def test_wrong_kernel_fails(self, tmp_path):
        """Optimized kernel that zeroes output should fail correctness."""
        ref_path = tmp_path / "baseline_standalone.py"
        ref_path.write_text(self.BASELINE_STANDALONE)

        from kernelopt.functions.correctness import check_correctness_against_pytorch
        passed, msg = check_correctness_against_pytorch(
            original_pytorch_path=str(ref_path),
            optimized_source=self.OPTIMIZED_WRONG,
            optimized_kernel_path=str(tmp_path / "optimized.py"),
        )
        assert passed is False, f"Expected fail but got pass: {msg}"
