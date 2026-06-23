"""Tests for the layered defense against Triton kernel hallucination."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))

from kernelopt.functions.correctness import validate_kernel_runs


class TestValidateKernelRuns:
    """Layer 1: Dry-run validation gate."""

    def test_valid_kernel_passes(self, tmp_path):
        kernel = tmp_path / "good.py"
        kernel.write_text(textwrap.dedent("""\
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                y = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x + y, mask=mask)

            def kernel_function(*inputs):
                x, y = inputs
                out = torch.empty_like(x)
                n = x.numel()
                grid = (triton.cdiv(n, 1024),)
                add_kernel[grid](x, y, out, n, BLOCK=1024)
                return out

            def get_inputs():
                return [torch.randn(1024, device='cuda'), torch.randn(1024, device='cuda')]

            if __name__ == "__main__":
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--profile-only", action="store_true")
                args = parser.parse_args()
                inputs = get_inputs()
                kernel_function(*inputs)
                torch.cuda.synchronize()
        """))
        ok, err = validate_kernel_runs(str(kernel))
        assert ok is True
        assert err == ""

    def test_missing_import_fails(self, tmp_path):
        kernel = tmp_path / "bad_import.py"
        kernel.write_text(textwrap.dedent("""\
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def k(x_ptr, n, BLOCK: tl.constexpr):
                pass

            def kernel_function(*inputs):
                x = inputs[0]
                x = copy_if_misaligned(x)  # undefined!
                return x

            def get_inputs():
                return [torch.randn(64, device='cuda')]

            if __name__ == "__main__":
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--profile-only", action="store_true")
                args = parser.parse_args()
                kernel_function(*get_inputs())
                torch.cuda.synchronize()
        """))
        ok, err = validate_kernel_runs(str(kernel))
        assert ok is False
        assert "copy_if_misaligned" in err

    def test_syntax_error_fails(self, tmp_path):
        kernel = tmp_path / "bad_syntax.py"
        kernel.write_text("def broken(\n  pass\n")
        ok, err = validate_kernel_runs(str(kernel))
        assert ok is False
        assert "SyntaxError" in err or "exit" in err

    def test_timeout_fails(self, tmp_path):
        kernel = tmp_path / "hang.py"
        kernel.write_text(textwrap.dedent("""\
            import time
            if __name__ == "__main__":
                time.sleep(999)
        """))
        ok, err = validate_kernel_runs(str(kernel), timeout=2)
        assert ok is False
        assert "timed out" in err


from kernelopt.agents.executor.triton_api_reference import build_triton_api_reference


class TestBuildTritonApiReference:
    """Layer 2: Triton API reference for executor prompt."""

    def test_returns_nonempty_string(self):
        ref = build_triton_api_reference()
        assert isinstance(ref, str)
        assert len(ref) > 200

    def test_contains_core_tl_functions(self):
        ref = build_triton_api_reference()
        for fn in ["load", "store", "program_id", "arange", "dot", "exp", "where", "maximum"]:
            assert fn in ref, f"Missing tl.{fn}"

    def test_contains_libdevice_functions(self):
        ref = build_triton_api_reference()
        for fn in ["sqrt", "exp", "tanh", "sin", "cos"]:
            assert fn in ref, f"Missing libdevice.{fn}"

    def test_contains_hardware_constraints(self):
        ref = build_triton_api_reference()
        assert "num_warps" in ref
        assert "num_stages" in ref
        assert "1024" in ref

    def test_contains_common_mistakes(self):
        ref = build_triton_api_reference()
        assert "triton_helpers" in ref
        assert "reinterpret_tensor" in ref

    def test_contains_triton_version(self):
        ref = build_triton_api_reference()
        import triton
        assert triton.__version__ in ref


from kernelopt.agents.executor.prompts import SYSTEM_PROMPT, build_initial_message


class TestExecutorPromptIntegration:
    """Layer 2+3: API reference in system prompt, context in initial message."""

    def test_system_prompt_contains_api_reference(self):
        assert "TRITON API REFERENCE" in SYSTEM_PROMPT
        assert "Valid tl." in SYSTEM_PROMPT
        assert "triton_helpers" in SYSTEM_PROMPT

    def test_initial_message_includes_profiling_summary(self):
        msg = build_initial_message(
            "increase_block_size", K=2,
            profiling_summary="GPU: H200\nBaseline: 5000 ns\nTop: MemoryBound 15%",
        )
        assert "H200" in msg.content
        assert "5000 ns" in msg.content

    def test_initial_message_without_profiling_summary(self):
        msg = build_initial_message("increase_block_size", K=2)
        assert "Hardware Context" not in msg.content

    def test_initial_message_includes_reference_snippets(self):
        msg = build_initial_message(
            "add_autotune", K=1,
            reference_snippets="@triton.autotune example here",
        )
        assert "autotune example" in msg.content

    def test_initial_message_without_reference_snippets(self):
        msg = build_initial_message("increase_block_size", K=2)
        assert "Reference" not in msg.content


class TestExecutorContextBuilder:
    """Layer 3: Profiling context preservation."""

    def test_build_executor_context_empty_state(self):
        from kernelopt.langgraph_pipeline import _build_executor_context
        result = _build_executor_context({})
        assert result == ""

    def test_build_executor_context_with_data(self):
        from kernelopt.langgraph_pipeline import _build_executor_context
        state = {
            "baseline_profiling_context": {
                "gpu_specs": {
                    "gpu_name": "NVIDIA H200",
                    "sm_count": 132,
                    "shared_memory_per_sm_kb": 228,
                },
                "launch": {
                    "block_size": 256,
                    "registers_per_thread": 32,
                    "waves_per_sm": 4,
                },
                "rules": [
                    {"rule_name": "MemoryBound", "speedup_pct": 25.0},
                    {"rule_name": "LowOccupancy", "speedup_pct": 10.0},
                    {"rule_name": "RegisterSpill", "speedup_pct": 5.0},
                    {"rule_name": "SmallBottleneck", "speedup_pct": 1.0},
                ],
            },
            "baseline_latency_ns": 5000,
        }
        result = _build_executor_context(state)
        assert "H200" in result
        assert "132 SMs" in result
        assert "5000 ns" in result
        assert "MemoryBound" in result
        assert "25.0%" in result
        # Only top 3 rules
        assert "SmallBottleneck" not in result


class TestReferenceSnippets:
    """Layer 2: Reference kernel injection."""

    def test_matmul_plan_gets_reference(self):
        from kernelopt.langgraph_pipeline import _build_reference_snippets
        plan = {"direction": "optimize_matmul_tile_sizes"}
        result = _build_reference_snippets(plan)
        assert result is not None
        assert "matmul" in result.lower()

    def test_softmax_plan_gets_reference(self):
        from kernelopt.langgraph_pipeline import _build_reference_snippets
        plan = {"direction": "fuse_softmax_with_mask"}
        result = _build_reference_snippets(plan)
        if result:
            assert "softmax" in result.lower()

    def test_unrelated_plan_gets_none(self):
        from kernelopt.langgraph_pipeline import _build_reference_snippets
        plan = {"direction": "increase_xblock_num_warps"}
        result = _build_reference_snippets(plan)
        assert result is None
