"""Tests for codegen strategy extraction via regex parsing."""

from __future__ import annotations

import pytest

from kernelopt.compiler.extraction.codegen_capture import extract_codegen_strategies


def _make_kernel(name: str, body: str) -> str:
    return f"{name} = async_compile.triton('{name}', '''\n{body}\n''')\n"


class TestCodegenCapture:
    def test_pointwise_detection(self):
        code = _make_kernel("triton_poi_fused_relu_0", """
@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    num_warps = 4
    num_stages = 1
""")
        strats = extract_codegen_strategies(code)
        assert len(strats) == 1
        assert strats[0].kernel_category == "pointwise"
        assert strats[0].inductor_num_warps == 4

    def test_persistent_reduction_detection(self):
        code = _make_kernel("triton_per_fused_sum_0", """
@triton.jit
def triton_per_fused_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 1024
    reduction_hint = ReductionHint.INNER
    num_warps = 8
""")
        strats = extract_codegen_strategies(code)
        assert strats[0].kernel_category == "persistent_reduction"
        assert strats[0].persistent_reduction
        assert strats[0].inductor_block_sizes.get("RBLOCK") == 1024
        assert strats[0].reduction_hint == "INNER"

    def test_cooperative_reduction_detection(self):
        code = _make_kernel("triton_red_fused_norm_0", """
@triton.jit
def triton_red_fused_norm_0(in_ptr0, XBLOCK: tl.constexpr, RSPLIT: tl.constexpr):
    rsplit_start = tl.program_id(1)
    num_warps = 8
""")
        strats = extract_codegen_strategies(code)
        assert strats[0].kernel_category == "cooperative_reduction"
        assert strats[0].cooperative_reduction

    def test_template_with_block_pointers(self):
        code = _make_kernel("triton_tem_fused_mm_0", """
@triton.jit
def triton_tem_fused_mm_0(A, B, C):
    tl.make_block_ptr(A, shape=(M, K))
    "BLOCK_M": 128
    "BLOCK_N": 256
    "BLOCK_K": 64
    num_warps = 4
    num_stages = 3
""")
        strats = extract_codegen_strategies(code)
        assert strats[0].kernel_category == "template"
        assert strats[0].uses_block_pointers
        assert strats[0].inductor_block_sizes == {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}
        assert strats[0].inductor_num_stages == 3

    def test_no_xblock_detection(self):
        code = _make_kernel("triton_poi_fused_0", """
@triton.jit
def triton_poi_fused_0(in_ptr0, rnumel):
    num_warps = 2
""")
        strats = extract_codegen_strategies(code)
        assert strats[0].no_x_dim

    def test_xblock_constexpr_one(self):
        code = _make_kernel("triton_poi_fused_0", """
@triton.jit
def triton_poi_fused_0(in_ptr0, XBLOCK: tl.constexpr = 1):
    pass
""")
        strats = extract_codegen_strategies(code)
        assert strats[0].no_x_dim

    def test_multiple_kernels(self):
        code = (
            _make_kernel("triton_poi_fused_relu_0", "num_warps = 4\nXBLOCK = 256") +
            _make_kernel("triton_per_fused_sum_1", "num_warps = 8\nRBLOCK: tl.constexpr = 512") +
            _make_kernel("triton_tem_fused_mm_2", "num_warps = 4")
        )
        strats = extract_codegen_strategies(code)
        assert len(strats) == 3
        categories = [s.kernel_category for s in strats]
        assert "pointwise" in categories
        assert "persistent_reduction" in categories
        assert "template" in categories

    def test_empty_input(self):
        assert extract_codegen_strategies("") == []
        assert extract_codegen_strategies("no triton here") == []

    def test_foreach_detection(self):
        code = _make_kernel("triton_for_fused_0", "num_warps = 4\nXBLOCK = 128")
        strats = extract_codegen_strategies(code)
        assert strats[0].kernel_category == "foreach"


class TestCodegenCaptureRendering:
    def test_rendering_from_strategies(self):
        from kernelopt.compiler.rendering import build_compiler_context_section
        from kernelopt.compiler.context import CompilerContext, CodegenStrategy

        ctx = CompilerContext(
            extraction_success=True,
            codegen_strategies=[CodegenStrategy(
                kernel_name="triton_per_fused_sum_0",
                kernel_category="persistent_reduction",
                persistent_reduction=True,
                inductor_num_warps=8,
                inductor_block_sizes={"RBLOCK": 1024},
                reduction_hint="INNER",
            )],
        )
        section = build_compiler_context_section(ctx.to_json(), "triton_per_fused_sum_0")
        assert "persistent" in section
        assert "RBLOCK" in section
        assert "INNER" in section
        assert "{" not in section  # no JSON in output
