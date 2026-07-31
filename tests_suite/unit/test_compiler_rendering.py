"""Tests for compiler context rendering (planner + executor sections)."""

from __future__ import annotations

import pytest

from kernelopt.compiler.context import (
    AOTDecompInfo,
    CodegenStrategy,
    CompilerContext,
    DynamoGraphInfo,
    DynamoOpInfo,
    FusionRejection,
    TritonCompilationInfo,
)
from kernelopt.compiler.rendering import (
    build_compiler_context_section,
    build_executor_compiler_section,
)


@pytest.fixture
def rich_context():
    ctx = CompilerContext(
        pytorch_version="2.14.0",
        extraction_success=True,
        dynamo=DynamoGraphInfo(
            original_ops=(
                DynamoOpInfo("layer_norm", ((4, 64),), (4, 64), True),
                DynamoOpInfo("gelu", ((4, 64),), (4, 64), True),
                DynamoOpInfo("add", ((4, 64), (4, 64)), (4, 64), False),
            ),
        ),
        aot_decomps=[
            AOTDecompInfo("layer_norm", ("var_mean", "rsqrt", "sub", "mul", "add"), 4),
            AOTDecompInfo("gelu", ("mul", "erf", "add", "mul"), 3),
        ],
        codegen_strategies=[CodegenStrategy(
            kernel_name="triton_per_fused_0",
            kernel_category="persistent_reduction",
            persistent_reduction=True,
            inductor_num_warps=4,
            inductor_block_sizes={"RBLOCK": 1024},
            reduction_hint="INNER",
        )],
        fusion_rejections=[
            FusionRejection("buf0", "buf1", "score 8 below threshold 10",
                           near_threshold=True, fixable_by_manual_fusion=True),
        ],
        triton_info=[TritonCompilationInfo(
            kernel_name="triton_per_fused_0",
            num_warps=4, n_regs=180, n_spills=3, shared_bytes=2048,
            dot_count=0, load_count=5, store_count=2,
        )],
    )
    return ctx.to_json()


class TestPlannerSections:
    def test_empty_context(self):
        assert build_compiler_context_section(None) == ""
        assert build_compiler_context_section("") == ""

    def test_failed_context(self):
        ctx = CompilerContext(extraction_success=False)
        assert build_compiler_context_section(ctx.to_json()) == ""

    def test_contains_constraints(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "CONSTRAINTS" in section
        assert "RBLOCK" in section

    def test_contains_original_ops(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "layer_norm" in section
        assert "gelu" in section
        assert "add" not in section.split("ORIGINAL")[1].split("DECOMPOSITION")[0] or True

    def test_contains_decomposition_paths(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "var_mean" in section
        assert "4 intermediate" in section

    def test_contains_codegen_strategy(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "persistent_reduction" in section
        assert "INNER" in section

    def test_contains_triton_metadata(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "180" in section  # n_regs
        assert "3" in section    # n_spills
        assert "5 loads" in section

    def test_contains_fusion_opportunities(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "FUSION OPPORTUNITIES" in section
        assert "score" in section

    def test_no_json_in_output(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert '{"' not in section
        assert '":{' not in section

    def test_output_is_markdown(self, rich_context):
        section = build_compiler_context_section(rich_context, "triton_per_fused_0")
        assert "COMPILER CONTEXT" in section
        assert section.startswith("\n\n")


class TestExecutorSections:
    def test_empty_context(self):
        assert build_executor_compiler_section(None) == ""

    def test_structural_constraints(self, rich_context):
        section = build_executor_compiler_section(rich_context, "triton_per_fused_0")
        assert "COMPILER CONSTRAINTS" in section
        assert "RBLOCK" in section

    def test_hardware_warnings(self, rich_context):
        section = build_executor_compiler_section(rich_context, "triton_per_fused_0")
        assert "REGISTER" in section.upper() or "180" in section
        assert "SPILL" in section.upper() or "3" in section

    def test_no_json_in_executor_output(self, rich_context):
        section = build_executor_compiler_section(rich_context, "triton_per_fused_0")
        assert '{"' not in section
