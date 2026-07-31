"""Tests for kernelopt.compiler data model and serialization."""

from __future__ import annotations

import json

import pytest

from kernelopt.compiler.context import (
    AOTDecompInfo,
    CodegenStrategy,
    CompilerContext,
    DynamoGraphInfo,
    DynamoOpInfo,
    FusionGroup,
    FusionRejection,
    GraphBreakInfo,
    IRNodeInfo,
    IntermediateBuffer,
    ReductionInfo,
    TritonCompilationInfo,
    XLAContext,
    XLAFusionDecision,
)


class TestCompilerContextConstruction:
    def test_empty_context(self):
        ctx = CompilerContext()
        assert ctx.extraction_success is False
        assert ctx.ir_nodes == []
        assert ctx.dynamo is None

    def test_full_context(self):
        ctx = CompilerContext(
            pytorch_version="2.14.0",
            extraction_success=True,
            dynamo=DynamoGraphInfo(
                original_ops=(DynamoOpInfo("layer_norm", ((4, 64),), (4, 64), True),),
                num_subgraphs=1,
            ),
            aot_decomps=[AOTDecompInfo("layer_norm", ("var_mean", "rsqrt", "sub", "mul", "add"), 4)],
            ir_nodes=[IRNodeInfo(
                buffer_name="buf0", node_type="Pointwise",
                iteration_ranges=(4, 64), total_elements=256,
                dtype="torch.float32", bytes_per_element=4,
                total_memory_bytes=1024,
            )],
            codegen_strategies=[CodegenStrategy(
                kernel_name="triton_per_fused_0", kernel_category="persistent_reduction",
                persistent_reduction=True, no_x_dim=True,
            )],
            triton_info=[TritonCompilationInfo(
                kernel_name="triton_per_fused_0", num_warps=4, n_regs=32, shared_bytes=8,
            )],
        )
        assert ctx.extraction_success
        assert len(ctx.ir_nodes) == 1
        assert ctx.dynamo.original_ops[0].is_compound


class TestCompilerContextJSON:
    def test_roundtrip_empty(self):
        ctx = CompilerContext(pytorch_version="2.14.0", extraction_success=True)
        restored = CompilerContext.from_json(ctx.to_json())
        assert restored.pytorch_version == "2.14.0"
        assert restored.extraction_success

    def test_roundtrip_with_data(self):
        ctx = CompilerContext(
            pytorch_version="2.14.0",
            extraction_success=True,
            ir_nodes=[IRNodeInfo(
                buffer_name="buf0", node_type="Reduction",
                iteration_ranges=(8192,), total_elements=8192,
                dtype="torch.float32", bytes_per_element=4,
                total_memory_bytes=32768,
                reduction=ReductionInfo(
                    reduction_type="sum", reduction_ranges=(1024,),
                    reduction_hint="INNER", src_dtype="torch.float32",
                    dst_dtype="torch.float32", rnumel_padded=1024,
                    wasted_sram_fraction=0.0,
                ),
            )],
            fusion_rejections=[FusionRejection(
                node1_name="buf0", node2_name="buf1",
                reason="score below threshold",
                near_threshold=True, fixable_by_manual_fusion=True,
            )],
        )
        json_str = ctx.to_json()
        restored = CompilerContext.from_json(json_str)
        assert restored.ir_nodes[0].reduction.reduction_type == "sum"
        assert restored.ir_nodes[0].reduction.rnumel_padded == 1024
        assert restored.fusion_rejections[0].near_threshold

    def test_roundtrip_with_xla(self):
        ctx = CompilerContext(
            pytorch_version="2.14.0",
            extraction_success=True,
            xla_context=XLAContext(
                stablehlo_text="func @main(%arg0: tensor<4xf32>)",
                cost_analysis={"flops": 1024.0},
                fusion_decisions=[XLAFusionDecision(
                    step_type="Fusion", producer_name="p", consumer_name="c",
                )],
            ),
        )
        restored = CompilerContext.from_json(ctx.to_json())
        assert restored.xla_context.stablehlo_text.startswith("func")
        assert restored.xla_context.cost_analysis["flops"] == 1024.0

    def test_json_is_valid(self):
        ctx = CompilerContext(pytorch_version="2.14.0", extraction_success=True)
        parsed = json.loads(ctx.to_json())
        assert parsed["pytorch_version"] == "2.14.0"


class TestCompilerContextQueries:
    @pytest.fixture
    def ctx(self):
        return CompilerContext(
            pytorch_version="2.14.0",
            extraction_success=True,
            fusion_groups=[FusionGroup(
                kernel_name="triton_poi_fused_relu_0",
                constituent_buffers=("buf0", "buf1"),
            )],
            ir_nodes=[
                IRNodeInfo(buffer_name="buf0", node_type="Pointwise",
                           iteration_ranges=(256,), total_elements=256,
                           dtype="torch.float32", bytes_per_element=4,
                           total_memory_bytes=1024),
                IRNodeInfo(buffer_name="buf1", node_type="Pointwise",
                           iteration_ranges=(256,), total_elements=256,
                           dtype="torch.float32", bytes_per_element=4,
                           total_memory_bytes=1024),
                IRNodeInfo(buffer_name="buf2", node_type="Reduction",
                           iteration_ranges=(16,), total_elements=16,
                           dtype="torch.float32", bytes_per_element=4,
                           total_memory_bytes=64),
            ],
            codegen_strategies=[CodegenStrategy(
                kernel_name="triton_poi_fused_relu_0",
                kernel_category="pointwise",
                no_x_dim=True, uses_block_pointers=True,
            )],
            fusion_rejections=[
                FusionRejection("a", "b", "score below threshold",
                                near_threshold=True, fixable_by_manual_fusion=True),
                FusionRejection("c", "d", "will create cycle",
                                near_threshold=False, fixable_by_manual_fusion=False),
            ],
            triton_info=[TritonCompilationInfo(
                kernel_name="triton_poi_fused_relu_0",
                n_regs=200, n_spills=5,
            )],
        )

    def test_get_kernel_ir(self, ctx):
        nodes = ctx.get_kernel_ir("triton_poi_fused_relu_0")
        assert len(nodes) == 2
        assert all(n.buffer_name in ("buf0", "buf1") for n in nodes)

    def test_get_kernel_ir_missing(self, ctx):
        assert ctx.get_kernel_ir("nonexistent") == []

    def test_get_codegen_strategy(self, ctx):
        strat = ctx.get_codegen_strategy("triton_poi_fused_relu_0")
        assert strat.no_x_dim
        assert strat.uses_block_pointers

    def test_get_structural_constraints(self, ctx):
        constraints = ctx.get_structural_constraints("triton_poi_fused_relu_0")
        assert any("XBLOCK" in c for c in constraints)
        assert any("block_ptr" in c for c in constraints)
        assert any("register" in c.lower() for c in constraints)
        assert any("spill" in c.lower() for c in constraints)

    def test_get_fusion_opportunities(self, ctx):
        opps = ctx.get_fusion_opportunities()
        assert len(opps) == 1
        assert opps[0].reason == "score below threshold"

    def test_get_triton_info(self, ctx):
        tri = ctx.get_triton_info("triton_poi_fused_relu_0")
        assert tri.n_regs == 200
        assert tri.n_spills == 5
