"""Tests for JAX/XLA extraction modules.

Tests the XLA-layer extraction: stages API, fusion dump parsing,
autotune result parsing, Pallas bridge, and XLA rendering sections.
All tests are self-contained — JAX is NOT required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernelopt.compiler.context import (
    CompilerContext,
    XLAAutotuneResult,
    XLAContext,
    XLAFusionDecision,
)


# =========================================================================
# Fusion dump parsing
# =========================================================================

class TestFusionDumpParsing:
    def test_empty_dir(self, tmp_path):
        from kernelopt.compiler.extraction.xla.fusion_dump import parse_fusion_dump_text
        assert parse_fusion_dump_text(str(tmp_path)) == []

    def test_nonexistent_dir(self):
        from kernelopt.compiler.extraction.xla.fusion_dump import parse_fusion_dump_text
        assert parse_fusion_dump_text("/nonexistent/dir") == []

    def test_parse_fusion_text(self, tmp_path):
        from kernelopt.compiler.extraction.xla.fusion_dump import parse_fusion_dump_text

        dump_file = tmp_path / "module.fusion_process_dump.txt"
        dump_file.write_text(
            "Fusing producer reduce_sum into consumer add\n"
            "Cannot fuse broadcast: ineligible\n"
        )
        results = parse_fusion_dump_text(str(tmp_path))
        assert len(results) == 2
        assert results[0]["step_type"] == "Fusion"
        assert results[1]["step_type"] == "ProducerIneligible"

    def test_fallback_to_hlo_text(self, tmp_path):
        from kernelopt.compiler.extraction.xla.fusion_dump import parse_fusion_dump_text

        hlo_file = tmp_path / "module.after_optimizations.txt"
        hlo_file.write_text(
            "ENTRY main {\n"
            "  %fused_reduce(p0, p1)\n"
            "  %fused_add(p2)\n"
            "}\n"
        )
        results = parse_fusion_dump_text(str(tmp_path))
        assert len(results) == 2
        assert all(r["step_type"] == "Fusion" for r in results)
        assert "fused_reduce" in results[0]["producer_name"]

    def test_setup_xla_dump_flags(self):
        from kernelopt.compiler.extraction.xla.fusion_dump import setup_xla_dump_flags

        old = os.environ.get("XLA_FLAGS", "")
        try:
            os.environ["XLA_FLAGS"] = ""
            setup_xla_dump_flags("/tmp/xla_dump")
            assert "--xla_dump_to=/tmp/xla_dump" in os.environ["XLA_FLAGS"]
            assert "--xla_dump_hlo_as_text" in os.environ["XLA_FLAGS"]
        finally:
            os.environ["XLA_FLAGS"] = old


# =========================================================================
# Autotune parsing
# =========================================================================

class TestAutotuneResultParsing:
    def test_nonexistent_file(self):
        from kernelopt.compiler.extraction.xla.autotune import parse_autotune_results
        assert parse_autotune_results("/nonexistent/file.txt") == []

    def test_parse_text_proto(self, tmp_path):
        from kernelopt.compiler.extraction.xla.autotune import parse_autotune_results

        proto_file = tmp_path / "autotune.textproto"
        proto_file.write_text(
            'results {\n'
            '  device: "gpu:0"\n'
            '  hlo: "fused_dot_general.1"\n'
            '  result {\n'
            '    triton {\n'
            '      block_m: 128\n'
            '      block_n: 128\n'
            '      block_k: 32\n'
            '      num_stages: 3\n'
            '      num_warps: 4\n'
            '    }\n'
            '    run_time {\n'
            '      nanos: 5000\n'
            '    }\n'
            '  }\n'
            '}\n'
        )
        results = parse_autotune_results(str(proto_file))
        assert len(results) == 1
        assert results[0]["fusion_name"] == "1"
        assert results[0]["block_m"] == 128
        assert results[0]["block_n"] == 128
        assert results[0]["block_k"] == 32
        assert results[0]["num_stages"] == 3
        assert results[0]["num_warps"] == 4
        assert results[0]["run_time_us"] == 5.0

    def test_parse_multiple_results(self, tmp_path):
        from kernelopt.compiler.extraction.xla.autotune import parse_autotune_results

        proto_file = tmp_path / "results.txt"
        proto_file.write_text(
            'results {\n  device: "gpu:0"\n  hlo: "fusion1"\n'
            '  result {\n    triton {\n      block_m: 64\n    }\n  }\n}\n'
            'results {\n  device: "gpu:0"\n  hlo: "fusion2"\n'
            '  result {\n    triton {\n      block_m: 256\n    }\n  }\n}\n'
        )
        results = parse_autotune_results(str(proto_file))
        assert len(results) == 2
        assert results[0]["block_m"] == 64
        assert results[1]["block_m"] == 256

    def test_parse_per_fusion_cache(self, tmp_path):
        from kernelopt.compiler.extraction.xla.autotune import parse_per_fusion_cache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        f1 = cache_dir / "fused_reduce"
        f1.write_text(
            'results {\n  device: "gpu:0"\n  hlo: "fused_reduce"\n'
            '  result {\n    triton {\n      block_m: 32\n    }\n  }\n}\n'
        )
        results = parse_per_fusion_cache(str(cache_dir))
        assert len(results) == 1
        assert results[0]["block_m"] == 32

    def test_empty_cache_dir(self, tmp_path):
        from kernelopt.compiler.extraction.xla.autotune import parse_per_fusion_cache
        assert parse_per_fusion_cache(str(tmp_path)) == []

    def test_nonexistent_cache_dir(self):
        from kernelopt.compiler.extraction.xla.autotune import parse_per_fusion_cache
        assert parse_per_fusion_cache("/nonexistent/dir") == []

    def test_setup_autotune_dump_flags(self):
        from kernelopt.compiler.extraction.xla.autotune import setup_autotune_dump_flags

        old = os.environ.get("XLA_FLAGS", "")
        try:
            os.environ["XLA_FLAGS"] = ""
            setup_autotune_dump_flags("/tmp/autotune.pb")
            assert "--xla_gpu_dump_autotune_results_to=/tmp/autotune.pb" in os.environ["XLA_FLAGS"]
        finally:
            os.environ["XLA_FLAGS"] = old


# =========================================================================
# Stages extraction (with mocked JAX)
# =========================================================================

class TestStagesExtraction:
    def test_no_jax_returns_error(self):
        from kernelopt.compiler.extraction.xla.stages import extract_xla_stages
        with patch.dict("sys.modules", {"jax": None}):
            result = extract_xla_stages(lambda x: x)
            assert "error" in result

    def test_extract_jaxpr_no_jax(self):
        from kernelopt.compiler.extraction.xla.stages import extract_jaxpr
        with patch.dict("sys.modules", {"jax": None}):
            result = extract_jaxpr(lambda x: x)
            assert "error" in result


# =========================================================================
# Pallas bridge
# =========================================================================

class TestPallasBridge:
    def test_check_pallas_unavailable(self):
        from kernelopt.compiler.extraction.xla.pallas_bridge import check_pallas_available
        with patch.dict("sys.modules", {"jax": None, "jax.experimental": None}):
            assert not check_pallas_available()

    def test_create_wrapper_no_jax(self):
        from kernelopt.compiler.extraction.xla.pallas_bridge import create_pallas_kernel_wrapper
        with patch.dict("sys.modules", {"jax": None, "jax.experimental": None}):
            with pytest.raises((ImportError, RuntimeError)):
                wrapper = create_pallas_kernel_wrapper(
                    lambda: None, None, (1,)
                )
                wrapper()


# =========================================================================
# XLA data model
# =========================================================================

class TestXLADataModel:
    def test_xla_fusion_decision_creation(self):
        d = XLAFusionDecision(
            step_type="Fusion",
            producer_name="reduce_sum",
            consumer_name="add",
            reason="cost model accepted",
        )
        assert d.step_type == "Fusion"
        assert d.producer_name == "reduce_sum"

    def test_xla_autotune_result_creation(self):
        r = XLAAutotuneResult(
            fusion_name="fused_dot",
            block_m=128, block_n=128, block_k=32,
            num_stages=3, num_warps=4,
            run_time_us=5.0,
        )
        assert r.fusion_name == "fused_dot"
        assert r.block_m == 128
        assert r.run_time_us == 5.0

    def test_xla_context_creation(self):
        ctx = XLAContext(
            stablehlo_text="func.func @main",
            optimized_hlo_text="ENTRY main",
            cost_analysis={"flops": 1_000_000, "bytes accessed": 50_000},
            memory_analysis={"temp_size_bytes": 4096},
            fusion_decisions=[
                XLAFusionDecision("Fusion", "a", "b", "ok"),
                XLAFusionDecision("ProducerIneligible", "c", "", "too large"),
            ],
            autotune_results=[
                XLAAutotuneResult("fused_dot", 128, 128, 32, 3, 4, 5.0),
            ],
        )
        assert ctx.stablehlo_text.startswith("func")
        assert len(ctx.fusion_decisions) == 2
        assert len(ctx.autotune_results) == 1

    def test_compiler_context_with_xla(self):
        xla = XLAContext(
            stablehlo_text="test",
            cost_analysis={"flops": 1000},
        )
        ctx = CompilerContext(
            extraction_success=True,
            xla_context=xla,
        )
        assert ctx.xla_context is not None
        assert ctx.xla_context.cost_analysis["flops"] == 1000


# =========================================================================
# XLA rendering
# =========================================================================

class TestXLAPlannerRendering:
    @pytest.fixture
    def xla_context_json(self):
        xla = XLAContext(
            cost_analysis={"flops": 2_000_000, "bytes accessed": 100_000},
            memory_analysis={"temp_size_bytes": 8192},
            fusion_decisions=[
                XLAFusionDecision("Fusion", "a", "b", "ok"),
                XLAFusionDecision("ProducerIneligible", "c", "", "producer too large"),
            ],
            autotune_results=[
                XLAAutotuneResult("fused_dot", 128, 128, 32, 3, 4, 5.0),
            ],
        )
        ctx = CompilerContext(
            extraction_success=True,
            xla_context=xla,
        )
        return ctx.to_json()

    def test_xla_section_in_planner(self, xla_context_json):
        from kernelopt.compiler.rendering import build_compiler_context_section
        section = build_compiler_context_section(xla_context_json)
        assert "XLA COMPILATION CONTEXT" in section
        assert "FLOPs" in section
        assert "2,000,000" in section
        assert "20.0 FLOPs/byte" in section
        assert "Temp buffer" in section
        assert "rejected 1" in section
        assert "producer too large" in section
        assert "autotuned 1" in section
        assert "fused_dot" in section

    def test_xla_section_empty_when_no_xla(self):
        from kernelopt.compiler.rendering import build_compiler_context_section
        ctx = CompilerContext(extraction_success=True)
        section = build_compiler_context_section(ctx.to_json())
        assert "XLA" not in section

    def test_xla_executor_autotune_reference(self, xla_context_json):
        from kernelopt.compiler.rendering import build_executor_compiler_section
        section = build_executor_compiler_section(xla_context_json)
        assert "XLA AUTOTUNE REFERENCE" in section
        assert "block_m=128" in section
        assert "fused_dot" in section


class TestXLAExecutorRendering:
    def test_no_xla_no_section(self):
        from kernelopt.compiler.rendering import build_executor_compiler_section
        ctx = CompilerContext(extraction_success=True)
        section = build_executor_compiler_section(ctx.to_json())
        assert "XLA" not in section

    def test_xla_without_autotune(self):
        from kernelopt.compiler.rendering import build_executor_compiler_section
        xla = XLAContext(stablehlo_text="test")
        ctx = CompilerContext(extraction_success=True, xla_context=xla)
        section = build_executor_compiler_section(ctx.to_json())
        assert "XLA AUTOTUNE" not in section


# =========================================================================
# XLA JSON round-trip
# =========================================================================

class TestXLAJsonRoundTrip:
    def test_compiler_context_xla_serialization(self):
        xla = XLAContext(
            stablehlo_text="func.func @main",
            optimized_hlo_text="ENTRY main",
            cost_analysis={"flops": 500},
            fusion_decisions=[
                XLAFusionDecision("Fusion", "a", "b", "ok"),
            ],
            autotune_results=[
                XLAAutotuneResult("fused_dot", 64, 64, 16, 2, 4, 3.5),
            ],
        )
        ctx = CompilerContext(
            extraction_success=True,
            xla_context=xla,
        )
        json_str = ctx.to_json()

        restored = CompilerContext.from_json(json_str)
        assert restored.xla_context is not None
        assert restored.xla_context.stablehlo_text == "func.func @main"
        assert restored.xla_context.cost_analysis["flops"] == 500
        assert len(restored.xla_context.fusion_decisions) == 1
        assert restored.xla_context.fusion_decisions[0].step_type == "Fusion"
        assert len(restored.xla_context.autotune_results) == 1
        assert restored.xla_context.autotune_results[0].block_m == 64

    def test_compiler_context_no_xla_roundtrip(self):
        ctx = CompilerContext(extraction_success=True)
        json_str = ctx.to_json()
        restored = CompilerContext.from_json(json_str)
        assert restored.xla_context is None
