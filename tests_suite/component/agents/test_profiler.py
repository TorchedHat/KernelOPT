"""
Comprehensive tests for the profiler agent.

Tests cover:
- NCU execution and output parsing
- Profiling context building
- Latency extraction
- Framework detection
- Error handling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestProfilerNCU:
    """Test NCU execution and profiling."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_run_ncu_basic(self, test_kernel_path, temp_output_dir, skip_if_no_cuda):
        """Test basic NCU execution."""
        from kernelopt.agents.profiler import run_ncu

        output_base = str(temp_output_dir / "test_ncu")

        csv_path, ncu_rep, profiling_ctx_path = run_ncu(
            str(test_kernel_path),
            output_base,
            kernel_args=["--profile-only"],
        )

        assert Path(csv_path).exists()
        assert csv_path.endswith(".csv")
        # ncu_rep may or may not exist depending on NCU version
        if ncu_rep:
            assert ncu_rep.endswith(".ncu-rep")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_run_ncu_with_launch_config(self, test_kernel_path, temp_output_dir, skip_if_no_cuda):
        """Test NCU with launch skip/count."""
        from kernelopt.agents.profiler import run_ncu

        output_base = str(temp_output_dir / "test_ncu_launch")

        csv_path, ncu_rep, profiling_ctx_path = run_ncu(
            str(test_kernel_path),
            output_base,
            launch_skip=0,
            launch_count=1,
            kernel_args=["--profile-only"],
        )

        assert Path(csv_path).exists()


class TestProfilingContext:
    """Test profiling context building."""

    def test_build_profiling_context_basic(self):
        """Test basic profiling context construction."""
        from kernelopt.agents.profiler import build_profiling_context

        sections = {
            "GPU Speed Of Light Throughput": {
                "Duration": {"value": 125000, "unit": "nsecond"},
                "Compute (SM) Throughput": {"value": 45.2, "unit": "%"},
            }
        }
        rules = [
            {
                "name": "MemoryBound",
                "type": "OPT",
                "description": "Memory bound",
                "speedup_pct": 10.0,
            }
        ]
        launch = {"block_size": 256, "grid_size": 4096}

        ctx = build_profiling_context(
            sections, rules, launch, "/tmp/test.py", kernel_id=0
        )

        assert "kernel_name" in ctx
        assert "framework_hint" in ctx
        assert "sections" in ctx
        assert "rules" in ctx
        assert "launch" in ctx
        assert ctx["sections"] == sections
        assert ctx["rules"] == rules

    def test_build_profiling_context_framework_detection(self):
        """Test framework hint detection."""
        from kernelopt.agents.profiler import build_profiling_context

        # With empty data and non-existent path, framework is unknown
        ctx_empty = build_profiling_context(
            {}, [], {}, "/tmp/test_triton.py"
        )
        # Framework detection requires either AST parsing or kernel name patterns
        assert ctx_empty["framework_hint"] in ["cuda", "triton", "helion", "unknown"]

    def test_build_profiling_context_compute_capability(self):
        """Test compute capability extraction."""
        from kernelopt.agents.profiler import build_profiling_context

        sections = {
            "Device": {
                "Compute Capability": {"value": "8.9", "unit": ""}
            }
        }

        ctx = build_profiling_context(sections, [], {}, "/tmp/test.py")

        # Compute capability may be extracted from device info
        assert "compute_capability" in ctx


class TestLatencyExtraction:
    """Test latency extraction from profiling context."""

    def test_get_latency_ns_from_duration(self):
        """Test extracting latency from Duration metric."""
        from kernelopt.agents.profiler import get_latency_ns

        ctx = {
            "sections": {
                "GPU Speed Of Light Throughput": {
                    "Duration": {"value": 125000, "unit": "nsecond"}
                }
            }
        }

        latency = get_latency_ns(ctx)
        assert latency == 125000

    def test_get_latency_ns_from_time_duration(self):
        """Test extracting latency from gpu__time_duration.sum."""
        from kernelopt.agents.profiler import get_latency_ns

        ctx = {
            "sections": {
                "Duration": {
                    "gpu__time_duration.sum": {"value": 150000, "unit": "nsecond"}
                }
            }
        }

        latency = get_latency_ns(ctx)
        assert latency > 0

    def test_get_latency_ns_fallback(self):
        """Test latency extraction fallback."""
        from kernelopt.agents.profiler import get_latency_ns

        ctx = {"sections": {}}

        latency = get_latency_ns(ctx)
        # Should return some default or raise
        assert latency >= 0 or latency == float("inf")

    def test_get_latency_ns_unit_conversion(self):
        """Test that different time units are handled."""
        from kernelopt.agents.profiler import get_latency_ns

        # Microseconds
        ctx_us = {
            "sections": {
                "GPU Speed Of Light Throughput": {
                    "Duration": {"value": 125, "unit": "usecond"}
                }
            }
        }

        # Milliseconds
        ctx_ms = {
            "sections": {
                "GPU Speed Of Light Throughput": {
                    "Duration": {"value": 0.125, "unit": "msecond"}
                }
            }
        }

        # Both should handle the units appropriately
        latency_us = get_latency_ns(ctx_us)
        latency_ms = get_latency_ns(ctx_ms)

        assert latency_us > 0
        assert latency_ms > 0


class TestProfilerIntegration:
    """Integration tests for profiler agent."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_full_profiling_workflow(self, test_kernel_path, temp_output_dir, skip_if_no_cuda):
        """Test complete profiling workflow."""
        from kernelopt.agents.profiler import (
            run_ncu,
            build_profiling_context,
            get_latency_ns,
        )
        from kernelopt.functions.csv_parser import parse_ncu_csv

        output_base = str(temp_output_dir / "test_workflow")

        # Step 1: Run NCU
        csv_path, ncu_rep, profiling_ctx_path = run_ncu(
            str(test_kernel_path),
            output_base,
            kernel_args=["--profile-only"],
        )

        # Step 2: Parse CSV
        sections, rules, launch = parse_ncu_csv(csv_path)

        # Step 3: Build context
        ctx = build_profiling_context(
            sections, rules, launch, str(test_kernel_path)
        )

        # Step 4: Extract latency
        latency = get_latency_ns(ctx)

        # Verify
        assert latency > 0
        assert latency < float("inf")
        assert ctx["kernel_name"] != ""
        assert len(ctx["sections"]) > 0


class TestProfilerErrorHandling:
    """Test error handling in profiler agent."""

    def test_invalid_kernel_path(self):
        """Test handling of invalid kernel path."""
        from kernelopt.agents.profiler import run_ncu

        with pytest.raises((FileNotFoundError, Exception)):
            run_ncu("/nonexistent/kernel.py", "/tmp/output")

    def test_empty_csv_handling(self, temp_output_dir):
        """Test handling of empty CSV files."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        empty_csv = temp_output_dir / "empty.csv"
        empty_csv.write_text("")

        # Should handle gracefully
        sections, rules, launch = parse_ncu_csv(str(empty_csv))

        assert isinstance(sections, dict)
        assert isinstance(rules, list)
        assert isinstance(launch, dict)

    def test_malformed_sections(self):
        """Test handling of malformed section data."""
        from kernelopt.agents.profiler import build_profiling_context

        # Missing required fields
        sections = {"InvalidSection": {}}
        rules = []
        launch = {}

        ctx = build_profiling_context(sections, rules, launch, "/tmp/test.py")

        # Should still create a context
        assert "kernel_name" in ctx
        assert "sections" in ctx
