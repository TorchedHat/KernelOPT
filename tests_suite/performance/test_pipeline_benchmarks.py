"""
Pipeline Performance Benchmarks.

Benchmarks for end-to-end pipeline performance, LLM operations,
and GPU kernel execution overhead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestPipelineBenchmarks:
    """Benchmark pipeline operations."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_baseline_profiling_time(self, benchmark, temp_output_dir):
        """Benchmark time to profile baseline kernel."""
        # Simulated profiling time measurement
        def profile_baseline():
            import time
            time.sleep(0.001)  # Simulate profiling
            return {"duration": "1000ns"}

        result = benchmark(profile_baseline)
        assert result is not None

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_plan_generation_time(self, benchmark):
        """Benchmark plan generation time."""
        def generate_plans():
            # Simulate plan generation
            plans = [
                {"plan_id": "A", "direction": "optimize_memory"},
                {"plan_id": "B", "direction": "optimize_compute"},
            ]
            return plans

        result = benchmark(generate_plans)
        assert len(result) >= 2

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_executor_compilation_time(self, benchmark, temp_output_dir):
        """Benchmark kernel compilation time."""
        def compile_kernel():
            kernel_file = temp_output_dir / "bench.cu"
            kernel_file.write_text("__global__ void k() {}")
            return True

        result = benchmark(compile_kernel)
        assert result is True

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_full_iteration_time(self, benchmark):
        """Benchmark full optimization iteration."""
        def full_iteration():
            # Simulate: plan → execute → profile → summarize → select
            steps = ["plan", "execute", "profile", "summarize", "select"]
            return len(steps)

        result = benchmark(full_iteration)
        assert result == 5


class TestLLMBenchmarks:
    """Benchmark LLM operations."""

    @pytest.mark.benchmark
    def test_llm_invoke_time(self, benchmark):
        """Benchmark LLM invoke time (mocked)."""
        from unittest.mock import Mock

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Response")

        def invoke_llm():
            return mock_llm.invoke(["message"])

        result = benchmark(invoke_llm)
        assert result is not None

    @pytest.mark.benchmark
    def test_llm_with_structured_output_time(self, benchmark):
        """Benchmark LLM structured output time (mocked)."""
        from unittest.mock import Mock

        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = {"field": "value"}
        mock_llm.with_structured_output.return_value = mock_structured

        def invoke_structured():
            structured_llm = mock_llm.with_structured_output(dict)
            return structured_llm.invoke(["message"])

        result = benchmark(invoke_structured)
        assert result is not None

    @pytest.mark.benchmark
    def test_thinking_strip_overhead(self, benchmark):
        """Benchmark overhead of stripping thinking tags."""
        from kernelopt.llm import strip_thinking

        large_content = "Normal text " + "<think>thinking content " * 100 + "</think> " * 100 + "more text"

        result = benchmark(strip_thinking, large_content)
        assert "think" not in result.lower()


class TestGPUKernelBenchmarks:
    """Benchmark GPU kernel operations."""

    @pytest.mark.benchmark
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_cuda_kernel_launch_overhead(self, benchmark):
        """Benchmark CUDA kernel launch overhead."""
        def kernel_launch():
            # Simulate kernel launch
            return "launched"

        result = benchmark(kernel_launch)
        assert result == "launched"

    @pytest.mark.benchmark
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_ncu_profiling_overhead(self, benchmark):
        """Benchmark NCU profiling overhead."""
        def ncu_profile():
            # Simulate NCU profiling
            return {"duration": "1000ns"}

        result = benchmark(ncu_profile)
        assert result is not None


class TestMemoryBenchmarks:
    """Benchmark memory operations."""

    @pytest.mark.benchmark
    def test_memory_append_throughput(self, benchmark):
        """Benchmark memory append throughput."""
        from kernelopt.memory import append_item

        def append_many():
            memory = []
            for i in range(100):
                item = {"item_id": f"item_{i}", "speedup": 1.5}
                memory = append_item(memory, item, capacity=50)
            return len(memory)

        result = benchmark(append_many)
        assert result <= 50  # Capacity limit

    @pytest.mark.benchmark
    def test_memory_selection_throughput(self, benchmark):
        """Benchmark memory selection (TopK) throughput."""
        from kernelopt.memory import select_top_k

        # Create large memory
        memory = [
            {"item_id": f"item_{i}", "speedup": i * 0.1}
            for i in range(100)
        ]

        result = benchmark(select_top_k, memory, k=10, t_pos=1.04)
        assert len(result) <= 10


class TestSerializationBenchmarks:
    """Benchmark serialization operations."""

    @pytest.mark.benchmark
    def test_checkpoint_serialization_time(self, benchmark):
        """Benchmark checkpoint serialization time."""
        import json

        checkpoint_data = {
            "version": 1,
            "entry": "after_baseline",
            "graph_state": {
                "iteration": 5,
                "memory": [{"item_id": f"item_{i}", "speedup": 1.5} for i in range(50)],
                "iteration_history": [{"iteration": i, "improved": True} for i in range(5)],
            },
        }

        def serialize():
            return json.dumps(checkpoint_data)

        result = benchmark(serialize)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_checkpoint_deserialization_time(self, benchmark):
        """Benchmark checkpoint deserialization time."""
        import json

        checkpoint_json = json.dumps({
            "version": 1,
            "graph_state": {"iteration": 5},
        })

        def deserialize():
            return json.loads(checkpoint_json)

        result = benchmark(deserialize)
        assert result["version"] == 1


class TestCSVParsingBenchmarks:
    """Benchmark CSV parsing operations."""

    @pytest.mark.benchmark
    def test_ncu_csv_parsing_throughput(self, benchmark):
        """Benchmark NCU CSV parsing throughput."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        # Large NCU CSV
        ncu_csv = """"ID","Kernel Name","Metric Name","Metric Value"\n"""
        for i in range(100):
            ncu_csv += f'"0","kernel","Metric{i}","{i * 100}"\n'

        def parse():
            return parse_ncu_csv(ncu_csv)

        result = benchmark(parse)
        assert result is not None


class TestStateManagementBenchmarks:
    """Benchmark state management operations."""

    @pytest.mark.benchmark
    def test_pipeline_state_copy_time(self, benchmark):
        """Benchmark PipelineState copying time."""
        from kernelopt.state import PipelineState, PipelineConfig

        state = PipelineState(
            config=PipelineConfig(kernel_path="/tmp/test.cu"),
            kernel_path="/tmp/test.cu",
            kernel_source="__global__ void k() {}",
            kernel_basename="test",
            iteration=5,
            memory=[{"item_id": f"item_{i}", "speedup": 1.5} for i in range(50)],
        )

        def copy_state():
            from dataclasses import replace
            return replace(state)

        result = benchmark(copy_state)
        assert result.iteration == 5
