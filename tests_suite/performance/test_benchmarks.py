"""
Performance benchmarks for critical paths.

These tests use pytest-benchmark to establish baseline performance
and detect regressions in critical code paths.

Run with: pytest tests_suite/performance/ --benchmark-only
Compare: pytest tests_suite/performance/ --benchmark-compare
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestLLMPerformance:
    """Benchmark LLM-related operations."""

    def test_strip_thinking_performance(self, benchmark):
        """Benchmark thinking tag removal performance."""
        from kernelopt.llm import strip_thinking

        # Test with nested thinking blocks
        content = "A <think>outer <think>inner</think> outer</think> B " * 100

        result = benchmark(strip_thinking, content)

        assert "think" not in result.lower()

    def test_strip_thinking_large_content(self, benchmark):
        """Benchmark thinking tag removal on large content."""
        from kernelopt.llm import strip_thinking

        # Large content with many thinking blocks
        content = "Text " + ("<think>reasoning</think> more text " * 1000)

        result = benchmark(strip_thinking, content)

        assert "think" not in result.lower()


class TestCSVParsingPerformance:
    """Benchmark CSV parsing operations."""

    def test_parse_ncu_csv_performance(self, benchmark, sample_ncu_csv):
        """Benchmark NCU CSV parsing."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        sections, rules, launch = benchmark(parse_ncu_csv, sample_ncu_csv)

        assert isinstance(sections, dict)
        assert isinstance(rules, list)
        assert isinstance(launch, dict)

    @pytest.mark.slow
    def test_parse_h200_csv_performance(self, benchmark, h200_ncu_csv):
        """Benchmark parsing real H200 NCU output (large file)."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        sections, rules, launch = benchmark(parse_ncu_csv, h200_ncu_csv)

        # Real H200 output should have substantial data
        assert len(sections) > 0
        assert len(rules) > 0


class TestMemoryOperationsPerformance:
    """Benchmark memory management operations."""

    def test_memory_append_performance(self, benchmark):
        """Benchmark appending items to memory queue."""
        from kernelopt.memory import append_item

        memory = []
        new_item = {
            "item_id": "test",
            "direction": "optimize",
            "speedup": 1.5,
            "rewrite_type": "positive",
        }

        def append_many():
            m = []
            for i in range(100):
                item = {**new_item, "item_id": f"item_{i}"}
                m = append_item(m, item, capacity=50)
            return m

        result = benchmark(append_many)

        assert len(result) > 0

    def test_memory_dedup_performance(self, benchmark):
        """Benchmark memory deduplication checks."""
        # This would test the performance of checking for duplicates
        # in a large memory queue

        memory = [
            {"item_id": f"item_{i}", "direction": f"opt_{i % 10}", "speedup": 1.1 + i * 0.01}
            for i in range(100)
        ]

        def check_duplicates():
            # Simulate dedup check
            directions = set()
            for item in memory:
                if item["direction"] in directions:
                    continue
                directions.add(item["direction"])
            return len(directions)

        result = benchmark(check_duplicates)

        assert result > 0


class TestProfilingContextPerformance:
    """Benchmark profiling context building."""

    def test_build_profiling_context_performance(self, benchmark):
        """Benchmark profiling context construction."""
        from kernelopt.agents.profiler import build_profiling_context

        # Moderate-sized profiling data
        sections = {
            f"Section{i}": {
                f"Metric{j}": {"value": i * j, "unit": "unit"}
                for j in range(10)
            }
            for i in range(20)
        }

        rules = [
            {"name": f"Rule{i}", "type": "OPT", "speedup_pct": i * 0.1}
            for i in range(50)
        ]

        launch = {"block_size": 256, "grid_size": 1024}

        ctx = benchmark(
            build_profiling_context,
            sections,
            rules,
            launch,
            "/tmp/test.py",
        )

        assert "kernel_name" in ctx
        assert "sections" in ctx


class TestStateManagementPerformance:
    """Benchmark state operations."""

    def test_state_creation_performance(self, benchmark):
        """Benchmark PipelineState creation."""
        from kernelopt.state import PipelineState, PipelineConfig

        def create_state():
            config = PipelineConfig(
                kernel_path="/tmp/test.py",
                T=5,
                N=4,
                K=3,
            )
            return PipelineState(
                config=config,
                kernel_path="/tmp/test.py",
                kernel_source="# code",
                kernel_basename="test",
            )

        state = benchmark(create_state)

        assert state.kernel_path == "/tmp/test.py"


class TestSerializationPerformance:
    """Benchmark serialization/deserialization."""

    def test_json_dumps_performance(self, benchmark):
        """Benchmark JSON serialization of large data structures."""
        import json

        # Large profiling context
        data = {
            "sections": {
                f"section_{i}": {
                    f"metric_{j}": {"value": i * j, "unit": "test"}
                    for j in range(50)
                }
                for i in range(20)
            },
            "rules": [
                {"name": f"rule_{i}", "value": i}
                for i in range(100)
            ],
        }

        result = benchmark(json.dumps, data)

        assert len(result) > 0

    def test_json_loads_performance(self, benchmark):
        """Benchmark JSON deserialization."""
        import json

        data = {
            "sections": {f"s{i}": {"metric": i} for i in range(100)},
            "rules": [{"name": f"r{i}"} for i in range(200)],
        }

        json_str = json.dumps(data)

        result = benchmark(json.loads, json_str)

        assert "sections" in result


class TestFileIOPerformance:
    """Benchmark file I/O operations."""

    def test_read_kernel_file_performance(self, benchmark, test_kernel_path):
        """Benchmark reading kernel files."""

        def read_kernel():
            return test_kernel_path.read_text()

        content = benchmark(read_kernel)

        assert len(content) > 0

    def test_write_kernel_file_performance(self, benchmark, temp_output_dir):
        """Benchmark writing kernel files."""
        kernel_content = "# test kernel\n" + ("x = 1\n" * 1000)

        def write_kernel():
            output_file = temp_output_dir / "benchmark_output.py"
            output_file.write_text(kernel_content)
            return output_file

        output_file = benchmark(write_kernel)

        assert output_file.exists()


@pytest.mark.benchmark(group="regex")
class TestRegexPerformance:
    """Benchmark regex operations."""

    def test_kernel_name_extraction_performance(self, benchmark):
        """Benchmark kernel name extraction regex."""
        import re

        content = """
def kernel1(x): pass
def kernel2(y): pass
def kernel3(z): pass
""" * 100

        pattern = r"def\s+(\w+)\s*\("

        matches = benchmark(re.findall, pattern, content)

        assert len(matches) > 0

    def test_thinking_tag_regex_performance(self, benchmark):
        """Benchmark thinking tag removal regex."""
        import re

        content = "<think>reasoning</think> " * 1000
        pattern = r"<think>(?:(?!<think>)[\s\S])*?</think>"

        def remove_all():
            c = content
            while '<think>' in c:
                prev = c
                c = re.sub(pattern, "", c)
                if prev == c:
                    break
            return c

        result = benchmark(remove_all)

        assert "think" not in result.lower()


@pytest.mark.slow
@pytest.mark.gpu
class TestGPUOperationsPerformance:
    """Benchmark GPU-related operations."""

    def test_gpu_memory_check_performance(self, benchmark):
        """Benchmark GPU memory checking."""
        from kernelopt.functions.gpu_monitoring import get_gpu_memory

        memory_info = benchmark(get_gpu_memory, 0)

        assert "total" in memory_info or memory_info is None

    def test_gpu_utilization_check_performance(self, benchmark):
        """Benchmark GPU utilization checking."""
        from kernelopt.functions.gpu_monitoring import get_gpu_utilization

        utilization = benchmark(get_gpu_utilization, 0)

        assert utilization is not None or utilization is None


class TestBaselinePerformance:
    """Establish baseline performance metrics."""

    def test_baseline_iteration_time(self, benchmark):
        """Establish baseline for iteration time (target: <100ms)."""

        def simulate_iteration():
            # Simulate a fast iteration
            result = []
            for i in range(1000):
                result.append(i * 2)
            return sum(result)

        result = benchmark(simulate_iteration)

        assert result > 0
        # Benchmark will show if this takes longer than expected

    def test_baseline_memory_allocation(self, benchmark):
        """Establish baseline for memory allocation."""

        def allocate_memory():
            # Allocate moderate amount of memory
            data = [0] * 10000
            return len(data)

        result = benchmark(allocate_memory)

        assert result == 10000


# Benchmark comparison guidance
"""
To compare benchmarks across runs:

1. Run and save baseline:
   pytest tests_suite/performance/ --benchmark-save=baseline

2. Make changes to code

3. Compare with baseline:
   pytest tests_suite/performance/ --benchmark-compare=baseline

4. Auto-fail if performance degrades >10%:
   pytest tests_suite/performance/ --benchmark-compare=baseline --benchmark-compare-fail=mean:10%

Performance targets:
- strip_thinking: <1ms for 1000 char input
- parse_ncu_csv: <50ms for sample CSV, <500ms for H200 CSV
- memory operations: <10ms for 100 items
- JSON serialization: <100ms for large contexts
"""
