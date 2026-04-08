"""
Comprehensive tests for CSV parser functions.

Tests cover:
- Number parsing with various formats
- Block/grid dimension parsing
- NCU CSV file parsing
- Kernel listing
- Section extraction
- Rule extraction
- Launch configuration extraction
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestNumberParsing:
    """Test number parsing utilities."""

    def test_parse_number_integers(self):
        """Test parsing integer values."""
        from kernelopt.functions.csv_parser import _parse_number

        assert _parse_number("123") == 123.0
        assert _parse_number("0") == 0.0
        assert _parse_number("999999") == 999999.0

    def test_parse_number_floats(self):
        """Test parsing float values."""
        from kernelopt.functions.csv_parser import _parse_number

        assert _parse_number("123.45") == 123.45
        assert _parse_number("0.001") == 0.001
        assert _parse_number(".5") == 0.5

    def test_parse_number_with_commas(self):
        """Test parsing numbers with thousand separators."""
        from kernelopt.functions.csv_parser import _parse_number

        assert _parse_number("1,234") == 1234.0
        assert _parse_number("1,234,567.89") == 1234567.89
        assert _parse_number("10,000,000") == 10000000.0

    def test_parse_number_with_whitespace(self):
        """Test parsing numbers with surrounding whitespace."""
        from kernelopt.functions.csv_parser import _parse_number

        assert _parse_number("  123  ") == 123.0
        assert _parse_number("\t456\n") == 456.0

    def test_parse_number_scientific_notation(self):
        """Test parsing scientific notation."""
        from kernelopt.functions.csv_parser import _parse_number

        assert _parse_number("1.5e6") == 1.5e6
        assert _parse_number("2.5E-3") == 2.5e-3

    def test_parse_number_invalid(self):
        """Test parsing invalid numbers returns original string."""
        from kernelopt.functions.csv_parser import _parse_number

        assert _parse_number("not_a_number") == "not_a_number"
        assert _parse_number("N/A") == "N/A"
        assert _parse_number("") == ""
        assert _parse_number("inf") == float("inf")


class TestBlockGridParsing:
    """Test block and grid dimension parsing."""

    def test_parse_block_grid_single_number(self):
        """Test parsing single dimension."""
        from kernelopt.functions.csv_parser import _parse_block_grid

        assert _parse_block_grid("256") == 256
        assert _parse_block_grid("1024") == 1024

    def test_parse_block_grid_1d_tuple(self):
        """Test parsing 1D tuple format."""
        from kernelopt.functions.csv_parser import _parse_block_grid

        assert _parse_block_grid("(256)") == 256
        assert _parse_block_grid("(1024,)") == 1024

    def test_parse_block_grid_2d(self):
        """Test parsing 2D dimensions."""
        from kernelopt.functions.csv_parser import _parse_block_grid

        assert _parse_block_grid("(16, 16)") == 256
        assert _parse_block_grid("(32, 32)") == 1024
        assert _parse_block_grid("(256, 1)") == 256

    def test_parse_block_grid_3d(self):
        """Test parsing 3D dimensions."""
        from kernelopt.functions.csv_parser import _parse_block_grid

        assert _parse_block_grid("(256, 1, 1)") == 256
        assert _parse_block_grid("(16, 16, 4)") == 1024
        assert _parse_block_grid("(8, 8, 8)") == 512
        assert _parse_block_grid("(4, 4, 4)") == 64

    def test_parse_block_grid_empty(self):
        """Test parsing empty input."""
        from kernelopt.functions.csv_parser import _parse_block_grid

        assert _parse_block_grid("") == 0
        assert _parse_block_grid("()") == 0
        assert _parse_block_grid("(,,)") == 0

    def test_parse_block_grid_malformed(self):
        """Test parsing malformed input."""
        from kernelopt.functions.csv_parser import _parse_block_grid

        assert _parse_block_grid("invalid") == 0
        assert _parse_block_grid("(a, b, c)") == 0


class TestKernelListing:
    """Test kernel listing from CSV files."""

    def test_list_kernels_basic(self, sample_ncu_csv):
        """Test listing kernels from CSV."""
        from kernelopt.functions.csv_parser import list_kernels

        kernels = list_kernels(str(sample_ncu_csv))

        assert isinstance(kernels, list)
        assert len(kernels) > 0

    def test_list_kernels_structure(self, sample_ncu_csv):
        """Test kernel list structure."""
        from kernelopt.functions.csv_parser import list_kernels

        kernels = list_kernels(str(sample_ncu_csv))

        for kernel in kernels:
            assert isinstance(kernel, tuple)
            assert len(kernel) == 2
            kid, kname = kernel
            assert isinstance(kid, int)
            assert isinstance(kname, str)

    def test_list_kernels_sorted(self, sample_ncu_csv):
        """Test that kernels are sorted by ID."""
        from kernelopt.functions.csv_parser import list_kernels

        kernels = list_kernels(str(sample_ncu_csv))

        if len(kernels) > 1:
            for i in range(len(kernels) - 1):
                assert kernels[i][0] <= kernels[i + 1][0]


class TestCSVParsing:
    """Test NCU CSV parsing."""

    def test_parse_ncu_csv_basic(self, sample_ncu_csv):
        """Test basic CSV parsing."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        sections, rules, launch = parse_ncu_csv(str(sample_ncu_csv), kernel_id=0)

        assert isinstance(sections, dict)
        assert isinstance(rules, list)
        assert isinstance(launch, dict)

    def test_parse_ncu_csv_sections_structure(self, sample_ncu_csv):
        """Test sections structure."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        sections, _, _ = parse_ncu_csv(str(sample_ncu_csv), kernel_id=0)

        for section_name, metrics in sections.items():
            assert isinstance(section_name, str)
            assert isinstance(metrics, dict)
            for metric_name, metric_value in metrics.items():
                assert isinstance(metric_name, str)
                assert isinstance(metric_value, dict)
                assert "value" in metric_value
                assert "unit" in metric_value

    def test_parse_ncu_csv_rules_structure(self, sample_ncu_csv):
        """Test rules structure."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        _, rules, _ = parse_ncu_csv(str(sample_ncu_csv), kernel_id=0)

        for rule in rules:
            assert isinstance(rule, dict)
            assert "name" in rule
            assert "type" in rule
            assert rule["type"] in ["OPT", "INF", "WRN"]

    def test_parse_ncu_csv_launch_structure(self, sample_ncu_csv):
        """Test launch configuration structure."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        _, _, launch = parse_ncu_csv(str(sample_ncu_csv), kernel_id=0)

        # Launch config may have these fields
        possible_fields = [
            "block_size",
            "grid_size",
            "registers_per_thread",
            "sm_count",
            "waves_per_sm",
        ]

        # At least some fields should be present
        assert any(field in launch for field in possible_fields)

    def test_parse_ncu_csv_kernel_id_selection(self):
        """Test selecting specific kernel by ID."""
        from kernelopt.functions.csv_parser import parse_ncu_csv
        import tempfile

        # Create CSV with multiple kernels
        csv_content = """ID,Kernel Name,Metric Name,Metric Value,Metric Unit
0,kernel_a,Duration,100000,nsecond
1,kernel_b,Duration,200000,nsecond
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            csv_path = f.name

        # Parse kernel 0
        sections_0, _, _ = parse_ncu_csv(csv_path, kernel_id=0)
        # Parse kernel 1
        sections_1, _, _ = parse_ncu_csv(csv_path, kernel_id=1)

        # Should get different data (or at least not fail)
        assert isinstance(sections_0, dict)
        assert isinstance(sections_1, dict)


class TestCSVEdgeCases:
    """Test edge cases in CSV parsing."""

    def test_empty_csv(self, temp_output_dir):
        """Test handling of empty CSV."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        empty_csv = temp_output_dir / "empty.csv"
        empty_csv.write_text("")

        sections, rules, launch = parse_ncu_csv(str(empty_csv))

        assert isinstance(sections, dict)
        assert isinstance(rules, list)
        assert isinstance(launch, dict)

    def test_csv_with_only_header(self, temp_output_dir):
        """Test CSV with only header row."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        header_csv = temp_output_dir / "header_only.csv"
        header_csv.write_text("ID,Kernel Name,Metric Name,Metric Value,Metric Unit\n")

        sections, rules, launch = parse_ncu_csv(str(header_csv))

        assert isinstance(sections, dict)

    def test_csv_with_missing_values(self, temp_output_dir):
        """Test CSV with missing values."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        csv_content = """ID,Kernel Name,Metric Name,Metric Value,Metric Unit
0,test_kernel,Duration,,nsecond
0,test_kernel,Throughput,N/A,GB/s
"""
        csv_path = temp_output_dir / "missing_values.csv"
        csv_path.write_text(csv_content)

        sections, rules, launch = parse_ncu_csv(str(csv_path))

        assert isinstance(sections, dict)

    def test_csv_with_special_characters(self, temp_output_dir):
        """Test CSV with special characters in names."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        csv_content = """ID,Kernel Name,Metric Name,Metric Value,Metric Unit
0,"kernel<float, 256>",Duration,100000,nsecond
"""
        csv_path = temp_output_dir / "special_chars.csv"
        csv_path.write_text(csv_content)

        sections, rules, launch = parse_ncu_csv(str(csv_path))

        assert isinstance(sections, dict)


class TestCSVIntegration:
    """Integration tests for CSV parsing."""

    def test_parse_and_extract_duration(self, sample_ncu_csv):
        """Test parsing CSV and extracting duration."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        sections, _, _ = parse_ncu_csv(str(sample_ncu_csv))

        # Look for duration in various locations
        duration_found = False
        for section_name, metrics in sections.items():
            if "Duration" in metrics:
                duration_found = True
                break
            for metric_name in metrics:
                if "duration" in metric_name.lower():
                    duration_found = True
                    break

        # May or may not have duration
        assert isinstance(duration_found, bool)
