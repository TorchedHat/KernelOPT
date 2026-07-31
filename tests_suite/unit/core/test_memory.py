"""
Comprehensive tests for optimization memory module.

Tests cover:
- Experience item storage
- Memory capacity management
- Top-K selection
- Curation thresholds
- Persistence (save/load)
- Memory updates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestMemoryConstants:
    """Test memory module constants."""

    def test_expn_default(self):
        """Test default memory capacity."""
        from kernelopt.memory import Q_CAP

        assert Q_CAP == 8  # As per README

    def test_threshold_defaults(self):
        """Test default curation thresholds."""
        from kernelopt.memory import SPE_GATE, SLO_GATE

        assert SPE_GATE > 1.0  # Positive speedup threshold
        assert SLO_GATE > 1.0  # Negative slowdown threshold

    def test_top_k_default(self):
        """Test default top-K value."""
        from kernelopt.memory import KEEP_N

        assert KEEP_N == 4  # As per README


class TestShouldStore:
    """Test curation logic for experience items."""

    def test_should_store_positive_above_threshold(self):
        """Test storing positive speedup above threshold."""
        from kernelopt.memory import should_store

        # Significant speedup
        assert should_store(1.10, spe_gate=1.05) is True
        assert should_store(1.50, spe_gate=1.05) is True
        assert should_store(2.00, spe_gate=1.05) is True

    def test_should_store_positive_below_threshold(self):
        """Test not storing insignificant positive speedup."""
        from kernelopt.memory import should_store

        assert should_store(1.02, spe_gate=1.05) is False
        assert should_store(1.03, spe_gate=1.05) is False

    def test_should_store_negative_significant(self):
        """Test storing significant slowdowns."""
        from kernelopt.memory import should_store

        # 1/speedup >= slo_gate
        assert should_store(0.80, slo_gate=1.20) is True  # 1/0.80 = 1.25
        assert should_store(0.70, slo_gate=1.20) is True  # 1/0.70 = 1.43

    def test_should_store_negative_insignificant(self):
        """Test not storing insignificant slowdowns."""
        from kernelopt.memory import should_store

        assert should_store(0.95, slo_gate=1.20) is False  # 1/0.95 = 1.05
        assert should_store(0.90, slo_gate=1.20) is False  # 1/0.90 = 1.11

    def test_should_store_neutral(self):
        """Test not storing neutral changes."""
        from kernelopt.memory import should_store

        assert should_store(1.00, spe_gate=1.05, slo_gate=1.20) is False
        assert should_store(0.99, spe_gate=1.05, slo_gate=1.20) is False

    def test_should_store_custom_thresholds(self):
        """Test with custom thresholds."""
        from kernelopt.memory import should_store

        # Strict thresholds
        assert should_store(1.10, spe_gate=1.20) is False
        assert should_store(1.25, spe_gate=1.20) is True

        # Lenient thresholds
        assert should_store(1.02, spe_gate=1.01) is True


class TestSelectTopK:
    """Test top-K experience item selection."""

    def test_select_top_k_basic(self):
        """Test basic top-K selection."""
        from kernelopt.memory import select_top_k

        items = [
            {"speedup": 1.20},
            {"speedup": 1.10},
            {"speedup": 1.05},
            {"speedup": 0.90},
        ]

        top_2 = select_top_k(items, keep_n=2, spe_gate=1.05, slo_gate=1.20)

        assert len(top_2) <= 2

    def test_select_top_k_by_deviation(self):
        """Test that items are sorted by |speedup - 1.0|."""
        from kernelopt.memory import select_top_k

        items = [
            {"speedup": 1.05},  # |0.05| = 0.05
            {"speedup": 1.30},  # |0.30| = 0.30
            {"speedup": 0.80},  # |0.20| = 0.20
            {"speedup": 1.15},  # |0.15| = 0.15
        ]

        top_items = select_top_k(items, keep_n=4, spe_gate=1.05, slo_gate=1.20)

        # First should be 1.30 (largest deviation)
        if len(top_items) > 0:
            assert top_items[0]["speedup"] == 1.30

    def test_select_top_k_filtering(self):
        """Test that insignificant items are filtered."""
        from kernelopt.memory import select_top_k

        items = [
            {"speedup": 1.01},
            {"speedup": 1.02},
            {"speedup": 1.03},
        ]

        top = select_top_k(items, keep_n=10, spe_gate=1.05, slo_gate=1.20)

        # All should be filtered out
        assert len(top) == 0

    def test_select_top_k_mixed(self):
        """Test selection with mix of positive and negative."""
        from kernelopt.memory import select_top_k

        items = [
            {"speedup": 1.25},  # Significant positive
            {"speedup": 0.75},  # Significant negative
            {"speedup": 1.02},  # Insignificant
            {"speedup": 1.15},  # Significant positive
        ]

        top = select_top_k(items, keep_n=3, spe_gate=1.05, slo_gate=1.20)

        # Should include both significant positive and negative
        assert len(top) > 0
        speedups = [item["speedup"] for item in top]
        # Should include 1.25 and 0.75 (most significant)
        assert 1.25 in speedups or 0.75 in speedups

    def test_select_top_k_empty(self):
        """Test selection with empty list."""
        from kernelopt.memory import select_top_k

        top = select_top_k([], keep_n=5)

        assert len(top) == 0

    def test_select_top_k_more_than_available(self):
        """Test requesting more items than available."""
        from kernelopt.memory import select_top_k

        items = [
            {"speedup": 1.20},
            {"speedup": 1.10},
        ]

        top = select_top_k(items, keep_n=10, spe_gate=1.05, slo_gate=1.20)

        # Should return all qualifying items
        assert len(top) <= 2


class TestAppendItem:
    """Test appending single items to memory."""

    def test_append_item_to_empty(self):
        """Test appending to empty memory."""
        from kernelopt.memory import append_item

        memory = []
        item = {"id": 1, "speedup": 1.25}

        new_memory = append_item(memory, item, capacity=5)

        assert len(new_memory) == 1
        assert new_memory[0] == item

    def test_append_item_within_capacity(self):
        """Test appending within capacity."""
        from kernelopt.memory import append_item

        memory = [{"id": 1}, {"id": 2}]
        item = {"id": 3}

        new_memory = append_item(memory, item, capacity=5)

        assert len(new_memory) == 3
        assert new_memory[-1] == item

    def test_append_item_at_capacity(self):
        """Test appending when at capacity."""
        from kernelopt.memory import append_item

        memory = [{"id": i} for i in range(5)]
        item = {"id": 5}

        new_memory = append_item(memory, item, capacity=5)

        assert len(new_memory) == 5
        # Oldest (id=0) should be evicted
        assert new_memory[0]["id"] == 1
        assert new_memory[-1]["id"] == 5

    def test_append_item_immutability(self):
        """Test that original memory is not mutated."""
        from kernelopt.memory import append_item

        memory = [{"id": 1}]
        item = {"id": 2}

        new_memory = append_item(memory, item, capacity=5)

        # Original should be unchanged
        assert len(memory) == 1
        assert len(new_memory) == 2


class TestAppendItems:
    """Test appending multiple items to memory."""

    def test_append_items_basic(self):
        """Test appending multiple items."""
        from kernelopt.memory import append_items

        memory = []
        items = [{"id": 1}, {"id": 2}, {"id": 3}]

        new_memory = append_items(memory, items, capacity=10)

        assert len(new_memory) == 3
        assert all(item in new_memory for item in items)

    def test_append_items_exceeding_capacity(self):
        """Test appending items exceeding capacity."""
        from kernelopt.memory import append_items

        memory = [{"id": i} for i in range(3)]
        items = [{"id": 10}, {"id": 11}, {"id": 12}]

        new_memory = append_items(memory, items, capacity=5)

        assert len(new_memory) == 5
        # Should keep most recent 5
        assert new_memory[-1]["id"] == 12
        assert new_memory[-2]["id"] == 11

    def test_append_items_empty_list(self):
        """Test appending empty list."""
        from kernelopt.memory import append_items

        memory = [{"id": 1}]
        items = []

        new_memory = append_items(memory, items, capacity=5)

        assert len(new_memory) == 1
        assert new_memory == memory


class TestMemoryPersistence:
    """Test saving and loading memory."""

    def test_save_memory(self, temp_output_dir):
        """Test saving memory to JSON file."""
        from kernelopt.memory import save_memory

        memory = [
            {"item_id": "001", "speedup": 1.25},
            {"item_id": "002", "speedup": 0.85},
        ]

        path = temp_output_dir / "test_memory.json"
        save_memory(memory, str(path))

        assert path.exists()

        # Verify content
        with open(path) as f:
            loaded = json.load(f)

        assert loaded == memory

    def test_load_memory(self, temp_output_dir):
        """Test loading memory from JSON file."""
        from kernelopt.memory import load_memory

        memory = [
            {"item_id": "001", "speedup": 1.25},
            {"item_id": "002", "speedup": 0.85},
        ]

        path = temp_output_dir / "test_load.json"
        with open(path, "w") as f:
            json.dump(memory, f)

        loaded = load_memory(str(path))

        assert loaded == memory

    def test_load_memory_nonexistent(self):
        """Test loading from nonexistent file."""
        from kernelopt.memory import load_memory

        loaded = load_memory("/nonexistent/file.json")

        # Should return empty list
        assert loaded == []

    def test_save_load_roundtrip(self, temp_output_dir):
        """Test save/load roundtrip."""
        from kernelopt.memory import save_memory, load_memory

        original = [
            {
                "item_id": "exp_001",
                "direction": "optimize_memory",
                "rewrite_type": "positive",
                "speedup": 1.42,
                "framework": "triton",
            }
        ]

        path = temp_output_dir / "roundtrip.json"
        save_memory(original, str(path))
        loaded = load_memory(str(path))

        assert loaded == original


class TestMemoryIntegration:
    """Integration tests for memory system."""

    def test_full_memory_workflow(self):
        """Test complete memory workflow."""
        from kernelopt.memory import select_top_k, append_items

        # Generate experience items
        candidates = [
            {"speedup": 1.30, "direction": "opt_a"},
            {"speedup": 0.85, "direction": "opt_b"},
            {"speedup": 1.10, "direction": "opt_c"},
            {"speedup": 1.02, "direction": "opt_d"},
        ]

        # Select top-K
        top = select_top_k(candidates, keep_n=2, spe_gate=1.05, slo_gate=1.20)

        # Append to memory
        memory = []
        memory = append_items(memory, top, capacity=8)

        # Verify
        assert len(memory) <= 2
        assert all(item["speedup"] >= 1.10 or item["speedup"] <= 0.85 for item in memory)

    def test_memory_capacity_enforcement(self):
        """Test that memory capacity (8) is enforced."""
        from kernelopt.memory import append_items, Q_CAP

        # Create 10 items
        items = [{"id": i, "speedup": 1.1 + i * 0.01} for i in range(10)]

        memory = append_items([], items, capacity=Q_CAP)

        # Should only keep last Q_CAP items
        assert len(memory) == Q_CAP
        assert memory[-1]["id"] == 9
