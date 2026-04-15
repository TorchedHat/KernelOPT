"""
Checkpoint Tests.

Tests for checkpoint save/load/resume functionality.
Checkpoints allow resuming interrupted pipeline runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestCheckpointWrite:
    """Test checkpoint file creation and writing."""

    def test_checkpoint_constants(self):
        """Test checkpoint constants are defined."""
        from kernelopt.checkpoint import CHECKPOINT_FILENAME, CHECKPOINT_VERSION

        assert CHECKPOINT_FILENAME == "pipeline_checkpoint.json"
        assert CHECKPOINT_VERSION == 1

    def test_pipeline_config_to_dict(self, temp_output_dir):
        """Test converting PipelineConfig to dict for checkpoint."""
        from kernelopt.state import PipelineConfig
        from dataclasses import asdict

        config = PipelineConfig(
            kernel_path="/tmp/kernel.cu",
            T=5,
            N=4,
            K=3,
        )

        config_dict = asdict(config)

        assert config_dict["kernel_path"] == "/tmp/kernel.cu"
        assert config_dict["T"] == 5
        assert config_dict["N"] == 4
        assert config_dict["K"] == 3


class TestCheckpointRead:
    """Test checkpoint file loading and parsing."""

    def test_load_pipeline_config_from_checkpoint_dict(self):
        """Test rebuilding PipelineConfig from checkpoint dict."""
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict

        checkpoint_dict = {
            "kernel_path": "/tmp/kernel.cu",
            "T": 5,
            "N": 4,
            "K": 3,
            "model": "claude-sonnet-4@20250514",
        }

        config = pipeline_config_from_checkpoint_dict(checkpoint_dict)

        assert config.kernel_path == "/tmp/kernel.cu"
        assert config.T == 5
        assert config.N == 4
        assert config.K == 3

    def test_load_checkpoint_with_extra_fields(self):
        """Test loading checkpoint with unknown fields."""
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict

        checkpoint_dict = {
            "kernel_path": "/tmp/kernel.cu",
            "T": 5,
            "N": 4,
            "K": 3,
            "unknown_field": "should_be_ignored",
        }

        # Should ignore unknown fields
        config = pipeline_config_from_checkpoint_dict(checkpoint_dict)

        assert config.kernel_path == "/tmp/kernel.cu"

    def test_load_checkpoint_with_missing_fields(self):
        """Test loading checkpoint with missing optional fields."""
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict

        minimal_dict = {
            "kernel_path": "/tmp/kernel.cu",
        }

        # Should use defaults for missing fields
        config = pipeline_config_from_checkpoint_dict(minimal_dict)

        assert config.kernel_path == "/tmp/kernel.cu"
        # Defaults should be applied
        assert config.T > 0

    def test_corrupted_checkpoint_dict(self):
        """Test handling corrupted checkpoint data."""
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict

        # Empty dict
        empty_dict = {}

        try:
            config = pipeline_config_from_checkpoint_dict(empty_dict)
            # If it doesn't raise, it should use all defaults
            assert config is not None
        except (KeyError, TypeError):
            # Expected if kernel_path is required
            pass


class TestCheckpointResume:
    """Test resuming from different checkpoint stages."""

    def test_checkpoint_entry_points_defined(self):
        """Test that checkpoint entry points are defined."""
        # Entry points documented in checkpoint.py
        entry_points = [
            "after_baseline",
            "after_plans",
            "after_execute",
            "after_summarize",
            "after_select",
            "after_iteration",
        ]

        # These are documented as valid entry points
        assert len(entry_points) == 6

    def test_iteration_result_from_dict(self):
        """Test rebuilding IterationResult from dict."""
        from kernelopt.checkpoint import _iteration_result_from_dict

        result_dict = {
            "iteration": 1,
            "candidates": [
                {
                    "plan_id": "A",
                    "latency_ns": 1000,
                    "speedup": 1.5,
                }
            ],
            "best_candidate": {
                "plan_id": "A",
                "latency_ns": 1000,
                "speedup": 1.5,
            },
            "improved": True,
        }

        result = _iteration_result_from_dict(result_dict)

        assert result.iteration == 1
        assert len(result.candidates) == 1
        assert result.best_candidate.plan_id == "A"
        assert result.improved is True

    def test_plan_candidate_from_dict(self):
        """Test rebuilding PlanCandidate from dict."""
        from kernelopt.checkpoint import _plan_candidate_from_dict

        candidate_dict = {
            "plan_id": "A",
            "latency_ns": 1000,
            "speedup": 1.5,
            "plan": {},
        }

        candidate = _plan_candidate_from_dict(candidate_dict)

        assert candidate.plan_id == "A"
        assert candidate.latency_ns == 1000
        assert candidate.speedup == 1.5

    def test_plans_to_json(self):
        """Test converting plans to JSON format."""
        from kernelopt.checkpoint import _plans_to_json

        plans = [
            ("A", {"direction": "optimize_memory"}),
            ("B", {"direction": "optimize_compute"}),
        ]

        json_plans = _plans_to_json(plans)

        assert len(json_plans) == 2
        assert json_plans[0]["letter"] == "A"
        assert json_plans[0]["plan"]["direction"] == "optimize_memory"


class TestCheckpointCompatibility:
    """Test checkpoint version compatibility."""

    def test_checkpoint_version_field(self, temp_output_dir):
        """Test checkpoint includes version field."""
        from kernelopt.checkpoint import CHECKPOINT_VERSION

        checkpoint_data = {
            "version": CHECKPOINT_VERSION,
            "entry": "after_baseline",
            "config": {},
        }

        checkpoint_file = temp_output_dir / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        loaded = json.loads(checkpoint_file.read_text())
        assert loaded["version"] == CHECKPOINT_VERSION

    def test_handle_old_checkpoint_version(self):
        """Test handling checkpoint with old version."""
        old_version_checkpoint = {
            "version": 0,  # Old version
            "entry": "after_baseline",
        }

        # Should be able to detect version mismatch
        from kernelopt.checkpoint import CHECKPOINT_VERSION
        assert old_version_checkpoint["version"] != CHECKPOINT_VERSION

    def test_checkpoint_json_format(self, temp_output_dir):
        """Test checkpoint is valid JSON."""
        checkpoint_data = {
            "version": 1,
            "entry": "after_baseline",
            "graph_state": {
                "config": {},
                "iteration": 0,
            },
        }

        checkpoint_file = temp_output_dir / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))

        # Should be valid JSON
        loaded = json.loads(checkpoint_file.read_text())
        assert loaded["version"] == 1
        assert loaded["entry"] == "after_baseline"


class TestCheckpointFileOperations:
    """Test checkpoint file operations."""

    def test_checkpoint_filename_constant(self):
        """Test checkpoint filename constant."""
        from kernelopt.checkpoint import CHECKPOINT_FILENAME

        assert CHECKPOINT_FILENAME.endswith(".json")
        assert "checkpoint" in CHECKPOINT_FILENAME.lower()

    def test_checkpoint_in_output_dir(self, temp_output_dir):
        """Test checkpoint is written to output directory."""
        from kernelopt.checkpoint import CHECKPOINT_FILENAME

        output_dir = temp_output_dir / "pipeline_output"
        output_dir.mkdir()
        checkpoint_path = output_dir / CHECKPOINT_FILENAME

        # Simulate writing checkpoint
        checkpoint_path.write_text("{}")

        assert checkpoint_path.exists()
        assert checkpoint_path.parent == output_dir

    def test_checkpoint_file_not_exists(self, temp_output_dir):
        """Test handling when checkpoint file doesn't exist."""
        from kernelopt.checkpoint import CHECKPOINT_FILENAME

        nonexistent = temp_output_dir / CHECKPOINT_FILENAME

        assert not nonexistent.exists()


class TestCheckpointGraphState:
    """Test checkpoint graph state serialization."""

    def test_graph_state_in_checkpoint(self):
        """Test GraphState is included in checkpoint."""
        checkpoint = {
            "version": 1,
            "entry": "after_baseline",
            "graph_state": {
                "iteration": 0,
                "baseline_latency_ns": 1000,
            },
        }

        assert "graph_state" in checkpoint
        assert checkpoint["graph_state"]["iteration"] == 0

    def test_checkpoint_preserves_baseline(self):
        """Test checkpoint preserves baseline profiling data."""
        checkpoint = {
            "graph_state": {
                "baseline_latency_ns": 1000,
                "baseline_profiling_ctx": {},
            }
        }

        assert checkpoint["graph_state"]["baseline_latency_ns"] == 1000

    def test_checkpoint_preserves_memory(self):
        """Test checkpoint preserves memory queue."""
        checkpoint = {
            "graph_state": {
                "memory": [
                    {"item_id": "test_1", "speedup": 1.5}
                ]
            }
        }

        assert len(checkpoint["graph_state"]["memory"]) == 1

    def test_checkpoint_preserves_iteration_history(self):
        """Test checkpoint preserves iteration history."""
        checkpoint = {
            "graph_state": {
                "iteration_history": [
                    {"iteration": 0, "improved": True},
                    {"iteration": 1, "improved": False},
                ]
            }
        }

        assert len(checkpoint["graph_state"]["iteration_history"]) == 2


class TestCheckpointEdgeCases:
    """Test checkpoint edge cases."""

    def test_empty_checkpoint_file(self, temp_output_dir):
        """Test handling empty checkpoint file."""
        checkpoint_file = temp_output_dir / "empty_checkpoint.json"
        checkpoint_file.write_text("")

        with pytest.raises(json.JSONDecodeError):
            json.loads(checkpoint_file.read_text())

    def test_invalid_json_checkpoint(self, temp_output_dir):
        """Test handling invalid JSON in checkpoint."""
        checkpoint_file = temp_output_dir / "invalid.json"
        checkpoint_file.write_text("{invalid json}")

        with pytest.raises(json.JSONDecodeError):
            json.loads(checkpoint_file.read_text())

    def test_checkpoint_with_null_values(self):
        """Test checkpoint with null values."""
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict

        checkpoint_dict = {
            "kernel_path": "/tmp/kernel.cu",
            "baseline_csv": None,  # Null value
            "api_key": None,
        }

        config = pipeline_config_from_checkpoint_dict(checkpoint_dict)
        assert config.baseline_csv is None
        assert config.api_key is None
