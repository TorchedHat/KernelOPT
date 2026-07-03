"""
Tests for Profiler Agent's three operational modes.

The profiler agent has three distinct modes for resolving NCU configuration:
1. Full bypass mode: ncu_config is provided (candidate re-profiling)
2. User bypass mode: kernel_name provided in pipeline_config (skip LLM)
3. LLM mode: agent decides kernel_name, launch_count, kernel_args

These tests verify each mode works correctly and produces valid profiling results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestProfilerFullBypassMode:
    """Test full bypass mode - reuse existing ncu_config."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_full_bypass_reuses_config(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test that full bypass mode reuses provided ncu_config without calling LLM."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        # Pre-defined ncu_config (simulating candidate re-profiling)
        existing_config = {
            "kernel_name": "softmax_kernel",
            "launch_skip": 0,
            "launch_count": 1,
            "kernel_args": [],
            "replay_mode": "kernel",
        }

        pipeline_config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            T=5,
            N=2,
            K=3,
        )

        output_base = str(temp_output_dir / "test_full_bypass")

        # Mock LLM to ensure it's NOT called
        mock_llm = Mock()

        profiling_ctx, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=existing_config,  # Provide config for full bypass
            pipeline_config=pipeline_config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            llm_override=mock_llm,
        )

        # Verify config was reused
        assert resolved_config == existing_config
        assert resolved_config["kernel_name"] == "softmax_kernel"
        assert resolved_config["launch_count"] == 1

        # Verify LLM was NOT called (full bypass)
        mock_llm.invoke.assert_not_called()

        # Verify profiling context was built
        assert profiling_ctx is not None
        assert "kernel_name" in profiling_ctx
        assert "sections" in profiling_ctx


class TestProfilerUserBypassMode:
    """Test user bypass mode - kernel_name provided, skip LLM."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_user_bypass_with_kernel_name(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test user bypass when kernel_name is provided in pipeline_config."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        pipeline_config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            T=5,
            N=2,
            K=3,
            kernel_name="softmax_kernel",  # User-provided kernel name
        )

        output_base = str(temp_output_dir / "test_user_bypass")

        # Mock LLM to ensure it's NOT called
        mock_llm = Mock()

        profiling_ctx, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=None,  # No config provided
            pipeline_config=pipeline_config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            llm_override=mock_llm,
        )

        # Verify user's kernel_name was used
        assert resolved_config["kernel_name"] == "softmax_kernel"

        # Verify LLM was NOT called (user bypass)
        mock_llm.invoke.assert_not_called()

        # Verify profiling context was built
        assert profiling_ctx is not None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_user_bypass_with_launch_config(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test user bypass with kernel_name and launch configuration."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        pipeline_config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            T=5,
            N=2,
            K=3,
            kernel_name="softmax_kernel",
            launch_skip=1,
            launch_count=2,
        )

        output_base = str(temp_output_dir / "test_user_bypass_launch")

        _, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=None,
            pipeline_config=pipeline_config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Verify all user settings were used
        assert resolved_config["kernel_name"] == "softmax_kernel"
        assert resolved_config["launch_skip"] == 1
        assert resolved_config["launch_count"] == 2

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_user_bypass_without_launch_count(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test user bypass with kernel_name but no launch_count (profiles all matching)."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        pipeline_config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            T=5,
            N=2,
            K=3,
            kernel_name="softmax_kernel",
            # No launch_count specified
        )

        output_base = str(temp_output_dir / "test_user_bypass_no_count")

        _, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=None,
            pipeline_config=pipeline_config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # launch_count should be None (profiles all matching kernels)
        assert resolved_config["kernel_name"] == "softmax_kernel"
        assert resolved_config["launch_count"] is None


class TestProfilerModeSelection:
    """Test that correct mode is selected based on inputs."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_mode_priority_full_bypass_wins(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test that full bypass takes priority over user bypass."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        existing_config = {
            "kernel_name": "from_config",
            "launch_skip": 0,
            "launch_count": 1,
            "kernel_args": [],
            "replay_mode": "kernel",
        }

        # Even with kernel_name in pipeline_config, full bypass should win
        pipeline_config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            T=5,
            N=2,
            K=3,
            kernel_name="from_pipeline",  # This should be ignored
        )

        output_base = str(temp_output_dir / "test_priority")

        mock_llm = Mock()

        _, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=existing_config,  # Full bypass config provided
            pipeline_config=pipeline_config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            llm_override=mock_llm,
        )

        # Full bypass should win - use config, not pipeline_config.kernel_name
        assert resolved_config["kernel_name"] == "from_config"

        # LLM should not be called
        mock_llm.invoke.assert_not_called()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_mode_priority_user_bypass_over_llm(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test that user bypass takes priority over LLM mode."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        pipeline_config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            T=5,
            N=2,
            K=3,
            kernel_name="user_specified",
        )

        output_base = str(temp_output_dir / "test_user_over_llm")

        mock_llm = Mock()

        _, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=None,  # No full bypass
            pipeline_config=pipeline_config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            llm_override=mock_llm,
        )

        # User bypass should be used
        assert resolved_config["kernel_name"] == "user_specified"

        # LLM should not be called
        mock_llm.invoke.assert_not_called()
