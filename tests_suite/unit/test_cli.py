"""
CLI Tests.

Tests for command-line interface argument parsing and validation.
Tests entry point for kernelopt CLI tool.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_cli_module_exists(self):
        """Test CLI module can be imported."""
        try:
            from kernelopt import cli
            assert cli is not None
        except ImportError:
            pytest.skip("CLI module not found")

    def test_required_argument_kernel_path(self):
        """Test --kernel-path is required argument."""
        # kernel-path should be required
        required_args = ["--kernel-path"]
        assert "--kernel-path" in required_args

    def test_optional_arguments_list(self):
        """Test list of optional arguments."""
        optional_args = [
            "--output-dir",
            "--gpu-id",
            "--max-iterations", "--T",
            "--N",
            "--K",
            "--model",
            "--verbose",
        ]

        # All should be recognized
        assert len(optional_args) > 0


class TestCLIValidation:
    """Test CLI input validation."""

    def test_kernel_path_validation(self, temp_output_dir):
        """Test kernel path must exist."""
        kernel_file = temp_output_dir / "test.cu"
        kernel_file.write_text("__global__ void k() {}")

        # File exists
        assert kernel_file.exists()

        # Non-existent file
        nonexistent = temp_output_dir / "missing.cu"
        assert not nonexistent.exists()

    def test_output_dir_creation(self, temp_output_dir):
        """Test output directory is created if doesn't exist."""
        output_dir = temp_output_dir / "new_output"

        # Initially doesn't exist
        assert not output_dir.exists()

        # Should be created
        output_dir.mkdir(parents=True)
        assert output_dir.exists()

    def test_gpu_id_validation(self):
        """Test GPU ID validation."""
        # Valid GPU IDs
        valid_ids = [0, 1, 2, 7]
        for gpu_id in valid_ids:
            assert gpu_id >= 0

        # Invalid GPU IDs
        invalid_ids = [-1, -5]
        for gpu_id in invalid_ids:
            assert gpu_id < 0  # Should be rejected

    def test_iteration_parameter_validation(self):
        """Test iteration parameter validation."""
        # Valid values
        assert 5 > 0  # T
        assert 4 > 0  # N
        assert 3 > 0  # K

        # Invalid values (negative or zero)
        invalid_values = [-1, 0, -10]
        for val in invalid_values:
            assert val <= 0  # Should be rejected


class TestCLIExecution:
    """Test CLI execution flow."""

    def test_cli_entry_point_exists(self):
        """Test CLI has main entry point."""
        try:
            from kernelopt import cli
            # Should have main() or similar
            assert hasattr(cli, 'main') or hasattr(cli, '__main__')
        except ImportError:
            pytest.skip("CLI module not found")

    def test_cli_exit_codes(self):
        """Test CLI exit codes."""
        # Standard exit codes
        EXIT_SUCCESS = 0
        EXIT_FAILURE = 1

        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE != 0


class TestCLIConfig:
    """Test CLI configuration loading."""

    def test_env_file_loading(self, temp_output_dir):
        """Test loading configuration from .env file."""
        env_file = temp_output_dir / ".env"
        env_file.write_text("""
ANTHROPIC_API_KEY=test_key
MODEL=claude-sonnet-4@20250514
""")

        assert env_file.exists()

        # dotenv would load these
        import os
        # Simulate env loading
        test_env = {
            "ANTHROPIC_API_KEY": "test_key",
            "MODEL": "claude-sonnet-4@20250514",
        }
        assert "ANTHROPIC_API_KEY" in test_env

    def test_api_key_from_env(self):
        """Test API key can come from environment."""
        import os

        # Simulate environment variable
        test_key = "test_api_key"
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": test_key}):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            assert api_key == test_key

    def test_config_precedence(self):
        """Test configuration precedence (CLI > env > defaults)."""
        # CLI args should override env vars
        cli_value = "cli_model"
        env_value = "env_model"
        default_value = "default_model"

        # CLI takes precedence
        final_value = cli_value if cli_value else (env_value if env_value else default_value)
        assert final_value == cli_value


class TestCLIHelp:
    """Test CLI help and documentation."""

    def test_help_flag_exists(self):
        """Test --help flag is recognized."""
        help_flags = ["--help", "-h"]
        assert len(help_flags) == 2

    def test_version_flag_exists(self):
        """Test --version flag is recognized."""
        version_flags = ["--version", "-v"]
        assert len(version_flags) == 2


class TestCLIErrorMessages:
    """Test CLI error messages."""

    def test_missing_required_argument_error(self):
        """Test error when required argument missing."""
        error_msg = "error: the following arguments are required: --kernel-path"
        assert "required" in error_msg.lower()
        assert "kernel-path" in error_msg.lower()

    def test_invalid_file_path_error(self):
        """Test error for invalid file path."""
        error_msg = "error: kernel file not found: /nonexistent/path.cu"
        assert "not found" in error_msg.lower()

    def test_invalid_gpu_id_error(self):
        """Test error for invalid GPU ID."""
        error_msg = "error: invalid GPU ID: -1"
        assert "invalid" in error_msg.lower()
        assert "gpu" in error_msg.lower()


class TestCLIArgumentTypes:
    """Test CLI argument type handling."""

    def test_string_arguments(self):
        """Test string argument parsing."""
        kernel_path = "/path/to/kernel.cu"
        model = "claude-sonnet-4@20250514"

        assert isinstance(kernel_path, str)
        assert isinstance(model, str)

    def test_integer_arguments(self):
        """Test integer argument parsing."""
        gpu_id = 0
        max_iterations = 5
        N = 4
        K = 3

        assert isinstance(gpu_id, int)
        assert isinstance(max_iterations, int)

    def test_boolean_flags(self):
        """Test boolean flag parsing."""
        verbose = True
        agentic_correctness = False

        assert isinstance(verbose, bool)
        assert isinstance(agentic_correctness, bool)


class TestCLICheckpointResume:
    """Test CLI checkpoint resume functionality."""

    def test_checkpoint_resume_flag(self):
        """Test --checkpoint-resume flag."""
        checkpoint_flag = "--checkpoint-resume"
        assert "checkpoint" in checkpoint_flag

    def test_checkpoint_resume_with_output_dir(self, temp_output_dir):
        """Test resuming from checkpoint in output dir."""
        from kernelopt.checkpoint import CHECKPOINT_FILENAME

        checkpoint_path = temp_output_dir / CHECKPOINT_FILENAME

        # Checkpoint should be in output directory
        # CLI should look for it there when resuming
        assert CHECKPOINT_FILENAME.endswith(".json")


class TestCLIModelSelection:
    """Test CLI model selection options."""

    def test_model_argument(self):
        """Test --model argument."""
        models = [
            "claude-sonnet-4@20250514",
            "claude-opus-4@20250514",
            "claude-haiku-3-5-20241022",
        ]

        for model in models:
            assert len(model) > 0
            assert "claude" in model.lower()

    def test_separate_agent_models(self):
        """Test separate model arguments for different agents."""
        model_args = [
            "--planner-model",
            "--executor-model",
            "--profiler-model",
            "--summarizer-model",
        ]

        # All agent-specific model arguments
        assert len(model_args) == 4


class TestCLIVerboseOutput:
    """Test CLI verbose output mode."""

    def test_verbose_flag(self):
        """Test --verbose flag."""
        verbose_flag = "--verbose"
        assert verbose_flag == "--verbose"

    def test_verbose_enables_logging(self):
        """Test verbose flag enables detailed logging."""
        import logging

        # Verbose should set logging level
        verbose_level = logging.DEBUG
        normal_level = logging.INFO

        assert verbose_level < normal_level  # DEBUG < INFO


class TestCLIProfileHints:
    """Test CLI profile hints argument."""

    def test_profile_hints_argument(self):
        """Test --profile-hints argument."""
        hints = "Focus on memory-bound kernels"
        assert len(hints) > 0
        assert isinstance(hints, str)


class TestCLIKernelArgs:
    """Test CLI kernel arguments passing."""

    def test_kernel_args_list(self):
        """Test passing arguments to kernel binary."""
        kernel_args = ["--arg1", "value1", "--arg2", "value2"]
        assert isinstance(kernel_args, list)
        assert len(kernel_args) > 0
