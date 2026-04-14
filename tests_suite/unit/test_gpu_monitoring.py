"""Unit tests for GPU monitoring utilities."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from kernelopt.functions.gpu_monitoring import (
    GPUMonitoringError,
    check_ps_available,
    get_gpu_metrics,
    get_gpu_processes,
    get_gpu_processes_detailed,
    get_system_memory,
    monitor_gpu_activity,
    select_gpu,
    validate_gpu_for_profiling,
)


class TestGetSystemMemory:
    """Tests for get_system_memory function."""

    def test_get_system_memory_success(self):
        """Test system memory retrieval from /proc/meminfo."""
        mock_meminfo = (
            "MemTotal:       32000000 kB\n"
            "MemFree:        10000000 kB\n"
            "MemAvailable:   20000000 kB\n"
            "Buffers:         1000000 kB\n"
        )
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = \
                iter(mock_meminfo.splitlines())

            mem = get_system_memory()

            assert mem["total_mb"] == 32000000 / 1024  # ~31250 MB
            assert mem["available_mb"] == 20000000 / 1024  # ~19531 MB
            assert mem["free_mb"] == 10000000 / 1024  # ~9765 MB
            assert mem["used_mb"] > 0

    def test_get_system_memory_error(self):
        """Test error handling when /proc/meminfo unavailable."""
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = FileNotFoundError("/proc/meminfo")

            with pytest.raises(GPUMonitoringError):
                get_system_memory()


class TestCheckPsAvailable:
    """Tests for check_ps_available function."""

    def test_check_ps_available_true(self):
        """Test ps availability check when ps exists."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_ps_available() is True

    def test_check_ps_available_false(self):
        """Test ps availability check when ps not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ps not found")
            assert check_ps_available() is False


class TestGetGPUMetrics:
    """Tests for get_gpu_metrics function."""

    def test_get_gpu_metrics_success(self):
        """Test GPU metrics retrieval with nvidia-smi."""
        mock_output = "NVIDIA H200, 141000, 20000, 121000, 5.2, 42"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                returncode=0
            )

            metrics = get_gpu_metrics(0)

            assert metrics["name"] == "NVIDIA H200"
            assert metrics["memory_total_mb"] == 141000.0
            assert metrics["memory_used_mb"] == 20000.0
            assert metrics["memory_free_mb"] == 121000.0
            assert metrics["utilization_pct"] == 5.2
            assert metrics["temperature_c"] == 42.0

    def test_get_gpu_metrics_pytorch_fallback(self):
        """Test fallback to PyTorch when nvidia-smi fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=1):
                    with patch("torch.cuda.get_device_properties") as mock_props:
                        # Create a proper mock object with name attribute
                        device_props = MagicMock()
                        device_props.name = "Test GPU"
                        device_props.total_memory = 141000 * 1024**2
                        mock_props.return_value = device_props

                        with patch("torch.cuda.memory_reserved", return_value=20000 * 1024**2):
                            metrics = get_gpu_metrics(0)

                            assert metrics["name"] == "Test GPU"
                            assert metrics["utilization_pct"] == 0.0
                            assert metrics["temperature_c"] == 0.0

    def test_get_gpu_metrics_invalid_gpu(self):
        """Test error handling for invalid GPU."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

            with patch("torch.cuda.is_available", return_value=False):
                with pytest.raises(GPUMonitoringError):
                    get_gpu_metrics(999)


class TestGetGPUProcesses:
    """Tests for get_gpu_processes function."""

    def test_get_gpu_processes_empty(self):
        """Test GPU process listing when no processes running."""
        mock_output = ""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                returncode=0
            )

            processes = get_gpu_processes(0)
            assert processes == []

    def test_get_gpu_processes_multiple(self):
        """Test GPU process listing with multiple processes."""
        mock_output = "12345, python, 1024\n67890, ncu, 2048\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                returncode=0
            )

            processes = get_gpu_processes(0)
            assert len(processes) == 2
            assert processes[0]["pid"] == 12345
            assert processes[0]["name"] == "python"
            assert processes[0]["memory_mb"] == 1024.0
            assert processes[1]["pid"] == 67890
            assert processes[1]["name"] == "ncu"

    def test_get_gpu_processes_error(self):
        """Test error handling when query fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

            with pytest.raises(GPUMonitoringError):
                get_gpu_processes(0)


class TestGetGPUProcessesDetailed:
    """Tests for get_gpu_processes_detailed function."""

    def test_get_gpu_processes_detailed_success(self):
        """Test detailed process info retrieval."""
        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_processes"
        ) as mock_get_procs:
            mock_get_procs.return_value = [
                {"pid": 12345, "name": "python", "memory_mb": 1024.0}
            ]

            # Mock ps subprocess call
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="testuser 15.5\n",
                    returncode=0
                )

                processes = get_gpu_processes_detailed(0)
                assert len(processes) == 1
                assert processes[0]["pid"] == 12345
                assert processes[0]["user"] == "testuser"
                assert processes[0]["gpu_memory_mb"] == 1024.0
                assert processes[0]["cpu_percent"] == 15.5

    def test_get_gpu_processes_detailed_empty(self):
        """Test detailed process info when no processes."""
        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_processes"
        ) as mock_get:
            mock_get.return_value = []
            processes = get_gpu_processes_detailed(0)
            assert processes == []


class TestSelectGPU:
    """Tests for select_gpu function."""

    def test_select_gpu_default(self):
        """Test GPU selection defaults to 0."""
        with patch.dict(os.environ, {}, clear=True):
            gpu_id = select_gpu()
            assert gpu_id == 0

    def test_select_gpu_from_env(self):
        """Test GPU selection from CUDA_VISIBLE_DEVICES."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True):
            gpu_id = select_gpu()
            assert gpu_id == 2

    def test_select_gpu_single_device(self):
        """Test GPU selection with single device in env."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True):
            gpu_id = select_gpu()
            assert gpu_id == 1


class TestValidateGPUForProfiling:
    """Tests for validate_gpu_for_profiling function."""

    def test_validate_gpu_success(self):
        """Test GPU validation with good metrics."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            with patch(
                "kernelopt.functions.gpu_monitoring."
                "get_gpu_processes_detailed"
            ) as mock_get_processes:
                with patch(
                    "kernelopt.functions.gpu_monitoring.get_system_memory"
                ) as mock_sys_mem:
                    mock_get_metrics.return_value = {
                        "name": "NVIDIA H200",
                        "memory_total_mb": 141000.0,
                        "memory_used_mb": 20000.0,
                        "memory_free_mb": 121000.0,
                        "utilization_pct": 5.2,
                        "temperature_c": 42.0,
                    }
                    mock_get_processes.return_value = []
                    mock_sys_mem.return_value = {
                        "total_mb": 32000.0,
                        "available_mb": 16000.0,
                        "used_mb": 16000.0,
                        "free_mb": 10000.0,
                    }

                    metrics = validate_gpu_for_profiling(
                        0, console=mock_console
                    )

                    assert metrics["name"] == "NVIDIA H200"
                    assert mock_console.print.called

    def test_validate_gpu_low_memory_warning(self):
        """Test GPU validation warns on low memory."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            with patch(
                "kernelopt.functions.gpu_monitoring."
                "get_gpu_processes_detailed"
            ) as mock_get_processes:
                with patch(
                    "kernelopt.functions.gpu_monitoring.get_system_memory"
                ) as mock_sys_mem:
                    mock_get_metrics.return_value = {
                        "name": "Test GPU",
                        "memory_total_mb": 100000.0,
                        "memory_used_mb": 70000.0,
                        "memory_free_mb": 30000.0,  # 30% free < 40%
                        "utilization_pct": 10.0,
                        "temperature_c": 50.0,
                    }
                    mock_get_processes.return_value = []
                    mock_sys_mem.return_value = {
                        "total_mb": 32000.0,
                        "available_mb": 16000.0,
                        "used_mb": 16000.0,
                        "free_mb": 10000.0,
                    }

                    validate_gpu_for_profiling(0, console=mock_console)

                    # Check that warning was printed
                    calls = [
                        str(call) for call in mock_console.print.call_args_list
                    ]
                    warning_found = any(
                        "Warning" in str(call) and "Low GPU memory" in str(call)
                        for call in calls
                    )
                    assert warning_found

    def test_validate_gpu_high_utilization_warning(self):
        """Test GPU validation warns on high utilization."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            with patch(
                "kernelopt.functions.gpu_monitoring."
                "get_gpu_processes_detailed"
            ) as mock_get_processes:
                with patch(
                    "kernelopt.functions.gpu_monitoring.get_system_memory"
                ) as mock_sys_mem:
                    mock_get_metrics.return_value = {
                        "name": "Test GPU",
                        "memory_total_mb": 100000.0,
                        "memory_used_mb": 20000.0,
                        "memory_free_mb": 80000.0,
                        "utilization_pct": 45.0,  # > 20%
                        "temperature_c": 60.0,
                    }
                    mock_get_processes.return_value = []
                    mock_sys_mem.return_value = {
                        "total_mb": 32000.0,
                        "available_mb": 16000.0,
                        "used_mb": 16000.0,
                        "free_mb": 10000.0,
                    }

                    validate_gpu_for_profiling(0, console=mock_console)

                    # Check that warning was printed
                    calls = [
                        str(call) for call in mock_console.print.call_args_list
                    ]
                    warning_found = any(
                        "Warning" in str(call) and "GPU is busy" in str(call)
                        for call in calls
                    )
                    assert warning_found

    def test_validate_gpu_error_handling(self):
        """Test GPU validation handles errors gracefully."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.side_effect = GPUMonitoringError("Test error")

            metrics = validate_gpu_for_profiling(0, console=mock_console)

            assert metrics == {}
            # Should print warning message
            assert mock_console.print.called


class TestMonitorGPUActivity:
    """Tests for monitor_gpu_activity function."""

    def test_monitor_gpu_idle(self):
        """Test monitoring when GPU is idle (no output)."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            with patch(
                "kernelopt.functions.gpu_monitoring."
                "get_gpu_processes_detailed"
            ) as mock_get_processes:
                mock_get_metrics.return_value = {
                    "utilization_pct": 5.0,  # Low utilization
                }
                mock_get_processes.return_value = []  # No processes

                monitor_gpu_activity(0, 1, console=mock_console)

                # Should not print anything when idle
                assert not mock_console.print.called

    def test_monitor_gpu_high_utilization(self):
        """Test monitoring displays warning on high utilization."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            with patch(
                "kernelopt.functions.gpu_monitoring."
                "get_gpu_processes_detailed"
            ) as mock_get_processes:
                mock_get_metrics.return_value = {
                    "utilization_pct": 45.0,  # > 30%
                }
                mock_get_processes.return_value = []

                monitor_gpu_activity(0, 1, console=mock_console)

                # Should print warning
                assert mock_console.print.called
                calls = [str(call) for call in mock_console.print.call_args_list]
                assert any("GPU activity" in str(call) for call in calls)

    def test_monitor_gpu_active_processes(self):
        """Test monitoring displays table with active processes."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            with patch(
                "kernelopt.functions.gpu_monitoring."
                "get_gpu_processes_detailed"
            ) as mock_get_processes:
                mock_get_metrics.return_value = {
                    "utilization_pct": 15.0,
                }
                mock_get_processes.return_value = [
                    {
                        "pid": 12345,
                        "user": "testuser",
                        "gpu_memory_mb": 1024.0,
                        "cpu_percent": 15.5,
                    }
                ]

                monitor_gpu_activity(0, 1, console=mock_console)

                # Should print warning and table
                assert mock_console.print.called
                calls = [str(call) for call in mock_console.print.call_args_list]
                assert any("processes" in str(call) for call in calls)

    def test_monitor_gpu_error_silent(self):
        """Test monitoring silently ignores errors."""
        mock_console = MagicMock()

        with patch(
            "kernelopt.functions.gpu_monitoring.get_gpu_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.side_effect = GPUMonitoringError("Test error")

            # Should not raise exception
            monitor_gpu_activity(0, 1, console=mock_console)

            # Should not print anything
            assert not mock_console.print.called
