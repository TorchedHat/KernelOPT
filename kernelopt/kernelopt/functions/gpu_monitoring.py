"""GPU monitoring utilities for kernel profiling pipeline.

Provides:
- Pre-run GPU validation and selection
- Per-iteration GPU activity monitoring
- Non-blocking warnings about GPU contention
"""

from __future__ import annotations

import os
import subprocess

import torch


class GPUMonitoringError(Exception):
    """Raised when GPU monitoring fails (non-fatal)."""
    pass


def get_system_memory() -> dict:
    """Get system (CPU) memory information.

    Returns:
        Dictionary with system memory metrics:
        {
            "total_mb": float,
            "available_mb": float,
            "used_mb": float,
            "free_mb": float,
        }

    Raises:
        GPUMonitoringError: If memory info cannot be read
    """
    try:
        # Read /proc/meminfo (works on all Linux systems)
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    # Parse value (usually in kB)
                    value_str = parts[1].strip().split()[0]
                    meminfo[key] = int(value_str) / 1024  # Convert kB to MB

        total_mb = meminfo.get("MemTotal", 0)
        available_mb = meminfo.get("MemAvailable", 0)
        free_mb = meminfo.get("MemFree", 0)
        used_mb = total_mb - available_mb

        return {
            "total_mb": total_mb,
            "available_mb": available_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
        }
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise GPUMonitoringError(f"Cannot read system memory: {e}")


def get_gpu_metrics(gpu_id: int = 0) -> dict:
    """Get current GPU metrics via nvidia-smi.

    Args:
        gpu_id: GPU device ID to query

    Returns:
        Dictionary with GPU metrics:
        {
            "name": str,
            "memory_total_mb": float,
            "memory_used_mb": float,
            "memory_free_mb": float,
            "utilization_pct": float,
            "temperature_c": float,
        }

    Raises:
        GPUMonitoringError: If nvidia-smi fails and PyTorch fallback
            unavailable
    """
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=name,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]

        return {
            "name": parts[0],
            "memory_total_mb": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_free_mb": float(parts[3]),
            "utilization_pct": float(parts[4]),
            "temperature_c": float(parts[5]) if len(parts) > 5 else 0.0,
        }
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        IndexError,
        FileNotFoundError
    ) as e:
        # Fallback to PyTorch if nvidia-smi unavailable
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_id)
            total_mem = props.total_memory / (1024**2)
            used_mem = torch.cuda.memory_reserved(gpu_id) / (1024**2)
            return {
                "name": props.name,
                "memory_total_mb": total_mem,
                "memory_used_mb": used_mem,
                "memory_free_mb": total_mem - used_mem,
                "utilization_pct": 0.0,  # Not available from PyTorch
                "temperature_c": 0.0,
            }
        raise GPUMonitoringError(f"Cannot query GPU {gpu_id}: {e}")


def get_gpu_processes(gpu_id: int = 0) -> list[dict]:
    """Get list of compute processes running on GPU.

    Args:
        gpu_id: GPU device ID to query

    Returns:
        List of process dictionaries:
        [{"pid": int, "name": str, "memory_mb": float}, ...]

    Raises:
        GPUMonitoringError: If query fails
    """
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        processes = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                processes.append({
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "memory_mb": float(parts[2]),
                })
        return processes

    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        FileNotFoundError
    ) as e:
        raise GPUMonitoringError(f"Cannot query GPU processes: {e}")


def get_gpu_processes_detailed(gpu_id: int = 0) -> list[dict]:
    """Get detailed process information including user and CPU usage.

    Args:
        gpu_id: GPU device ID to query

    Returns:
        List of detailed process dictionaries:
        [{
            "pid": int,
            "user": str,
            "gpu_memory_mb": float,
            "cpu_percent": float
        }, ...]

    Raises:
        GPUMonitoringError: If query fails
    """
    try:
        # First get GPU processes
        base_processes = get_gpu_processes(gpu_id)
        if not base_processes:
            return []

        # Then get detailed info for each PID
        detailed_processes = []

        for proc in base_processes:
            pid = proc["pid"]
            user = "host"  # Default for processes outside container
            cpu_percent = 0.0

            # Use ps command directly
            ps_paths = ["/usr/bin/ps", "/bin/ps"]
            ps_worked = False
            for ps_path in ps_paths:
                try:
                    # Use ps to get user and CPU% directly
                    ps_result = subprocess.run(
                        [ps_path, "-o", "user=,pcpu=", "-p", str(pid)],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=1
                    )
                    lines = ps_result.stdout.strip().split("\n")
                    if lines and lines[0].strip():
                        # Parse output (format: "username cpu%")
                        parts = lines[0].strip().split(None, 1)
                        if len(parts) >= 1:
                            user = parts[0]
                        if len(parts) >= 2:
                            try:
                                cpu_percent = float(parts[1])
                            except ValueError:
                                pass
                        ps_worked = True
                    break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue

            # Fallback: Try /proc if ps didn't work
            if not ps_worked:
                try:
                    import os
                    import pwd
                    stat_info = os.stat(f"/proc/{pid}")
                    uid = stat_info.st_uid
                    try:
                        user = pwd.getpwuid(uid).pw_name
                    except KeyError:
                        user = str(uid)
                except (FileNotFoundError, PermissionError, OSError):
                    # Process not in container namespace
                    user = "host"

            detailed_processes.append({
                "pid": pid,
                "user": user,
                "gpu_memory_mb": proc["memory_mb"],
                "cpu_percent": cpu_percent,
            })

        return detailed_processes

    except GPUMonitoringError:
        raise
    except Exception as e:
        raise GPUMonitoringError(
            f"Cannot get detailed process info: {e}"
        )


def check_ps_available() -> bool:
    """Check if ps command is available in the system.

    Returns:
        True if ps is available, False otherwise
    """
    ps_paths = ["/usr/bin/ps", "/bin/ps"]
    for ps_path in ps_paths:
        try:
            subprocess.run(
                [ps_path, "--version"],
                capture_output=True,
                check=True,
                timeout=1
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return False


def install_procps_ng() -> bool:
    """Try to install procps-ng package (provides ps command).

    Tries sudo first, then falls back to direct dnf.
    Non-fatal - returns False if installation fails.

    Returns:
        True if installation succeeded, False otherwise
    """
    install_commands = [
        ["sudo", "dnf", "install", "-y", "procps-ng"],
        ["dnf", "install", "-y", "procps-ng"],
    ]

    for cmd in install_commands:
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=30
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    return False


def select_gpu() -> int:
    """Select GPU device to use (defaults to 0, or CUDA_VISIBLE_DEVICES).

    Returns:
        GPU device ID
    """
    # Honor CUDA_VISIBLE_DEVICES if set
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "":
        try:
            # Get first GPU from comma-separated list
            return int(visible.split(",")[0])
        except ValueError:
            pass

    # Default to GPU 0
    return 0


def validate_gpu_for_profiling(
    gpu_id: int, *, console, _ps_available_cache: list = []
) -> dict:
    """Display GPU status for pre-run validation (non-blocking).

    Shows GPU device info, memory status, utilization, and temperature.
    Displays warnings if memory is low or GPU is busy, but never blocks
    execution.

    Args:
        gpu_id: GPU device to validate
        console: Rich console for output
        _ps_available_cache: Internal cache for ps availability check

    Returns:
        GPU metrics dict (empty dict if validation fails)
    """
    try:
        # Check ps availability once (cached)
        if not _ps_available_cache:
            ps_available = check_ps_available()

            # If ps not available, try to auto-install procps-ng
            if not ps_available:
                console.print(
                    "  [dim]Installing procps-ng for user info...[/]"
                )
                if install_procps_ng():
                    ps_available = check_ps_available()
                    if ps_available:
                        console.print("  [dim]✓ procps-ng installed[/]")

            _ps_available_cache.append(ps_available)
        ps_available = _ps_available_cache[0]

        metrics = get_gpu_metrics(gpu_id)
        processes_detailed = get_gpu_processes_detailed(gpu_id)

        # Get system memory info
        try:
            sys_mem = get_system_memory()
            sys_mem_available = True
        except GPUMonitoringError:
            sys_mem_available = False

        # GPU memory calculations
        gpu_free_pct = (
            metrics["memory_free_mb"] / metrics["memory_total_mb"]
        ) * 100

        console.print("\n[bold]System Status:[/]")

        # Display system (CPU) memory if available
        if sys_mem_available:
            sys_free_pct = (
                sys_mem["available_mb"] / sys_mem["total_mb"]
            ) * 100
            console.print(
                f"  CPU Memory (RAM): {sys_mem['available_mb']:.0f} MB "
                f"available / {sys_mem['total_mb']:.0f} MB total "
                f"({sys_free_pct:.1f}% available)"
            )

        # Display GPU info
        console.print(
            f"  GPU Device: [cyan]GPU {gpu_id}[/] ({metrics['name']})\n"
            f"  GPU Memory (VRAM): {metrics['memory_free_mb']:.0f} MB free / "
            f"{metrics['memory_total_mb']:.0f} MB total "
            f"({gpu_free_pct:.1f}% free)\n"
            f"  GPU Utilization: [cyan]{metrics['utilization_pct']:.1f}%[/]"
        )

        if metrics["temperature_c"] > 0:
            console.print(
                f"  GPU Temperature: {metrics['temperature_c']:.0f}°C"
            )

        # Show active processes if any
        if len(processes_detailed) > 0:
            total_gpu_mem = sum(
                p["gpu_memory_mb"] for p in processes_detailed
            )
            console.print(
                f"  Active processes: "
                f"[yellow]{len(processes_detailed)}[/] "
                f"({total_gpu_mem:.0f} MB GPU memory)"
            )

        # Warnings (non-blocking)
        if gpu_free_pct < 40.0:
            console.print(
                f"  [yellow]Warning:[/] Low GPU memory "
                f"({gpu_free_pct:.1f}%) — profiling may fail or be inaccurate"
            )

        if metrics["utilization_pct"] > 20.0:
            console.print(
                f"  [yellow]Warning:[/] GPU is busy "
                f"({metrics['utilization_pct']:.1f}%) — "
                f"profiling may be affected by other workloads"
            )

        # Show ps install message if not available and processes exist
        if not ps_available and len(processes_detailed) > 0:
            console.print(
                f"  [dim]Tip: Install ps for user info: "
                f"sudo dnf install -y procps-ng[/]"
            )

        # Show detailed process table if processes exist
        if len(processes_detailed) > 0:
            from rich.table import Table

            # Check if any processes are in container (not "host")
            has_container_procs = any(
                p["user"] != "host" for p in processes_detailed
            )
            show_cpu = ps_available and has_container_procs

            console.print("")  # blank line before table
            table = Table(
                title="Active GPU Processes",
                show_header=True,
                header_style="bold cyan",
                box=None,
                padding=(0, 1),
            )

            # Conditionally add User column if ps is available
            if ps_available:
                table.add_column("User", style="cyan")
            table.add_column("PID", justify="right", style="dim")
            table.add_column("GPU Memory", justify="right")
            # Only show CPU% if we have container processes
            if show_cpu:
                table.add_column("CPU%", justify="right", style="dim")

            # Sort by GPU memory (descending)
            sorted_procs = sorted(
                processes_detailed,
                key=lambda p: p["gpu_memory_mb"],
                reverse=True
            )

            for proc in sorted_procs:
                if ps_available:
                    if show_cpu:
                        table.add_row(
                            f"{proc['user']}",
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                            f"{proc['cpu_percent']:.1f}",
                        )
                    else:
                        table.add_row(
                            f"{proc['user']}",
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                        )
                else:
                    if show_cpu:
                        table.add_row(
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                            f"{proc['cpu_percent']:.1f}",
                        )
                    else:
                        table.add_row(
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                        )

            console.print(table)

            # Explain "host" label if any processes show it
            if ps_available and any(
                p["user"] == "host" for p in processes_detailed
            ):
                console.print(
                    "  [dim]Note: 'host' = process running outside "
                    "container namespace[/]"
                )

        console.print("")  # blank line
        # Extra keys for HUD / callers (backward compatible with get_gpu_metrics shape)
        metrics["gpu_id"] = gpu_id
        metrics["active_process_count"] = len(processes_detailed)
        if sys_mem_available:
            metrics["sys_mem_total_mb"] = float(sys_mem["total_mb"])
            metrics["sys_mem_available_mb"] = float(sys_mem["available_mb"])
        return metrics

    except GPUMonitoringError as e:
        console.print(f"[yellow]Warning:[/] Cannot validate GPU: {e}\n")
        return {}


def monitor_gpu_activity(gpu_id: int, iteration: int, *, console) -> None:
    """Display GPU activity monitoring (software warnings, non-blocking).

    Shows warnings about GPU utilization and active processes if they exceed
    thresholds. Displays detailed process table when processes are active.
    Silent if GPU is idle (non-intrusive).

    Args:
        gpu_id: GPU device to monitor
        iteration: Current iteration number (for display)
        console: Rich console for output
    """
    try:
        # Check ps availability
        ps_available = check_ps_available()

        metrics = get_gpu_metrics(gpu_id)
        processes_detailed = get_gpu_processes_detailed(gpu_id)

        # Only show warnings if there's something noteworthy
        show_warning = (
            metrics["utilization_pct"] > 30.0 or len(processes_detailed) > 0
        )

        if not show_warning:
            return  # Silent when idle

        # Summary line
        warnings = []
        if metrics["utilization_pct"] > 30.0:
            warnings.append(
                f"GPU utilization: "
                f"[yellow]{metrics['utilization_pct']:.1f}%[/]"
            )

        if len(processes_detailed) > 0:
            total_gpu_mem = sum(p["gpu_memory_mb"] for p in processes_detailed)
            warnings.append(
                f"[yellow]{len(processes_detailed)}[/] processes "
                f"using [yellow]{total_gpu_mem:.0f} MB[/]"
            )

        console.print(f"  [dim]GPU activity:[/] " + " | ".join(warnings))

        # Detailed process table (always show if processes exist)
        if len(processes_detailed) > 0:
            from rich.table import Table

            # Check if any processes are in container (not "host")
            has_container_procs = any(
                p["user"] != "host" for p in processes_detailed
            )
            show_cpu = ps_available and has_container_procs

            table = Table(
                show_header=True,
                header_style="bold cyan",
                box=None,
                padding=(0, 1),
            )

            # Conditionally add User column if ps is available
            if ps_available:
                table.add_column("User", style="cyan")
            table.add_column("PID", justify="right", style="dim")
            table.add_column("GPU Mem", justify="right")
            # Only show CPU% if we have container processes
            if show_cpu:
                table.add_column("CPU%", justify="right", style="dim")

            # Sort by GPU memory (descending)
            sorted_procs = sorted(
                processes_detailed,
                key=lambda p: p["gpu_memory_mb"],
                reverse=True
            )

            for proc in sorted_procs:
                if ps_available:
                    if show_cpu:
                        table.add_row(
                            f"{proc['user']}",
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                            f"{proc['cpu_percent']:.1f}",
                        )
                    else:
                        table.add_row(
                            f"{proc['user']}",
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                        )
                else:
                    if show_cpu:
                        table.add_row(
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                            f"{proc['cpu_percent']:.1f}",
                        )
                    else:
                        table.add_row(
                            f"{proc['pid']}",
                            f"[yellow]{proc['gpu_memory_mb']:.0f} MB[/]",
                        )

            console.print(table)

            # Explain "host" label if any processes show it
            if ps_available and any(
                p["user"] == "host" for p in processes_detailed
            ):
                console.print(
                    "  [dim]Note: 'host' = process outside container[/]"
                )

            console.print("")  # blank line

    except GPUMonitoringError:
        # Silently ignore monitoring errors (non-critical)
        pass
