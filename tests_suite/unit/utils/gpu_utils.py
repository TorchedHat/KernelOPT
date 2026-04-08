"""
GPU utility functions for testing on H200 GPUs.

This module provides:
- GPU availability checking
- GPU resource monitoring (memory, utilization)
- GPU selection for tests
- H200-specific optimizations
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

import torch


class GPUResourceError(Exception):
    """Raised when GPU resources are insufficient."""
    pass


def get_gpu_info(gpu_id: int = 0) -> dict:
    """
    Get detailed GPU information using nvidia-smi.

    Args:
        gpu_id: GPU device ID

    Returns:
        Dictionary with GPU metrics
    """
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        parts = result.stdout.strip().split(", ")

        return {
            "name": parts[0],
            "memory_total_mb": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_free_mb": float(parts[3]),
            "gpu_utilization_pct": float(parts[4]),
            "memory_utilization_pct": float(parts[5]),
            "temperature_c": float(parts[6]) if len(parts) > 6 else 0.0,
        }
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError) as e:
        # Fallback to PyTorch if nvidia-smi fails
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_id)
            mem_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**2)
            mem_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**2)
            total_mem = props.total_memory / (1024**2)

            return {
                "name": props.name,
                "memory_total_mb": total_mem,
                "memory_used_mb": mem_reserved,
                "memory_free_mb": total_mem - mem_reserved,
                "gpu_utilization_pct": 0.0,  # Not available from PyTorch
                "memory_utilization_pct": (mem_reserved / total_mem) * 100,
                "temperature_c": 0.0,
            }
        raise RuntimeError(f"Cannot get GPU info for GPU {gpu_id}: {e}")


def get_gpu_memory_free_percent(gpu_id: int = 0) -> float:
    """
    Get percentage of free GPU memory.

    Args:
        gpu_id: GPU device ID

    Returns:
        Percentage of free memory (0-100)
    """
    info = get_gpu_info(gpu_id)
    return (info["memory_free_mb"] / info["memory_total_mb"]) * 100


def is_gpu_free(gpu_id: int = 0, min_free_memory_pct: float = 40.0, max_utilization_pct: float = 20.0) -> bool:
    """
    Check if GPU is free enough for testing.

    Args:
        gpu_id: GPU device ID
        min_free_memory_pct: Minimum free memory percentage required (default: 40%)
        max_utilization_pct: Maximum allowed GPU utilization (default: 20%)

    Returns:
        True if GPU meets the criteria
    """
    try:
        info = get_gpu_info(gpu_id)

        free_memory_pct = (info["memory_free_mb"] / info["memory_total_mb"]) * 100
        gpu_util = info["gpu_utilization_pct"]

        is_free = (free_memory_pct >= min_free_memory_pct and
                   gpu_util <= max_utilization_pct)

        if not is_free:
            print(f"\nGPU {gpu_id} status:")
            print(f"  Free memory: {free_memory_pct:.1f}% (need ≥{min_free_memory_pct}%)")
            print(f"  GPU utilization: {gpu_util:.1f}% (need ≤{max_utilization_pct}%)")

        return is_free
    except Exception as e:
        print(f"Warning: Cannot check GPU {gpu_id} status: {e}")
        return False


def wait_for_gpu_free(gpu_id: int = 0, min_free_memory_pct: float = 40.0,
                      max_utilization_pct: float = 20.0, timeout: float = 300.0) -> bool:
    """
    Wait for GPU to become free.

    Args:
        gpu_id: GPU device ID
        min_free_memory_pct: Minimum free memory percentage
        max_utilization_pct: Maximum GPU utilization
        timeout: Maximum wait time in seconds

    Returns:
        True if GPU became free, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_gpu_free(gpu_id, min_free_memory_pct, max_utilization_pct):
            return True

        time.sleep(5)

    return False


def list_available_gpus() -> list[int]:
    """
    List all available GPU IDs.

    Returns:
        List of GPU device IDs
    """
    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def select_best_gpu(min_free_memory_pct: float = 40.0) -> Optional[int]:
    """
    Select the GPU with most free memory.

    Args:
        min_free_memory_pct: Minimum free memory percentage required

    Returns:
        GPU ID with most free memory, or None if no suitable GPU found
    """
    gpus = list_available_gpus()

    if not gpus:
        return None

    best_gpu = None
    max_free_memory = 0.0

    for gpu_id in gpus:
        try:
            info = get_gpu_info(gpu_id)
            free_memory_pct = (info["memory_free_mb"] / info["memory_total_mb"]) * 100

            if free_memory_pct >= min_free_memory_pct and free_memory_pct > max_free_memory:
                max_free_memory = free_memory_pct
                best_gpu = gpu_id
        except Exception:
            continue

    return best_gpu


def check_gpu_for_profiling(gpu_id: int = 0) -> None:
    """
    Check if GPU has sufficient resources for profiling tests.

    For H200 GPUs, we require:
    - At least 40% free memory (for NCU profiling overhead)
    - Less than 20% GPU utilization

    Args:
        gpu_id: GPU device ID

    Raises:
        GPUResourceError: If GPU doesn't meet requirements
    """
    if not torch.cuda.is_available():
        raise GPUResourceError("CUDA is not available")

    if gpu_id >= torch.cuda.device_count():
        raise GPUResourceError(f"GPU {gpu_id} not found (available: {torch.cuda.device_count()})")

    info = get_gpu_info(gpu_id)
    free_memory_pct = (info["memory_free_mb"] / info["memory_total_mb"]) * 100

    print(f"\nGPU {gpu_id} ({info['name']}):")
    print(f"  Total memory: {info['memory_total_mb']:.0f} MB")
    print(f"  Used memory: {info['memory_used_mb']:.0f} MB ({100 - free_memory_pct:.1f}%)")
    print(f"  Free memory: {info['memory_free_mb']:.0f} MB ({free_memory_pct:.1f}%)")
    print(f"  GPU utilization: {info['gpu_utilization_pct']:.1f}%")
    if info['temperature_c'] > 0:
        print(f"  Temperature: {info['temperature_c']:.0f}°C")

    # For H200 profiling, require at least 40% free memory
    MIN_FREE_MEMORY_PCT = 40.0
    MAX_GPU_UTIL_PCT = 20.0

    if free_memory_pct < MIN_FREE_MEMORY_PCT:
        raise GPUResourceError(
            f"GPU {gpu_id} has insufficient free memory: {free_memory_pct:.1f}% "
            f"(need ≥{MIN_FREE_MEMORY_PCT}% for profiling)"
        )

    if info['gpu_utilization_pct'] > MAX_GPU_UTIL_PCT:
        raise GPUResourceError(
            f"GPU {gpu_id} is too busy: {info['gpu_utilization_pct']:.1f}% utilization "
            f"(need ≤{MAX_GPU_UTIL_PCT}% for accurate profiling)"
        )

    print(f"✓ GPU {gpu_id} has sufficient resources for profiling\n")


def is_h200_gpu(gpu_id: int = 0) -> bool:
    """
    Check if the GPU is an H200.

    Args:
        gpu_id: GPU device ID

    Returns:
        True if GPU is H200
    """
    try:
        info = get_gpu_info(gpu_id)
        return "H200" in info["name"].upper()
    except Exception:
        return False


def get_h200_specific_settings() -> dict:
    """
    Get H200-specific test settings.

    Returns:
        Dictionary with H200-optimized settings
    """
    return {
        "min_free_memory_pct": 40.0,  # H200 has 141GB, be conservative with NCU
        "max_utilization_pct": 20.0,  # Low utilization for accurate profiling
        "warmup_iterations": 10,      # More warmup for H200
        "benchmark_iterations": 100,   # More iterations for stable results
        "recommended_block_sizes": [128, 256, 512, 1024],
        "recommended_num_warps": [4, 8, 16],
    }


def print_gpu_summary():
    """Print summary of all available GPUs."""
    gpus = list_available_gpus()

    if not gpus:
        print("No CUDA GPUs available")
        return

    print(f"\n{'='*70}")
    print(f"Available GPUs: {len(gpus)}")
    print(f"{'='*70}")

    for gpu_id in gpus:
        try:
            info = get_gpu_info(gpu_id)
            free_pct = (info["memory_free_mb"] / info["memory_total_mb"]) * 100

            status = "✓ FREE" if free_pct >= 40.0 and info['gpu_utilization_pct'] <= 20.0 else "✗ BUSY"
            h200_marker = " [H200]" if "H200" in info["name"].upper() else ""

            print(f"\nGPU {gpu_id}: {info['name']}{h200_marker} - {status}")
            print(f"  Memory: {info['memory_free_mb']:.0f} MB free / {info['memory_total_mb']:.0f} MB total ({free_pct:.1f}% free)")
            print(f"  Utilization: {info['gpu_utilization_pct']:.1f}% GPU, {info['memory_utilization_pct']:.1f}% Memory")
        except Exception as e:
            print(f"\nGPU {gpu_id}: Error getting info - {e}")

    print(f"\n{'='*70}\n")


def clear_gpu_memory(gpu_id: int = 0):
    """
    Clear GPU memory cache.

    Args:
        gpu_id: GPU device ID
    """
    if torch.cuda.is_available():
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
