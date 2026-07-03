# kernelopt/kernelopt/agents/profiler/context_builder.py
"""Build structured profiling_context from parsed NCU CSV data."""

from __future__ import annotations

import re

from kernelopt.agents.gpu_specs import get_gpu_specs
from kernelopt.agents.profiler.ncu_runner import (
    _detect_framework_from_name,
    extract_kernel_info,
)


def _extract_short_name(full_name: str) -> str:
    # Extract the base name (everything before angle-bracket template args or parens)
    m = re.search(r"(?:void\s+)?(\w+)\s*[<(]", full_name)
    if m:
        base = m.group(1)
    else:
        m2 = re.match(r"(\w+)", full_name)
        base = m2.group(1) if m2 else full_name[:50]

    # For Triton-style names like "triton__0d1d2d3_softmax_kernel", extract the
    # human-readable suffix by stripping leading hex-digest segments (e.g. "0d1d2d3_").
    # Pattern: remove leading "triton__<hex_digits>_" prefix if present.
    triton_m = re.match(r"triton__[0-9a-f]+_(.+)", base)
    if triton_m:
        return triton_m.group(1)

    return base


def _to_int(val, default=None):
    if val is None:
        return default
    try:
        return int(float(val)) if not isinstance(val, int) else val
    except (ValueError, TypeError):
        return default


def build_profiling_context(
    sections: dict,
    rules: list,
    launch: dict,
    kernel_path: str,
    kernel_id: int = 0,
) -> dict:
    """Assemble the profiling_context dict from parsed CSV data."""
    kernel_name_full = launch.get("kernel_name", "")
    kernel_name = _extract_short_name(kernel_name_full)

    framework_hint, _ = extract_kernel_info(kernel_path)
    if framework_hint == "unknown":
        framework_hint = _detect_framework_from_name(kernel_name_full)

    cc = str(launch.get("cc", ""))
    device = str(launch.get("device", ""))

    launch_stats = sections.get("Launch Statistics", {})
    block_size = launch.get("block_size", 0)
    grid_size = launch.get("grid_size", 0)

    regs = _to_int(launch_stats.get("Registers Per Thread", {}).get("value"))
    sm_count = _to_int(launch_stats.get("# SMs", {}).get("value"))
    dyn_smem = _to_int(launch_stats.get("Dynamic Shared Memory Per Block", {}).get("value"))
    static_smem = _to_int(launch_stats.get("Static Shared Memory Per Block", {}).get("value"))
    threads = _to_int(launch_stats.get("Threads", {}).get("value"))

    waves_per_sm_raw = launch_stats.get("Waves Per SM", {}).get("value")
    waves_per_sm = None
    if waves_per_sm_raw is not None:
        try:
            waves_per_sm = float(waves_per_sm_raw)
        except (ValueError, TypeError):
            if sm_count and sm_count > 0 and grid_size:
                waves_per_sm = round(grid_size / sm_count, 2)

    launch_config: dict = {
        "block_size": block_size,
        "grid_size": grid_size,
        "sm_count": sm_count,
        "waves_per_sm": waves_per_sm,
        "registers_per_thread": regs,
        "dynamic_smem_per_block_bytes": dyn_smem,
        "static_smem_per_block_bytes": static_smem,
        "threads": threads,
    }

    gpu_specs = get_gpu_specs(sm_count, cc)

    return {
        "kernel_name": kernel_name,
        "kernel_name_full": kernel_name_full,
        "kernel_id": kernel_id,
        "framework_hint": framework_hint,
        "device": device,
        "compute_capability": cc,
        "launch": launch_config,
        "gpu_specs": gpu_specs,
        "sections": sections,
        "rules": list(rules),
    }


def get_latency_ns(profiling_context: dict) -> float:
    """Extract Duration converted to nanoseconds from a profiling context."""
    sol = profiling_context.get("sections", {}).get(
        "GPU Speed Of Light Throughput", {}
    )
    dur_entry = sol.get("Duration", {})
    dur = dur_entry.get("value", float("inf"))
    unit = dur_entry.get("unit", "ns").strip().lower()

    if isinstance(dur, str):
        try:
            dur = float(dur.replace(",", ""))
        except ValueError:
            return float("inf")

    dur = float(dur)

    if unit in ("us", "usecond", "microsecond"):
        dur *= 1000.0
    elif unit in ("ms", "msecond", "millisecond"):
        dur *= 1_000_000.0
    elif unit in ("s", "second"):
        dur *= 1_000_000_000.0

    return dur
