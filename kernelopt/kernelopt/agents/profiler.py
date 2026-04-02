"""Profiler agent: run NCU, parse CSV, build profiling_context."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from kernelopt.functions.csv_parser import parse_ncu_csv
from kernelopt.agents.gpu_specs import get_gpu_specs


def extract_kernel_info(kernel_path: str) -> tuple[str, list[str]]:
    """
    Parse the kernel source file with AST (Python) or regex (CUDA) to detect
    the framework and extract the names of all JIT/GPU kernel functions.

    Returns:
        (framework, kernel_names)
        framework:    "triton" | "helion" | "cuda" | "unknown"
        kernel_names: list of function names suitable for NCU --kernel-name

    For Python files, walks the AST looking for:
      @triton.jit or @tl.jit       → Triton kernel
      @helion.kernel or @hl.kernel → Helion kernel

    For .cu files, uses regex to find __global__ function names.

    kernel_names can be passed directly to build_ncu_command():
      1 name  → used as-is
      N names → joined as "regex:(name1|name2|...)"
      0 names → no --kernel-name flag (NCU profiles all kernels)
    """
    path = Path(kernel_path)
    ext = path.suffix.lower()

    if ext == ".py":
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return "unknown", []
        return _extract_python_kernels(tree)

    elif ext == ".cu":
        try:
            source = path.read_text()
        except OSError:
            return "unknown", []
        return _extract_cuda_kernels(source)

    return "unknown", []


def _extract_python_kernels(tree: ast.AST) -> tuple[str, list[str]]:
    """Walk a Python AST and return (framework, [kernel_function_names])."""
    triton_kernels: list[str] = []
    helion_kernels: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            # @triton.jit  or  @tl.jit
            if isinstance(dec, ast.Attribute) and dec.attr == "jit":
                if isinstance(dec.value, ast.Name) and dec.value.id in ("triton", "tl"):
                    triton_kernels.append(node.name)
            # @helion.kernel  or  @hl.kernel
            elif isinstance(dec, ast.Attribute) and dec.attr == "kernel":
                if isinstance(dec.value, ast.Name) and dec.value.id in ("helion", "hl"):
                    helion_kernels.append(node.name)

    if triton_kernels:
        return "triton", triton_kernels
    if helion_kernels:
        return "helion", helion_kernels
    return "unknown", []


def _extract_cuda_kernels(source: str) -> tuple[str, list[str]]:
    """Extract __global__ kernel function names from CUDA source."""
    names = re.findall(r'__global__\s+[\w\s*]+?\s+(\w+)\s*[(<]', source)
    if names:
        return "cuda", names
    return "unknown", []


def _detect_framework_from_name(kernel_name: str) -> str:
    """Fallback: detect framework from the NCU demangled kernel name string."""
    if any(p in kernel_name for p in ["triton_", "__triton", "triton::"]):
        return "triton"
    if any(p in kernel_name for p in ["helion_", "helion::"]):
        return "helion"
    if any(p in kernel_name for p in ["at::", "pycuda_"]):
        return "cuda"
    if kernel_name.startswith("jit_"):
        return "cuda"
    if "__global__" in kernel_name or re.search(r"void\s+\w+<", kernel_name):
        return "cuda"
    return "unknown"


def _build_kernel_name_filter(kernel_names: list[str]) -> str | None:
    """Convert a list of kernel names into an NCU --kernel-name argument."""
    if not kernel_names:
        return None
    if len(kernel_names) == 1:
        return kernel_names[0]
    return "regex:(" + "|".join(kernel_names) + ")"


def build_ncu_command(
    kernel_path: str,
    output_basename: str,
    *,
    kernel_name: str | None = None,
    launch_skip: int = 0,
    launch_count: int | None = None,
    replay_mode: str = "kernel",
    ncu_extra_args: list[str] | None = None,
    kernel_args: list[str] | None = None,
) -> list[str]:
    """
    Build the ncu command line for profiling.

    Two-step approach:
      Step 1: Profile → writes {output_basename}.ncu-rep (no --csv)
      Step 2: Export  → ncu --import {ncu-rep} --csv > {output_basename}.csv

    kernel_name:
      - If provided explicitly, used as-is (supports "regex:..." syntax).
      - If None, auto-detected from the kernel source via AST/regex.
      - If detection finds nothing, no --kernel-name flag is added (NCU
        profiles all kernels and the first one is used).

    replay_mode:
      - "kernel" (default): replays only the kernel, lower overhead but
        higher peak GPU memory (NCU saves/restores GPU memory state).
      - "application": replays the full program, lower peak memory but
        slower. Use for large inputs that cause OOM with kernel replay.
    """
    import sys

    ext = Path(kernel_path).suffix
    if ext == ".py":
        runner = [sys.executable, kernel_path]
    else:
        runner = [kernel_path.replace(ext, "")]

    # Auto-detect kernel name from source when not explicitly provided
    if kernel_name is None:
        _, detected_names = extract_kernel_info(kernel_path)
        kernel_name = _build_kernel_name_filter(detected_names)

    cmd = [
        "ncu",
        "--set", "full",
        "--replay-mode", replay_mode,
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
    ]

    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])

    cmd.extend(["--launch-skip", str(launch_skip)])

    if launch_count is not None:
        cmd.extend(["--launch-count", str(launch_count)])

    cmd.extend(["-o", output_basename, "-f"])

    if ncu_extra_args:
        cmd.extend(ncu_extra_args)

    cmd.extend(runner)

    if kernel_args:
        cmd.extend(kernel_args)

    return cmd


def _export_csv_from_ncu_rep(ncu_rep_path: str, csv_path: str) -> None:
    """Export CSV from an existing .ncu-rep file using ncu --import --csv."""
    import os

    # Clean environment - NCU uses Python internally and can conflict with virtualenv
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    env.pop('VIRTUAL_ENV_PROMPT', None)
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONSTARTUP', None)

    # Remove venv from PATH
    if 'PATH' in env:
        path_parts = env['PATH'].split(':')
        path_parts = [p for p in path_parts if '.venv' not in p]
        env['PATH'] = ':'.join(path_parts)

    cmd = ["ncu", "--import", ncu_rep_path, "--csv"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"NCU CSV export failed (exit {result.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:500]}"
        )
    Path(csv_path).write_text(result.stdout)


def run_ncu(
    kernel_path: str,
    output_basename: str,
    *,
    kernel_name: str | None = None,
    launch_skip: int = 0,
    launch_count: int | None = None,
    replay_mode: str = "kernel",
    ncu_extra_args: list[str] | None = None,
    kernel_args: list[str] | None = None,
    kernel_id: int = 0,
    save_profiling_context: bool = True,
) -> tuple[str, str, str]:
    """
    Execute NCU profiling in three steps:
      1. Profile the kernel → .ncu-rep
      2. Export CSV from .ncu-rep → .csv
      3. Build and save profiling_context.json → profiling context

    Returns (csv_path, ncu_rep_path, profiling_context_path).
    """
    # Step 1: Profile
    # Inherit the full environment so the kernel script can import torch/triton.
    # Set both PYTHONIOENCODING and PYTHONUTF8 to prevent NCU's frozen Python
    # from hitting "LookupError: unknown encoding: utf-8-sig" inside
    # site.addpackage() when it opens .pth files with the locale encoding.
    import os
    prof_env = os.environ.copy()
    prof_env["PYTHONIOENCODING"] = "utf-8"
    prof_env["PYTHONUTF8"] = "1"

    cmd = build_ncu_command(
        kernel_path,
        output_basename,
        kernel_name=kernel_name,
        launch_skip=launch_skip,
        launch_count=launch_count,
        replay_mode=replay_mode,
        ncu_extra_args=ncu_extra_args,
        kernel_args=kernel_args,
    )
    # Let NCU's stdout stream to the terminal (shows profiling progress).
    # Only capture stderr so we have it for error messages on failure.
    result = subprocess.run(cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=600, env=prof_env)

    ncu_rep_path = f"{output_basename}.ncu-rep"

    if result.returncode != 0:
        raise RuntimeError(
            f"NCU profiling failed (exit {result.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:1000]}"
        )
    csv_path = f"{output_basename}.csv"

    # Step 2: Export CSV from .ncu-rep
    _export_csv_from_ncu_rep(ncu_rep_path, csv_path)

    # Step 3: Build and save profiling_context.json (profiler agent's primary output)
    profiling_context_path = f"{output_basename}_profiling_context.json"
    if save_profiling_context:
        from kernelopt.functions.csv_parser import parse_ncu_csv
        import json
        sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
        ctx = build_profiling_context(sections, rules, launch, kernel_path, kernel_id)
        Path(profiling_context_path).write_text(json.dumps(ctx, indent=2))

    return csv_path, ncu_rep_path, profiling_context_path


def _extract_short_name(full_name: str) -> str:
    """Extract a short readable name from the innermost function name
    before the first < or (."""
    m = re.search(r"(?:void\s+)?(\w+)\s*[<(]", full_name)
    if m:
        return m.group(1)
    m = re.match(r"(\w+)", full_name)
    return m.group(1) if m else full_name[:50]


def _to_int(val, default=None):
    """Convert a value to int if possible."""
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
    """
    Assemble the profiling_context dict from parsed CSV data.

    Output matches the skill spec schema:
      kernel_name, kernel_name_full, kernel_id, framework_hint,
      device, compute_capability, launch{...}, sections{...}, rules[...]
    """
    kernel_name_full = launch.get("kernel_name", "")
    kernel_name = _extract_short_name(kernel_name_full)

    # Detect framework from source (AST) — reliable for any kernel name.
    # Falls back to demangled name pattern matching only if source unreadable.
    framework_hint, _ = extract_kernel_info(kernel_path)
    if framework_hint == "unknown":
        framework_hint = _detect_framework_from_name(kernel_name_full)

    cc = str(launch.get("cc", ""))
    device = str(launch.get("device", ""))

    # Build launch config from Launch Statistics section
    launch_stats = sections.get("Launch Statistics", {})

    block_size = launch.get("block_size", 0)
    grid_size = launch.get("grid_size", 0)

    regs = _to_int(
        launch_stats.get("Registers Per Thread", {}).get("value")
    )
    # NCU uses "# SMs", not "SM Count"
    sm_count = _to_int(
        launch_stats.get("# SMs", {}).get("value")
    )
    dyn_smem = _to_int(
        launch_stats.get("Dynamic Shared Memory Per Block", {}).get("value")
    )
    static_smem = _to_int(
        launch_stats.get("Static Shared Memory Per Block", {}).get("value")
    )
    threads = _to_int(
        launch_stats.get("Threads", {}).get("value")
    )

    # Extract waves_per_sm from CSV (more accurate than calculating from grid/sm_count)
    waves_per_sm_raw = launch_stats.get("Waves Per SM", {}).get("value")
    waves_per_sm = None
    if waves_per_sm_raw is not None:
        try:
            waves_per_sm = float(waves_per_sm_raw)
        except (ValueError, TypeError):
            # Fallback to calculation if parse fails
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

    # Get GPU hardware specs for roofline analysis
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

    # Normalize to nanoseconds
    if unit in ("us", "usecond", "microsecond"):
        dur *= 1000.0
    elif unit in ("ms", "msecond", "millisecond"):
        dur *= 1_000_000.0
    elif unit in ("s", "second"):
        dur *= 1_000_000_000.0

    return dur
