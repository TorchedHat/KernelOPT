# kernelopt/kernelopt/agents/profiler/ncu_runner.py
"""NCU subprocess runner: profile a kernel, export CSV, save profiling context."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path


def extract_kernel_info(kernel_path: str) -> tuple[str, list[str]]:
    """Parse the kernel source to detect framework and list all GPU kernel function names.

    Returns (framework, kernel_names).
    framework: "triton" | "helion" | "cuda" | "unknown"
    kernel_names: suitable for --kernel-name flag (exact or regex pattern)
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
    triton_kernels: list[str] = []
    helion_kernels: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "jit":
                if isinstance(dec.value, ast.Name) and dec.value.id in ("triton", "tl"):
                    triton_kernels.append(node.name)
            elif isinstance(dec, ast.Attribute) and dec.attr == "kernel":
                if isinstance(dec.value, ast.Name) and dec.value.id in ("helion", "hl"):
                    helion_kernels.append(node.name)

    if triton_kernels:
        return "triton", triton_kernels
    if helion_kernels:
        return "helion", helion_kernels
    return "unknown", []


def _extract_cuda_kernels(source: str) -> tuple[str, list[str]]:
    names = re.findall(r'__global__\s+[\w\s*]+?\s+(\w+)\s*[(<]', source)
    if names:
        return "cuda", names
    return "unknown", []


def _detect_framework_from_name(kernel_name: str) -> str:
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
    """Build the ncu command line for profiling."""
    ext = Path(kernel_path).suffix
    if ext == ".py":
        runner = [sys.executable, kernel_path]
    else:
        runner = [kernel_path.replace(ext, "")]

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
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("VIRTUAL_ENV_PROMPT", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)

    if "PATH" in env:
        path_parts = env["PATH"].split(":")
        path_parts = [p for p in path_parts if ".venv" not in p]
        env["PATH"] = ":".join(path_parts)

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
    """Execute NCU profiling: profile → .ncu-rep → export CSV → optionally build context JSON.

    Returns (csv_path, ncu_rep_path, profiling_context_path).
    """
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
    result = subprocess.run(
        cmd, stdout=None, stderr=subprocess.PIPE, text=True, timeout=600, env=prof_env
    )

    ncu_rep_path = f"{output_basename}.ncu-rep"

    if result.returncode != 0:
        raise RuntimeError(
            f"NCU profiling failed (exit {result.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:1000]}"
        )

    csv_path = f"{output_basename}.csv"
    _export_csv_from_ncu_rep(ncu_rep_path, csv_path)

    profiling_context_path = f"{output_basename}_profiling_context.json"
    if save_profiling_context:
        import json
        from kernelopt.agents.profiler.context_builder import build_profiling_context
        from kernelopt.functions.csv_parser import parse_ncu_csv

        sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
        ctx = build_profiling_context(sections, rules, launch, kernel_path, kernel_id)
        Path(profiling_context_path).write_text(json.dumps(ctx, indent=2))

    return csv_path, ncu_rep_path, profiling_context_path
