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
    framework: "triton" | "helion" | "cuda" | "pytorch" | "unknown"
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
        framework, names = _extract_python_kernels(tree)
        if framework != "unknown":
            return framework, names
        if _is_inductor_output(source):
            return "triton", []
        if _is_pytorch_source(source, tree):
            return "pytorch", []
        return "unknown", []

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


def _is_inductor_output(source: str) -> bool:
    """Detect torch.compile / Inductor-generated output.

    These files have Triton kernels compiled via ``async_compile.triton(...)``
    rather than module-level ``@triton.jit`` decorators.
    """
    return "async_compile.triton(" in source or "async_compile = AsyncCompile()" in source


def _is_pytorch_source(source: str, tree: ast.AST) -> bool:
    """Detect whether a Python file is a PyTorch module/function (not already Triton/Helion).

    Uses the same AST-based detection as the synthesis system so that
    framework identification and extraction always agree.
    """
    from kernelopt.synthesis.compile_extract import _detect_callable

    return _detect_callable(source) is not None


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
    binary_path: str | None = None,
    ncu_set: str = "full",
    clock_control: str | None = None,
) -> list[str]:
    """Build the ncu command line for profiling.

    Args:
        binary_path: For .cu kernels, path to the compiled binary. If None, derived from kernel_path.
        ncu_set: NCU metric set — "full" for optimization, "basic" for duration-only.
        clock_control: NCU clock control — None for default (locked), "none" for boost clocks.
    """
    ext = Path(kernel_path).suffix
    if ext == ".py":
        runner = [sys.executable, kernel_path]
    else:
        # Use provided binary_path or derive from kernel_path
        if binary_path:
            runner = [binary_path]
        else:
            runner = [kernel_path.replace(ext, "")]

    # kernel_name semantics:
    #   None: auto-detect from source (for backward compat)
    #   "": explicitly no filter (profile all kernels - for Triton/Helion)
    #   "foo": use this exact filter
    if kernel_name is None:
        _, detected_names = extract_kernel_info(kernel_path)
        kernel_name = _build_kernel_name_filter(detected_names)

    cmd = [
        "ncu",
        "--set", ncu_set,
        "--replay-mode", replay_mode,
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
    ]

    if clock_control is not None:
        cmd.extend(["--clock-control", clock_control])

    # Only add --kernel-name if we have a non-empty filter
    if kernel_name:
        # Ensure substring matching. NCU requires the "regex:" prefix for regex
        # patterns; without it the match is exact. Wrap bare names so they match
        # demangled CUDA names like "void softmax_kernel<1024>(float*, ...)".
        if not kernel_name.startswith("regex:"):
            kernel_name = f"regex:.*{kernel_name}.*"
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
    binary_path: str | None = None,
) -> tuple[str, str, str]:
    """Execute NCU profiling: profile → .ncu-rep → export CSV → optionally build context JSON.

    Args:
        binary_path: For .cu kernels, path to the compiled binary. If None, derived from kernel_path.

    Returns (csv_path, ncu_rep_path, profiling_context_path).
    """
    prof_env = os.environ.copy()
    prof_env["PYTHONIOENCODING"] = "utf-8"
    prof_env["PYTHONUTF8"] = "1"

    # Strip PyTorch source tree from PYTHONPATH. NCU application replay launches
    # the script as a subprocess; if PYTHONPATH points at an unbuilt pytorch source
    # directory, Python loads the source torch/__init__.py but the compiled .so
    # from the venv — causing ImportError on symbols not yet compiled in.
    if "PYTHONPATH" in prof_env:
        clean = [
            p for p in prof_env["PYTHONPATH"].split(":")
            if not (p.endswith("/pytorch") or "/pytorch/torch" in p)
        ]
        if clean:
            prof_env["PYTHONPATH"] = ":".join(clean)
        else:
            del prof_env["PYTHONPATH"]

    cmd = build_ncu_command(
        kernel_path,
        output_basename,
        kernel_name=kernel_name,
        launch_skip=launch_skip,
        launch_count=launch_count,
        replay_mode=replay_mode,
        ncu_extra_args=ncu_extra_args,
        kernel_args=kernel_args,
        binary_path=binary_path,
    )
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200, env=prof_env
    )
    if result.stdout:
        from kernelopt.pipeline_ui import emit_profiler_output

        emit_profiler_output(result.stdout)

    ncu_rep_path = f"{output_basename}.ncu-rep"

    if result.returncode != 0:
        raise RuntimeError(
            f"NCU profiling failed (exit {result.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:1000]}"
        )

    # Detect "No kernels were profiled" — NCU exits 0 but the .ncu-rep is empty.
    # This usually means the --kernel-name filter didn't match anything.
    # Auto-retry without the filter so we always capture something.
    if "No kernels were profiled" in result.stdout:
        has_kernel_filter = any(a == "--kernel-name" for a in cmd)
        if has_kernel_filter:
            kernel_filter = next(
                cmd[i + 1] for i, a in enumerate(cmd) if a == "--kernel-name"
            )
            from kernelopt.pipeline_ui import emit_profiler_output

            emit_profiler_output(
                f"[profiler] --kernel-name {kernel_filter!r} matched 0 kernels — "
                f"retrying without filter to capture all kernels.\n"
            )
            # Strip --kernel-name and its value from the command and retry.
            retry_cmd = [
                a for i, a in enumerate(cmd)
                if a != "--kernel-name" and (i == 0 or cmd[i - 1] != "--kernel-name")
            ]
            result = subprocess.run(
                retry_cmd, capture_output=True, text=True, timeout=1200, env=prof_env
            )
            if result.stdout:
                emit_profiler_output(result.stdout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"NCU profiling failed on retry (exit {result.returncode}):\n"
                    f"Command: {' '.join(retry_cmd)}\n"
                    f"stderr: {result.stderr[:1000]}"
                )
            if "No kernels were profiled" in result.stdout:
                raise RuntimeError(
                    f"NCU profiled 0 kernels even without a --kernel-name filter.\n"
                    f"NCU output:\n{result.stdout[:2000]}"
                )
        else:
            raise RuntimeError(
                f"NCU profiled 0 kernels (no filter was applied).\n"
                f"NCU output:\n{result.stdout[:2000]}"
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


def run_ncu_boost(
    wrapper_script_path: str,
    output_dir: str,
) -> float:
    """Run NCU with --clock-control none to get total kernel duration at boost clocks.

    Used for Amdahl's analysis: the returned T_kernel is directly comparable
    to wall-clock measurements (same GPU clock state, no lock).

    The wrapper_script_path should be a Python script that:
      1. Imports and warms up the kernel (3+ calls)
      2. Wraps the measured call in torch.cuda.nvtx.range_push("measure") / range_pop()

    Uses --set basic (duration only) and --nvtx-include "measure/" to isolate
    the profiled call from warmup/JIT/init.

    Returns total kernel duration in nanoseconds (sum of all GPU kernels in the
    measured range), or -1.0 on failure.
    """
    from kernelopt.functions.csv_parser import sum_all_kernel_durations

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ncu_base = str(Path(output_dir) / "boost_profile")
    ncu_rep = f"{ncu_base}.ncu-rep"
    ncu_csv = f"{ncu_base}.csv"

    cmd = build_ncu_command(
        wrapper_script_path,
        ncu_base,
        kernel_name="",  # no filter — capture all kernels
        ncu_set="basic",
        clock_control="none",
        replay_mode="kernel",
        ncu_extra_args=["--nvtx", "--nvtx-include", "measure/"],
    )

    prof_env = os.environ.copy()
    prof_env["PYTHONIOENCODING"] = "utf-8"
    prof_env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, env=prof_env,
    )
    if result.returncode != 0:
        return -1.0
    if "No kernels were profiled" in result.stdout:
        return -1.0

    try:
        _export_csv_from_ncu_rep(ncu_rep, ncu_csv)
        total_ns = sum_all_kernel_durations(ncu_csv)
        return total_ns if total_ns > 0 else -1.0
    except Exception:
        return -1.0
    finally:
        for f in [ncu_rep, ncu_csv]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
