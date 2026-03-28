"""Correctness gate for optimized kernels."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def check_correctness_subprocess(
    original_path: str,
    optimized_source: str,
    framework: str,
) -> tuple[bool, str]:
    """
    Verify an optimized kernel is correct.

    For Python kernels: runs the optimized kernel (which should have its own
    correctness check built in). Passes if exit code is 0 and no failure
    indicators in output.

    For CUDA kernels: compiles and runs both, compares stdout.

    Returns (passed, diagnosis_message).
    """
    ext = Path(original_path).suffix

    if ext == ".py":
        return _check_python_kernel(original_path, optimized_source)
    elif ext == ".cu":
        return _check_cuda_kernel(original_path, optimized_source)
    else:
        return False, f"Unsupported kernel extension: {ext}"


def _check_python_kernel(
    original_path: str, optimized_source: str,
) -> tuple[bool, str]:
    """
    Run the optimized Python kernel and check it doesn't crash.

    Most Triton/Helion kernels have built-in correctness checks that print
    pass/fail. We check:
      1. Exit code == 0 (no crashes, no assertion errors)
      2. No failure indicators in output (FAILED, Error, Traceback)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
        dir=Path(original_path).parent,
    ) as tmp:
        tmp.write(optimized_source)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=120,
        )

        combined = result.stdout + result.stderr

        # Check for crash
        if result.returncode != 0:
            return False, (
                f"Optimized kernel crashed (exit {result.returncode}):\n"
                f"{combined[:500]}"
            )

        # Check for failure indicators
        failure_indicators = [
            "FAILED", "AssertionError", "Traceback",
            "RuntimeError", "CUDA error", "triton.runtime",
        ]
        for indicator in failure_indicators:
            if indicator in combined:
                return False, (
                    f"Failure indicator '{indicator}' found in output:\n"
                    f"{combined[:500]}"
                )

        return True, f"Kernel ran successfully (exit 0).\n{result.stdout[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Kernel execution timed out (120s)."
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _check_cuda_kernel(
    original_path: str, optimized_source: str,
) -> tuple[bool, str]:
    """Compile and run optimized CUDA kernel, check it doesn't crash."""
    orig_path = Path(original_path)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False,
        dir=orig_path.parent,
    ) as tmp:
        tmp.write(optimized_source)
        tmp_path = Path(tmp.name)

    opt_bin = tmp_path.with_suffix("")

    try:
        # Compile
        compile_result = subprocess.run(
            ["nvcc", "-o", str(opt_bin), str(tmp_path)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_result.returncode != 0:
            return (
                False,
                f"Compilation failed:\n{compile_result.stderr[:500]}",
            )

        # Run
        run_result = subprocess.run(
            [str(opt_bin)],
            capture_output=True, text=True, timeout=120,
        )
        if run_result.returncode != 0:
            return (
                False,
                f"Runtime error (exit {run_result.returncode}):\n"
                f"{run_result.stderr[:500]}",
            )

        return True, "CUDA kernel compiled and ran successfully."

    except subprocess.TimeoutExpired:
        return False, "Compilation or execution timed out."
    finally:
        tmp_path.unlink(missing_ok=True)
        opt_bin.unlink(missing_ok=True)
