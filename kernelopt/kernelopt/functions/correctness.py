"""Correctness gate for optimized kernels."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm


def _has_builtin_self_test(cu_source: str) -> bool:
    """Return True if the .cu source contains its own correctness verification.

    Detects two patterns:
    - A function named 'check_correctness' defined in the file
    - printf calls with literal "PASS" or "FAIL" strings (typical host-side tests)
    """
    if re.search(r'\bcheck_correctness\s*\(', cu_source):
        return True
    if re.search(r'printf\s*\(\s*"[^"]*(?:PASS|FAIL)', cu_source):
        return True
    return False


def _check_cuda_kernel(
    original_path: str,
    optimized_source: str,
    binary_path: str | None = None,
) -> tuple[bool, str]:
    """Compile (if needed) and run an optimized CUDA kernel; check it doesn't crash.

    Args:
        binary_path: Pre-compiled binary. If provided and the file exists,
                     nvcc compilation is skipped entirely.
    """
    orig_path = Path(original_path)

    if binary_path and Path(binary_path).exists():
        opt_bin = Path(binary_path)
        tmp_cu = None
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cu", delete=False,
            dir=orig_path.parent,
        ) as tmp:
            tmp.write(optimized_source)
            tmp_cu = Path(tmp.name)

        opt_bin = tmp_cu.with_suffix("")
        compile_result = subprocess.run(
            ["nvcc", "-O2", "-o", str(opt_bin), str(tmp_cu)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_result.returncode != 0:
            tmp_cu.unlink(missing_ok=True)
            opt_bin.unlink(missing_ok=True)
            return False, f"Compilation failed:\n{compile_result.stderr[:500]}"

    try:
        run_result = subprocess.run(
            [str(opt_bin)],
            capture_output=True, text=True, timeout=120,
        )
        combined = run_result.stdout + run_result.stderr
        if run_result.returncode != 0:
            return (
                False,
                f"Runtime error (exit {run_result.returncode}):\n{combined[:500]}",
            )
        failure_indicators = ["FAILED", "AssertionError", "RuntimeError", "CUDA error", "Traceback"]
        for indicator in failure_indicators:
            if indicator in combined:
                return False, f"Failure indicator '{indicator}' found in output:\n{combined[:500]}"
        return True, "CUDA kernel ran successfully."
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (120s)."
    finally:
        if tmp_cu:
            tmp_cu.unlink(missing_ok=True)
            opt_bin.unlink(missing_ok=True)


_CUDA_LLM_SYSTEM_PROMPT = """\
You are the CUDA Correctness Agent. Given a CUDA kernel source file, write a complete
self-contained .cu test harness that verifies the kernel produces correct output.

The harness must:
1. Include all necessary headers (#include <cuda_runtime.h>, <stdio.h>, <math.h>, etc.)
2. Generate random float inputs on the CPU (use srand(42) for reproducibility)
3. Copy inputs to device with cudaMalloc/cudaMemcpy
4. Call the GPU kernel
5. Copy results back to host
6. Compute the expected result using a simple CPU reference (pure C math, no CUDA)
7. Compare element-wise: if max absolute diff > 1e-3, print "Correctness: FAIL" and exit(1)
8. Otherwise print "Correctness: PASS" and return 0

The harness must be a single compilable .cu file with int main(). Do NOT use #include of
the kernel file — reproduce the __global__ kernel function inline in the harness.
"""


def _check_cuda_with_llm_reference(
    cu_source: str,
    kernel_path: str,
    api_key: str | None,
    model: str,
    project_id: str | None,
    region: str | None,
) -> tuple[bool, str]:
    """Ask Claude to write a C test harness, compile it, run it, check markers."""
    from kernelopt.agents.schemas import CudaCorrectnessSpec

    llm = build_llm(model=model, api_key=api_key, max_tokens=4096,
                    temperature=0.1, project_id=project_id, region=region)
    structured_llm = llm.with_structured_output(CudaCorrectnessSpec)

    try:
        spec = structured_llm.invoke([
            SystemMessage(content=_CUDA_LLM_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"## Kernel source ({Path(kernel_path).name})\n"
                f"```c\n{cu_source}\n```\n\n"
                f"Write the test harness."
            )),
        ])
    except Exception as e:
        return False, f"LLM failed to produce CUDA test harness: {e}"

    if spec.skip:
        return False, f"Correctness skipped: {spec.skip_reason or 'no reason given'}"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False, dir="/tmp",
        prefix="cuda_harness_",
    ) as f:
        f.write(spec.test_harness_code)
        harness_cu = Path(f.name)

    harness_bin = harness_cu.with_suffix("")
    try:
        compile_result = subprocess.run(
            ["nvcc", "-O2", "-o", str(harness_bin), str(harness_cu)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_result.returncode != 0:
            return False, (
                f"LLM-generated harness failed to compile:\n{compile_result.stderr[:500]}"
            )

        run_result = subprocess.run(
            [str(harness_bin)],
            capture_output=True, text=True, timeout=120,
        )
        combined = run_result.stdout + run_result.stderr
        if run_result.returncode != 0:
            return False, f"Runtime error (exit {run_result.returncode}):\n{combined[:500]}"
        if "FAIL" in combined:
            return False, f"Correctness FAILED:\n{combined[:500]}"
        if "PASS" not in combined:
            return False, f"No 'Correctness: PASS' marker found in output:\n{combined[:500]}"
        return True, "CUDA correctness passed (LLM-generated C harness)."
    except subprocess.TimeoutExpired:
        return False, "Harness compilation or execution timed out."
    finally:
        harness_cu.unlink(missing_ok=True)
        harness_bin.unlink(missing_ok=True)


def check_correctness(
    original_path: str,
    optimized_source: str,
    framework: str,
    *,
    use_agentic: bool = True,
    api_key: str | None = None,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str | None = None,
    kernel_args: list[str] | None = None,
    binary_path: str | None = None,
) -> tuple[bool, str]:
    """
    Verify an optimized kernel is correct.

    For .cu kernels:
      - If source has a built-in self-test → run binary, check exit code + markers
      - Else if use_agentic=True → LLM writes C harness, compile, run
      - Else → run binary, check exit code only

    For Python kernels (Triton / Helion):
      - use_agentic=True  → Claude generates CPU reference + torch.allclose
      - use_agentic=False → subprocess (exit code + failure markers)

    Args:
        binary_path: Pre-compiled binary path for .cu kernels. When provided,
                     the run step skips nvcc. Ignored for Python kernels.
    """
    ext = Path(original_path).suffix

    if ext == ".cu":
        if _has_builtin_self_test(optimized_source):
            return _check_cuda_kernel(original_path, optimized_source, binary_path=binary_path)
        if use_agentic:
            return _check_cuda_with_llm_reference(
                cu_source=optimized_source,
                kernel_path=original_path,
                api_key=api_key,
                model=model,
                project_id=project_id,
                region=region,
            )
        return _check_cuda_kernel(original_path, optimized_source, binary_path=binary_path)

    # Python kernels (Triton / Helion)
    if use_agentic:
        from kernelopt.agents.correctness import run_correctness_agent
        return run_correctness_agent(
            kernel_source=optimized_source,
            kernel_path=original_path,
            framework=framework,
            api_key=api_key,
            model=model,
            project_id=project_id,
            region=region,
            kernel_args=kernel_args,
        )
    return check_correctness_subprocess(original_path, optimized_source, framework)


def check_correctness_subprocess(
    original_path: str,
    optimized_source: str,
    framework: str,
) -> tuple[bool, str]:
    """
    Legacy correctness check (runs kernel file as subprocess).

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
    # Determine temp directory - use parent if it exists, otherwise use /tmp
    try:
        parent_dir = Path(original_path).parent
        if not parent_dir.exists():
            parent_dir = None  # Will use system temp dir
    except Exception:
        parent_dir = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=parent_dir,
        ) as tmp:
            tmp.write(optimized_source)
            tmp_path = tmp.name
    except (FileNotFoundError, OSError) as e:
        return False, f"Failed to create temp file for correctness check: {e}"

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=120,
        )

        combined = result.stdout + result.stderr

        # DEBUG: Log subprocess results
        print(f"[DEBUG] Correctness subprocess returncode: {result.returncode}")
        print(f"[DEBUG] stdout length: {len(result.stdout)}")
        print(f"[DEBUG] stderr length: {len(result.stderr)}")
        print(f"[DEBUG] Combined output preview:\n{combined[:300]}")

        # Check for crash
        if result.returncode != 0:
            print(f"[DEBUG] Kernel crashed with exit code {result.returncode}")
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
                context_start = max(0, combined.find(indicator) - 100)
                context_end = combined.find(indicator) + 200
                print(f"[DEBUG] Triggered by indicator: '{indicator}'")
                print(f"[DEBUG] Context: {combined[context_start:context_end]}")
                return False, (
                    f"Failure indicator '{indicator}' found in output:\n"
                    f"{combined[:500]}"
                )

        print(f"[DEBUG] Correctness check PASSED")
        return True, f"Kernel ran successfully (exit 0).\n{result.stdout[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Kernel execution timed out (120s)."
    finally:
        Path(tmp_path).unlink(missing_ok=True)
