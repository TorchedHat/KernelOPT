"""Correctness gate for optimized kernels."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def validate_kernel_runs(
    kernel_path: str,
    *,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Run a kernel with --profile-only to verify it doesn't crash.

    This is a pre-flight check before the expensive correctness agent
    or NCU profiling. Catches syntax errors, missing imports, invalid
    Triton APIs, wrong constexpr types, and runtime crashes.

    Returns (ok, error_message).
    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, kernel_path, "--profile-only"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode == 0:
            return True, ""
        stderr = proc.stderr[-1500:] if proc.stderr else ""
        stdout = proc.stdout[-500:] if proc.stdout else ""
        return False, f"exit {proc.returncode}\nstderr:\n{stderr}\nstdout:\n{stdout}"
    except subprocess.TimeoutExpired:
        return False, f"kernel timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def check_correctness_against_pytorch(
    original_pytorch_path: str,
    optimized_source: str,
    optimized_kernel_path: str,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    debug: bool = False,
) -> tuple[bool, str]:
    """Verify an optimized kernel against the original PyTorch program.

    This is the compiler-first correctness approach: the original PyTorch
    source IS the specification. We run both, compare outputs.

    Used for synthesized kernels where _original_pytorch_path is available.
    """
    import json

    script = f"""\
import json, sys, os, torch, importlib.util

torch.cuda.set_device(0)
seeds = [0, 42, 1337]
failures = []

for seed in seeds:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load original PyTorch module
    spec1 = importlib.util.spec_from_file_location("_orig", "{original_pytorch_path}")
    mod1 = importlib.util.module_from_spec(spec1)
    sys.modules["_orig"] = mod1
    spec1.loader.exec_module(mod1)

    # Load optimized kernel module
    spec2 = importlib.util.spec_from_file_location("_opt", "{optimized_kernel_path}")
    mod2 = importlib.util.module_from_spec(spec2)
    sys.modules["_opt"] = mod2
    spec2.loader.exec_module(mod2)

    # For compiler-first correctness: the original_pytorch_path is the
    # SYNTHESIZED Triton file (the unmodified Inductor output), not the
    # raw PyTorch source. Both reference and candidate use the same
    # Inductor argument format (kernel_function + get_inputs).
    #
    # If original_pytorch_path is actually a KernelBench .py with Model:
    # fall back to using the optimized file's own get_inputs for both,
    # since PyTorch Model inputs are incompatible with Inductor format.

    # Both reference (unmodified synthesized kernel) and candidate
    # (LLM-optimized kernel) use the same Inductor format:
    # kernel_function(*get_inputs()). Use the REFERENCE file's get_inputs
    # for both, since inputs must be identical for comparison.
    get_inputs = getattr(mod1, "get_inputs", None) or getattr(mod2, "get_inputs", None)
    if get_inputs is None:
        print(json.dumps({{"passed": False, "error": "no get_inputs found"}}))
        sys.exit(0)

    inputs = get_inputs()
    # Clone inputs so in-place kernels don't corrupt the comparison.
    ref_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
    opt_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]

    # Run reference kernel
    ref_fn = getattr(mod1, "kernel_function", None) or getattr(mod1, "run", None) or getattr(mod1, "call", None)
    if ref_fn is None:
        print(json.dumps({{"passed": False, "error": "no callable found in reference"}}))
        sys.exit(0)
    with torch.no_grad():
        ref_out = ref_fn(*ref_inputs)

    # Run optimized kernel with SAME (cloned) inputs
    opt_fn = getattr(mod2, "kernel_function", None) or getattr(mod2, "run", None) or getattr(mod2, "call", None)
    if opt_fn is None:
        print(json.dumps({{"passed": False, "error": "no kernel_function in optimized"}}))
        sys.exit(0)
    with torch.no_grad():
        opt_out = opt_fn(*opt_inputs)

    # Compare — handle multi-output models
    if not isinstance(ref_out, (tuple, list)):
        ref_out = [ref_out]
    if not isinstance(opt_out, (tuple, list)):
        opt_out = [opt_out]

    if len(ref_out) != len(opt_out):
        failures.append(f"seed={{seed}}: output count mismatch ref={{len(ref_out)}} opt={{len(opt_out)}}")
    else:
        for _oi, (_r, _o) in enumerate(zip(ref_out, opt_out)):
            if not isinstance(_r, torch.Tensor) or not isinstance(_o, torch.Tensor):
                continue
            ref_f = _r.float().cpu()
            opt_f = _o.float().cpu()

            if ref_f.shape != opt_f.shape:
                failures.append(f"seed={{seed}} output[{{_oi}}]: shape mismatch ref={{ref_f.shape}} opt={{opt_f.shape}}")
                continue

            # Handle NaN/Inf: both must produce NaN/Inf at the same positions
            ref_nan = torch.isnan(ref_f) | torch.isinf(ref_f)
            opt_nan = torch.isnan(opt_f) | torch.isinf(opt_f)
            if not torch.equal(ref_nan, opt_nan):
                mismatched = (ref_nan != opt_nan).sum().item()
                failures.append(f"seed={{seed}} output[{{_oi}}]: NaN/Inf position mismatch ({{mismatched}} elements)")
            else:
                finite_mask = ~ref_nan
                if finite_mask.any():
                    diff = (ref_f[finite_mask] - opt_f[finite_mask]).abs()
                    max_diff = diff.max().item()
                    passed = torch.allclose(ref_f[finite_mask], opt_f[finite_mask], atol={atol}, rtol={rtol})
                    if not passed:
                        failures.append(f"seed={{seed}} output[{{_oi}}]: max_diff={{max_diff:.3e}} > atol={atol}")
                # else: all NaN/Inf and positions match — that's fine

if failures:
    print(json.dumps({{"passed": False, "failures": failures}}))
else:
    print(json.dumps({{"passed": True, "message": "Correctness passed (3 seeds, PyTorch reference)"}}))
"""

    try:
        # Write optimized source to the expected path
        Path(optimized_kernel_path).write_text(optimized_source)

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0:
            import json as _json
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    result = _json.loads(line)
                    if result.get("passed"):
                        return True, result.get("message", "Passed")
                    failures = result.get("failures", [result.get("error", "unknown")])
                    return False, "Correctness FAILED:\n" + "\n".join(failures)
        return False, f"Correctness subprocess failed (exit {proc.returncode}): {(proc.stderr or '')[-300:]}"
    except subprocess.TimeoutExpired:
        return False, "Correctness check timed out (300s)"
    except Exception as e:
        return False, f"Correctness check error: {e}"


def check_model_correctness(
    original_pytorch_path: str,
    restitched_source: str,
    restitched_path: str,
    synthesized_path: str | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple[bool, str]:
    """Verify re-stitched compiled model against original compiled model.

    Runs the unmodified synthesized_triton.py and the re-stitched version
    with the same Inductor-format inputs (from get_inputs()). If outputs
    match, the kernel optimization preserved model-level correctness.

    We compare compiled-vs-compiled rather than eager-vs-compiled because
    the Inductor arg ordering (interleaved params + inputs) is determined
    during compilation and can't be reproduced from the eager model's
    state_dict without re-running torch.compile.
    """
    import json

    if not synthesized_path:
        return False, "No synthesized_path provided for model correctness check"

    script = f"""\
import json, sys, os, torch, importlib.util

torch.cuda.set_device(0)
seeds = [0, 42]
failures = []

for seed in seeds:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load original compiled model (unmodified synthesized_triton.py)
    spec1 = importlib.util.spec_from_file_location("_orig_" + str(seed), "{synthesized_path}")
    mod1 = importlib.util.module_from_spec(spec1)
    sys.modules["_orig_" + str(seed)] = mod1
    spec1.loader.exec_module(mod1)

    # Load restitched compiled model
    spec2 = importlib.util.spec_from_file_location("_rs_" + str(seed), "{restitched_path}")
    mod2 = importlib.util.module_from_spec(spec2)
    sys.modules["_rs_" + str(seed)] = mod2
    spec2.loader.exec_module(mod2)

    # Use the ORIGINAL model's get_inputs for both
    get_inputs = getattr(mod1, "get_inputs", None)
    if get_inputs is None:
        print(json.dumps({{"passed": False, "error": "no get_inputs in synthesized file"}}))
        sys.exit(0)

    inputs = get_inputs()
    orig_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
    rs_inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]

    # Run original compiled model
    orig_fn = getattr(mod1, "kernel_function", None) or getattr(mod1, "call", None)
    with torch.no_grad():
        orig_out = orig_fn(*orig_inputs)

    # Run restitched model
    rs_fn = getattr(mod2, "kernel_function", None) or getattr(mod2, "call", None)
    with torch.no_grad():
        rs_out = rs_fn(*rs_inputs)

    # Normalize outputs to lists for multi-output comparison
    if not isinstance(orig_out, (tuple, list)):
        orig_out = [orig_out]
    if not isinstance(rs_out, (tuple, list)):
        rs_out = [rs_out]

    if len(orig_out) != len(rs_out):
        failures.append(f"seed={{seed}}: output count mismatch orig={{len(orig_out)}} rs={{len(rs_out)}}")
        continue

    for _oi, (_orig_t, _rs_t) in enumerate(zip(orig_out, rs_out)):
        if not isinstance(_orig_t, torch.Tensor) or not isinstance(_rs_t, torch.Tensor):
            continue
        orig_f = _orig_t.float().cpu()
        rs_f = _rs_t.float().cpu()

        if orig_f.shape != rs_f.shape:
            failures.append(f"seed={{seed}} output[{{_oi}}]: shape mismatch orig={{list(orig_f.shape)}} vs restitched={{list(rs_f.shape)}}")
            continue

        # Handle NaN/Inf positions
        orig_nan = torch.isnan(orig_f) | torch.isinf(orig_f)
        rs_nan = torch.isnan(rs_f) | torch.isinf(rs_f)
        if not torch.equal(orig_nan, rs_nan):
            mismatched = (orig_nan != rs_nan).sum().item()
            failures.append(f"seed={{seed}} output[{{_oi}}]: NaN/Inf position mismatch ({{mismatched}} elements)")
            continue
        finite_mask = ~orig_nan
        if not finite_mask.any():
            continue
        max_diff = (orig_f[finite_mask] - rs_f[finite_mask]).abs().max().item()
        passed = torch.allclose(orig_f[finite_mask], rs_f[finite_mask], atol={atol}, rtol={rtol})
        if not passed:
            failures.append(f"seed={{seed}} output[{{_oi}}]: max_diff={{max_diff:.3e}} > atol={atol}")

if failures:
    print(json.dumps({{"passed": False, "failures": failures}}))
else:
    print(json.dumps({{"passed": True, "message": "Model correctness passed (compiled vs restitched)"}}))
"""

    try:
        Path(restitched_path).write_text(restitched_source)

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode == 0:
            import json as _json
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    result = _json.loads(line)
                    if result.get("passed"):
                        return True, result.get("message", "Passed")
                    failures = result.get("failures", [result.get("error", "unknown")])
                    return False, "Model correctness FAILED:\n" + "\n".join(failures)
        stderr_tail = (proc.stderr or "")[-500:]
        return False, f"Model correctness subprocess failed (exit {proc.returncode}): {stderr_tail}"
    except subprocess.TimeoutExpired:
        return False, "Model correctness check timed out (180s)"
    except Exception as e:
        return False, f"Model correctness check error: {e}"


def check_correctness(
    original_path: str,
    optimized_source: str,
    framework: str,
    *,
    use_agentic: bool = True,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    project_id: str | None = None,
    region: str | None = None,
    kernel_args: list[str] | None = None,
    original_pytorch_path: str | None = None,
    debug: bool = False,
    **llm_kwargs: object,
) -> tuple[bool, str]:
    """
    Verify an optimized kernel is correct.

    For Python kernels (Triton / Helion):
      - use_agentic=True  → Claude generates CPU reference + torch.allclose
      - use_agentic=False → subprocess (exit code + failure markers)
    """
    # Python kernels (Triton / Helion)
    # Compiler-first correctness: if we have the original PyTorch source,
    # use it as ground truth instead of asking the LLM to generate a reference.
    if original_pytorch_path and Path(original_pytorch_path).exists():
        opt_tmp = Path(tempfile.mktemp(suffix=".py", prefix="kpt_correctness_"))
        return check_correctness_against_pytorch(
            original_pytorch_path=original_pytorch_path,
            optimized_source=optimized_source,
            optimized_kernel_path=str(opt_tmp),
            debug=debug,
        )

    # No PyTorch reference available: subprocess-only checking cannot verify
    # numerical correctness, so always use the agentic path for actual
    # output comparison, even when use_agentic=False.
    if not use_agentic:
        import logging
        logging.getLogger(__name__).warning(
            "No PyTorch reference path available and use_agentic=False. "
            "Using LLM-generated reference for actual output comparison."
        )
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
        **llm_kwargs,
    )


def check_correctness_subprocess(
    original_path: str,
    optimized_source: str,
    debug: bool = False,
) -> tuple[bool, str]:
    """
    Legacy correctness check (runs kernel file as subprocess).

    For Python kernels: runs the optimized kernel (which should have its own
    correctness check built in). Passes if exit code is 0 and no failure
    indicators in output.

    Returns (passed, diagnosis_message).
    """
    return _check_python_kernel(original_path, optimized_source, debug=debug)


def _check_python_kernel(
    original_path: str, optimized_source: str, debug: bool = False,
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

        if debug:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.debug("Correctness subprocess returncode: %d", result.returncode)
            _logger.debug("stdout length: %d", len(result.stdout))
            _logger.debug("stderr length: %d", len(result.stderr))
            _logger.debug("Combined output preview:\n%s", combined[:300])

        # Check for crash
        if result.returncode != 0:
            if debug:
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
                if debug:
                    print(f"[DEBUG] Triggered by indicator: '{indicator}'")
                    print(f"[DEBUG] Context: {combined[context_start:context_end]}")
                return False, (
                    f"Failure indicator '{indicator}' found in output:\n"
                    f"{combined[:500]}"
                )

        if debug:
            print("[DEBUG] Correctness check PASSED")
        return True, f"Kernel ran successfully (exit 0).\n{result.stdout[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Kernel execution timed out (120s)."
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def time_model(
    model_path: str,
    warmup: int = 5,
    repeats: int = 20,
) -> tuple[float, str]:
    """Time a synthesized_triton.py model end-to-end.

    Returns (median_ms, message).  Uses time.perf_counter for true
    end-to-end wall-clock timing (includes CPU overhead, launch latency,
    synchronization gaps — not just GPU kernel execution).
    """

    script = f"""\
import json, sys, torch, importlib.util, time

torch.cuda.set_device(0)

spec = importlib.util.spec_from_file_location("_model", "{model_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["_model"] = mod
spec.loader.exec_module(mod)

fn = getattr(mod, "kernel_function", None) or getattr(mod, "call", None)
get_inputs = getattr(mod, "get_inputs", None)
if fn is None or get_inputs is None:
    print(json.dumps({{"error": "no kernel_function/get_inputs"}}))
    sys.exit(0)

# Warmup
for _ in range({warmup}):
    inputs = get_inputs()
    inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
    with torch.no_grad():
        fn(*inputs)
    torch.cuda.synchronize()

# Measure wall-clock
times = []
for _ in range({repeats}):
    inputs = get_inputs()
    inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        fn(*inputs)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
median = times[len(times) // 2]
print(json.dumps({{"median_ms": median, "min_ms": min(times), "max_ms": max(times)}}))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            import json as _json
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    result = _json.loads(line)
                    if "error" in result:
                        return -1.0, result["error"]
                    return result["median_ms"], (
                        f"median={result['median_ms']:.3f}ms "
                        f"min={result['min_ms']:.3f}ms "
                        f"max={result['max_ms']:.3f}ms"
                    )
        return -1.0, f"Timing failed (exit {proc.returncode}): {(proc.stderr or '')[-300:]}"
    except subprocess.TimeoutExpired:
        return -1.0, "Timing timed out (120s)"
    except Exception as e:
        return -1.0, f"Timing error: {e}"
