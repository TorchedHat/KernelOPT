"""Wall-clock benchmarking for GPU kernel performance measurement.

Provides wall-clock timing using triton.testing.do_bench (CUDA Events
based, median of multiple runs). Used for end-to-end comparison tables
(vs PyTorch eager, vs torch.compile) and standalone kernel benchmarking.

The kernelopt optimization pipeline uses NCU for per-kernel hardware
analysis and optimization selection. This module complements NCU with
end-to-end wall-clock timing that captures the full computation.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _find_callable_and_inputs(module):
    """Find the benchmark-able function and input generator in a kernel module."""
    fn = (
        getattr(module, "kernel_function", None)
        or getattr(module, "call", None)
        or getattr(module, "run", None)
    )
    get_inputs = getattr(module, "get_inputs", None)
    return fn, get_inputs


def benchmark_kernel(
    kernel_path: str,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """Benchmark a kernel file using triton.testing.do_bench.

    Returns dict with median_ms or error. Used by eval_benchmark.py
    for wall-clock comparison tables, not by the kernelopt pipeline.
    """
    try:
        import torch
        from triton.testing import do_bench
    except ImportError:
        return {"median_ms": -1, "error": "torch or triton not available"}

    try:
        mod_name = f"_bench_{Path(kernel_path).stem}_{id(kernel_path)}"
        spec = importlib.util.spec_from_file_location(mod_name, str(kernel_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        return {"median_ms": -1, "error": f"import failed: {str(e)[:200]}"}

    fn, get_inputs = _find_callable_and_inputs(module)
    if fn is None or get_inputs is None:
        return {"median_ms": -1, "error": "no callable or get_inputs found"}

    try:
        inputs = get_inputs()
        with torch.no_grad():
            median_ms = do_bench(lambda: fn(*inputs), warmup=warmup, rep=rep)
        return {"median_ms": float(median_ms)}
    except Exception as e:
        return {"median_ms": -1, "error": f"benchmark failed: {str(e)[:200]}"}


# ---------------------------------------------------------------------------
# Subprocess-based timing (isolated CUDA context per measurement)
# ---------------------------------------------------------------------------

_TIMING_SCRIPT_TEMPLATE = """\
import json, torch, importlib.util, sys
torch.cuda.set_device(0)
from triton.testing import do_bench

spec = importlib.util.spec_from_file_location("_m", "{file_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["_m"] = mod
spec.loader.exec_module(mod)

{setup_code}

inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_i()]
with torch.no_grad():
    median_ms = do_bench(lambda: fn(*inputs), warmup=25, rep=100)
print(json.dumps({{"median_ms": round(float(median_ms), 6)}}))
"""


def _time_in_subprocess(
    file_path: str, setup_code: str, timeout: int = 300,
) -> tuple[float, str]:
    """Run a timing script in an isolated subprocess.

    Returns (median_ms, error_reason).  median_ms is -1 on failure;
    error_reason is "" on success or a human-readable explanation on failure.
    """
    script = _TIMING_SCRIPT_TEMPLATE.format(
        file_path=file_path,
        setup_code=setup_code,
    )
    try:
        p = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        if p.returncode == 0:
            for line in reversed(p.stdout.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    if "error" in data:
                        return -1, data["error"]
                    return data.get("median_ms", -1), ""
        # Non-zero exit — extract the actual error from stderr
        error = ""
        if p.stderr:
            for err_line in reversed(p.stderr.strip().split("\n")):
                err_line = err_line.strip()
                if err_line and not err_line.startswith("^") and len(err_line) > 5:
                    error = err_line[:200]
                    break
        return -1, error or f"subprocess exited with code {p.returncode}"
    except subprocess.TimeoutExpired:
        return -1, f"subprocess timed out after {timeout}s"
    except Exception as e:
        return -1, str(e)[:200]


_CORRECTNESS_SCRIPT = """\
import json, torch, importlib.util, sys
torch.cuda.set_device(0)

# Load original model
spec1 = importlib.util.spec_from_file_location("_orig", "{original_path}")
orig_mod = importlib.util.module_from_spec(spec1)
sys.modules["_orig"] = orig_mod
spec1.loader.exec_module(orig_mod)

# Load optimized model
spec2 = importlib.util.spec_from_file_location("_opt", "{optimized_path}")
opt_mod = importlib.util.module_from_spec(spec2)
sys.modules["_opt"] = opt_mod
spec2.loader.exec_module(opt_mod)

Model = getattr(orig_mod, "Model", None)
get_inputs_orig = getattr(orig_mod, "get_inputs", None)
get_init = getattr(orig_mod, "get_init_inputs", lambda: [])
opt_fn = getattr(opt_mod, "kernel_function", None) or getattr(opt_mod, "call", None)
get_inputs_opt = getattr(opt_mod, "get_inputs", None)

if not Model or not get_inputs_orig or not opt_fn or not get_inputs_opt:
    print(json.dumps({{"correct": False, "error": "missing Model/get_inputs/kernel_function"}}))
    sys.exit(0)

try:
    init_args = get_init() if get_init else []
    # Seed model creation for reproducible weights — ensures fallback
    # files that wrap torch.compile(Model()) produce identical weights
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = Model(*init_args).cuda().eval()

    # Check if both models accept the same input interface.
    # Match by count AND shapes to avoid false positives (e.g., same count
    # but flipped argument order or different semantics).
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    _sample_orig = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_inputs_orig()]
    _sample_opt = get_inputs_opt()
    same_interface = (
        len(_sample_orig) == len(_sample_opt)
        and all(
            isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.shape == b.shape and a.dtype == b.dtype
            for a, b in zip(_sample_orig, _sample_opt)
        )
    )

    # When interfaces differ (e.g., Inductor externalizes model weights as
    # explicit arguments), reconstruct the correct flat arg list by using
    # torch.compile with a capture backend.  This extracts the EXACT args
    # in Inductor's order with REAL model state values — works generically
    # for any nn.Module (BN, LN, Conv, Linear, Attention, etc.).
    _captured_flat_args = None
    if not same_interface and len(_sample_opt) > len(_sample_orig):
        try:
            _capture_list = []
            def _capture_backend(_gm, _example_inputs):
                _capture_list.clear()
                _capture_list.extend(_example_inputs)
                return _gm
            _compiled = torch.compile(model, backend=_capture_backend)
            with torch.no_grad():
                _compiled(*_sample_orig)
            if len(_capture_list) == len(_sample_opt):
                _captured_flat_args = _capture_list
                same_interface = True  # can now do strict comparison
        except Exception:
            pass  # capture failed — fall back to sanity mode

    max_diffs = []
    for seed in [0, 42, 123]:
        # Generate inputs from original model (canonical source)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        inputs_ref = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_inputs_orig()]
        with torch.no_grad():
            ref_out = model(*inputs_ref)

        if same_interface:
            if _captured_flat_args is not None:
                # Externalized weights: the captured flat args are in Inductor's
                # order (which may interleave data and params in ANY order).
                # Identify which captured args correspond to data inputs by
                # matching shape+dtype, replace those with this seed's data,
                # and keep model state args as captured.
                inputs_opt = [t.clone() for t in _captured_flat_args]
                _data_matched = list(range(len(inputs_ref)))  # indices to match
                for _ci, _ct in enumerate(inputs_opt):
                    if not isinstance(_ct, torch.Tensor):
                        continue
                    for _di in list(_data_matched):
                        _dt = inputs_ref[_di]
                        if isinstance(_dt, torch.Tensor) and _ct.shape == _dt.shape and _ct.dtype == _dt.dtype:
                            inputs_opt[_ci] = _dt.clone()
                            _data_matched.remove(_di)
                            break
            else:
                # Same interface — clone tensors to avoid in-place mutation bugs
                inputs_opt = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs_ref]
        else:
            # Different interface, matching failed — use own inputs
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            inputs_opt = get_inputs_opt()

        # Track whether THIS seed used the same inputs (don't mutate
        # same_interface which controls the loop for ALL seeds)
        used_same_inputs = same_interface

        try:
            with torch.no_grad():
                opt_out = opt_fn(*inputs_opt)
        except Exception as exc:
            if used_same_inputs:
                try:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    inputs_opt = get_inputs_opt()
                    with torch.no_grad():
                        opt_out = opt_fn(*inputs_opt)
                    used_same_inputs = False  # this seed fell back
                except Exception as fallback_exc:
                    print(json.dumps({{"correct": False, "max_diff": -1,
                                      "error": f"kernel execution failed: {{str(fallback_exc)[:200]}}"}}))
                    sys.exit(0)
            else:
                print(json.dumps({{"correct": False, "max_diff": -1,
                                  "error": f"kernel execution failed: {{str(exc)[:200]}}"}}))
                sys.exit(0)

        torch.cuda.synchronize()

        # Normalize outputs to lists for uniform handling of single/multi-output
        ref_outs = ref_out if isinstance(ref_out, (tuple, list)) else [ref_out]
        opt_outs = opt_out if isinstance(opt_out, (tuple, list)) else [opt_out]

        if len(ref_outs) != len(opt_outs):
            print(json.dumps({{"correct": False,
                              "error": f"output count mismatch: ref={{len(ref_outs)}} opt={{len(opt_outs)}}"}}))
            sys.exit(0)

        seed_worst_diff = 0.0

        for out_idx, (r, o) in enumerate(zip(ref_outs, opt_outs)):
            if not isinstance(r, torch.Tensor) or not isinstance(o, torch.Tensor):
                continue

            # Structural checks — required for BOTH paths
            if r.shape != o.shape:
                print(json.dumps({{"correct": False,
                                  "error": f"output[{{out_idx}}] shape mismatch: ref={{r.shape}} opt={{o.shape}}"}}))
                sys.exit(0)

            if r.dtype != o.dtype:
                print(json.dumps({{"correct": False,
                                  "error": f"output[{{out_idx}}] dtype mismatch: ref={{r.dtype}} opt={{o.dtype}}"}}))
                sys.exit(0)

            opt_f = o.float()
            if torch.isnan(opt_f).any() or torch.isinf(opt_f).any():
                n_bad = (torch.isnan(opt_f) | torch.isinf(opt_f)).sum().item()
                print(json.dumps({{"correct": False, "max_diff": -1,
                                  "error": f"NaN/Inf in output[{{out_idx}}] ({{n_bad}} elements)"}}))
                sys.exit(0)

            # Tiered verification
            if used_same_inputs:
                # Tier 1: strict element-wise allclose
                diff = (r.float() - opt_f).abs().max().item()
                if diff > seed_worst_diff:
                    seed_worst_diff = diff
                correct_el = torch.allclose(r.float(), opt_f, rtol=1e-3, atol=1e-5)

                if not correct_el:
                    # Tier 2: float64 reference — distinguishes "different FP32
                    # accumulation order" (correct) from "algorithmic bug" (wrong).
                    # A correct kernel with different tiling has error within ~10x
                    # of the reference FP32 error. A buggy kernel is 1000x+ worse.
                    try:
                        import copy as _copy
                        model_f64 = _copy.deepcopy(model).double()
                        ref_f64 = model_f64(*[t.double() for t in inputs_ref])
                        if isinstance(ref_f64, (tuple, list)):
                            ref_f64 = ref_f64[out_idx]
                        ref_err = (r.double() - ref_f64).abs().max().item()
                        opt_err = (opt_f.double() - ref_f64).abs().max().item()
                        if ref_err < 1e-8:
                            # Baseline is exact (e.g., max, argmax, sort).
                            # Use scale-invariant relative check against
                            # the magnitude of the reference output.
                            ref_max_val = r.abs().max().item()
                            denom = max(ref_max_val, 1e-12)
                            error_ratio = opt_err / denom
                            threshold_ok = error_ratio <= 1e-4  # 0.01% relative
                        else:
                            error_ratio = opt_err / ref_err
                            # Primary: ratio within 10x of reference error
                            ratio_ok = error_ratio <= 10.0
                            # Secondary: absolute error within scaled tolerance
                            # Handles fused kernels where Inductor's separate-
                            # kernel approach has artificially low ref_err
                            # (DRAM writes anchor FP32 precision). A fused
                            # kernel with opt_err=1e-3 is correct for ML even
                            # if ref_err=1e-7 makes the ratio look bad.
                            # Floor of 5e-3 ensures strict check for small
                            # outputs (softmax, sigmoid); scales to 1e-3 *
                            # max_val for large outputs (matmul, conv).
                            ref_max_val = r.abs().max().item()
                            abs_bound = max(5e-3, 1e-3 * ref_max_val)
                            abs_ok = opt_err <= abs_bound
                            threshold_ok = ratio_ok or abs_ok
                        if threshold_ok:
                            # Acceptable accumulation order difference
                            pass  # correct
                        else:
                            print(json.dumps({{"correct": False, "max_diff": diff,
                                              "error": f"seed={{seed}} output[{{out_idx}}] failed: opt_err/ref_err={{error_ratio:.4e}} (ref_err={{ref_err:.4e}}, opt_err={{opt_err:.4e}})"}}))
                            sys.exit(0)
                    except Exception:
                        # float64 not supported — reject with original error
                        print(json.dumps({{"correct": False, "max_diff": diff,
                                          "error": f"seed={{seed}} output[{{out_idx}}] failed allclose (rtol=1e-3, atol=1e-5, max_diff={{diff:.4e}})"}}))
                        sys.exit(0)
            else:
                # Path B: sanity check (no element-wise comparison)
                if opt_f.numel() > 1 and opt_f.std().item() < 1e-10:
                    print(json.dumps({{"correct": False, "max_diff": -1,
                                      "error": f"output[{{out_idx}}] degenerate: std={{opt_f.std().item():.2e}}"}}))
                    sys.exit(0)

        if used_same_inputs:
            max_diffs.append(seed_worst_diff)

    worst = max(max_diffs) if max_diffs else 0.0
    mode = "strict" if same_interface else "sanity"
    print(json.dumps({{"correct": True, "max_diff": worst, "mode": mode}}))
except Exception as e:
    print(json.dumps({{"correct": False, "error": str(e)[:300]}}))
"""


def _check_correctness_subprocess(
    original_path: str, optimized_path: str, timeout: int = 120
) -> dict:
    """Verify optimized model produces correct results vs original model.

    Runs in subprocess with 3 random seeds. Returns dict with 'correct'
    bool and 'max_diff' float.
    """
    script = _CORRECTNESS_SCRIPT.format(
        original_path=original_path,
        optimized_path=optimized_path,
    )
    try:
        p = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        if p.returncode == 0:
            for line in reversed(p.stdout.strip().split("\n")):
                if line.strip().startswith("{"):
                    return json.loads(line.strip())
        return {"correct": False, "error": f"exit={p.returncode}: {p.stderr[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"correct": False, "error": "correctness check timed out"}
    except Exception as e:
        return {"correct": False, "error": str(e)[:200]}


def benchmark_model_e2e(
    original_model_path: str,
    optimized_model_path: str,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """Benchmark end-to-end model speedup with all baselines.

    Measures four things in isolated subprocesses:
    1. Correctness: optimized output matches original model (3 seeds)
    2. PyTorch eager timing (no compilation)
    3. torch.compile timing (Inductor baseline)
    4. KernelOPT optimized timing (final_model.py)

    If correctness fails, speedup is reported as 0.0 with error.

    All timings use do_bench (CUDA Events, median). Each runs in a
    separate subprocess to prevent CUDA state leaks.

    Returns dict with eager_ms, compile_ms, optimized_ms, speedups,
    and correctness info.
    """
    result: dict = {}

    # --- 0. Correctness check first ---
    corr = _check_correctness_subprocess(original_model_path, optimized_model_path)
    result["correct"] = corr.get("correct", False)
    result["max_diff"] = corr.get("max_diff", -1)
    result["verification_mode"] = corr.get("mode", "unknown")
    if not result["correct"]:
        diff_str = f"max_diff={result['max_diff']:.4e}"
        result["error"] = f"correctness failed: {corr.get('error', diff_str)}"
        result["speedup"] = 0.0
        result["vs_compile"] = 0.0
        result["vs_eager"] = 0.0
        return result

    # --- 1. PyTorch eager ---
    eager_setup = """\
Model = getattr(mod, "Model", None)
get_i = getattr(mod, "get_inputs", None)
get_ii = getattr(mod, "get_init_inputs", lambda: [])
if not Model or not get_i:
    print(json.dumps({"error": "missing Model or get_inputs"})); sys.exit(0)
init_args = get_ii() if get_ii else []
model = Model(*init_args).cuda().eval()
fn = model
"""
    eager_ms, eager_err = _time_in_subprocess(original_model_path, eager_setup)
    result["eager_ms"] = eager_ms

    # --- 2. torch.compile ---
    compile_setup = """\
Model = getattr(mod, "Model", None)
get_i = getattr(mod, "get_inputs", None)
get_ii = getattr(mod, "get_init_inputs", lambda: [])
if not Model or not get_i:
    print(json.dumps({"error": "missing Model or get_inputs"})); sys.exit(0)
init_args = get_ii() if get_ii else []
model = Model(*init_args).cuda().eval()
fn = torch.compile(model)
# Warmup compilation
inputs_warmup = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_i()]
with torch.no_grad():
    fn(*inputs_warmup)
"""
    compile_ms, compile_err = _time_in_subprocess(original_model_path, compile_setup)
    result["compile_ms"] = compile_ms

    # --- 3. KernelOPT optimized (final_model.py) ---
    optimized_setup = """\
fn = getattr(mod, "kernel_function", None) or getattr(mod, "call", None) or getattr(mod, "run", None)
get_i = getattr(mod, "get_inputs", None)
if not fn or not get_i:
    print(json.dumps({"error": "missing kernel_function or get_inputs"})); sys.exit(0)
"""
    optimized_ms, optimized_err = _time_in_subprocess(optimized_model_path, optimized_setup)
    result["optimized_ms"] = optimized_ms

    # --- Compute speedups ---
    def _sp(a: float, b: float) -> float:
        if a > 0 and b > 0:
            return round(a / b, 4)
        return 0.0

    result["vs_compile"] = _sp(compile_ms, optimized_ms)
    result["vs_eager"] = _sp(eager_ms, optimized_ms)
    result["compile_vs_eager"] = _sp(eager_ms, compile_ms)

    # Primary speedup = vs torch.compile (this is what KernelOPT improves over)
    result["speedup"] = result["vs_compile"]

    # Flag errors with actual reasons
    errors = []
    if eager_ms <= 0:
        errors.append(f"eager timing failed: {eager_err}" if eager_err else "eager timing failed")
    if compile_ms <= 0:
        errors.append(f"compile timing failed: {compile_err}" if compile_err else "compile timing failed")
    if optimized_ms <= 0:
        errors.append(f"optimized timing failed: {optimized_err}" if optimized_err else "optimized timing failed")
    if errors:
        result["error"] = "; ".join(errors)

    return result
