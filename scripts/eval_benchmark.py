#!/usr/bin/env python3
"""Phase 2 benchmark runner — runs kernelopt on KernelBench + SOL problems.

Designed for long unattended runs on a single GPU. Each GPU gets its own
process running this script with a different --assignments spec.

Features:
  - Resume: skips problems with existing result_<id>.json
  - Per-problem timeout (default 30 min)
  - Isolated memory bank per GPU (merged after Phase 1)
  - Baselines: measures eager + torch.compile before optimization
  - Robust error handling: never crashes the batch

Usage:
    python3 eval_benchmark.py --gpu 2 --assignments kb_l1 --phase 1
    python3 eval_benchmark.py --gpu 3 --assignments kb_l3,kb_l4,sol_train,sol_flashinfer --phase 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJ_ROOT = Path(__file__).resolve().parent
BENCH_KB = PROJ_ROOT / "tests_suite" / "benchmarks" / "kernelbench"
BENCH_SOL = PROJ_ROOT / "tests_suite" / "benchmarks" / "sol_execbench"
KERNELOPT_PKG = PROJ_ROOT / "kernelopt"

sys.path.insert(0, str(KERNELOPT_PKG))
sys.path.insert(0, str(PROJ_ROOT))

BASELINE_TIMEOUT_S = 1800  # 30 min safety net — prevents infinite hangs
                           # without cutting off legitimate slow compilations

# Ensure CUDA tools (ncu) are on PATH
_CUDA_BIN = "/usr/local/cuda-12.8/bin"
if _CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Problem discovery
# ---------------------------------------------------------------------------

BENCH_TRITON_TUTORIALS = PROJ_ROOT / "triton_officialdockernels"
BENCH_HELION_TUTORIALS = PROJ_ROOT / "helion_tutorials"

ASSIGNMENT_MAP = {
    "kb_l1": ("kernelbench", 1, None, None),
    "kb_l2": ("kernelbench", 2, None, None),
    "kb_l3": ("kernelbench", 3, None, None),
    "kb_l4": ("kernelbench", 4, None, None),
    "sol_train": ("sol", "train", None, None),
    "sol_flashinfer": ("sol", "FlashInfer-Bench", None, None),
    "sol_l1_lo": ("sol", "L1", 0, 46),
    "sol_l1_hi": ("sol", "L1", 47, 93),
    "sol_l2": ("sol", "L2", None, None),
    "sol_quant": ("sol", "Quant", None, None),
    "triton_tutorials": ("triton_tutorials", None, None, None),
    "helion_tutorials": ("helion_tutorials", None, None, None),
}


def discover_problems(assignments: list[str]) -> list[dict]:
    """Build the ordered list of problems for this GPU."""
    problems = []
    for assignment in assignments:
        assignment = assignment.strip().lower()
        if assignment not in ASSIGNMENT_MAP:
            print(f"WARNING: unknown assignment '{assignment}', skipping")
            continue

        bench_type, level_or_split, lo, hi = ASSIGNMENT_MAP[assignment]

        if bench_type == "kernelbench":
            level_dir = BENCH_KB / f"level_{level_or_split}"
            for py_path in sorted(level_dir.glob("kernel_*.py")):
                kid = int(py_path.stem.split("_")[1])
                if lo is not None and kid < lo:
                    continue
                if hi is not None and kid > hi:
                    continue
                problems.append({
                    "problem_id": f"kb_l{level_or_split}_{py_path.stem}",
                    "dataset": f"kb_l{level_or_split}",
                    "bench_type": "kernelbench",
                    "path": str(py_path),
                    "level": level_or_split,
                })
        elif bench_type == "sol":
            split = level_or_split
            if split == "train":
                split_dir = BENCH_SOL / "train"
            else:
                split_dir = BENCH_SOL / split / "train"
            if not split_dir.exists():
                print(f"WARNING: {split_dir} does not exist, skipping")
                continue
            for json_path in sorted(split_dir.glob("kernel_*.json")):
                kid = int(json_path.stem.split("_")[1])
                if lo is not None and kid < lo:
                    continue
                if hi is not None and kid > hi:
                    continue
                problems.append({
                    "problem_id": f"sol_{split.lower().replace('-','_')}_{json_path.stem}",
                    "dataset": f"sol_{split.lower().replace('-','_')}",
                    "bench_type": "sol",
                    "path": str(json_path),
                    "split": split,
                })
        elif bench_type == "triton_tutorials":
            if not BENCH_TRITON_TUTORIALS.exists():
                print(f"WARNING: {BENCH_TRITON_TUTORIALS} does not exist, skipping")
                continue
            for py_path in sorted(BENCH_TRITON_TUTORIALS.glob("*.py")):
                if py_path.suffix != ".py" or py_path.name.endswith(".bak"):
                    continue
                problems.append({
                    "problem_id": f"triton_{py_path.stem}",
                    "dataset": "triton_tutorials",
                    "bench_type": "triton",
                    "path": str(py_path),
                })
        elif bench_type == "helion_tutorials":
            if not BENCH_HELION_TUTORIALS.exists():
                print(f"WARNING: {BENCH_HELION_TUTORIALS} does not exist, skipping")
                continue
            for py_path in sorted(BENCH_HELION_TUTORIALS.glob("*.py")):
                if py_path.suffix != ".py":
                    continue
                problems.append({
                    "problem_id": f"helion_{py_path.stem}",
                    "dataset": "helion_tutorials",
                    "bench_type": "helion",
                    "path": str(py_path),
                })

    return problems


# ---------------------------------------------------------------------------
# Baseline measurement (eager + torch.compile)
# ---------------------------------------------------------------------------

def wall_clock_benchmark(kernel_path: str, gpu_id: int) -> dict:
    """Wall-clock benchmark a kernel file using triton.testing.do_bench.

    do_bench uses CUDA events internally (25 warmup, 100 reps, median).
    Same measurement method as measure_baselines_kb/sol/tutorial.
    Returns {"wall_ns": float, "error": str}.
    """
    script = f"""\
import json, sys, os, torch, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
torch.cuda.set_device(0)
from triton.testing import do_bench

spec = importlib.util.spec_from_file_location("_bench", "{kernel_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["_bench"] = mod
spec.loader.exec_module(mod)

fn = getattr(mod, "kernel_function", None) or getattr(mod, "call", None) or getattr(mod, "run", None)
get_inputs = getattr(mod, "get_inputs", None)

result = {{"wall_ns": -1, "error": ""}}

if fn is None or get_inputs is None:
    result["error"] = "no callable or get_inputs found"
    print(json.dumps(result))
    sys.exit(0)

try:
    inputs = get_inputs()
    with torch.no_grad():
        median_ms = do_bench(lambda: fn(*inputs), warmup=25, rep=100)
        result["wall_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] = f"benchmark: {{ex}}"

print(json.dumps(result))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            timeout=BASELINE_TIMEOUT_S,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        return {"wall_ns": -1,
                "error": f"exit={proc.returncode}: {(proc.stderr or '')[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"wall_ns": -1, "error": "benchmark timeout"}
    except Exception as e:
        return {"wall_ns": -1, "error": str(e)[:200]}


def ncu_boost_duration(kernel_path: str, gpu_id: int, output_dir: Path) -> dict:
    """Get total GPU kernel duration at boost clocks for Amdahl's analysis.

    Delegates to ncu_runner.run_ncu_boost() which uses:
      - --clock-control none (boost clocks, comparable to wall-clock)
      - --set basic (duration only, fast)
      - NVTX markers to isolate the measured call from warmup/JIT/init

    Returns {"kernel_ns": float, "error": str}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write wrapper script: warmup + NVTX-isolated measured call
    wrapper_path = str(output_dir / "boost_wrapper.py")
    wrapper_script = f"""\
import torch, sys, importlib.util
spec = importlib.util.spec_from_file_location("_k", "{kernel_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["_k"] = mod
spec.loader.exec_module(mod)
fn = getattr(mod, "kernel_function", None) or getattr(mod, "call", None) or getattr(mod, "run", None)
get_inputs = getattr(mod, "get_inputs", None)
inputs = get_inputs()
with torch.no_grad():
    for _ in range(3):
        fn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure")
    fn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
"""
    Path(wrapper_path).write_text(wrapper_script)

    try:
        env_prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from kernelopt.agents.profiler.ncu_runner import run_ncu_boost
        kernel_ns = run_ncu_boost(wrapper_path, str(output_dir))

        if kernel_ns > 0:
            return {"kernel_ns": kernel_ns, "error": ""}
        return {"kernel_ns": -1, "error": "ncu: no kernel durations captured"}
    except Exception as e:
        return {"kernel_ns": -1, "error": str(e)[:200]}
    finally:
        if env_prev is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_prev
        try:
            Path(wrapper_path).unlink(missing_ok=True)
        except Exception:
            pass


def measure_baselines_kb(problem_path: str, gpu_id: int) -> dict:
    """Measure PyTorch eager and torch.compile baselines for a KB problem.

    Uses triton.testing.do_bench (CUDA events, 25 warmup, 100 reps, median).
    """
    script = f"""\
import json, sys, os, torch, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
torch.cuda.set_device(0)
from triton.testing import do_bench

spec = importlib.util.spec_from_file_location("prob", "{problem_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Model(*mod.get_init_inputs()).cuda().eval()
inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in mod.get_inputs()]

result = {{"eager_ns": -1, "compile_ns": -1, "error": ""}}

try:
    with torch.no_grad():
        median_ms = do_bench(lambda: model(*inputs), warmup=25, rep=100)
        result["eager_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] = f"eager: {{ex}}"

try:
    compiled = torch.compile(model, mode="default")
    with torch.no_grad():
        median_ms = do_bench(lambda: compiled(*inputs), warmup=25, rep=100)
        result["compile_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] += f" compile: {{ex}}"

print(json.dumps(result))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            timeout=BASELINE_TIMEOUT_S,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        return {"eager_ns": -1, "compile_ns": -1,
                "error": f"exit={proc.returncode}: {(proc.stderr or '')[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"eager_ns": -1, "compile_ns": -1, "error": "baseline timeout"}
    except Exception as e:
        return {"eager_ns": -1, "compile_ns": -1, "error": str(e)[:200]}


def measure_baselines_sol(json_path: str, gpu_id: int) -> dict:
    """Measure baselines for a SOL problem (reference fn + torch.compile)."""
    script = f"""\
import json, sys, os, torch, math, types
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
torch.cuda.set_device(0)

raw = json.load(open("{json_path}"))
axes = json.loads(raw["axes"]) if isinstance(raw["axes"], str) else raw["axes"]
inputs_spec = json.loads(raw["inputs"]) if isinstance(raw["inputs"], str) else raw["inputs"]
workloads = json.loads(raw["workloads"]) if isinstance(raw["workloads"], str) else raw["workloads"]
ref_code = raw.get("reference", "")

resolved = {{}}
for name, spec in axes.items():
    if isinstance(spec, dict) and spec.get("type") == "const":
        resolved[name] = spec["value"]
if workloads:
    wl = workloads[0] if isinstance(workloads, list) else workloads
    for name, val in (wl.get("axes", {{}}) if isinstance(wl, dict) else {{}}).items():
        resolved[name] = val
for _ in range(3):
    for name, spec in axes.items():
        if name in resolved:
            continue
        if isinstance(spec, dict) and spec.get("type") == "expr":
            try:
                resolved[name] = int(eval(spec["expression"], {{"__builtins__": {{}}}}, resolved))
            except Exception:
                pass
unresolved = [n for n in axes if n not in resolved]
if unresolved:
    raise ValueError(f"SOL axes unresolved: {{unresolved}}")

DTYPE_MAP = {{"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
              "int32": torch.int32, "int64": torch.int64, "int8": torch.int8}}
args = []
for inp_name, inp_spec in inputs_spec.items():
    if not isinstance(inp_spec, dict):
        continue
    dtype_str = inp_spec.get("dtype", "float32")
    dt = DTYPE_MAP.get(dtype_str, torch.float32)
    shape = inp_spec.get("shape")
    if shape is None:
        args.append(torch.tensor(0.1, dtype=dt, device="cuda"))
    else:
        dims = []
        for d in shape:
            if isinstance(d, int):
                dims.append(d)
            elif isinstance(d, str) and d.isdigit():
                dims.append(int(d))
            elif isinstance(d, str):
                dims.append(resolved.get(d, 64))
            else:
                dims.append(64)
        sh = tuple(dims)
        if "int" in dtype_str:
            args.append(torch.randint(0, 10, sh, dtype=dt, device="cuda"))
        else:
            args.append(torch.randn(*sh, dtype=dt, device="cuda"))

import sys as _sys
mod = types.ModuleType("_sol_baseline_ref")
mod.__dict__["torch"] = torch
mod.__dict__["math"] = math
_sys.modules["_sol_baseline_ref"] = mod
exec(compile(ref_code, "<ref>", "exec"), mod.__dict__)
run_fn = mod.__dict__["run"]

result = {{"eager_ns": -1, "compile_ns": -1, "error": ""}}

from triton.testing import do_bench

try:
    with torch.no_grad():
        median_ms = do_bench(lambda: run_fn(*args), warmup=25, rep=100)
        result["eager_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] = f"eager: {{ex}}"

try:
    compiled_fn = torch.compile(run_fn, mode="default")
    with torch.no_grad():
        median_ms = do_bench(lambda: compiled_fn(*args), warmup=25, rep=100)
        result["compile_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] += f" compile: {{ex}}"

print(json.dumps(result))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            timeout=BASELINE_TIMEOUT_S,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        return {"eager_ns": -1, "compile_ns": -1,
                "error": f"exit={proc.returncode}: {(proc.stderr or '')[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"eager_ns": -1, "compile_ns": -1, "error": "baseline timeout"}
    except Exception as e:
        return {"eager_ns": -1, "compile_ns": -1, "error": str(e)[:200]}


def measure_baselines_tutorial(kernel_path: str, gpu_id: int) -> dict:
    """Measure eager + torch.compile baselines for a tutorial kernel with reference_function.

    Uses triton.testing.do_bench (CUDA events, 25 warmup, 100 reps, median).
    Looks for reference_function + get_inputs in the kernel file.
    """
    script = f"""\
import json, sys, os, torch, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
torch.cuda.set_device(0)
from triton.testing import do_bench

spec = importlib.util.spec_from_file_location("prob", "{kernel_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["prob"] = mod
spec.loader.exec_module(mod)

ref_fn = getattr(mod, "reference_function", None)
get_inputs = getattr(mod, "get_inputs", None)

result = {{"eager_ns": -1, "compile_ns": -1, "error": ""}}

if ref_fn is None or get_inputs is None:
    result["error"] = "no reference_function or get_inputs found"
    print(json.dumps(result))
    sys.exit(0)

inputs = get_inputs()

try:
    with torch.no_grad():
        median_ms = do_bench(lambda: ref_fn(*inputs), warmup=25, rep=100)
        result["eager_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] = f"eager: {{ex}}"

try:
    compiled = torch.compile(ref_fn, mode="default")
    with torch.no_grad():
        median_ms = do_bench(lambda: compiled(*inputs), warmup=25, rep=100)
        result["compile_ns"] = float(median_ms) * 1e6
except Exception as ex:
    result["error"] += f" compile: {{ex}}"

print(json.dumps(result))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            timeout=BASELINE_TIMEOUT_S,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        return {"eager_ns": -1, "compile_ns": -1,
                "error": f"exit={proc.returncode}: {(proc.stderr or '')[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"eager_ns": -1, "compile_ns": -1, "error": "baseline timeout"}
    except Exception as e:
        return {"eager_ns": -1, "compile_ns": -1, "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# SOL JSON -> .py conversion (for kernelopt run)
# ---------------------------------------------------------------------------

def _resolve_sol_axes(axes_spec: dict, workloads: list) -> dict:
    """Resolve all SOL axes including const, var (from workload), and expr types."""
    resolved = {}
    for name, spec in axes_spec.items():
        if isinstance(spec, dict) and spec.get("type") == "const":
            resolved[name] = spec["value"]
    if workloads:
        wl = workloads[0] if isinstance(workloads, list) else workloads
        for name, val in (wl.get("axes", {}) if isinstance(wl, dict) else {}).items():
            resolved[name] = val
    # Evaluate expr axes (may depend on const/var axes already resolved)
    # Multiple passes to handle exprs that depend on other exprs
    for _ in range(3):
        for name, spec in axes_spec.items():
            if name in resolved:
                continue
            if isinstance(spec, dict) and spec.get("type") == "expr":
                expr = spec.get("expression", "")
                try:
                    resolved[name] = int(eval(expr, {"__builtins__": {}}, resolved))
                except Exception:
                    pass
    unresolved = [n for n in axes_spec if n not in resolved]
    if unresolved:
        raise ValueError(
            f"SOL axes could not be resolved: {unresolved}. "
            f"Available: {list(resolved.keys())}"
        )
    return resolved


_DTYPE_MAP_STR = {
    "float32": "torch.float32", "float16": "torch.float16",
    "bfloat16": "torch.bfloat16", "int32": "torch.int32",
    "int64": "torch.int64", "int8": "torch.int8", "uint8": "torch.uint8",
    "bool": "torch.bool", "fp8_e4m3": "torch.float8_e4m3fn",
    "fp8_e5m2": "torch.float8_e5m2",
}


def sol_json_to_py(json_path: str, out_dir: Path) -> str:
    """Convert SOL JSON spec to standalone .py for kernelopt run."""
    data = json.load(open(json_path))
    reference = data.get("reference", "")
    if not reference:
        raise ValueError(f"No reference code in {json_path}")

    inputs_spec = json.loads(data.get("inputs", "{}"))
    axes_spec = json.loads(data.get("axes", "{}"))
    workloads = json.loads(data.get("workloads", "[]"))

    axis_values = _resolve_sol_axes(axes_spec, workloads)

    lines = ["def get_inputs():"]
    names = []
    for inp_name, inp_spec in inputs_spec.items():
        if not isinstance(inp_spec, dict):
            continue
        dtype = inp_spec.get("dtype", "float32")
        torch_dtype = _DTYPE_MAP_STR.get(dtype, "torch.float32")
        shape = inp_spec.get("shape")
        if shape is None:
            lines.append(f"    {inp_name} = torch.tensor(0.1, dtype={torch_dtype}, device='cuda')")
        else:
            parts = [str(int(d) if isinstance(d, str) and d.isdigit() else axis_values.get(d, 64) if isinstance(d, str) else d) for d in shape]
            shape_str = ", ".join(parts)
            if "int" in dtype or dtype in ("uint8",):
                lines.append(f"    {inp_name} = torch.randint(0, 10, ({shape_str},), dtype={torch_dtype}, device='cuda')")
            elif "fp8" in dtype:
                lines.append(f"    {inp_name} = torch.randn({shape_str}, device='cuda').to({torch_dtype})")
            elif dtype == "bool":
                lines.append(f"    {inp_name} = torch.randint(0, 2, ({shape_str},), device='cuda').bool()")
            else:
                lines.append(f"    {inp_name} = torch.randn({shape_str}, dtype={torch_dtype}, device='cuda')")
        names.append(inp_name)
    lines.append(f"    return [{', '.join(names)}]")

    py_code = reference.strip() + "\n\n" + "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(json_path).stem}.py"
    out_path.write_text(py_code)
    return str(out_path)


# ---------------------------------------------------------------------------
# Run kernelopt on a single problem
# ---------------------------------------------------------------------------

def run_kernelopt_problem(
    problem: dict,
    gpu_id: int,
    output_dir: Path,
    *,
    T: int = 4,
    N: int = 3,
    K: int = 2,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """Run `kernelopt run` on a single problem and parse results."""
    problem_id = problem["problem_id"]
    result = {
        "problem_id": problem_id,
        "dataset": problem["dataset"],
        "bench_type": problem["bench_type"],
        "gpu_id": gpu_id,
        "status": "unknown",
        "synthesis_status": "skip",
        "n_triton_kernels": 0,
        "has_extern_kernels": False,
        "eager_ns": -1,
        "compile_ns": -1,
        "baseline_latency_ns": -1,
        "best_latency_ns": -1,
        "speedup": 0.0,
        "speedup_vs_eager": 0.0,
        "speedup_vs_compile": 0.0,
        "synth_wall_ns": -1,
        "best_wall_ns": -1,
        "synthesis_speedup": 0.0,
        "optimization_speedup": 0.0,
        "best_kernel_boost_ns": -1,
        "kernel_pct": 0.0,
        "host_overhead_pct": 0.0,
        "best_kernel_path": "",
        "strategy_type": "",
        "optimization_strategy": "",
        "multi_kernel_summary": "",
        "plans_tried": 0,
        "iterations_completed": 0,
        "correctness_pass_rate": 0.0,
        "best_direction": "",
        "error": "",
        "wall_time_s": 0.0,
        "baseline_time_s": 0.0,
        "kpt_time_s": 0.0,
        "bench_time_s": 0.0,
    }

    problem_start = time.monotonic()

    # Determine kernel path for kernelopt
    if problem["bench_type"] == "sol":
        try:
            sol_py_dir = output_dir / "sol_converted"
            kernel_path = sol_json_to_py(problem["path"], sol_py_dir)
        except Exception as e:
            result["status"] = "convert_error"
            result["error"] = str(e)[:300]
            return result
    else:
        kernel_path = problem["path"]

    # Measure baselines (eager + torch.compile of the reference implementation).
    # All types have a PyTorch reference: KernelBench has Model, SOL has run(),
    # Triton/Helion tutorials have reference_function (PyTorch spec from website).
    baseline_start = time.monotonic()
    if problem["bench_type"] == "kernelbench":
        baselines = measure_baselines_kb(kernel_path, gpu_id)
        result["eager_ns"] = baselines.get("eager_ns", -1)
        result["compile_ns"] = baselines.get("compile_ns", -1)
    elif problem["bench_type"] == "sol":
        baselines = measure_baselines_sol(problem["path"], gpu_id)
        result["eager_ns"] = baselines.get("eager_ns", -1)
        result["compile_ns"] = baselines.get("compile_ns", -1)
    elif problem["bench_type"] in ("triton", "helion"):
        baselines = measure_baselines_tutorial(kernel_path, gpu_id)
        result["eager_ns"] = baselines.get("eager_ns", -1)
        result["compile_ns"] = baselines.get("compile_ns", -1)
    else:
        baselines = {}
    result["baseline_time_s"] = round(time.monotonic() - baseline_start, 1)
    if baselines.get("error"):
        result["error"] = f"baseline: {baselines['error']} | "

    # Build kernelopt run command
    kpt_output = output_dir / problem_id.replace("/", "_")
    cmd = [
        "kernelopt", "run", kernel_path,
        "-T", str(T), "-N", str(N), "-K", str(K),
        "-o", str(kpt_output),
        "--model", model,
    ]

    # Standalone Triton/Helion tutorial kernels need --kernel-args '--profile-only'
    # to skip their internal benchmarks during NCU profiling
    if problem["bench_type"] in ("triton", "helion"):
        cmd.extend(["--kernel-args", "--profile-only"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = output_dir / f"log_{problem_id.replace('/', '_')}.txt"

    kpt_start = time.monotonic()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                env=env, cwd=str(PROJ_ROOT),
            )
        result["kpt_time_s"] = round(time.monotonic() - kpt_start, 1)
        result["status"] = "pass" if proc.returncode == 0 else "kpt_error"
        if proc.returncode != 0:
            try:
                log_tail = log_path.read_text()[-500:]
            except Exception:
                log_tail = ""
            result["error"] += f"exit={proc.returncode}: {log_tail}"
    except Exception as e:
        result["kpt_time_s"] = round(time.monotonic() - kpt_start, 1)
        result["status"] = "crash"
        result["error"] += str(e)[:300]

    # Parse results from the run_* subdirectory
    _parse_kpt_results(kpt_output, result)

    # ── Wall-clock benchmarks ──
    # All use triton.testing.do_bench (CUDA events, 25 warmup, 100 reps, median).
    bench_start = time.monotonic()

    # 1. Synthesized Triton baseline (KB/SOL only — before kpt LLM optimization)
    synth_kp = result.get("synth_kernel_path", "")
    if synth_kp and Path(synth_kp).exists():
        bench = wall_clock_benchmark(synth_kp, gpu_id)
        result["synth_wall_ns"] = bench.get("wall_ns", -1)
        if bench.get("error"):
            result["error"] += f"synth_bench: {bench['error']} | "

    # 2. kpt-optimized best kernel
    best_kp = result.get("best_kernel_path", "")
    if best_kp and Path(best_kp).exists():
        bench = wall_clock_benchmark(best_kp, gpu_id)
        result["best_wall_ns"] = bench.get("wall_ns", -1)
        if bench.get("error"):
            result["error"] += f"best_bench: {bench['error']} | "

        # 3. NCU with --clock-control none (kernel duration at boost clocks)
        #    for Amdahl's analysis: kernel_pct = T_kernel / T_wall
        ncu_boost = ncu_boost_duration(best_kp, gpu_id, kpt_output / "_boost_profile")
        result["best_kernel_boost_ns"] = ncu_boost.get("kernel_ns", -1)
        if ncu_boost.get("error"):
            result["error"] += f"boost_ncu: {ncu_boost['error']} | "

    # ── Speedup calculations (all wall-clock / wall-clock) ──

    # Total end-to-end: original PyTorch vs kpt output
    if result["eager_ns"] > 0 and result["best_wall_ns"] > 0:
        result["speedup_vs_eager"] = round(
            result["eager_ns"] / result["best_wall_ns"], 3
        )
    if result["compile_ns"] > 0 and result["best_wall_ns"] > 0:
        result["speedup_vs_compile"] = round(
            result["compile_ns"] / result["best_wall_ns"], 3
        )

    # Decomposed: separate synthesis contribution from optimization contribution.
    # For KB/SOL: synthesis = eager → synth_wall, optimization = synth_wall → best_wall
    # For Triton/Helion tutorials: no synthesis, so optimization_speedup = vs_eager
    if result["synth_wall_ns"] > 0:
        # Has synthesis step (KB/SOL)
        if result["eager_ns"] > 0:
            result["synthesis_speedup"] = round(
                result["eager_ns"] / result["synth_wall_ns"], 3
            )
        if result["best_wall_ns"] > 0:
            result["optimization_speedup"] = round(
                result["synth_wall_ns"] / result["best_wall_ns"], 3
            )
    elif problem["bench_type"] in ("triton", "helion"):
        # No synthesis — kpt optimizes the kernel directly
        # optimization_speedup = vs_eager (the entire speedup is from kpt)
        if result["eager_ns"] > 0 and result["best_wall_ns"] > 0:
            result["optimization_speedup"] = result["speedup_vs_eager"]

    # Amdahl's analysis: kernel execution % of total wall time.
    # Both best_kernel_boost_ns and best_wall_ns use boost clocks → directly comparable.
    if result["best_kernel_boost_ns"] > 0 and result["best_wall_ns"] > 0:
        result["kernel_pct"] = round(
            min(result["best_kernel_boost_ns"] / result["best_wall_ns"] * 100, 100.0), 1
        )
        result["host_overhead_pct"] = round(100.0 - result["kernel_pct"], 1)

    result["bench_time_s"] = round(time.monotonic() - bench_start, 1)
    result["wall_time_s"] = round(time.monotonic() - problem_start, 1)

    return result


def _parse_kpt_results(kpt_output: Path, result: dict) -> None:
    """Extract metrics from kernelopt output directory."""
    # Find the run_* subdirectory
    run_dirs = sorted(kpt_output.glob("run_*"))
    if not run_dirs:
        return
    run_dir = run_dirs[-1]  # latest run

    # Parse checkpoint
    ckpt_path = run_dir / "pipeline_checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
            gs = ckpt.get("graph_state", {})

            result["baseline_latency_ns"] = gs.get("baseline_latency_ns", -1)
            result["best_latency_ns"] = gs.get("best_latency_ns", -1)
            result["framework"] = gs.get("framework", "")
            result["iterations_completed"] = gs.get("iteration", 0)

            if result["baseline_latency_ns"] > 0 and result["best_latency_ns"] > 0:
                result["speedup"] = round(
                    result["baseline_latency_ns"] / result["best_latency_ns"], 3
                )
            # NCU speedup above is valid internally (both use same NCU settings).
            # Cross-method comparisons (eager wall-clock vs Triton NCU) are done
            # separately using wall-clock for ALL measurements — see best_wall_ns.

            bpc = gs.get("baseline_profiling_context", {})
            result["multi_kernel_summary"] = bpc.get("multi_kernel_summary", "")[:200]

            result["optimization_strategy"] = gs.get("optimization_strategy", "")

            ir_list = gs.get("iteration_results", [])
            total_candidates = sum(len(ir.get("candidates", [])) for ir in ir_list)
            result["plans_tried"] = total_candidates

            if gs.get("_original_pytorch_path"):
                result["synthesis_status"] = "pass"
            elif gs.get("framework") == "triton":
                result["synthesis_status"] = "pass"

        except Exception:
            pass

    # Parse optimization_memory.json for strategy details
    mem_path = run_dir / "optimization_memory.json"
    if mem_path.exists():
        try:
            mem = json.loads(mem_path.read_text())
            successes = [m for m in mem if m.get("speedup", 0) > 1.0]
            # improvement_rate: fraction of candidates that achieved speedup > 1.0
            result["improvement_rate"] = round(
                len(successes) / max(len(mem), 1), 3
            )
            # correctness_pass_rate: fraction that passed correctness (not implementation_flaw)
            correct = [m for m in mem if m.get("rewrite_type") != "implementation_flaw"]
            result["correctness_pass_rate"] = round(
                len(correct) / max(len(mem), 1), 3
            )
            if successes:
                best = max(successes, key=lambda m: m.get("speedup", 0))
                result["best_direction"] = best.get("direction", "")
                result["strategy_type"] = best.get("rewrite_type", "")
            elif mem:
                result["strategy_type"] = mem[0].get("rewrite_type", "")
                result["best_direction"] = mem[0].get("direction", "")
        except Exception:
            pass

    # Parse optimization_log.md for additional info
    log_md = run_dir / "optimization_log.md"
    if log_md.exists():
        try:
            text = log_md.read_text()
            m = re.search(r"Synthesis:\s*(.*)", text)
            if m:
                result["synthesis_status"] = "pass"
            for line in text.split("\n"):
                if "n_triton_kernels" in line or "Triton kernel" in line:
                    km = re.search(r"(\d+)\s*Triton", line)
                    if km:
                        result["n_triton_kernels"] = int(km.group(1))
                if "extern_kernels" in line:
                    result["has_extern_kernels"] = True
        except Exception:
            pass

    # Check for synthesis artifacts
    synth_dir = run_dir / "synthesis"
    if synth_dir.exists():
        result["synthesis_status"] = "pass"
        synth_file = synth_dir / "synthesized_triton.py"
        if synth_file.exists():
            src = synth_file.read_text()
            result["n_triton_kernels"] = src.count("@triton.jit")
            result["has_extern_kernels"] = "extern_kernels" in src

    # Locate output kernel files for wall-clock benchmarking
    best_kernel_file = run_dir / "best_kernel.py"
    if best_kernel_file.exists():
        result["best_kernel_path"] = str(best_kernel_file)

    # Locate synthesized Triton file (for KB/SOL — measures synthesis-only speedup)
    synth_file_path = synth_dir / "synthesized_triton.py" if synth_dir and synth_dir.exists() else None
    if synth_file_path and synth_file_path.exists():
        result["synth_kernel_path"] = str(synth_file_path)


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="kernelopt Phase 2 benchmark runner")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index")
    parser.add_argument("--assignments", type=str, required=True,
                        help="Comma-separated: kb_l1,kb_l2,sol_train,sol_l1_lo,...")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("-T", type=int, default=4, help="Max iterations")
    parser.add_argument("-N", type=int, default=3, help="Plans per iteration")
    parser.add_argument("-K", type=int, default=2, help="Attempts per plan")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--output", type=str, default="eval_benchmark_results")
    args = parser.parse_args()

    gpu_id = args.gpu
    phase = args.phase
    assignments = [a.strip() for a in args.assignments.split(",")]

    output_base = Path(args.output) / f"phase{phase}" / f"gpu{gpu_id}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Per-GPU memory bank
    bank_dir = Path(args.output) / f"phase{phase}_memory_banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_path = bank_dir / f"memory_bank_gpu{gpu_id}.json"
    if phase == 2:
        merged_bank = bank_dir.parent / "phase1_memory_banks" / "memory_bank_merged.json"
        if merged_bank.exists():
            bank_path = merged_bank
    os.environ["KERNELOPT_MEMORY_BANK"] = str(bank_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    problems = discover_problems(assignments)

    # Check which are already done (resume support)
    results_file = output_base / "results.json"
    done_ids = set()
    all_results = []
    if results_file.exists():
        try:
            all_results = json.loads(results_file.read_text())
            done_ids = {r["problem_id"] for r in all_results}
        except Exception:
            pass

    remaining = [p for p in problems if p["problem_id"] not in done_ids]

    print(f"{'='*70}")
    print(f"  kernelopt Phase {phase} Benchmark — GPU {gpu_id}")
    print(f"  Assignments: {', '.join(assignments)}")
    print(f"  Total problems: {len(problems)} | Already done: {len(done_ids)} | Remaining: {len(remaining)}")
    print(f"  Config: T={args.T} N={args.N} K={args.K} TopK=4 model={args.model}")
    print(f"  Memory bank: {bank_path}")
    print(f"  Output: {output_base}")
    print(f"{'='*70}\n")

    wall_start = time.monotonic()
    successes = sum(1 for r in all_results if r.get("status") == "pass")
    failures = sum(1 for r in all_results if r.get("status") != "pass")
    consecutive_cascade = 0

    for i, problem in enumerate(remaining, 1):
        pid = problem["problem_id"]
        elapsed_total = time.monotonic() - wall_start
        avg_per_problem = elapsed_total / max(i - 1, 1) if i > 1 else 900
        eta_s = avg_per_problem * (len(remaining) - i + 1)
        eta_h = eta_s / 3600

        print(f"\n[{len(done_ids)+i}/{len(problems)}] {pid} "
              f"(ok={successes} fail={failures} eta={eta_h:.1f}h)")
        print(f"  path: {problem['path']}")

        try:
            result = run_kernelopt_problem(
                problem, gpu_id, output_base,
                T=args.T, N=args.N, K=args.K,
                model=args.model,
            )
        except Exception as e:
            result = {
                "problem_id": pid,
                "dataset": problem["dataset"],
                "bench_type": problem["bench_type"],
                "gpu_id": gpu_id,
                "status": "runner_crash",
                "error": f"{type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-200:]}",
                "wall_time_s": 0,
            }

        # Detect CUDA cascade: a crash poisons the GPU driver for all
        # subsequent subprocesses. When we see 2 consecutive cascade errors,
        # re-exec ourselves so the OS gives us a fresh driver handle.
        is_cascade = "reported at some other API call" in result.get("error", "")
        if is_cascade:
            consecutive_cascade += 1
        else:
            consecutive_cascade = 0

        if consecutive_cascade >= 2:
            print(f"\n  *** CUDA cascade detected ({consecutive_cascade} in a row). "
                  f"Re-launching with fresh process... ***\n")
            # Save current results first
            all_results.append(result)
            results_file.write_text(json.dumps(all_results, indent=2, default=str))
            indiv = output_base / f"result_{pid.replace('/', '_')}.json"
            indiv.write_text(json.dumps(result, indent=2, default=str))
            # Re-exec: replaces this process with a fresh one
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # Print summary
        status = result.get("status", "?")
        wall = result.get("wall_time_s", 0)

        if status == "pass":
            successes += 1

            eager_us = result.get("eager_ns", -1) / 1e3 if result.get("eager_ns", -1) > 0 else -1
            compile_us = result.get("compile_ns", -1) / 1e3 if result.get("compile_ns", -1) > 0 else -1
            synth_us = result.get("synth_wall_ns", -1) / 1e3 if result.get("synth_wall_ns", -1) > 0 else -1
            best_wall_us = result.get("best_wall_ns", -1) / 1e3 if result.get("best_wall_ns", -1) > 0 else -1
            synth_sp = result.get("synthesis_speedup", 0)
            opt_sp = result.get("optimization_speedup", 0)
            vs_eager = result.get("speedup_vs_eager", 0)
            vs_compile = result.get("speedup_vs_compile", 0)
            kernel_pct = result.get("kernel_pct", 0)
            bl_t = result.get("baseline_time_s", 0)
            kpt_t = result.get("kpt_time_s", 0)
            bench_t = result.get("bench_time_s", 0)

            # Line 1: timings
            parts = []
            if eager_us > 0:
                parts.append(f"eager={eager_us:.1f}us")
            if compile_us > 0:
                parts.append(f"compile={compile_us:.1f}us")
            if synth_us > 0:
                parts.append(f"synth={synth_us:.1f}us")
            if best_wall_us > 0:
                parts.append(f"best={best_wall_us:.1f}us")
            print(f"  PASS  {' '.join(parts)}")
            # Line 2: decomposed speedups
            sp_parts = []
            if synth_sp > 0:
                sp_parts.append(f"synth={synth_sp:.2f}x")
            if opt_sp > 0:
                sp_parts.append(f"opt={opt_sp:.2f}x")
            if vs_eager > 0:
                sp_parts.append(f"vs_eager={vs_eager:.2f}x")
            if vs_compile > 0:
                sp_parts.append(f"vs_compile={vs_compile:.2f}x")
            if kernel_pct > 0:
                sp_parts.append(f"kernel={kernel_pct:.0f}%")
            if sp_parts:
                print(f"        {' '.join(sp_parts)}")
            # Line 3: timing breakdown
            print(f"        [baseline={bl_t:.0f}s kpt={kpt_t:.0f}s bench={bench_t:.0f}s total={wall:.0f}s]")
        else:
            failures += 1
            err = result.get("error", "")[:80]
            eager_us = result.get("eager_ns", -1) / 1e3 if result.get("eager_ns", -1) > 0 else -1
            compile_us = result.get("compile_ns", -1) / 1e3 if result.get("compile_ns", -1) > 0 else -1
            print(f"  {status.upper()}  eager={eager_us:.1f}us compile={compile_us:.1f}us  [{wall:.0f}s]  {err}")

        all_results.append(result)

        # Save after every problem (crash-safe)
        results_file.write_text(json.dumps(all_results, indent=2, default=str))

        # Also save individual result
        indiv = output_base / f"result_{pid.replace('/', '_')}.json"
        indiv.write_text(json.dumps(result, indent=2, default=str))

    # Final summary
    total_wall = time.monotonic() - wall_start
    print(f"\n{'='*70}")
    print(f"  DONE — GPU {gpu_id} Phase {phase}")
    print(f"  Total: {len(all_results)} | Pass: {successes} | Fail: {failures}")
    print(f"  Wall time: {total_wall/3600:.1f}h")

    speedups = [r["speedup"] for r in all_results if r.get("speedup", 0) > 0]
    if speedups:
        speedups.sort()
        median = speedups[len(speedups)//2]
        gt1 = sum(1 for s in speedups if s > 1.0)
        gt12 = sum(1 for s in speedups if s > 1.2)
        gt15 = sum(1 for s in speedups if s > 1.5)
        print(f"  Median NCU speedup: {median:.2f}x")
        print(f"  >1.0x: {gt1}/{len(speedups)} | >1.2x: {gt12}/{len(speedups)} | >1.5x: {gt15}/{len(speedups)}")

    wall_speedups = [r.get("wall_speedup", 0) for r in all_results if r.get("wall_speedup", 0) > 0]
    if wall_speedups:
        wall_speedups.sort()
        w_median = wall_speedups[len(wall_speedups)//2]
        print(f"  Median wall-clock speedup: {w_median:.2f}x")

    print(f"  Results: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
