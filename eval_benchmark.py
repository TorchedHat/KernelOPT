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
    python eval_benchmark.py --gpu 2 --assignments kb_l1 --phase 1
    python eval_benchmark.py --gpu 3 --assignments kb_l3,kb_l4,sol_train,sol_flashinfer --phase 1
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

BASELINE_TIMEOUT_S = 180  # 3 min for eager+compile baselines

# Ensure CUDA tools (ncu) are on PATH
_CUDA_BIN = "/usr/local/cuda-12.8/bin"
if _CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Problem discovery
# ---------------------------------------------------------------------------

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

    return problems


# ---------------------------------------------------------------------------
# Baseline measurement (eager + torch.compile)
# ---------------------------------------------------------------------------

def measure_baselines_kb(problem_path: str, gpu_id: int) -> dict:
    """Measure PyTorch eager and torch.compile baselines for a KB problem."""
    script = f"""\
import json, sys, os, torch, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
torch.cuda.set_device(0)

spec = importlib.util.spec_from_file_location("prob", "{problem_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Model(*mod.get_init_inputs()).cuda().eval()
inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in mod.get_inputs()]

result = {{"eager_ns": -1, "compile_ns": -1, "error": ""}}

try:
    with torch.no_grad():
        for _ in range(25):
            model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(*inputs)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e6)
        times.sort()
        result["eager_ns"] = times[len(times)//2]
except Exception as ex:
    result["error"] = f"eager: {{ex}}"

try:
    compiled = torch.compile(model, mode="default")
    with torch.no_grad():
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            compiled(*inputs)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e6)
        times.sort()
        result["compile_ns"] = times[len(times)//2]
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

try:
    with torch.no_grad():
        for _ in range(25):
            run_fn(*args)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            run_fn(*args)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e6)
        times.sort()
        result["eager_ns"] = times[len(times)//2]
except Exception as ex:
    result["error"] = f"eager: {{ex}}"

try:
    compiled_fn = torch.compile(run_fn, mode="default")
    with torch.no_grad():
        for _ in range(5):
            compiled_fn(*args)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            compiled_fn(*args)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e6)
        times.sort()
        result["compile_ns"] = times[len(times)//2]
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
        "strategy_type": "",
        "optimization_strategy": "",
        "multi_kernel_summary": "",
        "plans_tried": 0,
        "iterations_completed": 0,
        "correctness_pass_rate": 0.0,
        "best_direction": "",
        "error": "",
        "wall_time_s": 0.0,
    }

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

    # Measure baselines
    if problem["bench_type"] == "kernelbench":
        baselines = measure_baselines_kb(kernel_path, gpu_id)
    else:
        baselines = measure_baselines_sol(problem["path"], gpu_id)
    result["eager_ns"] = baselines.get("eager_ns", -1)
    result["compile_ns"] = baselines.get("compile_ns", -1)
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = output_dir / f"log_{problem_id.replace('/', '_')}.txt"

    start = time.monotonic()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                env=env, cwd=str(PROJ_ROOT),
            )
        wall_time = time.monotonic() - start
        result["wall_time_s"] = round(wall_time, 1)
        result["status"] = "pass" if proc.returncode == 0 else "kpt_error"
        if proc.returncode != 0:
            try:
                log_tail = log_path.read_text()[-500:]
            except Exception:
                log_tail = ""
            result["error"] += f"exit={proc.returncode}: {log_tail}"
    except Exception as e:
        result["wall_time_s"] = round(time.monotonic() - start, 1)
        result["status"] = "crash"
        result["error"] += str(e)[:300]

    # Parse results from the run_* subdirectory
    _parse_kpt_results(kpt_output, result)

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
            if result["eager_ns"] > 0 and result["best_latency_ns"] > 0:
                result["speedup_vs_eager"] = round(
                    result["eager_ns"] / result["best_latency_ns"], 3
                )
            if result["compile_ns"] > 0 and result["best_latency_ns"] > 0:
                result["speedup_vs_compile"] = round(
                    result["compile_ns"] / result["best_latency_ns"], 3
                )

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
            failures = [m for m in mem if m.get("speedup", 0) <= 1.0]
            result["correctness_pass_rate"] = round(
                len(successes) / max(len(mem), 1), 3
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

        # Print summary
        status = result.get("status", "?")
        speedup = result.get("speedup", 0)
        eager_us = result.get("eager_ns", -1) / 1e3 if result.get("eager_ns", -1) > 0 else -1
        compile_us = result.get("compile_ns", -1) / 1e3 if result.get("compile_ns", -1) > 0 else -1
        baseline_us = result.get("baseline_latency_ns", -1) / 1e3 if result.get("baseline_latency_ns", -1) > 0 else -1
        best_us = result.get("best_latency_ns", -1) / 1e3 if result.get("best_latency_ns", -1) > 0 else -1
        wall = result.get("wall_time_s", 0)

        if status == "pass":
            successes += 1
            print(f"  PASS  eager={eager_us:.1f}µs compile={compile_us:.1f}µs "
                  f"triton_base={baseline_us:.1f}µs best={best_us:.1f}µs "
                  f"speedup={speedup:.2f}x  [{wall:.0f}s]")
        else:
            failures += 1
            err = result.get("error", "")[:80]
            print(f"  {status.upper()}  eager={eager_us:.1f}µs compile={compile_us:.1f}µs  [{wall:.0f}s]  {err}")

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
        print(f"  Median speedup: {median:.2f}x")
        print(f"  >1.0x: {gt1}/{len(speedups)} | >1.2x: {gt12}/{len(speedups)} | >1.5x: {gt15}/{len(speedups)}")

    print(f"  Results: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
