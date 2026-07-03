#!/usr/bin/env python3
"""Full model-level wall-clock benchmark with pipeline optimization.

Measures: eager, compile, synth (baseline compiled model), best (restitched optimized),
plus kernel-level NCU times and Amdahl's kernel execution percentage.

All wall-clock timings use triton.testing.do_bench (CUDA events internally,
25ms warmup, 100ms rep, median) — same method across all measurements.

Amdahl's kernel_pct uses NCU --clock-control none + NVTX isolation so that
T_kernel and T_wall are both at GPU boost clocks — directly comparable.

Usage:
    python3 run_full_bench.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJ_ROOT / "kernelopt"))
sys.path.insert(0, str(PROJ_ROOT))

# Ensure CUDA tools (ncu) are on PATH
_CUDA_BIN = "/usr/local/cuda-12.8/bin"
if _CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")

PROBLEMS = [
    ("kb_l1_005", "tests_suite/benchmarks/kernelbench/level_1/kernel_005.py"),
    ("kb_l1_010", "tests_suite/benchmarks/kernelbench/level_1/kernel_010.py"),
    ("kb_l2_003", "tests_suite/benchmarks/kernelbench/level_2/kernel_003.py"),
    ("kb_l2_005", "tests_suite/benchmarks/kernelbench/level_2/kernel_005.py"),
    ("kb_l3_003", "tests_suite/benchmarks/kernelbench/level_3/kernel_003.py"),
    ("kb_l3_005", "tests_suite/benchmarks/kernelbench/level_3/kernel_005.py"),
    ("sol_l1_003", "pipeline_output/benchmark_suite/sol_wrappers/sol_l1_003.py"),
    ("sol_l1_007", "pipeline_output/benchmark_suite/sol_wrappers/sol_l1_007.py"),
    ("sol_train_003", "pipeline_output/benchmark_suite/sol_wrappers/sol_train_003.py"),
    ("sol_train_007", "pipeline_output/benchmark_suite/sol_wrappers/sol_train_007.py"),
    ("sol_quant_003", "pipeline_output/benchmark_suite/sol_wrappers/sol_quant_003.py"),
]

OUT = Path("pipeline_output/full_bench")
OUT.mkdir(parents=True, exist_ok=True)

TIMEOUT_S = 1800  # 30 min safety net per subprocess


# ---------------------------------------------------------------------------
# Timing functions — all use triton.testing.do_bench for consistency
# ---------------------------------------------------------------------------

def time_pytorch_model(pytorch_file: str, mode: str = "eager") -> float:
    """Time a PyTorch model end-to-end via do_bench. Returns median ms or -1."""
    compile_line = "model = torch.compile(model)" if mode == "compile" else ""
    script = f"""\
import json, torch, importlib.util, sys
torch.cuda.set_device(0)
from triton.testing import do_bench

spec = importlib.util.spec_from_file_location("m", "{pytorch_file}")
mod = importlib.util.module_from_spec(spec); sys.modules["m"] = mod
spec.loader.exec_module(mod)
Model = getattr(mod, "Model", None)
get_ii = getattr(mod, "get_init_inputs", lambda: [])
get_i = getattr(mod, "get_inputs", None)
if not Model or not get_i:
    print(json.dumps({{"error": "missing Model or get_inputs"}})); sys.exit(0)
model = Model(*get_ii()).cuda().eval()
{compile_line}
inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in get_i()]
with torch.no_grad():
    median_ms = do_bench(lambda: model(*inputs), warmup=25, rep=100)
print(json.dumps({{"median_ms": round(float(median_ms), 4)}}))
"""
    return _run_timing_script(script)


def time_synth_model(synth_path: str) -> float:
    """Time a synthesized_triton.py or restitched model via do_bench. Returns median ms or -1."""
    script = f"""\
import json, torch, importlib.util, sys
torch.cuda.set_device(0)
from triton.testing import do_bench

spec = importlib.util.spec_from_file_location("m", "{synth_path}")
mod = importlib.util.module_from_spec(spec); sys.modules["m"] = mod
spec.loader.exec_module(mod)
fn = getattr(mod, "kernel_function", None) or getattr(mod, "call", None)
gi = getattr(mod, "get_inputs", None)
if not fn or not gi:
    print(json.dumps({{"error": "missing kernel_function or get_inputs"}})); sys.exit(0)
inputs = gi()
with torch.no_grad():
    median_ms = do_bench(lambda: fn(*inputs), warmup=25, rep=100)
print(json.dumps({{"median_ms": round(float(median_ms), 4)}}))
"""
    return _run_timing_script(script)


def _run_timing_script(script: str) -> float:
    """Execute a timing subprocess script. Returns median_ms or -1."""
    try:
        p = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        if p.returncode == 0:
            for line in p.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    if "error" in data:
                        sys.stderr.write(f"  timing error: {data['error']}\n")
                        return -1
                    return data.get("median_ms", -1)
        sys.stderr.write(f"  timing exit={p.returncode}: {(p.stderr or '')[-200:]}\n")
    except subprocess.TimeoutExpired:
        sys.stderr.write("  timing timeout\n")
    except Exception as e:
        sys.stderr.write(f"  timing exception: {e}\n")
    return -1


# ---------------------------------------------------------------------------
# NCU boost duration — Amdahl's kernel% at boost clocks
# ---------------------------------------------------------------------------

def measure_kernel_boost_ns(kernel_path: str, output_dir: Path) -> float:
    """Get total GPU kernel duration at boost clocks for Amdahl's analysis.

    Uses NCU --clock-control none + NVTX markers to isolate the measured
    call from warmup/JIT. Returns total kernel duration in ns, or -1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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
        from kernelopt.agents.profiler.ncu_runner import run_ncu_boost
        kernel_ns = run_ncu_boost(wrapper_path, str(output_dir))
        return kernel_ns if kernel_ns > 0 else -1
    except Exception as e:
        sys.stderr.write(f"  ncu_boost error: {e}\n")
        return -1
    finally:
        try:
            Path(wrapper_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pipeline runner + structured result extraction
# ---------------------------------------------------------------------------

def run_pipeline(name: str, pytorch_file: str, out_dir: Path) -> dict:
    """Run kernelopt pipeline and return structured metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "pipeline.log"

    cmd = [
        "kernelopt", "run", pytorch_file,
        "-T", "1", "-N", "3", "-K", "2",
        "--no-run-subdir", "-o", str(out_dir),
        "--kernel-args", "--profile-only",
        "--launch-skip", "0", "--launch-count", "10",
        "--search-policy", "ucb",
    ]
    try:
        with open(log_file, "w") as lf:
            subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT, text=True,
                timeout=TIMEOUT_S,
            )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    result = {"status": "ok"}

    # Extract NCU metrics from checkpoint (structured, not log regex)
    ckpt_path = out_dir / "pipeline_checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
            gs = ckpt.get("graph_state", {})
            result["kernel_baseline_ns"] = gs.get("baseline_latency_ns", -1)
            result["kernel_best_ns"] = gs.get("best_latency_ns", -1)
            result["ncu_speedup"] = gs.get("speedup", 0)
            if result["ncu_speedup"] == 0:
                bl = result["kernel_baseline_ns"]
                bt = result["kernel_best_ns"]
                if bl > 0 and bt > 0:
                    result["ncu_speedup"] = round(bl / bt, 3)
            result["iterations"] = gs.get("iteration", 0)
            result["correctness_pass_rate"] = 0.0
            ir_list = gs.get("iteration_results", [])
            total_cands = sum(len(ir.get("candidates", [])) for ir in ir_list)
            result["plans_tried"] = total_cands
        except Exception:
            pass

    # Find output files
    synth_path = out_dir / "synthesis" / "synthesized_triton.py"
    if synth_path.exists():
        result["synth_path"] = str(synth_path)

    best_kernel = out_dir / "best_kernel.py"
    if best_kernel.exists():
        result["best_kernel_path"] = str(best_kernel)

    restitched = sorted(out_dir.glob("*_restitched.py"))
    if restitched:
        result["best_restitched_path"] = str(restitched[-1])

    return result


# ---------------------------------------------------------------------------
# Speedup helper
# ---------------------------------------------------------------------------

def sp(a, b):
    """Compute speedup a/b, return 0 if either is invalid."""
    if a and b and a > 0 and b > 0:
        return round(a / b, 3)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    csv_lines = [
        "problem,eager_ms,compile_ms,synth_ms,best_ms,"
        "synth_sp,opt_sp,vs_eager,vs_compile,"
        "kernel_bl_ns,kernel_best_ns,ncu_sp,"
        "kernel_boost_ns,kernel_pct,"
        "baseline_s,pipeline_s,bench_s,total_s"
    ]

    total_start = time.monotonic()

    for name, path in PROBLEMS:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        if not Path(path).exists():
            print("  MISSING")
            results.append({"name": name, "status": "missing"})
            continue

        row = {"name": name}
        problem_start = time.monotonic()

        try:
            # ── 1. Baseline timings (eager + compile) ──
            baseline_start = time.monotonic()
            print("  Timing eager...", end=" ", flush=True)
            row["eager_ms"] = time_pytorch_model(path, "eager")
            print(f"{row['eager_ms']:.3f} ms")

            print("  Timing compile...", end=" ", flush=True)
            row["compile_ms"] = time_pytorch_model(path, "compile")
            print(f"{row['compile_ms']:.3f} ms")
            row["baseline_s"] = round(time.monotonic() - baseline_start, 1)

            # ── 2. Run optimization pipeline ──
            pipe_dir = OUT / name
            print(f"  Running pipeline (T=1, N=3, K=2, UCB)...")
            pipeline_start = time.monotonic()
            pipe = run_pipeline(name, path, pipe_dir)
            row["pipeline_s"] = round(time.monotonic() - pipeline_start, 1)
            row.update(pipe)

            # ── 3. Wall-clock timing of synth + best (all do_bench) ──
            bench_start = time.monotonic()

            synth_ms = -1.0
            if pipe.get("synth_path"):
                print("  Timing synth model...", end=" ", flush=True)
                synth_ms = time_synth_model(pipe["synth_path"])
                row["synth_ms"] = synth_ms
                print(f"{synth_ms:.3f} ms")

            best_ms = -1.0
            if pipe.get("best_restitched_path"):
                print("  Timing optimized model...", end=" ", flush=True)
                best_ms = time_synth_model(pipe["best_restitched_path"])
                row["best_ms"] = best_ms
                print(f"{best_ms:.3f} ms")
            else:
                print("  No restitched model produced")
                row["best_ms"] = synth_ms

            # ── 4. Amdahl's kernel_pct via NCU --clock-control none + NVTX ──
            kernel_boost_ns = -1.0
            kernel_file = pipe.get("best_kernel_path")
            if kernel_file and Path(kernel_file).exists():
                print("  Measuring kernel boost duration (NCU --clock-control none)...", end=" ", flush=True)
                kernel_boost_ns = measure_kernel_boost_ns(
                    kernel_file, pipe_dir / "_boost_profile"
                )
                row["kernel_boost_ns"] = kernel_boost_ns
                if kernel_boost_ns > 0:
                    print(f"{kernel_boost_ns/1e3:.1f} µs")
                else:
                    print("failed")

            row["bench_s"] = round(time.monotonic() - bench_start, 1)

            # ── 5. Compute all speedup metrics ──
            eager = row.get("eager_ms", -1)
            compile_ = row.get("compile_ms", -1)
            best = row.get("best_ms", -1)

            row["synth_sp"] = sp(eager, synth_ms)
            row["opt_sp"] = sp(synth_ms, best)
            row["vs_eager"] = sp(eager, best)
            row["vs_compile"] = sp(compile_, best)

            kb_ns = row.get("kernel_baseline_ns", -1)
            kbest_ns = row.get("kernel_best_ns", -1)
            row["ncu_sp"] = sp(kb_ns, kbest_ns)

            # Amdahl's kernel_pct: both at boost clocks → directly comparable
            if kernel_boost_ns > 0 and best > 0:
                best_wall_ns = best * 1e6
                row["kernel_pct"] = round(
                    min(kernel_boost_ns / best_wall_ns * 100, 100.0), 1
                )
            else:
                row["kernel_pct"] = 0

        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  ERROR: {row['error']}")

        row["total_s"] = round(time.monotonic() - problem_start, 1)

        # ── Print summary ──
        eager = row.get("eager_ms", -1)
        compile_ = row.get("compile_ms", -1)
        synth_ms = row.get("synth_ms", -1)
        best = row.get("best_ms", -1)

        print(f"  ---")
        print(f"  Eager: {eager:.2f}ms | Compile: {compile_:.2f}ms | "
              f"Synth: {synth_ms:.2f}ms | Best: {best:.2f}ms")
        print(f"  synth_sp: {row.get('synth_sp', 0)}x | "
              f"opt_sp: {row.get('opt_sp', 0)}x | "
              f"vs_eager: {row.get('vs_eager', 0)}x | "
              f"vs_compile: {row.get('vs_compile', 0)}x")
        print(f"  Kernel NCU: {row.get('kernel_baseline_ns', 0)/1e3:.1f}µs → "
              f"{row.get('kernel_best_ns', 0)/1e3:.1f}µs "
              f"(ncu={row.get('ncu_sp', 0)}x) | "
              f"kernel%: {row.get('kernel_pct', 0)}%")
        print(f"  [baseline={row.get('baseline_s', 0):.0f}s "
              f"pipeline={row.get('pipeline_s', 0):.0f}s "
              f"bench={row.get('bench_s', 0):.0f}s "
              f"total={row.get('total_s', 0):.0f}s]")

        results.append(row)
        _save_results(results, csv_lines, row)

    total_time = time.monotonic() - total_start
    _print_summary_table(results, total_time)
    print(f"\nResults saved to {OUT / 'results.csv'} and {OUT / 'results.json'}")


def _save_results(results: list, csv_lines: list, row: dict) -> None:
    """Save results incrementally after each problem."""
    eager = row.get("eager_ms", -1)
    compile_ = row.get("compile_ms", -1)
    synth_ms = row.get("synth_ms", -1)
    best = row.get("best_ms", -1)

    csv_lines.append(
        f"{row['name']},{eager},{compile_},{synth_ms},{best},"
        f"{row.get('synth_sp', 0)},{row.get('opt_sp', 0)},"
        f"{row.get('vs_eager', 0)},{row.get('vs_compile', 0)},"
        f"{row.get('kernel_baseline_ns', -1)},{row.get('kernel_best_ns', -1)},"
        f"{row.get('ncu_sp', 0)},"
        f"{row.get('kernel_boost_ns', -1)},{row.get('kernel_pct', 0)},"
        f"{row.get('baseline_s', 0)},{row.get('pipeline_s', 0)},"
        f"{row.get('bench_s', 0)},{row.get('total_s', 0)}"
    )

    (OUT / "results.csv").write_text("\n".join(csv_lines) + "\n")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))


def _print_summary_table(results: list, total_time: float) -> None:
    """Print final summary table."""
    print(f"\n{'='*150}")
    hdr = (
        f"{'Problem':<14} {'Eager':>8} {'Compile':>8} {'Synth':>8} {'Best':>8} "
        f"{'SynthSp':>8} {'OptSp':>8} {'vsEager':>8} {'vsCmpl':>8} "
        f"{'NCU_sp':>7} {'Kern%':>6} {'Time':>6}"
    )
    print(hdr)
    print(
        f"{'':14} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} "
        f"{'':>8} {'':>8} {'':>8} {'':>8} "
        f"{'':>7} {'':>6} {'(min)':>6}"
    )
    print(f"{'-'*150}")

    for r in results:
        if r.get("status") in ("missing", "timeout", "error"):
            print(f"{r['name']:<14} {r.get('status', '?').upper():>8}  "
                  f"{r.get('error', '')[:80]}")
            continue

        def f(v, suf=""):
            return f"{v:.2f}{suf}" if v and v > 0 else "—"

        mins = r.get("total_s", 0) / 60
        print(
            f"{r['name']:<14} "
            f"{f(r.get('eager_ms')):>8} "
            f"{f(r.get('compile_ms')):>8} "
            f"{f(r.get('synth_ms')):>8} "
            f"{f(r.get('best_ms')):>8} "
            f"{f(r.get('synth_sp'), 'x'):>8} "
            f"{f(r.get('opt_sp'), 'x'):>8} "
            f"{f(r.get('vs_eager'), 'x'):>8} "
            f"{f(r.get('vs_compile'), 'x'):>8} "
            f"{f(r.get('ncu_sp'), 'x'):>7} "
            f"{r.get('kernel_pct', 0):>5.1f}% "
            f"{mins:>5.1f}m"
        )

    total_mins = total_time / 60
    print(f"{'-'*150}")
    print(f"{'TOTAL':>14} {'':>66} {total_mins:>49.1f}m")
    print(f"{'='*150}")


if __name__ == "__main__":
    main()
