#!/usr/bin/env python3
"""Evaluate Feature 4 synthesis pipeline on KernelBench problems.

Usage:
    python eval_synthesis.py --level 1 --gpu 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_ROOT = Path(__file__).parent / "tests_suite" / "benchmarks" / "kernelbench"
KERNELOPT_PKG = Path(__file__).parent / "kernelopt"
RESULTS_DIR = Path(__file__).parent / "eval_synthesis_results"

sys.path.insert(0, str(KERNELOPT_PKG))


def run_one(problem_path: str, gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch, SynthesisError
    from kernelopt.synthesis.wrap_kernel import wrap_inductor_output

    result = {
        "problem": Path(problem_path).stem,
        "problem_path": str(problem_path),
        "gpu": gpu_id,
        "status": "unknown",
        "n_triton_kernels": 0,
        "has_extern_kernels": False,
        "op_description": "",
        "error": "",
        "compile_time_s": 0.0,
        "wrap_time_s": 0.0,
        "exec_time_s": 0.0,
        "total_time_s": 0.0,
        "exec_ok": False,
    }

    src = Path(problem_path).read_text()
    for line in src.split("\n"):
        if "return" in line and "self" not in line.lower() and "return []" not in line:
            result["op_description"] = line.strip()[:100]
            break

    t_total = time.time()

    t0 = time.time()
    try:
        extraction_results = extract_triton_from_pytorch(problem_path)
    except SynthesisError as e:
        result["compile_time_s"] = round(time.time() - t0, 2)
        result["total_time_s"] = round(time.time() - t_total, 2)
        result["status"] = "synthesis_error"
        result["error"] = str(e)[:500]
        return result
    except Exception as e:
        result["compile_time_s"] = round(time.time() - t0, 2)
        result["total_time_s"] = round(time.time() - t_total, 2)
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return result
    result["compile_time_s"] = round(time.time() - t0, 2)

    if not extraction_results:
        result["total_time_s"] = round(time.time() - t_total, 2)
        result["status"] = "synthesis_error"
        result["error"] = "torch.compile returned empty source list"
        return result

    primary = extraction_results[0]
    n_kernels = len(primary["triton_kernels"])
    has_extern = "extern_kernels" in primary["inductor_source"]
    result["n_triton_kernels"] = n_kernels
    result["has_extern_kernels"] = has_extern

    if n_kernels == 0 and not has_extern:
        result["total_time_s"] = round(time.time() - t_total, 2)
        result["status"] = "no_triton_kernel"
        return result

    t1 = time.time()
    try:
        wrapped = wrap_inductor_output(
            inductor_source=primary["inductor_source"],
            original_pytorch_source=src,
            triton_kernel_sources=primary["triton_kernels"],
        )
    except Exception as e:
        result["wrap_time_s"] = round(time.time() - t1, 2)
        result["total_time_s"] = round(time.time() - t_total, 2)
        result["status"] = "wrap_error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return result
    result["wrap_time_s"] = round(time.time() - t1, 2)

    outpath = f"/tmp/eval_synth_{Path(problem_path).stem}.py"
    Path(outpath).write_text(wrapped)

    t2 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, outpath, "--profile-only"],
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
        result["exec_time_s"] = round(time.time() - t2, 2)
        if r.returncode == 0:
            result["status"] = "pass"
            result["exec_ok"] = True
        else:
            err_lines = (r.stderr or r.stdout).strip().split("\n")
            last_err = err_lines[-1][:300] if err_lines else "?"
            result["status"] = "exec_fail"
            result["error"] = last_err
    except Exception as e:
        result["exec_time_s"] = round(time.time() - t2, 2)
        result["status"] = "exec_error"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    result["total_time_s"] = round(time.time() - t_total, 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=999)
    args = parser.parse_args()

    level_dir = BENCH_ROOT / f"level_{args.level}"
    problems = sorted(level_dir.glob("kernel_*.py"))
    problems = [p for p in problems if args.start <= int(p.stem.split("_")[1]) <= args.end]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"level_{args.level}_gpu{args.gpu}.json"

    print(f"{'='*70}")
    print(f"  KernelBench L{args.level} Synthesis Evaluation")
    print(f"  Problems: {len(problems)} | GPU: {args.gpu} | No timeouts")
    print(f"  Output: {out_file}")
    print(f"{'='*70}\n")

    all_results = []
    counts = {"pass": 0}
    total = len(problems)
    wall_start = time.time()

    for i, problem in enumerate(problems, 1):
        print(f"[{i:3d}/{total}] {problem.stem} ... ", end="", flush=True)
        result = run_one(str(problem), args.gpu)
        all_results.append(result)

        s = result["status"]
        counts[s] = counts.get(s, 0) + 1
        ct, et, tt = result["compile_time_s"], result["exec_time_s"], result["total_time_s"]

        if s == "pass":
            print(f"PASS  compile={ct:.1f}s exec={et:.1f}s total={tt:.1f}s  [{result['n_triton_kernels']}k]")
        else:
            print(f"{s.upper()}  {tt:.1f}s  {result['error'][:60]}")

        with open(out_file, "w") as f:
            json.dump({"level": args.level, "gpu": args.gpu, "total": total,
                        "completed": i, "wall_time_s": round(time.time()-wall_start,1),
                        "counts": counts, "results": all_results}, f, indent=2)

    wall_total = time.time() - wall_start
    print(f"\n{'='*70}")
    print(f"  SUMMARY: L{args.level} ({total} problems, {wall_total:.0f}s)")
    print(f"{'='*70}")
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {c:3d}/{total} ({100*c/total:5.1f}%)")
    print(f"\n  Results: {out_file}")


if __name__ == "__main__":
    main()
