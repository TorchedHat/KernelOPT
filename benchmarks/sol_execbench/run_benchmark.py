"""Batch runner for SOL-ExecBench comparative evaluation.

Orchestrates parallel execution of kernelopt and KernelAgent across
multiple GPUs on SOL-ExecBench problems with all workloads.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from benchmarks.sol_execbench.harness import (
    list_problems,
    load_problem,
    measure_baseline,
    save_baseline_results,
    needs_safetensors,
)
from benchmarks.sol_execbench.adapter_kernelopt import run_kernelopt_track_a
from benchmarks.sol_execbench.adapter_kernelagent import run_kernelagent_track_a


def _run_single_problem(
    problem_meta: dict,
    tool: str,
    track: str,
    results_dir: str,
    naive_kernels_dir: str,
    gpu_id: int,
    T: int,
    N: int,
    K: int,
    max_rounds: int,
    workload_mode: str,
) -> dict:
    """Run a single SOL-ExecBench problem. Called in worker process."""
    pid = problem_meta["problem_id"]
    problem_path = problem_meta["path"]

    naive_path = str(Path(naive_kernels_dir) / f"{pid}.py")
    if not Path(naive_path).exists():
        return {"problem_id": pid, "tool": tool, "track": track,
                "success": False, "error": f"naive kernel not found: {naive_path}"}

    problem = load_problem(problem_path)
    if workload_mode == "primary":
        workload_indices = [0]
    else:
        workload_indices = list(range(problem.num_workloads))

    try:
        if tool == "kernelopt":
            return run_kernelopt_track_a(
                naive_path, problem_path, pid, results_dir,
                T=T, N=N, K=K, gpu_id=gpu_id, workload_indices=workload_indices,
            )
        elif tool == "kernelagent":
            return run_kernelagent_track_a(
                naive_path, problem_path, pid, results_dir,
                max_rounds=max_rounds, gpu_id=gpu_id, workload_indices=workload_indices,
            )
        else:
            return {"problem_id": pid, "tool": tool, "track": track,
                    "success": False, "error": "unknown tool"}
    except Exception as e:
        return {"problem_id": pid, "tool": tool, "track": track,
                "success": False, "error": f"{e}\n{traceback.format_exc()}"}


def _save_result(result: dict, results_dir: str) -> None:
    pid = result.get("problem_id", "unknown")
    tool = result.get("tool", "unknown")
    track = result.get("track", "x")
    out = Path(results_dir) / tool / f"track_{track}" / f"{pid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))


def _measure_one_baseline(prob_meta: dict, gpu_id: int, workload_mode: str):
    """Top-level picklable worker for baseline measurement."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    prob = load_problem(prob_meta["path"])
    indices = [0] if workload_mode == "primary" else list(range(prob.num_workloads))
    results = []
    for wi in indices:
        results.append(measure_baseline(prob, wi))
    return results


def run_baselines(
    problems: list[dict],
    output_path: str,
    gpus: list[int],
    workload_mode: str = "all",
) -> None:
    """Measure baselines for all SOL-ExecBench problems."""
    print(f"Measuring baselines for {len(problems)} SOL-ExecBench problems...")
    all_results = []

    with ProcessPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}
        for i, prob in enumerate(problems):
            gpu = gpus[i % len(gpus)]
            f = pool.submit(_measure_one_baseline, prob, gpu, workload_mode)
            futures[f] = prob

        for f in as_completed(futures):
            prob = futures[f]
            try:
                results = f.result()
                all_results.extend(results)
                for r in results:
                    print(f"  {prob['name']} wl{r.workload_idx}: "
                          f"ref={r.ref_ns/1e6:.2f}ms compile={r.compile_ns/1e6:.2f}ms")
            except Exception as e:
                print(f"  {prob['name']}: FAILED ({e})")

    save_baseline_results(all_results, output_path)
    print(f"Baselines saved to {output_path} ({len(all_results)} workload measurements)")


def run_benchmark(
    problems: list[dict],
    tools: list[str],
    tracks: list[str],
    results_dir: str,
    naive_kernels_dir: str,
    gpus: list[int],
    T: int = 5,
    N: int = 4,
    K: int = 1,
    max_rounds: int = 5,
    workload_mode: str = "all",
) -> list[dict]:
    """Run the full SOL-ExecBench benchmark."""
    tasks = []
    for tool in tools:
        for track in tracks:
            for prob in problems:
                tasks.append((prob, tool, track))

    print(f"Running {len(tasks)} tasks ({len(problems)} problems x {len(tools)} tools x {len(tracks)} tracks)")
    print(f"  Workload mode: {workload_mode}")
    print(f"  GPUs: {gpus}")

    all_results = []
    start = time.monotonic()

    with ProcessPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}
        for i, (prob, tool, track) in enumerate(tasks):
            gpu = gpus[i % len(gpus)]
            f = pool.submit(
                _run_single_problem,
                prob, tool, track, results_dir, naive_kernels_dir, gpu,
                T, N, K, max_rounds, workload_mode,
            )
            futures[f] = (prob, tool, track)

        done = 0
        for f in as_completed(futures):
            done += 1
            prob, tool, track = futures[f]
            try:
                result = f.result()
                all_results.append(result)
                _save_result(result, results_dir)
                ok = "OK" if result.get("success", any(
                    w.get("success") for w in result.get("workloads", [])
                )) else "FAIL"
                elapsed = result.get("total_wall_clock_s", "?")
                print(f"  [{done}/{len(tasks)}] {prob['name']} | {tool} | track_{track} | {ok} | {elapsed}s")
            except Exception as e:
                print(f"  [{done}/{len(tasks)}] {prob['name']} | {tool} | track_{track} | ERROR: {e}")

    total = time.monotonic() - start
    print(f"\nDone: {len(all_results)} results in {total/60:.1f} min")

    summary_path = Path(results_dir) / "benchmark_summary.json"
    summary = {
        "total_tasks": len(tasks),
        "completed": len(all_results),
        "total_time_s": round(total, 1),
        "tools": tools,
        "tracks": tracks,
        "workload_mode": workload_mode,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="SOL-ExecBench comparative benchmark runner")
    parser.add_argument("--tool", type=str, default="both", choices=["kernelopt", "kernelagent", "both"])
    parser.add_argument("--track", type=str, default="a", choices=["a", "b", "both"])
    parser.add_argument("--category", type=str, default=None,
                        help="Filter: gemm, rmsnorm, fused_rmsnorm, gqa, mla, moe")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated problem indices")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output", type=str, default="benchmarks/sol_execbench/results")
    parser.add_argument("--naive-kernels", type=str, default="benchmarks/sol_execbench/naive_kernels")
    parser.add_argument("--workloads", type=str, default="all", choices=["all", "primary"])
    parser.add_argument("-T", type=int, default=5)
    parser.add_argument("-N", type=int, default=4)
    parser.add_argument("-K", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--skip-safetensors", action="store_true",
                        help="Skip problems that need safetensors data")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    problems = list_problems()

    if args.category:
        problems = [p for p in problems if p["category"] == args.category]
    if args.problems:
        ids = set(int(x) for x in args.problems.split(","))
        problems = [p for p in problems if int(p["problem_id"].split("_")[1]) in ids]
    if args.skip_safetensors:
        problems = [p for p in problems if not needs_safetensors(load_problem(p["path"]))]

    print(f"SOL-ExecBench Comparative Evaluation")
    print(f"  Problems: {len(problems)}")
    print(f"  GPUs: {gpus}")

    if args.baselines_only:
        run_baselines(problems, f"{args.output}/baselines.json", gpus, args.workloads)
        return

    tools = ["kernelopt", "kernelagent"] if args.tool == "both" else [args.tool]
    tracks = ["a", "b"] if args.track == "both" else [args.track]

    run_benchmark(
        problems, tools, tracks, args.output, args.naive_kernels, gpus,
        T=args.T, N=args.N, K=args.K, max_rounds=args.max_rounds,
        workload_mode=args.workloads,
    )


if __name__ == "__main__":
    main()
