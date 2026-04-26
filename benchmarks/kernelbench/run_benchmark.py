"""Batch runner for KernelBench comparative evaluation.

Orchestrates parallel execution of kernelopt and KernelAgent across
multiple GPUs on KernelBench problems.
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

from benchmarks.kernelbench.harness import (
    list_problems,
    load_problem,
    measure_baselines,
    save_baseline_results,
)
from benchmarks.kernelbench.adapter_kernelopt import (
    run_kernelopt_track_a,
    run_kernelopt_track_b,
)
from benchmarks.kernelbench.adapter_kernelagent import (
    run_kernelagent_track_a,
    run_kernelagent_track_b,
)


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
) -> dict:
    """Run a single problem with a single tool on a single track. Called in worker process."""
    pid = problem_meta["problem_id"]
    problem_path = problem_meta["path"]

    naive_path = str(Path(naive_kernels_dir) / f"{pid}.py")
    if not Path(naive_path).exists():
        return {
            "problem_id": pid,
            "tool": tool,
            "track": track,
            "success": False,
            "error": f"naive kernel not found: {naive_path}",
        }

    try:
        if tool == "kernelopt" and track == "a":
            return run_kernelopt_track_a(
                naive_path, pid, results_dir, T=T, N=N, K=K, gpu_id=gpu_id,
            )
        elif tool == "kernelopt" and track == "b":
            return run_kernelopt_track_b(
                problem_path, naive_path, pid, results_dir, T=T, N=N, K=K, gpu_id=gpu_id,
            )
        elif tool == "kernelagent" and track == "a":
            return run_kernelagent_track_a(
                problem_path, naive_path, pid, results_dir,
                max_rounds=max_rounds, gpu_id=gpu_id,
            )
        elif tool == "kernelagent" and track == "b":
            return run_kernelagent_track_b(
                problem_path, pid, results_dir,
                max_rounds=max_rounds, gpu_id=gpu_id,
            )
        else:
            return {"problem_id": pid, "tool": tool, "track": track, "success": False, "error": f"unknown combo"}
    except Exception as e:
        return {
            "problem_id": pid,
            "tool": tool,
            "track": track,
            "success": False,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def _save_result(result: dict, results_dir: str) -> None:
    """Save a single problem result to JSON."""
    pid = result.get("problem_id", "unknown")
    tool = result.get("tool", "unknown")
    track = result.get("track", "x")
    out = Path(results_dir) / tool / f"track_{track}" / f"{pid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if k != "best_kernel_source"}
    out.write_text(json.dumps(serializable, indent=2, default=str))


def run_baselines(
    problems: list[dict],
    output_path: str,
    gpus: list[int],
) -> None:
    """Measure eager + torch.compile baselines for all problems."""
    print(f"Measuring baselines for {len(problems)} problems on GPUs {gpus}...")
    results = []

    def _measure_one(prob_meta, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        prob = load_problem(prob_meta["path"])
        return measure_baselines(prob)

    with ProcessPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}
        for i, prob in enumerate(problems):
            gpu = gpus[i % len(gpus)]
            f = pool.submit(_measure_one, prob, gpu)
            futures[f] = prob

        for f in as_completed(futures):
            prob = futures[f]
            try:
                result = f.result()
                results.append(result)
                status = f"eager={result.eager_ns/1e6:.2f}ms compile={result.compile_ns/1e6:.2f}ms"
                print(f"  {prob['problem_id']}: {status}")
            except Exception as e:
                print(f"  {prob['problem_id']}: FAILED ({e})")

    save_baseline_results(results, output_path)
    print(f"Baselines saved to {output_path}")


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
) -> list[dict]:
    """Run the full benchmark across tools and tracks with GPU parallelism."""
    tasks = []
    for tool in tools:
        for track in tracks:
            for prob in problems:
                tasks.append((prob, tool, track))

    print(f"Running {len(tasks)} tasks ({len(problems)} problems x {len(tools)} tools x {len(tracks)} tracks)")
    print(f"  GPUs: {gpus} ({len(gpus)} workers)")

    all_results = []
    start = time.monotonic()

    with ProcessPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {}
        for i, (prob, tool, track) in enumerate(tasks):
            gpu = gpus[i % len(gpus)]
            f = pool.submit(
                _run_single_problem,
                prob, tool, track, results_dir, naive_kernels_dir, gpu,
                T, N, K, max_rounds,
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

                ok = "OK" if result.get("success") else "FAIL"
                elapsed = result.get("wall_clock_s", "?")
                print(f"  [{done}/{len(tasks)}] {prob['problem_id']} | {tool} | track_{track} | {ok} | {elapsed}s")
            except Exception as e:
                print(f"  [{done}/{len(tasks)}] {prob['problem_id']} | {tool} | track_{track} | ERROR: {e}")

    total = time.monotonic() - start
    print(f"\nDone: {len(all_results)} results in {total/60:.1f} min")

    summary_path = Path(results_dir) / "benchmark_summary.json"
    summary = {
        "total_tasks": len(tasks),
        "completed": len(all_results),
        "succeeded": sum(1 for r in all_results if r.get("success")),
        "failed": sum(1 for r in all_results if not r.get("success")),
        "total_time_s": round(total, 1),
        "tools": tools,
        "tracks": tracks,
        "levels": sorted(set(p["level"] for p in problems)),
        "gpus": gpus,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="KernelBench comparative benchmark runner")
    parser.add_argument("--tool", type=str, default="both", choices=["kernelopt", "kernelagent", "both"])
    parser.add_argument("--track", type=str, default="both", choices=["a", "b", "both"])
    parser.add_argument("--levels", type=str, default="1,2,3")
    parser.add_argument("--problems", type=str, default=None,
                        help="Problem range (e.g. 1-10) or comma-separated IDs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output", type=str, default="benchmarks/kernelbench/results")
    parser.add_argument("--naive-kernels", type=str, default="benchmarks/kernelbench/naive_kernels")
    parser.add_argument("-T", type=int, default=5, help="kernelopt iterations")
    parser.add_argument("-N", type=int, default=4, help="kernelopt plans per iteration")
    parser.add_argument("-K", type=int, default=1, help="kernelopt attempts per plan")
    parser.add_argument("--max-rounds", type=int, default=5, help="KernelAgent max optimization rounds")
    parser.add_argument("--baselines-only", action="store_true", help="Only measure baselines")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    gpus = [int(x) for x in args.gpus.split(",")]
    problems = list_problems(levels=levels)

    if args.problems:
        if "-" in args.problems:
            lo, hi = args.problems.split("-")
            ids = set(range(int(lo), int(hi) + 1))
        else:
            ids = set(int(x) for x in args.problems.split(","))
        problems = [p for p in problems if int(p["path"].split("kernel_")[1].split(".")[0]) in ids]

    print(f"KernelBench Comparative Evaluation")
    print(f"  Problems: {len(problems)} (levels {levels})")
    print(f"  GPUs: {gpus}")

    if args.baselines_only:
        run_baselines(problems, f"{args.output}/baselines.json", gpus)
        return

    tools = ["kernelopt", "kernelagent"] if args.tool == "both" else [args.tool]
    tracks = ["a", "b"] if args.track == "both" else [args.track]

    run_benchmark(
        problems, tools, tracks, args.output, args.naive_kernels, gpus,
        T=args.T, N=args.N, K=args.K, max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
