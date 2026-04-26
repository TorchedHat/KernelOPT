"""Analysis script for SOL-ExecBench comparative evaluation.

Per-category and per-problem metrics, strategy analysis, and paper outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_all_results(results_dir: str) -> list[dict]:
    results = []
    for f in sorted(Path(results_dir).rglob("*.json")):
        if f.name in ("benchmark_summary.json", "baselines.json", "generation_summary.json"):
            continue
        try:
            data = json.loads(f.read_text())
            if "problem_id" in data and "tool" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_baselines(path: str) -> dict:
    """Load baselines indexed by (problem_id, workload_idx)."""
    data = json.loads(Path(path).read_text())
    index = {}
    for d in data:
        key = (d["problem_id"], d["workload_idx"])
        index[key] = d
    return index


def compute_category_metrics(
    results: list[dict],
    baselines: dict,
) -> list[dict]:
    """Compute per-category aggregate metrics."""
    by_cat = defaultdict(list)
    for r in results:
        cat = r.get("category", "other")
        tool = r.get("tool", "unknown")
        by_cat[(cat, tool)].append(r)

    rows = []
    for (cat, tool), cat_results in sorted(by_cat.items()):
        total_wl = 0
        correct_wl = 0
        speedups_ref = []
        speedups_compile = []

        for r in cat_results:
            for wl in r.get("workloads", []):
                total_wl += 1
                if wl.get("correct", wl.get("success", False)):
                    correct_wl += 1
                    bl_key = (r["problem_id"], wl.get("workload_idx", 0))
                    bl = baselines.get(bl_key, {})
                    ref_ns = bl.get("ref_ns", float("inf"))
                    compile_ns = bl.get("compile_ns", float("inf"))
                    best_ns = wl.get("best_ns", wl.get("best_kernel_ns", float("inf")))

                    if best_ns > 0 and best_ns != float("inf"):
                        if ref_ns > 0 and ref_ns != float("inf"):
                            speedups_ref.append(math.log(ref_ns / best_ns))
                        if compile_ns > 0 and compile_ns != float("inf"):
                            speedups_compile.append(math.log(compile_ns / best_ns))

        pass_rate = correct_wl / total_wl if total_wl > 0 else 0
        geomean_ref = math.exp(sum(speedups_ref) / len(speedups_ref)) if speedups_ref else 0
        geomean_compile = math.exp(sum(speedups_compile) / len(speedups_compile)) if speedups_compile else 0

        rows.append({
            "category": cat,
            "tool": tool,
            "n_problems": len(cat_results),
            "n_workloads": total_wl,
            "pass_rate": round(pass_rate, 3),
            "geomean_vs_ref": round(geomean_ref, 3),
            "geomean_vs_compile": round(geomean_compile, 3),
        })

    return rows


def compute_per_problem(results: list[dict], baselines: dict) -> list[dict]:
    """Per-problem detail rows for the paper table."""
    rows = []
    for r in results:
        pid = r.get("problem_id", "")
        tool = r.get("tool", "")
        wl_speedups = []
        for wl in r.get("workloads", []):
            bl_key = (pid, wl.get("workload_idx", 0))
            bl = baselines.get(bl_key, {})
            ref_ns = bl.get("ref_ns", float("inf"))
            best_ns = wl.get("best_ns", float("inf"))
            if best_ns > 0 and ref_ns > 0 and best_ns != float("inf"):
                wl_speedups.append(ref_ns / best_ns)

        iterations = r.get("iterations", [])
        best_strategy = iterations[0].get("direction", "") if iterations else ""

        rows.append({
            "problem_id": pid,
            "problem_name": r.get("problem_name", ""),
            "category": r.get("category", ""),
            "tool": tool,
            "n_workloads": len(r.get("workloads", [])),
            "avg_speedup_vs_ref": round(sum(wl_speedups) / len(wl_speedups), 3) if wl_speedups else 0,
            "best_strategy": best_strategy,
            "total_wall_clock_s": r.get("total_wall_clock_s", 0),
        })
    return rows


def analyze_strategies(results: list[dict]) -> dict:
    stats = defaultdict(lambda: {"count": 0, "speedups": [], "tools": Counter()})
    for r in results:
        tool = r.get("tool", "unknown")
        for it in r.get("iterations", []):
            d = it.get("direction", "unknown")
            stats[d]["count"] += 1
            s = it.get("speedup", 0)
            if s and s > 0:
                stats[d]["speedups"].append(s)
            stats[d]["tools"][tool] += 1
    for v in stats.values():
        speeds = v["speedups"]
        v["avg_speedup"] = sum(speeds) / len(speeds) if speeds else 0
        v["tools"] = dict(v["tools"])
    return dict(stats)


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze SOL-ExecBench results")
    parser.add_argument("--results-dir", type=str, default="benchmarks/sol_execbench/results")
    parser.add_argument("--baselines", type=str, default="benchmarks/sol_execbench/results/baselines.json")
    parser.add_argument("--output-dir", type=str, default="benchmarks/sol_execbench/analysis")
    args = parser.parse_args()

    print("Loading results...")
    results = load_all_results(args.results_dir)
    print(f"  {len(results)} results loaded")

    baselines = {}
    if Path(args.baselines).exists():
        baselines = load_baselines(args.baselines)
        print(f"  {len(baselines)} baseline measurements")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cat_rows = compute_category_metrics(results, baselines)
    write_csv(cat_rows, str(out / "category_summary.csv"))
    print("\n  Category Summary:")
    for r in cat_rows:
        print(f"    {r['category']:20} {r['tool']:15} pass={r['pass_rate']:.0%} "
              f"geomean(ref)={r['geomean_vs_ref']:.2f}x "
              f"geomean(compile)={r['geomean_vs_compile']:.2f}x")

    prob_rows = compute_per_problem(results, baselines)
    write_csv(prob_rows, str(out / "per_problem.csv"))

    strat = analyze_strategies(results)
    strat_rows = [
        {"strategy": k, "count": v["count"], "avg_speedup": round(v["avg_speedup"], 3),
         "kernelopt": v["tools"].get("kernelopt", 0), "kernelagent": v["tools"].get("kernelagent", 0)}
        for k, v in sorted(strat.items(), key=lambda x: -x[1]["count"])
    ]
    write_csv(strat_rows, str(out / "strategies.csv"))

    print(f"\n  Analysis written to {out}/")


if __name__ == "__main__":
    main()
