"""Analysis script for KernelBench comparative evaluation.

Computes fast_p metrics, geometric mean speedups, strategy analysis,
and generates LaTeX tables + matplotlib figures for the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import csv


def load_all_results(results_dir: str) -> list[dict]:
    """Load all per-problem JSON results from the results directory."""
    results = []
    rdir = Path(results_dir)
    for json_file in sorted(rdir.rglob("*.json")):
        if json_file.name in ("benchmark_summary.json", "baselines.json", "generation_summary.json"):
            continue
        try:
            data = json.loads(json_file.read_text())
            if "problem_id" in data and "tool" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_baselines(baselines_path: str) -> dict[str, dict]:
    """Load baselines indexed by problem_id."""
    data = json.loads(Path(baselines_path).read_text())
    return {d["problem_id"]: d for d in data}


def _level_from_pid(pid: str) -> int:
    if "level_1" in pid:
        return 1
    elif "level_2" in pid:
        return 2
    elif "level_3" in pid:
        return 3
    return 0


def compute_fast_p(
    results: list[dict],
    baselines: dict[str, dict],
    baseline_key: str = "eager_ns",
    p: float = 1.0,
) -> float:
    """Compute fast_p: fraction of problems that are correct AND speedup > p."""
    if not results:
        return 0.0
    qualifying = 0
    for r in results:
        if not r.get("correct", r.get("success", False)):
            continue
        pid = r["problem_id"]
        bl = baselines.get(pid, {})
        bl_ns = bl.get(baseline_key, float("inf"))
        best_ns = r.get("best_kernel_ns", r.get("best_ns", float("inf")))
        if best_ns > 0 and bl_ns > 0:
            speedup = bl_ns / best_ns
            if speedup > p:
                qualifying += 1
    return qualifying / len(results)


def compute_geomean_speedup(
    results: list[dict],
    baselines: dict[str, dict],
    baseline_key: str = "eager_ns",
) -> float:
    """Geometric mean speedup across all correct results."""
    log_speedups = []
    for r in results:
        if not r.get("correct", r.get("success", False)):
            continue
        pid = r["problem_id"]
        bl = baselines.get(pid, {})
        bl_ns = bl.get(baseline_key, float("inf"))
        best_ns = r.get("best_kernel_ns", r.get("best_ns", float("inf")))
        if best_ns > 0 and bl_ns > 0 and best_ns != float("inf"):
            log_speedups.append(math.log(bl_ns / best_ns))
    if not log_speedups:
        return 0.0
    return math.exp(sum(log_speedups) / len(log_speedups))


def analyze_strategies(results: list[dict]) -> dict[str, dict]:
    """Analyze optimization strategies across all results."""
    strategy_stats = defaultdict(lambda: {"count": 0, "speedups": [], "tools": Counter()})

    for r in results:
        tool = r.get("tool", "unknown")
        for it in r.get("iterations", []):
            direction = it.get("direction", "unknown")
            speedup = it.get("speedup_vs_naive", it.get("speedup", 0))
            strategy_stats[direction]["count"] += 1
            if speedup and speedup > 0:
                strategy_stats[direction]["speedups"].append(speedup)
            strategy_stats[direction]["tools"][tool] += 1

    for v in strategy_stats.values():
        speeds = v["speedups"]
        v["avg_speedup"] = sum(speeds) / len(speeds) if speeds else 0
        v["tools"] = dict(v["tools"])

    return dict(strategy_stats)


def generate_summary_table(
    results: list[dict],
    baselines: dict[str, dict],
    tools: list[str],
    tracks: list[str],
    levels: list[int],
) -> list[dict]:
    """Generate the main summary table rows."""
    rows = []
    for tool in tools:
        for track in tracks:
            for level in levels:
                subset = [
                    r for r in results
                    if r.get("tool") == tool
                    and r.get("track") == track
                    and _level_from_pid(r["problem_id"]) == level
                ]
                if not subset:
                    continue
                rows.append({
                    "tool": tool,
                    "track": track,
                    "level": level,
                    "n_problems": len(subset),
                    "fast_0": round(compute_fast_p(subset, baselines, "eager_ns", 0), 3),
                    "fast_1_eager": round(compute_fast_p(subset, baselines, "eager_ns", 1.0), 3),
                    "fast_1_compile": round(compute_fast_p(subset, baselines, "compile_ns", 1.0), 3),
                    "geomean_eager": round(compute_geomean_speedup(subset, baselines, "eager_ns"), 3),
                    "geomean_compile": round(compute_geomean_speedup(subset, baselines, "compile_ns"), 3),
                })
    return rows


def write_latex_table(rows: list[dict], output_path: str, caption: str = "") -> None:
    """Write a LaTeX table from summary rows."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Tool & Level & fast\_0 & fast\_1 (eager) & fast\_1 (compile) & Geomean (eager) & Geomean (compile) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['tool']} & L{r['level']} & "
            f"{r['fast_0']:.1%} & {r['fast_1_eager']:.1%} & {r['fast_1_compile']:.1%} & "
            f"{r['geomean_eager']:.2f}x & {r['geomean_compile']:.2f}x \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines))


def write_csv_summary(rows: list[dict], output_path: str) -> None:
    """Write summary as CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_strategy_csv(strategy_stats: dict, output_path: str) -> None:
    """Write strategy analysis as CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, stats in sorted(strategy_stats.items(), key=lambda x: -x[1]["count"]):
        rows.append({
            "strategy": name,
            "total_count": stats["count"],
            "avg_speedup": round(stats["avg_speedup"], 3),
            "kernelopt_count": stats["tools"].get("kernelopt", 0),
            "kernelagent_count": stats["tools"].get("kernelagent", 0),
        })
    if not rows:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_per_problem_narrative(results: list[dict], output_path: str) -> None:
    """Write 1-line optimization narrative per problem per tool (for appendix)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sorted(results, key=lambda x: (x.get("tool", ""), x.get("problem_id", ""))):
        iterations = r.get("iterations", [])
        if iterations:
            strategies = "; ".join(it.get("direction", "?") for it in iterations[:3])
            best_summary = iterations[0].get("summary", "") if iterations else ""
        else:
            strategies = "none"
            best_summary = ""
        rows.append({
            "problem_id": r.get("problem_id", ""),
            "tool": r.get("tool", ""),
            "track": r.get("track", ""),
            "success": r.get("success", False),
            "speedup": r.get("speedup_vs_compile", r.get("speedup", "")),
            "strategies_tried": strategies,
            "best_summary": best_summary[:120],
        })
    if not rows:
        return
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze KernelBench benchmark results")
    parser.add_argument("--results-dir", type=str, default="benchmarks/kernelbench/results")
    parser.add_argument("--baselines", type=str, default="benchmarks/kernelbench/results/baselines.json")
    parser.add_argument("--output-dir", type=str, default="benchmarks/kernelbench/analysis")
    args = parser.parse_args()

    print("Loading results...")
    results = load_all_results(args.results_dir)
    print(f"  Loaded {len(results)} result files")

    if not Path(args.baselines).exists():
        print(f"  WARNING: baselines file not found at {args.baselines}")
        baselines = {}
    else:
        baselines = load_baselines(args.baselines)
        print(f"  Loaded {len(baselines)} baselines")

    tools = sorted(set(r.get("tool", "") for r in results))
    tracks = sorted(set(r.get("track", "") for r in results))
    levels = sorted(set(_level_from_pid(r["problem_id"]) for r in results))

    print(f"  Tools: {tools}, Tracks: {tracks}, Levels: {levels}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for track in tracks:
        track_results = [r for r in results if r.get("track") == track]
        rows = generate_summary_table(track_results, baselines, tools, [track], levels)

        write_csv_summary(rows, str(out_dir / f"summary_track_{track}.csv"))
        write_latex_table(rows, str(out_dir / f"table_track_{track}.tex"),
                          caption=f"KernelBench Track {track.upper()} Results")
        print(f"\n  Track {track.upper()} Summary:")
        for r in rows:
            print(f"    {r['tool']:15} L{r['level']} | fast_0={r['fast_0']:.1%} "
                  f"fast_1(eager)={r['fast_1_eager']:.1%} "
                  f"fast_1(compile)={r['fast_1_compile']:.1%} "
                  f"geomean(compile)={r['geomean_compile']:.2f}x")

    print("\n  Strategy Analysis:")
    strategy_stats = analyze_strategies(results)
    write_strategy_csv(strategy_stats, str(out_dir / "strategies.csv"))
    for name, stats in sorted(strategy_stats.items(), key=lambda x: -x[1]["count"])[:10]:
        print(f"    {name:40} count={stats['count']:3d} avg_speedup={stats['avg_speedup']:.2f}x")

    write_per_problem_narrative(results, str(out_dir / "per_problem_narrative.csv"))

    print(f"\n  Analysis written to {out_dir}/")
    print(f"    summary_track_*.csv, table_track_*.tex, strategies.csv, per_problem_narrative.csv")


if __name__ == "__main__":
    main()
