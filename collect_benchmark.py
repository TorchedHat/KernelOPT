#!/usr/bin/env python3
"""Collect and analyze benchmark results from all GPUs.

Usage:
    python collect_benchmark.py --phase 1
    python collect_benchmark.py --phase 1 --merge-banks
    python collect_benchmark.py --phase 2 --report
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


RESULTS_BASE = Path("eval_benchmark_results")


def collect_results(phase: int) -> list[dict]:
    """Merge all per-GPU results into a single list."""
    phase_dir = RESULTS_BASE / f"phase{phase}"
    all_results = []
    seen_ids = set()

    for gpu_dir in sorted(phase_dir.glob("gpu*")):
        results_file = gpu_dir / "results.json"
        if not results_file.exists():
            print(f"  WARNING: {results_file} not found")
            continue
        data = json.loads(results_file.read_text())
        for r in data:
            pid = r.get("problem_id", "")
            if pid not in seen_ids:
                all_results.append(r)
                seen_ids.add(pid)
        print(f"  {gpu_dir.name}: {len(data)} results")

    combined = phase_dir / "combined_results.json"
    combined.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nCombined: {len(all_results)} results -> {combined}")
    return all_results


def merge_memory_banks(phase: int) -> None:
    """Merge per-GPU memory banks into one."""
    bank_dir = RESULTS_BASE / f"phase{phase}_memory_banks"
    if not bank_dir.exists():
        print(f"No memory banks found at {bank_dir}")
        return

    all_entries = []
    seen_ids = set()

    for bank_file in sorted(bank_dir.glob("memory_bank_gpu*.json")):
        try:
            entries = json.loads(bank_file.read_text())
            for e in entries:
                eid = e.get("id", "")
                if eid not in seen_ids:
                    all_entries.append(e)
                    seen_ids.add(eid)
            print(f"  {bank_file.name}: {len(entries)} entries")
        except Exception as e:
            print(f"  {bank_file.name}: ERROR {e}")

    merged = bank_dir / "memory_bank_merged.json"
    merged.write_text(json.dumps(all_entries, indent=2))
    print(f"\nMerged: {len(all_entries)} unique entries -> {merged}")


def generate_report(results: list[dict], phase: int) -> str:
    """Generate the full benchmark analysis report."""
    lines = [
        f"# kernelopt Phase {phase} Benchmark Results\n",
        f"**Total problems:** {len(results)}\n",
    ]

    # Status breakdown
    status_counts = {}
    for r in results:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    lines.append("## Status Breakdown\n")
    lines.append("| Status | Count | % |")
    lines.append("|--------|-------|---|")
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {s} | {c} | {100*c/len(results):.1f}% |")
    lines.append("")

    # Synthesis stats
    synth_pass = sum(1 for r in results if r.get("synthesis_status") == "pass")
    synth_fail = sum(1 for r in results if r.get("synthesis_status") in ("fail", "skip"))
    lines.append(f"## Synthesis\n")
    lines.append(f"- Pass: {synth_pass}/{len(results)} ({100*synth_pass/max(len(results),1):.1f}%)")
    lines.append(f"- Fail/Skip: {synth_fail}/{len(results)}")
    lines.append("")

    # Per-dataset breakdown
    datasets = sorted(set(r.get("dataset", "unknown") for r in results))
    lines.append("## Per-Dataset Results\n")
    lines.append("| Dataset | Total | Pass | Median Speedup | >1.0x | >1.2x | >1.5x | Median Eager (µs) | Median Compile (µs) | Median Best (µs) |")
    lines.append("|---------|-------|------|----------------|-------|-------|-------|-------------------|---------------------|------------------|")

    for ds in datasets:
        ds_results = [r for r in results if r.get("dataset") == ds]
        ds_pass = [r for r in ds_results if r.get("status") == "pass"]
        speedups = [r["speedup"] for r in ds_pass if r.get("speedup", 0) > 0]
        eagers = [r["eager_ns"]/1e3 for r in ds_results if r.get("eager_ns", -1) > 0]
        compiles = [r["compile_ns"]/1e3 for r in ds_results if r.get("compile_ns", -1) > 0]
        bests = [r["best_latency_ns"]/1e3 for r in ds_pass if r.get("best_latency_ns", -1) > 0]

        med_sp = f"{statistics.median(speedups):.2f}x" if speedups else "—"
        gt1 = sum(1 for s in speedups if s > 1.0) if speedups else 0
        gt12 = sum(1 for s in speedups if s > 1.2) if speedups else 0
        gt15 = sum(1 for s in speedups if s > 1.5) if speedups else 0
        med_eager = f"{statistics.median(eagers):.1f}" if eagers else "—"
        med_comp = f"{statistics.median(compiles):.1f}" if compiles else "—"
        med_best = f"{statistics.median(bests):.1f}" if bests else "—"

        lines.append(f"| {ds} | {len(ds_results)} | {len(ds_pass)} | {med_sp} | {gt1} | {gt12} | {gt15} | {med_eager} | {med_comp} | {med_best} |")

    lines.append("")

    # Overall speedup distribution
    all_speedups = [r["speedup"] for r in results if r.get("speedup", 0) > 0]
    if all_speedups:
        all_speedups.sort()
        lines.append("## Overall Speedup Distribution\n")
        lines.append(f"- **Median:** {statistics.median(all_speedups):.2f}x")
        lines.append(f"- **Mean:** {statistics.mean(all_speedups):.2f}x")
        lines.append(f"- **P10:** {all_speedups[len(all_speedups)//10]:.2f}x")
        lines.append(f"- **P90:** {all_speedups[9*len(all_speedups)//10]:.2f}x")
        lines.append(f"- **Min:** {min(all_speedups):.2f}x")
        lines.append(f"- **Max:** {max(all_speedups):.2f}x")
        lines.append(f"- **>1.0x:** {sum(1 for s in all_speedups if s > 1.0)}/{len(all_speedups)}")
        lines.append(f"- **>1.2x:** {sum(1 for s in all_speedups if s > 1.2)}/{len(all_speedups)}")
        lines.append(f"- **>1.5x:** {sum(1 for s in all_speedups if s > 1.5)}/{len(all_speedups)}")
        lines.append("")

    # Comparative analysis table (the detailed one user asked for)
    pass_results = [r for r in results if r.get("status") == "pass" and r.get("speedup", 0) > 0]
    if pass_results:
        pass_results.sort(key=lambda r: r.get("speedup", 0), reverse=True)
        lines.append("## Comparative Analysis (Top Optimizations)\n")
        lines.append("| Problem | Dataset | Eager (µs) | Compile (µs) | Triton Base (µs) | kernelopt (µs) | vs Eager | vs Compile | vs Triton | Strategy |")
        lines.append("|---------|---------|-----------|-------------|-----------------|---------------|----------|-----------|-----------|----------|")
        for r in pass_results[:50]:
            pid = r["problem_id"]
            ds = r.get("dataset", "")
            eager = r.get("eager_ns", -1) / 1e3
            comp = r.get("compile_ns", -1) / 1e3
            base = r.get("baseline_latency_ns", -1) / 1e3
            best = r.get("best_latency_ns", -1) / 1e3
            vs_e = f"{r.get('speedup_vs_eager', 0):.2f}x" if r.get("speedup_vs_eager", 0) > 0 else "—"
            vs_c = f"{r.get('speedup_vs_compile', 0):.2f}x" if r.get("speedup_vs_compile", 0) > 0 else "—"
            vs_t = f"{r.get('speedup', 0):.2f}x"
            strat = r.get("best_direction", r.get("strategy_type", ""))[:40]
            lines.append(f"| {pid} | {ds} | {eager:.1f} | {comp:.1f} | {base:.1f} | {best:.1f} | {vs_e} | {vs_c} | {vs_t} | {strat} |")
        lines.append("")

    # Strategy analysis
    strategies = {}
    for r in results:
        s = r.get("strategy_type", "unknown") or "unknown"
        if s not in strategies:
            strategies[s] = {"count": 0, "speedups": []}
        strategies[s]["count"] += 1
        if r.get("speedup", 0) > 0:
            strategies[s]["speedups"].append(r["speedup"])

    if strategies:
        lines.append("## Strategy Effectiveness\n")
        lines.append("| Strategy | Count | Median Speedup | >1.0x |")
        lines.append("|----------|-------|----------------|-------|")
        for s, info in sorted(strategies.items(), key=lambda x: -x[1]["count"]):
            med = f"{statistics.median(info['speedups']):.2f}x" if info["speedups"] else "—"
            gt1 = sum(1 for sp in info["speedups"] if sp > 1.0)
            lines.append(f"| {s} | {info['count']} | {med} | {gt1}/{len(info['speedups'])} |")
        lines.append("")

    # Timing
    wall_times = [r.get("wall_time_s", 0) for r in results if r.get("wall_time_s", 0) > 0]
    if wall_times:
        lines.append("## Timing\n")
        lines.append(f"- Total wall time: {sum(wall_times)/3600:.1f}h")
        lines.append(f"- Avg per problem: {statistics.mean(wall_times)/60:.1f}min")
        lines.append(f"- Median per problem: {statistics.median(wall_times)/60:.1f}min")
        lines.append(f"- Max: {max(wall_times)/60:.1f}min")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--merge-banks", action="store_true",
                        help="Merge per-GPU memory banks")
    parser.add_argument("--report", action="store_true",
                        help="Generate benchmark_results.md")
    parser.add_argument("--all", action="store_true",
                        help="Collect + merge banks + report")
    args = parser.parse_args()

    if args.all:
        args.merge_banks = True
        args.report = True

    print(f"Collecting Phase {args.phase} results...")
    results = collect_results(args.phase)

    if args.merge_banks:
        print(f"\nMerging memory banks...")
        merge_memory_banks(args.phase)

    if args.report:
        print(f"\nGenerating report...")
        report = generate_report(results, args.phase)
        report_path = Path("docs") / f"benchmark_results_phase{args.phase}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"Report: {report_path}")
        print(report[:2000])


if __name__ == "__main__":
    main()
