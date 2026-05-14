#!/usr/bin/env python3
"""Run PyTorch-to-Triton synthesis on KernelBench files (no optimization loop).

Tests only the torch.compile → extract → wrap pipeline.
Reports per-kernel: success/fail, kernel count, graph breaks, errors.

Usage:
    python run_synthesis_only.py [--level 1] [--start 1] [--end 5] [--timeout 180]
    python run_synthesis_only.py --all
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernelopt"))

from kernelopt.synthesis.compile_extract import (
    SynthesisError,
    extract_triton_from_pytorch,
)
from kernelopt.synthesis.wrap_kernel import wrap_inductor_output

BENCH_ROOT = Path(__file__).resolve().parent / "kernelbench"


def find_kernel_files(level: int | None, start: int | None, end: int | None) -> list[Path]:
    levels = [level] if level else [1, 2, 3, 4]
    files = []
    for lv in levels:
        level_dir = BENCH_ROOT / f"level_{lv}"
        if not level_dir.exists():
            continue
        py_files = sorted(level_dir.glob("kernel_*.py"))
        for f in py_files:
            num = int(f.stem.split("_")[1])
            if start and num < start:
                continue
            if end and num > end:
                continue
            files.append(f)
    return files


def run_one(kernel_path: Path, timeout: int) -> dict:
    record = {
        "file": str(kernel_path),
        "level": kernel_path.parent.name,
        "kernel": kernel_path.stem,
        "status": "UNKNOWN",
        "triton_kernel_count": 0,
        "template_kernel_count": 0,
        "graph_break_count": 0,
        "graph_breaks": "",
        "wrap_valid": False,
        "error": "",
        "elapsed_s": 0.0,
    }

    t0 = time.time()
    try:
        pytorch_source = kernel_path.read_text()
        results = extract_triton_from_pytorch(str(kernel_path), timeout=timeout)

        if not results:
            record["status"] = "NO_OUTPUT"
            record["error"] = "torch.compile returned no sources"
            return record

        primary = results[0]
        triton_blocks = primary.get("triton_kernel_blocks", [])
        triton_kernels = primary.get("triton_kernels", [])
        record["triton_kernel_count"] = len(triton_kernels)
        record["template_kernel_count"] = sum(1 for b in triton_blocks if b.get("is_template"))
        record["graph_break_count"] = primary.get("graph_break_count", 0)
        record["graph_breaks"] = "; ".join(primary.get("graph_breaks", []))

        if not triton_kernels:
            record["status"] = "NO_TRITON"
            record["error"] = "torch.compile produced Inductor output but no @triton.jit kernels (likely cuBLAS-only)"
            return record

        # Try wrapping the first kernel
        best_kernel_src = triton_kernels[0]
        call_source = primary.get("call_source", "")
        benchmark_source = primary.get("benchmark_source", "")
        inductor_source = primary.get("inductor_source", "")

        wrapped = wrap_inductor_output(
            triton_kernel_source=best_kernel_src,
            call_source=call_source,
            benchmark_source=benchmark_source,
            original_pytorch_source=pytorch_source,
            inductor_source=inductor_source,
        )

        # Verify the wrapped output is syntactically valid Python
        try:
            ast.parse(wrapped)
            record["wrap_valid"] = True
        except SyntaxError as e:
            record["wrap_valid"] = False
            record["error"] = f"Wrapped output has syntax error: {e}"
            record["status"] = "WRAP_ERROR"
            return record

        record["status"] = "OK"

    except SynthesisError as e:
        record["status"] = "SYNTHESIS_ERROR"
        record["error"] = str(e)[:300]
    except Exception as e:
        record["status"] = "ERROR"
        record["error"] = f"{type(e).__name__}: {str(e)[:250]}"
    finally:
        record["elapsed_s"] = round(time.time() - t0, 1)

    return record


def main():
    parser = argparse.ArgumentParser(description="Run synthesis-only on KernelBench")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], help="Level to test (default: all)")
    parser.add_argument("--start", type=int, help="Start kernel number (inclusive)")
    parser.add_argument("--end", type=int, help="End kernel number (inclusive)")
    parser.add_argument("--all", action="store_true", help="Run all levels")
    parser.add_argument("--timeout", type=int, default=180, help="Per-kernel timeout (seconds)")
    parser.add_argument("--csv", type=str, help="Output CSV file path")
    args = parser.parse_args()

    if not args.level and not args.all:
        parser.error("Specify --level N or --all")

    files = find_kernel_files(args.level, args.start, args.end)
    print(f"Found {len(files)} kernel files to test\n")

    results = []
    counts = {"OK": 0, "NO_TRITON": 0, "SYNTHESIS_ERROR": 0, "WRAP_ERROR": 0, "ERROR": 0, "NO_OUTPUT": 0}

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.parent.name}/{f.name} ... ", end="", flush=True)
        record = run_one(f, args.timeout)
        results.append(record)

        status = record["status"]
        counts[status] = counts.get(status, 0) + 1

        if status == "OK":
            detail = f"{record['triton_kernel_count']} kernels"
            if record["template_kernel_count"]:
                detail += f" ({record['template_kernel_count']} template)"
            if record["graph_break_count"]:
                detail += f", {record['graph_break_count']} graph breaks"
            print(f"OK ({detail}) [{record['elapsed_s']}s]")
        elif status == "NO_TRITON":
            print(f"NO_TRITON (cuBLAS-only) [{record['elapsed_s']}s]")
        else:
            print(f"{status} [{record['elapsed_s']}s]")
            if record["error"]:
                print(f"    {record['error'][:120]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SYNTHESIS-ONLY RESULTS: {len(files)} kernels")
    print(f"{'='*60}")
    for status, count in sorted(counts.items()):
        if count > 0:
            pct = 100 * count / len(files)
            print(f"  {status:20s}: {count:4d} ({pct:5.1f}%)")
    print(f"{'='*60}")

    # CSV output
    if args.csv:
        csv_path = Path(args.csv)
        fieldnames = ["level", "kernel", "status", "triton_kernel_count",
                      "template_kernel_count", "graph_break_count", "wrap_valid",
                      "elapsed_s", "error"]
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV written to {csv_path}")

    # Return non-zero if any errors
    if counts.get("ERROR", 0) + counts.get("SYNTHESIS_ERROR", 0) + counts.get("WRAP_ERROR", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
