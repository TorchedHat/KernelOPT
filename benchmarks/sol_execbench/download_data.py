"""Download safetensors trace data for SOL-ExecBench problems.

Parses the 26 JSON specs, identifies which require external safetensors
files, and downloads them from the HuggingFace nvidia/SOL-ExecBench repo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.sol_execbench.harness import (
    list_problems,
    load_problem,
    needs_safetensors,
    get_safetensors_paths,
    SAFETENSORS_ROOT,
)


def collect_all_paths() -> dict[str, list[str]]:
    """Collect all unique safetensors paths grouped by problem."""
    result = {}
    for meta in list_problems():
        prob = load_problem(meta["path"])
        if needs_safetensors(prob):
            paths = get_safetensors_paths(prob)
            if paths:
                result[prob.name] = paths
    return result


def download_safetensors(output_dir: Path | None = None) -> dict:
    """Download all required safetensors from HuggingFace.

    Returns summary dict with counts.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        return {"error": "huggingface_hub not installed"}

    output_dir = output_dir or SAFETENSORS_ROOT
    all_paths = collect_all_paths()

    unique_paths = set()
    for paths in all_paths.values():
        unique_paths.update(paths)

    print(f"Problems needing safetensors: {len(all_paths)}")
    print(f"Unique safetensors files: {len(unique_paths)}")

    downloaded = 0
    skipped = 0
    failed = 0

    for st_path in sorted(unique_paths):
        local_path = output_dir / st_path
        if local_path.exists():
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {st_path}...", end=" ", flush=True)
        try:
            hf_hub_download(
                repo_id="nvidia/SOL-ExecBench",
                filename=st_path,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            downloaded += 1
            print("OK")
        except Exception as e:
            failed += 1
            print(f"FAIL: {e}")

    summary = {
        "problems_needing_data": len(all_paths),
        "unique_files": len(unique_paths),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }

    readiness = check_readiness(output_dir)
    summary["problems_ready"] = readiness["ready"]
    summary["problems_blocked"] = readiness["blocked"]

    return summary


def check_readiness(data_dir: Path | None = None) -> dict:
    """Check which problems have all required safetensors present."""
    data_dir = data_dir or SAFETENSORS_ROOT
    ready = []
    blocked = []

    for meta in list_problems():
        prob = load_problem(meta["path"])
        if not needs_safetensors(prob):
            ready.append(prob.name)
            continue

        paths = get_safetensors_paths(prob)
        missing = [p for p in paths if not (data_dir / p).exists()]
        if missing:
            blocked.append({"name": prob.name, "missing": missing})
        else:
            ready.append(prob.name)

    return {"ready": ready, "blocked": blocked}


def main():
    parser = argparse.ArgumentParser(description="Download SOL-ExecBench safetensors data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {SAFETENSORS_ROOT})")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check readiness, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.check_only:
        readiness = check_readiness(output_dir)
        print(f"Ready: {len(readiness['ready'])} problems")
        for name in readiness["ready"]:
            print(f"  + {name}")
        print(f"Blocked: {len(readiness['blocked'])} problems")
        for item in readiness["blocked"]:
            print(f"  - {item['name']} (missing {len(item['missing'])} files)")
        return

    summary = download_safetensors(output_dir)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
