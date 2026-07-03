#!/usr/bin/env python3
"""Download and set up all kernel benchmark datasets."""

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from datasets import load_dataset

BENCHMARKS_DIR = Path(__file__).parent
REPOS_DIR = Path("/mnt/podman_storage/ahpoddar/home/kernelopt")


def download_kernelbench():
    """Download KernelBench dataset (270 problems)."""
    print("=" * 60)
    print("Downloading KernelBench...")
    dataset = load_dataset("ScalingIntelligence/KernelBench")

    output_dir = BENCHMARKS_DIR / "kernelbench"
    output_dir.mkdir(exist_ok=True)

    total = 0
    for level in ["level_1", "level_2", "level_3", "level_4"]:
        if level not in dataset:
            continue
        level_dir = output_dir / level
        level_dir.mkdir(exist_ok=True)

        for item in dataset[level]:
            pid = item["problem_id"]
            (level_dir / f"kernel_{pid:03d}.py").write_text(item["code"])
            (level_dir / f"kernel_{pid:03d}.json").write_text(
                json.dumps(
                    {
                        "name": item["name"],
                        "problem_id": pid,
                        "level": item["level"],
                    },
                    indent=2,
                )
            )
            total += 1

    print(f"  KernelBench: {total} problems saved to {output_dir}")


def download_sol_execbench():
    """Download SOL-ExecBench dataset (235 problems)."""
    print("=" * 60)
    print("Downloading SOL-ExecBench...")

    output_dir = BENCHMARKS_DIR / "sol_execbench"
    output_dir.mkdir(exist_ok=True)

    subsets = ["FlashInfer-Bench", "L1", "L2", "Quant"]
    total = 0

    for subset in subsets:
        ds = load_dataset("nvidia/SOL-ExecBench", subset, split="train")
        subset_dir = output_dir / subset / "train"
        subset_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(ds):
            out_path = subset_dir / f"kernel_{idx:03d}.json"
            out_path.write_text(json.dumps(dict(item), indent=2))
            total += 1

        print(f"  {subset}/train: {len(ds)} problems saved")

    print(f"  SOL-ExecBench: {total} problems saved to {output_dir}")


def setup_backendbench():
    """Set up BackendBench (271 PyTorch OpInfo ops)."""
    print("=" * 60)
    print("Setting up BackendBench...")

    repo_dir = REPOS_DIR / "backendbench"
    if not repo_dir.exists():
        print("  Cloning BackendBench...")
        subprocess.run(
            ["git", "clone", "https://github.com/meta-pytorch/BackendBench.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"  Repo already exists at {repo_dir}")

    output_dir = BENCHMARKS_DIR / "backendbench"
    output_dir.mkdir(exist_ok=True)

    config = {
        "name": "BackendBench",
        "source": "https://github.com/meta-pytorch/BackendBench",
        "repo_path": str(repo_dir),
        "description": "271 PyTorch OpInfo operators for testing LLM-generated backend kernels",
        "problem_count": 271,
        "metric": "Speedup + Correctness (OpInfo/FACTO)",
        "precision": "FP32/FP16/BF16",
        "suites": {
            "opinfo": "Correctness testing via PyTorch OpInfo (271 bfloat16 ops)",
            "facto": "FACTO test suite for advanced correctness",
            "torchbench": "Performance benchmarks from real model traces",
            "smoke": "Quick validation suite",
        },
        "run_commands": {
            "smoke": "python BackendBench/scripts/main.py --suite smoke --backend aten",
            "opinfo": "python BackendBench/scripts/main.py --suite opinfo --backend directory",
            "torchbench": "python BackendBench/scripts/main.py --suite torchbench --backend directory",
        },
    }

    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  BackendBench: config saved to {output_dir}/config.json")


def download_computeeval():
    """Download and extract ComputeEval (566 problems, 2026.1 release)."""
    print("=" * 60)
    print("Setting up ComputeEval...")

    repo_dir = REPOS_DIR / "computeeval"
    if not repo_dir.exists():
        print("  Cloning ComputeEval...")
        subprocess.run(
            ["git", "clone", "https://github.com/NVIDIA/compute-eval.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"  Repo already exists at {repo_dir}")

    tarball = repo_dir / "data" / "releases" / "2026-1-problems.tar.gz"
    if not tarball.exists():
        print(f"  ERROR: Release tarball not found at {tarball}")
        return

    extract_dir = repo_dir / "data" / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    problems_file = extract_dir / "problems.jsonl"
    if not problems_file.exists():
        print("  ERROR: problems.jsonl not found after extraction")
        return

    output_dir = BENCHMARKS_DIR / "computeeval"
    output_dir.mkdir(exist_ok=True)

    with open(problems_file) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    total = 0
    for p in problems:
        group = p.get("group", "unknown")
        group_dir = output_dir / group
        group_dir.mkdir(exist_ok=True)

        task_id = p.get("task_id", f"task_{total:04d}")
        safe_id = task_id.replace("/", "_")

        (group_dir / f"{safe_id}.json").write_text(json.dumps(p, indent=2))
        total += 1

    print(f"  ComputeEval: {total} problems saved to {output_dir}")


def download_cudabench():
    """Download and extract CUDABench (500 tasks)."""
    print("=" * 60)
    print("Setting up CUDABench...")

    repo_dir = REPOS_DIR / "cudabench"
    if not repo_dir.exists():
        print("  Cloning CUDABench...")
        subprocess.run(
            ["git", "clone", "https://github.com/CUDA-Bench/CUDABench.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"  Repo already exists at {repo_dir}")

    dataset_file = repo_dir / "Datasets" / "CUDABench-Set.jsonl"
    if not dataset_file.exists():
        print(f"  ERROR: Dataset not found at {dataset_file}")
        return

    output_dir = BENCHMARKS_DIR / "cudabench" / "tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_file) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    for task in tasks:
        task_id = task.get("id", 0)
        task_name = task.get("task_name", f"task_{task_id}")
        safe_name = task_name.replace(" ", "_").replace("/", "_")

        (output_dir / f"task_{task_id:04d}_{safe_name}.json").write_text(
            json.dumps(task, indent=2)
        )

    print(f"  CUDABench: {len(tasks)} tasks saved to {output_dir}")


def verify_all():
    """Verify all benchmarks are set up correctly."""
    print("=" * 60)
    print("Verification:")

    checks = [
        ("KernelBench", "kernelbench", ["level_1", "level_2", "level_3", "level_4"]),
        ("SOL-ExecBench", "sol_execbench", ["FlashInfer-Bench/train", "L1/train", "L2/train", "Quant/train"]),
        ("BackendBench", "backendbench", []),
        ("ComputeEval", "computeeval", ["cuda-kernels", "cudnn", "mathlibs", "cublas", "cccl", "cutile", "cuda-runtime"]),
        ("CUDABench", "cudabench", ["tasks"]),
    ]

    for name, dirname, subdirs in checks:
        base = BENCHMARKS_DIR / dirname
        if not base.exists():
            print(f"  {name}: MISSING")
            continue

        if subdirs:
            total = 0
            for sub in subdirs:
                sub_dir = base / sub
                if sub_dir.exists():
                    count = len([f for f in os.listdir(sub_dir) if f.endswith((".json", ".py"))])
                    total += count
            print(f"  {name}: {total} files across {len(subdirs)} subdirs")
        else:
            config = base / "config.json"
            if config.exists():
                with open(config) as f:
                    cfg = json.load(f)
                print(f"  {name}: {cfg.get('problem_count', '?')} ops (runtime-discovery, repo at {cfg.get('repo_path', '?')})")
            else:
                print(f"  {name}: directory exists but no config")


if __name__ == "__main__":
    download_kernelbench()
    download_sol_execbench()
    setup_backendbench()
    download_computeeval()
    download_cudabench()
    verify_all()
    print("\n" + "=" * 60)
    print("All datasets downloaded/set up successfully!")
