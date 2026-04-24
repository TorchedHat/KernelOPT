#!/usr/bin/env python3
"""Download KernelBench and SOL-ExecBench datasets."""

from datasets import load_dataset
from pathlib import Path
import json

def download_kernelbench():
    """Download KernelBench dataset."""
    print("Downloading KernelBench...")
    dataset = load_dataset("ScalingIntelligence/KernelBench")

    # Save to kernelbench directory
    output_dir = Path(__file__).parent / "kernelbench"
    output_dir.mkdir(exist_ok=True)

    # Save each level
    for level in ["level_1", "level_2", "level_3", "level_4"]:
        if level in dataset:
            level_dir = output_dir / level
            level_dir.mkdir(exist_ok=True)

            for idx, item in enumerate(dataset[level]):
                kernel_file = level_dir / f"kernel_{item['problem_id']:03d}.py"
                kernel_file.write_text(item['code'])

                # Save metadata
                meta_file = level_dir / f"kernel_{item['problem_id']:03d}.json"
                meta_file.write_text(json.dumps({
                    "name": item['name'],
                    "problem_id": item['problem_id'],
                    "level": item['level']
                }, indent=2))

    print(f"✓ KernelBench saved to {output_dir}")

def download_sol_execbench():
    """Download SOL-ExecBench dataset."""
    print("Downloading SOL-ExecBench...")
    dataset = load_dataset("nvidia/SOL-ExecBench")

    # Save to sol_execbench directory
    output_dir = Path(__file__).parent / "sol_execbench"
    output_dir.mkdir(exist_ok=True)

    # Save dataset
    for split_name, split_data in dataset.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        for idx, item in enumerate(split_data):
            # Save kernel spec as JSON
            kernel_file = split_dir / f"kernel_{idx:03d}.json"
            kernel_file.write_text(json.dumps(dict(item), indent=2))

    print(f"✓ SOL-ExecBench saved to {output_dir}")

if __name__ == "__main__":
    download_kernelbench()
    download_sol_execbench()
    print("\n✓ All datasets downloaded successfully!")
