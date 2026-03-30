"""Run the full LangGraph pipeline with direct transformers inference (no server)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--iterations", type=int, default=4)
parser.add_argument("-N", "--plans", type=int, default=4)
parser.add_argument("-K", "--attempts", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--kernel", type=str, default=str(Path(__file__).parent / "softmaxtri.py"))
parser.add_argument("--kernel-args", type=str, default="--profile-only")
parser.add_argument("--kernel-name", type=str, default=None)
parser.add_argument("-o", "--output-dir", type=str, default="pipeline_output")
args = parser.parse_args()

from test_agents import patch_build_llm

patch_build_llm(gpu_id=args.gpu)

from kernelopt.langgraph_pipeline import run_langgraph_pipeline
from kernelopt.state import PipelineConfig, PipelineState

kernel_path = Path(args.kernel)

cfg = PipelineConfig(
    T=args.iterations,
    N=args.plans,
    K=args.attempts,
    TopK=4,
    model="Qwen/Qwen3-30B-A3B",
    kernel_name_regex=args.kernel_name,
    kernel_args=args.kernel_args.split() if args.kernel_args else [],
    output_dir=args.output_dir,
    kernel_path=str(kernel_path),
)

state = PipelineState(
    config=cfg,
    kernel_path=str(kernel_path),
    kernel_source=kernel_path.read_text(),
    best_kernel=kernel_path.read_text(),
    framework="auto",
    kernel_basename=kernel_path.stem,
)

print(f"Running pipeline: T={cfg.T}, N={cfg.N}, K={cfg.K}")
final = run_langgraph_pipeline(state)
print(f"\nDone. Best latency: {final.best_latency_ns:.0f} ns")
