from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PlanCandidate:
    """One optimization plan and its execution/profiling results."""

    plan_id: str = ""
    plan_letter: str = ""            # A, B, C, D ...
    direction: str = ""
    optimization_plan: dict = field(default_factory=dict)
    optimized_kernel: str = ""
    execution_result: dict = field(default_factory=dict)
    correctness_passed: bool = False
    correctness_message: str = ""
    profiling_context: dict = field(default_factory=dict)
    csv_path: str = ""
    ncu_rep_path: str = ""
    latency_ns: float = float("inf")
    speedup: float = 1.0


@dataclass
class IterationResult:
    """Summary of one iteration across all plan candidates."""

    iteration: int = 0
    candidates: list[PlanCandidate] = field(default_factory=list)
    best_candidate: PlanCandidate | None = None
    improved: bool = False


@dataclass
class PipelineConfig:
    """User-controllable pipeline parameters."""

    T: int = 5          # max iterations
    N: int = 4          # plans per iteration
    K: int = 1          # executor attempts per plan
    TopK: int = 4       # max experience items added to memory per iteration
    ExpN: int = 8       # memory queue capacity
    t_pos: float = 1.04  # min speedup to store positive item
    t_neg: float = 1.15  # min slowdown factor to store negative item

    # NCU options
    kernel_name_regex: str | None = None
    launch_skip: int = 0
    launch_count: int | None = None
    ncu_extra_args: list[str] = field(default_factory=list)
    kernel_args: list[str] = field(default_factory=list)  # args passed to kernel program

    # LLM options
    model: str = "Qwen/Qwen3-32B"
    api_key: str | None = None
    base_url: str = "http://localhost:8000/v1"

    # Paths
    output_dir: str = "pipeline_output"
    kernel_path: str = ""
    baseline_csv: str | None = None

    # Runtime
    verbose: bool = False


@dataclass
class PipelineState:
    """Full mutable state of the optimization pipeline."""

    config: PipelineConfig = field(default_factory=PipelineConfig)

    # Kernel
    kernel_path: str = ""
    kernel_source: str = ""
    framework: str = "auto"
    kernel_id: int = 0
    kernel_basename: str = ""       # e.g. "softmax" for naming files

    # Current best
    best_kernel: str = ""
    best_latency_ns: float = float("inf")
    best_ncu_rep: str = ""

    # Baseline profiling
    baseline_profiling_context: dict = field(default_factory=dict)
    baseline_csv: str = ""
    baseline_ncu_rep: str = ""

    # Current iteration state
    current_profiling_context: dict = field(default_factory=dict)
    iteration: int = 0
    no_improvement_streak: int = 0

    # Memory
    optimization_memory: list[dict] = field(default_factory=list)

    # History
    iteration_results: list[IterationResult] = field(default_factory=list)
