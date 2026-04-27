"""
Pipeline state management.

Defines PipelineConfig, PipelineState, GraphState, and related types
for the kernel optimization pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict


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
    compiled_binary_path: str = ""        # set after compile in execute_plans; reused in profile_candidates
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
    kernel_name: str | None = None
    launch_skip: int = 0
    launch_count: int | None = None
    replay_mode: str = "kernel"   # "kernel" (default) or "application" (less GPU memory, slower)
    ncu_extra_args: list[str] = field(default_factory=list)
    kernel_args: list[str] = field(default_factory=list)  # args passed to kernel program

    # LLM options (defaults, used when per-agent options are not set)
    model: str = "claude-sonnet-4-6"
    api_key: str | None = None
    project_id: str = ""
    region: str = "us-east5"

    # Per-agent model overrides
    planner_model: str | None = None
    summarizer_model: str | None = None
    executor_model: str | None = None
    profiler_model: str | None = None

    # Paths
    output_dir: str = "pipeline_output"
    kernel_path: str = ""
    baseline_csv: str | None = None

    # Runtime
    verbose: bool = False
    debug: bool = False
    agentic_correctness: bool = True  # Use Claude to generate CPU reference tests
    # Same-terminal HUD (--hud); full stream + NCU text → kernelopt_terminal.log; not in checkpoint
    hud: bool = False

    # Profiler agent
    profile_hints: str = ""


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
    # NCU config determined at baseline; reused for all candidate profiling
    ncu_config: Optional[dict] = None

    # Current iteration state
    current_profiling_context: dict = field(default_factory=dict)
    iteration: int = 0
    no_improvement_streak: int = 0

    # Memory
    optimization_memory: list[dict] = field(default_factory=list)

    # History
    iteration_results: list[IterationResult] = field(default_factory=list)

    # Resume: directory from --checkpoint (prior run output folder)
    resume_run_dir: str | None = None

    # Pre-run metrics from validate_gpu_for_profiling (for HUD; same source as CLI System Status)
    pre_run_system_snapshot: dict = field(default_factory=dict)


class GraphState(TypedDict, total=False):
    """LangGraph-compatible state for the optimization pipeline."""

    # Immutable config
    config: PipelineConfig
    kernel_path: str
    kernel_source: str
    kernel_basename: str
    kernel_id: int
    framework: str
    output_dir: str
    ext: str

    # Baseline
    baseline_profiling_context: dict
    baseline_csv: str
    baseline_ncu_rep: str
    baseline_latency_ns: float
    # NCU config from baseline profiling — reused for all candidates
    ncu_config: Optional[dict]

    # Current best
    best_kernel: str
    best_latency_ns: float
    best_ncu_rep: str
    current_profiling_context: dict

    # Per-iteration
    iteration: int
    no_improvement_streak: int
    plans: list[tuple[str, dict]]
    candidates: list[PlanCandidate]
    correctness_failure_items: list   # implementation_flaw items from correctness failures

    # Memory
    optimization_memory: list[dict]

    # History
    iteration_results: list[IterationResult]

    # Internal: LangGraph entry routing (fresh | after_baseline | after_iteration)
    _entry: str
