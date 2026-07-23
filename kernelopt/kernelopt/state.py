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
class ActiveKernel:
    """One active optimization chain in the multi-chain forest search.

    The pipeline maintains a beam of B active kernels. Each iteration,
    plans are generated for each active kernel, candidates are profiled,
    and the top-B candidates form the next beam. This implements a
    ThoughtForest-style multi-chain search where B concurrent optimization
    paths are explored in parallel.
    """
    chain_id: str = "chain_0"        # stable ID across iterations
    kernel_source: str = ""          # current kernel source code
    latency_ns: float = float("inf") # NCU latency
    wall_ms: float = float("inf")    # wall-clock latency
    speedup: float = 1.0             # vs baseline
    profiling_context: dict = field(default_factory=dict)  # most recent NCU context
    ncu_summary: str = ""            # compact metrics for logging
    history: list[str] = field(default_factory=list)  # directions applied in order
    depth: int = 0                   # transformations from baseline
    alive: bool = True               # False if chain produced 0 correct candidates


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
    wall_ms: float = float("inf")           # wall-clock end-to-end timing
    speedup: float = 1.0
    wall_speedup: float = 1.0               # primary speedup metric (wall-clock)
    chain_id: str = ""                      # which ActiveKernel this was derived from


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
    keep_n: int = 4       # max experience items added to memory per iteration
    q_cap: int = 8        # memory queue capacity
    spe_gate: float = 1.05  # min speedup to store positive item
    slo_gate: float = 1.20  # min slowdown factor to store negative item

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
    llm_provider: str = "vertex"
    openai_base_url: str | None = None
    hf_inference_provider: str = "auto"

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

    # Search policy: "greedy" (single chain) or "beam" (multi-chain forest)
    search_policy: str = "greedy"
    beam_width: int = 1              # number of concurrent optimization chains

    # E2E correctness retries: after re-stitching an optimized sub-kernel,
    # verify full-model correctness against the original PyTorch model. If
    # it fails, feed the error back and retry optimization with a targeted
    # 1-iteration run. Catches bugs where per-kernel optimization breaks
    # full-model correctness (e.g., accumulation errors at full scale).
    e2e_retries: int = 2

    # E2E performance gate: after re-stitching, measure wall-clock timing
    # of the re-stitched model vs baseline. If the model is slower by more
    # than e2e_perf_margin (per-kernel speedup didn't translate to
    # model-level), reject and retry with layout/stride feedback.
    # Uses e2e_retries for max attempts.
    e2e_perf_gate: bool = True
    e2e_perf_margin: float = 1.03  # 3% tolerance — reject if opt > baseline * margin

    # E2E correctness timeout: seconds for the correctness subprocess to
    # complete. Includes torch.compile capture backend for weight extraction,
    # model compilation, and 3-seed verification. Increase for large models
    # or complex compilation paths.
    e2e_correctness_timeout: int = 180

    # Graph tracing timeout: seconds for the torch.compile tracing subprocess.
    # Includes torch.export, Inductor compilation with autotuning, and kernel
    # extraction. Complex architectures (Inception, DenseNet, ViT) with many
    # modules require longer autotuning. Scale with model complexity.
    trace_timeout: int = 1800

    # Ablation flags
    no_memory: bool = False          # disable experience accumulation (q_cap=0)
    skip_ncu: bool = False           # skip NCU profiling, planner gets no bottleneck context
    optimize_all_kernels: bool = False  # disable Inductor-aware synthesis (optimize extern too)
    replay_profiles: str = ""        # replay NCU profiles from a prior run directory

    # Synthesis options (PyTorch → Triton)
    entry_point: str | None = None
    input_shapes: list[tuple[int, ...]] | None = None


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
    best_wall_ms: float = float("inf")       # wall-clock end-to-end (primary speedup metric)
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

    # Multi-kernel: benchmark source for rebuild_standalone_from_body
    benchmark_kernel_source: str | None = None


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
    baseline_wall_ms: float              # wall-clock end-to-end timing
    # NCU config from baseline profiling — reused for all candidates
    ncu_config: Optional[dict]

    # Current best (legacy scalar — kept for backward compatibility)
    best_kernel: str
    best_latency_ns: float
    best_wall_ms: float                  # wall-clock end-to-end (primary speedup metric)
    best_ncu_rep: str
    current_profiling_context: dict

    # Multi-chain forest search (ThoughtForest-style)
    active_kernels: list[dict]           # list of ActiveKernel as dicts (beam)
    global_best_kernel: str              # best kernel ever found across all chains
    global_best_latency_ns: float        # corresponding latency

    # Per-iteration
    iteration: int
    no_improvement_streak: int
    plans: list[tuple[str, dict]] | list[tuple[str, dict, str]]  # (letter, plan) or (letter, plan, chain_id)
    candidates: list[PlanCandidate]
    correctness_failure_items: list   # implementation_flaw items from correctness failures

    # Memory
    optimization_memory: list[dict]

    # History
    iteration_results: list[IterationResult]

    # Strategy analysis (two-phase planning)
    optimization_strategy: Optional[dict]

    # Synthesis (Feature 4): original PyTorch source when input was auto-converted
    _original_pytorch_path: Optional[str]
    _original_pytorch_source: Optional[str]
    _benchmark_kernel_source: Optional[str]
    _all_benchmark_files: Optional[list]
    _baseline_standalone_path: Optional[str]
    _target_triton_kernel_name: Optional[str]

    # Internal: LangGraph entry routing (fresh | after_baseline | after_iteration)
    _entry: str
