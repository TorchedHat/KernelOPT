"""Pydantic schemas for structured outputs from agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Evidence supporting an optimization plan."""
    ncu_rule: str | None = Field(
        None,
        description="NCU rule name that motivated this optimization"
    )
    ncu_rule_speedup_pct: float | None = Field(
        None,
        description="Potential speedup percentage from NCU rule"
    )
    supporting_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Key metrics from profiling context (Section.Metric: value)"
    )
    reasoning: str = Field(
        ...,
        description="2-3 sentences explaining what the data says and why it points to this change"
    )


class SuccessCriteria(BaseModel):
    """Expected impact of optimization."""
    primary_metric: str = Field(
        ...,
        description="Primary metric to track (e.g., Memory Throughput, Duration)"
    )
    expected_direction: Literal["decrease", "increase", "stable"] = Field(
        ...,
        description="Expected direction of change"
    )
    expected_magnitude: str = Field(
        ...,
        description="Expected magnitude of change (e.g., '10-20%')"
    )
    secondary_effects: str = Field(
        "",
        description="Other metrics that may change"
    )


class Step(BaseModel):
    """Implementation step for optimization."""
    action: Literal[
        "reduce", "increase", "add", "modify", "hoist",
        "fuse", "tile", "pad", "split", "reorder", "replace"
    ] = Field(
        ...,
        description="Type of transformation to apply"
    )
    target: str = Field(
        ...,
        description="What to modify (e.g., loop, variable, parameter)"
    )
    change: str = Field(
        ...,
        description="Exact change description for executor"
    )
    predicted_effect: str = Field(
        ...,
        description="Which metric should improve, direction, and magnitude"
    )
    implementation_hints: list[str] = Field(
        default_factory=list,
        description="Concrete guidance with code snippets or parameter values"
    )
    success_criteria: SuccessCriteria | None = Field(
        None,
        description="Expected impact metrics"
    )


class OptimizationPlan(BaseModel):
    """Complete optimization plan from Planner agent."""
    plan_id: str = Field(
        ...,
        description="Short unique identifier for this plan"
    )
    direction: str = Field(
        ...,
        description="Descriptive snake_case name for this optimization"
    )
    framework: Literal["triton", "helion", "unknown"] = Field(
        ...,
        description="Detected kernel framework"
    )
    evidence: Evidence = Field(
        ...,
        description="Profiling evidence supporting this optimization"
    )
    step: Step = Field(
        ...,
        description="Implementation instructions"
    )
    correctness_constraints: list[str] = Field(
        default_factory=list,
        description="Invariants executor must preserve"
    )
    architecture_notes: str | None = Field(
        None,
        description="Why this optimization works on detected GPU architecture"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level in this optimization"
    )
    confidence_reason: str = Field(
        "",
        description="Explanation for confidence level"
    )


class ProfilingSignal(BaseModel):
    """Profiling metrics before and after optimization."""
    ncu_rule_triggered: str | None = Field(
        None,
        description="NCU rule that was triggered"
    )
    key_metric_before: dict[str, float] = Field(
        default_factory=dict,
        description="Key metrics before optimization"
    )
    key_metric_after: dict[str, float] = Field(
        default_factory=dict,
        description="Key metrics after optimization"
    )


class ExperienceItem(BaseModel):
    """Experience item from Summarizer agent."""
    item_id: str = Field(
        ...,
        description="Unique identifier for this experience"
    )
    iteration: int = Field(
        ...,
        description="Which iteration this came from"
    )
    speedup: float = Field(
        ...,
        description="Speedup achieved (>1.0 = improvement, <1.0 = regression)"
    )
    rewrite_type: Literal["positive", "negative", "implementation_flaw"] = Field(
        ...,
        description="Classification: positive (worked), negative (strategy wrong), implementation_flaw (strategy ok, code bad)"
    )
    framework: str = Field(
        ...,
        description="Framework used (triton, helion, unknown, multi)"
    )
    direction: str = Field(
        ...,
        description="Snake_case strategy name"
    )
    profiling_signal: ProfilingSignal = Field(
        ...,
        description="Profiling metrics before and after"
    )
    strategy_title: str = Field(
        ...,
        description="Short title (≤10 words)"
    )
    strategy_description: str = Field(
        ...,
        description="2-3 sentences, generalizable insight"
    )
    slow_pseudocode: str = Field(
        ...,
        description="Minimal before code"
    )
    fast_pseudocode: str = Field(
        ...,
        description="Minimal after code"
    )
    applicable_when: str = Field(
        ...,
        description="Profiling conditions that make this strategy relevant"
    )
    do_not_apply_when: str = Field(
        ...,
        description="Conditions where this causes regression"
    )
    framework_notes: str | None = Field(
        None,
        description="Framework-specific syntax notes"
    )


class TestInput(BaseModel):
    """Specification for one input passed to the kernel wrapper.

    Supports two kinds:
      - "tensor" (default): generates a PyTorch tensor from shape/dtype/distribution.
      - "scalar": passes a Python float/int/bool directly (e.g. scaling factors, dims).
    """
    kind: str = Field(
        "tensor",
        description="'tensor' for a PyTorch tensor, 'scalar' for a Python float/int/bool"
    )
    # --- Tensor fields (used when kind="tensor") ---
    shape: list[int] | None = Field(None, description="Tensor shape, e.g. [4096, 1024]. Required for kind='tensor'.")
    dtype: str | None = Field(None, description="PyTorch dtype: 'float32', 'float16', 'bfloat16', 'int32', 'int64'. Required for kind='tensor'.")
    distribution: str = Field(
        "randn",
        description="'randn' (normal), 'rand' (uniform 0-1), 'randint', 'ones', 'zeros'"
    )
    low: float | None = Field(None, description="Lower bound for randint")
    high: float | None = Field(None, description="Upper bound for randint")
    # --- Scalar fields (used when kind="scalar") ---
    value: float | int | bool | None = Field(None, description="The scalar value to pass (e.g. 1.0, 3, True). Required for kind='scalar'.")
    scalar_dtype: str | None = Field(None, description="'float', 'int', or 'bool'. Defaults to 'float'.")


class CorrectnessSpec(BaseModel):
    """Structured spec the correctness agent returns. Python enforces the test."""
    wrapper_function: str = Field(
        ...,
        description=(
            "Name of the plain Python wrapper function in the kernel file. "
            "E.g. 'run_triton_softmax'. NOT the @triton.jit kernel itself."
        )
    )
    reference_code: str = Field(
        ...,
        description=(
            "Complete Python source defining a function named exactly 'reference(*args)'. "
            "Must use only torch CPU operations. Receives the same positional args as "
            "wrapper_function but as CPU tensors. "
            "Start with 'import torch' if needed. "
            "Example: 'def reference(x):\\n    return torch.softmax(x, dim=-1)'"
        )
    )
    test_inputs: list[TestInput] = Field(
        ...,
        description="Ordered specs for each input tensor passed to wrapper_function"
    )
    atol: float = Field(1e-3, description="Absolute tolerance for torch.allclose")
    rtol: float = Field(1e-3, description="Relative tolerance for torch.allclose")
    skip: bool = Field(False, description="True only if automated testing is impossible")
    skip_reason: str | None = Field(None, description="Required when skip=True")


class MemoryUpdate(BaseModel):
    """Memory update action from Summarizer."""
    action: Literal["append", "replace", "skip"] = Field(
        ...,
        description="What to do with this experience item"
    )
    replace_item_id: str | None = Field(
        None,
        description="If action=replace, which item to replace"
    )
    reason: str = Field(
        ...,
        description="One sentence explanation"
    )


class SummarizerOutput(BaseModel):
    """Combined output from Summarizer agent."""
    experience_item: ExperienceItem = Field(
        ...,
        description="The experience item to add to memory"
    )
    memory_update: MemoryUpdate = Field(
        ...,
        description="How to update the memory queue"
    )


