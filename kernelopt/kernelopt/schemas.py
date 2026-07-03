from __future__ import annotations

from pydantic import BaseModel, Field


class MetricValue(BaseModel):
    value: float | str
    unit: str


class LaunchConfig(BaseModel):
    block_size: int
    grid_size: int
    sm_count: int | None = None
    waves_per_sm: float | None = None
    registers_per_thread: int | None = None


class NCURule(BaseModel):
    name: str
    type: str  # OPT | INF | WRN
    description: str
    speedup_type: str | None = None  # global | local
    speedup_pct: float | None = None


class ProfilingContext(BaseModel):
    kernel_name: str
    kernel_name_full: str
    kernel_id: int = 0
    framework_hint: str = "cuda"
    compute_capability: str = ""
    launch: dict = Field(default_factory=dict)
    sections: dict[str, dict[str, MetricValue]] = Field(default_factory=dict)
    rules: list[NCURule] = Field(default_factory=list)


class PlanEvidence(BaseModel):
    ncu_rule: str
    ncu_rule_speedup_pct: float | None = None
    supporting_metrics: dict[str, float | str] = Field(default_factory=dict)
    reasoning: str = ""


class PlanStep(BaseModel):
    action: str  # add | modify | remove | replace
    target: str
    change: str
    predicted_effect: str = ""


class OptimizationPlan(BaseModel):
    plan_id: str
    direction: str
    framework: str = "cuda"
    evidence: PlanEvidence
    step: PlanStep
    correctness_constraints: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class ProfilingSignal(BaseModel):
    ncu_rule_triggered: str = ""
    key_metric_before: dict[str, float | str] = Field(default_factory=dict)
    key_metric_after: dict[str, float | str] = Field(default_factory=dict)


class ExperienceItem(BaseModel):
    item_id: str
    iteration: int
    speedup: float
    rewrite_type: str  # positive | negative
    framework: str = "cuda"
    direction: str = ""
    profiling_signal: ProfilingSignal = Field(default_factory=ProfilingSignal)
    strategy_title: str = ""
    strategy_description: str = ""
    slow_pseudocode: str = ""
    fast_pseudocode: str = ""
    applicable_when: str = ""
    do_not_apply_when: str = ""
    framework_notes: str = ""
