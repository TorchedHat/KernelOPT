"""Resume checkpoints for kernelopt run (written under output_dir)."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from kernelopt.state import GraphState, IterationResult, PipelineConfig, PlanCandidate

CHECKPOINT_FILENAME = "pipeline_checkpoint.json"
CHECKPOINT_VERSION = 1

# `entry` values — next run starts at the node after the completed stage:
#   after_baseline → generate_plans
#   after_plans → execute_plans (same plans, no re-plan)
#   after_execute → profile_candidates (re-runs NCU if interrupted mid-profile)
#   after_summarize → select_best
#   after_select → increment_iteration
#   after_iteration → continuation (should_continue → more iterations or END)


def pipeline_config_from_checkpoint_dict(d: dict) -> PipelineConfig:
    """Rebuild PipelineConfig from graph_state.config in a checkpoint JSON file."""
    known = {f.name for f in fields(PipelineConfig)}
    kwargs = {k: d[k] for k in known if k in d}
    return PipelineConfig(**kwargs)


def _plan_candidate_from_dict(d: dict) -> PlanCandidate:
    known = {f.name for f in fields(PlanCandidate)}
    return PlanCandidate(**{k: v for k, v in d.items() if k in known})


def _iteration_result_from_dict(d: dict) -> IterationResult:
    cands = [_plan_candidate_from_dict(x) for x in d.get("candidates", [])]
    bc = d.get("best_candidate")
    best = _plan_candidate_from_dict(bc) if bc else None
    return IterationResult(
        iteration=d.get("iteration", 0),
        candidates=cands,
        best_candidate=best,
        improved=d.get("improved", False),
    )


def _plans_to_json(plans: list[tuple[str, dict]]) -> list[dict[str, Any]]:
    return [{"letter": letter, "plan": plan} for letter, plan in plans]


def _plans_from_json(data: list[dict[str, Any]]) -> list[tuple[str, dict]]:
    return [(x["letter"], x["plan"]) for x in data]


def serialize_for_checkpoint(state: GraphState) -> dict[str, Any]:
    """Build JSON-serializable payload (excluding non-data helpers)."""
    cfg = state["config"]
    cfg_d = asdict(cfg) if isinstance(cfg, PipelineConfig) else dict(cfg)

    iteration_results = state.get("iteration_results", [])
    ir_json = []
    for ir in iteration_results:
        ir_json.append(
            {
                "iteration": ir.iteration,
                "candidates": [asdict(c) for c in ir.candidates],
                "best_candidate": asdict(ir.best_candidate) if ir.best_candidate else None,
                "improved": ir.improved,
            }
        )

    candidates = state.get("candidates", [])
    cand_json = [asdict(c) for c in candidates]

    plans = state.get("plans", [])
    plans_json = _plans_to_json(plans) if plans else []

    return {
        "config": cfg_d,
        "kernel_path": state["kernel_path"],
        "kernel_source": state["kernel_source"],
        "kernel_basename": state["kernel_basename"],
        "kernel_id": state.get("kernel_id", 0),
        "framework": state["framework"],
        "output_dir": state["output_dir"],
        "ext": state["ext"],
        "baseline_profiling_context": state.get("baseline_profiling_context", {}),
        "baseline_csv": state.get("baseline_csv", ""),
        "baseline_ncu_rep": state.get("baseline_ncu_rep", ""),
        "baseline_latency_ns": state.get("baseline_latency_ns", float("inf")),
        "ncu_config": state.get("ncu_config"),
        "best_kernel": state.get("best_kernel", ""),
        "best_latency_ns": state.get("best_latency_ns", float("inf")),
        "best_ncu_rep": state.get("best_ncu_rep", ""),
        "current_profiling_context": state.get("current_profiling_context", {}),
        "iteration": state.get("iteration", 0),
        "no_improvement_streak": state.get("no_improvement_streak", 0),
        "optimization_memory": list(state.get("optimization_memory", [])),
        "iteration_results": ir_json,
        "candidates": cand_json,
        "plans": plans_json,
        "correctness_failure_items": list(state.get("correctness_failure_items", [])),
        "optimization_strategy": state.get("optimization_strategy"),
        "_original_pytorch_path": state.get("_original_pytorch_path"),
        "_original_pytorch_source": state.get("_original_pytorch_source"),
        "baseline_wall_ms": state.get("baseline_wall_ms", -1),
        "best_wall_ms": state.get("best_wall_ms", -1),
    }


def save_checkpoint(
    output_dir: Path,
    state: GraphState,
    entry: str,
) -> None:
    """Write pipeline_checkpoint.json (entry: after_baseline | after_iteration)."""
    payload = {
        "version": CHECKPOINT_VERSION,
        "entry": entry,
        "graph_state": serialize_for_checkpoint(state),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CHECKPOINT_FILENAME
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_checkpoint_file(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / CHECKPOINT_FILENAME
    if not path.is_file():
        return None
    with path.open() as f:
        data = json.load(f)
    if data.get("version") != CHECKPOINT_VERSION:
        return None
    return data


def graph_state_from_checkpoint_payload(
    payload: dict[str, Any],
    resume_dir: Path,
    *,
    live_config: PipelineConfig,
    live_kernel_path: str,
    live_kernel_source: str,
    live_kernel_basename: str,
    live_framework: str,
) -> tuple[GraphState, str]:
    """
    Rebuild GraphState from checkpoint graph_state.
    Merges live CLI config and kernel file with saved optimization progress.
    Returns (graph_state, entry) where entry is after_baseline | after_iteration.
    """
    entry = payload.get("entry", "after_baseline")
    gs = payload["graph_state"]

    saved_kernel = gs.get("kernel_path", "")
    if saved_kernel and Path(saved_kernel).name != Path(live_kernel_path).name:
        raise ValueError(
            f"Checkpoint kernel {saved_kernel!r} does not match {live_kernel_path!r} "
            "(basename must match to resume)."
        )

    iteration_results = [
        _iteration_result_from_dict(x) for x in gs.get("iteration_results", [])
    ]
    candidates = [_plan_candidate_from_dict(x) for x in gs.get("candidates", [])]
    plans_raw = gs.get("plans", [])
    plans = _plans_from_json(plans_raw) if plans_raw else []

    graph_state: GraphState = {
        "config": live_config,
        "kernel_path": live_kernel_path,
        "kernel_source": live_kernel_source,
        "kernel_basename": live_kernel_basename,
        "kernel_id": gs.get("kernel_id", 0),
        "framework": gs.get("framework", live_framework),
        "output_dir": str(resume_dir.resolve()),
        "ext": gs.get("ext", Path(live_kernel_path).suffix),
        "baseline_profiling_context": gs.get("baseline_profiling_context", {}),
        "baseline_csv": gs.get("baseline_csv", ""),
        "baseline_ncu_rep": gs.get("baseline_ncu_rep", ""),
        "baseline_latency_ns": gs.get("baseline_latency_ns", float("inf")),
        "ncu_config": gs.get("ncu_config"),
        "best_kernel": gs.get("best_kernel", live_kernel_source),
        "best_latency_ns": gs.get("best_latency_ns", float("inf")),
        "best_ncu_rep": gs.get("best_ncu_rep", ""),
        "current_profiling_context": gs.get("current_profiling_context", {}),
        "iteration": gs.get("iteration", 0),
        "no_improvement_streak": gs.get("no_improvement_streak", 0),
        "optimization_memory": list(gs.get("optimization_memory", [])),
        "iteration_results": iteration_results,
        "candidates": candidates,
        "plans": plans,
        "correctness_failure_items": list(gs.get("correctness_failure_items", [])),
        "optimization_strategy": gs.get("optimization_strategy"),
        "_original_pytorch_path": gs.get("_original_pytorch_path"),
        "_original_pytorch_source": gs.get("_original_pytorch_source"),
        "baseline_wall_ms": gs.get("baseline_wall_ms", -1),
        "best_wall_ms": gs.get("best_wall_ms", -1),
    }

    return graph_state, entry


def load_baseline_artifacts(run_dir: Path) -> dict[str, Any] | None:
    """
    If baseline profiling finished but no checkpoint, load baseline CSV + context JSON.
    Returns dict suitable for merging into initial GraphState, or None.
    """
    ctx_path = run_dir / "baseline_profiling_context.json"
    csv_path = run_dir / "baseline.csv"
    if not ctx_path.is_file() or not csv_path.is_file():
        return None
    with ctx_path.open() as f:
        baseline_ctx = json.load(f)
    rep = run_dir / "baseline.ncu-rep"
    return {
        "baseline_profiling_context": baseline_ctx,
        "baseline_csv": str(csv_path.resolve()),
        "baseline_ncu_rep": str(rep.resolve()) if rep.is_file() else str(rep),
    }
