"""
LangGraph-based optimization pipeline.

Graph:
  profile_baseline → generate_plans → execute_plans → profile_candidates
       → summarize_results → select_best → increment_iteration
       → {should_continue: continue → generate_plans | stop → END}
"""

from __future__ import annotations

import difflib
import json
import shutil
import string
import uuid
from pathlib import Path

from langgraph.graph import END, StateGraph

try:
    from langgraph.graph import START
except ImportError:  # pragma: no cover
    START = "__start__"
from rich.console import Console
from rich.table import Table

from kernelopt.agents.executor import run_executor_agent
from kernelopt.agents.summarizer import run_summarizer
from kernelopt.functions.compiler import compile_cuda
from kernelopt.functions.correctness import check_correctness
from kernelopt.functions.gpu_monitoring import (
    monitor_gpu_activity,
    select_gpu,
)
from kernelopt.checkpoint import (
    CHECKPOINT_FILENAME,
    graph_state_from_checkpoint_payload,
    load_baseline_artifacts,
    load_checkpoint_file,
    save_checkpoint,
)
from kernelopt.errors import KerneloptUserError
from kernelopt.iteration_metric_deltas import (
    LOG_METRIC_ALTERNATIVES as _LOG_METRIC_ALTERNATIVES,
    LOG_METRIC_COMPARISON_ROWS,
    delta_cell as _delta_cell,
    duration_ns_display as _duration_ns_display,
    metric_numeric as _metric_numeric,
    metric_value_str as _m,
    metric_value_str_any as _m_any,
)
from kernelopt.memory import append_items, load_memory, save_memory, select_top_k
from kernelopt.pipeline_ui import (
    HUD_TERMINAL_LOG_NAME,
    get_hud_log_console,
    get_pipeline_ui,
    pipe_print,
    run_primary_hud_session,
)
from kernelopt.progress import phase_spinner
from kernelopt.state import (
    GraphState,
    IterationResult,
    PipelineConfig,
    PipelineState,
    PlanCandidate,
)

console = Console()
PLAN_LETTERS = list(string.ascii_uppercase)


def _use_spinners() -> bool:
    """Show animated spinners only when HUD is off (non-HUD terminal mode)."""
    return get_pipeline_ui() is None


def _checkpoint_stage(state: GraphState, updates: dict, stage: str) -> None:
    """Persist merged state after a pipeline stage (for --checkpoint resume)."""
    try:
        merged: GraphState = {**state, **updates}  # type: ignore[misc]
        save_checkpoint(Path(str(merged["output_dir"])), merged, stage)
    except Exception as e:
        pipe_print(f"[yellow]Warning:[/] checkpoint [{stage}] failed: {e}")


def _compile_cuda(cu_path: str, output_dir: str | None = None) -> str:
    """Compile a .cu file with nvcc and return the binary path. Raises on failure."""
    binary_path, err = compile_cuda(cu_path, output_dir=output_dir)
    if err:
        raise RuntimeError(f"nvcc failed:\n{err}")
    return binary_path


# ---------------------------------------------------------------------------
# Node 1: Profile baseline kernel
# ---------------------------------------------------------------------------

def profile_baseline(state: GraphState) -> dict:
    """Run NCU on the original kernel and establish baseline metrics."""
    from kernelopt.agents.profiler import (
        build_profiling_context,
        get_latency_ns,
        run_profiler_agent,
    )
    from kernelopt.functions.csv_parser import parse_ncu_csv

    cfg = state["config"]
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = state["kernel_path"]
    kernel_id = state.get("kernel_id", 0)

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Baseline profiling", Path(kernel_path).name)

    # For .cu kernels: compile the baseline binary once so NCU can run it.
    ext = Path(kernel_path).suffix
    compiled_binary = None
    if ext == ".cu":
        compiled_binary, err = compile_cuda(kernel_path, output_dir=str(output_dir))
        if err:
            raise RuntimeError(
                f"Baseline .cu compilation failed:\n{err}\n\n"
                f"Fix the kernel source and rerun."
            )
        pipe_print(f"  [dim]Compiled baseline binary: {compiled_binary}[/]")

    llm_config = {
        "model": cfg.profiler_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    if cfg.baseline_csv and Path(cfg.baseline_csv).exists():
        pipe_print("[dim]Using provided baseline CSV[/]")
        sections, rules, launch = parse_ncu_csv(cfg.baseline_csv, kernel_id)
        baseline_ctx = build_profiling_context(
            sections, rules, launch, kernel_path, kernel_id=kernel_id
        )
        ncu_config = None
    else:
        pipe_print("[bold]Profiling baseline kernel...[/]")
        try:
            if _use_spinners():
                with phase_spinner("baseline", Path(kernel_path).name):
                    baseline_ctx, ncu_config = run_profiler_agent(
                        kernel_path,
                        str(output_dir / "baseline"),
                        ncu_config=None,
                        pipeline_config=cfg,
                        kernel_id=kernel_id,
                        llm_config=llm_config,
                        binary_path=compiled_binary,
                    )
            else:
                baseline_ctx, ncu_config = run_profiler_agent(
                    kernel_path,
                    str(output_dir / "baseline"),
                    ncu_config=None,
                    pipeline_config=cfg,
                    kernel_id=kernel_id,
                    llm_config=llm_config,
                    binary_path=compiled_binary,
                )
        except RuntimeError as e:
            raise RuntimeError(f"Baseline profiling failed: {e}") from e
        ctx_json = str(output_dir / "baseline_profiling_context.json")
        pipe_print(f"  [dim]Profiling context: {ctx_json}[/]")

    latency = get_latency_ns(baseline_ctx)
    framework = state["framework"]
    if framework == "auto":
        framework = baseline_ctx["framework_hint"]

    pipe_print(
        f"  Baseline: [cyan]{latency:.0f} ns[/]  "
        f"Framework: [green]{framework}[/]"
    )

    ui_done = get_pipeline_ui()
    if ui_done:
        ui_done.ingest_post_baseline(
            baseline_latency_ns=latency,
            best_latency_ns=latency,
            framework=str(framework),
            iteration_0based=state.get("iteration", 0),
        )

    out = {
        "baseline_profiling_context": baseline_ctx,
        "baseline_csv": cfg.baseline_csv or str(output_dir / "baseline.csv"),
        "baseline_ncu_rep": str(output_dir / "baseline.ncu-rep"),
        "baseline_latency_ns": latency,
        "best_latency_ns": latency,
        "current_profiling_context": baseline_ctx,
        "framework": framework,
        "ncu_config": ncu_config,
    }
    _checkpoint_stage(state, out, "after_baseline")
    return out


# ---------------------------------------------------------------------------
# Node 2: Generate N optimization plans (concurrent for TorchServe batching)
# ---------------------------------------------------------------------------

def generate_plans(state: GraphState) -> dict:
    """Generate N diverse optimization plans concurrently via parallel planner sub-graphs."""
    from kernelopt.agents.planner import run_planner_agent

    cfg = state["config"]
    profiling_ctx = state["current_profiling_context"]
    iteration = state.get("iteration", 0)
    baseline = state.get("baseline_latency_ns", float("inf"))
    best_speedup = round(baseline / state["best_latency_ns"], 2)

    ui = get_pipeline_ui()
    if ui:
        ui.sync_iteration_header(
            iteration,
            baseline,
            state["best_latency_ns"],
        )
        ui.set_phase("Planning", f"{cfg.N} optimization plans")

    pipe_print(
        f"\n[bold]Iteration {iteration + 1}/{cfg.T}[/]  "
        f"Best: [green]{best_speedup}x[/]"
    )

    # Per-iteration GPU monitoring (non-blocking warnings; main terminal)
    gpu_id = select_gpu()
    monitor_gpu_activity(gpu_id, iteration + 1, console=console)

    pipe_print(f"  Generating {cfg.N} optimization plans...")

    # Build cross-iteration direction history
    all_tried_directions = [
        c.direction
        for result in state.get("iteration_results", [])
        for c in result.candidates
    ]

    llm_config = {
        "model": cfg.planner_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    _plan_args = dict(
        profiling_context=profiling_ctx,
        kernel_source=state["best_kernel"],
        memory=state.get("optimization_memory", []),
        N=cfg.N,
        llm_config=llm_config,
        all_tried_directions=all_tried_directions,
    )
    if _use_spinners():
        with phase_spinner("planning", f"{cfg.N} optimization plans"):
            plans = run_planner_agent(**_plan_args)
    else:
        plans = run_planner_agent(**_plan_args)

    for letter, plan in plans:
        direction = plan.get("direction", "unknown")
        confidence = plan.get("confidence", "?")
        pipe_print(
            f"    Plan {letter}: {direction} [dim](confidence: {confidence})[/]"
        )

    out = {"plans": plans}
    _checkpoint_stage(state, out, "after_plans")
    return out


# ---------------------------------------------------------------------------
# Node 3: Execute each plan (K attempts) + correctness gate
# ---------------------------------------------------------------------------

def execute_plans(state: GraphState) -> dict:
    """Apply each plan to produce optimized kernels and check correctness."""
    cfg = state["config"]
    plans = state["plans"]
    current_kernel = state["best_kernel"]
    framework = state["framework"]
    output_dir = Path(state["output_dir"])
    ext = state["ext"]
    iteration = state.get("iteration", 0)
    kernel_basename = state["kernel_basename"]
    kernel_path = state["kernel_path"]

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Execute + correctness", f"{len(plans)} plans")

    llm_config = {
        "model": cfg.executor_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    candidates: list[PlanCandidate] = []
    correctness_failure_items: list[dict] = []

    for plan_letter, plan in plans:
        direction = plan.get("direction", "unknown")
        safe_dir = direction.replace(" ", "_")[:40]
        file_stem = f"{kernel_basename}_iter{iteration + 1}_plan{plan_letter}_{safe_dir}"

        pipe_print(f"  [dim]Executing plan {plan_letter} ({direction})...[/]")

        last_compile_err: str | None = None
        optimized_source: str | None = None
        execution_result: dict = {}
        compiled_binary: str | None = None
        outer_attempts = cfg.K if ext == ".cu" else 1
        executor_K = 1 if ext == ".cu" else cfg.K

        for attempt_idx in range(outer_attempts):
            _exec_label = f"plan {plan_letter} — {direction}"
            try:
                _exec_args = dict(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    memory=state.get("optimization_memory", []),
                    K=executor_K,
                    llm_config=llm_config,
                    compile_error=last_compile_err,
                )
                if _use_spinners():
                    with phase_spinner("executing", _exec_label):
                        optimized_source, execution_result = run_executor_agent(**_exec_args)
                else:
                    optimized_source, execution_result = run_executor_agent(**_exec_args)
            except Exception as e:
                pipe_print(f"    [red]Executor failed (plan {plan_letter}): {e}[/]")
                optimized_source = None
                break

            pipe_print(f"  [dim]Execution status: {execution_result.get('status')}[/]")
            pipe_print(f"  [dim]Ready for profiling: {execution_result.get('ready_for_profiling')}[/]")
            pipe_print(f"  [dim]Lines changed: {execution_result.get('lines_changed')}[/]")

            if not execution_result.get("ready_for_profiling", False):
                reason = execution_result.get("reason", execution_result.get("status", "failed"))
                pipe_print(f"    [red]Plan {plan_letter}: {reason}[/]")
                optimized_source = None
                break

            # Save source to disk
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)

            if ext == ".cu":
                binary_path, err = compile_cuda(kernel_file, output_dir=str(output_dir))
                if err:
                    last_compile_err = err
                    pipe_print(
                        f"    [yellow]Plan {plan_letter} attempt {attempt_idx + 1}/"
                        f"{outer_attempts}: compile failed[/]\n"
                        f"    {err.splitlines()[0] if err else ''}"
                    )
                    if attempt_idx == outer_attempts - 1:
                        # All K attempts exhausted — record for planner memory
                        correctness_failure_items.append({
                            "item_id": uuid.uuid4().hex[:8],
                            "iteration": iteration + 1,
                            "speedup": 0.0,
                            "rewrite_type": "implementation_flaw",
                            "framework": framework,
                            "direction": direction,
                            "profiling_signal": {
                                "ncu_rule_triggered": plan.get("evidence", {}).get("ncu_rule"),
                                "key_metric_before": {},
                                "key_metric_after": {},
                            },
                            "strategy_title": f"{direction} — compile failed after {outer_attempts} attempts",
                            "strategy_description": (
                                f"Plan '{direction}' produced a .cu file that nvcc rejected "
                                f"after {outer_attempts} attempt(s). "
                                f"Last error: {(last_compile_err or '')[:300]}"
                            ),
                            "slow_pseudocode": "",
                            "fast_pseudocode": "",
                            "applicable_when": plan.get("evidence", {}).get("reasoning", ""),
                            "do_not_apply_when": f"Compilation failed: {(last_compile_err or '')[:200]}",
                            "framework_notes": None,
                        })
                        optimized_source = None
                    continue  # retry with error feedback

                compiled_binary = binary_path

            break  # compiled successfully (or non-.cu)

        if optimized_source is None:
            continue

        execution_result["correctness"] = "unverified"

        pipe_print(f"  [dim]Running correctness check...[/]")
        _cc_args = dict(
            original_path=kernel_path,
            optimized_source=optimized_source,
            framework=framework,
            use_agentic=cfg.agentic_correctness,
            api_key=cfg.api_key,
            project_id=cfg.project_id,
            region=cfg.region,
            kernel_args=cfg.kernel_args,
            binary_path=compiled_binary,
        )
        if _use_spinners():
            with phase_spinner("correctness", f"plan {plan_letter}"):
                passed, msg = check_correctness(**_cc_args)
        else:
            passed, msg = check_correctness(**_cc_args)
        execution_result["correctness"] = "passed" if passed else "failed"

        status = "[green]passed[/]" if passed else "[red]failed[/]"
        pipe_print(f"    Plan {plan_letter}: correctness {status}")

        if not passed:
            pipe_print(f"    [dim]Failure: {msg[:300]}[/]")
            correctness_failure_items.append({
                "item_id": uuid.uuid4().hex[:8],
                "iteration": iteration + 1,
                "speedup": 0.0,
                "rewrite_type": "implementation_flaw",
                "framework": framework,
                "direction": direction,
                "profiling_signal": {
                    "ncu_rule_triggered": plan.get("evidence", {}).get("ncu_rule"),
                    "key_metric_before": {},
                    "key_metric_after": {},
                },
                "strategy_title": f"{direction} — correctness failure",
                "strategy_description": (
                    f"Plan '{direction}' was applied but failed correctness. "
                    f"Error: {msg[:300]}"
                ),
                "slow_pseudocode": "",
                "fast_pseudocode": "",
                "applicable_when": plan.get("evidence", {}).get("reasoning", ""),
                "do_not_apply_when": f"Previous implementation failed: {msg[:200]}",
                "framework_notes": None,
            })
            continue

        candidate = PlanCandidate(
            plan_id=plan.get("plan_id", ""),
            plan_letter=plan_letter,
            direction=direction,
            optimization_plan=plan,
            optimized_kernel=optimized_source,
            execution_result=execution_result,
            correctness_passed=passed,
            correctness_message=msg,
            compiled_binary_path=compiled_binary or "",
        )
        candidates.append(candidate)

    out = {"candidates": candidates, "correctness_failure_items": correctness_failure_items}
    _checkpoint_stage(state, out, "after_execute")
    return out


# ---------------------------------------------------------------------------
# Node 4: Profile valid candidates with NCU
# ---------------------------------------------------------------------------

def profile_candidates(state: GraphState) -> dict:
    """Run NCU on each candidate that passed correctness."""
    from kernelopt.agents.profiler import get_latency_ns, run_profiler_agent

    cfg = state["config"]
    candidates = list(state["candidates"])
    output_dir = Path(state["output_dir"])
    ext = state["ext"]
    iteration = state.get("iteration", 0)
    kernel_basename = state["kernel_basename"]
    kernel_id = state.get("kernel_id", 0)
    best_latency = state["best_latency_ns"]
    saved_ncu_config = state.get("ncu_config")  # reuse baseline config — full bypass

    llm_config = {
        "model": cfg.profiler_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Profiling (NCU)", f"{len(candidates)} candidates")

    pipe_print(f"  Profiling {len(candidates)} valid candidates...")

    for c in candidates:
        safe_dir = c.direction.replace(" ", "_")[:40]
        file_stem = (
            f"{kernel_basename}_iter{iteration + 1}"
            f"_plan{c.plan_letter}_{safe_dir}"
        )
        output_base = str(output_dir / file_stem)
        kernel_file = f"{output_base}{ext}"
        Path(kernel_file).write_text(c.optimized_kernel)

        candidate_binary = None
        if ext == ".cu":
            try:
                candidate_binary = _compile_cuda(kernel_file, output_dir=str(output_dir))
            except RuntimeError as e:
                pipe_print(f"    [red]Compile failed (plan {c.plan_letter}): {e}[/]")
                c.latency_ns = float("inf")
                c.speedup = 0.0
                continue

        try:
            _prof_label = f"plan {c.plan_letter} — {c.direction[:30]}"
            _prof_args = dict(
                kernel_path=kernel_file,
                output_basename=output_base,
                ncu_config=saved_ncu_config,
                pipeline_config=cfg,
                kernel_id=kernel_id,
                binary_path=candidate_binary,
                llm_config=llm_config,
            )
            if _use_spinners():
                with phase_spinner("profiling", _prof_label):
                    ctx, _ = run_profiler_agent(**_prof_args)
            else:
                ctx, _ = run_profiler_agent(**_prof_args)
            c.profiling_context = ctx
            c.csv_path = f"{output_base}.csv"
            c.ncu_rep_path = f"{output_base}.ncu-rep"
            c.latency_ns = get_latency_ns(ctx)
            c.speedup = (
                round(best_latency / c.latency_ns, 4)
                if c.latency_ns > 0 else 0.0
            )
            pipe_print(
                f"    Plan {c.plan_letter} ({c.direction}): "
                f"{c.latency_ns:.0f} ns -> {c.speedup:.2f}x"
            )
        except Exception as e:
            pipe_print(
                f"    [red]Profiling plan {c.plan_letter} failed: {e}[/]"
            )
            c.latency_ns = float("inf")
            c.speedup = 0.0

    if ui and getattr(ui, "is_hud_sink", False):
        try:
            try:
                term_cols = max(40, int(shutil.get_terminal_size().columns))
            except OSError:
                term_cols = 100
            _bl = state.get("baseline_latency_ns", float("inf"))
            try:
                _bl_f = float(_bl)
            except (TypeError, ValueError):
                _bl_f = float("inf")
            _bl_arg = (
                _bl_f
                if _bl_f == _bl_f and 0 < _bl_f < float("inf")
                else None
            )
            ui.set_plan_delta_from_profile(
                state.get("baseline_profiling_context") or {},
                candidates,
                baseline_latency_ns=_bl_arg,
                terminal_columns=term_cols,
            )
        except Exception:
            pass

    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node 5: Summarize results + update memory (concurrent)
# ---------------------------------------------------------------------------

def summarize_results(state: GraphState) -> dict:
    """Summarize optimization results and update experience memory."""
    cfg = state["config"]
    candidates = state["candidates"]
    current_kernel = state["best_kernel"]
    profiling_before = state["current_profiling_context"]
    framework = state["framework"]
    iteration = state.get("iteration", 0)
    memory = list(state.get("optimization_memory", []))

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Summarize + memory", "")

    _show_summarize_spinner = _use_spinners()

    summarize_candidates = sorted(
        [c for c in candidates if c.latency_ns < float("inf")],
        key=lambda c: abs(c.speedup - 1.0),
        reverse=True,
    )

    experience_items: list[dict] = []

    for c in summarize_candidates:
        try:
            _sum_args = dict(
                profiling_before=profiling_before,
                profiling_after=c.profiling_context,
                optimization_plan=c.optimization_plan,
                kernel_before=current_kernel,
                kernel_after=c.optimized_kernel,
                iteration=iteration + 1,
                framework=framework,
                speedup=c.speedup,
                existing_memory=memory,
                execution_result=c.execution_result,
                api_key=cfg.api_key,
                model=cfg.summarizer_model or cfg.model,
                project_id=cfg.project_id,
                region=cfg.region,
            )
            if _show_summarize_spinner:
                with phase_spinner("summarize", f"plan {c.plan_letter}"):
                    item, memory_update = run_summarizer(**_sum_args)
            else:
                item, memory_update = run_summarizer(**_sum_args)

            # Apply deduplication immediately
            if memory_update["action"] == "append":
                experience_items.append(item)
            elif memory_update["action"] == "replace":
                replace_id = memory_update["replace_item_id"]
                memory = [m for m in memory if m.get("item_id") != replace_id]
                experience_items.append(item)
            # else: skip (don't add)

        except Exception as e:
            pipe_print(
                f"    [red]Summarizer failed (plan {c.plan_letter}): {e}[/]"
            )

    # Merge implementation_flaw items from correctness failures — these bypass
    # the speedup curation threshold and go directly into memory so the planner
    # can avoid repeating broken implementations.
    failure_items = state.get("correctness_failure_items", [])
    if failure_items:
        memory = append_items(memory, failure_items, cfg.ExpN)

    # Apply memory curation for profiled candidates
    top_items = select_top_k(experience_items, cfg.TopK, cfg.t_pos, cfg.t_neg)
    if top_items:
        memory = append_items(memory, top_items, cfg.ExpN)

    out = {"optimization_memory": memory}
    _checkpoint_stage(state, out, "after_summarize")
    return out


# ---------------------------------------------------------------------------
# Node 6: Select best candidate
# ---------------------------------------------------------------------------

def select_best_candidate(state: GraphState) -> dict:
    """Pick the best candidate, update best kernel if improved."""
    candidates = state["candidates"]
    best_latency = state["best_latency_ns"]
    best_kernel = state["best_kernel"]
    best_ncu_rep = state.get("best_ncu_rep", "")
    current_ctx = state["current_profiling_context"]
    baseline_latency = state["baseline_latency_ns"]
    framework = state["framework"]
    iteration = state.get("iteration", 0)
    no_improvement_streak = state.get("no_improvement_streak", 0)
    iteration_results = list(state.get("iteration_results", []))
    memory = state.get("optimization_memory", [])

    iter_result = IterationResult(
        iteration=iteration + 1,
        candidates=candidates,
    )

    improved = False
    if candidates:
        best_c = min(candidates, key=lambda c: c.latency_ns)
        if best_c.latency_ns < best_latency:
            best_kernel = best_c.optimized_kernel
            best_latency = best_c.latency_ns
            best_ncu_rep = best_c.ncu_rep_path
            current_ctx = best_c.profiling_context
            iter_result.best_candidate = best_c
            iter_result.improved = True
            improved = True

    if improved:
        no_improvement_streak = 0
    else:
        no_improvement_streak += 1

    iteration_results.append(iter_result)

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Select best", "")

    # Print iteration table
    overall_speedup = round(baseline_latency / best_latency, 2)
    cfg = state["config"]
    t = Table(
        title=(
            f"Iteration {iteration + 1}/{cfg.T}  |  "
            f"Framework: {framework}  |  "
            f"Best: {overall_speedup}x"
        ),
        show_header=True,
    )
    t.add_column("Direction")
    t.add_column("Speedup", justify="right")
    t.add_column("Correctness")
    t.add_column("Kept?")
    for c in candidates:
        speedup_str = (
            f"{c.speedup:.2f}x" if c.latency_ns < float("inf") else "fail"
        )
        correct = "passed" if c.correctness_passed else "failed"
        is_best = (
            iter_result.best_candidate
            and c.plan_letter == iter_result.best_candidate.plan_letter
        )
        t.add_row(c.direction, speedup_str, correct, "Y" if is_best else "")

    (get_hud_log_console() or console).print(t)
    if ui and ui.is_hud_sink:
        summary_bits = [
            f"{c.plan_letter}: "
            f"{'%.2fx' % c.speedup if c.latency_ns < float('inf') else 'fail'}"
            for c in candidates
        ]
        ui.add_event(
            f"Iter {iteration + 1}/{cfg.T} cumulative {overall_speedup}x | "
            + (" | ".join(summary_bits) if summary_bits else "no candidates")
        )
        ui.sync_iteration_header(
            state.get("iteration", 0),
            baseline_latency,
            best_latency,
        )

    out = {
        "best_kernel": best_kernel,
        "best_latency_ns": best_latency,
        "best_ncu_rep": best_ncu_rep,
        "current_profiling_context": current_ctx,
        "no_improvement_streak": no_improvement_streak,
        "iteration_results": iteration_results,
    }
    _checkpoint_stage(state, out, "after_select")
    return out


# ---------------------------------------------------------------------------
# Node 7: Increment iteration
# ---------------------------------------------------------------------------

def increment_iteration(state: GraphState) -> dict:
    """Bump the iteration counter."""
    return {"iteration": state.get("iteration", 0) + 1}


# ---------------------------------------------------------------------------
# Conditional edge: should we loop or stop?
# ---------------------------------------------------------------------------

def should_continue(state: GraphState) -> str:
    """Decide whether to continue iterating or stop."""
    cfg = state["config"]
    iteration = state.get("iteration", 0)

    if iteration >= cfg.T:
        return "stop"
    if state.get("no_improvement_streak", 0) >= 2:
        pipe_print(
            "[yellow]Stopping:[/] 2 consecutive iterations "
            "with no improvement."
        )
        return "stop"

    # Check if all plans had low confidence in last iteration
    results = state.get("iteration_results", [])
    if results:
        latest = results[-1]
        if latest.candidates and all(
            c.optimization_plan.get("confidence") == "low"
            for c in latest.candidates
        ):
            pipe_print(
                "[yellow]Stopping:[/] All plans report low confidence "
                "-- NCU rule opportunities exhausted."
            )
            return "stop"

    return "continue"


# ---------------------------------------------------------------------------
# Resume / checkpoint helpers
# ---------------------------------------------------------------------------

def route_start(state: GraphState) -> str:
    """Route from START to the first node to run (fresh or --checkpoint resume)."""
    entry = state.get("_entry", "fresh")
    resume_targets = {
        "after_baseline": "generate_plans",
        "after_plans": "execute_plans",
        "after_execute": "profile_candidates",
        "after_summarize": "select_best",
        "after_select": "increment_iteration",
        "after_iteration": "continuation_entry",
    }
    if entry in resume_targets:
        return resume_targets[entry]
    return "profile_baseline"


def continuation_entry(state: GraphState) -> dict:
    """No-op node before should_continue when resuming mid-pipeline."""
    return {}


def save_iter_checkpoint(state: GraphState) -> dict:
    """Persist full graph state between iterations (enables --checkpoint resume)."""
    try:
        save_checkpoint(Path(state["output_dir"]), state, "after_iteration")
    except Exception as e:
        pipe_print(f"[yellow]Warning:[/] could not save checkpoint: {e}")
    return {}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph optimization pipeline."""
    graph = StateGraph(GraphState)

    graph.add_node("profile_baseline", profile_baseline)
    graph.add_node("generate_plans", generate_plans)
    graph.add_node("execute_plans", execute_plans)
    graph.add_node("profile_candidates", profile_candidates)
    graph.add_node("summarize_results", summarize_results)
    graph.add_node("select_best", select_best_candidate)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("continuation_entry", continuation_entry)
    graph.add_node("save_iter_checkpoint", save_iter_checkpoint)

    graph.add_conditional_edges(
        START,
        route_start,
        {
            "profile_baseline": "profile_baseline",
            "generate_plans": "generate_plans",
            "execute_plans": "execute_plans",
            "profile_candidates": "profile_candidates",
            "select_best": "select_best",
            "increment_iteration": "increment_iteration",
            "continuation_entry": "continuation_entry",
        },
    )
    graph.add_edge("profile_baseline", "generate_plans")
    graph.add_conditional_edges(
        "continuation_entry",
        should_continue,
        {"continue": "generate_plans", "stop": END},
    )
    graph.add_edge("generate_plans", "execute_plans")
    graph.add_edge("execute_plans", "profile_candidates")
    graph.add_edge("profile_candidates", "summarize_results")
    graph.add_edge("summarize_results", "select_best")
    graph.add_edge("select_best", "increment_iteration")

    graph.add_edge("increment_iteration", "save_iter_checkpoint")
    graph.add_conditional_edges(
        "save_iter_checkpoint",
        should_continue,
        {"continue": "generate_plans", "stop": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point: convert PipelineState ↔ GraphState
# ---------------------------------------------------------------------------

def _build_fresh_initial(state: PipelineState, cfg: PipelineConfig, ext: str) -> GraphState:
    initial: GraphState = {
        "config": cfg,
        "kernel_path": state.kernel_path,
        "kernel_source": state.kernel_source,
        "kernel_basename": state.kernel_basename,
        "kernel_id": state.kernel_id,
        "framework": state.framework,
        "output_dir": cfg.output_dir,
        "ext": ext,
        "best_kernel": state.best_kernel,
        "best_latency_ns": float("inf"),
        "best_ncu_rep": "",
        "current_profiling_context": {},
        "optimization_memory": state.optimization_memory,
        "iteration": 0,
        "no_improvement_streak": 0,
        "plans": [],
        "candidates": [],
        "iteration_results": [],
        "ncu_config": state.ncu_config,
        "_entry": "fresh",
    }
    if state.baseline_csv:
        initial["baseline_csv"] = state.baseline_csv
    return initial


def run_langgraph_pipeline(state: PipelineState) -> PipelineState:
    """Run the LangGraph optimization pipeline."""
    from kernelopt.agents.profiler import get_latency_ns

    cfg = state.config
    ext = Path(state.kernel_path).suffix

    graph = build_graph()

    resume_path = (
        Path(state.resume_run_dir).resolve()
        if state.resume_run_dir
        else None
    )

    if resume_path:
        if not resume_path.is_dir():
            raise KerneloptUserError(
                f"--checkpoint path is not an existing directory:\n{resume_path}"
            )
        ckpt_raw = load_checkpoint_file(resume_path)
        if ckpt_raw:
            entry = ckpt_raw.get("entry", "after_baseline")
            try:
                initial, _ = graph_state_from_checkpoint_payload(
                    ckpt_raw,
                    resume_path,
                    live_config=cfg,
                    live_kernel_path=state.kernel_path,
                    live_kernel_source=state.kernel_source,
                    live_kernel_basename=state.kernel_basename,
                    live_framework=state.framework,
                )
            except ValueError as exc:
                raise KerneloptUserError(str(exc)) from exc
            initial["_entry"] = entry
            pipe_print(
                f"[bold green]Resuming[/] from [cyan]{resume_path}/[/] "
                f"(checkpoint: [dim]{entry}[/])"
            )
        else:
            base = load_baseline_artifacts(resume_path)
            if not base:
                ckpt_path = resume_path / CHECKPOINT_FILENAME
                has_ckpt_file = ckpt_path.is_file()
                lines = [
                    "Cannot resume: this run directory has no usable checkpoint or baseline.",
                    "",
                    "Expected one of:",
                    f"  • {CHECKPOINT_FILENAME} — resume mid-run, or",
                    "  • baseline.csv and baseline_profiling_context.json — continue after baseline.",
                    "",
                    f"Directory: {resume_path}",
                ]
                if has_ckpt_file:
                    lines += [
                        "",
                        f"(Found {CHECKPOINT_FILENAME} but it could not be loaded — "
                        "wrong or missing version field.)",
                    ]
                lines.append("")
                lines.append(
                    "Omit --checkpoint to start fresh, or pass a pipeline_output/run_* "
                    "folder produced by kernelopt."
                )
                raise KerneloptUserError("\n".join(lines))
            mem_path = resume_path / "optimization_memory.json"
            mem = load_memory(mem_path) if mem_path.is_file() else []
            bctx = base["baseline_profiling_context"]
            latency = get_latency_ns(bctx)
            fw = state.framework
            if fw == "auto":
                fw = bctx.get("framework_hint", "triton")
            initial = {
                "config": cfg,
                "kernel_path": state.kernel_path,
                "kernel_source": state.kernel_source,
                "kernel_basename": state.kernel_basename,
                "kernel_id": state.kernel_id,
                "framework": fw,
                "output_dir": str(resume_path),
                "ext": ext,
                "best_kernel": state.kernel_source,
                "best_latency_ns": latency,
                "best_ncu_rep": "",
                "baseline_profiling_context": bctx,
                "baseline_csv": base["baseline_csv"],
                "baseline_ncu_rep": base["baseline_ncu_rep"],
                "baseline_latency_ns": latency,
                "current_profiling_context": bctx,
                "optimization_memory": mem,
                "iteration": 0,
                "no_improvement_streak": 0,
                "plans": [],
                "candidates": [],
                "iteration_results": [],
                "ncu_config": None,
                "_entry": "after_baseline",
            }
            pipe_print(
                f"[bold green]Resuming[/] from [cyan]{resume_path}/[/] "
                f"(baseline artifacts only — starting at iteration 1)"
            )
    else:
        initial = _build_fresh_initial(state, cfg, ext)

    watch_dir = Path(cfg.output_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    log_path = watch_dir / HUD_TERMINAL_LOG_NAME

    if cfg.hud:
        with run_primary_hud_session(log_path) as hud:
            if state.pre_run_system_snapshot:
                hud.set_pre_run_system_snapshot(state.pre_run_system_snapshot)
            hud.seed_from_graph(initial)
            final = graph.invoke(initial)
        pipe_print(f"\n  [dim]HUD:[/] full terminal stream was appended to [cyan]{log_path}[/]")
    else:
        final = graph.invoke(initial)

    # Map back to PipelineState
    state.best_kernel = final["best_kernel"]
    state.best_latency_ns = final["best_latency_ns"]
    state.best_ncu_rep = final.get("best_ncu_rep", "")
    state.baseline_csv = str(final.get("baseline_csv") or state.baseline_csv or "")
    state.baseline_ncu_rep = str(
        final.get("baseline_ncu_rep") or state.baseline_ncu_rep or ""
    )
    state.baseline_profiling_context = final.get("baseline_profiling_context", {})
    state.current_profiling_context = final.get("current_profiling_context", {})
    state.optimization_memory = final.get("optimization_memory", [])
    state.iteration_results = final.get("iteration_results", [])
    state.framework = final.get("framework", state.framework)
    state.ncu_config = final.get("ncu_config")
    if "iteration" in final:
        state.iteration = int(final["iteration"])
    if "no_improvement_streak" in final:
        state.no_improvement_streak = int(final["no_improvement_streak"])

    # Write final outputs
    output_dir = Path(cfg.output_dir)
    baseline_latency = final.get("baseline_latency_ns", state.best_latency_ns)
    _write_final_outputs(state, output_dir, ext, baseline_latency)

    return state


def _best_direction_for_latency(
    state: PipelineState, latency_ns: float, tol: float = 0.5
) -> str:
    """Find a candidate direction whose measured latency matches the run best."""
    for r in reversed(state.iteration_results):
        for c in r.candidates:
            if c.latency_ns < float("inf") and abs(c.latency_ns - latency_ns) <= tol:
                return c.direction
    return "baseline"


def _baseline_kernel_text_for_log(state: PipelineState) -> str:
    """Original kernel source for optimization_log diff (unchanged from run start)."""
    src = state.kernel_source or ""
    if src.strip():
        return src
    if state.kernel_path:
        try:
            return Path(state.kernel_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return ""


def _kernel_diff_log_lines(
    state: PipelineState,
    ext: str,
    *,
    max_diff_chars: int = 800_000,
) -> list[str]:
    """Markdown fragment: unified diff best_kernel vs baseline for optimization_log.md."""
    baseline = _baseline_kernel_text_for_log(state)
    best = state.best_kernel or ""
    a = baseline.splitlines()
    b = best.splitlines()
    body = "\n".join(
        difflib.unified_diff(
            a,
            b,
            fromfile="baseline_kernel",
            tofile="best_kernel",
            n=3,
        )
    )
    out: list[str] = ["\n## Best kernel vs baseline (diff)\n\n"]
    if not body.strip():
        out.append(
            "_No textual differences between baseline and best kernel "
            "(or baseline source was unavailable)._ \n\n"
        )
        return out
    if len(body) > max_diff_chars:
        body = (
            body[:max_diff_chars]
            + f"\n\n... (diff truncated; see `best_kernel{ext}` in this output folder.)\n"
        )
    out.extend(["```diff\n", body, "\n```\n\n"])
    return out


def _write_best_profiling_artifacts(
    state: PipelineState,
    output_dir: Path,
    baseline_latency: float,
) -> None:
    """Write stable best_kernel.ncu-rep, best_kernel.csv, best_kernel_profiling_context.json.

    If no candidate beats baseline (within tolerance), copies baseline artifacts.
    Otherwise copies from the winning candidate's NCU output stem (best_ncu_rep).
    """
    tol = 0.5
    best_ns = state.best_latency_ns
    best_ns_ok = best_ns == best_ns and 0 < best_ns < float("inf")
    beats_baseline = best_ns_ok and best_ns + tol < baseline_latency

    dest_rep = output_dir / "best_kernel.ncu-rep"
    dest_csv = output_dir / "best_kernel.csv"
    dest_ctx = output_dir / "best_kernel_profiling_context.json"

    if beats_baseline and (state.best_ncu_rep or "").strip():
        rep_path = Path(state.best_ncu_rep).resolve()
        if rep_path.is_file():
            shutil.copy2(rep_path, dest_rep)
        csv_path = rep_path.with_suffix(".csv")
        if csv_path.is_file():
            shutil.copy2(csv_path, dest_csv)
        ctx = state.current_profiling_context or {}
        if ctx:
            dest_ctx.write_text(
                json.dumps(ctx, indent=2),
                encoding="utf-8",
            )
        return

    # Best equals baseline: use baseline NCU outputs in this run directory (or external CSV).
    ctx_out = state.baseline_profiling_context or state.current_profiling_context or {}

    csv_src: Path | None = None
    cfg_csv = (state.config.baseline_csv or "").strip()
    if cfg_csv and Path(cfg_csv).is_file():
        csv_src = Path(cfg_csv)
    else:
        p = output_dir / "baseline.csv"
        if p.is_file():
            csv_src = p

    rep_src: Path | None = None
    local_rep = output_dir / "baseline.ncu-rep"
    if local_rep.is_file():
        rep_src = local_rep
    else:
        alt = (state.baseline_ncu_rep or "").strip()
        if alt:
            ap = Path(alt)
            if ap.is_file():
                rep_src = ap

    if rep_src is not None:
        shutil.copy2(rep_src, dest_rep)
    if csv_src is not None:
        shutil.copy2(csv_src, dest_csv)

    bl_json = output_dir / "baseline_profiling_context.json"
    if bl_json.is_file():
        shutil.copy2(bl_json, dest_ctx)
    elif ctx_out:
        dest_ctx.write_text(
            json.dumps(ctx_out, indent=2),
            encoding="utf-8",
        )


def _write_final_outputs(
    state: PipelineState,
    output_dir: Path,
    ext: str,
    baseline_latency: float,
) -> None:
    """Write best_kernel, best NCU artifacts, optimization_memory.json, and optimization_log.md."""
    best_path = output_dir / f"best_kernel{ext}"
    best_path.write_text(state.best_kernel)
    _write_best_profiling_artifacts(state, output_dir, baseline_latency)

    save_memory(
        state.optimization_memory,
        output_dir / "optimization_memory.json",
    )

    if (
        state.best_latency_ns > 0
        and state.best_latency_ns < float("inf")
    ):
        overall_speedup = round(baseline_latency / state.best_latency_ns, 4)
    else:
        overall_speedup = 1.0
    log_path = output_dir / "optimization_log.md"

    # --- Environment ---
    try:
        import torch
        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda or "—"
    except Exception:
        torch_ver = cuda_ver = "—"
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = "—"

    gpu_specs = state.baseline_profiling_context.get("gpu_specs", {})
    gpu_model = gpu_specs.get("gpu_model", "—")
    sm_count  = gpu_specs.get("sm_count", "—")
    mem_gb    = gpu_specs.get("total_memory_gb", "—")
    bw_gbs    = gpu_specs.get("peak_hbm_bandwidth_gbs", "—")
    cc        = state.baseline_profiling_context.get("compute_capability", "—")

    baseline_ctx = state.baseline_profiling_context
    baseline_launch = baseline_ctx.get("launch", {})
    kernel_name = baseline_ctx.get("kernel_name", "—")
    block = baseline_launch.get("block_size", "—")
    grid = baseline_launch.get("grid_size", "—")
    waves = baseline_launch.get("waves_per_sm", "—")
    regs = baseline_launch.get("registers_per_thread", "—")

    # Build a memory lookup: direction → memory item (latest / highest speedup)
    mem_by_dir: dict[str, dict] = {}
    for item in state.optimization_memory:
        d = item.get("direction", "")
        existing = mem_by_dir.get(d)
        if existing is None or item.get("speedup", 0) > existing.get("speedup", 0):
            mem_by_dir[d] = item

    best_ns = state.best_latency_ns
    best_ns_ok = best_ns < float("inf") and best_ns > 0
    beat_baseline = best_ns_ok and best_ns + 0.5 < baseline_latency
    best_dir = (
        _best_direction_for_latency(state, best_ns)
        if beat_baseline
        else "—"
    )
    n_iters = len(state.iteration_results)
    outcome = (
        f"Best configuration **`{best_dir}`** beats baseline latency."
        if beat_baseline
        else "**No** candidate beat baseline latency (best equals or slower than starting point)."
    )
    best_ns_display = f"{best_ns:.0f}" if best_ns_ok else "—"

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Kernel Optimization Log\n\n",
        "## Summary\n\n",
        f"- **Kernel**: `{kernel_name}` on **{gpu_model}** (CC {cc})\n",
        f"- **Baseline latency**: {baseline_latency:.0f} ns — **Best latency**: "
        f"{best_ns_display} ns (**{overall_speedup:.4f}×** vs baseline)\n",
        f"- **Iterations completed**: {n_iters}\n",
        f"- **Outcome**: {outcome}\n\n",
        "## Environment\n\n",
        f"- **Device**: {gpu_model}, {sm_count} SMs, {mem_gb} GB HBM\n",
        f"- **Compute Capability**: {cc}\n",
        f"- **Peak HBM Bandwidth**: {bw_gbs} GB/s\n",
        f"- **CUDA**: {cuda_ver} | **Torch**: {torch_ver} | **Triton**: {triton_ver}\n\n",
    ]

    # ── Baseline ────────────────────────────────────────────────────────────
    lines += [
        "## Baseline Kernel\n\n",
        f"- **Name**: `{kernel_name}`\n",
        f"- **File**: `{state.kernel_path}`\n",
        f"- **Framework**: {state.framework}\n",
        f"- **Launch**: Block ({block},1,1), Grid ({grid},1,1)"
        + (f", Waves/SM: {waves:.2f}" if isinstance(waves, float) else "") + "\n",
        f"- **Registers/thread**: {regs}\n",
        f"- **Duration**: {baseline_latency:.0f} ns\n",
        f"- **SM Throughput**: {_m(baseline_ctx, 'GPU Speed Of Light Throughput', 'Compute (SM) Throughput')}"
        f" | **Memory**: {_m(baseline_ctx, 'GPU Speed Of Light Throughput', 'Memory Throughput')}"
        f" | **DRAM**: {_m_any(baseline_ctx, _LOG_METRIC_ALTERNATIVES['DRAM Throughput (%)'])}\n",
        f"- **Achieved Occupancy**: {_m(baseline_ctx, 'Occupancy', 'Achieved Occupancy')}\n",
        f"- **IPC Active**: {_m_any(baseline_ctx, _LOG_METRIC_ALTERNATIVES['IPC Active'])}"
        f" | **No Eligible**: {_m(baseline_ctx, 'Scheduler Statistics', 'No Eligible')}\n\n",
    ]

    # Cumulative best tracking
    cumulative_best_ns   = baseline_latency
    cumulative_best_desc = "Baseline"
    cumulative_history: list[tuple[int, str, float, float]] = []  # iter, direction, ns, speedup

    # ── Per-iteration ────────────────────────────────────────────────────────
    for result in state.iteration_results:
        iter_num = result.iteration
        mem_count = sum(1 for r in state.iteration_results[:iter_num] for _ in r.candidates)

        lines.append(f"## Iteration {iter_num}\n\n")

        # Results summary table
        lines += [
            f"**Best so far**: {round(baseline_latency / cumulative_best_ns, 4)}×"
            f" | **Memory items**: {len(state.optimization_memory)}\n\n",
            "| Plan | Direction | Duration (ns) | Speedup | Correctness | Kept? |\n",
            "|------|-----------|--------------|---------|-------------|-------|\n",
        ]
        for c in result.candidates:
            dur_str = (
                f"{c.latency_ns:.0f} ns"
                if c.latency_ns < float("inf")
                else "fail"
            )
            spd_str = f"{c.speedup:.3f}×" if c.latency_ns < float("inf") else "fail"
            correct = "✓" if c.correctness_passed else "✗"
            is_best = result.best_candidate and c.plan_letter == result.best_candidate.plan_letter
            kept    = "**✓ best**" if is_best else ""
            lines.append(
                f"| {c.plan_letter} | {c.direction} | {dur_str} | {spd_str} | {correct} | {kept} |\n"
            )
        lines.append("\n")

        # Metric comparison table (only profiled candidates)
        profiled = [c for c in result.candidates if c.profiling_context]
        if profiled:
            metrics_rows = LOG_METRIC_COMPARISON_ROWS
            # Baseline ctx for this iteration (previous best)
            iter_baseline_ctx = baseline_ctx  # simplified — always compare to original

            col_max = 18
            header = "| Metric | Baseline |"
            sep = "|--------|----------|"
            for c in profiled:
                short = (
                    c.direction[: col_max - 1] + "..."
                    if len(c.direction) > col_max
                    else c.direction
                )
                header += f" {short} | Δ |"
                sep += "----------|-----|"
            lines += ["\n### Metric Comparison\n\n", header + "\n", sep + "\n"]
            for label, section, metric in metrics_rows:
                alts = _LOG_METRIC_ALTERNATIVES.get(label)
                base_num = _metric_numeric(
                    iter_baseline_ctx, label, section, metric, alts
                )
                if label == "Duration (ns)":
                    base_val = _duration_ns_display(iter_baseline_ctx)
                    row = f"| {label} | {base_val} |"
                    for c in profiled:
                        cand_val = _duration_ns_display(c.profiling_context)
                        cn = _metric_numeric(
                            c.profiling_context, label, section, metric, alts
                        )
                        row += f" {cand_val} | {_delta_cell(label, base_num, cn)} |"
                    lines.append(row + "\n")
                elif alts:
                    base_val = _m_any(iter_baseline_ctx, alts)
                    row = f"| {label} | {base_val} |"
                    for c in profiled:
                        row += f" {_m_any(c.profiling_context, alts)} |"
                        cn = _metric_numeric(
                            c.profiling_context, label, section, metric, alts
                        )
                        row += f" {_delta_cell(label, base_num, cn)} |"
                    lines.append(row + "\n")
                else:
                    base_val = _m(iter_baseline_ctx, section, metric)
                    row = f"| {label} | {base_val} |"
                    for c in profiled:
                        row += f" {_m(c.profiling_context, section, metric)} |"
                        cn = _metric_numeric(
                            c.profiling_context, label, section, metric, None
                        )
                        row += f" {_delta_cell(label, base_num, cn)} |"
                    lines.append(row + "\n")
            lines.append("\n")

        # Key learnings from optimization_memory
        learnings: list[str] = []
        for c in result.candidates:
            item = mem_by_dir.get(c.direction)
            if item:
                rtype  = item.get("rewrite_type", "")
                speedup = item.get("speedup", 1.0)
                desc   = item.get("strategy_description", "")
                app    = item.get("applicable_when", "")
                avoid  = item.get("do_not_apply_when", "")
                emoji  = "✅" if rtype == "positive" else ("⚠️" if rtype == "implementation_flaw" else "❌")
                entry  = f"#### {emoji} Plan {c.plan_letter}: `{c.direction}` ({speedup:.3f}×)\n\n"
                if desc:  entry += f"{desc}\n\n"
                if app:   entry += f"> **Applicable when**: {app}\n\n"
                if avoid: entry += f"> **Avoid when**: {avoid}\n\n"
                learnings.append(entry)
            elif not c.correctness_passed:
                learnings.append(
                    f"#### ✗ Plan {c.plan_letter}: `{c.direction}` — correctness failed\n\n"
                    f"```\n{c.correctness_message[:300]}\n```\n\n"
                )

        if learnings:
            lines += ["\n### Key Learnings\n\n"] + learnings

        # Track cumulative best
        if result.best_candidate and result.best_candidate.latency_ns < cumulative_best_ns:
            cumulative_best_ns   = result.best_candidate.latency_ns
            cumulative_best_desc = result.best_candidate.direction
            cumulative_history.append((
                iter_num,
                cumulative_best_desc,
                cumulative_best_ns,
                round(baseline_latency / cumulative_best_ns, 4),
            ))

        lines.append("---\n\n")

    # ── Cumulative progress ──────────────────────────────────────────────────
    lines += [
        "## Cumulative Progress\n\n",
        f"```\n",
        f"Baseline:  {baseline_latency:.0f} ns  (1.000×)\n",
    ]
    for iter_num, direction, ns, speedup in cumulative_history:
        lines.append(f"Iter {iter_num:<3}   {ns:.0f} ns  ({speedup:.3f}×)  — {direction}\n")
    lines += [
        f"\nFinal best: {best_ns_display} ns  ({overall_speedup:.4f}×)\n",
        "```\n\n",
    ]

    lines += _kernel_diff_log_lines(state, ext)

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = state.config
    lines += [
        "## Config\n\n",
        f"- T={cfg.T}, N={cfg.N}, K={cfg.K}, TopK={cfg.TopK}, ExpN={cfg.ExpN}\n",
        f"- t_pos={cfg.t_pos}, t_neg={cfg.t_neg}\n",
        f"- replay_mode={cfg.replay_mode}\n",
        f"- kernel_args: `{' '.join(cfg.kernel_args) or '—'}`\n",
    ]

    log_path.write_text("".join(lines))
    pipe_print(f"\n  Log written to [cyan]{log_path}[/]")
