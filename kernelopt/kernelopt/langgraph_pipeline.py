"""
LangGraph-based optimization pipeline.

Graph:
  profile_baseline → generate_plans → execute_plans → profile_candidates
       → summarize_results → select_best → increment_iteration
       → {should_continue: continue → generate_plans | stop → END}
"""

from __future__ import annotations

import string
import uuid
from pathlib import Path

from langgraph.graph import END, StateGraph
from rich.console import Console
from rich.table import Table

from kernelopt.agents.executor import run_executor_agent
from kernelopt.agents.summarizer import run_summarizer
from kernelopt.functions.compiler import compile_cuda
from kernelopt.functions.correctness import check_correctness
from kernelopt.memory import append_items, save_memory, select_top_k
from kernelopt.state import (
    GraphState,
    IterationResult,
    PipelineConfig,
    PipelineState,
    PlanCandidate,
)

console = Console()
PLAN_LETTERS = list(string.ascii_uppercase)


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

    # For .cu kernels: compile the baseline binary once so NCU can run it.
    ext = Path(kernel_path).suffix
    if ext == ".cu":
        binary_path, err = compile_cuda(kernel_path, output_dir=str(output_dir))
        if err:
            raise RuntimeError(
                f"Baseline .cu compilation failed:\n{err}\n\n"
                f"Fix the kernel source and rerun."
            )
        console.print(f"  [dim]Compiled baseline binary: {binary_path}[/]")

    llm_config = {
        "model": cfg.profiler_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    if cfg.baseline_csv and Path(cfg.baseline_csv).exists():
        console.print("[dim]Using provided baseline CSV[/]")
        sections, rules, launch = parse_ncu_csv(cfg.baseline_csv, kernel_id)
        baseline_ctx = build_profiling_context(
            sections, rules, launch, kernel_path, kernel_id=kernel_id
        )
        ncu_config = None
    else:
        console.print("[bold]Profiling baseline kernel...[/]")
        try:
            baseline_ctx, ncu_config = run_profiler_agent(
                kernel_path,
                str(output_dir / "baseline"),
                ncu_config=None,
                pipeline_config=cfg,
                kernel_id=kernel_id,
                llm_config=llm_config,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Baseline profiling failed: {e}") from e
        ctx_json = str(output_dir / "baseline_profiling_context.json")
        console.print(f"  [dim]Profiling context: {ctx_json}[/]")

    latency = get_latency_ns(baseline_ctx)
    framework = state["framework"]
    if framework == "auto":
        framework = baseline_ctx["framework_hint"]

    console.print(
        f"  Baseline: [cyan]{latency:.0f} ns[/]  "
        f"Framework: [green]{framework}[/]"
    )

    return {
        "baseline_profiling_context": baseline_ctx,
        "baseline_csv": cfg.baseline_csv or str(output_dir / "baseline.csv"),
        "baseline_ncu_rep": str(output_dir / "baseline.ncu-rep"),
        "baseline_latency_ns": latency,
        "best_latency_ns": latency,
        "current_profiling_context": baseline_ctx,
        "framework": framework,
        "ncu_config": ncu_config,
    }


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

    console.print(
        f"\n[bold]Iteration {iteration + 1}/{cfg.T}[/]  "
        f"Best: [green]{best_speedup}x[/]  "
        f"Memory: {len(state.get('optimization_memory', []))} items"
    )
    console.print(f"  Generating {cfg.N} optimization plans...")

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

    plans = run_planner_agent(
        profiling_context=profiling_ctx,
        kernel_source=state["best_kernel"],
        memory=state.get("optimization_memory", []),
        N=cfg.N,
        llm_config=llm_config,
        all_tried_directions=all_tried_directions,
    )

    for letter, plan in plans:
        direction = plan.get("direction", "unknown")
        confidence = plan.get("confidence", "?")
        console.print(
            f"    Plan {letter}: {direction} [dim](confidence: {confidence})[/]"
        )

    return {"plans": plans}


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

        console.print(f"  [dim]Executing plan {plan_letter} ({direction})...[/]")

        # For .cu kernels: up to K outer attempts (executor + compile).
        # Each attempt feeds the previous nvcc error back to the executor.
        # For non-.cu kernels: single executor call (existing behaviour).
        last_compile_err: str | None = None
        optimized_source: str | None = None
        execution_result: dict = {}
        compiled_binary: str | None = None
        outer_attempts = cfg.K if ext == ".cu" else 1
        executor_K = 1 if ext == ".cu" else cfg.K

        for attempt_idx in range(outer_attempts):
            try:
                optimized_source, execution_result = run_executor_agent(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    memory=state.get("optimization_memory", []),
                    K=executor_K,
                    llm_config=llm_config,
                    compile_error=last_compile_err,
                )
            except Exception as e:
                console.print(f"    [red]Executor failed (plan {plan_letter}): {e}[/]")
                optimized_source = None
                break

            console.print(f"  [dim]Execution status: {execution_result.get('status')}[/]")
            console.print(f"  [dim]Ready for profiling: {execution_result.get('ready_for_profiling')}[/]")
            console.print(f"  [dim]Lines changed: {execution_result.get('lines_changed')}[/]")

            if not execution_result.get("ready_for_profiling", False):
                reason = execution_result.get("reason", execution_result.get("status", "failed"))
                console.print(f"    [red]Plan {plan_letter}: {reason}[/]")
                optimized_source = None
                break

            # Save source to disk
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)

            if ext == ".cu":
                binary_path, err = compile_cuda(kernel_file, output_dir=str(output_dir))
                if err:
                    last_compile_err = err
                    console.print(
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

        # Correctness check
        console.print(f"  [dim]Running correctness check...[/]")
        passed, msg = check_correctness(
            kernel_path, optimized_source, framework,
            use_agentic=cfg.agentic_correctness,
            api_key=cfg.api_key,
            project_id=cfg.project_id,
            region=cfg.region,
            kernel_args=cfg.kernel_args,
            binary_path=compiled_binary,
        )
        execution_result["correctness"] = "passed" if passed else "failed"

        status = "[green]passed[/]" if passed else "[red]failed[/]"
        console.print(f"    Plan {plan_letter}: correctness {status}")

        if not passed:
            console.print(f"    [dim]Failure: {msg[:300]}[/]")
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

    return {"candidates": candidates, "correctness_failure_items": correctness_failure_items}


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

    console.print(f"  Profiling {len(candidates)} valid candidates...")

    for c in candidates:
        safe_dir = c.direction.replace(" ", "_")[:40]
        file_stem = (
            f"{kernel_basename}_iter{iteration + 1}"
            f"_plan{c.plan_letter}_{safe_dir}"
        )
        output_base = str(output_dir / file_stem)
        kernel_file = f"{output_base}{ext}"
        Path(kernel_file).write_text(c.optimized_kernel)

        if ext == ".cu":
            try:
                _compile_cuda(kernel_file, output_dir=str(output_dir))
            except RuntimeError as e:
                console.print(f"    [red]Compile failed (plan {c.plan_letter}): {e}[/]")
                c.latency_ns = float("inf")
                c.speedup = 0.0
                continue

        try:
            ctx, _ = run_profiler_agent(
                kernel_file,
                output_base,
                ncu_config=saved_ncu_config,  # full bypass — no LLM call
                pipeline_config=cfg,
                kernel_id=kernel_id,
                llm_config=llm_config,
            )
            c.profiling_context = ctx
            c.csv_path = f"{output_base}.csv"
            c.ncu_rep_path = f"{output_base}.ncu-rep"
            c.latency_ns = get_latency_ns(ctx)
            c.speedup = (
                round(best_latency / c.latency_ns, 4)
                if c.latency_ns > 0 else 0.0
            )
            console.print(
                f"    Plan {c.plan_letter} ({c.direction}): "
                f"{c.latency_ns:.0f} ns -> {c.speedup:.2f}x"
            )
        except Exception as e:
            console.print(
                f"    [red]Profiling plan {c.plan_letter} failed: {e}[/]"
            )
            c.latency_ns = float("inf")
            c.speedup = 0.0

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

    summarize_candidates = sorted(
        [c for c in candidates if c.latency_ns < float("inf")],
        key=lambda c: abs(c.speedup - 1.0),
        reverse=True,
    )

    experience_items: list[dict] = []

    # Sequential execution with updated memory for proper deduplication
    for c in summarize_candidates:
        try:
            item, memory_update = run_summarizer(
                profiling_before=profiling_before,
                profiling_after=c.profiling_context,
                optimization_plan=c.optimization_plan,
                kernel_before=current_kernel,
                kernel_after=c.optimized_kernel,
                iteration=iteration + 1,
                framework=framework,
                speedup=c.speedup,
                existing_memory=memory,  # Current state (updated each iteration)
                execution_result=c.execution_result,
                api_key=cfg.api_key,
                model=cfg.summarizer_model or cfg.model,
                project_id=cfg.project_id,
                region=cfg.region,
            )

            # Apply deduplication immediately
            if memory_update["action"] == "append":
                experience_items.append(item)
            elif memory_update["action"] == "replace":
                replace_id = memory_update["replace_item_id"]
                memory = [m for m in memory if m.get("item_id") != replace_id]
                experience_items.append(item)
            # else: skip (don't add)

        except Exception as e:
            console.print(
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

    return {"optimization_memory": memory}


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

    # Print iteration table
    overall_speedup = round(baseline_latency / best_latency, 2)
    cfg = state["config"]
    t = Table(
        title=(
            f"Iteration {iteration + 1}/{cfg.T}  |  "
            f"Framework: {framework}  |  "
            f"Best: {overall_speedup}x  |  "
            f"Memory: {len(memory)} items"
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
    console.print(t)

    return {
        "best_kernel": best_kernel,
        "best_latency_ns": best_latency,
        "best_ncu_rep": best_ncu_rep,
        "current_profiling_context": current_ctx,
        "no_improvement_streak": no_improvement_streak,
        "iteration_results": iteration_results,
    }


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
        console.print(
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
            console.print(
                "[yellow]Stopping:[/] All plans report low confidence "
                "-- NCU rule opportunities exhausted."
            )
            return "stop"

    return "continue"


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

    graph.set_entry_point("profile_baseline")
    graph.add_edge("profile_baseline", "generate_plans")
    graph.add_edge("generate_plans", "execute_plans")
    graph.add_edge("execute_plans", "profile_candidates")
    graph.add_edge("profile_candidates", "summarize_results")
    graph.add_edge("summarize_results", "select_best")
    graph.add_edge("select_best", "increment_iteration")

    graph.add_conditional_edges(
        "increment_iteration",
        should_continue,
        {"continue": "generate_plans", "stop": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point: convert PipelineState ↔ GraphState
# ---------------------------------------------------------------------------

def run_langgraph_pipeline(state: PipelineState) -> PipelineState:
    """Run the LangGraph optimization pipeline."""
    cfg = state.config
    ext = Path(state.kernel_path).suffix

    graph = build_graph()

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
    }

    # Provide baseline CSV if available
    if state.baseline_csv:
        initial["baseline_csv"] = state.baseline_csv

    final = graph.invoke(initial)

    # Map back to PipelineState
    state.best_kernel = final["best_kernel"]
    state.best_latency_ns = final["best_latency_ns"]
    state.best_ncu_rep = final.get("best_ncu_rep", "")
    state.baseline_profiling_context = final.get("baseline_profiling_context", {})
    state.current_profiling_context = final.get("current_profiling_context", {})
    state.optimization_memory = final.get("optimization_memory", [])
    state.iteration_results = final.get("iteration_results", [])
    state.framework = final.get("framework", state.framework)
    state.ncu_config = final.get("ncu_config")

    # Write final outputs
    output_dir = Path(cfg.output_dir)
    baseline_latency = final.get("baseline_latency_ns", state.best_latency_ns)
    _write_final_outputs(state, output_dir, ext, baseline_latency)

    return state


def _m(ctx: dict, section: str, metric: str, default: str = "—") -> str:
    """Safely extract a metric value from a profiling context."""
    try:
        entry = ctx["sections"][section][metric]
        val = entry.get("value", default)
        unit = entry.get("unit", "").strip()
        if val is None or val != val:   # NaN check
            return default
        if isinstance(val, float):
            return f"{val:.2f} {unit}".strip() if unit else f"{val:.2f}"
        return f"{val} {unit}".strip() if unit else str(val)
    except (KeyError, TypeError):
        return default


def _write_final_outputs(
    state: PipelineState,
    output_dir: Path,
    ext: str,
    baseline_latency: float,
) -> None:
    """Write best_kernel, optimization_memory.json, and optimization_log.md."""
    best_path = output_dir / f"best_kernel{ext}"
    best_path.write_text(state.best_kernel)

    save_memory(
        state.optimization_memory,
        output_dir / "optimization_memory.json",
    )

    overall_speedup = round(baseline_latency / state.best_latency_ns, 4)
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

    # Build a memory lookup: direction → memory item (latest / highest speedup)
    mem_by_dir: dict[str, dict] = {}
    for item in state.optimization_memory:
        d = item.get("direction", "")
        existing = mem_by_dir.get(d)
        if existing is None or item.get("speedup", 0) > existing.get("speedup", 0):
            mem_by_dir[d] = item

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Kernel Optimization Log\n\n",
        "## Environment\n\n",
        f"- **Device**: {gpu_model}, {sm_count} SMs, {mem_gb} GB HBM\n",
        f"- **Compute Capability**: {cc}\n",
        f"- **Peak HBM Bandwidth**: {bw_gbs} GB/s\n",
        f"- **CUDA**: {cuda_ver} | **Torch**: {torch_ver} | **Triton**: {triton_ver}\n\n",
    ]

    # ── Baseline ────────────────────────────────────────────────────────────
    kernel_name = baseline_ctx.get("kernel_name", "—")
    block  = baseline_launch.get("block_size", "—")
    grid   = baseline_launch.get("grid_size", "—")
    waves  = baseline_launch.get("waves_per_sm", "—")
    regs   = baseline_launch.get("registers_per_thread", "—")

    lines += [
        "## Baseline Kernel\n\n",
        f"- **Name**: `{kernel_name}`\n",
        f"- **File**: `{state.kernel_path}`\n",
        f"- **Framework**: {state.framework}\n",
        f"- **Launch**: Block ({block},1,1), Grid ({grid},1,1)"
        + (f", Waves/SM: {waves:.2f}" if isinstance(waves, float) else "") + "\n",
        f"- **Registers/thread**: {regs}\n",
        f"- **Duration**: {baseline_latency/1000:.2f} µs ({baseline_latency:.0f} ns)\n",
        f"- **SM Throughput**: {_m(baseline_ctx, 'GPU Speed Of Light Throughput', 'Compute (SM) Throughput')}"
        f" | **Memory**: {_m(baseline_ctx, 'GPU Speed Of Light Throughput', 'Memory Throughput')}"
        f" | **DRAM**: {_m(baseline_ctx, 'Memory Workload Analysis', 'DRAM Throughput')}\n",
        f"- **Achieved Occupancy**: {_m(baseline_ctx, 'Occupancy', 'Achieved Occupancy')}\n",
        f"- **IPC Active**: {_m(baseline_ctx, 'Instruction Statistics', 'IPC Active')}"
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
            dur_str = f"{c.latency_ns:.0f}" if c.latency_ns < float("inf") else "fail"
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
            metrics_rows = [
                ("Duration (ns)", "GPU Speed Of Light Throughput", "Duration"),
                ("SM Throughput (%)", "GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
                ("Memory Throughput (%)", "GPU Speed Of Light Throughput", "Memory Throughput"),
                ("DRAM Throughput (%)", "Memory Workload Analysis", "DRAM Throughput"),
                ("Achieved Occupancy (%)", "Occupancy", "Achieved Occupancy"),
                ("Registers/Thread", "Launch Statistics", "Registers Per Thread"),
                ("Waves/SM", "Launch Statistics", "Waves Per SM"),
                ("No Eligible (%)", "Scheduler Statistics", "No Eligible"),
                ("IPC Active", "Instruction Statistics", "IPC Active"),
            ]
            # Baseline ctx for this iteration (previous best)
            iter_baseline_ctx = baseline_ctx  # simplified — always compare to original

            header = "| Metric | Baseline |" + "".join(f" {c.direction[:22]} |" for c in profiled) + "\n"
            sep    = "|--------|---------|" + "".join("-----|" for _ in profiled) + "\n"
            lines += ["\n### Metric Comparison\n\n", header, sep]
            for label, section, metric in metrics_rows:
                base_val = _m(iter_baseline_ctx, section, metric)
                row = f"| {label} | {base_val} |"
                for c in profiled:
                    row += f" {_m(c.profiling_context, section, metric)} |"
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
        f"\nFinal best: {state.best_latency_ns:.0f} ns  ({overall_speedup:.4f}×)\n",
        "```\n\n",
    ]

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
    console.print(f"\n  Log written to [cyan]{log_path}[/]")
