"""
LangGraph-based optimization pipeline.

Graph:
  profile_baseline → generate_plans → execute_plans → profile_candidates
       → summarize_results → select_best → increment_iteration
       → {should_continue: continue → generate_plans | stop → END}
"""

from __future__ import annotations

import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langgraph.graph import END, StateGraph
from rich.console import Console
from rich.table import Table

from kernelopt.agents.executor import run_executor
from kernelopt.agents.planner import run_planner
from kernelopt.agents.profiler import (
    build_profiling_context,
    get_latency_ns,
    run_ncu,
)
from kernelopt.agents.summarizer import run_summarizer
from kernelopt.functions.correctness import check_correctness
from kernelopt.functions.csv_parser import parse_ncu_csv
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


# ---------------------------------------------------------------------------
# Node 1: Profile baseline kernel
# ---------------------------------------------------------------------------

def profile_baseline(state: GraphState) -> dict:
    """Run NCU on the original kernel and establish baseline metrics."""
    cfg = state["config"]
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = state["kernel_path"]
    kernel_id = state.get("kernel_id", 0)

    # Check for existing CSV
    baseline_csv = cfg.baseline_csv
    if baseline_csv and Path(baseline_csv).exists():
        console.print("[dim]Using provided baseline CSV[/]")
        csv_path = baseline_csv
        ncu_rep = ""
    else:
        console.print("[bold]Profiling baseline kernel...[/]")
        baseline_base = str(output_dir / "baseline")
        csv_path, ncu_rep, ctx_json = run_ncu(
            kernel_path,
            baseline_base,
            kernel_name=cfg.kernel_name,
            launch_skip=cfg.launch_skip,
            launch_count=cfg.launch_count,
            ncu_extra_args=cfg.ncu_extra_args or None,
            kernel_args=cfg.kernel_args or None,
            kernel_id=kernel_id,
            save_profiling_context=True,
        )

    sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
    baseline_ctx = build_profiling_context(
        sections, rules, launch, kernel_path, kernel_id=kernel_id,
    )
    latency = get_latency_ns(baseline_ctx)

    framework = state["framework"]
    if framework == "auto":
        framework = baseline_ctx["framework_hint"]

    console.print(
        f"  Baseline: [cyan]{latency:.0f} ns[/]  "
        f"Framework: [green]{framework}[/]"
    )
    console.print(f"  [dim]Profiling context: {ctx_json}[/]")

    return {
        "baseline_profiling_context": baseline_ctx,
        "baseline_csv": csv_path,
        "baseline_ncu_rep": ncu_rep,
        "baseline_latency_ns": latency,
        "best_latency_ns": latency,
        "current_profiling_context": baseline_ctx,
        "framework": framework,
    }


# ---------------------------------------------------------------------------
# Node 2: Generate N optimization plans (concurrent for TorchServe batching)
# ---------------------------------------------------------------------------

def generate_plans(state: GraphState) -> dict:
    """Generate N diverse optimization plans, concurrently for batch inference."""
    cfg = state["config"]
    N = cfg.N
    current_kernel = state["best_kernel"]
    profiling_ctx = state["current_profiling_context"]
    memory = state.get("optimization_memory", [])
    iteration = state.get("iteration", 0)
    baseline = state.get("baseline_latency_ns", float("inf"))
    best_speedup = round(baseline / state["best_latency_ns"], 2)

    console.print(
        f"\n[bold]Iteration {iteration + 1}/{cfg.T}[/]  "
        f"Best: [green]{best_speedup}x[/]  "
        f"Memory: {len(memory)} items"
    )
    console.print(f"  Generating {N} optimization plans...")

    plans: list[tuple[str, dict]] = []

    def _generate(i: int) -> tuple[int, dict]:
        return i, run_planner(
            profiling_context=profiling_ctx,
            kernel_source=current_kernel,
            optimization_memory=memory,
            api_key=cfg.api_key,
            model=cfg.planner_model or cfg.model,
            project_id=cfg.project_id,
            region=cfg.region,
            plan_index=i,
            total_plans=N,
            previous_directions=[],
        )

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = {pool.submit(_generate, i): i for i in range(N)}
        results: dict[int, dict] = {}
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, plan = future.result()
                results[i] = plan
                direction = plan.get("direction", "unknown")
                confidence = plan.get("confidence", "?")
                console.print(
                    f"    Plan {PLAN_LETTERS[i]}: {direction} "
                    f"[dim](confidence: {confidence})[/]"
                )
            except Exception as e:
                console.print(f"    [red]Plan {PLAN_LETTERS[i]} failed: {e}[/]")

    # Collect in order
    for i in range(N):
        if i in results:
            plans.append((PLAN_LETTERS[i], results[i]))

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

    candidates: list[PlanCandidate] = []

    for plan_letter, plan in plans:
        direction = plan.get("direction", "unknown")
        safe_dir = direction.replace(" ", "_")[:40]
        file_stem = f"{kernel_basename}_iter{iteration + 1}_plan{plan_letter}_{safe_dir}"

        for attempt in range(1, cfg.K + 1):
            console.print(f"  [dim]Executing plan {plan_letter}, attempt {attempt}...[/]")
            try:
                optimized_source, execution_result = run_executor(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    framework=framework,
                    api_key=cfg.api_key,
                    model=cfg.executor_model or cfg.model,
                    project_id=cfg.project_id,
                    region=cfg.region,
                    attempt=attempt,
                )

                # DEBUG: Log execution result
                console.print(f"  [dim]Execution status: {execution_result.get('status')}[/]")
                console.print(f"  [dim]Ready for profiling: {execution_result.get('ready_for_profiling')}[/]")
                console.print(f"  [dim]Lines changed: {execution_result.get('lines_changed')}[/]")
                if "debug_info" in execution_result:
                    console.print(f"  [dim]Debug info: {execution_result['debug_info']}[/]")

            except Exception as e:
                console.print(
                    f"    [red]Executor failed "
                    f"(plan {plan_letter}, attempt {attempt}): {e}[/]"
                )
                import traceback
                console.print(f"  [dim]{traceback.format_exc()[:500]}[/]")
                continue

            if not execution_result.get("ready_for_profiling", False):
                console.print(
                    f"    [red]Plan {plan_letter}: "
                    f"{execution_result.get('status', 'failed')}[/]"
                )
                if "reason" in execution_result:
                    console.print(f"    [dim]Reason: {execution_result['reason']}[/]")
                continue

            # Save kernel file
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)
            console.print(f"  [dim]Saved kernel to: {kernel_file}[/]")

            # Correctness check
            console.print(f"  [dim]Running correctness check...[/]")
            passed, msg = check_correctness(
                kernel_path, optimized_source, framework,
                use_agentic=cfg.agentic_correctness,
                api_key=cfg.api_key,
                project_id=cfg.project_id,
                region=cfg.region,
                kernel_args=cfg.kernel_args,
            )
            execution_result["correctness"] = "passed" if passed else "failed"

            # DEBUG: Log correctness result details
            console.print(f"  [dim]Correctness message: {msg[:200]}[/]")

            candidate = PlanCandidate(
                plan_id=plan.get("plan_id", ""),
                plan_letter=plan_letter,
                direction=direction,
                optimization_plan=plan,
                optimized_kernel=optimized_source,
                execution_result=execution_result,
                correctness_passed=passed,
                correctness_message=msg,
            )

            status = "[green]passed[/]" if passed else "[red]failed[/]"
            console.print(f"    Plan {plan_letter}: correctness {status}")
            if not passed:
                console.print(f"    [dim]Failure details: {msg[:300]}[/]")

            if passed:
                candidates.append(candidate)
                break

    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node 4: Profile valid candidates with NCU
# ---------------------------------------------------------------------------

def profile_candidates(state: GraphState) -> dict:
    """Run NCU on each candidate that passed correctness."""
    cfg = state["config"]
    candidates = list(state["candidates"])
    output_dir = Path(state["output_dir"])
    ext = state["ext"]
    iteration = state.get("iteration", 0)
    kernel_basename = state["kernel_basename"]
    kernel_id = state.get("kernel_id", 0)
    best_latency = state["best_latency_ns"]

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

        try:
            csv_path, ncu_rep_path, ctx_json_path = run_ncu(
                kernel_file,
                output_base,
                kernel_name=cfg.kernel_name,
                launch_skip=cfg.launch_skip,
                launch_count=cfg.launch_count,
                ncu_extra_args=cfg.ncu_extra_args or None,
                kernel_args=cfg.kernel_args or None,
                kernel_id=kernel_id,
                save_profiling_context=True,
            )
            sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
            ctx = build_profiling_context(
                sections, rules, launch, kernel_file, kernel_id=kernel_id,
            )
            c.profiling_context = ctx
            c.csv_path = csv_path
            c.ncu_rep_path = ncu_rep_path
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

    def _summarize(c: PlanCandidate) -> tuple[dict, dict]:
        return run_summarizer(
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

    with ThreadPoolExecutor(max_workers=len(summarize_candidates) or 1) as pool:
        futures = {
            pool.submit(_summarize, c): c for c in summarize_candidates
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                item, _ = future.result()
                experience_items.append(item)
            except Exception as e:
                console.print(
                    f"    [red]Summarizer failed "
                    f"(plan {c.plan_letter}): {e}[/]"
                )

    # Apply memory curation
    top_items = select_top_k(
        experience_items, cfg.TopK, cfg.t_pos, cfg.t_neg
    )
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

    # Write final outputs
    output_dir = Path(cfg.output_dir)
    baseline_latency = final.get("baseline_latency_ns", state.best_latency_ns)
    _write_final_outputs(state, output_dir, ext, baseline_latency)

    return state


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

    overall_speedup = round(baseline_latency / state.best_latency_ns, 2)
    log_path = output_dir / "optimization_log.md"
    lines = [
        "# Optimization Log\n\n",
        f"- **Kernel**: `{state.kernel_path}`\n",
        f"- **Framework**: {state.framework}\n",
        f"- **Baseline latency**: {baseline_latency:.0f} ns\n",
        f"- **Best latency**: {state.best_latency_ns:.0f} ns\n",
        f"- **Overall speedup**: {overall_speedup}x\n",
        f"- **Iterations**: {len(state.iteration_results)}\n",
        f"- **Config**: T={state.config.T}, N={state.config.N}, "
        f"K={state.config.K}, TopK={state.config.TopK}\n\n",
    ]

    for result in state.iteration_results:
        lines.append(f"## Iteration {result.iteration}\n\n")
        lines.append(
            "| Direction | Speedup | Correctness | Kept | ncu-rep |\n"
        )
        lines.append(
            "|-----------|---------|-------------|------|---------|\n"
        )
        for c in result.candidates:
            speedup_str = (
                f"{c.speedup:.2f}x"
                if c.latency_ns < float("inf") else "fail"
            )
            correct = "passed" if c.correctness_passed else "failed"
            is_best = (
                result.best_candidate
                and c.plan_letter == result.best_candidate.plan_letter
            )
            kept = "Y" if is_best else ""
            ncu = f"[link]({c.ncu_rep_path})" if c.ncu_rep_path else ""
            lines.append(
                f"| {c.direction} | {speedup_str} | "
                f"{correct} | {kept} | {ncu} |\n"
            )
        lines.append("\n")

    log_path.write_text("".join(lines))
    console.print(f"\n  Log written to [cyan]{log_path}[/]")
