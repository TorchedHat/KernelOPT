"""
Pipeline orchestrator: multi-plan-per-iteration optimization loop.

Flow per iteration (6 phases, matching orchestrator skill):
  Phase 1: Profile current kernel (or use existing CSV)
  Phase 2: Generate N optimization plans (fan-out, diverse)
  Phase 3: Execute each plan K times
  Phase 4: Correctness gate → Profile valid candidates with NCU → CSV + .ncu-rep
  Phase 5: Summarize top-K pairs → update memory
  Phase 6: Select best candidate → next iteration or stop
"""

from __future__ import annotations

import string
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kernelopt.agents.executor import run_executor_agent
from kernelopt.agents.planner import run_planner  # Legacy API - graph.py is deprecated
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
    IterationResult,
    PipelineConfig,
    PipelineState,
    PlanCandidate,
)

console = Console()
PLAN_LETTERS = list(string.ascii_uppercase)


def run_pipeline(state: PipelineState) -> PipelineState:
    """Execute the full optimization pipeline."""
    cfg = state.config
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(state.kernel_path).suffix

    # --- Baseline profiling ---
    if state.baseline_csv and Path(state.baseline_csv).exists():
        console.print("[dim]Using existing baseline CSV[/]")
        csv_path = state.baseline_csv
        ncu_rep = ""
    elif cfg.baseline_csv and Path(cfg.baseline_csv).exists():
        console.print("[dim]Using provided baseline CSV[/]")
        csv_path = cfg.baseline_csv
        ncu_rep = ""
    else:
        console.print("[bold]Profiling baseline kernel...[/]")
        baseline_base = str(output_dir / "baseline")
        csv_path, ncu_rep, ctx_json = run_ncu(
            state.kernel_path,
            baseline_base,
            kernel_name=cfg.kernel_name,
            launch_skip=cfg.launch_skip,
            launch_count=cfg.launch_count,
            ncu_extra_args=cfg.ncu_extra_args or None,
            kernel_args=cfg.kernel_args or None,
            kernel_id=state.kernel_id,
            save_profiling_context=True,
        )

    sections, rules, launch = parse_ncu_csv(csv_path, state.kernel_id)
    baseline_ctx = build_profiling_context(
        sections, rules, launch, state.kernel_path,
        kernel_id=state.kernel_id,
    )

    state.baseline_profiling_context = baseline_ctx
    state.baseline_csv = csv_path
    state.baseline_ncu_rep = ncu_rep
    state.current_profiling_context = baseline_ctx
    state.best_latency_ns = get_latency_ns(baseline_ctx)
    baseline_latency = state.best_latency_ns

    # Detect framework
    if state.framework == "auto":
        state.framework = baseline_ctx["framework_hint"]

    console.print(
        f"  Baseline: [cyan]{state.best_latency_ns:.0f} ns[/]  "
        f"Framework: [green]{state.framework}[/]"
    )

    # --- Iteration loop ---
    for iteration in range(cfg.T):
        state.iteration = iteration
        best_speedup = round(baseline_latency / state.best_latency_ns, 2)
        console.print(
            f"\n[bold]Iteration {iteration + 1}/{cfg.T}[/]  "
            f"Best: [green]{best_speedup}x[/]  "
            f"Memory: {len(state.optimization_memory)} items"
        )

        iter_result = _run_iteration(state, cfg, output_dir, ext)
        state.iteration_results.append(iter_result)

        if iter_result.improved:
            state.no_improvement_streak = 0
        else:
            state.no_improvement_streak += 1

        # Print iteration summary table
        _print_iteration_table(
            iter_result, iteration + 1, cfg.T, state, baseline_latency
        )

        # Check stopping conditions
        if state.no_improvement_streak >= 2:
            console.print(
                "[yellow]Stopping:[/] 2 consecutive iterations "
                "with no improvement."
            )
            break

        # Check if all plans had low confidence
        if iter_result.candidates:
            all_low = all(
                c.optimization_plan.get("confidence") == "low"
                for c in iter_result.candidates
            )
            if all_low:
                console.print(
                    "[yellow]Stopping:[/] All plans report low confidence "
                    "— NCU rule opportunities exhausted."
                )
                break

    # --- Write final outputs ---
    _write_final_outputs(state, output_dir, ext, baseline_latency)

    return state


def _run_iteration(
    state: PipelineState,
    cfg: PipelineConfig,
    output_dir: Path,
    ext: str,
) -> IterationResult:
    """Run one iteration: profile → N plans × K attempts → profile → summarize → select."""
    iteration = state.iteration
    candidates: list[PlanCandidate] = []
    current_kernel = state.best_kernel

    # Phase 1: Profile current kernel (re-profile at each iteration)
    # For iteration 0, baseline is already profiled.
    # For iteration > 0, use the profiling context from the best kernel.
    # (The best kernel was profiled when it became a candidate.)

    # Phase 2: Generate N plans with diversity
    console.print(f"  Generating {cfg.N} optimization plans...")
    plans = []
    previous_directions: list[str] = []

    for i in range(cfg.N):
        try:
            plan = run_planner(
                profiling_context=state.current_profiling_context,
                kernel_source=current_kernel,
                optimization_memory=state.optimization_memory,
                api_key=cfg.api_key,
                model=cfg.planner_model or cfg.model,
                project_id=cfg.project_id,
                region=cfg.region,
                plan_index=i,
                total_plans=cfg.N,
                previous_directions=previous_directions,
            )
            direction = plan.get("direction", "unknown")
            confidence = plan.get("confidence", "?")
            plans.append((PLAN_LETTERS[i], plan))
            previous_directions.append(direction)
            console.print(
                f"    Plan {PLAN_LETTERS[i]}: {direction} "
                f"[dim](confidence: {confidence})[/]"
            )
        except Exception as e:
            console.print(f"    [red]Plan {PLAN_LETTERS[i]} failed: {e}[/]")

    # Phase 3: Execute each plan K times + correctness gate
    for plan_letter, plan in plans:
        direction = plan.get("direction", "unknown")
        safe_dir = direction.replace(" ", "_")[:40]
        file_stem = (
            f"{state.kernel_basename}_iter{iteration + 1}"
            f"_plan{plan_letter}_{safe_dir}"
        )

        for attempt in range(1, cfg.K + 1):
            try:
                optimized_source, execution_result = run_executor_agent(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    framework=state.framework,
                    api_key=cfg.api_key,
                    model=cfg.executor_model or cfg.model,
                    project_id=cfg.project_id,
                    region=cfg.region,
                    attempt=attempt,
                )
            except Exception as e:
                console.print(
                    f"    [red]Executor failed "
                    f"(plan {plan_letter}, attempt {attempt}): {e}[/]"
                )
                continue

            # Skip if executor reported failure
            if not execution_result.get("ready_for_profiling", False):
                console.print(
                    f"    [red]Plan {plan_letter}: "
                    f"{execution_result.get('status', 'failed')}[/]"
                )
                continue

            # Save optimized kernel to flat output dir
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)

            # Correctness check
            passed, msg = check_correctness(
                state.kernel_path, optimized_source, state.framework,
                use_agentic=cfg.agentic_correctness,
                api_key=cfg.api_key,
                project_id=cfg.project_id,
                region=cfg.region,
                kernel_args=cfg.kernel_args,
            )

            # Update execution_result with correctness outcome
            execution_result["correctness"] = "passed" if passed else "failed"

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
            console.print(
                f"    Plan {plan_letter}: correctness {status}"
            )

            if passed:
                candidates.append(candidate)
                break  # one successful attempt per plan

    # Phase 4: Profile valid candidates
    console.print(
        f"  Profiling {len(candidates)} valid candidates..."
    )
    for c in candidates:
        safe_dir = c.direction.replace(" ", "_")[:40]
        file_stem = (
            f"{state.kernel_basename}_iter{iteration + 1}"
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
                kernel_id=state.kernel_id,
                save_profiling_context=True,
            )
            sections, rules, launch = parse_ncu_csv(
                csv_path, state.kernel_id
            )
            ctx = build_profiling_context(
                sections, rules, launch, kernel_file,
                kernel_id=state.kernel_id,
            )
            c.profiling_context = ctx
            c.csv_path = csv_path
            c.ncu_rep_path = ncu_rep_path
            c.latency_ns = get_latency_ns(ctx)
            c.speedup = round(
                state.best_latency_ns / c.latency_ns, 4
            ) if c.latency_ns > 0 else 0.0

            console.print(
                f"    Plan {c.plan_letter} ({c.direction}): "
                f"{c.latency_ns:.0f} ns → {c.speedup:.2f}x"
            )
        except Exception as e:
            console.print(
                f"    [red]Profiling plan {c.plan_letter} failed: {e}[/]"
            )
            c.latency_ns = float("inf")
            c.speedup = 0.0

    # Phase 5: Summarize — sort by |speedup - 1.0| descending, take top-K
    experience_items = []
    memory_updates = []
    summarize_candidates = sorted(
        [c for c in candidates if c.latency_ns < float("inf")],
        key=lambda c: abs(c.speedup - 1.0),
        reverse=True,
    )

    for c in summarize_candidates:
        try:
            item, mem_update = run_summarizer(
                profiling_before=state.current_profiling_context,
                profiling_after=c.profiling_context,
                optimization_plan=c.optimization_plan,
                kernel_before=current_kernel,
                kernel_after=c.optimized_kernel,
                iteration=iteration + 1,
                framework=state.framework,
                speedup=c.speedup,
                existing_memory=state.optimization_memory,
                execution_result=c.execution_result,
                api_key=cfg.api_key,
                model=cfg.summarizer_model or cfg.model,
                project_id=cfg.project_id,
                region=cfg.region,
            )
            experience_items.append(item)
            memory_updates.append(mem_update)
        except Exception as e:
            console.print(
                f"    [red]Summarizer failed "
                f"(plan {c.plan_letter}): {e}[/]"
            )

    # Apply memory updates: select top-K items that meet curation thresholds
    top_items = select_top_k(
        experience_items, cfg.TopK, cfg.t_pos, cfg.t_neg
    )
    if top_items:
        state.optimization_memory = append_items(
            state.optimization_memory, top_items, cfg.ExpN
        )

    # Phase 6: Select best candidate
    iter_result = IterationResult(
        iteration=iteration + 1,
        candidates=candidates,
    )

    if candidates:
        best_c = min(candidates, key=lambda c: c.latency_ns)
        if best_c.latency_ns < state.best_latency_ns:
            state.best_kernel = best_c.optimized_kernel
            state.best_latency_ns = best_c.latency_ns
            state.best_ncu_rep = best_c.ncu_rep_path
            state.current_profiling_context = best_c.profiling_context
            iter_result.best_candidate = best_c
            iter_result.improved = True

    return iter_result


def _print_iteration_table(
    result: IterationResult,
    iter_num: int,
    total: int,
    state: PipelineState,
    baseline_latency: float,
) -> None:
    """Print a Rich table summarizing the iteration."""
    overall_speedup = round(baseline_latency / state.best_latency_ns, 2)
    t = Table(
        title=(
            f"Iteration {iter_num}/{total}  |  "
            f"Framework: {state.framework}  |  "
            f"Best: {overall_speedup}x  |  "
            f"Memory: {len(state.optimization_memory)} items"
        ),
        show_header=True,
    )
    t.add_column("Direction")
    t.add_column("Speedup", justify="right")
    t.add_column("Correctness")
    t.add_column("Kept?")

    for c in result.candidates:
        speedup_str = (
            f"{c.speedup:.2f}x" if c.latency_ns < float("inf")
            else "fail"
        )
        correct = "passed" if c.correctness_passed else "failed"
        is_best = (
            result.best_candidate
            and c.plan_letter == result.best_candidate.plan_letter
        )
        kept = "Y" if is_best else ""
        t.add_row(c.direction, speedup_str, correct, kept)

    console.print(t)


def _write_final_outputs(
    state: PipelineState,
    output_dir: Path,
    ext: str,
    baseline_latency: float,
) -> None:
    """Write best_kernel, optimization_memory.json, and optimization_log.md."""
    # Best kernel
    best_path = output_dir / f"best_kernel{ext}"
    best_path.write_text(state.best_kernel)

    # Memory
    save_memory(
        state.optimization_memory,
        output_dir / "optimization_memory.json",
    )

    # Optimization log
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
            ncu = (
                f"[link]({c.ncu_rep_path})" if c.ncu_rep_path else ""
            )
            lines.append(
                f"| {c.direction} | {speedup_str} | "
                f"{correct} | {kept} | {ncu} |\n"
            )
        lines.append("\n")

    log_path.write_text("".join(lines))
    console.print(f"\n  Log written to [cyan]{log_path}[/]")
