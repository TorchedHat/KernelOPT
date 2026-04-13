"""CLI entry point for kernelopt. Built with Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

import warnings

# ADC user creds without a quota project — noisy on every Vertex call.
warnings.filterwarnings(
    "ignore",
    message=r".*authenticated using end user credentials.*quota project.*",
    category=UserWarning,
    module=r"google\.auth\._default",
)


app = typer.Typer(
    name="kernelopt",
    help="Multi-agent GPU kernel optimizer. Powered by LangGraph + NCU.",
    add_completion=True,
)
console = Console()


@app.command()
def run(
    kernel_path: Path = typer.Argument(
        ..., help="Kernel file (.cu, .py, .hl)"
    ),
    framework: str = typer.Option(
        "auto", help="cuda|triton|helion|auto"
    ),
    iterations: int = typer.Option(
        5, "-T", "--iterations", help="Max optimization iterations"
    ),
    plans: int = typer.Option(
        4, "-N", "--plans", help="Plans generated per iteration"
    ),
    attempts: int = typer.Option(
        1, "-K", "--attempts", help="Executor attempts per plan"
    ),
    top_k: int = typer.Option(
        4, "--top-k", help="Max experience items per iteration"
    ),
    output_dir: Path = typer.Option(
        Path("pipeline_output"), "-o", "--output-dir", help="Output directory"
    ),
    csv: Optional[Path] = typer.Option(
        None, "--csv", help="Existing baseline NCU CSV (skip baseline profiling)"
    ),
    memory: Optional[Path] = typer.Option(
        None, "--memory", help="Load existing optimization_memory.json"
    ),
    kernel_name: Optional[str] = typer.Option(
        None, "--kernel-name",
        help="Regex for NCU --kernel-name (e.g. 'softmax_triton_forward_kernel')"
    ),
    launch_skip: int = typer.Option(
        0, "--launch-skip", help="NCU --launch-skip value"
    ),
    launch_count: Optional[int] = typer.Option(
        None, "--launch-count", help="NCU --launch-count value"
    ),
    replay_mode: str = typer.Option(
        "kernel", "--replay-mode",
        help="NCU replay mode: 'kernel' (default, lower overhead) or 'application' (lower peak GPU memory, use for large inputs that OOM with kernel mode)"
    ),
    kernel_args: Optional[str] = typer.Option(
        None, "--kernel-args",
        help="Extra args passed to the kernel program (e.g. '--profile-only --m 4096')"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key (not needed for Vertex AI)"
    ),
    model: str = typer.Option(
        "claude-sonnet-4@20250514", help="Default model name (used when per-agent models not set)"
    ),
    project_id: Optional[str] = typer.Option(
        None, "--project-id", envvar="ANTHROPIC_VERTEX_PROJECT_ID",
        help="Google Cloud project ID for Vertex AI"
    ),
    region: str = typer.Option(
        "us-east5", "--region", help="Vertex AI region"
    ),
    planner_model: Optional[str] = typer.Option(
        None, "--planner-model", help="Model for planner agent"
    ),
    summarizer_model: Optional[str] = typer.Option(
        None, "--summarizer-model", help="Model for summarizer agent"
    ),
    executor_model: Optional[str] = typer.Option(
        None, "--executor-model", help="Model for executor agent"
    ),
    profile_hints: str = typer.Option(
        "", "--profile-hints",
        help="Natural language hint for the profiler agent (e.g. 'profile only the forward pass')"
    ),
    profiler_model: Optional[str] = typer.Option(
        None, "--profiler-model", help="Model for profiler agent config-decider"
    ),
    verbose: bool = typer.Option(False, help="Verbose output"),
    agentic_correctness: bool = typer.Option(
        True, "--agentic-correctness/--no-agentic-correctness",
        help="Use Claude to generate CPU reference tests (default: True)"
    ),
):
    """Run the full optimization pipeline.

    Generates N plans per iteration, profiles each valid candidate with NCU,
    and selects the best. Outputs best_kernel, optimization_log.md, and
    .ncu-rep files for every candidate.

    Example:
        kernelopt run softmax.py -T 3 -N 4 --kernel-name 'softmax_kernel' --kernel-args '--profile-only'
    """
    from kernelopt.memory import load_memory
    from kernelopt.state import PipelineConfig, PipelineState

    if not kernel_path.exists():
        console.print(f"[red]Error:[/] kernel file not found: {kernel_path}")
        raise typer.Exit(1)

    kernel_basename = kernel_path.stem

    cfg = PipelineConfig(
        T=iterations,
        N=plans,
        K=attempts,
        TopK=top_k,
        kernel_name=kernel_name,
        launch_skip=launch_skip,
        launch_count=launch_count,
        replay_mode=replay_mode,
        kernel_args=kernel_args.split() if kernel_args else [],
        model=model,
        api_key=api_key,
        project_id=project_id or "",
        region=region,
        planner_model=planner_model,
        summarizer_model=summarizer_model,
        executor_model=executor_model,
        profile_hints=profile_hints,
        profiler_model=profiler_model,
        output_dir=str(output_dir),
        kernel_path=str(kernel_path),
        baseline_csv=str(csv) if csv else None,
        verbose=verbose,
        agentic_correctness=agentic_correctness,
    )

    state = PipelineState(
        config=cfg,
        kernel_path=str(kernel_path),
        kernel_source=kernel_path.read_text(),
        best_kernel=kernel_path.read_text(),
        framework=framework,
        kernel_basename=kernel_basename,
        optimization_memory=load_memory(memory) if memory else [],
    )

    console.print(
        f"[bold green]kernelopt[/] starting on "
        f"[cyan]{kernel_path.name}[/]\n"
        f"  T={iterations} iterations, N={plans} plans/iter, "
        f"K={attempts} attempts/plan, TopK={top_k}\n"
        f"  Backend: [cyan]Claude on Vertex AI[/] ({model})"
    )

    from kernelopt.agents.profiler import get_latency_ns
    from kernelopt.langgraph_pipeline import run_langgraph_pipeline

    final = run_langgraph_pipeline(state)

    baseline_ns = get_latency_ns(final.baseline_profiling_context)
    best_ns = final.best_latency_ns

    def _lat_summary(ns: float) -> str:
        if ns != ns or ns == float("inf"):  # NaN or missing
            return "?"
        return f"{ns / 1000:.2f} µs ({ns:.0f} ns)"

    console.print(
        f"\n[bold]Done.[/] "
        f"Baseline: {_lat_summary(baseline_ns)} -> "
        f"Best: [green]{_lat_summary(best_ns)}[/]\n"
        f"Results in [cyan]{output_dir}/[/]"
    )


@app.command()
def profile(
    kernel_path: Path = typer.Argument(
        ..., help="Kernel file to profile"
    ),
    output: Path = typer.Option(
        Path("pipeline_output/baseline"), "-o",
        help="Output basename (generates _output.csv + _profile.ncu-rep)"
    ),
    kernel_id: int = typer.Option(
        0, "--id", help="Kernel ID in multi-kernel profiles"
    ),
    kernel_name: Optional[str] = typer.Option(
        None, "--kernel-name",
        help="Regex for NCU --kernel-name"
    ),
    launch_skip: int = typer.Option(
        0, "--launch-skip", help="NCU --launch-skip"
    ),
    launch_count: Optional[int] = typer.Option(
        None, "--launch-count", help="NCU --launch-count"
    ),
    kernel_args: Optional[str] = typer.Option(
        None, "--kernel-args",
        help="Extra args for the kernel program (e.g. '--profile-only')"
    ),
):
    """Profile a kernel with NCU.

    Generates both .csv and .ncu-rep files.
    """
    from kernelopt.agents.profiler import (
        build_profiling_context,
        run_ncu,
    )
    from kernelopt.functions.csv_parser import parse_ncu_csv

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold green]kernelopt profile[/] "
        f"[cyan]{kernel_path.name}[/] → {output}"
    )
    csv_path, ncu_rep, ctx_json = run_ncu(
        str(kernel_path),
        str(output),
        kernel_name=kernel_name,
        launch_skip=launch_skip,
        launch_count=launch_count,
        kernel_args=kernel_args.split() if kernel_args else None,
        kernel_id=kernel_id,
        save_profiling_context=True,
    )
    console.print(f"  CSV: [cyan]{csv_path}[/]")
    console.print(f"  ncu-rep: [cyan]{ncu_rep}[/]")
    console.print(f"  profiling context: [cyan]{ctx_json}[/]")

    sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
    ctx = build_profiling_context(
        sections, rules, launch, str(kernel_path)
    )
    _print_profile(ctx)


@app.command()
def show(
    csv_path: Path = typer.Argument(
        ..., help="NCU CSV file to inspect"
    ),
    kernel_id: int = typer.Option(
        0, "--id", help="Kernel ID"
    ),
):
    """Pretty-print a profile CSV (no LLM involved)."""
    from kernelopt.agents.profiler import build_profiling_context
    from kernelopt.functions.csv_parser import parse_ncu_csv

    sections, rules, launch = parse_ncu_csv(str(csv_path), kernel_id)
    ctx = build_profiling_context(sections, rules, launch, "")
    _print_profile(ctx)


@app.command()
def plan(
    csv_path: Path = typer.Argument(..., help="NCU CSV file"),
    kernel_path: Path = typer.Argument(..., help="Kernel source file"),
    num_plans: int = typer.Option(
        1, "-N", help="Number of plans to generate"
    ),
    memory: Optional[Path] = typer.Option(
        None, help="Existing optimization_memory.json"
    ),
    api_key: Optional[str] = typer.Option(
        None, envvar="ANTHROPIC_API_KEY"
    ),
    model: str = typer.Option("claude-sonnet-4@20250514"),
    project_id: Optional[str] = typer.Option(
        None, "--project-id", envvar="ANTHROPIC_VERTEX_PROJECT_ID",
        help="Google Cloud project ID for Vertex AI"
    ),
    region: str = typer.Option(
        "us-east5", "--region", help="Vertex AI region"
    ),
):
    """Generate optimization plan(s) from a profile CSV (no execution)."""
    import json

    from kernelopt.agents.planner import run_planner_agent
    from kernelopt.agents.profiler import build_profiling_context
    from kernelopt.functions.csv_parser import parse_ncu_csv
    from kernelopt.memory import load_memory

    sections, rules, launch = parse_ncu_csv(str(csv_path))
    ctx = build_profiling_context(
        sections, rules, launch, str(kernel_path)
    )
    mem = load_memory(memory) if memory else []

    llm_config = {
        "model": model,
        "api_key": api_key,
        "project_id": project_id,
        "region": region,
    }
    plans = run_planner_agent(
        profiling_context=ctx,
        kernel_source=kernel_path.read_text(),
        memory=mem,
        N=num_plans,
        llm_config=llm_config,
        all_tried_directions=[],
    )
    for letter, result in plans:
        if num_plans > 1:
            console.print(f"\n[bold]Plan {letter}:[/]")
        console.print_json(json.dumps(result))


@app.command()
def kernels(
    csv_path: Path = typer.Argument(..., help="NCU CSV file"),
):
    """List all kernels found in an NCU CSV file."""
    from rich.table import Table

    from kernelopt.functions.csv_parser import list_kernels

    kerns = list_kernels(str(csv_path))
    t = Table(title="Kernels in CSV", show_header=True)
    t.add_column("ID", justify="right")
    t.add_column("Kernel Name")
    for kid, kname in kerns:
        display = kname if len(kname) <= 100 else kname[:97] + "..."
        t.add_row(str(kid), display)
    console.print(t)


def _print_profile(ctx: dict) -> None:
    """Rich-formatted profile summary."""
    from rich.table import Table

    console.print(
        f"\n[bold]Kernel:[/] {ctx['kernel_name']}  "
        f"[dim]({ctx['framework_hint']}  "
        f"CC {ctx['compute_capability']})[/]"
    )

    lc = ctx.get("launch", {})
    if lc:
        console.print(
            f"  Block: {lc.get('block_size', '?')}  "
            f"Grid: {lc.get('grid_size', '?')}  "
            f"Regs: {lc.get('registers_per_thread', '?')}"
        )

    # Speed of Light
    t = Table(title="Speed of Light", show_header=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    sol = ctx["sections"].get("GPU Speed Of Light Throughput", {})
    for k in [
        "Compute (SM) Throughput",
        "Memory Throughput",
        "DRAM Throughput",
        "L1/TEX Cache Throughput",
        "L2 Cache Throughput",
        "Duration",
    ]:
        if k in sol:
            t.add_row(k, f"{sol[k]['value']} {sol[k]['unit']}")
    console.print(t)

    # NCU rules
    rules = ctx.get("rules", [])
    if rules:
        r = Table(title="NCU Optimization Rules", show_header=True)
        r.add_column("Rule")
        r.add_column("Type")
        r.add_column("Speedup", justify="right")
        r.add_column("Description")
        for rule in sorted(
            rules,
            key=lambda x: x.get("speedup_pct") or 0,
            reverse=True,
        ):
            pct = rule.get("speedup_pct")
            stype = rule.get("speedup_type", "")
            speedup_str = f"{pct}% ({stype})" if pct else "-"
            desc = rule.get("description", "")
            if len(desc) > 100:
                desc = desc[:97] + "..."
            r.add_row(rule["name"], rule["type"], speedup_str, desc)
        console.print(r)


if __name__ == "__main__":
    app()
