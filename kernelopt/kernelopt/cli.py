"""CLI entry point for kernelopt. Built with Typer."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Optional

import typer
from click.core import ParameterSource
from rich.console import Console

import warnings

from kernelopt.functions.gpu_monitoring import (
    select_gpu,
    validate_gpu_for_profiling,
)

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


# With --memory + checkpoint, these may not be set on the CLI (config comes from checkpoint).
_RESUME_CKPT_FORBIDDEN = frozenset({
    "framework",
    "top_k",
    "output_dir",
    "run_subdir",
    "csv",
    "kernel_name",
    "launch_skip",
    "launch_count",
    "replay_mode",
    "kernel_args",
    "api_key",
    "model",
    "project_id",
    "region",
    "planner_model",
    "summarizer_model",
    "executor_model",
    "profile_hints",
    "profiler_model",
    "verbose",
    "agentic_correctness",
})


def _cli_set_param(ctx: typer.Context, name: str) -> bool:
    """True if the user supplied this option on the command line (not defaults/env)."""
    try:
        return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
    except (ValueError, AttributeError):
        return False


def _run_output_dir(base: Path, *, run_subdir: bool) -> Path:
    """Place this run under base/run_YYYYmmdd_HHMMSS_microseconds/ when run_subdir is True."""
    if not run_subdir:
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return base / f"run_{stamp}"


@app.command()
def run(
    ctx: typer.Context,
    kernel_path: Path = typer.Argument(
        ..., help="Kernel file (.cu, .py, .hl)"
    ),
    framework: str = typer.Option(
        "auto", help="cuda|triton|helion|auto"
    ),
    iterations: Optional[int] = typer.Option(
        None, "-T", "--iterations",
        help="Max optimization iterations (default 5; with --memory + checkpoint: from checkpoint if omitted)",
    ),
    plans: Optional[int] = typer.Option(
        None, "-N", "--plans",
        help="Plans per iteration (default 4; with --memory + checkpoint: from checkpoint if omitted)",
    ),
    attempts: Optional[int] = typer.Option(
        None, "-K", "--attempts",
        help="Executor attempts per plan (default 1; with --memory + checkpoint: from checkpoint if omitted)",
    ),
    top_k: int = typer.Option(
        4, "--top-k", help="Max experience items per iteration"
    ),
    output_dir: Path = typer.Option(
        Path("pipeline_output"), "-o", "--output-dir",
        help="Output base directory for this kernelopt run; default: new run_<timestamp>/ here",
    ),
    run_subdir: bool = typer.Option(
        True, "--run-subdir/--no-run-subdir",
        help="Every kernelopt run: write under <output-dir>/run_<timestamp>/ (default: on)",
    ),
    csv: Optional[Path] = typer.Option(
        None, "--csv", help="Existing baseline NCU CSV (skip baseline profiling)"
    ),
    memory: Optional[Path] = typer.Option(
        None, "--memory",
        help="Resume from a prior run directory (pipeline_output/run_…/) with checkpoint or baseline artifacts",
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
        "us-east5", "--region", envvar="VERTEX_REGION", help="Vertex AI region"
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
    hud: bool = typer.Option(
        False,
        "--hud",
        help="Same terminal: alternate-screen HUD; NCU + full stream → kernelopt_terminal.log. Press [t] to toggle HUD / log tail (TTY only).",
    ),
    agentic_correctness: bool = typer.Option(
        True, "--agentic-correctness/--no-agentic-correctness",
        help="Use Claude to generate CPU reference tests (default: True)"
    ),
):
    """Run the full optimization pipeline.

    Generates N plans per iteration, profiles each valid candidate with NCU,
    and selects the best. Outputs best_kernel, optimization_log.md, and
    .ncu-rep files for every candidate. Each kernelopt run uses a new
    run_<timestamp>/ folder under --output-dir unless --no-run-subdir.
    With --memory and a pipeline_checkpoint.json, NCU/LLM options are taken from the
    checkpoint; only -T, -N, -K and --hud may be set (omit T/N/K to use saved values).

    Example:
        kernelopt run softmax.py -T 3 -N 4 --kernel-name 'softmax_kernel' --kernel-args '--profile-only'
    """
    from kernelopt.config import load_config as _load_config
    _file_cfg = _load_config().get("vertex", {})
    if not project_id:
        project_id = _file_cfg.get("project_id") or ""
    if (
        ctx.get_parameter_source("region") not in (
            ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT
        )
        and _file_cfg.get("region")
    ):
        region = _file_cfg["region"]

    from kernelopt.checkpoint import load_checkpoint_file, pipeline_config_from_checkpoint_dict
    from kernelopt.state import PipelineConfig, PipelineState

    ckpt: dict | None = None
    if memory is not None:
        run_dir_pre = memory.resolve()
        if not run_dir_pre.is_dir():
            console.print(f"[red]Error:[/] --memory must be an existing directory: {run_dir_pre}")
            raise typer.Exit(1)
        ckpt = load_checkpoint_file(run_dir_pre)
        gs_pre = ckpt.get("graph_state") if ckpt else None
        saved_pre = (gs_pre or {}).get("config") if isinstance(gs_pre, dict) else None
        will_resume_ckpt = (
            ckpt is not None
            and isinstance(saved_pre, dict)
            and bool(saved_pre)
        )
        if will_resume_ckpt:
            bad = sorted(
                p for p in _RESUME_CKPT_FORBIDDEN if _cli_set_param(ctx, p)
            )
            if bad:
                def _flag_label(name: str) -> str:
                    if name == "agentic_correctness":
                        return "--agentic-correctness / --no-agentic-correctness"
                    return "--" + name.replace("_", "-")

                console.print(
                    "[red]Error:[/] With [cyan]--memory[/] and a [cyan]pipeline_checkpoint.json[/], "
                    "config is loaded from the checkpoint.\n"
                    "  Only these options are allowed on the command line:\n"
                    "    • kernel file (positional)\n"
                    "    • [cyan]--memory[/] DIR\n"
                    "    • [cyan]-T[/] / [cyan]-N[/] / [cyan]-K[/] (optional overrides)\n"
                    "    • [cyan]--hud[/] (optional; same-terminal HUD + log file)\n"
                    f"  Remove: {', '.join(_flag_label(p) for p in bad)}"
                )
                raise typer.Exit(2)
        else:
            for p in ("output_dir", "run_subdir"):
                if _cli_set_param(ctx, p):
                    console.print(
                        "[red]Error:[/] With [cyan]--memory[/], output is the run directory; "
                        "do not pass [cyan]-o[/] / [cyan]--output-dir[/] or "
                        "[cyan]--run-subdir[/] / [cyan]--no-run-subdir[/]."
                    )
                    raise typer.Exit(2)

    if not kernel_path.exists():
        console.print(f"[red]Error:[/] kernel file not found: {kernel_path}")
        raise typer.Exit(1)

    kernel_basename = kernel_path.stem
    resume_from_ckpt = False
    run_framework = framework

    def _defaults_tnk() -> tuple[int, int, int]:
        return (
            iterations if iterations is not None else 5,
            plans if plans is not None else 4,
            attempts if attempts is not None else 1,
        )

    if memory is not None:
        run_dir = memory.resolve()
        resolved_output = run_dir
        gs = ckpt.get("graph_state") if ckpt else None
        saved_cfg_dict = (gs or {}).get("config") if isinstance(gs, dict) else None

        if ckpt is not None and isinstance(saved_cfg_dict, dict) and saved_cfg_dict:
            cfg = pipeline_config_from_checkpoint_dict(saved_cfg_dict)
            cfg.output_dir = str(resolved_output)
            cfg.kernel_path = str(kernel_path.resolve())
            if iterations is not None:
                cfg.T = iterations
            if plans is not None:
                cfg.N = plans
            if attempts is not None:
                cfg.K = attempts
            run_framework = gs.get("framework", framework) if isinstance(gs, dict) else framework
            resume_from_ckpt = True
        else:
            t0, n0, k0 = _defaults_tnk()
            cfg = PipelineConfig(
                T=t0,
                N=n0,
                K=k0,
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
                output_dir=str(resolved_output),
                kernel_path=str(kernel_path),
                baseline_csv=str(csv) if csv else None,
                verbose=verbose,
                agentic_correctness=agentic_correctness,
            )
    else:
        resolved_output = _run_output_dir(output_dir, run_subdir=run_subdir)
        t0, n0, k0 = _defaults_tnk()
        cfg = PipelineConfig(
            T=t0,
            N=n0,
            K=k0,
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
            output_dir=str(resolved_output),
            kernel_path=str(kernel_path),
            baseline_csv=str(csv) if csv else None,
            verbose=verbose,
            agentic_correctness=agentic_correctness,
        )

    cfg.hud = hud

    state = PipelineState(
        config=cfg,
        kernel_path=str(kernel_path),
        kernel_source=kernel_path.read_text(),
        best_kernel=kernel_path.read_text(),
        framework=run_framework,
        kernel_basename=kernel_basename,
        optimization_memory=[],
        resume_run_dir=str(resolved_output) if memory is not None else None,
    )

    extra = ""
    if resume_from_ckpt:
        extra = (
            "\n  [dim]Resume:[/] pipeline config from checkpoint "
            "(other CLI flags ignored; only -T/-N/-K override T/N/K; "
            "[cyan]--hud[/] is always from this shell)"
        )
    hud_note = ""
    if cfg.hud:
        hud_note = (
            "\n  [dim]HUD:[/] same-terminal panel; stream + NCU → "
            f"[cyan]{resolved_output / 'kernelopt_terminal.log'}[/] "
            "(press [t] to toggle HUD / log tail when stdin is a TTY)"
        )
    console.print(
        f"[bold green]kernelopt[/] starting on "
        f"[cyan]{kernel_path.name}[/]\n"
        f"  T={cfg.T} iterations, N={cfg.N} plans/iter, "
        f"K={cfg.K} attempts/plan, TopK={cfg.TopK}\n"
        f"  Output: [cyan]{resolved_output}/[/]\n"
        f"  Backend: [cyan]Claude on Vertex AI[/] ({cfg.model})"
        f"{extra}{hud_note}"
    )

    # Pre-run GPU validation (snapshot reused by HUD so it matches System Status)
    gpu_id = select_gpu()
    pre_snap = validate_gpu_for_profiling(gpu_id, console=console)
    if isinstance(pre_snap, dict) and pre_snap:
        state.pre_run_system_snapshot = pre_snap
    else:
        state.pre_run_system_snapshot = {}

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
        f"Results in [cyan]{resolved_output}/[/]"
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
    experience: Optional[Path] = typer.Option(
        None, "-E", "--experience",
        help="Path to optimization_memory.json (experience items for the planner)",
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
    mem = load_memory(experience) if experience else []

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


@app.command()
def setup():
    """Interactive wizard — configure Vertex AI credentials and verify environment."""
    import shutil
    import subprocess as _sp

    from rich.table import Table

    from kernelopt.config import config_path, load_config, save_config
    from kernelopt.functions.setup import (
        check_auth,
        check_gpu,
        check_ncu,
        check_vertex_connection,
    )

    console.print("\n[bold green]kernelopt setup[/]\n")
    console.print("Configures Vertex AI and verifies your environment.\n")

    existing = load_config().get("vertex", {})
    default_project = (
        os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or existing.get("project_id") or ""
    )
    default_region = (
        os.environ.get("VERTEX_REGION") or existing.get("region") or "us-east5"
    )

    if default_project:
        project_id = typer.prompt("Google Cloud project ID", default=default_project)
    else:
        project_id = typer.prompt("Google Cloud project ID")

    region = typer.prompt("Vertex AI region", default=default_region)

    path = save_config(project_id, region)
    console.print(f"\n[green]✓[/] Config saved → [cyan]{path}[/]\n")

    console.print("[bold]Checking authentication...[/]")
    auth_ok, auth_msg = check_auth()
    if auth_ok:
        console.print(f"[green]✓[/] {auth_msg}")
    else:
        console.print(f"[yellow]⚠[/] {auth_msg}")
        if shutil.which("gcloud"):
            console.print("\nLaunching: gcloud auth application-default login")
            try:
                _sp.run(["gcloud", "auth", "application-default", "login"], check=True)
                auth_ok, auth_msg = check_auth()
                if auth_ok:
                    console.print("[green]✓[/] Authenticated")
                else:
                    console.print("[red]✗[/] Auth failed after gcloud login")
            except _sp.CalledProcessError:
                console.print("[red]✗[/] gcloud auth failed")
        else:
            console.print("\nOptions to set up credentials:")
            console.print("  • Install gcloud: https://cloud.google.com/sdk/docs/install")
            console.print("    Then run: gcloud auth application-default login")
            console.print("  • Or set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")

    console.print("\n[bold]Verifying environment...[/]\n")

    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Check", style="dim", min_width=22)
    t.add_column("Status", justify="center", min_width=6)
    t.add_column("Detail")

    def _row(label: str, ok: bool, detail: str) -> None:
        t.add_row(label, "[green]✓[/]" if ok else "[red]✗[/]", detail)

    ok, msg = check_gpu()
    _row("GPU", ok, msg)

    ok, msg = check_ncu()
    _row("NCU binary", ok, msg)

    if auth_ok:
        ok, msg = check_vertex_connection(project_id, region)
        _row("Vertex AI", ok, msg)

    console.print(t)
    console.print(f"\n[bold green]Ready![/] Run: [cyan]kernelopt run <kernel.py>[/]")


@app.command()
def doctor():
    """Check environment health — config, credentials, GPU, NCU, Vertex AI."""
    from rich.table import Table

    from kernelopt.config import config_path, load_config
    from kernelopt.functions.setup import (
        check_auth,
        check_gpu,
        check_ncu,
        check_python_version,
        check_vertex_connection,
    )

    cfg = load_config().get("vertex", {})
    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or cfg.get("project_id", "")
    region = os.environ.get("VERTEX_REGION") or cfg.get("region", "")

    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Check", style="dim", min_width=22)
    t.add_column("Status", justify="center", min_width=6)
    t.add_column("Detail")

    all_ok = True

    def _row(label: str, ok: bool, detail: str) -> None:
        nonlocal all_ok
        if not ok:
            all_ok = False
        t.add_row(label, "[green]✓[/]" if ok else "[red]✗[/]", detail)

    cp = config_path()
    _row("Config file", cp.exists(), str(cp) if cp.exists() else "not found — run: kernelopt setup")
    _row("project_id", bool(project_id), project_id or "not set — run: kernelopt setup")
    _row("region", bool(region), region or "not set — run: kernelopt setup")

    ok, msg = check_python_version()
    _row("Python", ok, msg)

    ok, msg = check_auth()
    _row("Auth credentials", ok, msg)

    ok, msg = check_gpu()
    _row("GPU", ok, msg)

    ok, msg = check_ncu()
    _row("NCU binary", ok, msg)

    if project_id and region:
        ok, msg = check_vertex_connection(project_id, region)
        _row("Vertex AI", ok, msg)
    else:
        _row("Vertex AI", False, "skipped — project_id or region not set")

    console.print(t)
    raise typer.Exit(0 if all_ok else 1)


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
