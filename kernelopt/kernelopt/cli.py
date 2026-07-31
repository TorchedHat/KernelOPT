"""CLI entry point for kernelopt. Built with Typer."""

from __future__ import annotations

from datetime import datetime
import os
import re
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


# With --checkpoint + pipeline_checkpoint.json, these may not be set on the CLI (config from checkpoint).
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
    "provider",
    "base_url",
    "hf_inference_provider",
    "entry_point",
    "input_shapes",
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


def _parse_shapes(s: str) -> list[tuple[int, ...]]:
    """Parse ``"(8,512) (8,512,512)"`` into ``[(8, 512), (8, 512, 512)]``."""
    shapes = []
    for m in re.finditer(r'\(([^)]+)\)', s):
        shapes.append(tuple(int(x.strip()) for x in m.group(1).split(',')))
    return shapes


def _write_compile_fallback(final_path: Path, original_path: Path) -> None:
    """Replace final_model.py with a torch.compile fallback.

    When the synthesized kernel is slower than torch.compile, replace
    final_model.py with a wrapper that runs torch.compile on the original
    model.  Uses a fixed seed (42) for model creation so that the E2E
    correctness check (which also seeds with 42) produces identical
    weights and max_diff=0.0.
    """
    try:
        orig_source = original_path.read_text()
    except OSError as e:
        raise RuntimeError(
            f"Cannot read original kernel {original_path}: {e}"
        ) from e
    fallback = (
        '"""KernelOPT output: torch.compile baseline.\n'
        '\n'
        'KernelOPT could not produce a faster kernel for this model.\n'
        'This file wraps the original model with torch.compile.\n'
        '"""\n'
        f'{orig_source}\n\n'
        '# ── torch.compile baseline ──\n'
        'import torch as _torch\n'
        '_torch.manual_seed(42)\n'
        '_torch.cuda.manual_seed(42)\n'
        '_init = get_init_inputs() if "get_init_inputs" in dir() else []\n'
        '_model = Model(*_init).cuda().eval()\n'
        '_compiled = _torch.compile(_model)\n'
        '# Warmup\n'
        'with _torch.no_grad():\n'
        '    _compiled(*[t.cuda() if isinstance(t, _torch.Tensor) else t '
        'for t in get_inputs()])\n'
        '\n'
        'def kernel_function(*inputs):\n'
        '    with _torch.no_grad():\n'
        '        return _compiled(*inputs)\n'
    )
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(fallback)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write fallback to {final_path}: {e}"
        ) from e


@app.command()
def run(
    ctx: typer.Context,
    kernel_path: Path = typer.Argument(
        ..., help="Kernel file (.py, .hl)"
    ),
    framework: str = typer.Option(
        "auto", help="triton|helion|auto"
    ),
    iterations: Optional[int] = typer.Option(
        None, "-T", "--iterations",
        help="Max optimization iterations (default 5; with --checkpoint: from checkpoint if omitted)",
    ),
    plans: Optional[int] = typer.Option(
        None, "-N", "--plans",
        help="Plans per iteration (default 4; with --checkpoint: from checkpoint if omitted)",
    ),
    attempts: Optional[int] = typer.Option(
        None, "-K", "--attempts",
        help="Executor attempts per plan (default 1; with --checkpoint: from checkpoint if omitted)",
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
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint",
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
    search_policy: str = typer.Option(
        "greedy", "--search-policy",
        help='Search policy: "greedy" (single chain) or "beam" (multi-chain forest)',
    ),
    beam_width: int = typer.Option(
        1, "--beam-width",
        help="Number of concurrent optimization chains (beam width). Only used with --search-policy beam.",
    ),
    e2e_retries: int = typer.Option(
        2, "--e2e-retries",
        help="Max retries when optimized sub-kernel breaks full-model correctness or performance. "
             "Feeds e2e error back to executor for targeted fix attempts.",
    ),
    e2e_perf_gate: bool = typer.Option(
        True, "--e2e-perf-gate/--no-e2e-perf-gate",
        help="After re-stitching, verify the full model is not slower than baseline. "
             "Rejects per-kernel-fast but model-slow optimizations (e.g., stride/layout mismatches).",
    ),
    e2e_perf_margin: float = typer.Option(
        1.03, "--e2e-perf-margin",
        help="Noise margin for e2e performance gate. Reject if optimized_ms > baseline_ms * margin. "
             "Default 1.03 (3%% tolerance for do_bench variance).",
    ),
    e2e_correctness_timeout: int = typer.Option(
        180, "--e2e-correctness-timeout",
        help="Timeout in seconds for e2e correctness subprocess. Includes torch.compile "
             "capture, model compilation, and 3-seed verification. Increase for large models.",
    ),
    trace_timeout: int = typer.Option(
        1800, "--trace-timeout",
        help="Timeout in seconds for graph tracing subprocess (torch.export + torch.compile "
             "with autotuning). Complex architectures need longer. Default 1800s (30 min).",
    ),
    no_memory: bool = typer.Option(
        False, "--no-memory",
        help="Disable experience memory accumulation (ablation). Queue capacity set to 0.",
    ),
    skip_ncu: bool = typer.Option(
        False, "--skip-ncu",
        help="Skip NCU profiling (ablation). Planner receives no hardware bottleneck context.",
    ),
    optimize_all_kernels: bool = typer.Option(
        False, "--optimize-all-kernels",
        help="Disable Inductor-aware synthesis (ablation). LLM may optimize extern_kernels too.",
    ),
    replay_profiles: Optional[str] = typer.Option(
        None, "--replay-profiles",
        help="Replay NCU profiles from a prior run directory instead of live profiling.",
    ),
    use_compiler_context: bool = typer.Option(
        True, "--use-compiler-context/--no-compiler-context",
        help="Extract compiler IR/fusion/autotune context for planner/executor prompts.",
    ),
    multi_kernel: bool = typer.Option(
        True, "--multi-kernel/--single-kernel",
        help="Optimize ALL Triton kernels in a PyTorch model sequentially, re-stitching each back",
    ),
    entry_point: Optional[str] = typer.Option(
        None, "--entry-point",
        help='Callable to compile: "ClassName", "ClassName.method", or "function_name". Auto-detected if omitted.',
    ),
    input_shapes: Optional[str] = typer.Option(
        None, "--input-shapes",
        help='Tensor shapes for synthesis, e.g. "(8,512) (8,512,512)". Each becomes a torch.randn(shape, device=cuda) input.',
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key (not needed for Vertex AI)"
    ),
    model: str = typer.Option(
        "claude-sonnet-4-6", help="Default model name (used when per-agent models not set)"
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
    debug: bool = typer.Option(False, "--debug", help="Enable debug prints (profiler config, correctness subprocess)"),
    hud: bool = typer.Option(
        False,
        "--hud",
        help="Same terminal: alternate-screen HUD; NCU + full stream → kernelopt_terminal.log. Press [t] to toggle HUD / log tail (TTY only).",
    ),
    agentic_correctness: bool = typer.Option(
        True, "--agentic-correctness/--no-agentic-correctness",
        help="Use Claude to generate CPU reference tests (default: True)"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider",
        help="LLM provider: vertex, anthropic_api, openai, google_genai, huggingface, local (default: from config or 'vertex')",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url",
        help="Base URL for OpenAI-compatible / local server (e.g. http://localhost:8000/v1)",
    ),
    hf_inference_provider: str = typer.Option(
        "auto", "--hf-inference-provider",
        help="Hugging Face inference provider (default: auto)",
    ),
):
    """Run the full optimization pipeline.

    Generates N plans per iteration, profiles each valid candidate with NCU,
    and selects the best. Outputs best_kernel, optimization_log.md, and
    .ncu-rep files for every candidate. Each kernelopt run uses a new
    run_<timestamp>/ folder under --output-dir unless --no-run-subdir.
    With --checkpoint and a pipeline_checkpoint.json, NCU/LLM options are taken from the
    checkpoint; only -T, -N, -K and --hud may be set (omit T/N/K to use saved values).

    Example:
        kernelopt run softmax.py -T 3 -N 4 --kernel-name 'softmax_kernel' --kernel-args '--profile-only'
    """
    from kernelopt.config import load_config as _load_config
    _file_cfg = _load_config()
    _vertex_cfg = _file_cfg.get("vertex", {})
    _provider_cfg = _file_cfg.get("provider", {})
    if not project_id:
        project_id = _vertex_cfg.get("project_id") or ""
    if (
        ctx.get_parameter_source("region") not in (
            ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT
        )
        and _vertex_cfg.get("region")
    ):
        region = _vertex_cfg["region"]
    if provider is None:
        provider = _provider_cfg.get("name", "vertex")
    if base_url is None:
        base_url = _provider_cfg.get("base_url")

    from kernelopt.checkpoint import load_checkpoint_file, pipeline_config_from_checkpoint_dict
    from kernelopt.state import PipelineConfig, PipelineState

    ckpt: dict | None = None
    if checkpoint is not None:
        run_dir_pre = checkpoint.resolve()
        if not run_dir_pre.is_dir():
            console.print(f"[red]Error:[/] --checkpoint must be an existing directory: {run_dir_pre}")
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
                    "[red]Error:[/] With [cyan]--checkpoint[/] and a [cyan]pipeline_checkpoint.json[/], "
                    "config is loaded from the checkpoint.\n"
                    "  Only these options are allowed on the command line:\n"
                    "    • kernel file (positional)\n"
                    "    • [cyan]--checkpoint[/] DIR\n"
                    "    • [cyan]-T[/] / [cyan]-N[/] / [cyan]-K[/] (optional overrides)\n"
                    "    • [cyan]--hud[/] (optional; same-terminal HUD + log file)\n"
                    f"  Remove: {', '.join(_flag_label(p) for p in bad)}"
                )
                raise typer.Exit(2)
        else:
            for p in ("output_dir", "run_subdir"):
                if _cli_set_param(ctx, p):
                    console.print(
                        "[red]Error:[/] With [cyan]--checkpoint[/], output is the run directory; "
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

    if checkpoint is not None:
        run_dir = checkpoint.resolve()
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
                keep_n=top_k,
                kernel_name=kernel_name,
                launch_skip=launch_skip,
                launch_count=launch_count,
                replay_mode=replay_mode,
                kernel_args=kernel_args.split() if kernel_args else [],
                model=model,
                api_key=api_key,
                project_id=project_id or "",
                region=region,
                llm_provider=provider,
                openai_base_url=base_url,
                hf_inference_provider=hf_inference_provider,
                planner_model=planner_model,
                summarizer_model=summarizer_model,
                executor_model=executor_model,
                profile_hints=profile_hints,
                profiler_model=profiler_model,
                output_dir=str(resolved_output),
                kernel_path=str(kernel_path),
                baseline_csv=str(csv) if csv else None,
                verbose=verbose,
                debug=debug,
                agentic_correctness=agentic_correctness,
                entry_point=entry_point,
                input_shapes=_parse_shapes(input_shapes) if input_shapes else None,
                search_policy=search_policy,
                beam_width=beam_width,
                e2e_retries=e2e_retries,
                e2e_perf_gate=e2e_perf_gate,
                e2e_perf_margin=e2e_perf_margin,
                e2e_correctness_timeout=e2e_correctness_timeout,
                trace_timeout=trace_timeout,
                no_memory=no_memory,
                skip_ncu=skip_ncu,
                optimize_all_kernels=optimize_all_kernels,
                replay_profiles=replay_profiles or "",
                use_compiler_context=use_compiler_context,
            )
    else:
        resolved_output = _run_output_dir(output_dir, run_subdir=run_subdir)
        t0, n0, k0 = _defaults_tnk()
        cfg = PipelineConfig(
            T=t0,
            N=n0,
            K=k0,
            keep_n=top_k,
            kernel_name=kernel_name,
            launch_skip=launch_skip,
            launch_count=launch_count,
            replay_mode=replay_mode,
            kernel_args=kernel_args.split() if kernel_args else [],
            model=model,
            api_key=api_key,
            project_id=project_id or "",
            region=region,
            llm_provider=provider,
            openai_base_url=base_url,
            hf_inference_provider=hf_inference_provider,
            planner_model=planner_model,
            summarizer_model=summarizer_model,
            executor_model=executor_model,
            profile_hints=profile_hints,
            profiler_model=profiler_model,
            output_dir=str(resolved_output),
            kernel_path=str(kernel_path),
            baseline_csv=str(csv) if csv else None,
            verbose=verbose,
            debug=debug,
            agentic_correctness=agentic_correctness,
            entry_point=entry_point,
            input_shapes=_parse_shapes(input_shapes) if input_shapes else None,
            search_policy=search_policy,
            beam_width=beam_width,
            e2e_retries=e2e_retries,
            e2e_perf_gate=e2e_perf_gate,
            e2e_perf_margin=e2e_perf_margin,
            e2e_correctness_timeout=e2e_correctness_timeout,
            no_memory=no_memory,
            skip_ncu=skip_ncu,
            optimize_all_kernels=optimize_all_kernels,
            replay_profiles=replay_profiles or "",
            use_compiler_context=use_compiler_context,
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
        resume_run_dir=str(resolved_output) if checkpoint is not None else None,
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
        f"K={cfg.K} attempts/plan, keep_n={cfg.keep_n}\n"
        f"  Output: [cyan]{resolved_output}/[/]\n"
        f"  Backend: [cyan]{cfg.llm_provider}[/] ({cfg.model})"
        f"{extra}{hud_note}"
    )

    # Pre-run GPU validation (snapshot reused by HUD so it matches System Status)
    gpu_id = select_gpu()
    pre_snap = validate_gpu_for_profiling(gpu_id, console=console)
    if isinstance(pre_snap, dict) and pre_snap:
        state.pre_run_system_snapshot = pre_snap
    else:
        state.pre_run_system_snapshot = {}

    import time as _time

    from kernelopt.agents.profiler import get_latency_ns
    from kernelopt.errors import KerneloptUserError
    from kernelopt.langgraph_pipeline import run_langgraph_pipeline
    from kernelopt.progress import print_run_summary

    _run_start = _time.monotonic()

    # Multi-kernel mode: optimize all Triton kernels sequentially
    if multi_kernel and state.framework in ("pytorch", "auto", ""):
        try:
            from kernelopt.synthesis.graph_tracer import trace_model
            from kernelopt.synthesis.fuse_groups import merge_fusible_groups
            from kernelopt.synthesis.llm_codegen import generate_triton_kernel
            from kernelopt.synthesis.validate_eager import validate_against_eager
            from kernelopt.multi_kernel import optimize_all_kernels

            console.print("[bold]Multi-kernel mode: synthesizing + optimizing all Triton kernels[/]")

            # Phase 1: Trace computation graph + get Inductor fusion decisions
            console.print("  [dim]Tracing computation graph...[/]")
            kernel_groups = trace_model(str(kernel_path), timeout=cfg.trace_timeout)
            console.print(f"  [dim]Inductor generated {len(kernel_groups)} kernel(s)[/]")

            # Ablation C6: disable Inductor-aware synthesis
            if getattr(cfg, "optimize_all_kernels", False):
                for kg in kernel_groups:
                    kg["needs_triton_replacement"] = True
                console.print("  [yellow]--optimize-all-kernels: extern kernels included in optimization[/]")

            # Phase 2: Merge fusible kernel groups
            kernel_groups = merge_fusible_groups(kernel_groups)
            console.print(f"  [green]After fusion: {len(kernel_groups)} kernel group(s)[/]")

            from kernelopt.llm import pipeline_llm_base_kwargs
            llm_config_dict = {"model": cfg.model, **pipeline_llm_base_kwargs(cfg)}

            synth_dir = Path(resolved_output) / "synthesis"
            synth_dir.mkdir(parents=True, exist_ok=True)

            # Phase 1b: LLM generates clean Triton kernels (with K retries)
            codegen_attempts = cfg.K
            # Include original PyTorch source so the LLM can see the exact formula
            _pytorch_src = Path(kernel_path).read_text() if Path(kernel_path).exists() else ""
            generated_kernels = []
            for i, kg in enumerate(kernel_groups):
                kname = kg.get("kernel_name", f"kernel_{i}")
                ktype = kg.get("kernel_type", "unknown")
                aten_ops = [op["name"] for op in kg.get("aten_graph", [])]
                console.print(f"  [dim]Generating kernel {i}: {kname} ({ktype}) ops={aten_ops}[/]")
                kg["pytorch_source"] = _pytorch_src

                source = None
                last_error = None
                ok = False
                msg = ""
                for attempt in range(codegen_attempts):
                    if attempt > 0:
                        console.print(f"    [yellow]Retry {attempt}/{codegen_attempts} with error feedback...[/]")
                        kg["_codegen_error"] = last_error

                    source = generate_triton_kernel(kg, llm_config_dict)
                    if source is None:
                        console.print(f"    [red]LLM codegen failed for {kname}[/]")
                        break

                    kernel_path_gen = synth_dir / f"kernel_{i}_{kname[:40]}.py"
                    kernel_path_gen.write_text(source)

                    console.print(f"    [dim]Validating against PyTorch eager (attempt {attempt+1}/{codegen_attempts})...[/]")
                    ok, msg = validate_against_eager(
                        str(kernel_path), str(kernel_path_gen),
                    )
                    if ok:
                        break
                    last_error = msg
                    console.print(f"    [dim]Attempt {attempt+1} failed: {msg[:150]}[/]")
                    source = None

                if source and ok:
                    console.print(f"    [green]Correctness passed[/]")
                    generated_kernels.append({
                        "kernel_name": kname,
                        "kernel_type": ktype,
                        "source": source,
                        "path": str(kernel_path_gen),
                    })
                else:
                    console.print(f"    [red]Correctness failed: {msg[:200]}[/]")

            if not generated_kernels:
                console.print("[yellow]No valid kernels generated — falling back to Inductor path[/]")
                # Fallback to old Inductor extraction path
                from kernelopt.synthesis import extract_triton_from_pytorch, wrap_inductor_output
                results = extract_triton_from_pytorch(
                    str(kernel_path),
                    entry_point=entry_point,
                    input_shapes=_parse_shapes(input_shapes) if input_shapes else None,
                )
                has_triton = any(
                    r.get("triton_kernels") for r in results
                ) if results else False
                if not has_triton:
                    console.print(
                        "[yellow]Inductor generated no Triton kernels for this model. "
                        "Writing torch.compile baseline.[/]"
                    )
                    resolved_output = Path(cfg.output_dir)
                    final_path = resolved_output / "final_model.py"
                    _write_compile_fallback(final_path, kernel_path)
                    console.print(f"  Final model: [cyan]{final_path}[/]")
                    raise typer.Exit(0)
                primary = results[0]
                orig_source = kernel_path.read_text()
                wrapped = wrap_inductor_output(
                    primary["inductor_source"], orig_source, primary["triton_kernels"],
                )
                synth_path = synth_dir / "synthesized_triton.py"
                synth_path.write_text(wrapped)
                state.kernel_path = str(synth_path)
                state.kernel_source = wrapped
                bk_files = primary.get("benchmark_kernel_files", [])
                final_source, mk_results = optimize_all_kernels(state, bk_files, orig_source)
            else:
                # Phase 3: Optimize each generated kernel through Profile→Plan→Execute
                orig_source = kernel_path.read_text()
                bk_files = [{"source": gk["source"], "kernel_name": gk["kernel_name"]} for gk in generated_kernels]

                # Use the first generated kernel as the standalone for optimization
                # (multi-kernel assembly happens in Phase 4)
                standalone_path = Path(generated_kernels[0]["path"])
                state.kernel_path = str(standalone_path)
                state.kernel_source = generated_kernels[0]["source"]
                state.framework = "triton"
                state.best_kernel = generated_kernels[0]["source"]

                from kernelopt.langgraph_pipeline import run_langgraph_pipeline
                console.print(f"\n  [bold]Optimizing {len(generated_kernels)} kernel(s)...[/]")

                mk_results = []
                for gk in generated_kernels:
                    from kernelopt.state import PipelineConfig, PipelineState
                    kernel_output_dir = Path(resolved_output) / f"opt_{gk['kernel_name'][:30]}"
                    kernel_output_dir.mkdir(parents=True, exist_ok=True)

                    # Resolve per-sub-kernel replay directory
                    _sub_replay = cfg.replay_profiles
                    if _sub_replay:
                        _kn = gk['kernel_name'][:30]
                        _sub_rp = Path(_sub_replay) / f"opt_{_kn}"
                        if not _sub_rp.exists():
                            for _sd in Path(_sub_replay).iterdir():
                                if _sd.is_dir() and _kn in _sd.name and (_sd / "baseline_profiling_context.json").exists():
                                    _sub_rp = _sd
                                    break
                        _sub_replay = str(_sub_rp) if _sub_rp.exists() else ""

                    kernel_cfg = PipelineConfig(
                        T=cfg.T, N=cfg.N, K=cfg.K, keep_n=cfg.keep_n,
                        model=cfg.model, api_key=cfg.api_key,
                        project_id=cfg.project_id, region=cfg.region,
                        llm_provider=cfg.llm_provider,
                        openai_base_url=cfg.openai_base_url,
                        hf_inference_provider=cfg.hf_inference_provider,
                        planner_model=cfg.planner_model,
                        executor_model=cfg.executor_model,
                        profiler_model=cfg.profiler_model,
                        summarizer_model=cfg.summarizer_model,
                        output_dir=str(kernel_output_dir),
                        kernel_path=gk["path"],
                        agentic_correctness=cfg.agentic_correctness,
                        search_policy=cfg.search_policy,
                        beam_width=cfg.beam_width,
                        replay_mode=cfg.replay_mode,
                        replay_profiles=_sub_replay,
                        skip_ncu=cfg.skip_ncu,
                        no_memory=cfg.no_memory,
                        debug=cfg.debug, verbose=cfg.verbose,
                    )
                    kernel_state = PipelineState(
                        config=kernel_cfg,
                        kernel_path=gk["path"],
                        kernel_source=gk["source"],
                        kernel_basename=gk["kernel_name"][:40],
                        best_kernel=gk["source"],
                        framework="triton",
                        optimization_memory=list(state.optimization_memory),
                    )
                    kernel_state._baseline_standalone_path = gk["path"]

                    try:
                        console.print(f"\n  [bold]━━ {gk['kernel_name'][:50]} ━━[/]")
                        result_state = run_langgraph_pipeline(kernel_state)
                        from kernelopt.agents.profiler import get_latency_ns
                        bl_ns = get_latency_ns(result_state.baseline_profiling_context)
                        best_ns = result_state.best_latency_ns
                        speedup = round(bl_ns / best_ns, 2) if best_ns > 0 and best_ns < float("inf") and bl_ns > 0 else 1.0

                        # Save best kernel
                        best_path = kernel_output_dir / "best_kernel.py"
                        best_path.write_text(result_state.best_kernel)

                        mk_results.append({
                            "kernel_name": gk["kernel_name"],
                            "status": "optimized",
                            "speedup": speedup,
                        })
                        console.print(f"    {gk['kernel_name'][:40]}: {speedup:.2f}x")
                        state.optimization_memory = list(result_state.optimization_memory)
                    except Exception as e:
                        console.print(f"    [red]Optimization failed: {e}[/]")
                        mk_results.append({
                            "kernel_name": gk["kernel_name"],
                            "status": "failed",
                            "reason": str(e)[:200],
                            "speedup": 1.0,
                        })

                # Phase 4: Assemble final output
                from kernelopt.synthesis.assemble import assemble_output
                final_source = assemble_output(
                    generated_kernels,
                    optimized_dir=Path(resolved_output),
                    original_pytorch_path=str(kernel_path),
                )

            # Save final model
            final_path = Path(resolved_output) / "final_model.py"
            final_path.write_text(final_source)

            elapsed = _time.monotonic() - _run_start
            mins, secs = divmod(int(elapsed), 60)
            console.print(f"\n[bold]Multi-kernel complete[/] in {mins}m {secs:02d}s")
            for r in mk_results:
                name = r["kernel_name"][:40]
                if r["status"] == "optimized" and r.get("speedup", 1) > 1:
                    console.print(f"  [green]{name}: {r['speedup']:.2f}x (per-kernel)[/]")
                else:
                    console.print(f"  [dim]{name}: {r['status']}[/]")
            console.print(f"  Final model: [cyan]{final_path}[/]")

            # End-to-end model benchmark: eager vs compile vs KernelOPT
            try:
                from kernelopt.functions.benchmark import benchmark_model_e2e
                e2e = benchmark_model_e2e(
                    original_model_path=str(kernel_path),
                    optimized_model_path=str(final_path),
                )
                # Correctness check first
                if not e2e.get("correct", True):
                    console.print(
                        f"\n  [bold red]E2E correctness FAILED: {e2e.get('error', 'unknown')}[/]"
                    )
                    console.print(
                        f"  [red]Optimized kernel produces wrong results — falling back to torch.compile baseline.[/]"
                    )
                    _write_compile_fallback(final_path, kernel_path)
                else:
                    eager = e2e.get("eager_ms", -1)
                    compile_ = e2e.get("compile_ms", -1)
                    opt = e2e.get("optimized_ms", -1)
                    vs_compile = e2e.get("vs_compile", 0)
                    vs_eager = e2e.get("vs_eager", 0)
                    max_diff = e2e.get("max_diff", -1)
                    console.print(f"\n  [bold]End-to-end (do_bench, subprocess-isolated, correctness verified max_diff={max_diff:.2e}):[/]")
                    if eager > 0:
                        console.print(f"    Eager:     {eager:.3f} ms")
                    if compile_ > 0:
                        console.print(f"    Compile:   {compile_:.3f} ms")
                    if opt > 0:
                        console.print(f"    KernelOPT: {opt:.3f} ms")
                    _NOISE_THRESHOLD = 0.97  # 3% do_bench variance tolerance
                    if vs_compile > 0 and vs_compile >= 1.0:
                        console.print(
                            f"  [bold green]vs compile: {vs_compile:.2f}x[/]"
                        )
                    elif vs_compile >= _NOISE_THRESHOLD:
                        # Within measurement noise — report as baseline
                        console.print(
                            f"  [dim]vs compile: ~1.00x (within measurement noise, raw={vs_compile:.4f}x)[/]"
                        )
                    elif vs_compile > 0 and vs_compile < _NOISE_THRESHOLD:
                        console.print(
                            f"  [yellow]No improvement: KernelOPT output is {vs_compile:.2f}x vs torch.compile "
                            f"(compile={compile_:.3f}ms, optimized={opt:.3f}ms). "
                            f"Falling back to torch.compile baseline.[/]"
                        )
                        _write_compile_fallback(final_path, kernel_path)
                    if vs_eager > 0:
                        console.print(
                            f"  [bold green]vs eager:   {vs_eager:.2f}x[/]"
                        )
                    if "error" in e2e:
                        console.print(f"  [dim]E2E issues: {e2e['error']}[/]")
            except Exception as exc:
                console.print(f"  [dim]E2E benchmark failed: {exc}[/]")
            raise typer.Exit(0)
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[red]Multi-kernel failed: {exc}[/]")
            raise typer.Exit(1) from None

    try:
        final = run_langgraph_pipeline(state)
    except KerneloptUserError as exc:
        console.print("[bold red]kernelopt cannot continue[/]")
        console.print(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        elapsed = _time.monotonic() - _run_start
        mins, secs = divmod(int(elapsed), 60)
        console.print(
            f"\n[yellow]Interrupted[/] after {mins}m {secs:02d}s. "
            f"Checkpoint saved — resume with:\n"
            f"  [cyan]kernelopt run {kernel_path.name} "
            f"--checkpoint {resolved_output}/[/]"
        )
        raise typer.Exit(130) from None
    except Exception as exc:
        if cfg.hud and not cfg.verbose:
            console.print("[bold red]Run failed[/]")
            console.print(str(exc))
            console.print(
                "[dim]For a full Python traceback, run without [cyan]--hud[/] "
                "and/or add [cyan]--verbose[/].[/]"
            )
            raise typer.Exit(1) from None
        raise

    total_elapsed = _time.monotonic() - _run_start
    baseline_ns = get_latency_ns(final.baseline_profiling_context)
    best_ns = final.best_latency_ns

    iteration_results = getattr(final, "iteration_results", None) or []
    best_direction = ""
    # Find the direction of the overall best candidate
    for result in iteration_results:
        if result.improved and result.best_candidate:
            best_direction = result.best_candidate.direction

    if not cfg.hud:
        print_run_summary(
            baseline_ns=baseline_ns,
            best_ns=best_ns,
            best_direction=best_direction,
            iteration_results=iteration_results,
            output_dir=str(resolved_output),
            total_elapsed=total_elapsed,
            console=console,
        )
    else:
        def _lat_summary(ns: float) -> str:
            if ns != ns or ns == float("inf"):
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
    model: str = typer.Option("claude-sonnet-4-6"),
    project_id: Optional[str] = typer.Option(
        None, "--project-id", envvar="ANTHROPIC_VERTEX_PROJECT_ID",
        help="Google Cloud project ID for Vertex AI"
    ),
    region: str = typer.Option(
        "us-east5", "--region", help="Vertex AI region"
    ),
    provider: str = typer.Option(
        "vertex", "--provider", help="LLM provider"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Base URL for OpenAI-compatible / local server"
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
        "provider": provider,
        "openai_base_url": base_url,
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
    """Interactive wizard — choose LLM provider, configure credentials, verify environment."""
    import shutil
    import subprocess as _sp

    from rich.table import Table

    from kernelopt.config import config_path, load_config, save_config
    from kernelopt.functions.setup import (
        check_auth,
        check_gpu,
        check_local_connection,
        check_ncu,
        check_vertex_connection,
    )
    from kernelopt.llm import PROVIDERS

    console.print("\n[bold green]kernelopt setup[/]\n")

    # Step 1: Choose provider
    existing = load_config()
    existing_provider = existing.get("provider", {}).get("name", "vertex")

    console.print("[bold]Choose your LLM provider:[/]\n")
    for i, (pid, label) in enumerate(PROVIDERS, 1):
        marker = " [dim](current)[/]" if pid == existing_provider else ""
        console.print(f"  {i}. {label}{marker}")
    console.print()

    default_idx = next(
        (i for i, (pid, _) in enumerate(PROVIDERS, 1) if pid == existing_provider), 1
    )
    choice = typer.prompt("Enter number", default=str(default_idx))
    try:
        chosen_provider = PROVIDERS[int(choice) - 1][0]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice[/]")
        raise typer.Exit(1)

    console.print(f"\n[green]✓[/] Provider: [cyan]{chosen_provider}[/]\n")

    # Step 2: Provider-specific configuration
    project_id = ""
    region = "us-east5"
    base_url = ""
    model_name = ""

    if chosen_provider == "vertex":
        vertex_cfg = existing.get("vertex", {})
        default_project = (
            os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            or vertex_cfg.get("project_id") or ""
        )
        default_region = (
            os.environ.get("VERTEX_REGION")
            or vertex_cfg.get("region") or "us-east5"
        )
        if default_project:
            project_id = typer.prompt("Google Cloud project ID", default=default_project)
        else:
            project_id = typer.prompt("Google Cloud project ID")
        region = typer.prompt("Vertex AI region", default=default_region)

    elif chosen_provider == "local":
        default_url = existing.get("provider", {}).get("base_url", "http://localhost:8000/v1")
        base_url = typer.prompt("Local server URL", default=default_url)
        model_name = typer.prompt("Model name (as served by vLLM/SGLang)", default="")

    elif chosen_provider == "anthropic_api":
        console.print("  Set [cyan]ANTHROPIC_API_KEY[/] in your environment.")

    elif chosen_provider == "openai":
        base_url = typer.prompt("Base URL (leave empty for api.openai.com)", default="")
        console.print("  Set [cyan]OPENAI_API_KEY[/] in your environment.")

    elif chosen_provider == "google_genai":
        console.print("  Set [cyan]GOOGLE_API_KEY[/] or [cyan]GEMINI_API_KEY[/] in your environment.")

    elif chosen_provider == "huggingface":
        console.print("  Set [cyan]HF_TOKEN[/] in your environment.")

    path = save_config(
        project_id=project_id, region=region,
        provider=chosen_provider, base_url=base_url, model=model_name,
    )
    console.print(f"\n[green]✓[/] Config saved → [cyan]{path}[/]\n")

    # Step 3: Provider-specific auth check
    auth_ok = False
    if chosen_provider == "vertex":
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

    # Step 4: Environment checks
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

    if chosen_provider == "vertex" and auth_ok:
        ok, msg = check_vertex_connection(project_id, region)
        _row("Vertex AI", ok, msg)
    elif chosen_provider == "local" and base_url:
        ok, msg = check_local_connection(base_url)
        _row("Local server", ok, msg)

    console.print(t)
    console.print(f"\n[bold green]Ready![/] Run: [cyan]kernelopt run <kernel.py>[/]")


@app.command()
def doctor():
    """Check environment health — config, credentials, GPU, NCU, LLM provider."""
    from rich.table import Table

    from kernelopt.config import config_path, load_config
    from kernelopt.functions.setup import (
        check_auth,
        check_gpu,
        check_local_connection,
        check_ncu,
        check_python_version,
        check_vertex_connection,
    )

    file_cfg = load_config()
    vertex_cfg = file_cfg.get("vertex", {})
    provider_cfg = file_cfg.get("provider", {})
    chosen_provider = provider_cfg.get("name", "vertex")
    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or vertex_cfg.get("project_id", "")
    region = os.environ.get("VERTEX_REGION") or vertex_cfg.get("region", "")

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
    _row("Provider", True, chosen_provider)

    ok, msg = check_python_version()
    _row("Python", ok, msg)

    ok, msg = check_gpu()
    _row("GPU", ok, msg)

    ok, msg = check_ncu()
    _row("NCU binary", ok, msg)

    if chosen_provider == "vertex":
        _row("project_id", bool(project_id), project_id or "not set — run: kernelopt setup")
        _row("region", bool(region), region or "not set — run: kernelopt setup")
        ok, msg = check_auth()
        _row("Auth credentials", ok, msg)
        if project_id and region:
            ok, msg = check_vertex_connection(project_id, region)
            _row("Vertex AI", ok, msg)
        else:
            _row("Vertex AI", False, "skipped — project_id or region not set")
    elif chosen_provider == "local":
        base_url = provider_cfg.get("base_url") or os.environ.get("LOCAL_LLM_BASE_URL", "")
        if base_url:
            ok, msg = check_local_connection(base_url)
            _row("Local server", ok, msg)
        else:
            _row("Local server", False, "base_url not configured — run: kernelopt setup")

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
