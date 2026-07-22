"""
LangGraph-based optimization pipeline.

Graph:
  [maybe_synthesize] → profile_baseline → analyze_strategy → generate_plans → execute_plans
       → profile_candidates → summarize_results
       → select_best → increment_iteration
       → {should_continue: continue → generate_plans | stop → END}

maybe_synthesize runs only when framework=="pytorch" (auto-detected).
It converts PyTorch input to Triton via torch.compile, then the rest
of the pipeline proceeds on the synthesized Triton kernel.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import string
import uuid
from pathlib import Path

from langgraph.graph import END, StateGraph

from kernelopt.llm import pipeline_llm_base_kwargs

try:
    from langgraph.graph import START
except ImportError:  # pragma: no cover
    START = "__start__"
from rich.console import Console
from rich.table import Table

from kernelopt.agents.executor import run_executor_agent
from kernelopt.agents.summarizer import run_summarizer
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


def _maybe_spin(phase: str, label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with an optional spinner when HUD is off."""
    if _use_spinners():
        with phase_spinner(phase, label):
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def _get_primary_kernel(state):
    """Get the primary active kernel source (first in beam, or legacy best_kernel)."""
    active = state.get("active_kernels")
    if active:
        return active[0]["kernel_source"]
    return state.get("best_kernel", state.get("global_best_kernel", ""))


def _get_primary_context(state):
    """Get the primary active kernel's profiling context."""
    active = state.get("active_kernels")
    if active:
        return active[0].get("profiling_context", {})
    return state.get("current_profiling_context", {})


def _build_ncu_summary(ctx):
    """Build compact NCU summary string for tree/beam storage."""
    if not ctx:
        return ""
    sol = ctx.get("sections", {}).get("GPU Speed Of Light Throughput", {})
    dur = sol.get("Duration", {}).get("value", "?")
    mem = sol.get("Memory [%]", {}).get("value", "?")
    sm = sol.get("SM [%]", {}).get("value", "?")
    rules = ctx.get("rules", [])
    top_rules = ", ".join(r["name"][:30] for r in rules[:3])
    return f"dur={dur} mem%={mem} sm%={sm} rules=[{top_rules}]"


def _checkpoint_stage(state: GraphState, updates: dict, stage: str) -> None:
    """Persist merged state after a pipeline stage (for --checkpoint resume)."""
    try:
        merged: GraphState = {**state, **updates}  # type: ignore[misc]
        save_checkpoint(Path(str(merged["output_dir"])), merged, stage)
    except Exception as e:
        pipe_print(f"[yellow]Warning:[/] checkpoint [{stage}] failed: {e}")


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
    from kernelopt.functions.csv_parser import parse_ncu_csv, find_bottleneck_kernel

    cfg = state["config"]
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = state["kernel_path"]
    kernel_id = state.get("kernel_id", 0)
    auto_kernel_id = (kernel_id == 0 and not cfg.kernel_name)
    _target_triton_name = None  # may be set in the multi-kernel branch below

    # Ablation: skip NCU profiling entirely
    if getattr(cfg, "skip_ncu", False):
        from kernelopt.functions.benchmark import benchmark_kernel
        pipe_print("[dim]NCU profiling skipped (--skip-ncu ablation)[/]")
        wall_result = benchmark_kernel(kernel_path)
        wall_ms = wall_result.get("median_ms", 1.0) if isinstance(wall_result, dict) else 1.0
        wall_ns = wall_ms * 1e6
        return {
            "baseline_profiling_context": {},
            "best_latency_ns": wall_ns,
            "best_wall_ms": wall_ms,
            "ncu_config": None,
            "active_kernels": [{
                "chain_id": "chain_0",
                "kernel_source": state.get("kernel_source", ""),
                "latency_ns": wall_ns,
                "wall_ms": wall_ms,
                "speedup": 1.0,
                "profiling_context": {},
                "history": [],
                "depth": 0,
                "alive": True,
            }],
        }

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Baseline profiling", Path(kernel_path).name)

    ext = Path(kernel_path).suffix

    llm_config = {"model": cfg.profiler_model or cfg.model, **pipeline_llm_base_kwargs(cfg)}

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
            baseline_ctx, ncu_config = _maybe_spin(
                "baseline", Path(kernel_path).name,
                run_profiler_agent,
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
        pipe_print(f"  [dim]Profiling context: {ctx_json}[/]")

        # Multi-kernel bottleneck discovery: if kernel_id was auto (0) and
        # the CSV has multiple kernels, find the most expensive one and
        # re-parse the context for that kernel.
        csv_path = str(output_dir / "baseline.csv")
        if auto_kernel_id and Path(csv_path).exists():
            bottleneck_id, all_kernels = find_bottleneck_kernel(csv_path)
            if all_kernels and len(all_kernels) > 1:
                pipe_print(f"  [dim]Multi-kernel file: {len(all_kernels)} kernels detected[/]")
                for k in all_kernels[:8]:
                    is_target = k["kernel_id"] == bottleneck_id
                    marker = "[bold cyan]→[/]" if is_target else "  "
                    dur_us = k["duration_ns"] / 1000
                    name = k["kernel_name"][:55]
                    kind = k.get("kind", "")
                    tag = f" [dim]({kind})[/]" if kind and kind not in ("user", "triton") else ""
                    pipe_print(f"  {marker} [{k['kernel_id']}] {dur_us:>10.2f} µs  {name}{tag}")
                if len(all_kernels) > 8:
                    pipe_print(f"    ... and {len(all_kernels) - 8} more")

            if all_kernels and len(all_kernels) > 1:
                is_synth = bool(state.get("_original_pytorch_path"))
                if is_synth:
                    # Synthesized path: only target Triton kernels — cuBLAS,
                    # cuDNN, and runtime kernels can't be rewritten.
                    triton_kernels = [k for k in all_kernels if k.get("kind") == "triton"]
                    if triton_kernels:
                        kernel_id = triton_kernels[0]["kernel_id"]
                        _target_triton_name = triton_kernels[0]["kernel_name"]
                        pipe_print(
                            f"  [bold]Baseline target: "
                            f"ID={kernel_id} ({_target_triton_name[:50]})[/]"
                        )
                    else:
                        pipe_print(
                            "  [yellow]No Triton kernels found — all kernels are "
                            "library/runtime calls (cuBLAS, cuDNN, etc.). "
                            "Nothing to optimize.[/]"
                        )
                        return {"best_latency_ns": float("inf")}
                else:
                    kernel_id = bottleneck_id
                    _bottleneck_k = next(
                        (k for k in all_kernels if k["kernel_id"] == bottleneck_id),
                        all_kernels[0],
                    )
                    pipe_print(
                        f"  [bold]Bottleneck: "
                        f"ID={kernel_id} ({_bottleneck_k['kernel_name'][:50]})[/]"
                    )

                sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
                baseline_ctx = build_profiling_context(
                    sections, rules, launch, kernel_path, kernel_id=kernel_id
                )

                # Inject multi-kernel summary into profiling context so the planner
                # knows about ALL kernels (including the cuBLAS GEMM bottleneck).
                if all_kernels and len(all_kernels) > 1:
                    kernel_summary = []
                    for k in all_kernels:
                        dur_us = k["duration_ns"] / 1000
                        kind = k.get("kind", "user")
                        kernel_summary.append(
                            f"  [{k['kernel_id']}] {dur_us:>10.2f} µs  {k['kernel_name'][:60]}  ({kind})"
                        )
                    user_kernel_count = sum(1 for k in all_kernels if k.get("kind") == "user")
                    if user_kernel_count == 0:
                        guidance = (
                            "No user @triton.jit kernels found — computation is entirely "
                            "library calls. Optimization scope is very limited."
                        )
                    elif user_kernel_count == 1:
                        guidance = (
                            "Focus optimization on the single @triton.jit user kernel. "
                            "Library calls (cuBLAS, cuDNN) are already hardware-optimized."
                        )
                    else:
                        guidance = (
                            f"{user_kernel_count} @triton.jit user kernels found. "
                            "Consider fusing adjacent user kernels or optimizing the "
                            "most expensive one."
                        )

                    _bn_k = next(
                        (k for k in all_kernels if k["kernel_id"] == kernel_id),
                        all_kernels[0],
                    )
                    # Kernel time fraction (for display in multi_kernel_summary only)
                    _total_kernel_ncu_ns = sum(k["duration_ns"] for k in all_kernels)
                    _bn_pct = round(_bn_k["duration_ns"] / _total_kernel_ncu_ns * 100) if _total_kernel_ncu_ns > 0 else 100

                    baseline_ctx["multi_kernel_summary"] = (
                        f"This file launches {len(all_kernels)} GPU kernels. "
                        f"The bottleneck is kernel [{_bn_k['kernel_id']}] "
                        f"({_bn_k['kernel_name'][:40]}) at "
                        f"{_bn_k['duration_ns']/1000:.1f} µs "
                        f"({_bn_k.get('kind','user')}, "
                        f"{_bn_pct}% of kernel time). "
                        f"{guidance}\n"
                        + "\n".join(kernel_summary)
                    )

    latency = get_latency_ns(baseline_ctx)
    framework = state["framework"]
    if framework == "auto":
        framework = baseline_ctx["framework_hint"]

    # Wall-clock benchmark for end-to-end comparison
    _baseline_wall_ms = float("inf")
    try:
        from kernelopt.functions.benchmark import benchmark_kernel
        _wc = benchmark_kernel(kernel_path)
        if _wc.get("median_ms", -1) > 0:
            _baseline_wall_ms = _wc["median_ms"]
    except Exception:
        pass

    _wall_str = f"  Wall: {_baseline_wall_ms:.3f} ms" if _baseline_wall_ms < float("inf") else ""
    pipe_print(
        f"  Baseline: [cyan]{latency:.0f} ns[/] (NCU){_wall_str}  "
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

    import re as _re_bl

    _triton_name = _target_triton_name
    is_synth = bool(state.get("_original_pytorch_path"))
    if not _triton_name and is_synth:
        # Fallback: extract kernel name from the standalone source itself
        _standalone_match = _re_bl.search(
            r'def\s+(triton_\w+)\s*\(', _get_primary_kernel(state),
        )
        if _standalone_match:
            _triton_name = _standalone_match.group(1)
        else:
            _triton_name = baseline_ctx.get("kernel_name", "")
    out = {
        "baseline_profiling_context": baseline_ctx,
        "baseline_csv": cfg.baseline_csv or str(output_dir / "baseline.csv"),
        "baseline_ncu_rep": str(output_dir / "baseline.ncu-rep"),
        "baseline_latency_ns": latency,
        "baseline_wall_ms": _baseline_wall_ms,
        "best_latency_ns": latency,
        "best_wall_ms": _baseline_wall_ms,
        "current_profiling_context": baseline_ctx,
        "framework": framework,
        "ncu_config": ncu_config,
        "kernel_id": kernel_id,
        # Multi-chain forest search fields
        "active_kernels": [
            {
                "chain_id": "chain_0",
                "kernel_source": _get_primary_kernel(state),
                "latency_ns": latency,
                "wall_ms": _baseline_wall_ms,
                "speedup": 1.0,
                "profiling_context": baseline_ctx,
                "ncu_summary": _build_ncu_summary(baseline_ctx),
                "history": [],
                "depth": 0,
                "alive": True,
            }
        ],
        "global_best_kernel": state.get("best_kernel", ""),
        "global_best_latency_ns": latency,
    }
    if _triton_name:
        out["_target_triton_kernel_name"] = _triton_name

    # If the target kernel differs from the current standalone, rebuild the
    # standalone from the correct benchmark file so optimizations target the
    # right kernel and re-stitching can match by name.
    if _triton_name and is_synth:
        _cur_standalone = _get_primary_kernel(state)
        _cur_name_m = _re_bl.search(r'def\s+(triton_\w+)\s*\(', _cur_standalone)
        _cur_name = _cur_name_m.group(1) if _cur_name_m else ""
        if _cur_name and _cur_name != _triton_name:
            all_bk = state.get("_all_benchmark_files") or []
            # Find benchmark files where the @triton.jit function matches
            _jit_pattern = _re_bl.compile(
                r'@triton\.jit\s*\n\s*def\s+' + _re_bl.escape(_triton_name) + r'\s*\('
            )
            match_bk = [
                bk for bk in all_bk
                if _jit_pattern.search(bk.get("source", ""))
            ]
            # Prefer per-kernel files (only one @triton.jit function)
            per_kernel = [
                bk for bk in match_bk
                if len(_re_bl.findall(r'@triton\.jit', bk.get("source", ""))) == 1
            ]
            match_bk = per_kernel if per_kernel else match_bk
            if match_bk:
                from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone
                new_bk = match_bk[0]
                try:
                    new_standalone = build_benchmark_standalone(
                        benchmark_source=new_bk["source"],
                        original_pytorch_source=state.get("_original_pytorch_source", ""),
                    )
                    out["best_kernel"] = new_standalone
                    # Use the standalone (with kernel_function, get_inputs)
                    # as benchmark source — rebuild_standalone_from_body uses
                    # it as a template for the executor's optimized body.
                    out["_benchmark_kernel_source"] = new_standalone
                    synth_dir = Path(state.get("kernel_path", "")).parent
                    bl_path = synth_dir / "baseline_standalone.py"
                    bl_path.write_text(new_standalone)
                    out["_baseline_standalone_path"] = str(bl_path)
                    pipe_print(
                        f"  [green]Rebuilt standalone for target: "
                        f"{_triton_name[:50]}[/]"
                    )
                except Exception as e:
                    pipe_print(
                        f"  [dim]Could not rebuild standalone for "
                        f"{_triton_name[:40]}: {e}[/]"
                    )
            else:
                pipe_print(
                    f"  [dim]No benchmark file found for {_triton_name[:40]} "
                    f"(standalone has {_cur_name[:40]})[/]"
                )

    # Initialize search tree with baseline as root
    # Always store the baseline standalone as the template for
    # rebuild_standalone_from_body(). This is the kernel file that passed
    # dry-run — it has correct kernel_function(), get_inputs(), grid.
    if "_benchmark_kernel_source" not in out:
        out["_benchmark_kernel_source"] = _get_primary_kernel(state)

    from kernelopt.search_tree import SearchTree
    tree = SearchTree(Path(output_dir) / "search_tree.json")
    tree.set_root(_get_primary_kernel(state), latency)

    _checkpoint_stage(state, out, "after_baseline")
    return out


# ---------------------------------------------------------------------------
# Node 1.5: Strategy Analysis (two-phase planning)
# ---------------------------------------------------------------------------

STRATEGY_SYSTEM_PROMPT = """\
You are the Strategy Analyst in a GPU kernel optimization pipeline.

Given a kernel's multi-kernel profile and source code, decide the HIGH-LEVEL
optimization strategy. You do NOT write code — you decide WHAT to optimize.

IMPORTANT GUIDANCE: Library calls (extern_kernels.mm, cuBLAS GEMM,
extern_kernels.convolution, cuDNN) are already hardware-optimized by NVIDIA.
Focus primarily on @triton.jit kernel functions and their parameters —
this is where real acceleration happens. Rewriting library calls as Triton
is extremely difficult and has a very low success rate. Only recommend it
if strong NCU evidence shows the library call is suboptimal for this
specific workload size and hardware.

Return a JSON object with:
{
  "strategy_type": "optimize_kernel" | "fuse_pointwise" | "tune_parameters",
  "reasoning": "one paragraph explaining why",
  "primary_target": "description of what to optimize in the @triton.jit kernel(s)",
  "fusion_opportunity": "specific pointwise fusion description or null",
  "plan_guidance": ["guidance string for plan A", "for plan B", "for plan C"]
}

Strategy types:
- "optimize_kernel": The Triton kernel has inefficiencies. Optimize tile sizes,
  memory access patterns, occupancy, vectorization, or precision.
- "fuse_pointwise": Multiple Triton pointwise kernels can be fused into one to
  eliminate intermediate memory traffic. Do NOT fuse with library calls.
- "tune_parameters": The Triton kernel is already well-structured. Focus on
  constexpr parameters (XBLOCK, num_warps, num_stages, eviction policies).
"""


def _compute_ncu_tier(profiling_ctx: dict) -> dict:
    """Derive optimization tier from NCU SOL% and occupancy metrics.

    Returns a dict with strategy_type, reasoning, and recommended approaches
    based on whether the kernel is memory-bound, compute-bound, or underutilized.
    """
    throughput = profiling_ctx.get("GPU Speed Of Light Throughput", {})
    mem_section = profiling_ctx.get("Memory Workload Analysis", {})

    mem_sol_str = throughput.get("Memory [%]", throughput.get("DRAM [%]", "0"))
    compute_sol_str = throughput.get("Compute (SM) [%]", throughput.get("SM [%]", "0"))

    try:
        mem_sol = float(str(mem_sol_str).replace("%", "").strip())
    except (ValueError, TypeError):
        mem_sol = 0.0
    try:
        compute_sol = float(str(compute_sol_str).replace("%", "").strip())
    except (ValueError, TypeError):
        compute_sol = 0.0

    if mem_sol > 80 or compute_sol > 80:
        return {
            "tier": 0,
            "tier_name": "near_optimal",
            "strategy_type": "tune_parameters",
            "reasoning": f"Kernel is near hardware limits (Memory SOL: {mem_sol:.0f}%, Compute SOL: {compute_sol:.0f}%). "
                         f"Only minor parameter tuning may help.",
            "primary_target": "constexpr parameters (tile sizes, num_warps)",
            "plan_guidance": [
                "Try different XBLOCK/RBLOCK values to find the sweet spot for this specific workload size",
                "Experiment with num_warps (2, 4, 8) and num_stages",
                "Try eviction_policy hints on loads/stores",
            ],
        }
    elif mem_sol > compute_sol:
        return {
            "tier": 1,
            "tier_name": "memory_bound",
            "strategy_type": "optimize_kernel",
            "reasoning": f"Kernel is memory-bound (Memory SOL: {mem_sol:.0f}% > Compute SOL: {compute_sol:.0f}%). "
                         f"Focus on reducing memory traffic and improving access patterns.",
            "primary_target": "memory access patterns and data movement",
            "plan_guidance": [
                "Improve memory coalescing in loads/stores",
                "Add vectorized loads (float4) to increase bandwidth utilization",
                "Use L2 compression hints (eviction_policy) to reduce memory traffic",
                "Increase tile sizes to improve L2 cache hit rate",
            ],
        }
    elif compute_sol > mem_sol:
        return {
            "tier": 2,
            "tier_name": "compute_bound",
            "strategy_type": "optimize_kernel",
            "reasoning": f"Kernel is compute-bound (Compute SOL: {compute_sol:.0f}% > Memory SOL: {mem_sol:.0f}%). "
                         f"Focus on reducing instruction count and improving occupancy.",
            "primary_target": "computation efficiency and occupancy",
            "plan_guidance": [
                "Reduce register pressure to increase occupancy",
                "Eliminate branch divergence",
                "Use BF16/FP16 for intermediate computations where precision allows",
                "Reduce unnecessary operations (common subexpression elimination)",
            ],
        }
    else:
        return {
            "tier": 3,
            "tier_name": "underutilized",
            "strategy_type": "optimize_kernel",
            "reasoning": f"Kernel is underutilized (Memory SOL: {mem_sol:.0f}%, Compute SOL: {compute_sol:.0f}%). "
                         f"Significant room for improvement in both memory and compute.",
            "primary_target": "overall kernel efficiency",
            "plan_guidance": [
                "Increase tile sizes to improve parallelism and cache utilization",
                "Fuse adjacent pointwise operations to eliminate intermediate buffers",
                "Add vectorized loads and stores",
                "Consider register tiling for compute-intensive sections",
            ],
        }


def analyze_strategy(state: GraphState) -> dict:
    """Phase 1: Decide the macro optimization strategy before generating plans.

    For multi-kernel files: one LLM call to analyze the kernel landscape and
    decide whether to fuse, optimize, or rewrite. The strategy guides all N
    planners in Phase 2.

    For single-kernel files: pass-through (strategy = optimize_kernel).
    """
    profiling_ctx = _get_primary_context(state)
    multi_kernel_summary = profiling_ctx.get("multi_kernel_summary", "")

    # Derive optimization tier from NCU metrics
    ncu_tier = _compute_ncu_tier(profiling_ctx)

    # Inject technique hints from the optimization technique library
    from kernelopt.optimization_techniques import select_techniques
    tech_list = select_techniques(profiling_ctx, _get_primary_kernel(state), max_techniques=3)
    if tech_list:
        technique_hints = [f"Consider '{t.name}': {t.description}" for t, _ in tech_list]
        ncu_tier["plan_guidance"] = technique_hints + ncu_tier.get("plan_guidance", [])

    if not multi_kernel_summary or "single kernel" in multi_kernel_summary.lower():
        return {"optimization_strategy": {
            "strategy_type": ncu_tier.get("strategy_type", "optimize_kernel"),
            "reasoning": ncu_tier.get("reasoning", "Single kernel file — optimize the kernel directly."),
            "primary_target": ncu_tier.get("primary_target", "the kernel"),
            "fusion_opportunity": None,
            "plan_guidance": ncu_tier.get("plan_guidance", []),
            "ncu_tier": ncu_tier,
        }}

    kernel_source = _get_primary_kernel(state)

    # If there are no @triton.jit kernels in the source, optimization scope
    # is extremely limited (computation is entirely library calls).
    if "@triton.jit" not in kernel_source and "No user @triton.jit" in multi_kernel_summary:
        pipe_print("  [yellow]No @triton.jit kernels — computation is all library calls. "
                   "Limited optimization scope.[/]")
        return {"optimization_strategy": {
            "strategy_type": "tune_parameters",
            "reasoning": "No user @triton.jit kernels found. The computation is entirely "
                         "library calls (cuBLAS/cuDNN). Only minor parameter tuning of the "
                         "computation graph may help.",
            "primary_target": "launch parameters and memory layout",
            "fusion_opportunity": None,
            "plan_guidance": [
                "Check if batch dimensions or tensor layouts can be improved",
                "Look for redundant memory copies or suboptimal strides",
            ],
            "ncu_tier": ncu_tier,
        }}

    from kernelopt.llm import build_llm, strip_thinking
    import json as _json

    cfg = state["config"]
    llm_config = {"model": cfg.planner_model or cfg.model, **pipeline_llm_base_kwargs(cfg)}

    pipe_print("  Analyzing optimization strategy...")

    # Query memory bank for past experiences with similar kernels
    from kernelopt.memory_bank import MemoryBank, classify_kernel_structure, get_hardware_info
    from kernelopt.reference_kernels import get_relevant_references
    bank = MemoryBank()
    kernel_struct = classify_kernel_structure(kernel_source, profiling_ctx)
    hw_info = get_hardware_info()
    past_failures = bank.get_failures(kernel_struct, hw_info)
    past_successes = bank.get_successes(kernel_struct, hw_info)
    memory_context = bank.format_for_llm(past_failures + past_successes)

    constraint_context = bank.format_constraints_for_llm()
    if constraint_context:
        memory_context = (memory_context + "\n\n" + constraint_context) if memory_context else constraint_context

    templates = bank.get_templates(kernel_source, multi_kernel_summary)
    if templates:
        template_context = bank.format_for_llm(templates, max_entries=2)
        memory_context = (memory_context + "\n\n" + template_context) if memory_context else template_context

    # Get reference kernel examples for this kernel type
    references = get_relevant_references(kernel_struct)
    if references:
        ref_context = "\n\n## Reference kernel examples (from Triton official tutorials):\n"
        for name, code in references[:2]:
            ref_context += f"\n### {name}\n```python\n{code[:1000]}\n```\n"
        memory_context = (memory_context + ref_context) if memory_context else ref_context

    if past_failures or references:
        pipe_print(
            f"  [dim]Memory bank: {len(past_failures)} failures, "
            f"{len(past_successes)} successes, {len(references)} reference kernels[/]"
        )

    try:
        llm = build_llm(**llm_config)
        from langchain_core.messages import HumanMessage, SystemMessage

        tier_context = (
            f"## NCU-Guided Optimization Tier\n"
            f"Tier {ncu_tier['tier']} ({ncu_tier['tier_name']}): {ncu_tier['reasoning']}\n"
            f"Recommended approaches: {', '.join(ncu_tier.get('plan_guidance', []))}\n\n"
        )
        user_msg = (
            f"## Multi-kernel profile\n{multi_kernel_summary}\n\n"
            f"{tier_context}"
            f"## Kernel source (first 200 lines)\n```python\n{kernel_source[:8000]}\n```\n\n"
        )
        if memory_context:
            user_msg += f"\n{memory_context}\n\n"
        user_msg += "Decide the optimization strategy. Return ONLY the JSON object."

        response = llm.invoke([
            SystemMessage(content=STRATEGY_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        text = strip_thinking(response.content)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        strategy = _json.loads(text)
        pipe_print(
            f"  Strategy: [bold]{strategy.get('strategy_type', '?')}[/] — "
            f"{strategy.get('reasoning', '')[:80]}"
        )
        return {"optimization_strategy": strategy}

    except Exception as e:
        pipe_print(f"  [yellow]Strategy analysis failed: {e}. Defaulting to optimize_kernel.[/]")
        return {"optimization_strategy": {
            "strategy_type": "optimize_kernel",
            "reasoning": f"Strategy analysis failed ({e}). Defaulting to single-kernel optimization.",
            "primary_target": "the dominant kernel",
            "fusion_opportunity": None,
            "plan_guidance": [],
        }}


# ---------------------------------------------------------------------------
# Node 2: Generate N optimization plans (concurrent)
# ---------------------------------------------------------------------------

def generate_plans(state: GraphState) -> dict:
    """Generate N diverse optimization plans concurrently via parallel planner sub-graphs."""
    from kernelopt.agents.planner import run_planner_agent

    cfg = state["config"]
    profiling_ctx = _get_primary_context(state)
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

    # Build structured iteration log so the planner knows what was tried
    # and what worked — not just direction names.
    all_tried_directions = []
    iteration_log_lines = []
    for result in state.get("iteration_results", []):
        iter_entries = []
        for c in result.candidates:
            all_tried_directions.append(c.direction)
            speedup_str = f"{c.speedup:.2f}x" if c.latency_ns < float("inf") else "failed"
            correct = "correct" if c.correctness_passed else "INCORRECT"
            iter_entries.append(f"{c.direction}: {speedup_str} ({correct})")
        if iter_entries:
            best = "improved" if result.improved else "no improvement"
            iteration_log_lines.append(
                f"Iter {result.iteration}: {best} — " + ", ".join(iter_entries)
            )

    # Add partial credit: correctness failures with their error signals
    failure_items = state.get("correctness_failure_items", [])
    if failure_items:
        failure_lines = []
        for item in failure_items[-6:]:
            desc = item.get("strategy_description", "")[:150]
            failure_lines.append(f"  - {item.get('direction', '?')}: {desc}")
        if failure_lines:
            iteration_log_lines.append(
                "Approaches that FAILED correctness (do not repeat as-is, "
                "but the direction may be salvageable with a different implementation):\n"
                + "\n".join(failure_lines)
            )

    llm_config = {"model": cfg.planner_model or cfg.model, **pipeline_llm_base_kwargs(cfg)}

    # Inject strategy context into profiling context for the planner
    strategy = state.get("optimization_strategy", {})
    if strategy:
        profiling_ctx = {**profiling_ctx, "optimization_strategy": strategy}

    # Inject iteration log so the planner knows what was tried and what worked
    if iteration_log_lines:
        profiling_ctx = {
            **profiling_ctx,
            "iteration_history": "\n".join(iteration_log_lines),
        }

    # Meltdown detection: if recent plans are repetitive, inject diversity boost
    if len(all_tried_directions) >= 3:
        _normalized = [re.sub(r'[_\s]+', ' ', d.lower().strip()) for d in all_tried_directions]
        _recent = _normalized[-6:]
        _unique = set(_recent)
        if len(_unique) <= 2:
            diversity_msg = (
                "WARNING: The last several iterations produced very similar strategies "
                f"({', '.join(_unique)}). These approaches have been exhausted. "
                "You MUST propose fundamentally different optimization directions: "
                "loop tiling, shared memory usage, warp-level primitives, "
                "algorithmic changes, or data layout transformations. "
                "Do NOT repeat block-size tuning or autotune variations."
            )
            profiling_ctx = {**profiling_ctx, "diversity_required": diversity_msg}
            pipe_print("  [yellow]Meltdown detected — injecting diversity boost[/]")

    # Inject strategy rankings from optimization memory
    opt_mem = _get_opt_memory(state)
    if opt_mem:
        kt = _get_kernel_type(state)
        past_results = opt_mem.format_for_planner(kt)
        if past_results:
            profiling_ctx = {**profiling_ctx, "past_strategy_results": past_results}

    # Detect transposed/permuted layouts — warn planner about stride constraints
    _target_name = state.get("_target_triton_kernel_name", "")
    _orig_src = state.get("_original_pytorch_source") or ""
    _has_permute = (
        "permute" in _target_name
        or ".T" in _orig_src
        or "transpose" in _orig_src
    )
    if _has_permute:
        profiling_ctx = {
            **profiling_ctx,
            "layout_warning": (
                "LAYOUT WARNING: This kernel involves a transposed matrix layout. "
                "While standard configurations rely on hardware-optimized libraries, "
                "your custom Triton implementation must handle this layout without "
                "sacrificing memory coalescing.\n\n"
                "PLANNING STRATEGY:\n"
                "- Advise the code generator to modify the memory access strides "
                "directly in the main loop (e.g., swapping row/col pointer increments) "
                "rather than reshaping tensors in global memory.\n"
                "- Ensure global memory loads are vectorized along the contiguous "
                "dimension to avoid interleaved, uncoalesced memory requests that "
                "tank global E2E model performance."
            ),
        }

    # --- Multi-chain forest: generate plans per active kernel ---
    active = state.get("active_kernels", [])
    beam_width = getattr(cfg, "beam_width", 1)

    if len(active) > 1 and beam_width > 1:
        # Multi-chain: generate plans for each active kernel independently.
        # Distribute N plans across chains as evenly as possible.
        n_chains = len(active)
        n_per_chain = max(1, cfg.N // n_chains)
        all_plans: list[tuple[str, dict, str]] = []  # (letter, plan, chain_id)
        letter_idx = 0
        for ak in active:
            if not ak.get("alive", True):
                continue
            chain_id = ak["chain_id"]
            # Build per-chain profiling context
            chain_ctx = {**profiling_ctx}
            chain_ncu = ak.get("profiling_context", {})
            if chain_ncu:
                chain_ctx = {**chain_ctx, **chain_ncu}
            chain_plan_args = dict(
                profiling_context=chain_ctx,
                kernel_source=ak["kernel_source"],
                memory=state.get("optimization_memory", []),
                N=n_per_chain,
                llm_config=llm_config,
                all_tried_directions=all_tried_directions + ak.get("history", []),
            )
            chain_plans = _maybe_spin(
                "planning", f"{n_per_chain} plans for {chain_id}",
                run_planner_agent, **chain_plan_args,
            )
            for _, plan in chain_plans:
                letter = PLAN_LETTERS[letter_idx % len(PLAN_LETTERS)]
                all_plans.append((letter, plan, chain_id))
                letter_idx += 1
        plans = all_plans
    else:
        # Single-chain (beam_width=1 or no active_kernels): original behavior
        _plan_args = dict(
            profiling_context=profiling_ctx,
            kernel_source=_get_primary_kernel(state),
            memory=state.get("optimization_memory", []),
            N=cfg.N,
            llm_config=llm_config,
            all_tried_directions=all_tried_directions,
        )
        raw_plans = _maybe_spin("planning", f"{cfg.N} optimization plans",
                            run_planner_agent, **_plan_args)
        # Normalize to 3-tuples with empty chain_id for backward compat
        plans = [(letter, plan, "") for letter, plan in raw_plans]

    # Ensure at least one simple "tune_parameters" plan exists as a safe backup.
    # If all plans are complex (fusion, rewrite), append a mechanical tuning plan.
    has_safe = any(
        not _is_complex_plan(p)
        for _, p, _ in plans
    )
    if not has_safe and plans:
        safe_plan = {
            "plan_id": "safe_autotune",
            "direction": "add_triton_autotune",
            "framework": plans[0][1].get("framework", "triton") if plans else "triton",
            "evidence": plans[0][1].get("evidence", {}) if plans else {},
            "step": {
                "action": "tune",
                "target": "kernel constexpr parameters",
                "change": (
                    "Add @triton.autotune decorator with 4 configs exploring "
                    "XBLOCK (128, 256, 512, 1024) and num_warps (4, 8). "
                    "Keep kernel body unchanged."
                ),
                "predicted_effect": "Explore parameter space for potential speedup",
                "implementation_hints": [
                    "Add @triton.autotune(configs=[...], key=['xnumel']) before @triton.jit",
                    "Create 4 triton.Config entries: {XBLOCK: 128, num_warps: 4}, "
                    "{XBLOCK: 256, num_warps: 4}, {XBLOCK: 512, num_warps: 8}, "
                    "{XBLOCK: 1024, num_warps: 8}",
                    "Do NOT modify the kernel body",
                ],
            },
            "correctness_constraints": [],
            "confidence": "high",
            "confidence_reason": "Mechanical parameter exploration, no code changes",
        }
        next_letter = chr(ord(plans[-1][0]) + 1) if plans else "A"
        # Use chain_id from the first plan (or empty for single-chain)
        safe_chain_id = plans[0][2] if plans else ""
        plans.append((next_letter, safe_plan, safe_chain_id))
        pipe_print(f"    [dim]Added safe backup plan {next_letter}: add_triton_autotune[/]")

    for letter, plan, chain_id in plans:
        direction = plan.get("direction", "unknown")
        confidence = plan.get("confidence", "?")
        chain_tag = f" [{chain_id}]" if chain_id else ""
        pipe_print(
            f"    Plan {letter}: {direction} [dim](confidence: {confidence}){chain_tag}[/]"
        )

    out = {"plans": plans}
    _checkpoint_stage(state, out, "after_plans")
    return out


# ---------------------------------------------------------------------------
# Executor helpers: profiling context + reference snippets for anti-hallucination
# ---------------------------------------------------------------------------

def _get_opt_memory(state: GraphState) -> "OptimizationMemory | None":
    """Get or create the optimization memory for this run."""
    from kernelopt.optimization_memory import OptimizationMemory
    output_dir = state.get("output_dir", "")
    if not output_dir:
        return None
    mem_dir = Path(output_dir).parent / "opt_memory"
    return OptimizationMemory(mem_dir)


def _get_kernel_type(state: GraphState) -> str:
    """Classify the current kernel being optimized."""
    from kernelopt.optimization_memory import OptimizationMemory
    return OptimizationMemory.classify_kernel(_get_primary_kernel(state))


def _build_executor_context(state: GraphState) -> str:
    """Build a compact profiling summary for the executor agent."""
    ctx = state.get("baseline_profiling_context", {})
    if not ctx:
        return ""

    gpu = ctx.get("gpu_specs", {})
    launch = ctx.get("launch", {})
    rules = ctx.get("rules", [])

    top_rules = sorted(
        rules, key=lambda r: r.get("speedup_pct") or 0, reverse=True
    )[:3]
    rule_lines = "\n".join(
        f"  - {r.get('rule_name', '?')}: {(r.get('speedup_pct') or 0):.1f}% potential"
        for r in top_rules
    )

    sections = ctx.get("sections", {})
    sol = sections.get("GPU Speed Of Light Throughput", {})
    occ_section = sections.get("Occupancy", {})

    def _metric_val(section_dict, key):
        m = section_dict.get(key, {})
        return m.get("value", "?") if isinstance(m, dict) else m

    mem_pct = _metric_val(sol, "Memory [%]")
    compute_pct = _metric_val(sol, "SM [%]")
    achieved_occ = _metric_val(occ_section, "Achieved Occupancy")
    duration = _metric_val(sol, "Duration")

    base = (
        f"GPU: {gpu.get('gpu_name', 'unknown')} "
        f"({gpu.get('sm_count', '?')} SMs, "
        f"{gpu.get('shared_memory_per_sm_kb', '?')} KB shmem/SM)\n"
        f"Baseline latency: {state.get('baseline_latency_ns', 0):.0f} ns\n"
        f"SOL: Memory={mem_pct}%, Compute(SM)={compute_pct}%\n"
        f"Occupancy: {achieved_occ}%\n"
        f"Launch: {launch.get('block_size', '?')} threads/block, "
        f"{launch.get('registers_per_thread', '?')} regs/thread, "
        f"{launch.get('waves_per_sm', '?')} waves/SM\n"
        f"Top bottlenecks:\n{rule_lines}"
    )

    # Detect reduction kernels and inject optimization guidance
    kernel_src = _get_primary_kernel(state).lower()
    is_reduction = any(k in kernel_src for k in (
        "tl.sum", "tl.max", "tl.min", "rblock", "r_block",
        "max_pool", "softmax", "tl.reduce",
    ))
    has_multi_load_max = kernel_src.count("tl.maximum") >= 3
    if is_reduction or has_multi_load_max:
        base += (
            "\n\n## REDUCTION KERNEL — OPTIMIZATION GUIDANCE\n"
            "This kernel contains reduction operations. The kernel body logic "
            "(loop bounds, accumulation, masks, index calculations) is "
            "correctness-critical and must be preserved exactly.\n\n"
            "SAFE and ENCOURAGED optimizations:\n"
            "- Add @triton.autotune to explore different XBLOCK, RBLOCK, "
            "YBLOCK tile sizes via multiple triton.Config entries. This is "
            "the BEST way to optimize reductions — Triton runtime picks "
            "the fastest config automatically.\n"
            "- Vary num_warps (4, 8, 16) and num_stages (1, 2, 3, 4) in configs\n"
            "- Add eviction_policy='evict_first' or 'evict_last' to tl.load/store\n"
            "- Change constexpr default values (XBLOCK, RBLOCK)\n\n"
            "UNSAFE — do NOT modify:\n"
            "- Loop bounds or iteration counts in the kernel body\n"
            "- Accumulation logic (tl.sum, tl.max, += patterns)\n"
            "- Index calculations or mask expressions\n"
            "- The number or order of tl.load / tl.store calls\n\n"
            "The correctness checker will validate your output against the "
            "original — if results diverge, the kernel is rejected."
        )

    # Detect template matmul kernels and inject safety constraints
    is_template = any(k in kernel_src for k in (
        "tl.dot", "triton_tem_", "block_m", "block_n", "block_k",
        "accumulator", "acc = tl.zeros",
    ))
    if is_template and not is_reduction:
        base += (
            "\n\n## TEMPLATE MATMUL KERNEL — OPTIMIZATION GUIDANCE\n"
            "This is an Inductor-generated GEMM template with tl.dot. "
            "The tiling loop and accumulator logic are tightly coupled.\n\n"
            "SAFE and ENCOURAGED optimizations:\n"
            "- Add @triton.autotune to explore BLOCK_M, BLOCK_N, BLOCK_K "
            "sizes via multiple triton.Config entries\n"
            "- Vary num_warps (4, 8, 16) and num_stages (1, 2, 3, 4, 5)\n"
            "- Set allow_tf32=True in tl.dot for faster F32 matmuls\n"
            "- Add eviction_policy hints to tl.load/tl.store\n"
            "- Set enable_fp_fusion=True for fused multiply-add\n\n"
            "UNSAFE — do NOT modify:\n"
            "- The accumulator loop structure (for k_idx in range(...))\n"
            "- The tl.dot call pattern or its dtype arguments\n"
            "- Index calculations for A and B matrix loads\n"
            "- The order of loads/stores around tl.dot\n\n"
            "The correctness checker will validate your output against the "
            "original — if results diverge, the kernel is rejected."
        )

    # Inject reference example from optimization memory
    opt_mem = _get_opt_memory(state)
    if opt_mem:
        kt = _get_kernel_type(state)
        ref_prompt = opt_mem.format_for_executor(kt)
        if ref_prompt:
            base += "\n\n" + ref_prompt

    # Inject applicable optimization techniques from the technique library
    from kernelopt.optimization_techniques import select_techniques, format_techniques_for_executor
    techniques = select_techniques(ctx, _get_primary_kernel(state), max_techniques=2)
    if techniques:
        base += "\n\n" + format_techniques_for_executor(techniques)

    return base


def _build_reference_snippets(plan: dict) -> str | None:
    """Get reference kernel snippets relevant to the plan direction."""
    from kernelopt.reference_kernels import get_reference_kernel

    direction = plan.get("direction", "").lower()
    keywords_to_techniques = {
        "matmul": "matmul", "gemm": "matmul", "mm": "matmul",
        "softmax": "fused_softmax",
        "layernorm": "layer_norm", "layer_norm": "layer_norm", "rmsnorm": "layer_norm",
        "attention": "fused_attention", "flash": "fused_attention",
        "dropout": "dropout",
    }

    for keyword, technique in keywords_to_techniques.items():
        if keyword in direction:
            ref = get_reference_kernel(technique)
            if ref:
                return f"### {technique} (Triton tutorial)\n```python\n{ref[:1500]}\n```"

    # Fallback: check if the plan direction matches an optimization technique name
    from kernelopt.optimization_techniques import get_technique_by_name, get_all_technique_names
    for tech_name in get_all_technique_names():
        if tech_name.replace("_", " ") in direction or tech_name in direction:
            tech = get_technique_by_name(tech_name)
            if tech and tech.code_pattern.after:
                return (
                    f"### {tech.name} technique\n"
                    f"**Before:**\n```python\n{tech.code_pattern.before}\n```\n"
                    f"**After:**\n```python\n{tech.code_pattern.after}\n```\n"
                    f"{tech.code_pattern.explanation}"
                )

    return None


def _is_complex_plan(plan: dict) -> bool:
    """Check if a plan is too complex for reliable executor implementation."""
    step = plan.get("step", {})
    action = step.get("action", "").lower()
    hints = step.get("implementation_hints", [])
    if action in ("fuse", "rewrite", "replace", "merge"):
        return True
    if len(hints) > 20:
        return True
    return False


def _simplify_plan(plan: dict) -> dict:
    """Downgrade a complex plan to a simpler version of the same direction."""
    import copy
    simple = copy.deepcopy(plan)
    simple["direction"] = plan.get("direction", "tune") + "_simplified"
    simple["confidence"] = "medium"
    simple["step"] = {
        "action": "tune",
        "target": plan.get("step", {}).get("target", "kernel"),
        "change": (
            "Add @triton.autotune with 6-8 configs exploring XBLOCK, RBLOCK "
            "(if present), num_warps, and num_stages. "
            "Keep the kernel body unchanged."
        ),
        "predicted_effect": "Explore parameter space for potential speedup",
        "implementation_hints": [
            "Add @triton.autotune decorator before @triton.jit",
            "Create 6-8 triton.Config entries varying XBLOCK and num_warps",
            "If the kernel has RBLOCK as a constexpr, also vary RBLOCK across configs",
            "Set key=['xnumel'] (or the appropriate numel parameters like 'rnumel')",
            "Do NOT modify the kernel body — only add the decorator",
        ],
    }
    simple.get("correctness_constraints", []).clear()
    return simple


# ---------------------------------------------------------------------------
# Node 3: Execute each plan (K attempts) + correctness gate
# ---------------------------------------------------------------------------


def _make_failure_item(
    *,
    iteration: int,
    framework: str,
    direction: str,
    plan: dict,
    title: str,
    description: str,
    do_not_apply_when: str,
    chain_id: str = "",
) -> dict:
    """Build a standardized correctness failure item for planner memory."""
    return {
        "item_id": uuid.uuid4().hex[:8],
        "iteration": iteration,
        "speedup": 0.0,
        "rewrite_type": "implementation_flaw",
        "framework": framework,
        "direction": direction,
        "chain_id": chain_id,
        "profiling_signal": {
            "ncu_rule_triggered": plan.get("evidence", {}).get("ncu_rule"),
            "key_metric_before": {},
            "key_metric_after": {},
        },
        "strategy_title": title,
        "strategy_description": description,
        "slow_pseudocode": "",
        "fast_pseudocode": "",
        "applicable_when": plan.get("evidence", {}).get("reasoning", ""),
        "do_not_apply_when": do_not_apply_when,
        "framework_notes": None,
    }


def execute_plans(state: GraphState) -> dict:
    """Apply each plan to produce optimized kernels and check correctness."""
    cfg = state["config"]
    plans = state["plans"]
    current_kernel = _get_primary_kernel(state)
    framework = state["framework"]
    output_dir = Path(state["output_dir"])
    ext = state["ext"]
    iteration = state.get("iteration", 0)
    kernel_basename = state["kernel_basename"]
    kernel_path = state["kernel_path"]

    # Multi-chain: build lookup from chain_id → ActiveKernel dict
    chain_lookup = {ak["chain_id"]: ak for ak in state.get("active_kernels", [])}

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Execute + correctness", f"{len(plans)} plans")

    llm_config = {"model": cfg.executor_model or cfg.model, **pipeline_llm_base_kwargs(cfg)}

    executor_context = _build_executor_context(state)
    benchmark_src = state.get("_benchmark_kernel_source")
    pipe_print(f"  [dim]Benchmark source: {len(benchmark_src) if benchmark_src else 0} chars[/]")

    candidates: list[PlanCandidate] = []
    correctness_failure_items: list[dict] = []

    for plan_entry in plans:
        # Support both 2-tuple (legacy) and 3-tuple (multi-chain) plan formats
        if len(plan_entry) == 3:
            plan_letter, plan, chain_id = plan_entry
        else:
            plan_letter, plan = plan_entry[0], plan_entry[1]
            chain_id = ""

        # Resolve the base kernel: use chain's kernel if available, else legacy best
        if chain_id and chain_id in chain_lookup:
            current_kernel = chain_lookup[chain_id]["kernel_source"]
        else:
            current_kernel = _get_primary_kernel(state)
        direction = plan.get("direction", "unknown")
        safe_dir = direction.replace(" ", "_")[:40]
        file_stem = f"{kernel_basename}_iter{iteration + 1}_plan{plan_letter}_{safe_dir}"

        pipe_print(f"  [dim]Executing plan {plan_letter} ({direction})...[/]")

        last_compile_err: str | None = None
        optimized_source: str | None = None
        execution_result: dict = {}
        correctness_retried = False
        # executor_K = cfg.K gives the executor K attempts WITHIN a single
        # conversation so the LLM sees its previous failed code + the
        # specific validation error in chat history, enabling it to reason
        # about what went wrong rather than repeating the same mistake.
        # outer_attempts = cfg.K keeps the full retry budget for dry-run
        # and correctness failures that need runtime feedback.
        executor_K = cfg.K
        outer_attempts = cfg.K

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
                    profiling_summary=executor_context,
                    reference_snippets=_build_reference_snippets(plan),
                    benchmark_source=benchmark_src,
                    framework=framework,
                )
                optimized_source, execution_result = _maybe_spin(
                    "executing", _exec_label,
                    run_executor_agent, **_exec_args)
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

                # Feed the validation error back to the executor for retry.
                # MASK CONSISTENCY, SIGNATURE MISMATCH, and other validation
                # errors are recoverable — the LLM just needs to fix the issue.
                last_compile_err = reason

                if attempt_idx == outer_attempts - 1:
                    # All K attempts exhausted
                    optimized_source = None
                    break

                pipe_print(
                    f"    [yellow]Plan {plan_letter} attempt {attempt_idx + 1}/"
                    f"{outer_attempts}: validation failed — retrying with error feedback[/]"
                )
                continue  # retry with error fed back via last_compile_err

            # If executor submitted body-only, rebuild the full standalone file
            bk_src = execution_result.get("benchmark_source")
            if execution_result.get("body_only") and bk_src:
                from kernelopt.synthesis.wrap_kernel import rebuild_standalone_from_body
                optimized_source = rebuild_standalone_from_body(optimized_source, bk_src)

            # Save source to disk
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)

            # Save rebuild diff for diagnosis when dry-run fails later
            if execution_result.get("body_only") and bk_src:
                diff_file = str(output_dir / f"{file_stem}_rebuild_diff.txt")
                import difflib as _dl
                _diff = "".join(_dl.unified_diff(
                    bk_src.splitlines(keepends=True),
                    optimized_source.splitlines(keepends=True),
                    fromfile="baseline", tofile="rebuilt", n=3,
                ))
                Path(diff_file).write_text(_diff)

            # Dry-run validation
            from kernelopt.functions.correctness import validate_kernel_runs
            ok, run_err = validate_kernel_runs(kernel_file)
            if not ok:
                last_compile_err = run_err
                pipe_print(
                    f"    [yellow]Plan {plan_letter} attempt {attempt_idx + 1}/"
                    f"{outer_attempts}: dry-run failed[/]\n"
                    f"    {run_err.strip() if run_err else 'unknown error'}"
                )
                if attempt_idx == outer_attempts - 1:
                    correctness_failure_items.append(_make_failure_item(
                        iteration=iteration + 1,
                        framework=framework,
                        direction=direction,
                        plan=plan,
                        title=f"{direction} — dry-run crashed after {outer_attempts} attempts",
                        description=(
                            f"Plan '{direction}' produced a kernel that crashed on execution "
                            f"after {outer_attempts} attempt(s). "
                            f"Last error: {(run_err or '')[:300]}"
                        ),
                        do_not_apply_when=f"Dry-run failed: {(run_err or '')[:200]}",
                        chain_id=chain_id,
                    ))
                    optimized_source = None
                continue

            break  # compiled/validated successfully

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
            provider=cfg.llm_provider,
            openai_base_url=cfg.openai_base_url,
            hf_inference_provider=cfg.hf_inference_provider,
            kernel_args=cfg.kernel_args,
            original_pytorch_path=state.get("_baseline_standalone_path") or (kernel_path if state.get("_original_pytorch_path") else None),
            debug=cfg.debug,
        )
        passed, msg = _maybe_spin("correctness", f"plan {plan_letter}",
                                   check_correctness, **_cc_args)
        execution_result["correctness"] = "passed" if passed else "failed"

        status = "[green]passed[/]" if passed else "[red]failed[/]"
        pipe_print(f"    Plan {plan_letter}: correctness {status}")

        if not passed and cfg.K > 1 and not correctness_retried:
            # Error-guided retry: feed the correctness error back to the
            # executor so it can fix the bug rather than abandoning the plan.
            pipe_print(f"    [yellow]Correctness failed — retrying with error feedback...[/]")
            correctness_retried = True
            try:
                _retry_args = dict(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    memory=state.get("optimization_memory", []),
                    K=1,
                    llm_config=llm_config,
                    correctness_error=msg[:500],
                    profiling_summary=executor_context,
                    reference_snippets=_build_reference_snippets(plan),
                    benchmark_source=benchmark_src,
                )
                optimized_source, execution_result = _maybe_spin(
                    "retry", f"plan {plan_letter}",
                    run_executor_agent, **_retry_args)

                if optimized_source and execution_result.get("ready_for_profiling"):
                    # Rebuild wrapper if body-only submission
                    retry_bk = execution_result.get("benchmark_source")
                    if execution_result.get("body_only") and retry_bk:
                        from kernelopt.synthesis.wrap_kernel import rebuild_standalone_from_body
                        optimized_source = rebuild_standalone_from_body(optimized_source, retry_bk)
                    kernel_file = str(output_dir / f"{file_stem}_retry{ext}")
                    Path(kernel_file).write_text(optimized_source)
                    passed, msg = check_correctness(**{**_cc_args, "optimized_source": optimized_source})
                    if passed:
                        pipe_print(f"    [green]Retry succeeded![/]")
                        execution_result["correctness"] = "passed"
            except Exception as e:
                pipe_print(f"    [dim]Retry failed: {e}[/]")

        if not passed:
            pipe_print(f"    [dim]Failure: {msg[:300]}[/]")
            correctness_failure_items.append(_make_failure_item(
                iteration=iteration + 1,
                framework=framework,
                direction=direction,
                plan=plan,
                title=f"{direction} — correctness failure",
                description=(
                    f"Plan '{direction}' was applied but failed correctness. "
                    f"Error: {msg[:300]}"
                ),
                do_not_apply_when=f"Previous implementation failed: {msg[:200]}",
                chain_id=chain_id,
            ))
            continue

        # Model-level correctness: re-stitch the optimized kernel back into
        # the full compiled model and verify against the original compiled output.
        _target_name = state.get("_target_triton_kernel_name")
        _orig_py = state.get("_original_pytorch_path")
        _has_ks = bool(state.get("kernel_source"))
        if not _target_name and _orig_py:
            pipe_print(f"  [dim]Skipping model correctness: no target kernel name[/]")
        elif _target_name and _orig_py and _has_ks:
            from kernelopt.synthesis.restitch import restitch_kernel, extract_kernel_body
            from kernelopt.functions.correctness import check_model_correctness
            try:
                kernel_body = extract_kernel_body(optimized_source)
                # Verify the optimized kernel matches the target — the standalone
                # may have been built from a different benchmark file than the
                # profiled bottleneck kernel.
                import re as _re
                _opt_name = _re.search(r'def\s+(triton_\w+)\s*\(', kernel_body)
                _opt_name = _opt_name.group(1) if _opt_name else ""
                if _opt_name and _opt_name != _target_name:
                    # Check parameter compatibility before renaming.
                    # If the signatures differ, these are different kernels.
                    _opt_sig = _re.search(r'def\s+\w+\s*\(([^)]+)\)', kernel_body)
                    _tgt_block = state["kernel_source"]
                    _tgt_sig = _re.search(
                        rf'def\s+{_re.escape(_target_name)}\s*\(([^)]+)\)',
                        _tgt_block,
                    )
                    if _opt_sig and _tgt_sig:
                        _opt_params = [p.strip().split(':')[0].strip() for p in _opt_sig.group(1).split(',')]
                        _tgt_params = [p.strip().split(':')[0].strip() for p in _tgt_sig.group(1).split(',')]
                        if _opt_params != _tgt_params:
                            pipe_print(
                                f"  [dim]Skipping re-stitch: kernel {_opt_name} has "
                                f"incompatible params ({len(_opt_params)} vs "
                                f"{len(_tgt_params)})[/]"
                            )
                            raise ValueError("incompatible kernel signatures")
                    kernel_body = kernel_body.replace(
                        f"def {_opt_name}(", f"def {_target_name}(",
                    )
                    pipe_print(
                        f"  [dim]Renamed kernel {_opt_name} → "
                        f"{_target_name} for re-stitching[/]"
                    )
                restitched = restitch_kernel(
                    state["kernel_source"], _target_name, kernel_body,
                )
                rs_path = str(output_dir / f"{file_stem}_restitched{ext}")
                synth_path = state.get("kernel_path")
                pipe_print(f"  [dim]Running model-level correctness (compiled vs restitched)...[/]")
                model_ok, model_msg = _maybe_spin(
                    "model correctness", f"plan {plan_letter}",
                    check_model_correctness,
                    _orig_py, restitched, rs_path,
                    synthesized_path=synth_path,
                )
                if model_ok:
                    pipe_print(f"    [green]Model correctness passed[/]")
                    execution_result["model_correctness"] = "passed"
                    execution_result["restitched_source"] = restitched

                    # Time original vs restitched model
                    from kernelopt.functions.correctness import time_model
                    try:
                        orig_ms, _ = time_model(synth_path)
                        rs_ms, _ = time_model(rs_path)
                        if orig_ms > 0 and rs_ms > 0:
                            model_speedup = round(orig_ms / rs_ms, 2)
                            pipe_print(
                                f"    [dim]Model timing: {orig_ms:.2f}ms → "
                                f"{rs_ms:.2f}ms ({model_speedup:.2f}x)[/]"
                            )
                            execution_result["model_speedup"] = model_speedup
                    except Exception:
                        pass
                else:
                    pipe_print(f"    [red]Model correctness failed[/]: {model_msg[:200]}")
                    execution_result["model_correctness"] = "failed"
            except Exception as e:
                pipe_print(f"    [dim]Re-stitch failed: {e}[/]")
                execution_result["model_correctness"] = "error"

        candidate = PlanCandidate(
            plan_id=plan.get("plan_id", ""),
            plan_letter=plan_letter,
            direction=direction,
            optimization_plan=plan,
            optimized_kernel=optimized_source,
            execution_result=execution_result,
            correctness_passed=passed,
            correctness_message=msg,
            compiled_binary_path="",
            chain_id=chain_id,
        )
        candidates.append(candidate)

    out = {"candidates": candidates, "correctness_failure_items": correctness_failure_items}
    _checkpoint_stage(state, out, "after_execute")
    return out


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Node 4: Profile valid candidates with NCU
# ---------------------------------------------------------------------------

def profile_candidates(state: GraphState) -> dict:
    """Run NCU on each candidate that passed correctness."""
    from kernelopt.agents.profiler import (
        build_profiling_context, get_latency_ns, run_profiler_agent,
    )

    cfg = state["config"]
    candidates = list(state["candidates"])
    output_dir = Path(state["output_dir"])
    ext = state["ext"]
    iteration = state.get("iteration", 0)
    kernel_basename = state["kernel_basename"]
    baseline_kernel_id = state.get("kernel_id", 0)
    best_latency = state["best_latency_ns"]
    saved_ncu_config = state.get("ncu_config")  # reuse baseline config — full bypass
    is_multi_kernel = baseline_kernel_id != 0 or bool(state.get("_original_pytorch_path"))

    llm_config = {"model": cfg.profiler_model or cfg.model, **pipeline_llm_base_kwargs(cfg)}

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

        try:
            _prof_label = f"plan {c.plan_letter} — {c.direction[:30]}"
            # Candidates may have different launch patterns than baseline:
            # - Multi-kernel files (synthesized): different kernel count/order
            # - @triton.autotune candidates: warmup launches before real kernel
            # Use fresh NCU config (LLM re-analyzes the candidate source) to
            # avoid stale launch_skip/launch_count from baseline.
            has_autotune = "@triton.autotune" in c.optimized_kernel
            candidate_ncu_config = None if (is_multi_kernel or has_autotune) else saved_ncu_config

            _prof_args = dict(
                kernel_path=kernel_file,
                output_basename=output_base,
                ncu_config=candidate_ncu_config,
                pipeline_config=cfg,
                kernel_id=0,


                llm_config=llm_config,
            )
            ctx, _ = _maybe_spin("profiling", _prof_label,
                                  run_profiler_agent, **_prof_args)
            c.csv_path = f"{output_base}.csv"
            c.ncu_rep_path = f"{output_base}.ncu-rep"

            # For multi-kernel candidates, find the user Triton kernel's latency
            # (same logic as baseline bottleneck discovery)
            if is_multi_kernel and Path(c.csv_path).exists():
                from kernelopt.functions.csv_parser import find_bottleneck_kernel, parse_ncu_csv
                cand_kid, cand_kernels = find_bottleneck_kernel(c.csv_path)
                user_kernels = [k for k in cand_kernels if k.get("kind") == "user"]
                if user_kernels:
                    cand_kid = user_kernels[0]["kernel_id"]
                if cand_kid != 0:
                    sections, rules, launch = parse_ncu_csv(c.csv_path, cand_kid)
                    ctx = build_profiling_context(
                        sections, rules, launch, kernel_file, kernel_id=cand_kid
                    )

            c.profiling_context = ctx
            c.latency_ns = get_latency_ns(ctx)
            # Compute speedup vs BASELINE (not rolling best) for user-facing display
            baseline_latency_ns = state["baseline_latency_ns"]
            c.speedup = (
                round(baseline_latency_ns / c.latency_ns, 4)
                if c.latency_ns > 0 else 0.0
            )

            # Wall-clock for end-to-end comparison
            try:
                from kernelopt.functions.benchmark import benchmark_kernel
                _wc = benchmark_kernel(kernel_file)
                c.wall_ms = _wc.get("median_ms", float("inf"))
            except Exception:
                c.wall_ms = float("inf")
            _bl_wall = state.get("baseline_wall_ms", float("inf"))
            c.wall_speedup = (
                round(_bl_wall / c.wall_ms, 4)
                if c.wall_ms > 0 and c.wall_ms < float("inf") and _bl_wall < float("inf")
                else 0.0
            )

            # Display both NCU and wall-clock metrics
            _wall_str = f" | {c.wall_ms:.3f} ms wall ({c.wall_speedup:.2f}x)" if c.wall_ms < float("inf") else ""
            pipe_print(
                f"    Plan {c.plan_letter} ({c.direction}): "
                f"{c.latency_ns:.0f} ns NCU ({c.speedup:.2f}x){_wall_str}"
            )

            # Update optimization memory with actual speedup
            _opt_mem = _get_opt_memory(state)
            _bl_ns = state.get("baseline_latency_ns", 0)
            _real_speedup = round(_bl_ns / c.latency_ns, 2) if c.latency_ns > 0 and _bl_ns > 0 else 0.0
            if _opt_mem and _real_speedup > 0 and _real_speedup < 1000:
                _kt = _get_kernel_type(state)
                _opt_mem.record(
                    kernel_type=_kt,
                    strategy=c.direction,
                    speedup=_real_speedup,
                    passed=c.correctness_passed and _real_speedup > 1.0,
                    before_kernel=_get_primary_kernel(state)[:2000],
                    after_kernel=c.optimized_kernel[:2000],
                    change_summary=c.execution_result.get("change_summary", c.direction),
                )

            # Add candidate to search tree
            from kernelopt.search_tree import SearchTree
            _tree = SearchTree(Path(state["output_dir"]) / "search_tree.json")
            _tree.add_candidate(
                parent_id=_tree.get_best().id if _tree.get_best() else "root",
                kernel_source=c.optimized_kernel,
                latency_ns=c.latency_ns,
                direction=c.direction,
                iteration=iteration + 1,
                correctness=c.correctness_passed,
                plan_letter=c.plan_letter,
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
    current_kernel = _get_primary_kernel(state)
    profiling_before = _get_primary_context(state)
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
                provider=cfg.llm_provider,
                openai_base_url=cfg.openai_base_url,
                hf_inference_provider=cfg.hf_inference_provider,
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
        memory = append_items(memory, failure_items, cfg.q_cap)

    # Apply memory curation for profiled candidates
    top_items = select_top_k(experience_items, cfg.keep_n, cfg.spe_gate, cfg.slo_gate)
    if top_items:
        memory = append_items(memory, top_items, cfg.q_cap)

    # Write to persistent memory bank (cross-run learning)
    try:
        from kernelopt.memory_bank import (
            MemoryBank, MemoryEntry, classify_kernel_structure, get_hardware_info,
            compute_metric_deltas,
        )
        bank = MemoryBank()
        kernel_struct = classify_kernel_structure(current_kernel, profiling_before)
        hw_info = get_hardware_info()
        strategy = state.get("optimization_strategy", {})

        import difflib as _difflib

        for c in candidates:
            if not c.direction:
                continue
            is_success = c.correctness_passed and c.speedup > 1.0

            # Compute unified diff for successful optimizations so
            # get_templates() can retrieve reusable code patterns.
            _code_diff = ""
            if is_success and c.optimized_kernel:
                _code_diff = "".join(_difflib.unified_diff(
                    current_kernel.splitlines(keepends=True),
                    c.optimized_kernel.splitlines(keepends=True),
                    fromfile="baseline", tofile="optimized", n=3,
                ))

            _metric_deltas = {}
            if profiling_before and getattr(c, "profiling_context", None):
                _base_sections = profiling_before.get("sections", profiling_before)
                _cand_sections = c.profiling_context.get("sections", c.profiling_context)
                _metric_deltas = compute_metric_deltas(_base_sections, _cand_sections)

            entry = MemoryEntry(
                kernel_structure=kernel_struct,
                hardware=hw_info,
                strategy={
                    "type": strategy.get("strategy_type", "optimize_kernel"),
                    "direction": c.direction,
                },
                outcome="success" if is_success else (
                    "failure" if not c.correctness_passed else "degradation"
                ),
                speedup=c.speedup if c.speedup > 0 else 0.0,
                failure_reason=c.correctness_message[:200] if not c.correctness_passed else "",
                optimization_insight=(
                    next((it.get("insight", "") for it in top_items if it.get("direction") == c.direction), "")
                    if top_items else ""
                ),
                code_pattern=c.optimized_kernel[:500] if is_success else "",
                code_diff=_code_diff,
                metric_deltas=_metric_deltas,
            )
            bank.add(entry)
    except Exception as e:
        pipe_print(f"  [dim]Memory bank write failed: {e}[/]")

    out = {"optimization_memory": memory}
    _checkpoint_stage(state, out, "after_summarize")
    return out


# ---------------------------------------------------------------------------
# Node 6: Select best candidate
# ---------------------------------------------------------------------------

def select_best_candidate(state: GraphState) -> dict:
    """Pick the best candidate, update best kernel if improved.

    When beam_width > 1, selects the top-B candidates (with diversity) to
    form the next beam of active_kernels. When beam_width == 1 (default),
    behaves identically to the original greedy selection.
    """
    candidates = state["candidates"]
    best_latency = state["best_latency_ns"]
    best_kernel = _get_primary_kernel(state)
    best_ncu_rep = state.get("best_ncu_rep", "")
    best_wall_ms = state.get("best_wall_ms", float("inf"))
    current_ctx = _get_primary_context(state)
    baseline_latency = state["baseline_latency_ns"]
    framework = state["framework"]
    iteration = state.get("iteration", 0)
    no_improvement_streak = state.get("no_improvement_streak", 0)
    iteration_results = list(state.get("iteration_results", []))
    memory = state.get("optimization_memory", [])

    cfg = state["config"]
    beam_width = getattr(cfg, "beam_width", 1)
    prev_active = state.get("active_kernels", [])
    global_best_kernel = state.get("global_best_kernel", best_kernel)
    global_best_latency = state.get("global_best_latency_ns", best_latency)

    iter_result = IterationResult(
        iteration=iteration + 1,
        candidates=candidates,
    )

    # --- Greedy selection (always runs, used for legacy fields) ---
    # Prefer wall-clock when all candidates have real timing data; fall back to NCU latency.
    improved = False
    all_have_wall = bool(candidates) and all(
        c.wall_ms < float("inf") for c in candidates if c.correctness_passed
    )
    if candidates:
        if all_have_wall:
            best_c = min(candidates, key=lambda c: c.wall_ms)
            candidate_improved = best_c.wall_ms < best_wall_ms
        else:
            best_c = min(candidates, key=lambda c: c.latency_ns)
            candidate_improved = best_c.latency_ns < best_latency

        if candidate_improved:
            best_kernel = best_c.optimized_kernel
            best_latency = best_c.latency_ns
            best_wall_ms = best_c.wall_ms
            best_ncu_rep = best_c.ncu_rep_path
            current_ctx = best_c.profiling_context
            iter_result.best_candidate = best_c
            iter_result.improved = True
            improved = True

    # Update global best across all chains (same wall-clock preference as greedy selection)
    global_best_wall_ms = state.get("global_best_wall_ms", float("inf"))
    if all_have_wall:
        if best_wall_ms < global_best_wall_ms:
            global_best_kernel = best_kernel
            global_best_latency = best_latency
            global_best_wall_ms = best_wall_ms
    else:
        if best_latency < global_best_latency:
            global_best_kernel = best_kernel
            global_best_latency = best_latency

    if improved:
        no_improvement_streak = 0
    else:
        no_improvement_streak += 1

    iteration_results.append(iter_result)

    ui = get_pipeline_ui()
    if ui:
        ui.set_phase("Select best", "")

    # Print iteration table
    # Prefer wall-clock speedup for display when available (matches selection logic)
    _bl_wall = state.get("baseline_wall_ms", float("inf"))
    _use_wall = _bl_wall < float("inf") and all_have_wall
    if _use_wall:
        overall_speedup = round(_bl_wall / best_wall_ms, 2) if best_wall_ms > 0 else 1.0
        _metric_label = "wall"
    else:
        overall_speedup = round(baseline_latency / best_latency, 2)
        _metric_label = "ncu"
    t = Table(
        title=(
            f"Iteration {iteration + 1}/{cfg.T}  |  "
            f"Framework: {framework}  |  "
            f"Best: {overall_speedup}x ({_metric_label})"
        ),
        show_header=True,
    )
    t.add_column("Direction")
    t.add_column("Wall", justify="right")
    t.add_column("NCU", justify="right")
    t.add_column("Correctness")
    t.add_column("Kept?")
    for c in candidates:
        wall_str = (
            f"{c.wall_speedup:.2f}x"
            if c.wall_ms < float("inf") and c.wall_speedup > 0
            else "—"
        )
        ncu_str = (
            f"{c.speedup:.2f}x" if c.latency_ns < float("inf") else "fail"
        )
        correct = "passed" if c.correctness_passed else "failed"
        is_best = (
            iter_result.best_candidate
            and c.plan_letter == iter_result.best_candidate.plan_letter
        )
        t.add_row(c.direction, wall_str, ncu_str, correct, "Y" if is_best else "")

    (get_hud_log_console() or console).print(t)
    if ui and ui.is_hud_sink:
        summary_bits = [
            f"{c.plan_letter}: "
            f"{'%.2fx' % (c.wall_speedup if c.wall_ms < float('inf') and c.wall_speedup > 0 else c.speedup) if c.latency_ns < float('inf') else 'fail'}"
            for c in candidates
        ]
        ui.add_event(
            f"Iter {iteration + 1}/{cfg.T} cumulative {overall_speedup}x ({_metric_label}) | "
            + (" | ".join(summary_bits) if summary_bits else "no candidates")
        )
        ui.sync_iteration_header(
            state.get("iteration", 0),
            baseline_latency,
            best_latency,
        )

    # Search tree: log state
    try:
        from kernelopt.search_tree import SearchTree
        _tree = SearchTree(Path(state["output_dir"]) / "search_tree.json")
        _ts = _tree.stats()
        pipe_print(
            f"  [dim]Search tree: {_ts['total_nodes']} nodes, "
            f"{_ts['correct_nodes']} correct, "
            f"best {_ts['best_speedup']:.2f}x[/]"
        )
    except Exception:
        pass

    # --- Beam selection (multi-chain forest) ---
    new_active: list[dict] = []
    if beam_width > 1 and candidates:
        # Collect all viable candidates: correct + finite latency
        viable = [
            c for c in candidates
            if c.correctness_passed and c.latency_ns < float("inf")
        ]
        # Sort by latency (best first)
        viable.sort(key=lambda c: c.latency_ns)

        # Select top-B with diversity: avoid picking multiple candidates
        # from the same chain unless necessary
        seen_chains: set[str] = set()
        selected_beam: list[PlanCandidate] = []
        # First pass: one per chain (diversity)
        for c in viable:
            if len(selected_beam) >= beam_width:
                break
            cid = c.chain_id or "chain_0"
            if cid not in seen_chains:
                selected_beam.append(c)
                seen_chains.add(cid)
        # Second pass: fill remaining slots with best remaining
        for c in viable:
            if len(selected_beam) >= beam_width:
                break
            if c not in selected_beam:
                selected_beam.append(c)

        # If no viable candidates, carry forward previous active kernels
        if not selected_beam and prev_active:
            new_active = [ak for ak in prev_active if ak.get("alive", True)]
        else:
            for idx, c in enumerate(selected_beam):
                # Reuse chain_id if the candidate came from an existing chain,
                # otherwise assign a new one
                cid = c.chain_id or f"chain_{idx}"
                # Find parent chain to inherit history
                parent_ak = next(
                    (ak for ak in prev_active if ak["chain_id"] == cid),
                    None,
                )
                parent_history = parent_ak["history"] if parent_ak else []
                parent_depth = parent_ak["depth"] if parent_ak else 0

                new_active.append({
                    "chain_id": cid,
                    "kernel_source": c.optimized_kernel,
                    "latency_ns": c.latency_ns,
                    "wall_ms": c.wall_ms,
                    "speedup": c.speedup,
                    "profiling_context": c.profiling_context,
                    "ncu_summary": _build_ncu_summary(c.profiling_context),
                    "history": parent_history + [c.direction],
                    "depth": parent_depth + 1,
                    "alive": True,
                })
        if new_active:
            pipe_print(
                f"  [bold cyan]Beam: {len(new_active)} active chains "
                f"(best {new_active[0]['latency_ns']:.0f} ns)[/]"
            )
    else:
        # beam_width == 1: single-chain — update active_kernels to track
        # the current best, preserving the simple greedy behavior.
        parent_ak = prev_active[0] if prev_active else {}
        parent_history = parent_ak.get("history", [])
        parent_depth = parent_ak.get("depth", 0)
        best_direction = ""
        if improved and iter_result.best_candidate:
            best_direction = iter_result.best_candidate.direction
        new_active = [{
            "chain_id": "chain_0",
            "kernel_source": best_kernel,
            "latency_ns": best_latency,
            "wall_ms": best_wall_ms,
            "speedup": round(baseline_latency / best_latency, 4) if best_latency > 0 else 0.0,
            "profiling_context": current_ctx,
            "ncu_summary": _build_ncu_summary(current_ctx),
            "history": (parent_history + [best_direction]) if best_direction else parent_history,
            "depth": (parent_depth + 1) if improved else parent_depth,
            "alive": True,
        }]

        # Legacy search tree policy (only for beam_width=1)
        try:
            from kernelopt.search_tree import SearchTree
            _tree = SearchTree(Path(state["output_dir"]) / "search_tree.json")
            policy = cfg.search_policy if hasattr(cfg, "search_policy") else "greedy"
            if policy != "greedy":
                selected = _tree.select_nodes(policy, n=1)
                if selected and selected[0].kernel_source:
                    chosen = selected[0]
                    if chosen.id != (_tree.get_best() or chosen).id:
                        pipe_print(
                            f"  [bold cyan]Search policy '{policy}': expanding "
                            f"'{chosen.direction[:40]}' ({chosen.speedup:.2f}x) "
                            f"instead of global best[/]"
                        )
                        # Override best_kernel for next iteration's exploration
                        best_kernel = chosen.kernel_source
                        best_latency = chosen.latency_ns
                        # Keep active_kernels[0] consistent with best_kernel
                        new_active[0]["kernel_source"] = chosen.kernel_source
                        new_active[0]["latency_ns"] = chosen.latency_ns
                        new_active[0]["ncu_summary"] = chosen.ncu_summary if hasattr(chosen, "ncu_summary") else ""
        except Exception:
            pass

    # Build out dict with both legacy and multi-chain fields
    out = {
        # Legacy fields (backward compat)
        "best_kernel": best_kernel,
        "best_latency_ns": best_latency,
        "best_wall_ms": best_wall_ms,
        "best_ncu_rep": best_ncu_rep,
        "current_profiling_context": current_ctx,
        "no_improvement_streak": no_improvement_streak,
        "iteration_results": iteration_results,
        # Multi-chain forest fields
        "active_kernels": new_active,
        "global_best_kernel": global_best_kernel,
        "global_best_latency_ns": global_best_latency,
        "global_best_wall_ms": global_best_wall_ms,
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

    framework = state.get("framework", "")
    if framework == "pytorch":
        return "maybe_synthesize"
    if framework in ("auto", ""):
        from kernelopt.agents.profiler.ncu_runner import extract_kernel_info
        detected, _ = extract_kernel_info(state["kernel_path"])
        if detected == "pytorch":
            return "maybe_synthesize"
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
# Node 0 (conditional): Synthesize Triton from PyTorch via torch.compile
# ---------------------------------------------------------------------------

def maybe_synthesize(state: GraphState) -> dict:
    """If input is PyTorch, convert to Triton via torch.compile.

    Transparent to the rest of the pipeline -- after this node,
    kernel_path points to a Triton .py file regardless of input format.
    No-op when framework is not "pytorch".
    """
    framework = state.get("framework", "")
    if framework not in ("pytorch", "auto", ""):
        return {}

    if framework in ("auto", ""):
        from kernelopt.agents.profiler.ncu_runner import extract_kernel_info
        detected, _ = extract_kernel_info(state["kernel_path"])
        if detected != "pytorch":
            return {}

    from kernelopt.synthesis import extract_triton_from_pytorch, wrap_inductor_output
    from kernelopt.synthesis.compile_extract import SynthesisError

    kernel_path = state["kernel_path"]
    output_dir = state["output_dir"]

    pipe_print("[bold]Detected PyTorch input — synthesizing Triton via torch.compile...[/]")

    original_source = Path(kernel_path).read_text()

    try:
        cfg = state["config"]
        results = extract_triton_from_pytorch(
            kernel_path,
            entry_point=cfg.entry_point,
            input_shapes=cfg.input_shapes,
        )

        if not results:
            raise SynthesisError("torch.compile produced no output for this input.")

        primary = results[0]

        wrapped_source = wrap_inductor_output(
            inductor_source=primary["inductor_source"],
            original_pytorch_source=original_source,
            triton_kernel_sources=primary["triton_kernels"],
        )

        synth_dir = Path(output_dir) / "synthesis"
        synth_dir.mkdir(parents=True, exist_ok=True)

        wrapped_path = synth_dir / "synthesized_triton.py"
        wrapped_path.write_text(wrapped_source)

        (synth_dir / "original_pytorch.py").write_text(original_source)
        (synth_dir / "raw_inductor_output.py").write_text(primary["inductor_source"])

        n_kernels = len(primary["triton_kernels"])
        has_extern = "extern_kernels" in primary["inductor_source"]
        pipe_print(
            f"  Synthesized: [green]{wrapped_path}[/] "
            f"({n_kernels} Triton kernel{'s' if n_kernels != 1 else ''}"
            f"{', + extern_kernels' if has_extern else ''})"
        )

        # If 0 Triton kernels (all cuBLAS/cuDNN), retry with forced Triton
        # codegen so re-stitching has something to work with.
        if n_kernels == 0 and has_extern:
            pipe_print("  [yellow]No Triton kernels — retrying with forced-Triton codegen...[/]")
            try:
                results_ft = extract_triton_from_pytorch(
                    kernel_path,
                    entry_point=cfg.entry_point,
                    input_shapes=cfg.input_shapes,
                    force_triton=True,
                )
                if results_ft and results_ft[0]["triton_kernels"]:
                    primary = results_ft[0]
                    wrapped_source = wrap_inductor_output(
                        inductor_source=primary["inductor_source"],
                        original_pytorch_source=original_source,
                        triton_kernel_sources=primary["triton_kernels"],
                    )
                    wrapped_path.write_text(wrapped_source)
                    n_kernels = len(primary["triton_kernels"])
                    pipe_print(
                        f"  [green]Forced-Triton synthesis: {n_kernels} "
                        f"Triton kernel{'s' if n_kernels != 1 else ''}[/]"
                    )
            except Exception as e:
                pipe_print(f"  [dim]Forced-Triton retry failed: {e}[/]")

        # Use benchmark kernel files (correct by construction) if available,
        # otherwise fall back to the wrapped Inductor file.
        from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone
        benchmark_files = primary.get("benchmark_kernel_files", [])
        benchmark_kernel_src = None
        if benchmark_files:
            import re as _re
            # Extract kernel names from the synthesized output to filter
            # stale benchmark files from previous compilation runs.
            _synth_kernel_names = set(_re.findall(
                r'def\s+(triton_\w+)\s*\(', primary["inductor_source"],
            ))
            matching = [
                bk for bk in benchmark_files
                if any(kn in bk.get("source", "") for kn in _synth_kernel_names)
            ] if _synth_kernel_names else benchmark_files

            # Among matching files, prefer per-kernel benchmark files
            per_kernel = [
                bk for bk in matching
                if _re.search(r'return\s+\w+.*,\s*\d{2,}\s*,', bk.get("source", ""))
            ]
            candidates_bk = per_kernel if per_kernel else (matching if matching else benchmark_files)
            target_bk = max(candidates_bk, key=lambda b: len(b.get("source", "")))
            benchmark_kernel_src = target_bk["source"]
            standalone_kernel = build_benchmark_standalone(
                benchmark_source=benchmark_kernel_src,
                original_pytorch_source=original_source,
            )
            pipe_print(
                f"  Benchmark kernel: [green]{target_bk['name'][:50]}[/] "
                f"({len(benchmark_files)} total)"
            )
        else:
            standalone_kernel = wrapped_source
            pipe_print("  [dim]No Triton benchmark kernels — using wrapped Inductor file[/]")

        baseline_standalone_path = synth_dir / "baseline_standalone.py"
        baseline_standalone_path.write_text(standalone_kernel)

        return {
            "kernel_path": str(wrapped_path),
            "kernel_source": wrapped_source,
            "kernel_basename": "synthesized_triton",
            "framework": "triton",
            "ext": ".py",
            "best_kernel": standalone_kernel,
            "_original_pytorch_path": kernel_path,
            "_original_pytorch_source": original_source,
            "_benchmark_kernel_source": benchmark_kernel_src,
            "_all_benchmark_files": benchmark_files,
            "_baseline_standalone_path": str(baseline_standalone_path),
        }

    except Exception as e:
        pipe_print(
            f"  [yellow]Synthesis failed:[/] {e}\n"
            f"  [dim]Continuing with original PyTorch file (no conversion)[/]"
        )
        return {
            "_original_pytorch_path": kernel_path,
            "_original_pytorch_source": original_source,
        }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph optimization pipeline."""
    graph = StateGraph(GraphState)

    graph.add_node("maybe_synthesize", maybe_synthesize)
    graph.add_node("profile_baseline", profile_baseline)
    graph.add_node("analyze_strategy", analyze_strategy)
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
            "maybe_synthesize": "maybe_synthesize",
            "profile_baseline": "profile_baseline",
            "generate_plans": "generate_plans",
            "execute_plans": "execute_plans",
            "profile_candidates": "profile_candidates",
            "select_best": "select_best",
            "increment_iteration": "increment_iteration",
            "continuation_entry": "continuation_entry",
        },
    )
    graph.add_edge("maybe_synthesize", "profile_baseline")
    graph.add_edge("profile_baseline", "analyze_strategy")
    graph.add_edge("analyze_strategy", "generate_plans")
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
    if state.benchmark_kernel_source:
        initial["_benchmark_kernel_source"] = state.benchmark_kernel_source
    if hasattr(state, "_baseline_standalone_path") and state._baseline_standalone_path:
        initial["_baseline_standalone_path"] = state._baseline_standalone_path
    return initial


def run_langgraph_pipeline(state: PipelineState) -> PipelineState:
    """Run the LangGraph optimization pipeline."""
    from kernelopt.agents.profiler import get_latency_ns

    cfg = state.config

    # Ablation overrides
    if getattr(cfg, "no_memory", False):
        cfg.q_cap = 0
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
            baseline_standalone = resume_path / "synthesis" / "baseline_standalone.py"
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
                "_baseline_standalone_path": str(baseline_standalone) if baseline_standalone.exists() else None,
                # Multi-chain forest search initialization
                "active_kernels": [{
                    "chain_id": "chain_0",
                    "kernel_source": state.kernel_source,
                    "latency_ns": latency,
                    "wall_ms": float("inf"),
                    "speedup": 1.0,
                    "profiling_context": bctx,
                    "ncu_summary": _build_ncu_summary(bctx),
                    "history": [],
                    "depth": 0,
                    "alive": True,
                }],
                "global_best_kernel": state.kernel_source,
                "global_best_latency_ns": latency,
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

    # Map back to PipelineState — prefer global_best (actual best across all
    # chains and iterations) over best_kernel (which may be a SearchTree
    # exploration target, not the true optimum).
    state.best_kernel = final.get("global_best_kernel") or final["best_kernel"]
    state.best_latency_ns = final.get("global_best_latency_ns") or final["best_latency_ns"]
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
            "| Plan | Direction | Wall | NCU | Correctness | Kept? |\n",
            "|------|-----------|------|-----|-------------|-------|\n",
        ]
        for c in result.candidates:
            wall_str = (
                f"{c.wall_speedup:.3f}×"
                if c.wall_ms < float("inf") and c.wall_speedup > 0
                else "—"
            )
            ncu_str = f"{c.speedup:.3f}×" if c.latency_ns < float("inf") else "fail"
            correct = "✓" if c.correctness_passed else "✗"
            is_best = result.best_candidate and c.plan_letter == result.best_candidate.plan_letter
            kept    = "**✓ best**" if is_best else ""
            lines.append(
                f"| {c.plan_letter} | {c.direction} | {wall_str} | {ncu_str} | {correct} | {kept} |\n"
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
        f"- T={cfg.T}, N={cfg.N}, K={cfg.K}, keep_n={cfg.keep_n}, q_cap={cfg.q_cap}\n",
        f"- spe_gate={cfg.spe_gate}, slo_gate={cfg.slo_gate}\n",
        f"- replay_mode={cfg.replay_mode}\n",
        f"- kernel_args: `{' '.join(cfg.kernel_args) or '—'}`\n",
    ]

    log_path.write_text("".join(lines))
    pipe_print(f"\n  Log written to [cyan]{log_path}[/]")
