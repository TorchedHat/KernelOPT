# kernelopt/kernelopt/agents/profiler/__init__.py
"""Profiler Agent package.

Exports:
    run_profiler_agent      — profile a kernel, return (profiling_context, ncu_config)
    profiler_subgraph       — compiled single-instance sub-graph (for testing/reuse)

Re-exports for backward compatibility (used by cli.py):
    run_ncu, build_ncu_command, extract_kernel_info
    build_profiling_context, get_latency_ns
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from kernelopt.agents.profiler.context_builder import (
    build_profiling_context,
    get_latency_ns,
)
from kernelopt.agents.profiler.graph import profiler_subgraph
from kernelopt.agents.profiler.ncu_runner import (
    _build_kernel_name_filter,
    build_ncu_command,
    extract_kernel_info,
    run_ncu,
)
from kernelopt.agents.profiler.prompts import build_initial_message
from kernelopt.agents.profiler.tools import ProfilerToolServer
from kernelopt.functions.csv_parser import parse_ncu_csv


def run_profiler_agent(
    kernel_path: str,
    output_basename: str,
    *,
    ncu_config: dict | None = None,
    pipeline_config,                    # PipelineConfig — imported lazily to avoid circular
    kernel_id: int = 0,
    llm_config: dict,
    llm_override=None,
    binary_path: str | None = None,     # For .cu kernels: path to compiled binary
) -> tuple[dict, dict]:
    """Profile a kernel with NCU and return (profiling_context, ncu_config).

    Phase 1 — resolve NCU config (one of three paths):
      a) Full bypass: ncu_config is not None → reuse (candidate re-profiling)
      b) User bypass: kernel_name + launch_count both set in pipeline_config
      c) LLM sub-graph: reads source, decides kernel_name / launch_count / kernel_args

    Phase 2 — deterministic: run NCU, export CSV, parse, build profiling_context.

    Args:
        binary_path: For .cu kernels, path to the pre-compiled binary that NCU should execute.
                     If None, binary is assumed to be at kernel_path with extension stripped.

    Returns (profiling_context, resolved_ncu_config).
    """
    resolved = _resolve_ncu_config(
        kernel_path, ncu_config, pipeline_config, llm_config, llm_override
    )

    csv_path, ncu_rep_path, _ = run_ncu(
        kernel_path,
        output_basename,
        kernel_name=resolved["kernel_name"],  # "" means no filter, None means auto-detect
        launch_skip=resolved["launch_skip"],
        launch_count=resolved["launch_count"],
        replay_mode=resolved["replay_mode"],
        kernel_args=resolved["kernel_args"] or None,
        kernel_id=kernel_id,
        save_profiling_context=False,   # we build it below
        binary_path=binary_path,
    )

    sections, rules, launch = parse_ncu_csv(csv_path, kernel_id)
    ctx = build_profiling_context(sections, rules, launch, kernel_path, kernel_id=kernel_id)

    # Save context JSON for inspection
    ctx_json_path = f"{output_basename}_profiling_context.json"
    Path(ctx_json_path).write_text(json.dumps(ctx, indent=2))

    return ctx, resolved


def _resolve_ncu_config(
    kernel_path: str,
    ncu_config: dict | None,
    pipeline_config,
    llm_config: dict,
    llm_override,
) -> dict:
    """Determine the NCU config through bypass, user-provided, or LLM sub-graph."""

    # (a) Full bypass — reuse existing config (candidate re-profiling)
    if ncu_config is not None:
        return ncu_config

    # (b) User bypass — kernel_name provided (launch_count optional).
    # When kernel_name is given without launch_count, use None (no limit) so NCU
    # profiles all matching kernels regardless of how many non-matching launches
    # precede the target (e.g. torch.randn before a Triton kernel).
    if pipeline_config.kernel_name:
        import sys as _sys
        _sys.stdout.write("[profiler] user bypass — skipping LLM (kernel_name provided)\n")
        _sys.stdout.flush()

        # Detect if CUDA file has main() — need application mode for standalone binaries
        replay_mode = pipeline_config.replay_mode
        if kernel_path.endswith(".cu"):
            kernel_source = Path(kernel_path).read_text()
            if "int main(" in kernel_source or "void main(" in kernel_source:
                print(f"[DEBUG] Detected CUDA binary with main(), forcing replay_mode='application'")
                replay_mode = "application"

        return {
            "kernel_name": pipeline_config.kernel_name,
            "launch_skip": pipeline_config.launch_skip,
            "launch_count": pipeline_config.launch_count,  # None = no limit
            "kernel_args": list(pipeline_config.kernel_args),
            "replay_mode": replay_mode,
        }

    # (c) LLM sub-graph
    import sys as _sys
    _sys.stdout.write("[profiler] running LLM config-decider...\n")
    _sys.stdout.flush()
    kernel_source = Path(kernel_path).read_text()
    server = ProfilerToolServer(kernel_path, kernel_source)
    profile_hints = getattr(pipeline_config, "profile_hints", "")
    initial_msg = build_initial_message(kernel_path, profile_hints)

    config = RunnableConfig(configurable={
        "tool_server": server,
        "llm_config": llm_config,
        "llm": llm_override,
    })

    result = profiler_subgraph.invoke(
        {"messages": [initial_msg], "ncu_config": None},
        config=config,
    )

    llm_result = result.get("ncu_config")

    # Detect framework ONCE before any logic
    framework, detected_names = extract_kernel_info(kernel_path)

    if llm_result is not None:
        # Honor explicit user-provided replay_mode over LLM decision.
        # Default is "kernel"; if user changed it, they know better.
        if pipeline_config.replay_mode != "kernel":
            llm_result = {**llm_result, "replay_mode": pipeline_config.replay_mode}

        # Force kernel_name to be empty for Triton/Helion frameworks.
        # JIT-compiled kernels have mangled names that don't match Python function names.
        if framework in ("triton", "helion"):
            print(f"[DEBUG] Detected {framework} kernel, forcing kernel_name='' (was: {llm_result.get('kernel_name')})")
            llm_result = {**llm_result, "kernel_name": ""}

        # Force application mode for CUDA files with main() — kernel mode doesn't work with standalone binaries.
        if framework == "cuda" and kernel_path.endswith(".cu"):
            kernel_source = Path(kernel_path).read_text()
            if "int main(" in kernel_source or "void main(" in kernel_source:
                print(f"[DEBUG] Detected CUDA binary with main(), forcing replay_mode='application' (was: {llm_result.get('replay_mode')})")
                llm_result = {**llm_result, "replay_mode": "application"}

        print(f"[DEBUG] Final LLM config: {llm_result}")
        return llm_result

    # Fallback — deterministic extraction if LLM never called submit_ncu_config.
    # For CUDA, function names match NCU's demangled names directly.
    # For Triton/Helion, JIT compilers mangle names — don't filter by name.
    kernel_name_filter = _build_kernel_name_filter(detected_names) if framework == "cuda" else None

    # Detect if CUDA file has main() — need application mode for standalone binaries
    replay_mode = pipeline_config.replay_mode
    if framework == "cuda" and kernel_path.endswith(".cu"):
        kernel_source = Path(kernel_path).read_text()
        if "int main(" in kernel_source or "void main(" in kernel_source:
            print(f"[DEBUG] Detected CUDA binary with main(), forcing replay_mode='application'")
            replay_mode = "application"

    fallback_config = {
        "kernel_name": kernel_name_filter or "",
        "launch_skip": pipeline_config.launch_skip,
        "launch_count": pipeline_config.launch_count or 1,
        "kernel_args": list(pipeline_config.kernel_args),
        "replay_mode": replay_mode,
    }
    print(f"[DEBUG] Using fallback config (LLM didn't call submit_ncu_config): {fallback_config}")
    return fallback_config


__all__ = [
    "run_profiler_agent",
    "profiler_subgraph",
    # backward compat
    "run_ncu",
    "build_ncu_command",
    "extract_kernel_info",
    "build_profiling_context",
    "get_latency_ns",
]
