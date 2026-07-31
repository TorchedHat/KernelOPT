"""Build compiler context sections for the Planner agent prompt.

Converts CompilerContext into concise natural-language sections that
give the planner structural facts about the kernel. All output is
markdown-style text — no JSON reaches the LLM.

Sections are ordered by actionability:
1. Structural constraints (what NOT to do)
2. Original op identity (what the kernel IS)
3. Decomposition path (how it was lowered)
4. Codegen strategy (how Inductor compiled it)
5. Triton compilation metadata (register pressure, shared memory)
6. Fusion opportunities (near-threshold rejections)
"""

from __future__ import annotations

from typing import Optional


def build_compiler_context_section(
    compiler_context_json: Optional[str],
    kernel_name: str = "",
) -> str:
    """Build the compiler context prompt section for a specific kernel.

    Returns empty string if no compiler context is available.
    The output is plain text (not JSON) designed for token efficiency.
    """
    if not compiler_context_json:
        return ""

    try:
        from kernelopt.compiler.context import CompilerContext
        ctx = CompilerContext.from_json(compiler_context_json)
    except Exception:
        return ""

    if not ctx.extraction_success:
        return ""

    sections: list[str] = []

    _add_constraints(sections, ctx, kernel_name)
    _add_original_ops(sections, ctx)
    _add_decomposition_paths(sections, ctx)
    _add_codegen_strategy(sections, ctx, kernel_name)
    _add_triton_metadata(sections, ctx, kernel_name)
    _add_fusion_opportunities(sections, ctx)

    if not sections:
        return ""

    return (
        "\n\nCOMPILER CONTEXT (verified ground truth from torch.compile):\n"
        + "\n".join(sections)
        + "\n"
    )


def _add_constraints(sections: list[str], ctx, kernel_name: str) -> None:
    """Add structural constraints (DO NOT list)."""
    constraints = ctx.get_structural_constraints(kernel_name)
    if not constraints:
        return
    lines = ["CONSTRAINTS — do NOT suggest:"]
    for c in constraints:
        lines.append(f"  * {c}")
    sections.append("\n".join(lines))


def _add_original_ops(sections: list[str], ctx) -> None:
    """Add pre-decomposition op identity from Dynamo."""
    if ctx.dynamo is None or not ctx.dynamo.original_ops:
        return

    compound = [op for op in ctx.dynamo.original_ops if op.is_compound]
    if not compound:
        return

    lines = ["ORIGINAL OPERATIONS (pre-decomposition):"]
    for op in compound:
        shapes = " x ".join(str(s) for s in op.input_shapes) if op.input_shapes else ""
        lines.append(f"  {op.op_name} ({shapes})")
    sections.append("\n".join(lines))


def _add_decomposition_paths(sections: list[str], ctx) -> None:
    """Add AOTAutograd decomposition paths."""
    if not ctx.aot_decomps:
        return

    lines = ["DECOMPOSITION PATHS (how torch.compile lowered compound ops):"]
    for d in ctx.aot_decomps:
        intermediates = f", {d.num_intermediates} intermediate buffers" if d.num_intermediates > 0 else ""
        lines.append(
            f"  {d.original_op} -> {' + '.join(d.decomposed_into)}{intermediates}"
        )
    sections.append("\n".join(lines))


def _add_codegen_strategy(sections: list[str], ctx, kernel_name: str) -> None:
    """Add Inductor codegen strategy metadata."""
    strat = ctx.get_codegen_strategy(kernel_name)
    if strat is None:
        return

    blocks = ", ".join(f"{k}={v}" for k, v in strat.inductor_block_sizes.items())
    lines = [
        "CODEGEN STRATEGY (Inductor's decisions):",
        f"  Category: {strat.kernel_category}",
        f"  Block sizes: {blocks or 'N/A'}",
        f"  num_warps: {strat.inductor_num_warps}, num_stages: {strat.inductor_num_stages}",
    ]
    if strat.persistent_reduction:
        lines.append("  Reduction: persistent (RBLOCK is compile-time constant)")
    if strat.cooperative_reduction:
        lines.append("  Reduction: cooperative (RSPLIT across CTAs)")
    if strat.uses_block_pointers:
        lines.append("  Memory: block_ptr / TMA hardware loads")
    if strat.reduction_hint != "N/A":
        lines.append(f"  Reduction hint: {strat.reduction_hint}")

    sections.append("\n".join(lines))


def _add_triton_metadata(sections: list[str], ctx, kernel_name: str) -> None:
    """Add Triton compilation metadata (registers, shared memory)."""
    tri = ctx.get_triton_info(kernel_name)
    if tri is None:
        return

    lines = ["TRITON COMPILATION (hardware counters):"]
    lines.append(f"  Warps: {tri.num_warps}, Stages: {tri.num_stages}")
    lines.append(f"  Shared memory: {tri.shared_bytes} bytes")

    if tri.n_regs > 0:
        lines.append(f"  Registers/thread: {tri.n_regs}")
    if tri.n_spills > 0:
        lines.append(f"  Register spills: {tri.n_spills} (reduces performance)")

    if tri.dot_count > 0:
        lines.append(f"  Matrix ops: {tri.dot_count} tt.dot")
    if tri.load_count > 0 or tri.store_count > 0:
        lines.append(f"  Memory ops: {tri.load_count} loads, {tri.store_count} stores")
    if tri.has_async_copy:
        lines.append("  Uses async copy (software pipelining active)")
    if tri.has_warp_specialize:
        lines.append("  Uses warp specialization")
    if tri.mma_type:
        lines.append(f"  MMA instruction: {tri.mma_type}")

    sections.append("\n".join(lines))


def _add_fusion_opportunities(sections: list[str], ctx) -> None:
    """Add near-threshold fusion rejections the planner could address."""
    opportunities = ctx.get_fusion_opportunities()
    if not opportunities:
        return

    lines = ["FUSION OPPORTUNITIES (Inductor rejected these — consider manual fusion):"]
    for opp in opportunities[:5]:
        lines.append(f"  {opp.node1_name} <-> {opp.node2_name}: \"{opp.reason}\"")

    sections.append("\n".join(lines))
