"""Build compiler context sections for the Executor agent prompt.

Provides structural constraints and autotune avoid-lists that prevent
the executor from generating code that violates compiler invariants.
All output is plain text — no JSON reaches the LLM.
"""

from __future__ import annotations

from typing import Optional


def build_executor_compiler_section(
    compiler_context_json: Optional[str],
    kernel_name: str = "",
) -> str:
    """Build compiler constraints section for the executor prompt.

    Returns empty string if no compiler context is available.
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

    parts: list[str] = []

    constraints = ctx.get_structural_constraints(kernel_name)
    if constraints:
        lines = ["COMPILER CONSTRAINTS (violations will produce incorrect code):"]
        for c in constraints:
            lines.append(f"  * {c}")
        parts.append("\n".join(lines))

    avoid = ctx.get_autotune_avoid_list(kernel_name)
    if avoid:
        lines = ["AUTOTUNE AVOID LIST (Inductor proved these configs slow):"]
        for cfg in avoid[:10]:
            lines.append(f"  - {str(cfg)[:100]}")
        parts.append("\n".join(lines))

    tri = ctx.get_triton_info(kernel_name)
    if tri:
        warnings = []
        if tri.n_regs > 128:
            warnings.append(
                f"HIGH REGISTER PRESSURE: {tri.n_regs} regs/thread — "
                "prioritize reducing registers before adding computation"
            )
        if tri.n_spills > 0:
            warnings.append(
                f"REGISTER SPILLS: {tri.n_spills} spills detected — "
                "reducing register pressure is the top priority"
            )
        if tri.shared_bytes > 49152:
            warnings.append(
                f"HIGH SHARED MEMORY: {tri.shared_bytes} bytes — "
                "close to 48KB L1 limit, avoid increasing shared memory usage"
            )
        if warnings:
            parts.append("HARDWARE WARNINGS:\n" + "\n".join(f"  ! {w}" for w in warnings))

    _add_xla_autotune_constraints(parts, ctx)

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts) + "\n"


def _add_xla_autotune_constraints(parts: list[str], ctx) -> None:
    """Add XLA autotune results as reference configs for the executor."""
    if ctx.xla_context is None:
        return

    xla = ctx.xla_context
    if not xla.autotune_results:
        return

    lines = ["XLA AUTOTUNE REFERENCE (XLA's best configs — use as starting points):"]
    for at in xla.autotune_results[:5]:
        lines.append(
            f"  - {at.fusion_name}: "
            f"block_m={at.block_m}, block_n={at.block_n}, block_k={at.block_k}, "
            f"warps={at.num_warps}, stages={at.num_stages}"
        )
    parts.append("\n".join(lines))
