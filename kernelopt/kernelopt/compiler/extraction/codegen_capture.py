"""Extract codegen strategy metadata from generated Triton source code.

Uses regex parsing of the generated code rather than hooking into
TritonKernel internals. This is MORE STABLE than hooking codegen because
it depends only on the output format (which changes less often than
internal class fields).

Key patterns:
    triton_per_* -> persistent reduction
    triton_red_* -> streaming reduction (may be cooperative)
    triton_poi_* -> pointwise
    triton_tem_* -> template (GEMM, conv)
    triton_for_* -> foreach
    triton_spl_* -> split_scan
    RSPLIT in config -> cooperative reduction
    tl.constexpr in signature -> compile-time constant blocks

References:
    - Kernel naming: torch/_inductor/codegen/triton.py line 8051
    - Category mapping: torch/_inductor/wrapper_benchmark.py lines 21-28
    - triton_heuristics decorator: torch/_inductor/runtime/triton_heuristics.py
"""

from __future__ import annotations

import re

from kernelopt.compiler.context import CodegenStrategy


_CATEGORY_MAP = {
    "per": "persistent_reduction",
    "red": "streaming_reduction",
    "poi": "pointwise",
    "tem": "template",
    "for": "foreach",
    "spl": "split_scan",
    "unk": "unknown",
}


def extract_codegen_strategies(generated_code: str) -> list[CodegenStrategy]:
    """Parse generated Triton code to determine codegen strategies.

    Args:
        generated_code: Full Inductor output (the string from run_and_get_code).

    Returns:
        List of CodegenStrategy (one per kernel found in the source).
    """
    if not generated_code:
        return []

    strategies = []
    tq = "'''"
    pattern = rf"(triton_\w+)\s*=\s*async_compile\.triton\([^,]*,\s*{tq}(.*?){tq}"

    for match in re.finditer(pattern, generated_code, re.DOTALL):
        kernel_name = match.group(1)
        kernel_source = match.group(2)

        prefix = kernel_name.split("_")[1] if "_" in kernel_name else "unk"
        category = _CATEGORY_MAP.get(prefix[:3], "unknown")

        persistent = category == "persistent_reduction"
        cooperative = (
            ("RSPLIT" in kernel_source or "rsplit" in kernel_source)
            and prefix[:3] == "red"
        )
        if cooperative:
            category = "cooperative_reduction"

        no_x_dim = "XBLOCK" not in kernel_source
        if not no_x_dim:
            no_x_dim = bool(
                re.search(r"XBLOCK\s*:\s*tl\.constexpr\s*=\s*1\b", kernel_source)
            )

        uses_block_pointers = (
            "tl.make_block_ptr" in kernel_source or "block_ptr" in kernel_source
        )

        block_sizes: dict[str, int] = {}
        for bm in re.finditer(
            r'"(XBLOCK|RBLOCK|YBLOCK|BLOCK_M|BLOCK_N|BLOCK_K)"\s*:\s*(\d+)',
            kernel_source,
        ):
            block_sizes[bm.group(1)] = int(bm.group(2))
        for bm in re.finditer(
            r"(XBLOCK|RBLOCK|YBLOCK)\s*:\s*tl\.constexpr\s*=\s*(\d+)",
            kernel_source,
        ):
            block_sizes[bm.group(1)] = int(bm.group(2))

        nw_match = re.search(r"num_warps\s*=\s*(\d+)", kernel_source)
        num_warps = int(nw_match.group(1)) if nw_match else 4

        ns_match = re.search(r"num_stages\s*=\s*(\d+)", kernel_source)
        num_stages = int(ns_match.group(1)) if ns_match else 1

        hint_match = re.search(
            r"reduction_hint\s*=\s*ReductionHint\.(\w+)", kernel_source
        )
        reduction_hint = hint_match.group(1) if hint_match else "N/A"

        num_red_dims = len(re.findall(r"r\d+_numel|rnumel", kernel_source[:500]))
        if not num_red_dims and (persistent or cooperative or prefix[:3] == "red"):
            num_red_dims = 1

        strategies.append(CodegenStrategy(
            kernel_name=kernel_name,
            kernel_category=category,
            persistent_reduction=persistent,
            cooperative_reduction=cooperative,
            no_x_dim=no_x_dim,
            uses_block_pointers=uses_block_pointers,
            num_reduction_dims=num_red_dims,
            reduction_hint=reduction_hint,
            inductor_block_sizes=block_sizes,
            inductor_num_warps=num_warps,
            inductor_num_stages=num_stages,
        ))

    return strategies
