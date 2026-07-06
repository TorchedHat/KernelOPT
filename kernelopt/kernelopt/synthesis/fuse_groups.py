"""Merge kernel groups that share intermediate buffers into fused groups.

Takes the kernel groups from graph_tracer.trace_model() and merges adjacent
groups connected by intermediate buffers that are not model outputs.
The result is a smaller set of fused groups, each containing multiple
ATen ops that the LLM should implement in a single Triton kernel.
"""
from __future__ import annotations

from kernelopt.constants import MAX_GROUPS_FOR_SINGLE_FUSE, MAX_SOURCE_CHARS


def merge_fusible_groups(kernel_groups: list[dict]) -> list[dict]:
    """Merge kernel groups into fused groups based on data dependencies.

    Strategy:
    - If all kernel groups come from the same ATen graph (same model),
      merge them all into one group — let the LLM write one kernel
      that does the entire computation.
    - For very large graphs (>6 kernels), keep GEMM/template kernels
      separate from pointwise chains to avoid overly complex kernels.

    Args:
        kernel_groups: List of dicts from trace_model(), each with:
            kernel_name, kernel_type, fused_aten_ops, aten_graph,
            model_inputs, inductor_source

    Returns:
        List of merged group dicts, potentially fewer than input.
        Each merged group has:
            - kernel_name: combined name
            - kernel_type: "fused" if merged, else original type
            - fused_aten_ops: combined list of all ops
            - aten_graph: the full computation graph
            - model_inputs: model input specs
            - inductor_source: combined reference source
            - sub_kernels: list of original kernel names that were merged
    """
    if not kernel_groups:
        return []

    if len(kernel_groups) == 1:
        kg = kernel_groups[0]
        kg["sub_kernels"] = [kg["kernel_name"]]
        return [kg]

    # Collect all ATen ops across all groups
    all_aten_ops = []
    all_source_ops = []
    all_inductor_src = []
    sub_kernels = []
    kernel_types = set()

    for kg in kernel_groups:
        all_aten_ops.extend(kg.get("fused_aten_ops", []))
        all_source_ops.extend(kg.get("source_ops", []))
        if kg.get("inductor_source"):
            all_inductor_src.append(kg["inductor_source"])
        sub_kernels.append(kg["kernel_name"])
        kernel_types.add(kg["kernel_type"])

    # Use the ATen graph from the first group (they all share the same model)
    aten_graph = kernel_groups[0].get("aten_graph", [])
    model_inputs = kernel_groups[0].get("model_inputs", [])

    # Decision: merge all into one if small enough
    if len(kernel_groups) <= MAX_GROUPS_FOR_SINGLE_FUSE:
        # Merge everything into one fused kernel
        aten_ops_from_graph = [op["name"] for op in aten_graph] if aten_graph else all_aten_ops
        return [{
            "kernel_name": "fused_" + "_".join(sub_kernels[0].split("_")[2:4]),
            "kernel_type": "fused",
            "fused_aten_ops": aten_ops_from_graph or all_aten_ops,
            "source_ops": all_source_ops,
            "aten_graph": aten_graph,
            "model_inputs": model_inputs,
            "inductor_source": "\n\n".join(all_inductor_src)[:MAX_SOURCE_CHARS],
            "full_inductor_source": kernel_groups[0].get("full_inductor_source", ""),
            "sub_kernels": sub_kernels,
        }]

    # For larger graphs: keep template/GEMM kernels separate,
    # merge adjacent pointwise chains
    merged = []
    current_chain: list[dict] = []

    for kg in kernel_groups:
        if kg["kernel_type"] in ("template", "extern"):
            # Flush any accumulated pointwise chain
            if current_chain:
                merged.append(_merge_chain(current_chain, aten_graph, model_inputs))
                current_chain = []
            # Keep template as standalone
            kg["sub_kernels"] = [kg["kernel_name"]]
            merged.append(kg)
        else:
            current_chain.append(kg)

    if current_chain:
        merged.append(_merge_chain(current_chain, aten_graph, model_inputs))

    return merged


def _merge_chain(
    chain: list[dict],
    aten_graph: list[dict],
    model_inputs: list[dict],
) -> dict:
    """Merge a chain of pointwise/reduction kernels into one fused group."""
    if len(chain) == 1:
        chain[0]["sub_kernels"] = [chain[0]["kernel_name"]]
        return chain[0]

    all_ops = []
    all_src = []
    names = []
    for kg in chain:
        all_ops.extend(kg.get("fused_aten_ops", []))
        if kg.get("inductor_source"):
            all_src.append(kg["inductor_source"])
        names.append(kg["kernel_name"])

    return {
        "kernel_name": "fused_" + "_".join(names[0].split("_")[2:4]),
        "kernel_type": "fused",
        "fused_aten_ops": all_ops,
        "source_ops": [],
        "aten_graph": aten_graph,
        "model_inputs": model_inputs,
        "inductor_source": "\n\n".join(all_src)[:MAX_SOURCE_CHARS],
        "full_inductor_source": chain[0].get("full_inductor_source", ""),
        "sub_kernels": names,
    }
