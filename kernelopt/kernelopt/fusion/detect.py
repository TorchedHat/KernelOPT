"""Detect fusible kernel groups from a buffer dependency graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from kernelopt.fusion.graph import BufferGraph, BufferNode, KernelNode


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FusionGroup:
    group_id: str
    fusion_type: str  # "pointwise_chain" | "gemm_prologue" | "gemm_epilogue" | "full_gemm_fusion" | "reduction_prologue"
    kernels: list[KernelNode] = field(default_factory=list)
    kernel_sources: dict[str, str] = field(default_factory=dict)
    eliminated_buffers: list[BufferNode] = field(default_factory=list)
    input_buffers: list[str] = field(default_factory=list)
    output_buffers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fusion-type classification
# ---------------------------------------------------------------------------

# Maps (producer_type, consumer_type) -> fusion_type
_PAIR_FUSION_MAP: dict[tuple[str, str], str] = {
    ("poi", "poi"): "pointwise_chain",
    ("poi", "tem"): "gemm_prologue",
    ("tem", "poi"): "gemm_epilogue",
    ("poi", "red"): "reduction_prologue",
    ("poi", "per"): "reduction_prologue",
}


def _chain_type(kernel_types: list[str]) -> str:
    """Classify a multi-kernel chain by its type sequence."""
    if kernel_types == ["poi", "tem", "poi"]:
        return "full_gemm_fusion"
    # Homogeneous pointwise chains
    if all(t == "poi" for t in kernel_types):
        return "pointwise_chain"
    # Fallback: use first-pair type
    if len(kernel_types) >= 2:
        return _PAIR_FUSION_MAP.get(
            (kernel_types[0], kernel_types[1]), "pointwise_chain"
        )
    return "pointwise_chain"


# ---------------------------------------------------------------------------
# Helper: build producer / consumer maps
# ---------------------------------------------------------------------------


def _build_buf_maps(
    graph: BufferGraph,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (producer_map, consumer_map): buffer_name -> kernel_name."""
    producer_of: dict[str, str] = {}  # buf -> kernel that writes it
    consumer_of: dict[str, str] = {}  # buf -> kernel that reads it (unique)

    for knode in graph.kernels:
        for buf in knode.writes:
            producer_of[buf] = knode.name
        for buf in knode.reads:
            # Only track as consumer if the buffer is an intermediate (not model input)
            if buf not in graph.model_inputs:
                # If multiple kernels read the same buffer, mark as non-unique
                if buf in consumer_of:
                    consumer_of[buf] = "__multiple__"
                else:
                    consumer_of[buf] = knode.name

    return producer_of, consumer_of


def _fusible_buffers(graph: BufferGraph) -> set[str]:
    """Return set of buffers that can be fused away (1 producer, 1 consumer, not output)."""
    producer_of, consumer_of = _build_buf_maps(graph)
    output_set = set(graph.model_outputs)

    fusible: set[str] = set()
    for buf_name in graph.buffers:
        if buf_name in output_set:
            continue
        has_single_producer = buf_name in producer_of
        has_single_consumer = (
            buf_name in consumer_of and consumer_of[buf_name] != "__multiple__"
        )
        if has_single_producer and has_single_consumer:
            fusible.add(buf_name)

    return fusible


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_fusion_groups(
    graph: BufferGraph, kernel_sources: dict[str, str]
) -> list[FusionGroup]:
    """Detect fusible kernel groups from a buffer dependency graph.

    Parameters
    ----------
    graph:
        Parsed buffer dependency graph from ``parse_call_graph``.
    kernel_sources:
        Mapping from kernel name to its Triton source code.

    Returns
    -------
    list[FusionGroup]
        Each group contains a chain of kernels that can be fused, ordered by
        execution order, along with metadata about eliminated buffers and
        external I/O.
    """
    fusible = _fusible_buffers(graph)
    if not fusible:
        return []

    producer_of, consumer_of = _build_buf_maps(graph)

    # Build a name->KernelNode map
    kernel_map: dict[str, KernelNode] = {k.name: k for k in graph.kernels}

    # Build adjacency: kernel -> list of (buffer, consumer) pairs.
    # A kernel may produce multiple fusible buffers consumed by different kernels.
    next_kernel: dict[str, list[tuple[str, str]]] = {}
    prev_kernel: dict[str, str] = {}  # consumer -> producer

    for buf_name in fusible:
        prod = producer_of[buf_name]
        cons = consumer_of[buf_name]
        next_kernel.setdefault(prod, []).append((buf_name, cons))
        prev_kernel[cons] = prod

    # Find chain starts: kernels that are not a consumer in any fusible link
    chain_starts: list[str] = []
    for kname in [k.name for k in graph.kernels]:
        if kname not in prev_kernel:
            if kname in next_kernel:
                chain_starts.append(kname)

    # DFS traversal: enumerate all root-to-leaf paths through the fusion graph.
    # When a kernel has multiple consumers, each path becomes a separate FusionGroup.
    def _enumerate_chains(
        start: str,
    ) -> list[tuple[list[KernelNode], list[str]]]:
        results: list[tuple[list[KernelNode], list[str]]] = []

        def dfs(current: str, chain_kernels: list[KernelNode], chain_buffers: list[str]):
            if current not in next_kernel:
                # Leaf — path is complete
                if len(chain_kernels) >= 2:
                    results.append((list(chain_kernels), list(chain_buffers)))
                return
            for buf, nxt in next_kernel[current]:
                chain_kernels.append(kernel_map[nxt])
                chain_buffers.append(buf)
                dfs(nxt, chain_kernels, chain_buffers)
                chain_kernels.pop()
                chain_buffers.pop()

        dfs(start, [kernel_map[start]], [])
        return results

    groups: list[FusionGroup] = []
    group_counter = 0

    for start in chain_starts:
        all_chains = _enumerate_chains(start)
        for chain_kernels, chain_buffers in all_chains:
            # Eliminated buffers = the fusible intermediate buffers
            eliminated_buf_nodes = [graph.buffers[b] for b in chain_buffers]

            # Determine fusion type
            k_types = [k.kernel_type for k in chain_kernels]
            if len(k_types) == 2:
                ftype = _PAIR_FUSION_MAP.get(
                    (k_types[0], k_types[1]), "pointwise_chain"
                )
            else:
                ftype = _chain_type(k_types)

            # External inputs: reads that are not eliminated intermediate buffers
            eliminated_set = set(chain_buffers)
            seen_inputs: list[str] = []
            seen_input_set: set[str] = set()
            for k in chain_kernels:
                for r in k.reads:
                    if r not in eliminated_set and r not in seen_input_set:
                        seen_inputs.append(r)
                        seen_input_set.add(r)

            # External outputs: writes that are not eliminated
            seen_outputs: list[str] = []
            seen_output_set: set[str] = set()
            for k in chain_kernels:
                for w in k.writes:
                    if w not in eliminated_set and w not in seen_output_set:
                        seen_outputs.append(w)
                        seen_output_set.add(w)

            grp_sources = {k.name: kernel_sources.get(k.name, "") for k in chain_kernels}

            group = FusionGroup(
                group_id=f"fusion_group_{group_counter}",
                fusion_type=ftype,
                kernels=chain_kernels,
                kernel_sources=grp_sources,
                eliminated_buffers=eliminated_buf_nodes,
                input_buffers=seen_inputs,
                output_buffers=seen_outputs,
            )
            groups.append(group)
            group_counter += 1

    return groups
