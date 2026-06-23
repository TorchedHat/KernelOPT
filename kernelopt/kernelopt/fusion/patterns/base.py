"""Shared utilities for fusion pattern matching."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

from kernelopt.fusion.graph import (
    BufferGraph,
    BufferNode,
    ExternKernelNode,
    KernelNode,
)


def get_buffer_size_bytes(buf: BufferNode) -> int:
    """Compute buffer size in bytes from shape and dtype."""
    if not buf.shape:
        return 0
    numel = 1
    for d in buf.shape:
        numel *= d
    dtype_bytes = {
        "torch.float32": 4, "torch.float": 4,
        "torch.float16": 2, "torch.half": 2,
        "torch.bfloat16": 2,
        "torch.float64": 8, "torch.double": 8,
        "torch.int32": 4, "torch.int": 4,
        "torch.int64": 8, "torch.long": 8,
        "torch.int16": 2, "torch.short": 2,
        "torch.int8": 1, "torch.uint8": 1,
        "torch.bool": 1,
    }
    return numel * dtype_bytes.get(buf.dtype, 4)


def single_consumer(
    buf_name: str, graph: BufferGraph
) -> Optional[str]:
    """Return the sole consumer kernel name of buf_name, or None if 0 or 2+."""
    consumers = [c for (p, b, c) in graph.edges if b == buf_name]
    # Also check via aliases
    for alias, original in graph.buf_aliases.items():
        if original == buf_name:
            consumers.extend(c for (p, b, c) in graph.edges if b == alias)
    if len(consumers) == 1:
        return consumers[0]
    return None


def single_producer(
    buf_name: str, graph: BufferGraph
) -> Optional[str]:
    """Return the sole producer kernel name of buf_name, or None."""
    # Check direct
    producers = [p for (p, b, c) in graph.edges if b == buf_name]
    # Check via aliases
    resolved = graph.buf_aliases.get(buf_name, buf_name)
    if resolved != buf_name:
        producers.extend(p for (p, b, c) in graph.edges if b == resolved)
    if len(set(producers)) == 1:
        return producers[0]
    return None


def get_node_by_name(
    name: str, graph: BufferGraph
) -> Optional[Union[KernelNode, ExternKernelNode]]:
    """Find a node (Triton or extern) by name."""
    for k in graph.kernels:
        if k.name == name:
            return k
    for e in graph.extern_kernels:
        if e.name == name:
            return e
    return None


def kernel_name_contains(
    node: Union[KernelNode, ExternKernelNode], *substrings: str
) -> bool:
    """Check if kernel name contains any of the given substrings."""
    return any(s in node.name for s in substrings)


def collect_external_io(
    nodes: List[Union[KernelNode, ExternKernelNode]],
    eliminated_bufs: Set[str],
    graph: BufferGraph,
) -> Tuple[List[str], List[str]]:
    """Compute external inputs and outputs for a set of fused nodes.

    External inputs: buffers read by any node that are NOT in eliminated_bufs
    and NOT produced by another node in the set.

    External outputs: buffers written by any node that are NOT in eliminated_bufs.
    """
    node_names = {n.name for n in nodes}
    inputs: List[str] = []
    outputs: List[str] = []
    seen_inputs: Set[str] = set()

    for node in nodes:
        for buf in node.reads:
            resolved = graph.buf_aliases.get(buf, buf)
            if resolved in eliminated_bufs or buf in eliminated_bufs:
                continue
            # Check if produced by another node in the set
            producer = None
            for p, b, c in graph.edges:
                if b in (buf, resolved) and c == node.name and p in node_names:
                    producer = p
                    break
            if producer is None and buf not in seen_inputs:
                inputs.append(buf)
                seen_inputs.add(buf)

        for buf in node.writes:
            if buf not in eliminated_bufs:
                outputs.append(buf)

    return inputs, outputs


def is_model_output(buf_name: str, graph: BufferGraph) -> bool:
    """Check if a buffer is a model output."""
    resolved = graph.buf_aliases.get(buf_name, buf_name)
    return buf_name in graph.model_outputs or resolved in graph.model_outputs
