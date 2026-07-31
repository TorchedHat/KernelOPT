"""Extract IR metadata from the post-grad Inductor graph.

Walks GraphLowering.name_to_buffer after compilation to extract
shapes, dtypes, reduction parameters, and buffer dependencies.

Runs INSIDE the extraction subprocess (not the main KernelOPT process).

References:
    - Buffer class: torch/_inductor/ir.py line 5160
    - ComputedBuffer: torch/_inductor/ir.py line 5408
    - Reduction: torch/_inductor/ir.py line 1384
    - Pointwise: torch/_inductor/ir.py line 1219
    - GraphLowering.name_to_buffer: torch/_inductor/graph.py line 535
"""

from __future__ import annotations


def extract_ir_nodes(graph_lowering) -> list[dict]:
    """Walk GraphLowering.name_to_buffer and extract metadata.

    Args:
        graph_lowering: The torch._inductor.graph.GraphLowering instance
            available as V.graph during compilation.

    Returns:
        List of serializable dicts, each representing one IRNodeInfo.
    """
    try:
        from torch._inductor import ir as _ir
    except ImportError:
        return []

    results = []
    name_to_buffer = getattr(graph_lowering, "name_to_buffer", {})

    for buf_name, buf in name_to_buffer.items():
        if not isinstance(buf, _ir.ComputedBuffer):
            continue

        data = buf.data
        node_type = type(data).__name__

        ranges_raw = getattr(data, "ranges", [])
        iteration_ranges = _resolve_ranges(ranges_raw)
        if iteration_ranges is None:
            continue

        total_elements = 1
        for r in iteration_ranges:
            total_elements *= r

        try:
            layout = buf.layout
            dtype_str = str(layout.dtype)
            bytes_per_element = layout.dtype.itemsize
        except (AttributeError, TypeError):
            dtype_str = "unknown"
            bytes_per_element = 4

        num_reads = 0
        num_writes = 0
        read_names: tuple[str, ...] = ()
        write_names = (buf_name,)

        try:
            rw = buf.get_read_writes() if hasattr(buf, "get_read_writes") else None
            if rw:
                num_reads = len(rw.reads)
                num_writes = len(rw.writes)
                read_names = tuple(
                    dep.name for dep in rw.reads if hasattr(dep, "name")
                )
        except Exception:
            pass

        reduction_info = None
        if isinstance(data, _ir.Reduction):
            reduction_info = _extract_reduction(data, dtype_str)

        results.append({
            "buffer_name": buf_name,
            "node_type": node_type,
            "iteration_ranges": tuple(iteration_ranges),
            "total_elements": total_elements,
            "dtype": dtype_str,
            "bytes_per_element": bytes_per_element,
            "total_memory_bytes": total_elements * bytes_per_element,
            "num_reads": num_reads,
            "num_writes": num_writes,
            "read_buffer_names": read_names,
            "write_buffer_names": write_names,
            "reduction": reduction_info,
        })

    return results


def _resolve_ranges(ranges_raw) -> list[int] | None:
    """Convert symbolic/integer ranges to concrete ints. Returns None if any are unresolvable."""
    try:
        import sympy
    except ImportError:
        sympy = None

    resolved = []
    for r in ranges_raw:
        if isinstance(r, int):
            resolved.append(r)
        elif sympy and isinstance(r, sympy.Integer):
            resolved.append(int(r))
        else:
            try:
                resolved.append(int(r))
            except (TypeError, ValueError):
                return None
    return resolved


def _extract_reduction(data, dst_dtype: str) -> dict | None:
    """Extract reduction-specific metadata from an ir.Reduction node."""
    red_ranges_raw = getattr(data, "reduction_ranges", [])
    red_ranges = _resolve_ranges(list(red_ranges_raw))
    if red_ranges is None:
        return None

    rnumel = 1
    for r in red_ranges:
        rnumel *= r

    rnumel_padded = 1
    while rnumel_padded < rnumel:
        rnumel_padded *= 2
    wasted = (rnumel_padded - rnumel) / rnumel_padded if rnumel_padded > 0 else 0.0

    reduction_type = str(getattr(data, "reduction_type", "unknown"))
    reduction_hint_raw = getattr(data, "reduction_hint", "DEFAULT")
    reduction_hint = str(reduction_hint_raw).split(".")[-1]

    src_dtype = str(getattr(data, "src_dtype", "unknown"))

    return {
        "reduction_type": reduction_type,
        "reduction_ranges": tuple(red_ranges),
        "reduction_hint": reduction_hint,
        "src_dtype": src_dtype,
        "dst_dtype": dst_dtype,
        "is_persistent": False,
        "is_cooperative": False,
        "rnumel_padded": rnumel_padded,
        "wasted_sram_fraction": round(wasted, 4),
    }
