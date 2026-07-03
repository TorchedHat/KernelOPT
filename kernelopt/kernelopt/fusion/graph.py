"""Buffer dependency graph parser for Inductor synthesized model call() methods.

Parses the synthesized ``call()`` body to produce a DAG where nodes are Triton
kernels **and** extern_kernels calls, and edges are intermediate buffers flowing
between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass 
class BufferNode:
    """An intermediate (or output) tensor buffer."""

    name: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str  # e.g. "torch.float32"


@dataclass
class KernelNode:
    """A single Triton kernel invocation."""

    name: str
    kernel_type: str  # "poi" | "tem" | "per" | "red" | "unknown"
    reads: List[str] = field(default_factory=list)   # buffer/arg names this kernel reads
    writes: List[str] = field(default_factory=list)  # buffer names this kernel writes
    scalar_args: List[str] = field(default_factory=list)


@dataclass
class ExternKernelNode:
    """An extern_kernels call (cuBLAS mm, cuDNN convolution, etc.)."""

    name: str              # "extern_conv_0", "extern_mm_1"
    op: str                # "convolution" | "mm" | "addmm" | "bmm" | "bias_addmm"
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    raw_line: str = ""     # full source line for restitch
    scalar_args: List[str] = field(default_factory=list)
    reinterpret_tensors: Dict[str, str] = field(default_factory=dict)


@dataclass
class BufferGraph:
    """DAG of kernels connected by intermediate buffers.

    edges: list of (producer_kernel_name, buffer_name, consumer_kernel_name)
    """

    kernels: List[KernelNode]
    buffers: Dict[str, BufferNode]          # name -> BufferNode
    edges: List[Tuple[str, str, str]]       # (producer, buffer, consumer)
    model_inputs: Set[str]                  # arg names unpacked from `args`
    model_outputs: List[str]                # buffer names in the return tuple
    extern_kernels: List[ExternKernelNode] = field(default_factory=list)
    node_order: List[str] = field(default_factory=list)  # all nodes in execution order
    buf_aliases: Dict[str, str] = field(default_factory=dict)  # alias -> original


# ---------------------------------------------------------------------------
# Kernel-type classifier
# ---------------------------------------------------------------------------

_TYPE_PATTERNS = [
    ("poi", re.compile(r"_poi_")),
    ("tem", re.compile(r"_tem_")),
    ("per", re.compile(r"_per_")),
    ("red", re.compile(r"_red_")),
]


def _classify_kernel(name: str) -> str:
    for kind, pat in _TYPE_PATTERNS:
        if pat.search(name):
            return kind
    return "unknown"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# buf0 = empty_strided_cuda((16, 64, ...), (67..., ...), torch.float32)
_RE_BUFFER_ALLOC = re.compile(
    r"(\w+)\s*=\s*empty_strided_cuda\(\s*"
    r"\(([^)]+)\)"        # shape tuple
    r"\s*,\s*"
    r"\(([^)]+)\)"        # strides tuple
    r"\s*,\s*"
    r"(torch\.\w+)"       # dtype
    r"\s*\)"
)

# triton_poi_fused_convolution_0.run(arg1_1, buf0, 1024, ..., stream=raw_stream0)
_RE_KERNEL_CALL = re.compile(
    r"(triton_\w+)\.run\(([^)]+?),\s*stream="
)

# del buf0
_RE_DEL = re.compile(r"\bdel\s+(\w+)")

# return (buf2, )  or  return (buf2,)
_RE_RETURN = re.compile(r"\breturn\s*\(([^)]+)\)")

# arg0_1, arg1_1 = args
_RE_ARGS_UNPACK = re.compile(r"((?:arg\w+(?:,\s*)?)+)\s*=\s*args\b")

# buf1 = buf0; del buf0  # reuse
_RE_BUF_ALIAS = re.compile(
    r"(\w+)\s*=\s*(buf\w+)\s*;\s*del\s+\w+\s*#\s*reuse"
)

# buf6 = reinterpret_tensor(buf5, (10, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf5  # reuse
_RE_REINTERPRET_ALIAS = re.compile(
    r"(\w+)\s*=\s*reinterpret_tensor\((\w+),\s*"
)


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    """Convert a comma-separated integer string like '16, 64, 1024' to a tuple."""
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _is_buffer_token(token: str) -> bool:
    """True if this call argument token looks like a buffer or model-input arg."""
    t = token.strip()
    return t.startswith("buf") or t.startswith("arg")


# ---------------------------------------------------------------------------
# Extern kernels parser
# ---------------------------------------------------------------------------


def _extract_base_buffer(arg: str) -> str:
    """Extract base buffer name from reinterpret_tensor(buf5, ...) or buf5.reshape(...)."""
    arg = arg.strip()
    m = re.match(r"reinterpret_tensor\((\w+)", arg)
    if m:
        return m.group(1)
    m = re.match(r"(\w+)\.reshape\(", arg)
    if m:
        return m.group(1)
    # Just a plain token
    return arg.split("(")[0].strip()


def _parse_extern_kernel_line(
    line: str, extern_counter: Dict[str, int]
) -> Optional[ExternKernelNode]:
    """Parse an extern_kernels.op(...) call line.

    Handles forms:
      buf2 = extern_kernels.convolution(buf0, buf1, stride=(...), ...)
      extern_kernels.mm(arg2_1, arg0_1, out=buf0)
    """
    stripped = line.strip()

    # Check for extern_kernels call
    lhs_match = re.match(r"(\w+)\s*=\s*extern_kernels\.(\w+)\(", stripped)
    no_lhs_match = re.match(r"extern_kernels\.(\w+)\(", stripped)

    if not lhs_match and not no_lhs_match:
        return None

    if lhs_match:
        write_buf = lhs_match.group(1)
        op = lhs_match.group(2)
        args_start = lhs_match.end() - 1  # position of '('
    else:
        write_buf = None
        op = no_lhs_match.group(1)
        args_start = no_lhs_match.end() - 1

    # Balanced-paren scan to extract full argument string
    depth = 0
    args_end = args_start
    for i in range(args_start, len(stripped)):
        if stripped[i] == '(':
            depth += 1
        elif stripped[i] == ')':
            depth -= 1
            if depth == 0:
                args_end = i
                break

    full_args = stripped[args_start + 1:args_end]

    # Parse arguments: split on commas at depth 0
    args_tokens = []
    current = []
    depth = 0
    for ch in full_args:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            args_tokens.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args_tokens.append(''.join(current).strip())

    # Separate positional args from keyword args
    reads = []
    scalar_args = []
    reinterpret_tensors: Dict[str, str] = {}

    for tok in args_tokens:
        # Check for out= keyword
        if tok.strip().startswith("out="):
            write_buf = tok.strip()[4:].strip()
            continue
        # Check for other keyword args (stride=, padding=, etc.)
        if re.match(r"\w+=", tok.strip()) and not tok.strip().startswith("reinterpret_tensor"):
            scalar_args.append(tok.strip())
            continue
        # Positional arg — extract base buffer name
        base = _extract_base_buffer(tok)
        if _is_buffer_token(base):
            reads.append(base)
            # Track reinterpret_tensor wrapping
            if "reinterpret_tensor" in tok:
                reinterpret_tensors[base] = tok.strip()
        else:
            scalar_args.append(tok.strip())

    if write_buf is None:
        return None  # Can't determine output

    # Generate unique name
    count = extern_counter.get(op, 0)
    extern_counter[op] = count + 1
    name = f"extern_{op}_{count}"

    return ExternKernelNode(
        name=name,
        op=op,
        reads=reads,
        writes=[write_buf],
        raw_line=stripped,
        scalar_args=scalar_args,
        reinterpret_tensors=reinterpret_tensors,
    )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_call_graph(source: str) -> BufferGraph:
    """Parse an Inductor synthesized call() body and return a BufferGraph.

    Parameters
    ----------
    source:
        Full source text containing the synthesized model class/function,
        including the ``call()`` method.

    Returns
    -------
    BufferGraph
        A DAG whose nodes are :class:`KernelNode` and :class:`ExternKernelNode`
        objects, with edges as ``(producer, buffer, consumer)`` triples.
    """

    buffers: Dict[str, BufferNode] = {}
    kernels: List[KernelNode] = []
    extern_kernels_list: List[ExternKernelNode] = []
    edges: List[Tuple[str, str, str]] = []

    model_inputs: Set[str] = set()
    model_outputs: List[str] = []
    node_order: List[str] = []
    buf_aliases: Dict[str, str] = {}   # alias -> original

    written_by: Dict[str, str] = {}   # buffer_name -> kernel_name that first wrote it
    extern_counter: Dict[str, int] = {}  # op -> count for unique naming

    # Pass 1: collect model inputs
    m = _RE_ARGS_UNPACK.search(source)
    if m:
        raw = m.group(1)
        for token in raw.split(","):
            name = token.strip()
            if name:
                model_inputs.add(name)

    # Pass 2: sequential walk of the source lines to respect ordering
    for line in source.splitlines():
        line_stripped = line.strip()

        # Buffer allocation
        m = _RE_BUFFER_ALLOC.search(line_stripped)
        if m:
            bname, shape_str, stride_str, dtype = m.groups()
            buffers[bname] = BufferNode(
                name=bname,
                shape=_parse_int_tuple(shape_str),
                strides=_parse_int_tuple(stride_str),
                dtype=dtype,
            )
            continue

        # Buffer alias: buf1 = buf0; del buf0  # reuse
        m = _RE_BUF_ALIAS.match(line_stripped)
        if m:
            alias, original = m.group(1), m.group(2)
            buf_aliases[alias] = original
            # The alias inherits the written_by from the original
            if original in written_by:
                written_by[alias] = written_by[original]
            continue

        # reinterpret_tensor alias: buf6 = reinterpret_tensor(buf5, ...); del buf5
        m = _RE_REINTERPRET_ALIAS.match(line_stripped)
        if m and "# reuse" in line_stripped:
            alias, original = m.group(1), m.group(2)
            buf_aliases[alias] = original
            if original in written_by:
                written_by[alias] = written_by[original]
            continue

        # Extern kernel call
        if "extern_kernels." in line_stripped:
            enode = _parse_extern_kernel_line(line_stripped, extern_counter)
            if enode:
                extern_kernels_list.append(enode)
                node_order.append(enode.name)

                # Track writes
                for buf in enode.writes:
                    written_by[buf] = enode.name
                    # Create a synthetic BufferNode if not already allocated
                    if buf not in buffers:
                        buffers[buf] = BufferNode(
                            name=buf, shape=(), strides=(), dtype="unknown"
                        )

                # Build edges: reads from prior producers
                for buf in enode.reads:
                    # Resolve alias
                    resolved = buf_aliases.get(buf, buf)
                    if resolved in written_by:
                        producer = written_by[resolved]
                        edges.append((producer, resolved, enode.name))
                    elif buf in written_by:
                        producer = written_by[buf]
                        edges.append((producer, buf, enode.name))
                continue

        # Triton kernel call
        m = _RE_KERNEL_CALL.search(line_stripped)
        if m:
            kname, args_str = m.groups()
            ktype = _classify_kernel(kname)

            raw_args = [t.strip() for t in args_str.split(",") if t.strip()]
            reads: List[str] = []
            writes: List[str] = []
            scalars: List[str] = []

            for token in raw_args:
                if not _is_buffer_token(token):
                    scalars.append(token)
                    continue
                # Resolve alias
                resolved = buf_aliases.get(token, token)
                # Heuristic: if it is a known allocated buffer that has NOT
                # yet been written by any kernel, this kernel is its first
                # writer (i.e. it writes to it).  Otherwise it's a read.
                if resolved in buffers and resolved not in written_by:
                    writes.append(resolved)
                    written_by[resolved] = kname
                elif token in buffers and token not in written_by:
                    writes.append(token)
                    written_by[token] = kname
                else:
                    reads.append(token)

            kernel = KernelNode(
                name=kname,
                kernel_type=ktype,
                reads=reads,
                writes=writes,
                scalar_args=scalars,
            )
            kernels.append(kernel)
            node_order.append(kname)

            # Build edges: for every buffer this kernel reads that was
            # produced by a prior kernel, create an edge.
            for buf in reads:
                resolved = buf_aliases.get(buf, buf)
                if resolved in written_by:
                    producer = written_by[resolved]
                    edges.append((producer, resolved, kname))
                elif buf in written_by:
                    producer = written_by[buf]
                    edges.append((producer, buf, kname))
            continue

        # Return statement
        m = _RE_RETURN.search(line_stripped)
        if m:
            raw = m.group(1)
            for token in raw.split(","):
                name = token.strip()
                if name:
                    model_outputs.append(name)
            continue

    return BufferGraph(
        kernels=kernels,
        buffers=buffers,
        edges=edges,
        model_inputs=model_inputs,
        model_outputs=model_outputs,
        extern_kernels=extern_kernels_list,
        node_order=node_order,
        buf_aliases=buf_aliases,
    )
