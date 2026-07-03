"""Reference kernel database -- shipped with kernelopt.

Contains key code patterns from official Triton tutorials and known-good
optimization techniques. Used by the strategy analysis and planner to
provide concrete examples when proposing optimizations.

These are NOT starter kernels (we don't force the LLM to start from them).
They're reference material -- the LLM sees them as examples of techniques
that are known to work.
"""

from __future__ import annotations

from pathlib import Path

_REFERENCE_DIR = Path(__file__).parent.parent.parent / "tests_suite" / "benchmarks" / "triton_tutorials"


def get_reference_kernel(technique: str) -> str | None:
    """Get a reference kernel code snippet for a given technique.

    Args:
        technique: One of "matmul", "fused_softmax", "layer_norm",
                   "fused_attention", "persistent_matmul", "vector_add"

    Returns:
        The relevant code snippet, or None if not found.
    """
    mapping = {
        "matmul": "03-matrix-multiplication.py",
        "fused_softmax": "02-fused-softmax.py",
        "layer_norm": "05-layer-norm.py",
        "fused_attention": "06-fused-attention.py",
        "persistent_matmul": "09-persistent-matmul.py",
        "vector_add": "01-vector-add.py",
        "dropout": "04-low-memory-dropout.py",
        "grouped_gemm": "08-grouped-gemm.py",
    }

    filename = mapping.get(technique)
    if not filename:
        return None

    filepath = _REFERENCE_DIR / filename
    if not filepath.exists():
        return None

    try:
        source = filepath.read_text()
        return _extract_kernel_function(source)
    except Exception:
        return None


def get_relevant_references(kernel_structure: dict) -> list[tuple[str, str]]:
    """Get reference kernels relevant to a kernel structure.

    Args:
        kernel_structure: dict from classify_kernel_structure()

    Returns:
        List of (technique_name, code_snippet) tuples.
    """
    kernel_type = kernel_structure.get("type", "")
    references = []

    if kernel_type in ("gemm", "gemm_epilogue"):
        ref = get_reference_kernel("matmul")
        if ref:
            references.append(("triton_tutorial_matmul", ref))

    if kernel_type in ("softmax",):
        ref = get_reference_kernel("fused_softmax")
        if ref:
            references.append(("triton_tutorial_fused_softmax", ref))

    if kernel_type in ("layernorm", "rmsnorm"):
        ref = get_reference_kernel("layer_norm")
        if ref:
            references.append(("triton_tutorial_layer_norm", ref))

    if kernel_type in ("attention",):
        ref = get_reference_kernel("fused_attention")
        if ref:
            references.append(("triton_tutorial_fused_attention", ref))

    if kernel_type in ("pointwise",):
        ref = get_reference_kernel("vector_add")
        if ref:
            references.append(("triton_tutorial_vector_add", ref))

    return references


def _extract_kernel_function(source: str) -> str:
    """Extract the @triton.jit kernel from a tutorial file (first 100 lines)."""
    lines = source.split("\n")
    in_kernel = False
    kernel_lines = []

    for line in lines:
        if "@triton.jit" in line:
            in_kernel = True
            kernel_lines.append(line)
            continue
        if in_kernel:
            if line.strip() == "" and kernel_lines and not kernel_lines[-1].strip().startswith("def"):
                if len(kernel_lines) > 5:
                    break
            kernel_lines.append(line)
            if len(kernel_lines) > 80:
                break

    return "\n".join(kernel_lines) if kernel_lines else ""
