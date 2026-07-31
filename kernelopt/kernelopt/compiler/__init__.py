"""Compiler-aware optimization for KernelOPT.

Extracts structural metadata from the GPU kernel compilation pipeline
(Dynamo, AOTAutograd, Inductor, Triton) and surfaces it to the LLM
planner and executor agents as verified ground-truth context.

Public API:
    extract_compiler_context(pytorch_path, ...) -> CompilerContext
    CompilerContext  — immutable dataclass holding all extracted data
"""

from kernelopt.compiler.context import CompilerContext

__all__ = ["CompilerContext", "extract_compiler_context"]


def extract_compiler_context(
    pytorch_path: str,
    *,
    entry_point: str | None = None,
    timeout: int = 300,
) -> CompilerContext:
    """Extract compiler context from a PyTorch model file.

    Runs torch.compile in a subprocess with extraction hooks installed,
    then parses the output into a CompilerContext.

    Args:
        pytorch_path: Path to .py file with nn.Module or callable.
        entry_point: Explicit class/function name if ambiguous.
        timeout: Subprocess timeout in seconds.

    Returns:
        CompilerContext with extraction_success=True on success,
        or extraction_success=False with warnings on failure.
        Never raises — the pipeline proceeds without context on error.
    """
    try:
        from kernelopt.compiler.extraction import run_extraction_subprocess
        return run_extraction_subprocess(
            pytorch_path,
            entry_point=entry_point,
            timeout=timeout,
        )
    except Exception as e:
        return CompilerContext(
            pytorch_version="unknown",
            extraction_success=False,
            extraction_warnings=[f"Extraction failed: {e}"],
        )
