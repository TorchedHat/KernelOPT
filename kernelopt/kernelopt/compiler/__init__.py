"""Compiler-aware optimization for KernelOPT.

Extracts structural metadata from the GPU kernel compilation pipeline
(Dynamo, AOTAutograd, Inductor, Triton) and surfaces it to the LLM
planner and executor agents as verified ground-truth context.

Public API:
    extract_compiler_context(pytorch_path, ...) -> CompilerContext
    CompilerContext  — immutable dataclass holding all extracted data
"""

from __future__ import annotations

import logging
import traceback

from kernelopt.compiler.context import CompilerContext
from kernelopt.compiler.version_compat import detect_capabilities

__all__ = ["CompilerContext", "extract_compiler_context"]

logger = logging.getLogger(__name__)

_FRAMEWORK_EXTRACTION_MAP: dict[str, set[str]] = {
    "pytorch": {"inductor", "triton", "dynamo", "aot"},
    "triton": {"triton"},
    "helion": {"triton"},
    "jax": {"xla", "pallas"},
}


def extract_compiler_context(
    source_path: str,
    *,
    framework: str = "pytorch",
    entry_point: str | None = None,
    timeout: int = 300,
) -> CompilerContext:
    """Extract compiler context from a kernel source file.

    Routes extraction based on `framework`:
      - pytorch: Inductor IR + fusion + autotune + codegen + Triton listener
      - triton/helion: Triton compilation listener only (no torch.compile)
      - jax: XLA stages + fusion dumps + autotune protos

    Args:
        source_path: Path to .py file with the kernel/model.
        framework: One of "pytorch", "triton", "helion", "jax".
        entry_point: Explicit class/function name if ambiguous.
        timeout: Subprocess timeout in seconds.

    Returns:
        CompilerContext with extraction_success=True on success,
        or extraction_success=False with extraction_warnings on expected
        failures (timeout, subprocess crash, missing capabilities).
    """
    caps = detect_capabilities()

    applicable_layers = _FRAMEWORK_EXTRACTION_MAP.get(framework)
    if applicable_layers is None:
        logger.warning(
            "Unknown framework %r — skipping compiler extraction", framework
        )
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[f"Unknown framework: {framework!r}"],
        )

    if framework == "pytorch":
        if caps.get("pytorch_version") == "unknown":
            return CompilerContext(
                extraction_success=False,
                extraction_warnings=[
                    "PyTorch not available — cannot extract Inductor context. "
                    "Install with: pip install torch"
                ],
            )
        return _extract_pytorch(source_path, entry_point=entry_point, timeout=timeout)

    if framework in ("triton", "helion"):
        if caps.get("triton_version") == "unknown":
            return CompilerContext(
                extraction_success=False,
                extraction_warnings=[
                    "Triton not available — cannot extract compilation metadata. "
                    "Install with: pip install triton"
                ],
            )
        return _extract_triton_standalone(source_path, caps=caps)

    if framework == "jax":
        if caps.get("jax_version") == "unavailable":
            return CompilerContext(
                extraction_success=False,
                extraction_warnings=[
                    "JAX not available — cannot extract XLA context. "
                    "Install with: pip install jax[cuda12]"
                ],
            )
        return _extract_jax(source_path, entry_point=entry_point, timeout=timeout)

    return CompilerContext(
        extraction_success=False,
        extraction_warnings=[f"No extraction path for framework: {framework!r}"],
    )


def _extract_pytorch(
    pytorch_path: str,
    *,
    entry_point: str | None = None,
    timeout: int = 300,
) -> CompilerContext:
    """Full PyTorch extraction: Dynamo + AOT + Inductor + Triton listener."""
    try:
        from kernelopt.compiler.extraction import run_extraction_subprocess
        return run_extraction_subprocess(
            pytorch_path,
            entry_point=entry_point,
            timeout=timeout,
        )
    except Exception as e:
        logger.error("PyTorch extraction failed: %s", e)
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[f"Extraction failed: {e}"],
        )


def _extract_triton_standalone(
    source_path: str,
    *,
    caps: dict | None = None,
) -> CompilerContext:
    """Standalone Triton/Helion: extract only Triton compilation metadata.

    No torch.compile runs — the kernel IS the Triton kernel, so we only
    need the Triton listener to capture register pressure, shared mem, etc.
    """
    if caps is None:
        caps = detect_capabilities()

    if not caps.get("has_triton_compilation_listener"):
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                "Triton compilation listener not available in this Triton version. "
                f"Triton version: {caps.get('triton_version', 'unknown')}"
            ],
        )

    # For standalone Triton, we return a minimal context — the actual
    # compilation metadata is captured during profiling via the Triton listener.
    return CompilerContext(
        extraction_success=True,
        extraction_warnings=[
            "Standalone Triton kernel — Inductor layers not applicable"
        ],
    )


def _extract_jax(
    source_path: str,
    *,
    entry_point: str | None = None,
    timeout: int = 300,
) -> CompilerContext:
    """JAX extraction: XLA stages + fusion + autotune.

    Loads the source file, finds the JAX function to trace, then runs
    jax.jit → lower → compile to extract XLA metadata.
    """
    caps = detect_capabilities()

    if not caps.get("has_jax_stages_api"):
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                "JAX stages API not available — cannot lower JAX function. "
                f"JAX version: {caps.get('jax_version', 'unknown')}"
            ],
        )

    from kernelopt.compiler.context import XLAContext

    import importlib.util
    import sys
    from pathlib import Path

    module_path = Path(source_path)
    if not module_path.exists():
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[f"JAX source file not found: {source_path}"],
        )

    try:
        spec = importlib.util.spec_from_file_location("_jax_module", str(module_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_jax_module"] = mod
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("_jax_module", None)
        logger.error(
            "Failed to load JAX source file:\n%s", traceback.format_exc()
        )
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                f"Failed to load JAX source: {traceback.format_exc()[-500:]}"
            ],
        )

    fn_name = entry_point or "forward"
    fn = getattr(mod, fn_name, None)
    get_inputs = getattr(mod, "get_inputs", None)

    if fn is None or not callable(fn):
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                f"No callable '{fn_name}' found in {source_path}. "
                "Set --entry-point to the JAX function name."
            ],
        )

    if get_inputs is None:
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                f"No get_inputs() function in {source_path}. "
                "Required to provide example inputs for JAX tracing."
            ],
        )

    try:
        inputs = get_inputs()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
    except Exception:
        logger.error(
            "get_inputs() failed:\n%s", traceback.format_exc()
        )
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                f"get_inputs() raised: {traceback.format_exc()[-500:]}"
            ],
        )

    from kernelopt.compiler.extraction.xla.stages import extract_xla_stages

    try:
        xla_data = extract_xla_stages(fn, *inputs)
    except Exception:
        logger.error(
            "XLA extraction failed unexpectedly:\n%s", traceback.format_exc()
        )
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                f"XLA extraction raised: {traceback.format_exc()[-500:]}"
            ],
        )

    if "error" in xla_data:
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[f"XLA extraction error: {xla_data['error']}"],
        )

    xla_ctx = XLAContext(
        stablehlo_text=xla_data.get("stablehlo_text", ""),
        optimized_hlo_text=xla_data.get("optimized_hlo_text", ""),
        cost_analysis=xla_data.get("cost_analysis", {}),
        memory_analysis=xla_data.get("memory_analysis", {}),
    )

    return CompilerContext(
        extraction_success=True,
        extraction_warnings=[],
        xla_context=xla_ctx,
    )
