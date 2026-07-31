"""Compiler context extraction entry point.

Orchestrates all extraction modules (Dynamo, AOT, Inductor, Triton)
into a single subprocess call that piggybacks on torch.compile.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kernelopt.compiler.context import (
    AOTDecompInfo,
    CodegenStrategy,
    CompilerContext,
    DynamoGraphInfo,
    DynamoOpInfo,
    FusionGroup,
    FusionRejection,
    GraphBreakInfo,
    IRNodeInfo,
    IntermediateBuffer,
    ReductionInfo,
    TritonCompilationInfo,
)


def run_extraction_subprocess(
    pytorch_path: str,
    *,
    entry_point: str | None = None,
    timeout: int = 300,
) -> CompilerContext:
    """Run compiler context extraction in an isolated subprocess.

    The subprocess runs torch.compile with all extraction hooks installed,
    then serializes the extracted data as JSON on stdout.
    """
    from kernelopt.compiler.extraction.subprocess_script import (
        build_extraction_script,
    )

    script = build_extraction_script(pytorch_path, entry_point=entry_point)

    python = sys.executable
    pipeline_venv = Path(__file__).resolve().parents[3] / ".pipeline_venv" / "bin" / "python"
    if pipeline_venv.exists():
        python = str(pipeline_venv)

    import os
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)

    try:
        proc = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[f"Extraction timed out after {timeout}s"],
        )

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-1000:]
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[
                f"Subprocess failed (exit {proc.returncode}): {stderr_tail}"
            ],
        )

    return _parse_output(proc.stdout)


def _parse_output(stdout: str) -> CompilerContext:
    """Parse the subprocess JSON output into a CompilerContext."""
    start_marker = "__COMPILER_CONTEXT_START__"
    end_marker = "__COMPILER_CONTEXT_END__"

    start_idx = stdout.find(start_marker)
    end_idx = stdout.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=["No compiler context markers in subprocess output"],
        )

    json_str = stdout[start_idx + len(start_marker):end_idx].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return CompilerContext(
            extraction_success=False,
            extraction_warnings=[f"JSON parse error: {e}"],
        )

    return _build_context(data)


def _build_context(data: dict) -> CompilerContext:
    """Construct CompilerContext from raw extraction data."""
    warnings = data.get("extraction_warnings", [])

    dynamo = None
    dynamo_data = data.get("dynamo_info")
    if dynamo_data:
        dynamo = DynamoGraphInfo(
            original_ops=tuple(
                DynamoOpInfo(**op) for op in dynamo_data.get("original_ops", [])
            ),
            graph_breaks=tuple(
                GraphBreakInfo(**gb) for gb in dynamo_data.get("graph_breaks", [])
            ),
            num_subgraphs=dynamo_data.get("num_subgraphs", 1),
            unsupported_ops=tuple(dynamo_data.get("unsupported_ops", [])),
        )

    aot_decomps = [
        AOTDecompInfo(**ad) for ad in data.get("aot_decomps", [])
    ]

    ir_nodes = []
    for n in data.get("ir_nodes", []):
        red = None
        if n.get("reduction"):
            red = ReductionInfo(**n["reduction"])
        n_clean = {k: v for k, v in n.items() if k != "reduction"}
        ir_nodes.append(IRNodeInfo(reduction=red, **n_clean))

    fusion_data = data.get("fusion_data", {})
    fusion_groups = []
    for fg in fusion_data.get("fusion_groups", []):
        fg_norm = {
            "kernel_name": fg.get("kernel_name", fg.get("name", "")),
            "constituent_buffers": tuple(fg.get("constituent_buffers", fg.get("snodes", []))),
            "constituent_ops": tuple(fg.get("constituent_ops", ())),
            "shared_bytes": fg.get("shared_bytes", 0),
        }
        fusion_groups.append(FusionGroup(**fg_norm))
    fusion_rejections = []
    for fr in fusion_data.get("fusion_rejections", []):
        fr_norm = {
            "node1_name": fr.get("node1_name", fr.get("node1", "")),
            "node2_name": fr.get("node2_name", fr.get("node2", "")),
            "reason": fr.get("reason", ""),
            "near_threshold": fr.get("near_threshold", False),
            "fixable_by_manual_fusion": fr.get("fixable_by_manual_fusion", False),
        }
        fusion_rejections.append(FusionRejection(**fr_norm))
    intermediate_buffers = [
        IntermediateBuffer(**ib) for ib in fusion_data.get("intermediate_buffers", [])
    ]

    codegen = [
        CodegenStrategy(**cs) for cs in data.get("codegen_strategies", [])
    ]

    triton_info = [
        TritonCompilationInfo(**ti) for ti in data.get("triton_info", [])
    ]

    return CompilerContext(
        pytorch_version=data.get("pytorch_version", "unknown"),
        extraction_success=True,
        extraction_warnings=warnings,
        dynamo=dynamo,
        aot_decomps=aot_decomps,
        ir_nodes=ir_nodes,
        fusion_groups=fusion_groups,
        fusion_rejections=fusion_rejections,
        intermediate_buffers=intermediate_buffers,
        codegen_strategies=codegen,
        triton_info=triton_info,
    )
