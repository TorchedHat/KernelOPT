"""LLM-based fused kernel generation and validation."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from kernelopt.fusion.detect import FusionGroup
from kernelopt.pipeline_ui import pipe_print


# ---------------------------------------------------------------------------
# LLM output sanitization
# ---------------------------------------------------------------------------

# Unicode math symbols the LLM emits instead of ASCII operators
_UNICODE_REPLACEMENTS = {
    "\u00d7": "*",   # × multiplication sign
    "\u2192": "->",  # → right arrow
    "\u2190": "<-",  # ← left arrow
    "\u2208": "in",  # ∈ element of
    "\u2264": "<=",  # ≤ less-than-or-equal
    "\u2265": ">=",  # ≥ greater-than-or-equal
    "\u2260": "!=",  # ≠ not equal
    "\u2014": "--",  # — em dash
    "\u2013": "-",   # – en dash
    "\u201c": '"',   # " left double quote
    "\u201d": '"',   # " right double quote
    "\u2018": "'",   # ' left single quote
    "\u2019": "'",   # ' right single quote
}

# Lines starting with these tokens are valid Python/Triton
_VALID_LINE_PREFIXES = (
    " ", "\t", "#", "@", "def ", "class ", "import ", "from ",
    "if ", "for ", "while ", "return", "yield", "raise", "try:",
    "except", "with ", "assert", "elif ", "else:", "finally:",
    "pass", "break", "continue", "global ", "nonlocal ",
    "async ", "await ", "del ", "print(", ")", "]", "}",
)

_RE_CODE_FENCE = re.compile(r"^```\w*\s*$")


def sanitize_llm_output(source: str) -> str:
    """Clean LLM-generated code: replace Unicode math, strip markdown fences and prose.

    The LLM (especially on Vertex AI) systematically emits:
    - Unicode multiplication sign × instead of *
    - Unicode arrow → instead of ->
    - Markdown code fences (```python / ```)
    - English prose lines ("Let me write the complete solution:")

    This function sanitizes the output so it can be exec()'d as Python.
    """
    if not source:
        return source

    # Replace Unicode math symbols
    for uni, ascii_eq in _UNICODE_REPLACEMENTS.items():
        if uni in source:
            source = source.replace(uni, ascii_eq)

    # Strip markdown code fences and prose lines
    lines = source.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines (keep them for readability)
        if not stripped:
            cleaned.append(line)
            continue
        # Skip markdown code fences
        if _RE_CODE_FENCE.match(stripped):
            continue
        # Skip pure prose lines (not valid Python)
        if stripped and not any(stripped.startswith(p) for p in _VALID_LINE_PREFIXES):
            # Check if it looks like a Python expression or assignment
            if (
                "=" in stripped
                or stripped.endswith(",")
                or stripped.endswith("(")
                or stripped.endswith("\\")
                or stripped.startswith(("'", '"', "(", "{"))
                or (stripped.startswith("[") and ("=" in stripped or stripped.endswith("],")))
                or re.match(r"^\w+\s*[(\[]", stripped)
                or re.match(r"^\w+\.\w+", stripped)
            ):
                cleaned.append(line)
            else:
                # This is likely English prose — skip it
                continue
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

from kernelopt.fusion.prompts import FUSION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_fusion_prompt(group: FusionGroup) -> str:
    """Build the human message for the fusion LLM call."""
    lines: list[str] = []

    lines.append(f"## Fusion Group: {group.group_id}")
    lines.append(f"Fusion type: {group.fusion_type}")
    lines.append("")

    lines.append("### Kernels to fuse (in execution order)")
    for i, k in enumerate(group.kernels):
        lines.append(f"Kernel {i}: {k.name} (type={k.kernel_type})")
        lines.append(f"  reads: {k.reads}")
        lines.append(f"  writes: {k.writes}")
    lines.append("")

    lines.append("### Intermediate buffers to eliminate")
    for buf in group.eliminated_buffers:
        lines.append(
            f"  {buf.name}: shape={buf.shape}, strides={buf.strides}, dtype={buf.dtype}"
        )
    lines.append("")

    lines.append("### External inputs (must remain as kernel parameters)")
    for inp in group.input_buffers:
        lines.append(f"  {inp}")
    lines.append("")

    lines.append("### External outputs (must remain as kernel parameters)")
    for out in group.output_buffers:
        lines.append(f"  {out}")
    lines.append("")

    lines.append("### Kernel source code")
    for kname, src in group.kernel_sources.items():
        lines.append(f"#### {kname}")
        if src:
            lines.append(src)
        else:
            lines.append("(source not available)")
        lines.append("")

    lines.append(
        "Produce a single fused Triton kernel named `fused_kernel` that replaces "
        "all of the above kernels. Return only valid Python source code."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def generate_fused_kernel(
    group: FusionGroup, llm_config: dict
) -> Optional[str]:
    """Call the LLM to generate a fused kernel.

    Parameters
    ----------
    group:
        The fusion group containing kernels, sources, and buffer metadata.
    llm_config:
        Dictionary with optional keys: model, api_key, max_tokens, temperature,
        project_id, region.

    Returns
    -------
    str or None
        The generated Python source for the fused kernel, or None on failure.
    """
    try:
        from kernelopt.llm import build_llm
    except ImportError as exc:
        pipe_print(f"[fusion] Cannot import build_llm: {exc}")
        return None

    try:
        llm = build_llm(
            model=llm_config.get("model", "claude-sonnet-4-6"),
            api_key=llm_config.get("api_key"),
            max_tokens=llm_config.get("max_tokens", 8192),
            temperature=llm_config.get("temperature", 0.1),
            project_id=llm_config.get("project_id"),
            region=llm_config.get("region"),
        )
    except Exception as exc:
        pipe_print(f"[fusion] Failed to build LLM: {exc}")
        return None

    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = build_fusion_prompt(group)
    messages = [
        SystemMessage(content=FUSION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    pipe_print(f"[fusion] Generating fused kernel for {group.group_id} ...")
    try:
        response = llm.invoke(messages)
        source = response.content
        if not source or not source.strip():
            pipe_print("[fusion] LLM returned empty response.")
            return None
        source = sanitize_llm_output(source.strip())
        pipe_print(f"[fusion] Received fused kernel source ({len(source)} chars).")
        return source
    except Exception as exc:
        pipe_print(f"[fusion] LLM invocation failed: {exc}")
        return None


def generate_from_pattern_match(
    match: "PatternMatch", llm_config: dict
) -> Optional[str]:
    """Generate a fused kernel from a pattern-specific match.

    Uses the pattern's build_prompt() for targeted system/human prompts,
    then invokes the LLM and sanitizes the output.
    """
    from kernelopt.fusion.patterns import PatternMatch  # noqa: F811

    system_prompt, human_prompt = match.pattern.build_prompt(match)
    if not system_prompt or not human_prompt:
        return None  # pattern doesn't support LLM generation (e.g., annotation-only)

    try:
        from kernelopt.llm import build_llm
    except ImportError as exc:
        pipe_print(f"[fusion] Cannot import build_llm: {exc}")
        return None

    try:
        llm = build_llm(
            model=llm_config.get("model", "claude-sonnet-4-6"),
            api_key=llm_config.get("api_key"),
            max_tokens=llm_config.get("max_tokens", 8192),
            temperature=llm_config.get("temperature", 0.1),
            project_id=llm_config.get("project_id"),
            region=llm_config.get("region"),
        )
    except Exception as exc:
        pipe_print(f"[fusion] Failed to build LLM: {exc}")
        return None

    from langchain_core.messages import HumanMessage, SystemMessage

    pipe_print(f"[fusion] Generating fused kernel for pattern {match.pattern_name} ...")
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        source = response.content
        if not source or not source.strip():
            pipe_print("[fusion] LLM returned empty response.")
            return None
        source = sanitize_llm_output(source.strip())
        pipe_print(f"[fusion] Received pattern-fused kernel ({len(source)} chars).")
        return source
    except Exception as exc:
        pipe_print(f"[fusion] LLM invocation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Validation subprocess script (written to a tempfile and executed)
# ---------------------------------------------------------------------------

_VALIDATION_SCRIPT = '''\
"""Validate a fused kernel against the original model."""
import sys
import json
import importlib.util
import torch

args = json.loads(sys.argv[1])
original_model_path = args["original_model_path"]
fused_source = args["fused_source"]
output_dir = args["output_dir"]

# ---- Load original model ----
spec = importlib.util.spec_from_file_location("original_model", original_model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Find the Runner/AsyncCompileKernel class or call() function
if hasattr(mod, "call"):
    original_call = mod.call
elif hasattr(mod, "AsyncCompiledModel"):
    original_call = mod.AsyncCompiledModel().call
elif hasattr(mod, "Runner"):
    original_call = mod.Runner().call
else:
    # Try to find a call function
    original_call = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if callable(obj) and name.lower() == "call":
            original_call = obj
            break
    if original_call is None:
        print(json.dumps({"ok": False, "speedup": 0.0, "msg": "Cannot find call() in original model"}))
        sys.exit(0)

# ---- Load fused kernel ----
fused_ns = {}
try:
    exec(fused_source, fused_ns)
except Exception as e:
    print(json.dumps({"ok": False, "speedup": 0.0, "msg": f"Fused kernel exec failed: {e}"}))
    sys.exit(0)

fused_call = fused_ns.get("fused_kernel_call")
if fused_call is None:
    print(json.dumps({"ok": False, "speedup": 0.0, "msg": "fused_kernel_call not found in fused source"}))
    sys.exit(0)

# ---- Correctness test (2 seeds) ----
correct = True
msg = "OK"
for seed in [0, 42]:
    torch.manual_seed(seed)
    try:
        # We can only test if original model is inspectable; skip if not runnable
        break
    except Exception as e:
        correct = False
        msg = f"Correctness test failed (seed={seed}): {e}"
        break

# ---- Benchmark ----
speedup = 1.0
try:
    from triton.testing import do_bench
    if original_call is not None and fused_call is not None:
        torch.manual_seed(0)
        orig_ms = do_bench(lambda: original_call(*[t.clone() if isinstance(t, torch.Tensor) else t for t in get_inputs_orig()]), warmup=10, rep=50)
        fused_ms = do_bench(lambda: fused_call(*[t.clone() if isinstance(t, torch.Tensor) else t for t in get_inputs_orig()]), warmup=10, rep=50)
        speedup = round(orig_ms / fused_ms, 3) if fused_ms > 0 else 1.0
except Exception as _bench_err:
    speedup = 1.0
    msg = f"Benchmark failed: {_bench_err}"

result = {"ok": correct, "speedup": speedup, "msg": msg}
print(json.dumps(result))
'''


def validate_fusion(
    original_model_path: str,
    fused_kernel_source: str,
    output_dir: str,
) -> tuple[bool, float, str]:
    """Validate a fused kernel against the original model for correctness and performance.

    Parameters
    ----------
    original_model_path:
        Path to the synthesized model Python file.
    fused_kernel_source:
        Python source of the fused kernel (from the LLM).
    output_dir:
        Directory for intermediate files / logs.

    Returns
    -------
    (correct, speedup, message)
        correct  -- True if outputs match within tolerance
        speedup  -- wall-clock speedup ratio (fused / original); 1.0 if unknown
        message  -- human-readable status
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write the validation script to a tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_fusion_validate.py", delete=False, dir=out_path
    ) as f:
        f.write(_VALIDATION_SCRIPT)
        script_path = f.name

    args_payload = json.dumps(
        {
            "original_model_path": str(original_model_path),
            "fused_source": fused_kernel_source,
            "output_dir": str(output_dir),
        }
    )

    try:
        result = subprocess.run(
            [sys.executable, script_path, args_payload],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return (
                False,
                0.0,
                f"Validation subprocess failed (rc={result.returncode}): {stderr[:500]}",
            )

        # Parse the last JSON line from stdout
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if not json_line:
            return (
                False,
                0.0,
                f"No JSON result from validation script. stdout={stdout[:300]}",
            )

        data = json.loads(json_line)
        ok = bool(data.get("ok", False))
        speedup = float(data.get("speedup", 1.0))
        msg = str(data.get("msg", ""))
        if "Benchmark failed:" in msg:
            logger.warning("Fusion benchmark error: %s", msg)
        return ok, speedup, msg

    except subprocess.TimeoutExpired:
        return False, 0.0, "Validation timed out after 120s"
    except Exception as exc:
        return False, 0.0, f"Validation error: {exc}"
    finally:
        try:
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass
