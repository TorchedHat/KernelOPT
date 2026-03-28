"""Executor agent: apply an optimization plan to produce an optimized kernel."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

EXECUTOR_SYSTEM_PROMPT = """\
You are a GPU kernel code transformer. Given a kernel source file and an optimization
plan, produce the optimized version of the kernel.

## Rules
1. Apply EXACTLY the change described in the plan's "step" field.
2. Preserve all correctness_constraints listed in the plan.
3. Do NOT add unrelated optimizations — apply only what the plan specifies.
4. Add a comment above the changed section:
   # OPTIMIZED: <plan direction>
   # Original: <brief description of what was here before>
5. Output ONLY the complete kernel source code — no markdown fences, no explanation.
6. The output must be a complete, compilable/runnable file.

## Framework-specific guidance

### CUDA
- Tile size changes: modify #define TILE_* constants or template parameters
- Register pressure: add __launch_bounds__(block_size, min_blocks), split kernel, reduce unroll
- Shared memory: adjust array sizes, add __shared__ declarations
- Vectorization: replace scalar loads with float4/int4/half2 intrinsics
- Launch config: adjust grid/block dimensions at the call site
- Instruction fusion: replace a/b with reciprocal+multiply, 1/sqrt(x) with rsqrtf

### Triton
CRITICAL constraints — violating these causes compilation errors:
- ONLY use APIs that exist in triton.language (tl): tl.load, tl.store, tl.arange,
  tl.program_id, tl.exp, tl.log, tl.max, tl.sum, tl.dot, tl.zeros, tl.full,
  tl.where, tl.cdiv, tl.math.*
- There is NO tl.shared_memory, NO tl.barrier, NO tl.syncthreads — Triton manages
  shared memory automatically
- BLOCK_SIZE must be a tl.constexpr parameter, not a runtime value
- tl.dot requires both dimensions to be multiples of 16
- To change num_warps or num_stages, modify the kernel LAUNCH CALL, not the kernel body
- Preserve the EXACT function signature (parameter names, order, constexpr annotations)
- Keep all existing mask= arguments to prevent out-of-bounds access
- Do NOT hardcode tensor dimensions, grid sizes, or block sizes — use variables
- The softmax/reduction normalization logic must remain mathematically equivalent

### Helion
- Tile sizes: adjust parameters passed to @helion.kernel or tile loop
- Memory access ordering: reorder tile loop dimensions
- Explicit memory management: adjust when data is loaded into L1/L2

## Preservation requirements
- The main() function, argument parsing, benchmark, and correctness check code
  must be preserved EXACTLY as-is
- Do NOT change the grid launch dimensions unless the plan specifically asks for it
"""


def run_executor(
    kernel_source: str,
    optimization_plan: dict,
    framework: str,
    api_key: str | None = None,
    model: str = "Qwen/Qwen3-32B",
    base_url: str = "http://localhost:8000/v1",
    attempt: int = 1,
) -> tuple[str, dict]:
    """
    Call the LLM to apply the optimization plan to the kernel source.

    Returns (optimized_kernel_source, execution_result).
    """
    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=8192,
        temperature=0.0 if attempt == 1 else 0.2,
        base_url=base_url,
    )

    attempt_note = ""
    if attempt > 1:
        attempt_note = (
            f"\n\nThis is attempt {attempt} — try an ALTERNATIVE implementation "
            f"approach to the same plan (different loop order, different tile layout, "
            f"etc.) while achieving the same optimization goal."
        )

    human_msg = (
        f"## Framework: {framework}\n\n"
        f"## Optimization Plan\n"
        f"```json\n{json.dumps(optimization_plan, indent=2)}\n```\n\n"
        f"## Current Kernel Source\n"
        f"```\n{kernel_source}\n```\n\n"
        f"Apply the optimization plan and output the complete optimized kernel source."
        f"{attempt_note}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])

        from kernelopt.llm import strip_thinking
        content = strip_thinking(response.content)
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        # Count changed lines
        orig_lines = set(kernel_source.splitlines())
        new_lines = set(content.splitlines())
        lines_changed = len(new_lines - orig_lines)

        direction = optimization_plan.get("direction", "unknown")

        execution_result = {
            "plan_id": optimization_plan.get("plan_id", ""),
            "attempt": attempt,
            "framework": framework,
            "status": "success",
            "correctness": "unverified",
            "change_summary": f"Applied {direction}",
            "lines_changed": lines_changed,
            "ready_for_profiling": True,
        }

        return content, execution_result

    except Exception as e:
        execution_result = {
            "plan_id": optimization_plan.get("plan_id", ""),
            "attempt": attempt,
            "framework": framework,
            "status": "implementation_failure",
            "reason": str(e)[:500],
            "ready_for_profiling": False,
        }
        return kernel_source, execution_result
