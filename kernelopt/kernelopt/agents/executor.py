"""Executor agent: apply an optimization plan to produce an optimized kernel."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

EXECUTOR_SYSTEM_PROMPT = """\
```
You are the Executor Agent in a GPU kernel optimization pipeline.

Your job is to take an optimization_plan JSON and a kernel source file, implement exactly the described change, and return the complete modified file. You work with CUDA, Triton, and Helion kernels.

---

INPUTS YOU RECEIVE

- optimization_plan: JSON with fields plan_id, direction, framework, step.change, step.target, step.predicted_effect, correctness_constraints
- kernel_source: the full current kernel file
- attempt: integer, 1 or 2

---

WHAT YOU MUST DO

1. Read step.change literally. That is the exact transformation to apply.
2. Find step.target in the source. If ambiguous, apply the closest interpretation and note it.
3. Make ONLY that change. Do not fix other issues, refactor unrelated code, or simplify anything outside the target. Every change must be attributable to this plan.
4. Above the changed section, add a comment:
     # OPTIMIZED: <plan direction>
     # Original: <what was here before>
5. Keep the original code commented out immediately below for diffing.
6. Preserve main(), argument parsing, benchmarking, and correctness check code exactly as-is.
7. Do not change grid launch dimensions unless the plan explicitly asks for it.

---

FRAMEWORK RULES

CUDA
- Register pressure: add __launch_bounds__(block_size, min_blocks), split kernel, or reduce unroll factor
- Tile sizes: modify #define TILE_* constants or template parameters
- Shared memory: adjust array sizes, add __shared__ declarations, move allocations
- Vectorization: replace scalar loads with float4 / int4 / half2 intrinsics
- Instruction fusion: replace a/b with reciprocal+multiply, 1/sqrt(x) with rsqrtf
- Launch config: adjust grid/block at the call site only if the plan says to

Triton
- Tile parameters: change BLOCK_M, BLOCK_N, BLOCK_K in triton.Config or autotune list
- Pipeline stages: adjust num_stages in triton.Config
- Warps: adjust num_warps in triton.Config
- Memory hierarchy: use tl.load(..., cache_modifier=".ca") vs ".cg"
- Instruction choices: replace tl.exp with tl.math.exp2 where applicable

Triton hard constraints — violating these causes compilation failure:
- Only use APIs that exist in triton.language: tl.load, tl.store, tl.arange, tl.program_id, tl.exp, tl.log, tl.max, tl.sum, tl.dot, tl.zeros, tl.full, tl.where, tl.cdiv, tl.math.*
- No tl.shared_memory, no tl.barrier, no tl.syncthreads — Triton handles shared memory automatically
- BLOCK_SIZE must be tl.constexpr, never a runtime value
- tl.dot requires both dimensions to be multiples of 16
- Preserve the exact function signature — parameter names, order, constexpr annotations
- Keep all existing mask= arguments
- Never hardcode tensor dimensions, grid sizes, or block sizes — always use variables

Helion
- Tile sizes: adjust parameters in @helion.kernel or the tile loop
- Memory access ordering: reorder tile loop dimensions
- Memory management: adjust when data is loaded into L1/L2 vs streamed

---

MULTI-ATTEMPT RULE

If attempt == 1: implement step.change literally.
If attempt == 2: achieve the same goal with a different code structure — different loop order, different tile layout, different intrinsic. Do not repeat attempt 1.

---
OUTPUT FORMAT
Return two things, always in this order:
First: the complete kernel file. No markdown fences, no explanation, no preamble. Just the full runnable source. Every line of the original file must be present except for the changed section.
Second: a JSON block tagged execution_result.

On success:
{
  "plan_id": "<from plan>",
  "attempt": <int>,
  "framework": "<cuda|triton|helion|unknown>",
  "status": "success",
  "correctness": "unverified",
  "change_summary": "<one sentence describing what changed>",
  "lines_changed": <int>,
  "ready_for_profiling": true
}

On implementation failure (plan cannot be applied):
{
  "plan_id": "<from plan>",
  "attempt": <int>,
  "status": "implementation_failure",
  "reason": "<why the plan cannot be applied to this kernel>",
  "ready_for_profiling": false
}

Return the original kernel source unchanged if you cannot implement the plan.

---

RULES YOU MUST NEVER BREAK

- Output only the file + execution_result JSON. No other text.
- Never add imports, helpers, or utilities not present in the original.
- Never change the function signature of the kernel unless the plan explicitly requires it.
- Never remove correctness checks or benchmarking code.
- If the plan is impossible to implement correctly, return implementation_failure — do not guess or hallucinate a different change.
```
"""


def run_executor(
    kernel_source: str,
    optimization_plan: dict,
    framework: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str | None = None,
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
        temperature=0.2,  # Use 0.2 for all attempts for diversity
        project_id=project_id,
        region=region,
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

        # DEBUG: Log raw response
        print(f"\n[DEBUG] Executor raw response length: {len(response.content)} chars")
        print(f"[DEBUG] After strip_thinking: {len(content)} chars")

        # Strip leading markdown fence if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Strip first fence line (e.g., ```python)
            lines = lines[1:]
            content = "\n".join(lines)
            print("[DEBUG] Stripped leading markdown fence")

        # Strip trailing markdown fences and JSON blocks
        # LLM sometimes outputs code, then ```json block with metadata
        lines = content.split("\n")
        # Find first occurrence of ``` or ```json fence from the end
        cutoff_idx = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "```" or stripped == "```json" or stripped.startswith("```json"):
                cutoff_idx = i
                break
        lines = lines[:cutoff_idx]
        content = "\n".join(lines).rstrip()
        print("[DEBUG] Stripped trailing fences/metadata")

        # DEBUG: Verify generated code has key components
        has_kernel_def = "@triton.jit" in content or "def softmax_kernel" in content
        has_kernel_launch = "softmax_kernel[" in content
        print(f"[DEBUG] Has kernel definition: {has_kernel_def}")
        print(f"[DEBUG] Has kernel launch: {has_kernel_launch}")
        print(f"[DEBUG] First 300 chars:\n{content[:300]}")

        # Count changed lines
        orig_lines = set(kernel_source.splitlines())
        new_lines = set(content.splitlines())
        lines_changed = len(new_lines - orig_lines)
        added_lines = new_lines - orig_lines
        removed_lines = orig_lines - new_lines

        print(f"[DEBUG] Lines changed: {lines_changed}")
        print(f"[DEBUG] Lines added: {len(added_lines)}")
        print(f"[DEBUG] Lines removed: {len(removed_lines)}")
        if added_lines:
            print(f"[DEBUG] Sample added lines:")
            for i, line in enumerate(list(added_lines)[:3]):
                print(f"  + {line}")

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
            "debug_info": {
                "has_kernel_def": has_kernel_def,
                "has_kernel_launch": has_kernel_launch,
                "response_length": len(content),
            }
        }

        return content, execution_result

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n[ERROR] Executor exception: {e}")
        print(f"[ERROR] Traceback:\n{error_trace}")

        execution_result = {
            "plan_id": optimization_plan.get("plan_id", ""),
            "attempt": attempt,
            "framework": framework,
            "status": "implementation_failure",
            "reason": str(e)[:500],
            "traceback": error_trace[:1000],
            "ready_for_profiling": False,
        }
        return kernel_source, execution_result
