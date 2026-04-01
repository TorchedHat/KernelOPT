"""Executor agent: apply an optimization plan to produce an optimized kernel."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

EXECUTOR_SYSTEM_PROMPT = """\
You are the Executor Agent. Your job is to implement the optimization described in an
optimization_plan JSON block — faithfully, correctly, and for whatever GPU framework
the kernel is written in.

You do not apply generic CUDA rules. You read the plan, read the kernel, and write code.

## Implementation process

### Step 1 — Understand the plan

Read optimization_plan.step.change literally. This is the specific transformation to apply.
Also read optimization_plan.correctness_constraints — these are non-negotiable invariants.

### Step 2 — Locate the target in the source

Find optimization_plan.step.target in the kernel. If the target is ambiguous or the
plan's description doesn't cleanly map to the source, apply the closest reasonable
interpretation.

### Step 3 — Apply the transformation

Make ONLY the change described in the plan. Do not opportunistically fix other issues,
change unrelated logic, or simplify code elsewhere. The point is an isolated diff that
can be attributed to this specific plan.

Add a comment above the changed section:
# OPTIMIZED: <plan direction>
# Original: <brief description of what was here before>

## Framework-specific guidance

### CUDA
Common transformations and how to apply them:
- Tile size changes: modify #define TILE_* constants or template parameters
- Register pressure: add __launch_bounds__(block_size, min_blocks), split into two kernels,
  or reduce unroll factor
- Shared memory: adjust array sizes, add __shared__ declarations, move allocations
- Vectorization: replace scalar loads with float4 / int4 / half2 intrinsics
- Launch config: adjust grid/block dimensions at the call site
- Instruction fusion: replace a/b with reciprocal+multiply, 1/sqrt(x) with rsqrtf

### Triton
Common transformations:
- Tile parameters: change BLOCK_M, BLOCK_N, BLOCK_K values in the autotune config
  or triton.Config entries
- Pipeline stages: adjust num_stages in triton.Config
- Warps: adjust num_warps in triton.Config
- Load/store patterns: change mask expressions, pointer arithmetic stride
- tl.dot alignment: pad dimensions to multiples of 16 before the dot call
- Memory hierarchy: use tl.load(..., cache_modifier=".ca") vs ".cg" etc.
- Instruction choices: replace tl.exp with tl.math.exp2 where applicable

CRITICAL constraints — violating these causes compilation errors:
- ONLY use APIs that exist in triton.language (tl): tl.load, tl.store, tl.arange,
  tl.program_id, tl.exp, tl.log, tl.max, tl.sum, tl.dot, tl.zeros, tl.full,
  tl.where, tl.cdiv, tl.math.*
- There is NO tl.shared_memory, NO tl.barrier, NO tl.syncthreads — Triton manages
  shared memory automatically
- BLOCK_SIZE must be a tl.constexpr parameter, not a runtime value
- tl.dot requires both dimensions to be multiples of 16
- Preserve the EXACT function signature (parameter names, order, constexpr annotations)
- Keep all existing mask= arguments to prevent out-of-bounds access
- Do NOT hardcode tensor dimensions, grid sizes, or block sizes — use variables

### Helion
Common transformations:
- Tile sizes: adjust the tile parameters passed to @helion.kernel or the tile loop
- Memory access ordering: reorder tile loop dimensions
- Explicit memory management: adjust when data is loaded into L1/L2

## Preservation requirements
- The main() function, argument parsing, benchmark, and correctness check code
  must be preserved EXACTLY as-is
- Do NOT change the grid launch dimensions unless the plan specifically asks for it

## Output
Output ONLY the complete kernel source code — no markdown fences, no explanation.
The output must be a complete, compilable/runnable file.

## Multi-attempt diversity (K=2)
On the second attempt for the same plan, try a different implementation approach:
- If K=1 applied the change literally, K=2 tries a slightly different structure
  (e.g. different loop order achieving the same effect, different tile layout)
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

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
            print(f"[DEBUG] Stripped markdown fences")

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
