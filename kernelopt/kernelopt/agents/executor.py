"""Executor agent: apply an optimization plan to produce an optimized kernel."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

EXECUTOR_SYSTEM_PROMPT = """\
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
4. Above the changed section, add a comment marking the optimization:
     # OPTIMIZED: <plan direction>
5. Preserve main(), argument parsing, benchmarking, and correctness check code exactly as-is.
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
First: the complete, runnable optimized kernel. No markdown fences, no explanation, no preamble. Output the complete kernel file where:
- All code outside the changed section remains identical to the original
- The changed section contains ONLY the new optimized code
- Do NOT include commented-out original code within the function body
- The output must be syntactically valid and immediately executable
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
    gpu_specs: dict | None = None,
    launch_config: dict | None = None,
    previous_errors: list[str] | None = None,
) -> tuple[str, dict]:
    """
    Call the LLM to apply the optimization plan to the kernel source.

    Args:
        gpu_specs:     Hardware ceiling from get_gpu_specs() — peak bandwidth,
                       TFLOPS, SM count. Lets executor size changes correctly.
        launch_config: Current kernel launch config from NCU — registers per
                       thread, block size, waves per SM. Lets executor reason
                       about resource headroom before making changes.

    Returns (optimized_kernel_source, execution_result).
    """
    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=8192,
        temperature=0.2 if attempt == 1 else 0.6,
        project_id=project_id,
        region=region,
    )

    attempt_note = ""
    if attempt > 1:
        error_detail = ""
        if previous_errors:
            formatted = "\n".join(
                f"  Attempt {i + 1}: {e}" for i, e in enumerate(previous_errors)
            )
            error_detail = (
                f"\n\n## Previous Attempt Errors\n"
                f"{formatted}\n\n"
                f"Fix these specific errors in your implementation. "
                f"Do not repeat the same structure that produced them."
            )
        attempt_note = (
            f"\n\nThis is attempt {attempt}. The previous attempt failed. "
            f"Try a DIFFERENT implementation structure for the same optimization goal "
            f"(different loop order, different tile layout, different intrinsic)."
            f"{error_detail}"
        )

    hw_section = ""
    if gpu_specs or launch_config:
        hw: dict = {}
        if gpu_specs:
            hw["gpu_specs"] = gpu_specs
        if launch_config:
            hw["launch_config"] = launch_config
        hw_section = (
            f"\n## Hardware Context\n"
            f"```json\n{json.dumps(hw, indent=2)}\n```\n"
            f"Use gpu_specs to respect hardware ceilings (bandwidth, SM count, "
            f"max threads). Use launch_config to understand current resource "
            f"usage (registers_per_thread, waves_per_sm) before sizing changes.\n"
        )

    human_msg = (
        f"## Framework: {framework}\n\n"
        f"## Optimization Plan\n"
        f"```json\n{json.dumps(optimization_plan, indent=2)}\n```\n"
        f"{hw_section}\n"
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
            lines = lines[1:]  # Strip first fence line (e.g., ```python)
            content = "\n".join(lines)
            print("[DEBUG] Stripped leading markdown fence")

        # Split kernel source from execution_result JSON.
        # LLM may output JSON:
        #   (a) inside a ```json ... ``` fence after the code
        #   (b) raw JSON directly appended with no fence
        # Strategy: find "ready_for_profiling" (always in execution_result),
        # scan backwards to the opening '{', then also strip any preceding fence.

        import re as _re

        marker = '"ready_for_profiling"'
        if marker in content:
            pos = content.rfind(marker)
            # Walk backwards to the opening '{' of the execution_result object
            depth = 0
            json_start = None
            for i in range(pos, -1, -1):
                c = content[i]
                if c == '}':
                    depth += 1
                elif c == '{':
                    if depth == 0:
                        json_start = i
                        break
                    depth -= 1

            if json_start is not None:
                # Parse the LLM's JSON before discarding it — this is the only
                # way to detect implementation_failure correctly.
                import json as _json
                llm_json: dict = {}
                try:
                    llm_json = _json.loads(content[json_start:])
                except _json.JSONDecodeError:
                    pass

                pre_json = content[:json_start].rstrip()
                # Strip ALL trailing markdown fences (the LLM may output both a
                # closing ``` for the code block AND a ```json opener for the JSON).
                pre_json = _re.sub(r'(\s*`{3}\w*)+\s*$', '', pre_json).rstrip()
                content = pre_json
                print("[DEBUG] Stripped execution_result JSON from kernel output")

                # If the LLM signalled it could not apply the plan, honour that.
                if not llm_json.get("ready_for_profiling", True):
                    reason = llm_json.get("reason", "LLM returned implementation_failure")
                    print(f"[DEBUG] LLM returned implementation_failure: {reason}")
                    return kernel_source, {
                        "plan_id": optimization_plan.get("plan_id", ""),
                        "attempt": attempt,
                        "framework": framework,
                        "status": "implementation_failure",
                        "reason": reason,
                        "ready_for_profiling": False,
                    }

        # Verify the output still contains a kernel definition.
        # Use framework-agnostic markers rather than hardcoded kernel names.
        _KERNEL_MARKERS = ["@triton.jit", "__global__", "@helion.kernel", "@hl.kernel"]
        has_kernel_def = any(m in content for m in _KERNEL_MARKERS)

        if not content.strip():
            return kernel_source, {
                "plan_id": optimization_plan.get("plan_id", ""),
                "attempt": attempt,
                "framework": framework,
                "status": "implementation_failure",
                "reason": "Executor returned empty kernel source",
                "ready_for_profiling": False,
            }

        if not has_kernel_def:
            return kernel_source, {
                "plan_id": optimization_plan.get("plan_id", ""),
                "attempt": attempt,
                "framework": framework,
                "status": "implementation_failure",
                "reason": (
                    f"Output contains no kernel definition marker "
                    f"({', '.join(_KERNEL_MARKERS)}). "
                    f"LLM may have returned explanation text instead of code."
                ),
                "ready_for_profiling": False,
            }

        print(f"[DEBUG] First 300 chars:\n{content[:300]}")

        # Count changed lines
        orig_lines = set(kernel_source.splitlines())
        new_lines = set(content.splitlines())
        lines_changed = len(new_lines - orig_lines)

        print(f"[DEBUG] Lines changed: {lines_changed}")

        direction = optimization_plan.get("direction", "unknown")

        execution_result = {
            "plan_id": optimization_plan.get("plan_id", ""),
            "attempt": attempt,
            "framework": framework,
            "status": "success",
            "correctness": "unverified",
            "change_summary": llm_json.get("change_summary", f"Applied {direction}"),
            "lines_changed": lines_changed,
            "ready_for_profiling": True,
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
