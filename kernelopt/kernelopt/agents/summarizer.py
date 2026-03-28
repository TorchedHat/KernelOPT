"""Summarizer agent: compare slow/fast kernel pair, produce an experience item."""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

SUMMARIZER_SYSTEM_PROMPT = """\
You are a GPU optimization analyst. Given before/after profiling data and the
optimization that was applied, produce a structured experience item that captures
what was learned. Follow these steps:

## Step 1 — Identify the causal change
Diff the slow and fast kernels. Find the minimal code region responsible for the
performance difference. Verify against the actual diff, not what the plan intended.

## Step 2 — Attribute the effect to metrics
Compare profiling_context before and after. Which metrics changed, by how much?
Only attribute effects that appear in the profiling data — do not speculate.

## Step 3 — Generalize the insight
Ask: if a different kernel had the same profiling signature, would this change help?
Write the strategy_description at that level of generality. Reference metric names
and values, NOT specific variable names from this kernel.

## Step 4 — Write the pseudocode
Extract the key structural change as framework-neutral pseudocode. Strip boilerplate.
Keep loop structure, tile parameters, memory allocation sizes, and access patterns.

## Step 5 — Deduplication check
If existing_memory is provided, check if an item with the same direction exists.
Report your recommendation in the memory_update field.

## Output format

Your output MUST be TWO JSON objects separated by a blank line.

FIRST — the experience_item:
{
  "item_id": "<uuid>",
  "iteration": <int>,
  "speedup": <float ratio: old_latency / new_latency>,
  "rewrite_type": "<positive|negative>",
  "framework": "<cuda|triton|helion|unknown|multi>",
  "direction": "<snake_case strategy name>",
  "profiling_signal": {
    "ncu_rule_triggered": "<Rule Name or null>",
    "key_metric_before": { "<Section.Metric>": <value>, ... },
    "key_metric_after": { "<Section.Metric>": <value>, ... }
  },
  "strategy_title": "<10 words max>",
  "strategy_description": "<2-3 sentences, generalizable, references metric signals>",
  "slow_pseudocode": "<minimal before code>",
  "fast_pseudocode": "<minimal after code>",
  "applicable_when": "<profiling conditions that make this strategy relevant>",
  "do_not_apply_when": "<conditions where this causes regression or is inapplicable>",
  "framework_notes": "<framework-specific syntax notes, or null>"
}

SECOND — the memory_update:
{
  "action": "<append|replace|skip>",
  "replace_item_id": "<id of item to replace, or null>",
  "reason": "<one sentence>"
}

## For negative rewrites (speedup < 1.0)
The strategy_description MUST explain:
- Which metrics got worse and by how much
- Why the change caused the regression (from profiling data, not speculation)
- What the structural anti-pattern is

Output ONLY these two JSON objects, no markdown fences or extra text.
"""


def run_summarizer(
    profiling_before: dict,
    profiling_after: dict,
    optimization_plan: dict,
    kernel_before: str,
    kernel_after: str,
    iteration: int,
    framework: str,
    speedup: float,
    existing_memory: list[dict] | None = None,
    execution_result: dict | None = None,
    api_key: str | None = None,
    model: str = "Qwen/Qwen3-32B",
    base_url: str = "http://localhost:8000/v1",
) -> tuple[dict, dict]:
    """
    Call the LLM to produce an experience item from before/after data.

    Returns (experience_item, memory_update).
    """
    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.2,
        base_url=base_url,
    )

    rewrite_type = "positive" if speedup >= 1.0 else "negative"

    memory_context = ""
    if existing_memory:
        memory_context = (
            f"\n\n## Existing Memory Queue ({len(existing_memory)} items)\n"
            f"```json\n{json.dumps(existing_memory, indent=2)}\n```\n"
            "Check for duplicate directions. If one exists with higher speedup, "
            "recommend 'skip'. If lower, recommend 'replace'."
        )

    human_msg = (
        f"## Iteration: {iteration}\n"
        f"## Speedup: {speedup}x ({rewrite_type})\n"
        f"## Framework: {framework}\n\n"
        f"## Profiling BEFORE\n"
        f"```json\n{json.dumps(profiling_before, indent=2)}\n```\n\n"
        f"## Profiling AFTER\n"
        f"```json\n{json.dumps(profiling_after, indent=2)}\n```\n\n"
        f"## Optimization Plan Applied\n"
        f"```json\n{json.dumps(optimization_plan, indent=2)}\n```\n\n"
        f"## Kernel BEFORE (first 200 lines)\n"
        f"```\n{''.join(kernel_before.splitlines(True)[:200])}\n```\n\n"
        f"## Kernel AFTER (first 200 lines)\n"
        f"```\n{''.join(kernel_after.splitlines(True)[:200])}\n```"
        f"{memory_context}"
    )

    response = llm.invoke([
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    from kernelopt.llm import strip_thinking
    content = strip_thinking(response.content)
    # Strip markdown fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content[: content.rfind("```")]
    content = content.strip()

    # Parse two JSON objects from response
    item, memory_update = _parse_two_json(content)

    # Ensure required fields on experience_item
    if not item.get("item_id"):
        item["item_id"] = uuid.uuid4().hex[:8]
    item["iteration"] = iteration
    item["speedup"] = speedup
    item["rewrite_type"] = rewrite_type

    # Ensure memory_update has required fields
    if not memory_update:
        memory_update = {
            "action": "append",
            "replace_item_id": None,
            "reason": "New experience item",
        }

    return item, memory_update


def _parse_two_json(content: str) -> tuple[dict, dict]:
    """Parse two JSON objects from the LLM response using brace balancing."""
    objects = []
    i = 0
    while i < len(content):
        if content[i] == "{":
            # Find the matching closing brace
            depth = 0
            start = i
            in_string = False
            escape = False
            for j in range(i, len(content)):
                c = content[j]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = content[start : j + 1]
                        try:
                            obj = json.loads(candidate)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    if len(objects) >= 2:
        # Larger object is experience_item, smaller is memory_update
        if len(objects[0]) >= len(objects[1]):
            return objects[0], objects[1]
        else:
            return objects[1], objects[0]
    elif len(objects) == 1:
        return objects[0], {}
    else:
        raise ValueError(
            f"Could not parse JSON from summarizer response: {content[:500]}"
        )
