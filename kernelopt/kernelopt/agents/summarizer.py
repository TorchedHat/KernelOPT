"""Summarizer agent: compare slow/fast kernel pair, produce an experience item."""

from __future__ import annotations

import difflib
import json
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

from kernelopt.agents.summarizer_prompts import SUMMARIZER_SYSTEM_PROMPT


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
    model: str = "claude-sonnet-4-6",
    project_id: str | None = None,
    region: str | None = None,
) -> tuple[dict, dict]:
    """
    Call the LLM to produce an experience item from before/after data.

    Returns (experience_item, memory_update).
    """
    from kernelopt.agents.schemas import SummarizerOutput

    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.2,
        project_id=project_id,
        region=region,
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

    # Compute unified diff
    diff_lines = list(difflib.unified_diff(
        kernel_before.splitlines(keepends=True),
        kernel_after.splitlines(keepends=True),
        fromfile='kernel_before',
        tofile='kernel_after',
        lineterm='',
        n=5  # 5 lines of context around changes
    ))

    # If diff is too large (>500 lines), truncate to first/last 250
    if len(diff_lines) > 500:
        diff_output = ''.join(diff_lines[:250]) + "\n\n... (truncated) ...\n\n" + ''.join(diff_lines[-250:])
    else:
        diff_output = ''.join(diff_lines)

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
        f"## Code Changes (Unified Diff)\n"
        f"```diff\n{diff_output}\n```\n"
        f"{memory_context}"
    )

    # Use structured output for guaranteed schema compliance
    structured_llm = llm.with_structured_output(SummarizerOutput)

    response = structured_llm.invoke([
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    # response is already a Pydantic model with both experience_item and memory_update
    output = response.model_dump()

    item = output["experience_item"]
    memory_update = output["memory_update"]

    # Ensure iteration and speedup are set (should be from structured output, but validate)
    item["iteration"] = iteration
    item["speedup"] = speedup

    return item, memory_update


