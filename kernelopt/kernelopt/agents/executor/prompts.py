# kernelopt/kernelopt/agents/executor/prompts.py
"""System prompt and initial message builder for the Executor Agent."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


SYSTEM_PROMPT = """\
You are the Executor Agent in a GPU kernel optimization pipeline.

You receive an optimization plan and a kernel source file.
Your job is to implement exactly the change described in the plan.

Use your tools to understand the context:
  get_optimization_plan()  — read the plan you must implement
  get_kernel_source()      — read the current kernel source
  search_memory(query)     — check past experience for relevant patterns

When you have implemented the change, call submit_kernel(kernel_source, change_summary).
  kernel_source  — the complete modified file with ONLY the plan's change applied
  change_summary — one sentence describing what changed

If submit_kernel returns a validation error, diagnose what went wrong and resubmit.
"""


def build_initial_message(
    plan_direction: str,
    K: int,
    compile_error: str | None = None,
) -> HumanMessage:
    """Build the opening human message for the executor sub-graph."""
    content = (
        f"Implement the optimization plan.\n\n"
        f"Direction: {plan_direction}\n"
        f"You have up to {K} attempts via submit_kernel."
    )
    if compile_error:
        content += (
            f"\n\n## Previous attempt failed to compile\n"
            f"The kernel you produced in the previous attempt was rejected by nvcc:\n\n"
            f"```\n{compile_error}\n```\n\n"
            f"Fix the compilation error in your new implementation."
        )
    return HumanMessage(content=content)
