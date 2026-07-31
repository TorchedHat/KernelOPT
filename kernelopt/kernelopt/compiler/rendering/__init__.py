"""Prompt rendering for compiler context.

Converts CompilerContext into natural-language sections for LLM agents.
All output is plain text / markdown — no JSON reaches the LLM.
"""

from kernelopt.compiler.rendering.planner_sections import (
    build_compiler_context_section,
)
from kernelopt.compiler.rendering.executor_sections import (
    build_executor_compiler_section,
)

__all__ = [
    "build_compiler_context_section",
    "build_executor_compiler_section",
]
