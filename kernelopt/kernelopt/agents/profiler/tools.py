# kernelopt/kernelopt/agents/profiler/tools.py
"""ProfilerToolServer and LangChain tool definitions for the Profiler Agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool


class ProfilerToolServer:
    """Holds kernel path and source for the profiler sub-graph tool calls.

    Thread-safe for concurrent reads — no mutation after construction.
    """

    def __init__(self, kernel_path: str, kernel_source: str) -> None:
        self.kernel_path = kernel_path
        self.kernel_source = kernel_source


@tool
def read_kernel_source(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Return the full kernel source code.

    Read this before calling submit_ncu_config. Inspect:
    - GPU kernel function names and decorators (@triton.jit, __global__, @helion.kernel)
    - Benchmark harness structure (warmup loops, launch counts)
    - CLI argument parser (argparse/typer) to determine required kernel_args
    """
    server: ProfilerToolServer = config["configurable"]["tool_server"]
    return server.kernel_source


@tool
def submit_ncu_config(
    kernel_name: str,
    launch_skip: int,
    launch_count: int,
    kernel_args: list[str],
    replay_mode: str,
) -> str:
    """Submit the NCU profiling configuration. Terminates the config-decision phase.

    Args:
        kernel_name: Exact function name or 'regex:(name1|name2|...)' pattern
                     matching only the kernels to profile. Must be non-empty.
        launch_skip: Number of warmup launches to discard before capturing (>= 0).
        launch_count: Number of steady-state launches to capture (>= 1).
        kernel_args: CLI flags to pass to the kernel script, e.g.
                     ['--batch', '32', '--profile-only']. May be empty list.
        replay_mode: 'kernel' (default, lower overhead) or 'application'
                     (lower peak GPU memory, use for inputs >4 GB).
    """
    return "received"  # Body never called — tool_node intercepts submit_ncu_config and handles validation + state update directly


# Map of tool name → tool function for tool_node dispatch.
# submit_ncu_config is intentionally excluded — tool_node intercepts it for validation and state update.
_TOOL_MAP: dict[str, Any] = {"read_kernel_source": read_kernel_source}

# All tools to bind to the LLM (including submit_ncu_config for schema generation).
ALL_TOOLS = [read_kernel_source, submit_ncu_config]
