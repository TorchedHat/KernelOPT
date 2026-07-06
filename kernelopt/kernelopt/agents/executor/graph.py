# kernelopt/kernelopt/agents/executor/graph.py
"""Executor Agent LangGraph sub-graph definition."""

from __future__ import annotations

import difflib
import json
from typing import Annotated, Optional

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kernelopt.agents.executor.prompts import SYSTEM_PROMPT
from kernelopt.agents.executor.tools import ALL_TOOLS, _TOOL_MAP


class ExecutorState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    kernel_result: Optional[dict]
    attempt: int                 # number of submit_kernel calls made so far


_KERNEL_MARKERS = ["@triton.jit", "@helion.kernel", "@hl.kernel"]


def _infer_framework(kernel_source: str) -> str:
    if "@triton.jit" in kernel_source:
        return "triton"
    if "@helion.kernel" in kernel_source or "@hl.kernel" in kernel_source:
        return "helion"
    return "unknown"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: ExecutorState, config: RunnableConfig) -> dict:
    """Call the LLM with tools bound. Accepts a pre-built LLM via configurable for testing."""
    from kernelopt.llm import build_llm

    llm = config["configurable"].get("llm")
    if llm is None:
        llm_cfg = config["configurable"]["llm_config"]
        llm = build_llm(**llm_cfg)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: ExecutorState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last agent message.

    submit_kernel is intercepted: validates the kernel source, increments the attempt
    counter, and writes kernel_result to state on success or K-exhaustion.
    All other tools are executed normally via _TOOL_MAP.
    """
    last = state["messages"][-1]
    K: int = config["configurable"]["K"]
    plan_id: str = config["configurable"]["plan_id"]
    original_kernel: str = config["configurable"]["original_kernel"]

    tool_messages: list[ToolMessage] = []
    new_kernel_result: Optional[dict] = None
    new_attempt: int = state.get("attempt", 0)

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "submit_kernel":
            new_attempt = state.get("attempt", 0) + 1
            kernel_source: str = args.get("kernel_source", "")
            change_summary: str = args.get("change_summary", "")

            has_marker = any(m in kernel_source for m in _KERNEL_MARKERS)
            is_valid = bool(kernel_source.strip()) and has_marker

            # Validate signature matches original
            sig_error = ""
            if is_valid and "@triton.jit" in kernel_source:
                sig_error = _validate_signature(original_kernel, kernel_source)
                if sig_error:
                    is_valid = False

            if is_valid:
                try:
                    compile(kernel_source, "<submit_kernel>", "exec")
                except SyntaxError as e:
                    is_valid = False
                    sig_error = f"SYNTAX ERROR line {e.lineno}: {e.msg}. Fix and resubmit."

            if is_valid:
                _bad_apis = []
                for _line in kernel_source.splitlines():
                    _s = _line.strip()
                    if _s.startswith("#"):
                        continue
                    # Only reject APIs that genuinely don't exist in the installed Triton.
                    # tl.sigmoid, tl.sqrt, tl.log, tl.sin, tl.cos all exist in Triton 3.4.0+.
                    if "tl.tanh" in _s:
                        _bad_apis.append("tl.tanh does not exist — use libdevice.tanh")
                if _bad_apis:
                    is_valid = False
                    sig_error = "INVALID TRITON API:\n" + "\n".join(set(_bad_apis))

            # Check mask consistency: if the kernel uses masking on some loads
            # but not all, large XBLOCK configs will cause OOB memory access.
            if is_valid:
                _mask_warning = _validate_mask_consistency(kernel_source)
                if _mask_warning:
                    is_valid = False
                    sig_error = _mask_warning

            # Check for common Triton codegen errors that cause runtime crashes
            if is_valid:
                _codegen_errors = _validate_triton_codegen(kernel_source)
                if _codegen_errors:
                    is_valid = False
                    sig_error = _codegen_errors

            if is_valid:
                framework = _infer_framework(kernel_source)
                lines_changed = sum(
                    1 for line in difflib.unified_diff(
                        original_kernel.splitlines(), kernel_source.splitlines()
                    )
                    if line.startswith("+") and not line.startswith("+++")
                )

                new_kernel_result = {
                    "kernel_source": kernel_source,
                    "execution_result": {
                        "plan_id": plan_id,
                        "attempt": new_attempt,
                        "framework": framework,
                        "status": "success",
                        "correctness": "unverified",
                        "change_summary": change_summary,
                        "lines_changed": lines_changed,
                        "ready_for_profiling": True,
                        "body_only": True,
                    },
                }
                tool_messages.append(
                    ToolMessage(content="Kernel submitted successfully.", tool_call_id=tc_id)
                )

            elif new_attempt >= K:
                reason = sig_error or (
                    "Submitted kernel is empty."
                    if not kernel_source.strip()
                    else f"No kernel marker found. Expected one of: {_KERNEL_MARKERS}"
                )
                new_kernel_result = {
                    "kernel_source": original_kernel,
                    "execution_result": {
                        "plan_id": plan_id,
                        "attempt": new_attempt,
                        "framework": "unknown",
                        "status": "implementation_failure",
                        "reason": reason,
                        "ready_for_profiling": False,
                    },
                }
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed (attempt {new_attempt}/{K}): {reason}. "
                            f"All attempts exhausted."
                        ),
                        tool_call_id=tc_id,
                    )
                )

            else:
                reason = sig_error or (
                    "Submitted kernel is empty."
                    if not kernel_source.strip()
                    else f"No kernel marker found. Expected one of: {_KERNEL_MARKERS}"
                )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed (attempt {new_attempt}/{K}): {reason}. "
                            f"Fix and resubmit. Keep the EXACT original function signature."
                        ),
                        tool_call_id=tc_id,
                    )
                )

        else:
            tool_fn = _TOOL_MAP.get(name)
            if tool_fn is None:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool: {name!r}", tool_call_id=tc_id)
                )
                continue
            try:
                result = tool_fn.invoke(args, config=config)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=tc_id,
                    )
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(content=f"Tool error: {exc}", tool_call_id=tc_id)
                )

    updates: dict = {"messages": tool_messages, "attempt": new_attempt}
    if new_kernel_result is not None:
        updates["kernel_result"] = new_kernel_result
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_from_agent(state: ExecutorState) -> str:
    """Route to tools if the agent made tool calls, otherwise END (fallback)."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "tools"


def route_from_tools(state: ExecutorState) -> str:
    """Route to END if kernel_result is set, otherwise back to agent."""
    if state.get("kernel_result") is not None:
        return END
    return "agent"


# ---------------------------------------------------------------------------
# Signature validation
# ---------------------------------------------------------------------------

import re as _re


def _validate_mask_consistency(source: str) -> str:
    """Check that all tl.load calls use a mask when the kernel has masking.

    When @triton.autotune is used with large XBLOCK configs, threads with
    xindex >= xnumel compute out-of-bounds derived indices. If tl.load
    calls for derived indices (e.g., channel index, spatial index) don't
    apply the mask, they cause CUDA illegal memory access errors.

    Returns error message if inconsistency found, empty string if ok.
    """
    has_autotune = '@triton.autotune' in source

    # Find mask variables (e.g., xmask, mask, rmask, tmp_mask)
    mask_defs = _re.findall(r'(\w*mask\w*)\s*=\s*(?:xindex|rindex|offsets|tl\.)', source)
    mask_var = mask_defs[0] if mask_defs else "xmask"

    # When @triton.autotune is present, threads beyond xnumel/rnumel will
    # compute out-of-bounds indices.  ALL tl.load/tl.store MUST have a real
    # mask — passing None is NOT safe because it disables bounds checking.
    # This catches the case where the original kernel used mask=None (because
    # Inductor knew the grid was perfectly aligned) but the LLM adds autotune
    # configs that break that alignment.

    unmasked_loads = []
    none_masked_loads = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue

        # Check tl.load calls
        if 'tl.load(' in stripped:
            has_mask = ('mask=' in stripped or (mask_defs and (
                f', {mask_var}' in stripped or f',{mask_var}' in stripped)))
            has_none = ', None)' in stripped or ', None,' in stripped
            if not has_mask and not has_none:
                ptr_start = stripped.find('tl.load(') + 8
                ptr_expr = stripped[ptr_start:ptr_start + 40].split(',')[0]
                unmasked_loads.append(ptr_expr)
            elif has_none and has_autotune:
                ptr_start = stripped.find('tl.load(') + 8
                ptr_expr = stripped[ptr_start:ptr_start + 40].split(',')[0]
                none_masked_loads.append(ptr_expr)

        # Check tl.store calls
        if 'tl.store(' in stripped:
            has_mask = ('mask=' in stripped or (mask_defs and (
                f', {mask_var}' in stripped or f',{mask_var}' in stripped)))
            has_none = ', None)' in stripped or ', None,' in stripped
            if has_none and has_autotune:
                ptr_start = stripped.find('tl.store(') + 9
                ptr_expr = stripped[ptr_start:ptr_start + 40].split(',')[0]
                none_masked_loads.append(ptr_expr)

    if unmasked_loads and mask_defs:
        return (
            f"MASK CONSISTENCY: {len(unmasked_loads)} tl.load call(s) missing mask "
            f"(kernel uses '{mask_var}' for masking). Unmasked loads: "
            f"{', '.join(unmasked_loads[:3])}. "
            f"With @triton.autotune and large XBLOCK, threads beyond xnumel "
            f"will compute invalid indices and cause CUDA illegal memory access. "
            f"Add mask={mask_var} to ALL tl.load and tl.store calls."
        )

    if none_masked_loads and has_autotune:
        return (
            f"MASK CONSISTENCY: {len(none_masked_loads)} tl.load/tl.store call(s) "
            f"use mask=None with @triton.autotune. This is UNSAFE — autotune configs "
            f"with larger XBLOCK/RBLOCK will have threads beyond xnumel/rnumel that "
            f"compute invalid indices. Replace None with a proper bounds mask "
            f"(e.g., xindex < xnumel). Affected: {', '.join(none_masked_loads[:3])}."
        )
    return ""


def _validate_triton_codegen(source: str) -> str:
    """Check for common Triton codegen errors that cause runtime crashes.

    Returns error message if issues found, empty string if ok.
    """
    errors = []

    # 1. Check for tl.atomic_* on non-global pointers (shared memory atomics
    # are not supported on most operations). If the kernel creates shared
    # memory via tl.zeros/tl.full and then does atomic ops on it, reject.
    if 'tl.atomic_' in source:
        # Check if any atomic is applied to a locally-created tensor
        # (not a pointer parameter). This is a heuristic — can't fully
        # determine memory space statically, but catches common mistakes.
        for line in source.splitlines():
            s = line.strip()
            if s.startswith('#'):
                continue
            if 'tl.atomic_' in s:
                # Check if the pointer arg looks like a local variable
                # (not a function parameter ending in _ptr)
                atomic_match = _re.search(r'tl\.atomic_\w+\((\w+)', s)
                if atomic_match:
                    ptr_name = atomic_match.group(1)
                    if not ptr_name.endswith('_ptr') and not ptr_name.startswith('out_') and not ptr_name.startswith('in_'):
                        errors.append(
                            f"tl.atomic_* on '{ptr_name}' — atomics only work on "
                            f"global memory pointers (function parameters). If "
                            f"'{ptr_name}' is shared/local memory, use regular "
                            f"assignment instead."
                        )

    # 2. Check for pointer + float arithmetic (causes CompilationError:
    # 'invalid operands of type pointer<fp32> and triton.language.float32')
    # Only flag if the float cast is in the POINTER argument (before first comma),
    # not in the value or mask arguments. E.g.:
    #   tl.store(ptr + offset.to(tl.float32), val)  — BAD (float in pointer)
    #   tl.store(ptr, acc.to(tl.float32))            — OK (float cast on value)
    for line in source.splitlines():
        s = line.strip()
        if s.startswith('#'):
            continue
        for call in ('tl.load(', 'tl.store('):
            if call not in s:
                continue
            # Extract pointer argument (everything before first comma after call)
            call_start = s.find(call) + len(call)
            # Find first comma at depth 0 (skip nested parens)
            depth = 0
            ptr_end = len(s)
            for ci, ch in enumerate(s[call_start:], call_start):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    if depth == 0:
                        ptr_end = ci
                        break
                    depth -= 1
                elif ch == ',' and depth == 0:
                    ptr_end = ci
                    break
            ptr_arg = s[call_start:ptr_end]
            if '.to(tl.float' in ptr_arg or '.to(tl.bfloat' in ptr_arg:
                errors.append(
                    f"Float cast in pointer offset — pointer arithmetic must "
                    f"use integer types. Remove .to(tl.float*) from the pointer "
                    f"expression in {call[:-1]}."
                )

    # 3. Check for conflicting meta-parameters: if @triton.autotune defines
    # a config key AND the kernel_function passes it as an explicit kwarg,
    # Triton raises ValueError. Detect autotune config keys.
    autotune_keys = set()
    for m in _re.finditer(r"triton\.Config\(\s*\{([^}]+)\}", source):
        config_text = m.group(1)
        for km in _re.finditer(r"'(\w+)'", config_text):
            autotune_keys.add(km.group(1))
    if autotune_keys:
        # Check if any of these keys are also passed as explicit kwargs
        # in a kernel launch call (kernel_name[grid](..., KEY=val))
        for key in autotune_keys:
            if _re.search(rf',\s*{key}\s*=', source):
                # Check it's in a launch call, not in the Config definition
                for line in source.splitlines():
                    s = line.strip()
                    # Launch calls contain [grid] or [(...)] before (
                    if '][' in s or '[grid]' in s or '](' in s:
                        if f'{key}=' in s and 'Config' not in s:
                            errors.append(
                                f"Conflicting meta-parameter '{key}': defined in "
                                f"@triton.autotune config AND passed as explicit "
                                f"kwarg in launch call. Remove '{key}=...' from "
                                f"the launch call — the autotuner sets it."
                            )
                            break

        # Check that all meta['KEY'] references in grid lambdas exist in
        # autotune configs.  Missing keys cause KeyError at runtime during
        # autotuning — a common LLM mistake when restructuring tiling.
        meta_refs = set(_re.findall(r"meta\[(?:'|\")(\w+)(?:'|\")\]", source))
        missing_keys = meta_refs - autotune_keys
        if missing_keys:
            errors.append(
                f"Missing meta-parameter(s) {sorted(missing_keys)} referenced "
                f"in grid/lambda but not defined in any @triton.autotune config. "
                f"Available config keys: {sorted(autotune_keys)}. Either add the "
                f"missing key(s) to your triton.Config dicts or use the existing "
                f"keys in your grid function."
            )

    if errors:
        return "CODEGEN ERROR:\n" + "\n".join(errors)
    return ""


def _extract_params(source: str) -> list[str] | None:
    """Extract parameter names from the first @triton.jit function."""
    # Match any function decorated with @triton.jit, not just triton_* names
    # First try triton_* pattern, then fall back to any def after @triton.jit
    m = _re.search(r'def\s+triton_\w+\s*\(([^)]+)\)', source)
    if not m:
        m = _re.search(r'@triton\.jit\s*\ndef\s+\w+\s*\(([^)]+)\)', source)
    if not m:
        return None
    return [p.strip().split(':')[0].split('=')[0].strip() for p in m.group(1).split(',')]


def _validate_signature(original: str, submitted: str) -> str:
    """Check that submitted kernel preserves the original signature.

    Returns empty string if valid, or an error message describing the violation.
    """
    orig_params = _extract_params(original)
    sub_params = _extract_params(submitted)
    if orig_params is None or sub_params is None:
        return ""

    if orig_params != sub_params:
        # Show exactly what the function def line must look like
        orig_name_match = _re.search(r'def\s+(\w+)\s*\(', original)
        orig_name = orig_name_match.group(1) if orig_name_match else "kernel"
        orig_sig = f"def {orig_name}({', '.join(orig_params)}):"
        return (
            f"SIGNATURE MISMATCH: Your function has params {sub_params} "
            f"but the original has {orig_params}. "
            f"Your function definition MUST be exactly:\n"
            f"  {orig_sig}\n"
            f"Do NOT add, remove, rename, or reorder any parameters. "
            f"Do NOT restructure from 2D tiling (XBLOCK+YBLOCK) to 1D (BLOCK). "
            f"Keep the SAME parameter names and implement your optimization "
            f"WITHIN the existing signature."
        )
    return ""


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_executor_graph():
    """Return a compiled LangGraph sub-graph for one executor instance."""
    graph = StateGraph(ExecutorState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_from_tools, {"agent": "agent", END: END})

    return graph.compile()


executor_subgraph = compile_executor_graph()
