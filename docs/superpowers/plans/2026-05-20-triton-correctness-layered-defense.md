# Triton Correctness Layered Defense — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop broken Triton kernels from wasting LLM correctness-agent calls and NCU profiling time by adding a dry-run gate, injecting a Triton API reference into the executor prompt, and passing profiling context to the executor.

**Architecture:** Three layers inserted into the existing `execute_plans()` flow. Layer 1 adds a subprocess `--profile-only` validation between executor submission and correctness checking, with error-guided retry. Layer 2 creates a new module that introspects the installed Triton version and appends a valid-API reference to the executor system prompt. Layer 3 builds a compact profiling summary from `baseline_profiling_context` and threads it through to the executor's initial message.

**Tech Stack:** Python 3.12, subprocess, Triton introspection (`triton.language`, `triton.language.extra.libdevice`), LangChain messages

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `kernelopt/kernelopt/functions/correctness.py` | Modify | Add `validate_kernel_runs()` |
| `kernelopt/kernelopt/agents/executor/triton_api_reference.py` | **Create** | `build_triton_api_reference()` — introspects installed Triton |
| `kernelopt/kernelopt/agents/executor/prompts.py` | Modify | Append API reference to `SYSTEM_PROMPT`, add `profiling_summary` + `reference_snippets` params to `build_initial_message()` |
| `kernelopt/kernelopt/agents/executor/__init__.py` | Modify | Thread `profiling_summary` + `reference_snippets` through `run_executor_agent()` |
| `kernelopt/kernelopt/langgraph_pipeline.py` | Modify | Wire dry-run gate in `execute_plans()`, add `_build_executor_context()`, pass context/references to executor |
| `tests_suite/unit/test_triton_correctness_defense.py` | **Create** | Tests for all three layers |

---

### Task 1: Add `validate_kernel_runs()` function

**Files:**
- Modify: `kernelopt/kernelopt/functions/correctness.py` (add function near top, after imports)
- Create: `tests_suite/unit/test_triton_correctness_defense.py`

- [ ] **Step 1: Write the test file with dry-run validation tests**

Create `tests_suite/unit/test_triton_correctness_defense.py`:

```python
"""Tests for the layered defense against Triton kernel hallucination."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))

from kernelopt.functions.correctness import validate_kernel_runs


class TestValidateKernelRuns:
    """Layer 1: Dry-run validation gate."""

    def test_valid_kernel_passes(self, tmp_path):
        kernel = tmp_path / "good.py"
        kernel.write_text(textwrap.dedent("""\
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                y = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x + y, mask=mask)

            def kernel_function(*inputs):
                x, y = inputs
                out = torch.empty_like(x)
                n = x.numel()
                grid = (triton.cdiv(n, 1024),)
                add_kernel[grid](x, y, out, n, BLOCK=1024)
                return out

            def get_inputs():
                return [torch.randn(1024, device='cuda'), torch.randn(1024, device='cuda')]

            if __name__ == "__main__":
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--profile-only", action="store_true")
                args = parser.parse_args()
                inputs = get_inputs()
                kernel_function(*inputs)
                torch.cuda.synchronize()
        """))
        ok, err = validate_kernel_runs(str(kernel))
        assert ok is True
        assert err == ""

    def test_missing_import_fails(self, tmp_path):
        kernel = tmp_path / "bad_import.py"
        kernel.write_text(textwrap.dedent("""\
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def k(x_ptr, n, BLOCK: tl.constexpr):
                pass

            def kernel_function(*inputs):
                x = inputs[0]
                x = copy_if_misaligned(x)  # undefined!
                return x

            def get_inputs():
                return [torch.randn(64, device='cuda')]

            if __name__ == "__main__":
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--profile-only", action="store_true")
                args = parser.parse_args()
                kernel_function(*get_inputs())
                torch.cuda.synchronize()
        """))
        ok, err = validate_kernel_runs(str(kernel))
        assert ok is False
        assert "copy_if_misaligned" in err

    def test_syntax_error_fails(self, tmp_path):
        kernel = tmp_path / "bad_syntax.py"
        kernel.write_text("def broken(\n  pass\n")
        ok, err = validate_kernel_runs(str(kernel))
        assert ok is False
        assert "SyntaxError" in err or "exit" in err

    def test_timeout_fails(self, tmp_path):
        kernel = tmp_path / "hang.py"
        kernel.write_text(textwrap.dedent("""\
            import time
            if __name__ == "__main__":
                time.sleep(999)
        """))
        ok, err = validate_kernel_runs(str(kernel), timeout=2)
        assert ok is False
        assert "timed out" in err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py::TestValidateKernelRuns -v`
Expected: FAIL with `ImportError: cannot import name 'validate_kernel_runs'`

- [ ] **Step 3: Implement `validate_kernel_runs()`**

Add to `kernelopt/kernelopt/functions/correctness.py`, after the imports (after line 13):

```python
import os


def validate_kernel_runs(
    kernel_path: str,
    *,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Run a kernel with --profile-only to verify it doesn't crash.

    This is a pre-flight check before the expensive correctness agent
    or NCU profiling. Catches syntax errors, missing imports, invalid
    Triton APIs, wrong constexpr types, and runtime crashes.

    Returns (ok, error_message).
    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, kernel_path, "--profile-only"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode == 0:
            return True, ""
        stderr = proc.stderr[-1500:] if proc.stderr else ""
        stdout = proc.stdout[-500:] if proc.stdout else ""
        return False, f"exit {proc.returncode}\nstderr:\n{stderr}\nstdout:\n{stdout}"
    except subprocess.TimeoutExpired:
        return False, f"kernel timed out after {timeout}s"
    except Exception as e:
        return False, str(e)
```

Also add `import os` to the imports at the top if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py::TestValidateKernelRuns -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/functions/correctness.py tests_suite/unit/test_triton_correctness_defense.py
git commit -m "feat: add validate_kernel_runs() dry-run gate for Triton kernels"
```

---

### Task 2: Wire dry-run gate into `execute_plans()`

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py` (inside `execute_plans()`, lines 680-762)

- [ ] **Step 1: Change `outer_attempts` to apply to all frameworks**

In `execute_plans()`, find lines 684-685:

```python
        outer_attempts = cfg.K if ext == ".cu" else 1
        executor_K = 1 if ext == ".cu" else cfg.K
```

Replace with:

```python
        outer_attempts = cfg.K
        executor_K = 1
```

This gives all frameworks (CUDA and Python) K outer retry attempts. The executor gets one internal tool-level attempt per outer loop; the outer loop handles compile/dry-run error feedback.

- [ ] **Step 2: Add dry-run validation for Python kernels after line 762**

Find the `break  # compiled successfully (or non-.cu)` line (line 762). The block ending there currently handles CUDA compilation. Replace the section from line 718 through line 762 with logic that handles both CUDA compilation AND Python dry-run:

After line 720 (`Path(kernel_file).write_text(optimized_source)`), and after the existing CUDA compilation block (lines 722-761), add a new block for Python validation. The full structure becomes:

```python
            # Save source to disk
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)

            if ext == ".cu":
                binary_path, err = compile_cuda(kernel_file, output_dir=str(output_dir))
                if err:
                    last_compile_err = err
                    pipe_print(
                        f"    [yellow]Plan {plan_letter} attempt {attempt_idx + 1}/"
                        f"{outer_attempts}: compile failed[/]\n"
                        f"    {err.splitlines()[0] if err else ''}"
                    )
                    if attempt_idx == outer_attempts - 1:
                        correctness_failure_items.append({
                            "item_id": uuid.uuid4().hex[:8],
                            "iteration": iteration + 1,
                            "speedup": 0.0,
                            "rewrite_type": "implementation_flaw",
                            "framework": framework,
                            "direction": direction,
                            "profiling_signal": {
                                "ncu_rule_triggered": plan.get("evidence", {}).get("ncu_rule"),
                                "key_metric_before": {},
                                "key_metric_after": {},
                            },
                            "strategy_title": f"{direction} — compile failed after {outer_attempts} attempts",
                            "strategy_description": (
                                f"Plan '{direction}' produced a .cu file that nvcc rejected "
                                f"after {outer_attempts} attempt(s). "
                                f"Last error: {(last_compile_err or '')[:300]}"
                            ),
                            "slow_pseudocode": "",
                            "fast_pseudocode": "",
                            "applicable_when": plan.get("evidence", {}).get("reasoning", ""),
                            "do_not_apply_when": f"Compilation failed: {(last_compile_err or '')[:200]}",
                            "framework_notes": None,
                        })
                        optimized_source = None
                    continue

                compiled_binary = binary_path

            else:
                # Python kernels (Triton/Helion): dry-run validation
                from kernelopt.functions.correctness import validate_kernel_runs
                ok, run_err = validate_kernel_runs(kernel_file)
                if not ok:
                    last_compile_err = run_err
                    pipe_print(
                        f"    [yellow]Plan {plan_letter} attempt {attempt_idx + 1}/"
                        f"{outer_attempts}: dry-run failed[/]\n"
                        f"    {run_err.splitlines()[0] if run_err else ''}"
                    )
                    if attempt_idx == outer_attempts - 1:
                        correctness_failure_items.append({
                            "item_id": uuid.uuid4().hex[:8],
                            "iteration": iteration + 1,
                            "speedup": 0.0,
                            "rewrite_type": "implementation_flaw",
                            "framework": framework,
                            "direction": direction,
                            "profiling_signal": {
                                "ncu_rule_triggered": plan.get("evidence", {}).get("ncu_rule"),
                                "key_metric_before": {},
                                "key_metric_after": {},
                            },
                            "strategy_title": f"{direction} — dry-run crashed after {outer_attempts} attempts",
                            "strategy_description": (
                                f"Plan '{direction}' produced a kernel that crashed on execution "
                                f"after {outer_attempts} attempt(s). "
                                f"Last error: {(run_err or '')[:300]}"
                            ),
                            "slow_pseudocode": "",
                            "fast_pseudocode": "",
                            "applicable_when": plan.get("evidence", {}).get("reasoning", ""),
                            "do_not_apply_when": f"Dry-run failed: {(run_err or '')[:200]}",
                            "framework_notes": None,
                        })
                        optimized_source = None
                    continue

            break  # compiled/validated successfully
```

- [ ] **Step 3: Verify the existing tests still pass**

Run: `python -m pytest tests_suite/unit/ tests_suite/frameworks/ tests_suite/quality/ -x -q 2>&1 | tail -5`
Expected: All tests pass (no regressions)

- [ ] **Step 4: Commit**

```bash
git add kernelopt/kernelopt/langgraph_pipeline.py
git commit -m "feat: wire dry-run validation gate for Python kernels in execute_plans"
```

---

### Task 3: Create `triton_api_reference.py`

**Files:**
- Create: `kernelopt/kernelopt/agents/executor/triton_api_reference.py`
- Modify: `tests_suite/unit/test_triton_correctness_defense.py` (add tests)

- [ ] **Step 1: Add tests for the API reference builder**

Append to `tests_suite/unit/test_triton_correctness_defense.py`:

```python
from kernelopt.agents.executor.triton_api_reference import build_triton_api_reference


class TestBuildTritonApiReference:
    """Layer 2: Triton API reference for executor prompt."""

    def test_returns_nonempty_string(self):
        ref = build_triton_api_reference()
        assert isinstance(ref, str)
        assert len(ref) > 200

    def test_contains_core_tl_functions(self):
        ref = build_triton_api_reference()
        for fn in ["load", "store", "program_id", "arange", "dot", "exp", "where", "maximum"]:
            assert fn in ref, f"Missing tl.{fn}"

    def test_contains_libdevice_functions(self):
        ref = build_triton_api_reference()
        for fn in ["sqrt", "exp", "tanh", "sin", "cos"]:
            assert fn in ref, f"Missing libdevice.{fn}"

    def test_contains_hardware_constraints(self):
        ref = build_triton_api_reference()
        assert "num_warps" in ref
        assert "num_stages" in ref
        assert "1024" in ref  # max threads per block

    def test_contains_common_mistakes(self):
        ref = build_triton_api_reference()
        assert "triton_helpers" in ref
        assert "reinterpret_tensor" in ref

    def test_contains_triton_version(self):
        ref = build_triton_api_reference()
        import triton
        assert triton.__version__ in ref
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py::TestBuildTritonApiReference -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `triton_api_reference.py`**

Create `kernelopt/kernelopt/agents/executor/triton_api_reference.py`:

```python
"""Triton API reference for the executor agent.

Introspects the installed Triton version at import time and builds a compact
reference string listing all valid tl.* and libdevice.* functions, hardware
constraints, and common mistakes to avoid.
"""

from __future__ import annotations


def build_triton_api_reference() -> str:
    """Build a compact Triton API reference from the installed version.

    Returns a string (~800 tokens) suitable for appending to the executor
    system prompt. Built by introspecting the actual installed Triton,
    so it's always accurate for the user's environment.
    """
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    tl_names = sorted(
        x for x in dir(tl)
        if not x.startswith("_") and callable(getattr(tl, x, None))
    )
    lib_names = sorted(
        x for x in dir(libdevice)
        if not x.startswith("_") and callable(getattr(libdevice, x, None))
    )

    return (
        f"\nTRITON API REFERENCE (v{triton.__version__}):\n"
        f"\n"
        f"Valid tl.* functions:\n"
        f"  {', '.join(tl_names)}\n"
        f"\n"
        f"Valid libdevice.* functions (import: from triton.language.extra import libdevice):\n"
        f"  {', '.join(lib_names[:40])}\n"
        f"  (and {len(lib_names) - 40} more — use for transcendentals not on tl.*)\n"
        f"\n"
        f"Hardware constraints:\n"
        f"- num_warps: 1, 2, 4, or 8 (powers of 2, max 8)\n"
        f"- num_stages: 1-16 (pipeline depth)\n"
        f"- BLOCK sizes: must be powers of 2, typically 16-2048\n"
        f"- Max threads per block: 1024 (= num_warps * 32)\n"
        f"- tl.dot requires both dimensions >= 16\n"
        f"\n"
        f"Common mistakes to AVOID:\n"
        f"- triton_helpers.* does NOT exist in standalone Triton — only inside torch._inductor internals\n"
        f"- Use tl.maximum(a, b) instead of triton_helpers.maximum(a, b)\n"
        f"- Use tl.math.* or libdevice.* for transcendentals not directly available as tl.*\n"
        f"- Import libdevice as: from triton.language.extra import libdevice\n"
        f"- Do NOT use reinterpret_tensor inside @triton.jit — it is a PyTorch function, not a Triton function\n"
        f"- tl.full([1], 0, tl.int32) creates an int32 tensor — cast with .to(tl.float32) before mixing with float ops\n"
        f"- Do NOT import or use async_compile, empty_strided_cuda, or assert_size_stride — these are torch._inductor internals\n"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py::TestBuildTritonApiReference -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/agents/executor/triton_api_reference.py tests_suite/unit/test_triton_correctness_defense.py
git commit -m "feat: add Triton API reference builder from installed version"
```

---

### Task 4: Integrate API reference and context into executor prompts

**Files:**
- Modify: `kernelopt/kernelopt/agents/executor/prompts.py`
- Modify: `tests_suite/unit/test_triton_correctness_defense.py` (add tests)

- [ ] **Step 1: Add tests for updated prompts**

Append to `tests_suite/unit/test_triton_correctness_defense.py`:

```python
from kernelopt.agents.executor.prompts import SYSTEM_PROMPT, build_initial_message


class TestExecutorPromptIntegration:
    """Layer 2+3: API reference in system prompt, context in initial message."""

    def test_system_prompt_contains_api_reference(self):
        assert "TRITON API REFERENCE" in SYSTEM_PROMPT
        assert "tl.*" in SYSTEM_PROMPT or "Valid tl." in SYSTEM_PROMPT
        assert "triton_helpers" in SYSTEM_PROMPT

    def test_initial_message_includes_profiling_summary(self):
        msg = build_initial_message(
            "increase_block_size", K=2,
            profiling_summary="GPU: H200\nBaseline: 5000 ns\nTop: MemoryBound 15%",
        )
        assert "H200" in msg.content
        assert "5000 ns" in msg.content

    def test_initial_message_without_profiling_summary(self):
        msg = build_initial_message("increase_block_size", K=2)
        assert "Hardware Context" not in msg.content

    def test_initial_message_includes_reference_snippets(self):
        msg = build_initial_message(
            "add_autotune", K=1,
            reference_snippets="@triton.autotune example here",
        )
        assert "autotune example" in msg.content

    def test_initial_message_without_reference_snippets(self):
        msg = build_initial_message("increase_block_size", K=2)
        assert "Reference" not in msg.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py::TestExecutorPromptIntegration -v`
Expected: FAIL (SYSTEM_PROMPT doesn't contain API reference yet, build_initial_message doesn't accept new params)

- [ ] **Step 3: Update `SYSTEM_PROMPT` in `prompts.py`**

In `kernelopt/kernelopt/agents/executor/prompts.py`, after the existing imports (line 6), add:

```python
from kernelopt.agents.executor.triton_api_reference import build_triton_api_reference

_TRITON_API_REF = build_triton_api_reference()
```

Then append the reference to `SYSTEM_PROMPT`. Find the closing `"""` of `SYSTEM_PROMPT` (line 119) and change:

```python
If submit_kernel returns a validation error, diagnose what went wrong and resubmit.
"""
```

to:

```python
If submit_kernel returns a validation error, diagnose what went wrong and resubmit.
""" + _TRITON_API_REF
```

- [ ] **Step 4: Add `profiling_summary` and `reference_snippets` to `build_initial_message()`**

Replace the `build_initial_message` function (lines 122-153) with:

```python
def build_initial_message(
    plan_direction: str,
    K: int,
    compile_error: str | None = None,
    correctness_error: str | None = None,
    profiling_summary: str | None = None,
    reference_snippets: str | None = None,
) -> HumanMessage:
    """Build the opening human message for the executor sub-graph."""
    content = (
        f"Implement the optimization plan.\n\n"
        f"Direction: {plan_direction}\n"
        f"You have up to {K} attempts via submit_kernel."
    )
    if profiling_summary:
        content += f"\n\n## Hardware Context\n{profiling_summary}"
    if reference_snippets:
        content += f"\n\n## Reference Kernel Examples\n{reference_snippets}"
    if compile_error:
        content += (
            f"\n\n## Previous attempt failed to compile\n"
            f"The kernel you produced in the previous attempt failed:\n\n"
            f"```\n{compile_error}\n```\n\n"
            f"Fix the error in your new implementation."
        )
    if correctness_error:
        content += (
            f"\n\n## Previous attempt failed correctness\n"
            f"Your prior optimization produced wrong results:\n\n"
            f"```\n{correctness_error}\n```\n\n"
            f"Fix the numerical issue. Common causes:\n"
            f"- Wrong accumulation order or precision (use float32 accumulators)\n"
            f"- Off-by-one in mask or index calculations\n"
            f"- Missing boundary checks for edge tiles\n"
            f"- Incorrect reduction across threads/warps\n"
            f"Preserve the same algorithm but fix the implementation bug."
        )
    return HumanMessage(content=content)
```

Note: the `compile_error` message text is generalized from "rejected by nvcc" to "failed" since it now also carries Python dry-run errors.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py::TestExecutorPromptIntegration -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add kernelopt/kernelopt/agents/executor/prompts.py tests_suite/unit/test_triton_correctness_defense.py
git commit -m "feat: inject Triton API reference and profiling context into executor prompts"
```

---

### Task 5: Thread new params through `run_executor_agent()` and pipeline

**Files:**
- Modify: `kernelopt/kernelopt/agents/executor/__init__.py`
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py`

- [ ] **Step 1: Add `profiling_summary` and `reference_snippets` to `run_executor_agent()`**

In `kernelopt/kernelopt/agents/executor/__init__.py`, update the function signature (line 20-28):

```python
def run_executor_agent(
    kernel_source: str,
    optimization_plan: dict,
    memory: list[dict],
    K: int,
    llm_config: dict,
    llm_override=None,
    compile_error: str | None = None,
    correctness_error: str | None = None,
    profiling_summary: str | None = None,
    reference_snippets: str | None = None,
) -> tuple[str, dict]:
```

Update the `build_initial_message` call (lines 51-54):

```python
    initial_msg = build_initial_message(
        plan_direction, K,
        compile_error=compile_error,
        correctness_error=correctness_error,
        profiling_summary=profiling_summary,
        reference_snippets=reference_snippets,
    )
```

- [ ] **Step 2: Add `_build_executor_context()` to `langgraph_pipeline.py`**

Add this function before `execute_plans()` (around line 640):

```python
def _build_executor_context(state: GraphState) -> str:
    """Build a compact profiling summary for the executor agent."""
    ctx = state.get("baseline_profiling_context", {})
    if not ctx:
        return ""

    gpu = ctx.get("gpu_specs", {})
    launch = ctx.get("launch", {})
    rules = ctx.get("rules", [])

    top_rules = sorted(
        rules, key=lambda r: r.get("speedup_pct", 0), reverse=True
    )[:3]
    rule_lines = "\n".join(
        f"  - {r.get('rule_name', '?')}: {r.get('speedup_pct', 0):.1f}% potential"
        for r in top_rules
    )

    return (
        f"GPU: {gpu.get('gpu_name', 'unknown')} "
        f"({gpu.get('sm_count', '?')} SMs, "
        f"{gpu.get('shared_memory_per_sm_kb', '?')} KB shmem/SM)\n"
        f"Baseline latency: {state.get('baseline_latency_ns', 0):.0f} ns\n"
        f"Current launch: {launch.get('block_size', '?')} threads/block, "
        f"{launch.get('registers_per_thread', '?')} regs/thread, "
        f"{launch.get('waves_per_sm', '?')} waves/SM\n"
        f"Top bottlenecks:\n{rule_lines}"
    )
```

- [ ] **Step 3: Add `_build_reference_snippets()` to `langgraph_pipeline.py`**

Add this function right after `_build_executor_context()`:

```python
def _build_reference_snippets(plan: dict) -> str | None:
    """Get reference kernel snippets relevant to the plan direction."""
    from kernelopt.reference_kernels import get_reference_kernel

    direction = plan.get("direction", "").lower()
    keywords_to_techniques = {
        "matmul": "matmul", "gemm": "matmul", "mm": "matmul",
        "softmax": "fused_softmax",
        "layernorm": "layer_norm", "layer_norm": "layer_norm", "rmsnorm": "layer_norm",
        "attention": "fused_attention", "flash": "fused_attention",
        "dropout": "dropout",
    }

    for keyword, technique in keywords_to_techniques.items():
        if keyword in direction:
            ref = get_reference_kernel(technique)
            if ref:
                return f"### {technique} (Triton tutorial)\n```python\n{ref[:1500]}\n```"
    return None
```

- [ ] **Step 4: Wire context and references into `execute_plans()`**

In `execute_plans()`, after the `llm_config` dict construction (after line 667) and before the plan loop (line 672), add:

```python
    executor_context = _build_executor_context(state)
```

Then update the `_exec_args` dict (around line 690) to include the new params:

```python
                _exec_args = dict(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    memory=state.get("optimization_memory", []),
                    K=executor_K,
                    llm_config=llm_config,
                    compile_error=last_compile_err,
                    profiling_summary=executor_context,
                    reference_snippets=_build_reference_snippets(plan),
                )
```

Also update the correctness retry `_retry_args` dict (around line 799) to include the same new params:

```python
                _retry_args = dict(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    memory=state.get("optimization_memory", []),
                    K=1,
                    llm_config=llm_config,
                    correctness_error=msg[:500],
                    profiling_summary=executor_context,
                    reference_snippets=_build_reference_snippets(plan),
                )
```

- [ ] **Step 5: Verify all tests pass**

Run: `python -m pytest tests_suite/unit/ tests_suite/frameworks/ tests_suite/quality/ -x -q 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add kernelopt/kernelopt/agents/executor/__init__.py kernelopt/kernelopt/langgraph_pipeline.py
git commit -m "feat: thread profiling context and reference snippets through to executor"
```

---

### Task 6: Final integration test

**Files:**
- Modify: `tests_suite/unit/test_triton_correctness_defense.py` (add integration tests)

- [ ] **Step 1: Add integration tests**

Append to `tests_suite/unit/test_triton_correctness_defense.py`:

```python
class TestExecutorContextBuilder:
    """Layer 3: Profiling context preservation."""

    def test_build_executor_context_empty_state(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))
        from kernelopt.langgraph_pipeline import _build_executor_context
        result = _build_executor_context({})
        assert result == ""

    def test_build_executor_context_with_data(self):
        from kernelopt.langgraph_pipeline import _build_executor_context
        state = {
            "baseline_profiling_context": {
                "gpu_specs": {
                    "gpu_name": "NVIDIA H200",
                    "sm_count": 132,
                    "shared_memory_per_sm_kb": 228,
                },
                "launch": {
                    "block_size": 256,
                    "registers_per_thread": 32,
                    "waves_per_sm": 4,
                },
                "rules": [
                    {"rule_name": "MemoryBound", "speedup_pct": 25.0},
                    {"rule_name": "LowOccupancy", "speedup_pct": 10.0},
                    {"rule_name": "RegisterSpill", "speedup_pct": 5.0},
                    {"rule_name": "SmallBottleneck", "speedup_pct": 1.0},
                ],
            },
            "baseline_latency_ns": 5000,
        }
        result = _build_executor_context(state)
        assert "H200" in result
        assert "132 SMs" in result
        assert "5000 ns" in result
        assert "MemoryBound" in result
        assert "25.0%" in result
        # Only top 3 rules
        assert "SmallBottleneck" not in result


class TestReferenceSnippets:
    """Layer 2: Reference kernel injection."""

    def test_matmul_plan_gets_reference(self):
        from kernelopt.langgraph_pipeline import _build_reference_snippets
        plan = {"direction": "optimize_matmul_tile_sizes"}
        result = _build_reference_snippets(plan)
        assert result is not None
        assert "matmul" in result.lower()

    def test_softmax_plan_gets_reference(self):
        from kernelopt.langgraph_pipeline import _build_reference_snippets
        plan = {"direction": "fuse_softmax_with_mask"}
        result = _build_reference_snippets(plan)
        # May be None if reference files don't exist in this checkout
        if result:
            assert "softmax" in result.lower()

    def test_unrelated_plan_gets_none(self):
        from kernelopt.langgraph_pipeline import _build_reference_snippets
        plan = {"direction": "increase_xblock_num_warps"}
        result = _build_reference_snippets(plan)
        assert result is None
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests_suite/unit/test_triton_correctness_defense.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run the full test suite for regressions**

Run: `python -m pytest tests_suite/unit/ tests_suite/frameworks/ tests_suite/quality/ -x -q 2>&1 | tail -10`
Expected: No new failures beyond pre-existing ones (CUDA compiler tests, LLM model naming)

- [ ] **Step 4: Commit**

```bash
git add tests_suite/unit/test_triton_correctness_defense.py
git commit -m "test: add integration tests for layered defense (context builder, reference snippets)"
```

---

### Task 7: Commit spec and plan docs

**Files:**
- Add: `docs/superpowers/specs/2026-05-20-triton-correctness-layered-defense.md`
- Add: `docs/superpowers/plans/2026-05-20-triton-correctness-layered-defense.md`

- [ ] **Step 1: Commit documentation**

```bash
git add docs/superpowers/specs/2026-05-20-triton-correctness-layered-defense.md docs/superpowers/plans/2026-05-20-triton-correctness-layered-defense.md
git commit -m "docs: add spec and plan for Triton correctness layered defense"
```
