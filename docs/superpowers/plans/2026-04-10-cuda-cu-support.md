# Native `.cu` File Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to submit raw `.cu` files — compile once upfront, reuse the binary for correctness checking and NCU profiling, and retry with nvcc error feedback on compile failure (up to K times).

**Architecture:** New `functions/compiler.py` owns all nvcc compilation logic. `functions/correctness.py` gains built-in self-test detection and an LLM C harness fallback. `langgraph_pipeline.py` adds a compile step to `profile_baseline`, wraps `execute_plans` in a K-attempt compile-retry loop (`.cu` only), and drops the redundant compile in `profile_candidates`.

**Tech Stack:** Python, subprocess, LangChain structured output, LangGraph, pytest + unittest.mock

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `kernelopt/functions/compiler.py` | `compile_cuda()` — nvcc wrapper + error formatter |
| Create | `kernelopt/tests/functions/test_compiler.py` | Tests for compile_cuda |
| Modify | `kernelopt/agents/schemas.py` | Add `CudaCorrectnessSpec` |
| Modify | `kernelopt/agents/executor/prompts.py` | Add `compile_error` kwarg to `build_initial_message` |
| Modify | `kernelopt/agents/executor/__init__.py` | Thread `compile_error` through `run_executor_agent` |
| Modify | `kernelopt/functions/correctness.py` | `_has_builtin_self_test`, `_check_cuda_with_llm_reference`, update `_check_cuda_kernel` + dispatcher |
| Modify | `kernelopt/state.py` | Add `compiled_binary_path` to `PlanCandidate` |
| Modify | `kernelopt/langgraph_pipeline.py` | Compile in baseline; K-retry loop in execute_plans; drop redundant compile in profile_candidates |

---

### Task 1: `kernelopt/functions/compiler.py` — nvcc wrapper

**Files:**
- Create: `kernelopt/functions/compiler.py`
- Create: `kernelopt/tests/functions/__init__.py`
- Create: `kernelopt/tests/functions/test_compiler.py`

- [ ] **Step 1: Create the test file with failing tests**

```python
# kernelopt/tests/functions/test_compiler.py
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from kernelopt.functions.compiler import compile_cuda, _format_nvcc_errors


# --- _format_nvcc_errors ---

def test_format_nvcc_errors_keeps_error_lines():
    stderr = (
        "/tmp/tmpXYZ/kernel.cu(42): error: identifier 'foo' is undefined\n"
        "1 error detected in the compilation of 'kernel.cu'.\n"
    )
    result = _format_nvcc_errors(stderr)
    assert "kernel.cu(42): error: identifier 'foo' is undefined" in result


def test_format_nvcc_errors_strips_tmp_prefix():
    stderr = "/tmp/tmpABC123/softmax.cu(10): error: expected ';'\n"
    result = _format_nvcc_errors(stderr)
    assert "/tmp/tmpABC123" not in result
    assert "softmax.cu(10): error: expected ';'" in result


def test_format_nvcc_errors_keeps_warning_lines():
    stderr = "/tmp/x/k.cu(5): warning: unused variable 'x'\n"
    result = _format_nvcc_errors(stderr)
    assert "k.cu(5): warning: unused variable 'x'" in result


def test_format_nvcc_errors_empty_stderr():
    assert _format_nvcc_errors("") == ""


# --- compile_cuda ---

def test_compile_cuda_success_returns_binary_path_and_none():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        binary, err = compile_cuda("/tmp/kernel.cu")
    assert binary == "/tmp/kernel"
    assert err is None
    cmd = mock_run.call_args[0][0]
    assert "nvcc" in cmd
    assert "/tmp/kernel.cu" in cmd
    assert "/tmp/kernel" in cmd


def test_compile_cuda_failure_returns_binary_path_and_error():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "/tmp/tmpXYZ/kernel.cu(5): error: expected ';'\n"
    with patch("subprocess.run", return_value=mock_result):
        binary, err = compile_cuda("/tmp/kernel.cu")
    assert binary == "/tmp/kernel"
    assert err is not None
    assert "kernel.cu(5): error: expected ';'" in err


def test_compile_cuda_binary_path_is_always_predictable():
    """binary_path == cu_path with .cu stripped, regardless of success/failure."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result):
        binary, _ = compile_cuda("/some/dir/my_kernel.cu")
    assert binary == "/some/dir/my_kernel"


def test_compile_cuda_invokes_nvcc_with_O2():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        compile_cuda("/tmp/k.cu")
    cmd = mock_run.call_args[0][0]
    assert "-O2" in cmd
```

- [ ] **Step 2: Create empty `__init__.py` for the new test package**

```python
# kernelopt/tests/functions/__init__.py
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_compiler.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.functions.compiler'`

- [ ] **Step 4: Implement `kernelopt/functions/compiler.py`**

```python
# kernelopt/functions/compiler.py
"""CUDA kernel compiler: wraps nvcc, formats errors."""

from __future__ import annotations

import re
import subprocess


def compile_cuda(cu_path: str) -> tuple[str, str | None]:
    """Compile a .cu file with nvcc.

    Returns:
        (binary_path, None)           on success
        (binary_path, error_message)  on failure

    binary_path is always cu_path with .cu stripped — predictable without
    inspecting the return value.
    """
    binary = cu_path.replace(".cu", "")
    result = subprocess.run(
        ["nvcc", "-O2", "-o", binary, cu_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return binary, _format_nvcc_errors(result.stderr)
    return binary, None


def _format_nvcc_errors(stderr: str) -> str:
    """Strip temp-file path prefixes from nvcc stderr.

    Keeps lines matching: filename.cu(line): error/warning/note: message
    Strips any leading /tmp/... prefix so errors are readable.
    """
    if not stderr.strip():
        return ""
    lines = stderr.strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        # Strip leading /path/to/tmpXXX/ prefix before the filename
        line = re.sub(r"^[^\s]*?/([^/]+\.cu)", r"\1", line)
        cleaned.append(line)
    return "\n".join(cleaned)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_compiler.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/functions/compiler.py kernelopt/tests/functions/__init__.py kernelopt/tests/functions/test_compiler.py
git commit -m "feat: add functions/compiler.py — nvcc wrapper with error formatting"
```

---

### Task 2: `agents/schemas.py` — add `CudaCorrectnessSpec`

**Files:**
- Modify: `kernelopt/kernelopt/agents/schemas.py`

- [ ] **Step 1: Add `CudaCorrectnessSpec` at the end of `agents/schemas.py`**

Append after the final `class MemoryUpdate` block (after line 310 in the current file):

```python
class CudaCorrectnessSpec(BaseModel):
    """LLM-generated .cu test harness for kernels without a built-in self-test."""

    test_harness_code: str = Field(
        ...,
        description=(
            "Complete, self-contained .cu source with a main() that: "
            "(1) generates random float inputs on CPU, "
            "(2) copies to device, "
            "(3) calls the GPU kernel, "
            "(4) computes a CPU reference using standard C math, "
            "(5) compares element-wise within tolerance, "
            "(6) prints 'Correctness: PASS' or 'Correctness: FAIL' and exits 0/1."
        ),
    )
    skip: bool = Field(
        False,
        description="True only if automated testing is impossible (e.g. custom RNG).",
    )
    skip_reason: str | None = Field(None, description="Required when skip=True.")
```

- [ ] **Step 2: Verify the import works**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -c "from kernelopt.agents.schemas import CudaCorrectnessSpec; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/agents/schemas.py
git commit -m "feat: add CudaCorrectnessSpec schema for LLM-generated C test harness"
```

---

### Task 3: Executor `build_initial_message` — thread `compile_error`

**Files:**
- Modify: `kernelopt/kernelopt/agents/executor/prompts.py`
- Test: `kernelopt/tests/agents/test_executor_tools.py` (append tests)

- [ ] **Step 1: Append failing tests to `test_executor_tools.py`**

Add at the end of `kernelopt/tests/agents/test_executor_tools.py`:

```python
def test_build_initial_message_without_compile_error_has_no_nvcc_section():
    msg = build_initial_message("tiling", K=2)
    assert "nvcc" not in msg.content
    assert "compilation" not in msg.content.lower()


def test_build_initial_message_with_compile_error_includes_error():
    msg = build_initial_message("tiling", K=2, compile_error="kernel.cu(5): error: expected ';'")
    assert "kernel.cu(5): error: expected ';'" in msg.content


def test_build_initial_message_with_compile_error_instructs_fix():
    msg = build_initial_message("tiling", K=2, compile_error="kernel.cu(3): error: foo")
    # Message should tell the LLM this is a retry after a compile failure
    assert "compile" in msg.content.lower() or "nvcc" in msg.content.lower()
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/agents/test_executor_tools.py::test_build_initial_message_without_compile_error_has_no_nvcc_section tests/agents/test_executor_tools.py::test_build_initial_message_with_compile_error_includes_error tests/agents/test_executor_tools.py::test_build_initial_message_with_compile_error_instructs_fix -v
```

Expected: first test passes (no `compile_error` kwarg means it's already absent), last two fail with `TypeError: unexpected keyword argument`.

- [ ] **Step 3: Update `build_initial_message` in `prompts.py`**

Replace the entire `build_initial_message` function:

```python
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
```

- [ ] **Step 4: Run all executor tools tests**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/agents/test_executor_tools.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/agents/executor/prompts.py kernelopt/tests/agents/test_executor_tools.py
git commit -m "feat: add compile_error kwarg to build_initial_message for nvcc retry feedback"
```

---

### Task 4: `run_executor_agent` — thread `compile_error`

**Files:**
- Modify: `kernelopt/kernelopt/agents/executor/__init__.py`
- Test: `kernelopt/tests/agents/test_executor_graph.py` (append tests)

- [ ] **Step 1: Append failing test to `test_executor_graph.py`**

Add at the end of `kernelopt/tests/agents/test_executor_graph.py`:

```python
def test_run_executor_agent_passes_compile_error_in_initial_message(sample_memory):
    """compile_error kwarg is accepted without error and surfaces in initial message."""
    mock_llm = _make_mock_llm([
        [{"name": "submit_kernel", "args": {"kernel_source": TRITON_KERNEL, "change_summary": "fixed"}, "id": "tc1"}],
    ])
    # Should not raise TypeError
    kernel, result = run_executor_agent(
        kernel_source=TRITON_KERNEL,
        optimization_plan=SAMPLE_PLAN,
        memory=sample_memory,
        K=1,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        llm_override=mock_llm,
        compile_error="kernel.cu(5): error: expected ';'",
    )
    assert result["status"] == "success"
    # Verify the compile_error appeared in the messages sent to the LLM
    invoke_calls = mock_llm.invoke.call_args_list
    first_call_messages = invoke_calls[0][0][0]  # positional arg: list of messages
    combined = " ".join(str(m.content) for m in first_call_messages)
    assert "expected ';'" in combined
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/agents/test_executor_graph.py::test_run_executor_agent_passes_compile_error_in_initial_message -v
```

Expected: `TypeError: run_executor_agent() got an unexpected keyword argument 'compile_error'`

- [ ] **Step 3: Update `run_executor_agent` in `__init__.py`**

Replace the entire function with:

```python
def run_executor_agent(
    kernel_source: str,
    optimization_plan: dict,
    memory: list[dict],
    K: int,
    llm_config: dict,
    llm_override=None,
    compile_error: str | None = None,
) -> tuple[str, dict]:
    """Run the executor sub-graph to apply one optimization plan.

    Args:
        kernel_source:      Current kernel source to be optimized.
        optimization_plan:  Plan dict from the planner agent.
        memory:             Optimization memory items from prior iterations.
        K:                  Maximum submit_kernel attempts allowed.
        llm_config:         Dict with keys model, api_key, project_id, region.
        llm_override:       Optional pre-built LLM (used in tests to avoid API calls).
        compile_error:      nvcc error from a previous compile attempt, or None.
                            When set, the initial message includes the error so the
                            LLM can fix the specific issue.

    Returns:
        (optimized_kernel_source, execution_result) — drop-in for old run_executor().
    """
    plan_direction = optimization_plan.get("direction", "unknown")
    plan_id = optimization_plan.get("plan_id", "")

    server = ExecutorToolServer(kernel_source, optimization_plan, memory)
    initial_msg = build_initial_message(plan_direction, K, compile_error=compile_error)

    config = RunnableConfig(
        configurable={
            "tool_server": server,
            "llm_config": llm_config,
            "llm": llm_override,
            "K": K,
            "plan_id": plan_id,
            "original_kernel": kernel_source,
        }
    )

    try:
        result = executor_subgraph.invoke(
            {"messages": [initial_msg], "kernel_result": None, "attempt": 0},
            config=config,
        )
    except Exception as e:
        return kernel_source, {
            "plan_id": plan_id,
            "attempt": 0,
            "framework": "unknown",
            "status": "implementation_failure",
            "reason": str(e)[:500],
            "traceback": traceback.format_exc()[:1000],
            "ready_for_profiling": False,
        }

    kernel_result = result.get("kernel_result")
    if kernel_result is None:
        return kernel_source, {
            "plan_id": plan_id,
            "attempt": result.get("attempt", 0),
            "framework": "unknown",
            "status": "implementation_failure",
            "reason": "Agent did not call submit_kernel.",
            "ready_for_profiling": False,
        }

    return kernel_result["kernel_source"], kernel_result["execution_result"]
```

- [ ] **Step 4: Run all executor tests**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/agents/test_executor_graph.py tests/agents/test_executor_tools.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/agents/executor/__init__.py kernelopt/tests/agents/test_executor_graph.py
git commit -m "feat: thread compile_error kwarg through run_executor_agent"
```

---

### Task 5: `functions/correctness.py` — `_has_builtin_self_test`

**Files:**
- Modify: `kernelopt/kernelopt/functions/correctness.py`
- Create: `kernelopt/tests/functions/test_correctness.py`

- [ ] **Step 1: Write failing tests**

```python
# kernelopt/tests/functions/test_correctness.py
import pytest
from kernelopt.functions.correctness import _has_builtin_self_test


WITH_CHECK_FN = """\
#include <cuda_runtime.h>
static int check_correctness(int N) {
    // compares GPU vs CPU
    printf("Correctness: PASS\\n");
    return 1;
}
int main() { check_correctness(4096); }
"""

WITH_PASS_PRINTF = """\
#include <stdio.h>
int main() {
    int ok = run_test();
    if (ok) printf("PASS\\n");
    else printf("FAIL\\n");
}
"""

WITHOUT_SELF_TEST = """\
#include <cuda_runtime.h>
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    // just runs the kernel, no verification
    return 0;
}
"""

KERNEL_ONLY = """\
__global__ void softmax(float* x, int N) {
    // no main at all
}
"""


def test_has_builtin_self_test_detects_check_correctness_function():
    assert _has_builtin_self_test(WITH_CHECK_FN) is True


def test_has_builtin_self_test_detects_pass_printf():
    assert _has_builtin_self_test(WITH_PASS_PRINTF) is True


def test_has_builtin_self_test_returns_false_for_bare_kernel():
    assert _has_builtin_self_test(WITHOUT_SELF_TEST) is False


def test_has_builtin_self_test_returns_false_for_kernel_without_main():
    assert _has_builtin_self_test(KERNEL_ONLY) is False


def test_has_builtin_self_test_returns_false_for_empty_string():
    assert _has_builtin_self_test("") is False
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_has_builtin_self_test'`

- [ ] **Step 3: Add `_has_builtin_self_test` to `correctness.py`**

Add this function after the existing imports in `kernelopt/functions/correctness.py` (before `check_correctness`):

```python
def _has_builtin_self_test(cu_source: str) -> bool:
    """Return True if the .cu source contains its own correctness verification.

    Detects two patterns:
    - A function named 'check_correctness' defined in the file
    - printf calls with literal "PASS" or "FAIL" strings (typical host-side tests)
    """
    import re
    if re.search(r'\bcheck_correctness\s*\(', cu_source):
        return True
    if re.search(r'printf\s*\(\s*"[^"]*(?:PASS|FAIL)', cu_source):
        return True
    return False
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/functions/correctness.py kernelopt/tests/functions/test_correctness.py
git commit -m "feat: add _has_builtin_self_test heuristic to correctness.py"
```

---

### Task 6: `functions/correctness.py` — `_check_cuda_kernel` accepts `binary_path`

**Files:**
- Modify: `kernelopt/kernelopt/functions/correctness.py`
- Modify: `kernelopt/tests/functions/test_correctness.py`

- [ ] **Step 1: Append failing tests**

Add to `kernelopt/tests/functions/test_correctness.py`:

```python
import subprocess
from unittest.mock import patch, MagicMock
from kernelopt.functions.correctness import _check_cuda_kernel


def test_check_cuda_kernel_with_binary_path_skips_nvcc():
    """When binary_path is provided, nvcc should not be called."""
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Correctness: PASS\n"
    mock_run.return_value.stderr = ""

    with patch("subprocess.run", mock_run):
        passed, msg = _check_cuda_kernel(
            original_path="/tmp/k.cu",
            optimized_source="__global__ void k(){}\nint main(){}",
            binary_path="/tmp/k",
        )

    # Only one subprocess call: running the binary (not nvcc compile)
    calls = mock_run.call_args_list
    assert len(calls) == 1
    cmd = calls[0][0][0]
    assert "nvcc" not in cmd[0]
    assert "/tmp/k" in cmd[0]


def test_check_cuda_kernel_without_binary_path_calls_nvcc():
    """When no binary_path, nvcc is invoked to compile."""
    compile_result = MagicMock(returncode=0, stdout="", stderr="")
    run_result = MagicMock(returncode=0, stdout="ok\n", stderr="")

    with patch("subprocess.run", side_effect=[compile_result, run_result]) as mock_run:
        with patch("pathlib.Path.unlink"):
            _check_cuda_kernel(
                original_path="/tmp/k.cu",
                optimized_source="__global__ void k(){}\nint main(){}",
            )

    first_cmd = mock_run.call_args_list[0][0][0]
    assert "nvcc" in first_cmd


def test_check_cuda_kernel_passes_on_exit_zero():
    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="ok\n", stderr=""))
    with patch("subprocess.run", mock_run):
        passed, msg = _check_cuda_kernel("/tmp/k.cu", "__global__ void k(){}", binary_path="/tmp/k")
    assert passed is True


def test_check_cuda_kernel_fails_on_nonzero_exit():
    mock_run = MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="CUDA error"))
    with patch("subprocess.run", mock_run):
        passed, msg = _check_cuda_kernel("/tmp/k.cu", "__global__ void k(){}", binary_path="/tmp/k")
    assert passed is False
    assert "Runtime error" in msg or "exit 1" in msg or "1" in msg
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py::test_check_cuda_kernel_with_binary_path_skips_nvcc -v 2>&1 | head -20
```

Expected: `TypeError: _check_cuda_kernel() got an unexpected keyword argument 'binary_path'`

- [ ] **Step 3: Update `_check_cuda_kernel` signature and body**

In `kernelopt/functions/correctness.py`, replace `_check_cuda_kernel` (currently at lines ~145–190):

```python
def _check_cuda_kernel(
    original_path: str,
    optimized_source: str,
    binary_path: str | None = None,
) -> tuple[bool, str]:
    """Compile (if needed) and run an optimized CUDA kernel; check it doesn't crash.

    Args:
        binary_path: Pre-compiled binary. If provided and the file exists,
                     nvcc compilation is skipped entirely.
    """
    orig_path = Path(original_path)

    if binary_path and Path(binary_path).exists():
        # Binary already compiled — run it directly.
        opt_bin = Path(binary_path)
        tmp_cu = None
    else:
        # No pre-compiled binary; compile from source.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cu", delete=False,
            dir=orig_path.parent,
        ) as tmp:
            tmp.write(optimized_source)
            tmp_cu = Path(tmp.name)

        opt_bin = tmp_cu.with_suffix("")
        compile_result = subprocess.run(
            ["nvcc", "-O2", "-o", str(opt_bin), str(tmp_cu)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_result.returncode != 0:
            if tmp_cu:
                tmp_cu.unlink(missing_ok=True)
            return False, f"Compilation failed:\n{compile_result.stderr[:500]}"

    try:
        run_result = subprocess.run(
            [str(opt_bin)],
            capture_output=True, text=True, timeout=120,
        )
        combined = run_result.stdout + run_result.stderr
        if run_result.returncode != 0:
            return (
                False,
                f"Runtime error (exit {run_result.returncode}):\n{combined[:500]}",
            )
        failure_indicators = ["FAILED", "AssertionError", "RuntimeError", "CUDA error", "Traceback"]
        for indicator in failure_indicators:
            if indicator in combined:
                return False, f"Failure indicator '{indicator}' found in output:\n{combined[:500]}"
        return True, "CUDA kernel ran successfully."
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (120s)."
    finally:
        if tmp_cu:
            tmp_cu.unlink(missing_ok=True)
            opt_bin.unlink(missing_ok=True)
```

- [ ] **Step 4: Run correctness tests**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/functions/correctness.py kernelopt/tests/functions/test_correctness.py
git commit -m "feat: _check_cuda_kernel accepts binary_path to skip nvcc recompilation"
```

---

### Task 7: `functions/correctness.py` — `_check_cuda_with_llm_reference`

**Files:**
- Modify: `kernelopt/kernelopt/functions/correctness.py`
- Modify: `kernelopt/tests/functions/test_correctness.py`

- [ ] **Step 1: Append failing test**

Add to `kernelopt/tests/functions/test_correctness.py`:

```python
from unittest.mock import patch, MagicMock
from kernelopt.functions.correctness import _check_cuda_with_llm_reference

SIMPLE_CUDA_SOURCE = """\
#include <cuda_runtime.h>
#include <math.h>
__global__ void scale(float* x, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
extern "C" void run_scale(float* x, float factor, int n) {
    scale<<<(n+255)/256, 256>>>(x, factor, n);
    cudaDeviceSynchronize();
}
"""

HARNESS_PASS = """\
#include <cuda_runtime.h>
#include <stdio.h>
int main() { printf("Correctness: PASS\\n"); return 0; }
"""

HARNESS_FAIL = """\
#include <cuda_runtime.h>
#include <stdio.h>
int main() { printf("Correctness: FAIL\\n"); return 1; }
"""


def _make_mock_llm_with_spec(harness_code, skip=False):
    from kernelopt.agents.schemas import CudaCorrectnessSpec
    spec = CudaCorrectnessSpec(test_harness_code=harness_code, skip=skip)
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = spec
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


def test_check_cuda_with_llm_reference_pass():
    mock_llm = _make_mock_llm_with_spec(HARNESS_PASS)

    compile_ok = MagicMock(returncode=0, stdout="", stderr="")
    run_ok = MagicMock(returncode=0, stdout="Correctness: PASS\n", stderr="")

    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm):
        with patch("subprocess.run", side_effect=[compile_ok, run_ok]):
            with patch("pathlib.Path.unlink"):
                passed, msg = _check_cuda_with_llm_reference(
                    cu_source=SIMPLE_CUDA_SOURCE,
                    kernel_path="/tmp/scale.cu",
                    api_key=None, model="unused", project_id=None, region=None,
                )
    assert passed is True


def test_check_cuda_with_llm_reference_fail():
    mock_llm = _make_mock_llm_with_spec(HARNESS_FAIL)

    compile_ok = MagicMock(returncode=0, stdout="", stderr="")
    run_fail = MagicMock(returncode=1, stdout="Correctness: FAIL\n", stderr="")

    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm):
        with patch("subprocess.run", side_effect=[compile_ok, run_fail]):
            with patch("pathlib.Path.unlink"):
                passed, msg = _check_cuda_with_llm_reference(
                    cu_source=SIMPLE_CUDA_SOURCE,
                    kernel_path="/tmp/scale.cu",
                    api_key=None, model="unused", project_id=None, region=None,
                )
    assert passed is False


def test_check_cuda_with_llm_reference_skip():
    mock_llm = _make_mock_llm_with_spec("", skip=True)
    mock_llm.with_structured_output.return_value.invoke.return_value = \
        __import__("kernelopt.agents.schemas", fromlist=["CudaCorrectnessSpec"]).CudaCorrectnessSpec(
            test_harness_code="", skip=True, skip_reason="custom RNG"
        )

    with patch("kernelopt.functions.correctness.build_llm", return_value=mock_llm):
        passed, msg = _check_cuda_with_llm_reference(
            cu_source=SIMPLE_CUDA_SOURCE,
            kernel_path="/tmp/scale.cu",
            api_key=None, model="unused", project_id=None, region=None,
        )
    assert passed is False
    assert "skip" in msg.lower() or "custom RNG" in msg
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py::test_check_cuda_with_llm_reference_pass -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_check_cuda_with_llm_reference'`

- [ ] **Step 3: Add `_check_cuda_with_llm_reference` to `correctness.py`**

Add after `_check_cuda_kernel` and before `check_correctness`:

```python
_CUDA_LLM_SYSTEM_PROMPT = """\
You are the CUDA Correctness Agent. Given a CUDA kernel source file, write a complete
self-contained .cu test harness that verifies the kernel produces correct output.

The harness must:
1. Include all necessary headers (#include <cuda_runtime.h>, <stdio.h>, <math.h>, etc.)
2. Generate random float inputs on the CPU (use srand(42) for reproducibility)
3. Copy inputs to device with cudaMalloc/cudaMemcpy
4. Call the GPU kernel
5. Copy results back to host
6. Compute the expected result using a simple CPU reference (pure C math, no CUDA)
7. Compare element-wise: if max absolute diff > 1e-3, print "Correctness: FAIL" and exit(1)
8. Otherwise print "Correctness: PASS" and return 0

The harness must be a single compilable .cu file with int main(). Do NOT use #include of
the kernel file — reproduce the __global__ kernel function inline in the harness.
"""


def _check_cuda_with_llm_reference(
    cu_source: str,
    kernel_path: str,
    api_key: str | None,
    model: str,
    project_id: str | None,
    region: str | None,
) -> tuple[bool, str]:
    """Ask Claude to write a C test harness, compile it, run it, check markers."""
    from kernelopt.agents.schemas import CudaCorrectnessSpec
    from kernelopt.llm import build_llm

    llm = build_llm(model=model, api_key=api_key, max_tokens=4096,
                    temperature=0.1, project_id=project_id, region=region)
    structured_llm = llm.with_structured_output(CudaCorrectnessSpec)

    try:
        spec = structured_llm.invoke([
            SystemMessage(content=_CUDA_LLM_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"## Kernel source ({Path(kernel_path).name})\n"
                f"```c\n{cu_source}\n```\n\n"
                f"Write the test harness."
            )),
        ])
    except Exception as e:
        return False, f"LLM failed to produce CUDA test harness: {e}"

    if spec.skip:
        return False, f"Correctness skipped: {spec.skip_reason or 'no reason given'}"

    # Write harness to temp file, compile, run
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False, dir="/tmp",
        prefix="cuda_harness_",
    ) as f:
        f.write(spec.test_harness_code)
        harness_cu = Path(f.name)

    harness_bin = harness_cu.with_suffix("")
    try:
        compile_result = subprocess.run(
            ["nvcc", "-O2", "-o", str(harness_bin), str(harness_cu)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_result.returncode != 0:
            return False, (
                f"LLM-generated harness failed to compile:\n{compile_result.stderr[:500]}"
            )

        run_result = subprocess.run(
            [str(harness_bin)],
            capture_output=True, text=True, timeout=120,
        )
        combined = run_result.stdout + run_result.stderr
        if run_result.returncode != 0:
            return False, f"Runtime error (exit {run_result.returncode}):\n{combined[:500]}"
        if "FAIL" in combined:
            return False, f"Correctness FAILED:\n{combined[:500]}"
        return True, "CUDA correctness passed (LLM-generated C harness)."
    except subprocess.TimeoutExpired:
        return False, "Harness compilation or execution timed out."
    finally:
        harness_cu.unlink(missing_ok=True)
        harness_bin.unlink(missing_ok=True)
```

Also add the missing imports at the top of `correctness.py` if not already present:
```python
from langchain_core.messages import HumanMessage, SystemMessage
from kernelopt.llm import build_llm
```
(Check first — `build_llm` import may not exist in `functions/correctness.py` yet.)

- [ ] **Step 4: Run correctness tests**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/functions/correctness.py kernelopt/tests/functions/test_correctness.py
git commit -m "feat: add _check_cuda_with_llm_reference — LLM-generated C test harness path"
```

---

### Task 8: `functions/correctness.py` — update dispatcher

**Files:**
- Modify: `kernelopt/kernelopt/functions/correctness.py`
- Modify: `kernelopt/tests/functions/test_correctness.py`

- [ ] **Step 1: Append dispatcher routing tests**

Add to `kernelopt/tests/functions/test_correctness.py`:

```python
from kernelopt.functions.correctness import check_correctness

SOURCE_WITH_SELF_TEST = WITH_CHECK_FN       # defined earlier in this file
SOURCE_WITHOUT_SELF_TEST = WITHOUT_SELF_TEST  # defined earlier in this file


def test_check_correctness_cu_with_self_test_uses_subprocess_path():
    """When .cu has built-in self-test, uses _check_cuda_kernel (no LLM)."""
    with patch("kernelopt.functions.correctness._check_cuda_kernel",
               return_value=(True, "ok")) as mock_ck, \
         patch("kernelopt.functions.correctness._check_cuda_with_llm_reference") as mock_llm:
        passed, msg = check_correctness(
            original_path="/tmp/k.cu",
            optimized_source=SOURCE_WITH_SELF_TEST,
            framework="cuda",
            use_agentic=True,
            binary_path="/tmp/k",
        )
    mock_ck.assert_called_once()
    mock_llm.assert_not_called()
    assert passed is True


def test_check_correctness_cu_without_self_test_and_agentic_uses_llm():
    """When .cu has no built-in self-test and use_agentic=True, uses LLM path."""
    with patch("kernelopt.functions.correctness._check_cuda_kernel") as mock_ck, \
         patch("kernelopt.functions.correctness._check_cuda_with_llm_reference",
               return_value=(True, "ok")) as mock_llm:
        passed, msg = check_correctness(
            original_path="/tmp/k.cu",
            optimized_source=SOURCE_WITHOUT_SELF_TEST,
            framework="cuda",
            use_agentic=True,
            binary_path="/tmp/k",
        )
    mock_llm.assert_called_once()
    mock_ck.assert_not_called()
    assert passed is True


def test_check_correctness_cu_without_self_test_non_agentic_uses_subprocess():
    """When use_agentic=False, always uses _check_cuda_kernel even without self-test."""
    with patch("kernelopt.functions.correctness._check_cuda_kernel",
               return_value=(True, "ok")) as mock_ck, \
         patch("kernelopt.functions.correctness._check_cuda_with_llm_reference") as mock_llm:
        passed, msg = check_correctness(
            original_path="/tmp/k.cu",
            optimized_source=SOURCE_WITHOUT_SELF_TEST,
            framework="cuda",
            use_agentic=False,
            binary_path="/tmp/k",
        )
    mock_ck.assert_called_once()
    mock_llm.assert_not_called()


def test_check_correctness_python_kernel_uses_agentic_path():
    """Non-.cu kernels still use the agentic Python path when use_agentic=True."""
    with patch("kernelopt.functions.correctness.run_correctness_agent",
               return_value=(True, "ok")) as mock_agent:
        passed, msg = check_correctness(
            original_path="/tmp/softmax.py",
            optimized_source="@triton.jit\ndef k(): pass",
            framework="triton",
            use_agentic=True,
        )
    mock_agent.assert_called_once()
    assert passed is True
```

- [ ] **Step 2: Run to verify dispatcher tests fail**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py::test_check_correctness_cu_with_self_test_uses_subprocess_path -v 2>&1 | head -20
```

Expected: `TypeError: check_correctness() got an unexpected keyword argument 'binary_path'`

- [ ] **Step 3: Replace `check_correctness` in `correctness.py`**

Replace the entire `check_correctness` function:

```python
def check_correctness(
    original_path: str,
    optimized_source: str,
    framework: str,
    *,
    use_agentic: bool = True,
    api_key: str | None = None,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str | None = None,
    kernel_args: list[str] | None = None,
    binary_path: str | None = None,
) -> tuple[bool, str]:
    """
    Verify an optimized kernel is correct.

    For .cu kernels:
      - If source has a built-in self-test → run binary, check exit code + markers
      - Else if use_agentic=True → LLM writes C harness, compile, run
      - Else → run binary, check exit code only

    For Python kernels (Triton / Helion):
      - use_agentic=True  → Claude generates CPU reference + torch.allclose
      - use_agentic=False → subprocess (exit code + failure markers)

    Args:
        binary_path: Pre-compiled binary path for .cu kernels. When provided,
                     the run step skips nvcc. Ignored for Python kernels.
    """
    ext = Path(original_path).suffix

    if ext == ".cu":
        if _has_builtin_self_test(optimized_source):
            return _check_cuda_kernel(original_path, optimized_source, binary_path=binary_path)
        if use_agentic:
            return _check_cuda_with_llm_reference(
                cu_source=optimized_source,
                kernel_path=original_path,
                api_key=api_key,
                model=model,
                project_id=project_id,
                region=region,
            )
        return _check_cuda_kernel(original_path, optimized_source, binary_path=binary_path)

    # Python kernels (Triton / Helion)
    if use_agentic:
        from kernelopt.agents.correctness import run_correctness_agent
        return run_correctness_agent(
            kernel_source=optimized_source,
            kernel_path=original_path,
            framework=framework,
            api_key=api_key,
            model=model,
            project_id=project_id,
            region=region,
            kernel_args=kernel_args,
        )
    return check_correctness_subprocess(original_path, optimized_source, framework)
```

- [ ] **Step 4: Run all correctness tests**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/functions/test_correctness.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/functions/correctness.py kernelopt/tests/functions/test_correctness.py
git commit -m "feat: update check_correctness dispatcher — self-test detection + binary_path threading"
```

---

### Task 9: `state.py` — add `compiled_binary_path` to `PlanCandidate`

**Files:**
- Modify: `kernelopt/kernelopt/state.py`

- [ ] **Step 1: Add field to `PlanCandidate`**

In `kernelopt/kernelopt/state.py`, add one line to `PlanCandidate` after `correctness_message`:

```python
@dataclass
class PlanCandidate:
    """One optimization plan and its execution/profiling results."""

    plan_id: str = ""
    plan_letter: str = ""
    direction: str = ""
    optimization_plan: dict = field(default_factory=dict)
    optimized_kernel: str = ""
    execution_result: dict = field(default_factory=dict)
    correctness_passed: bool = False
    correctness_message: str = ""
    compiled_binary_path: str = ""        # ← NEW: set after compile in execute_plans
    profiling_context: dict = field(default_factory=dict)
    csv_path: str = ""
    ncu_rep_path: str = ""
    latency_ns: float = float("inf")
    speedup: float = 1.0
```

- [ ] **Step 2: Verify import still works**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -c "from kernelopt.state import PlanCandidate; c = PlanCandidate(); print(c.compiled_binary_path)"
```

Expected: empty string printed with no error.

- [ ] **Step 3: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/state.py
git commit -m "feat: add compiled_binary_path field to PlanCandidate"
```

---

### Task 10: `langgraph_pipeline.py` — compile in `profile_baseline`

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py`

- [ ] **Step 1: Replace the `_compile_cuda` helper and add the import**

At the top of `langgraph_pipeline.py`, replace:

```python
# REMOVE this entire function (lines 36-45):
def _compile_cuda(cu_path: str) -> str:
    """Compile a .cu file with nvcc and return the binary path. Raises on failure."""
    binary = cu_path.replace(".cu", "")
    result = subprocess.run(
        ["nvcc", "-O2", "-o", binary, cu_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stderr}")
    return binary
```

And add to the imports section:

```python
from kernelopt.functions.compiler import compile_cuda
```

- [ ] **Step 2: Add compile step to `profile_baseline`**

In the `profile_baseline` function, add after `output_dir.mkdir(...)` and before the `if cfg.baseline_csv` block:

```python
    # For .cu kernels: compile the baseline binary once so NCU can run it.
    ext = Path(kernel_path).suffix
    if ext == ".cu":
        binary_path, err = compile_cuda(kernel_path)
        if err:
            raise RuntimeError(
                f"Baseline .cu compilation failed:\n{err}\n\n"
                f"Fix the kernel source and rerun."
            )
        console.print(f"  [dim]Compiled baseline binary: {binary_path}[/]")
```

- [ ] **Step 3: Verify the pipeline module imports cleanly**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -c "from kernelopt.langgraph_pipeline import build_graph; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Run the full test suite to catch regressions**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/ -v --ignore=tests/agents/test_profiler_graph.py 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/langgraph_pipeline.py
git commit -m "feat: compile .cu baseline binary upfront in profile_baseline node"
```

---

### Task 11: `langgraph_pipeline.py` — K-retry compile loop in `execute_plans`

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py`

- [ ] **Step 1: Replace the per-plan block in `execute_plans`**

In `execute_plans`, the current flow per plan is:
```python
for plan_letter, plan in plans:
    ...
    try:
        optimized_source, execution_result = run_executor_agent(...)
    except Exception as e:
        ...
        continue
    if not execution_result.get("ready_for_profiling", False):
        ...
        continue
    kernel_file = ...
    Path(kernel_file).write_text(optimized_source)
    # correctness check follows
```

Replace the section from `try: optimized_source, execution_result = run_executor_agent(...)` through `Path(kernel_file).write_text(optimized_source)` with:

```python
        # For .cu kernels: K outer attempts (executor + compile). Each attempt
        # passes the previous nvcc error back so the LLM can fix it.
        # For non-.cu kernels: single executor call (existing behaviour).
        last_compile_err: str | None = None
        optimized_source: str | None = None
        execution_result: dict = {}
        compiled_binary: str | None = None
        outer_attempts = cfg.K if ext == ".cu" else 1
        executor_K = 1 if ext == ".cu" else cfg.K

        for attempt_idx in range(outer_attempts):
            try:
                optimized_source, execution_result = run_executor_agent(
                    kernel_source=current_kernel,
                    optimization_plan=plan,
                    memory=state.get("optimization_memory", []),
                    K=executor_K,
                    llm_config=llm_config,
                    compile_error=last_compile_err,
                )
            except Exception as e:
                console.print(f"    [red]Executor failed (plan {plan_letter}): {e}[/]")
                optimized_source = None
                break

            if not execution_result.get("ready_for_profiling", False):
                reason = execution_result.get("reason", execution_result.get("status", "failed"))
                console.print(f"    [red]Plan {plan_letter}: {reason}[/]")
                optimized_source = None
                break

            # Save source to disk
            kernel_file = str(output_dir / f"{file_stem}{ext}")
            Path(kernel_file).write_text(optimized_source)

            if ext == ".cu":
                binary_path, err = compile_cuda(kernel_file)
                if err:
                    last_compile_err = err
                    console.print(
                        f"    [yellow]Plan {plan_letter} attempt {attempt_idx + 1}/"
                        f"{outer_attempts}: compile failed[/]\n"
                        f"    {err.splitlines()[0] if err else ''}"
                    )
                    if attempt_idx == outer_attempts - 1:
                        # All K attempts exhausted
                        import uuid as _uuid
                        correctness_failure_items.append({
                            "item_id": _uuid.uuid4().hex[:8],
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
                    continue  # retry with error feedback
                compiled_binary = binary_path

            break  # compiled successfully (or non-.cu)

        if optimized_source is None:
            continue

        execution_result["correctness"] = "unverified"
```

Then ensure the correctness check call that follows passes `binary_path`:

```python
        passed, msg = check_correctness(
            kernel_path, optimized_source, framework,
            use_agentic=cfg.agentic_correctness,
            api_key=cfg.api_key,
            project_id=cfg.project_id,
            region=cfg.region,
            kernel_args=cfg.kernel_args,
            binary_path=compiled_binary,   # ← NEW: None for Python kernels
        )
```

And after the correctness check, when creating the `PlanCandidate`, set the new field:

```python
        candidate = PlanCandidate(
            plan_id=plan.get("plan_id", ""),
            plan_letter=plan_letter,
            direction=direction,
            optimization_plan=plan,
            optimized_kernel=optimized_source,
            execution_result=execution_result,
            correctness_passed=passed,
            correctness_message=msg,
            compiled_binary_path=compiled_binary or "",   # ← NEW
        )
```

- [ ] **Step 2: Verify the module still imports cleanly**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -c "from kernelopt.langgraph_pipeline import build_graph; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Run the full test suite**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/ -v --ignore=tests/agents/test_profiler_graph.py 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/langgraph_pipeline.py
git commit -m "feat: K-retry compile loop in execute_plans — feeds nvcc errors back to executor"
```

---

### Task 12: `langgraph_pipeline.py` — remove redundant compile in `profile_candidates`

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py`

- [ ] **Step 1: Remove the redundant compile block from `profile_candidates`**

In `profile_candidates`, find and delete this block (currently lines 325–329):

```python
        if ext == ".cu":
            try:
                _compile_cuda(kernel_file)
            except RuntimeError as e:
                console.print(f"    [red]Compile failed (plan {c.plan_letter}): {e}[/]")
                c.latency_ns = float("inf")
                c.speedup = 0.0
                continue
```

The binary is already on disk from `execute_plans`. NCU's `build_ncu_command()` strips `.cu` and runs the binary directly — no recompile needed.

Also remove the `import subprocess` at the top of `langgraph_pipeline.py` if it's no longer used after removing `_compile_cuda`. Check first:

```bash
grep -n "subprocess" /workspaces/pytorch-devcontainers/TOOL/kernelopt/kernelopt/langgraph_pipeline.py
```

If `subprocess` only appeared in `_compile_cuda` (now deleted), remove the import line.

- [ ] **Step 2: Verify imports and module load**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -c "from kernelopt.langgraph_pipeline import build_graph, run_langgraph_pipeline; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Run the full test suite one final time**

```bash
cd /workspaces/pytorch-devcontainers/TOOL/kernelopt
python -m pytest tests/ -v --ignore=tests/agents/test_profiler_graph.py 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL
git add kernelopt/kernelopt/langgraph_pipeline.py
git commit -m "refactor: remove redundant nvcc compile from profile_candidates — binary reused from execute_plans"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `functions/compiler.py` | New — `compile_cuda()`, `_format_nvcc_errors()` |
| `agents/schemas.py` | Add `CudaCorrectnessSpec` |
| `agents/executor/prompts.py` | `build_initial_message` gains `compile_error` kwarg |
| `agents/executor/__init__.py` | `run_executor_agent` gains `compile_error` kwarg |
| `functions/correctness.py` | `_has_builtin_self_test`, `_check_cuda_with_llm_reference`, updated `_check_cuda_kernel` + `check_correctness` |
| `state.py` | `PlanCandidate.compiled_binary_path: str = ""` |
| `langgraph_pipeline.py` | Baseline compile; K-retry loop; drop redundant compile; import swap |
