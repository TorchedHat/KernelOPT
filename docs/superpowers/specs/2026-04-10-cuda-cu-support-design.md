# Design: Native `.cu` File Support — Compilation + Correctness Pipeline

**Date**: 2026-04-10
**Status**: Approved

---

## Goal

Allow users to submit raw `.cu` files directly. The pipeline must compile the file with `nvcc`
upfront, reuse the binary for both correctness checking and NCU profiling, and return clean,
readable errors if compilation fails.

---

## Architecture Overview

One new module (`functions/compiler.py`), targeted changes to three existing files. No new
LangGraph nodes, no new agents beyond the existing LLM correctness path.

```
User submits .cu file
        │
        ▼
[profile_baseline node]
  compile_cuda(kernel_path) ──► binary at kernel_path (no .cu extension)
        │ fail ──► RuntimeError with clean nvcc errors (pipeline stops)
        ▼
  NCU runs binary (existing build_ncu_command() behavior unchanged)
        │
        ▼
[execute_plans node]
  executor produces optimized .cu source
  save source to file
  compile_cuda(kernel_file) ──► binary_path stored on PlanCandidate
        │ fail ──► retry executor with nvcc error (up to K attempts)
        │          after K failures → skip candidate, add to correctness_failure_items
        ▼
  check_correctness(..., binary_path=candidate.compiled_binary_path)
    ├─ _has_builtin_self_test(source) == True
    │     └─► run binary, check exit code + "PASS"/"FAIL" markers
    └─ _has_builtin_self_test(source) == False + use_agentic=True
          └─► LLM writes complete .cu test harness → compile → run → check markers
        │
        ▼
[profile_candidates node]
  binary already on disk — _compile_cuda() call removed
  NCU runs binary directly (unchanged)
```

---

## Components

### 1. New: `kernelopt/functions/compiler.py`

Single public function, no LLM, no external dependencies beyond `subprocess`.

```python
def compile_cuda(cu_path: str) -> tuple[str, str | None]:
    """
    Compile a .cu file with nvcc.

    Returns:
        (binary_path, None)          on success
        (binary_path, error_message) on failure

    binary_path is always cu_path with .cu stripped.
    Callers can predict it without inspecting the return value.
    """
```

**Error formatting** (`_format_nvcc_errors`): strips full temp-file paths from nvcc stderr,
keeps `filename.cu:line: error: message` lines. Result is a human-readable string suitable
for display in the CLI and for storage in `correctness_message`. No LLM involved.

The private `_compile_cuda()` currently in `langgraph_pipeline.py` is deleted and replaced
with an import of this function.

---

### 2. Modified: `kernelopt/functions/correctness.py`

#### `_has_builtin_self_test(cu_source: str) -> bool`

Regex scan of the `.cu` source. Returns `True` if the file contains patterns indicating
an embedded correctness check:
- A function named `check_correctness` defined in the file
- `printf` calls with literal `"PASS"` or `"FAIL"` strings inside `main()`

This is a heuristic — false negatives are safe (fall through to LLM path), false positives
are acceptable (binary's self-test runs, exit code and markers are checked).

#### `_check_cuda_kernel(original_path, optimized_source, binary_path=None)`

Updated signature. If `binary_path` is provided and the file exists, skip `nvcc` compilation.
Run the binary, check exit code and failure markers in stdout/stderr. Same failure indicator
list as the Python path (`FAILED`, `AssertionError`, `RuntimeError`, `CUDA error`).

#### `_check_cuda_with_llm_reference(cu_source, kernel_path, api_key, model, ...)`

New function. Only called when `_has_builtin_self_test()` returns `False` and
`use_agentic=True`.

Prompts Claude to write a **complete self-contained `.cu` test harness** — a full file
with `main()` that:
1. Generates random float inputs on the CPU
2. Copies to device
3. Calls the GPU kernel (by reproducing the kernel function inline or via `extern "C"`)
4. Computes a CPU reference using standard C math
5. Compares element-wise within a tolerance
6. Prints `"Correctness: PASS"` or `"Correctness: FAIL"` and exits with the appropriate code

The LLM returns the harness as a single string (analogous to `reference_code` in the Python
correctness agent). Python writes it to a temp `.cu` file, compiles with `compile_cuda()`,
runs the binary, and checks exit code + markers.

Schema addition in `agents/schemas.py`:
```python
class CudaCorrectnessSpec(BaseModel):
    test_harness_code: str   # complete .cu source, compilable standalone
    skip: bool = False
    skip_reason: str | None = None
```

#### `check_correctness()` dispatcher — `.cu` branch

```
ext == ".cu":
    1. _has_builtin_self_test(source) == True
           → _check_cuda_kernel(..., binary_path=binary_path)
    2. _has_builtin_self_test(source) == False AND use_agentic=True
           → _check_cuda_with_llm_reference(...)
    3. _has_builtin_self_test(source) == False AND use_agentic=False
           → _check_cuda_kernel(..., binary_path=binary_path)  # exit code only
```

`check_correctness()` gains an optional `binary_path: str | None = None` kwarg.

---

### 3. Modified: `kernelopt/state.py`

One field added to `PlanCandidate`:

```python
@dataclass
class PlanCandidate:
    ...
    compiled_binary_path: str = ""  # set after compile in execute_plans; reused in profile_candidates
```

`GraphState` unchanged — the binary path is always derivable as `cu_path.replace('.cu', '')`.

---

### 4. Modified: `kernelopt/langgraph_pipeline.py`

#### `profile_baseline` node

For `.cu` files, add before the NCU call:
```python
if ext == ".cu":
    binary_path, err = compile_cuda(kernel_path)
    if err:
        raise RuntimeError(f"Baseline compilation failed:\n{err}")
```
NCU's `build_ncu_command()` already strips `.cu` and uses the binary — no further changes
needed there.

#### `execute_plans` node

For `.cu` files, the executor call and compile step are wrapped in a K-attempt retry loop.
If compilation fails, the error is passed back to the executor as context for the next attempt.
Only after all K attempts are exhausted does the plan get skipped and recorded as a failure.

```python
last_compile_err = None
for attempt in range(cfg.K):
    optimized_source, execution_result = run_executor_agent(
        ...,
        compile_error=last_compile_err,   # None on first attempt
    )
    if ext == ".cu":
        binary_path, err = compile_cuda(kernel_file)
        if err:
            last_compile_err = err
            console.print(f"    [red]Plan {plan_letter} attempt {attempt+1}/{cfg.K}: compile failed[/]")
            continue   # retry executor with error context
    break  # compiled successfully (or non-.cu kernel — no compile step)
else:
    # All K attempts failed to compile
    correctness_failure_items.append({
        "item_id": uuid4().hex[:8],
        "iteration": iteration + 1,
        "speedup": 0.0,
        "rewrite_type": "implementation_flaw",
        "framework": framework,
        "direction": direction,
        "profiling_signal": {"ncu_rule_triggered": None, "key_metric_before": {}, "key_metric_after": {}},
        "strategy_title": f"{direction} — compile failed after {cfg.K} attempts",
        "strategy_description": f"Plan '{direction}' produced a .cu file that nvcc rejected after {cfg.K} attempts. Last error: {last_compile_err[:300]}",
        "slow_pseudocode": "", "fast_pseudocode": "",
        "applicable_when": "", "do_not_apply_when": f"Compilation failed: {last_compile_err[:200]}",
        "framework_notes": None,
    })
    continue

candidate.compiled_binary_path = binary_path
```

`run_executor_agent()` gains an optional `compile_error: str | None = None` kwarg. When set,
the executor's initial message includes the previous attempt's nvcc error so the LLM can
correct the specific issue rather than starting from scratch.

`check_correctness()` call gains `binary_path=candidate.compiled_binary_path`.

#### `profile_candidates` node

Remove the existing `if ext == ".cu": _compile_cuda(kernel_file)` block (lines 325–329).
Binary is already on disk from `execute_plans`.

Remove the now-unused `_compile_cuda()` helper (lines 36–45).
Replace top-of-file with: `from kernelopt.functions.compiler import compile_cuda`.

---

## Data Flow (CUDA path end-to-end)

```
kernelopt run softmax_cuda.cu
        │
        ▼  profile_baseline
compile_cuda("softmax_cuda.cu") → binary "softmax_cuda"
run_profiler_agent("softmax_cuda", ...)  ← NCU uses binary
        │
        ▼  generate_plans (unchanged)
        │
        ▼  execute_plans (per plan)
executor produces optimized .cu source
save → "softmax_cuda_iter1_planA_direction.cu"
compile_cuda("...planA....cu") → binary "...planA..."
        | err → retry executor with nvcc error (up to K attempts) | K exhausted → skip
check_correctness(..., binary_path="...planA...")
  _has_builtin_self_test() → True  → run binary, check markers
                           → False → LLM harness → compile → run
        │
        ▼  profile_candidates
run_profiler_agent("...planA....cu", ...)  ← binary already exists, no recompile
```

---

## Error Handling

| Failure point | Behavior |
|---|---|
| Baseline `.cu` won't compile | `RuntimeError` propagates, pipeline stops with clean message |
| Optimized `.cu` won't compile | Executor retried up to K times with the nvcc error as context; skipped and added to `correctness_failure_items` only after all K attempts fail |
| LLM harness won't compile | `(False, "LLM-generated harness failed to compile: ...")` |
| LLM harness exits non-zero | `(False, "Runtime error (exit N): ...")` |
| Built-in self-test prints FAIL | `(False, "Failure indicator 'FAIL' found in output: ...")` |

---

## What Is Not Changing

- `build_ncu_command()` in `ncu_runner.py` — already handles `.cu` by stripping the extension
- The Python/Triton/Helion correctness path — untouched
- LangGraph graph structure — no new nodes or edges
- `PipelineConfig` — no new config knobs; `agentic_correctness` flag already controls LLM use
