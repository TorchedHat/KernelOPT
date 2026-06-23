# Layered Defense Against Triton Kernel Hallucination

## Problem

The executor agent generates incorrect Triton kernels that crash before NCU can profile them, breaking the optimization loop. Three failure modes observed in production runs:

1. **Invalid API calls** — LLM invents `tl.*` functions that don't exist or uses wrong signatures (e.g., `triton_helpers.maximum` which only exists inside `torch._inductor`)
2. **Missing imports** — Generated code calls Inductor internals like `copy_if_misaligned` without importing them
3. **Type mismatches** — `tl.full([1], 0, tl.int32)` mixed with float tensors in `tl.maximum`, producing `inf` differences

Currently the pipeline has **zero validation for Triton kernels** between executor submission (which only checks for `@triton.jit` marker presence) and correctness checking (which requires an expensive LLM call). A kernel that can't even import wastes both the correctness-agent LLM call and, if it somehow passes, NCU profiling time.

The `benchmarks/kernelbench/generate_naive.py` has a `_validate_kernel()` function that runs `python kernel.py --profile-only` as a pre-flight check — this pattern exists but was never wired into the optimization pipeline.

## Design

Three layers, each catching errors the previous layer missed:

### Layer 1: Dry-Run Validation Gate

A subprocess check that runs the kernel once with `--profile-only` before any correctness or profiling work. On failure, the error is fed back to the executor for retry — same pattern as the existing CUDA compilation gate.

**New function** in `kernelopt/kernelopt/functions/correctness.py`:

```python
def validate_kernel_runs(kernel_path: str, timeout: int = 60) -> tuple[bool, str]:
    """Run kernel with --profile-only to verify it doesn't crash.
    
    Returns (ok, error_message). Catches syntax errors, missing imports,
    invalid Triton APIs, wrong constexpr types, and runtime crashes.
    """
```

Implemented as `subprocess.run([sys.executable, kernel_path, "--profile-only"])` with stderr capture, matching the existing `_validate_kernel()` in `benchmarks/kernelbench/generate_naive.py`.

**Integration** in `langgraph_pipeline.py:execute_plans()`:

- Change line 684: `outer_attempts = cfg.K` for all frameworks (was `cfg.K if ext == ".cu" else 1`). The outer loop retries the executor with error feedback; the inner `executor_K` stays `1` for `.cu` (CUDA uses outer compile retries) and `cfg.K` for `.py` (Triton uses inner tool retries). With the dry-run gate, `.py` also benefits from outer retries.
- Also change line 685: `executor_K = 1` for all frameworks when dry-run gate is active — the outer loop handles retries, the inner executor gets one shot per attempt.
- Insert dry-run check after line 720 (kernel written to disk), before line 769 (correctness check)
- For `.py` files: call `validate_kernel_runs(kernel_file)`. On failure, set `last_compile_err = run_err` and `continue` to retry the executor with error feedback (same path as CUDA compile errors)
- On final attempt exhaustion: record in `correctness_failure_items` as `"implementation_flaw"`, set `optimized_source = None`

**What it catches:** Everything that would crash NCU — syntax errors, NameError on undefined functions, ImportError on missing modules, RuntimeError from invalid Triton ops, TypeError from wrong constexpr types.

**Cost:** ~100ms per validation (one kernel launch). Saves the much more expensive correctness-agent LLM call (~$0.02 + 5s) when the kernel can't even run.

### Layer 2: Triton API Reference in Executor Prompt

Give the executor the actual list of valid Triton functions from the installed version so it doesn't hallucinate APIs.

**New file** `kernelopt/kernelopt/agents/executor/triton_api_reference.py`:

- `build_triton_api_reference() -> str` — introspects `triton.language`, `triton.language.extra.libdevice` at import time
- Returns a compact reference (~800 tokens) containing:
  - Triton version
  - All valid `tl.*` callable names
  - All valid `libdevice.*` callable names  
  - Hardware constraints: `num_warps` in [1,8], `num_stages` in [1,16], block sizes must be powers of 2, max 1024 threads/block
  - Common mistakes: `triton_helpers.*` doesn't exist in standalone Triton, don't use `reinterpret_tensor` inside `@triton.jit`, cast `tl.int32` to float before mixing with float ops

**Integration** in `executor/prompts.py`:

- Append `build_triton_api_reference()` to `SYSTEM_PROMPT` (cached at module load, not rebuilt per call)

**Reference kernel injection** in `executor/prompts.py:build_initial_message()`:

- New optional `reference_snippets` parameter
- When the plan targets a known pattern (autotune, softmax, matmul, layernorm), include the matching reference kernel from `reference_kernels.py`
- Called from `execute_plans()` which matches `plan["direction"]` against reference kernel names

### Layer 3: Profiling Context Preservation

Pass a compact hardware/profiling summary to the executor so it can validate that plan changes are feasible.

**New helper** in `langgraph_pipeline.py`:

```python
def _build_executor_context(state: GraphState) -> str:
    """Extract a compact profiling summary for the executor (~300 tokens)."""
```

Extracts from `baseline_profiling_context`:
- GPU name, SM count, shared memory per SM
- Baseline latency in ns
- Current launch config: threads/block, registers/thread, waves/SM
- Top 3 NCU rules by speedup potential

**Integration:**

- `execute_plans()` calls `_build_executor_context(state)` once before the plan loop
- Passes result to `run_executor_agent()` via new `profiling_summary` kwarg
- `run_executor_agent()` passes it to `build_initial_message()`
- `build_initial_message()` appends `## Hardware Context\n{profiling_summary}` to the human message

## Files Modified

| File | Change |
|---|---|
| `kernelopt/kernelopt/functions/correctness.py` | Add `validate_kernel_runs()` |
| `kernelopt/kernelopt/langgraph_pipeline.py` | Wire dry-run gate in `execute_plans()`, add `_build_executor_context()`, pass profiling summary |
| `kernelopt/kernelopt/agents/executor/triton_api_reference.py` | **New file** — `build_triton_api_reference()` |
| `kernelopt/kernelopt/agents/executor/prompts.py` | Append API reference to `SYSTEM_PROMPT`, add `profiling_summary` and `reference_snippets` to `build_initial_message()` |
| `kernelopt/kernelopt/agents/executor/__init__.py` | Thread `profiling_summary` kwarg through to `build_initial_message()` |

## What Does NOT Change

- Correctness agent (`agents/correctness.py`) — still runs after the dry-run gate; validates numerical correctness, not just "can it run"
- Planner agent — no changes
- Profiler agent — no changes
- Pipeline graph structure — no new nodes, just code within `execute_plans()`
- Checkpoint serialization — no new state fields

## Verification

1. Unit test `validate_kernel_runs()` with a valid kernel (should pass) and a kernel with `tl.nonexistent()` (should fail with clear error)
2. Unit test `build_triton_api_reference()` — verify it returns a non-empty string containing `tl.load`, `tl.store`, `libdevice.exp`
3. Integration test: run the pipeline on a kernel where the executor is likely to hallucinate (e.g., a synthesized kernel with `triton_helpers` patterns), verify the dry-run gate catches the error and the executor retries successfully
4. Verify the `run_kb_l2_kernel_005` failure case (missing `copy_if_misaligned` import) would be caught by the dry-run gate
5. Verify backward compatibility: existing Triton/CUDA/Helion kernels still optimize correctly
