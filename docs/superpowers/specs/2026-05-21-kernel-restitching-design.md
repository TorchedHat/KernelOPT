# Kernel Re-Stitching: End-to-End Correctness and Performance for Synthesized Models

**Date:** 2026-05-21
**Status:** Draft

## Problem

The pipeline optimizes a single Triton kernel extracted from a multi-kernel compiled model, but never verifies that the optimization is correct or beneficial at the model level. Today:

- Correctness compares standalone kernel vs standalone kernel (fixed in `2c5d0b6`)
- Speedup is kernel-level only — meaningless if the kernel is 1% of total model time
- No end-to-end validation against the original PyTorch eager output

## Goal

After optimizing a kernel, swap the optimized `@triton.jit` body back into the full compiled model (`synthesized_triton.py`), then:

1. **End-to-end correctness**: run the re-stitched model, compare outputs against eager PyTorch
2. **End-to-end timing**: measure wall-clock time of the full re-stitched model vs the unmodified compiled model

## Scope

This applies only to the **PyTorch→synthesis pathway** (when `_original_pytorch_path` is set). Native Triton/CUDA/Helion inputs are unaffected.

## Design

### New function: `restitch_kernel`

**Location:** `kernelopt/kernelopt/synthesis/restitch.py`

**Signature:**
```python
def restitch_kernel(
    synthesized_source: str,
    target_kernel_name: str,
    optimized_body: str,
) -> str:
```

**Behavior:**

1. Find the `async_compile.triton('{target_kernel_name}', '''...''', device_str='cuda')` block in `synthesized_source`.
2. Extract the triple-quoted string (the kernel source).
3. Within that string, find the `@triton.jit` function and everything after it (the kernel body).
4. Replace the kernel body with `optimized_body`.
5. If `optimized_body` contains `@triton.autotune`:
   - Strip the `@triton_heuristics.*` decorator from the string (it conflicts with autotune).
   - Keep the `@triton.autotune` + `@triton.jit` from the optimized body.
6. Return the modified full source.

**What it does NOT change:**
- The `call()` / `Runner.call()` function (buffer management, `.run()` invocations)
- Other Triton kernel definitions
- `extern_kernels` calls
- Imports, `async_compile.wait()`, class structure
- The kernel function name or signature (constrained by Option A)

### New function: `check_model_correctness`

**Location:** `kernelopt/kernelopt/functions/correctness.py`

**Signature:**
```python
def check_model_correctness(
    original_pytorch_path: str,
    restitched_source: str,
    restitched_path: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> tuple[bool, str]:
```

**Behavior:**

1. Run the original PyTorch model (eager) with seeded inputs → reference outputs
2. Write `restitched_source` to `restitched_path`, run it → optimized outputs
3. Compare with `torch.allclose`
4. Clone inputs for each run (in-place kernel fix from `2c5d0b6`)

### Pipeline integration

**Where:** `execute_plans()` in `langgraph_pipeline.py`, after kernel-level correctness passes.

**Flow:**
```
kernel-level correctness passes
    ↓
restitch_kernel(synthesized_source, kernel_name, optimized_body)
    ↓
check_model_correctness(original_pytorch_path, restitched_source)
    ↓
if fails: reject candidate (kernel is correct in isolation but breaks model)
if passes: candidate is valid
```

This is an additional gate, not a replacement for kernel-level correctness.

### End-to-end timing

**Where:** `profile_candidates()` in `langgraph_pipeline.py`.

**Approach:** After NCU profiling the standalone kernel, also time the full re-stitched model with `torch.cuda.Event` wall-clock measurement. Report both:
- Kernel speedup (NCU, apples-to-apples)
- Model speedup (wall-clock, what the user actually gets)

### Executor constraint

The executor prompt already says "Do NOT include imports, kernel_function(), get_inputs(), or __main__". Add an explicit constraint:

> Do NOT change the kernel function name or its parameter signature (names, order, count). Only modify the body and add decorators like @triton.autotune.

This eliminates edge cases 1, 2, 6, 9 from the brainstorm.

## Edge cases

| Case | Handling |
|------|----------|
| `@triton.autotune` added | Strip `@triton_heuristics.*` from the `async_compile.triton()` string, keep the autotune decorator |
| Multiple Triton kernels | Match by `target_kernel_name`, replace only that one |
| Same kernel called N times | One replacement covers all `.run()` calls (same object) |
| No Triton kernels | Early exit (already fixed in `9ddc53c`) |
| Kernel name changed by executor | Rejected by executor constraint — prompt forbids it |
| Signature changed by executor | Rejected by executor constraint — prompt forbids it |
| `extern_kernels` interleaved | Untouched — separate statements in `call()` |
| Buffer reuse / in-place ops | Untouched — in `call()`, not in kernel body |
| 2D/3D grid kernels | `.run()` args unchanged, grid computed by heuristics wrapper |

## Files to modify

1. **New:** `kernelopt/kernelopt/synthesis/restitch.py` — `restitch_kernel()`
2. **Modify:** `kernelopt/kernelopt/functions/correctness.py` — add `check_model_correctness()`
3. **Modify:** `kernelopt/kernelopt/langgraph_pipeline.py` — integrate re-stitch gate in `execute_plans()`, add model timing in `profile_candidates()`
4. **Modify:** `kernelopt/kernelopt/agents/executor/prompts.py` — add signature constraint
5. **New:** `tests_suite/unit/synthesis/test_restitch.py` — unit tests for `restitch_kernel()`

## Out of scope (for now)

- **Parallel top-K kernel optimization** — run sub-agents to optimize the top-K slowest Triton kernels simultaneously, then re-stitch all of them into the model. This is the next step after single-kernel re-stitching is proven.
- Re-stitching for non-PyTorch inputs (native Triton/CUDA/Helion)
- Replacing `extern_kernels` calls with Triton implementations
