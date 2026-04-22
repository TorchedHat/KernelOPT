# Design: CUDA Agentic Correctness — Exit-Code Contract + matmul Test Fixture

**Date:** 2026-04-21
**Status:** Approved

---

## Problem

`_check_cuda_with_llm_reference` in `kernelopt/kernelopt/functions/correctness.py` uses hardcoded output string checks (`"PASS"`, `"FAIL"`) to determine correctness, which is fragile for the same reason the previous `check_correctness` and `_has_builtin_self_test` fixes addressed: the LLM is free to vary output wording, and substring matches against short strings like `"FAIL"` are error-prone.

Additionally, there is no `.cu` test fixture without a `main()` to exercise the agentic CUDA correctness path in tests.

---

## Solution

Two changes:

1. Replace string-based pass/fail detection with exit-code-only semantics in `_check_cuda_with_llm_reference`, and update the system prompt to produce richer diagnostic output on failure instead of mandating specific output strings.
2. Add a `matmul_cuda.cu` fixture (kernel-only, no `main()`) and a test class that exercises the agentic CUDA correctness path end-to-end.

---

## Architecture

No structural changes. All changes are within existing files and one new fixture.

```
_check_cuda_with_llm_reference()   ← fix string checks → exit-code-only
_CUDA_LLM_SYSTEM_PROMPT            ← update points 7–9
tests_suite/fixtures/kernels/matmul_cuda.cu   ← new fixture (kernel-only)
tests_suite/component/functions/test_correctness.py  ← new test class
```

---

## Change 1: `_check_cuda_with_llm_reference` — exit-code-only

**File:** `kernelopt/kernelopt/functions/correctness.py`

Replace lines 151–157 (current string-check block):

```python
# BEFORE
if run_result.returncode != 0:
    return False, f"Runtime error (exit {run_result.returncode}):\n{combined[:500]}"
if "FAIL" in combined:
    return False, f"Correctness FAILED:\n{combined[:500]}"
if "PASS" not in combined:
    return False, f"No 'Correctness: PASS' marker found in output:\n{combined[:500]}"
return True, "CUDA correctness passed (LLM-generated C harness)."

# AFTER
if run_result.returncode != 0:
    return False, f"Correctness failed (exit {run_result.returncode}):\n{combined[:500]}"
return True, f"CUDA correctness passed.\n{combined[:200]}"
```

Exit code is the sole pass/fail contract. Output is included in messages only as diagnostic context.

---

## Change 2: `_CUDA_LLM_SYSTEM_PROMPT` — remove hardcoded string requirements

**File:** `kernelopt/kernelopt/functions/correctness.py`

Replace points 7–8 in the system prompt:

```
# REMOVE:
7. Compare element-wise: if max absolute diff > 1e-5, print "Correctness: FAIL" and exit(1)
8. Otherwise print "Correctness: PASS" and return 0

# REPLACE WITH:
7. Compare element-wise against the CPU reference using max absolute diff > 1e-5 as the threshold
8. On failure: print the first failing index, expected value, actual value, and max absolute diff
   across all elements, then call exit(1)
9. On success: call exit(0). No specific output string is required.
```

Also update the docstring of `_check_cuda_with_llm_reference` to reflect exit-code contract.

---

## Change 3: `matmul_cuda.cu` fixture

**File:** `tests_suite/fixtures/kernels/matmul_cuda.cu` *(new)*

Tiled matrix multiplication kernel (C = A × B, fp32, square N×N matrices). No `main()`, no correctness check — this is a pure kernel file that exercises `_has_builtin_self_test → False` → agentic path.

Implementation: standard tiled SMEM matmul with configurable tile size via `TILE_SIZE` define. Simple enough that the LLM-generated CPU reference (`C[i][j] = sum_k A[i][k] * B[k][j]`) is unambiguous.

---

## Change 4: Test class

**File:** `tests_suite/component/functions/test_correctness.py` *(extend)*

New class `TestCudaAgenticCorrectness`:

- `test_correct_matmul_passes` — reads `matmul_cuda.cu`, calls `check_correctness(..., use_agentic=True)`, asserts `passed == True`
- `test_broken_matmul_fails` — same kernel with a deliberate numerical error (e.g. off-by-one in accumulation), asserts `passed == False`

Both tests use `patch_build_llm` from `conftest.py` for unit runs and are marked `requires_llm` for integration runs against the real model.

---

## Error Handling

- LLM fails to generate harness → existing `except Exception` path returns `(False, "LLM failed...")`
- Harness fails to compile → existing compile check returns `(False, "LLM-generated harness failed to compile:...")`
- Harness exits non-zero → new check returns `(False, "Correctness failed (exit N):...")`
- Harness exits 0 → `(True, "CUDA correctness passed.\n<first 200 chars of output>")`

---

## Verification

1. Run existing CUDA correctness tests: `pytest tests_suite/component/functions/test_correctness.py -v`
2. Run new test class (unit, mocked LLM): `pytest tests_suite/component/functions/test_correctness.py::TestCudaAgenticCorrectness -v`
3. Run with real LLM (integration): `pytest tests_suite/component/functions/test_correctness.py::TestCudaAgenticCorrectness -m requires_llm --gpu 0`
4. Confirm no `"PASS"`/`"FAIL"` string checks remain in `_check_cuda_with_llm_reference`
