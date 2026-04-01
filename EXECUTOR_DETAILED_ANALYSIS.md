# Detailed Executor Agent Analysis

## Summary

**Finding:** The executor agent has a **fundamental correctness problem** - it failed to generate ANY candidates in the most recent run (`/pipeline_output/`), despite successfully generating candidates in a previous run (`/results/`).

## Evidence Comparison

### Previous Successful Run (`/results/`)
- **Date:** March 31, 2026 06:47-06:57
- **Generated:** 12+ optimized kernel files
- **Iterations:** 3 complete
- **Candidates:** Multiple per iteration (planA, planB, planC, planD)
- **Correctness:** All candidates passed
- **Profiling:** All profiled with NCU
- **Speedup:** Minimal (1.01x), but pipeline functioned correctly

### Current Failed Run (`/pipeline_output/`)
- **Generated:** 0 optimized kernel files (only best_kernel.py = unchanged baseline)
- **Iterations:** 2 attempted
- **Candidates:** ZERO in both iterations
- **Optimization log:** Empty tables
- **Speedup:** 1.0x (no change)

## What We Know Works

From `/results/softmaxtri_iter1_planB_optimize_grid_launch_configuration.py`, the executor successfully implemented a grid launch optimization:

```python
# OPTIMIZED: Round grid size to eliminate partial waves
# Original: grid size was M (4096) causing 3.88 waves per SM with tail effect
SM_COUNT = 132
TARGET_WAVES = 4
OPTIMAL_GRID_SIZE = SM_COUNT * TARGET_WAVES  # 528

# Round M to nearest multiple of optimal grid size
grid_size = ((M + OPTIMAL_GRID_SIZE - 1) // OPTIMAL_GRID_SIZE) * OPTIMAL_GRID_SIZE

# ... (padding logic)

softmax_kernel[(grid_size,)](  # Changed from [(M,)]
```

**Key observations:**
1. ✅ Executor CAN make substantial code changes
2. ✅ Executor CAN add multi-line logic and comments
3. ✅ Executor CAN change grid launch parameters
4. ✅ Generated code CAN pass correctness checks

## Current Plan vs. Previous Plan

### Previous Plan (iter1_planB) Approach:
- Rounded grid size to optimal value
- Padded input tensor
- **Did NOT modify kernel** - still 1 row per block
- Changes only in `run_triton_softmax()` function

### Current Plan 1 Requirement:
> "Adjust grid launch to use exactly 132 blocks (one per SM) or 264 blocks (2 per SM) instead of current M blocks, **requiring kernel to process multiple rows per thread block**"

**This is MORE complex** - it requires:
1. Changing grid launch to fixed size (132 or 264)
2. **Modifying the kernel itself** to process multiple rows
3. Adding loop/striding logic inside the @triton.jit kernel
4. Calculating which rows each block should process

## Possible Root Causes

### Hypothesis 1: Plan Complexity Exceeded LLM Capability
The current plan requires modifying the kernel itself (adding a loop inside the @triton.jit function), which is more complex than the previous approach.

**Evidence FOR:**
- Previous successful plan only modified the Python wrapper
- Current plan requires Triton kernel modification
- Temperature=0.0 for first attempt may cause deterministic failure

**Evidence AGAINST:**
- LLM should be capable of adding loops
- The prompt includes framework-specific guidance

### Hypothesis 2: Correctness Check Too Strict
The generated code fails the correctness check, preventing it from becoming a candidate.

**Evidence FOR:**
- Modifying kernel logic has higher risk of correctness bugs
- Multiple rows per block could cause off-by-one errors
- The correctness check output shows failures in some test runs

**Evidence AGAINST:**
- Previous run had 100% pass rate on correctness
- Simple grid launch changes shouldn't break correctness

### Hypothesis 3: Executor Exception/Crash
The executor crashes or throws an exception before producing output.

**Evidence FOR:**
- Zero candidates in BOTH iterations
- No partial successes
- Lines 198-203 in `langgraph_pipeline.py` catch exceptions

**Evidence AGAINST:**
- Should see error messages in console
- Exception handling should log the error

### Hypothesis 4: API/Model Configuration Change
Something changed between the two runs (API key, model, quotas).

**Evidence FOR:**
- Different run times (previous: early March 31, current: later March 31)
- Same codebase, different results

**Evidence AGAINST:**
- Planner successfully generated 3 plans
- Only executor failed

## The Executor Prompt

Generated prompt for Plan 1:

```
## Framework: triton

## Optimization Plan
{
  "direction": "reduce_partial_wave_tail_effect",
  "step": {
    "change": "Adjust grid launch to use exactly 132 blocks (one per SM) or 264 blocks (2 per SM) 
              instead of current M blocks, requiring kernel to process multiple rows per thread block"
  },
  ...
}

## Current Kernel Source
[155 lines of code]

Apply the optimization plan and output the complete optimized kernel source.
```

**Prompt Analysis:**
- ✅ Clear plan description
- ✅ Kernel source provided
- ✅ Framework specified
- ⚠️ Instruction emphasizes "requiring kernel to process multiple rows per thread block"
- ⚠️ Does not provide explicit example of how to modify Triton kernel for multi-row processing
- ⚠️ Temperature=0.0 means no randomness - will fail the same way every time

## Code Issues in executor.py

### Issue 1: No Verification Loop
```python
# executor.py:159-186
optimized_source, execution_result = run_executor(...)

# If ready_for_profiling is False, candidate is silently dropped
if not execution_result.get("ready_for_profiling", False):
    console.print(f"[red]Plan {plan_letter}: {execution_result.get('status', 'failed')}[/]")
    continue
```

**Problem:** No retry logic, no debugging output showing what the LLM generated

### Issue 2: Insufficient Plan Interpretation Guidance
```python
# executor.py:19-36
"""
### Step 1 — Understand the plan
Read optimization_plan.step.change literally.

### Step 2 — Locate the target in the source
Find optimization_plan.step.target in the kernel.

### Step 3 — Apply the transformation
Make ONLY the change described in the plan.
"""
```

**Problem:** For complex transformations like "process multiple rows per thread block," the executor needs more specific guidance on HOW to implement this in Triton.

### Issue 3: Temperature Settings
```python
# executor.py:118
temperature=0.0 if attempt == 1 else 0.2,
```

**Problem:** With temp=0.0 and K=1, if the LLM fails once, it will fail every time with the same prompt.

## Next Steps to Debug

1. **Add Console Logging**
   - Log execution_result for every plan attempt
   - Print LLM-generated code even if it fails
   - Show why ready_for_profiling is False

2. **Run Single Executor Test**
   - Manually call run_executor() with Plan 1
   - Capture the generated code
   - Check if it's syntactically valid Triton
   - Run correctness check manually

3. **Check API Logs**
   - Verify LLM API calls are succeeding
   - Check if rate limits or quotas were hit
   - Look for error responses from Anthropic API

4. **Simplify Plan 1**
   - Test with the simpler approach (padding like previous run)
   - See if executor can handle that

5. **Increase Temperature**
   - Try with temperature=0.3 for attempt 1
   - Add more diversity to implementations

## Recommended Fixes

### Fix 1: Add Detailed Logging
```python
# In langgraph_pipeline.py:execute_plans
console.print(f"[dim]Executor response preview:[/] {optimized_source[:200]}...")
console.print(f"[dim]Execution result:[/] {json.dumps(execution_result, indent=2)}")
```

### Fix 2: Enhance Executor Prompt
Add explicit examples for multi-row processing:
```python
# In executor.py system prompt
"""
Example: To process multiple rows per block in Triton:

# Calculate rows per block
ROWS_PER_BLOCK = tl.constexpr(4)
block_id = tl.program_id(0)
base_row = block_id * ROWS_PER_BLOCK

# Loop over rows
for row_offset in range(ROWS_PER_BLOCK):
    row_idx = base_row + row_offset
    if row_idx < M:
        # process row_idx
"""
```

### Fix 3: Add Retry with Higher Temperature
```python
# In langgraph_pipeline.py
for attempt in range(1, cfg.K + 1):
    temp = 0.0 if attempt == 1 else 0.3  # Increase from 0.2
```

### Fix 4: Verify Generated Code
```python
# After run_executor
if "softmax_kernel[" not in optimized_source:
    console.print("[yellow]Warning: Generated code missing kernel launch[/]")
```

## Conclusion

The executor agent is **not broken** - it has successfully generated optimized kernels before. However, it **failed completely** in the current run, producing zero candidates across two iterations.

The most likely cause is that the current optimization plans require more complex kernel modifications than the previous run, and the executor's prompt doesn't provide sufficient guidance for these transformations.

Without access to the actual LLM responses, we cannot determine if:
- The LLM generated invalid code
- The code failed correctness checks
- The executor crashed with an exception
- The API calls failed

**Immediate action required:** Add debug logging to capture executor outputs and failure reasons.
