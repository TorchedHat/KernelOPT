# Executor Agent Debugging - Changes Summary

## Date: 2026-03-31

## Problem Statement

The executor agent failed to generate any optimized kernel candidates in the most recent pipeline run, despite successfully generating candidates in a previous run. Three high-confidence optimization plans were created but resulted in zero candidates across two iterations.

## Root Cause Investigation

### Evidence Collected

1. **Previous Successful Run** (`/results/` directory):
   - Generated 12+ optimized kernel files
   - All passed correctness checks
   - Successfully applied grid launch optimizations
   - Example: `softmaxtri_iter1_planB_optimize_grid_launch_configuration.py`

2. **Current Failed Run** (`/pipeline_output/` directory):
   - Zero optimized kernel files generated
   - Empty iteration tables in optimization log
   - Baseline latency = Best latency (no improvement)

3. **Plan Complexity Analysis**:
   - Previous successful plans: Modified Python wrapper only
   - Current Plan 1: Requires modifying Triton kernel itself (adding loops for multi-row processing)
   - Hypothesis: Increased complexity exceeded LLM capability with temperature=0.0

## Changes Implemented

### 1. Debug Logging in Executor Agent ✓

**File:** `kernelopt/kernelopt/agents/executor.py`

**Changes:**
- Added logging of raw LLM response length
- Validation checks for kernel definition and launch
- Line count analysis (added/removed)
- Sample output of changed lines
- Exception traceback logging
- Debug info dictionary in execution_result

**Purpose:** Visibility into what code the LLM generates and why it might fail.

### 2. Debug Logging in Pipeline Orchestrator ✓

**File:** `kernelopt/kernelopt/langgraph_pipeline.py`

**Changes in `execute_plans()`:**
- Log when starting each executor attempt
- Display execution status and ready_for_profiling flag
- Show lines changed count
- Log debug info from executor
- Enhanced exception handling with traceback
- Display failure reasons
- Log kernel file save path
- Enhanced correctness check output with failure details

**Purpose:** Track where candidates are being filtered out in the pipeline.

### 3. Debug Logging in Correctness Checker ✓

**File:** `kernelopt/kernelopt/functions/correctness.py`

**Changes in `_check_python_kernel()`:**
- Log subprocess return code
- Display stdout/stderr lengths
- Show combined output preview
- Log which specific failure indicator was triggered
- Display context around failure indicators
- Log success message

**Purpose:** Understand why correctness checks might be failing.

### 4. Debug Logging in Agentic Correctness ✓

**File:** `kernelopt/kernelopt/agents/correctness.py`

**Changes in `run_correctness_agent()`:**
- Log generated test code length and preview
- Display test file path
- Log test return code and output
- Enhanced pass/fail logging

**Purpose:** Debug the LLM-generated correctness tests.

### 5. Temperature Adjustment ✓

**File:** `kernelopt/kernelopt/agents/executor.py` (line 118)

**Change:**
```python
# Before:
temperature=0.0 if attempt == 1 else 0.2,

# After:
temperature=0.2,  # Use 0.2 for all attempts for diversity
```

**Rationale:**
- Temperature=0.0 caused deterministic failures
- With temp=0.2, LLM can try different implementation approaches
- Aligns with default temperature in `llm.py`

### 6. Manual Plan 1 Implementation ✓

**File:** `pipeline_output/manual_plan1_implementation.py`

**Implementation:**
- Based on previous successful approach from `/results/`
- Uses padding strategy instead of modifying kernel
- Calculates optimal grid size (132 or 264 blocks)
- Pads input tensor to match grid size
- Returns only original M rows

**Test Results:**
```
Shape: M=4096, N=1024
✓ Correctness check passed
```

**Significance:**
- Proves the optimization is implementable
- Validates correctness checker works
- Provides reference for what executor should generate

## Debug Output Format

### Executor Debug Output
```
[DEBUG] Executor raw response length: XXXX chars
[DEBUG] After strip_thinking: XXXX chars
[DEBUG] Has kernel definition: True/False
[DEBUG] Has kernel launch: True/False
[DEBUG] First 300 chars:
<code preview>
[DEBUG] Lines changed: XX
[DEBUG] Lines added: XX
[DEBUG] Lines removed: XX
[DEBUG] Sample added lines:
  + <line>
  + <line>
```

### Correctness Debug Output
```
[DEBUG] Correctness subprocess returncode: X
[DEBUG] stdout length: XXX
[DEBUG] stderr length: XXX
[DEBUG] Combined output preview:
<output preview>
[DEBUG] Triggered by indicator: 'FAILED'
[DEBUG] Context: <surrounding text>
[DEBUG] Correctness check PASSED
```

### Pipeline Debug Output
```
[dim]Executing plan A, attempt 1...[/]
[dim]Execution status: success[/]
[dim]Ready for profiling: True[/]
[dim]Lines changed: 25[/]
[dim]Debug info: {...}[/]
[dim]Saved kernel to: path/to/kernel.py[/]
[dim]Running correctness check...[/]
[dim]Correctness message: ...[/]
```

## Testing Strategy

### Test Pipeline Command
```bash
kernelopt run test/softmaxtri.py \
  --output-dir pipeline_output_debug \
  --T 1 --N 2 --K 1 \
  --agentic-correctness false
```

**Parameters:**
- `--T 1`: One iteration only
- `--N 2`: Generate 2 plans
- `--K 1`: One attempt per plan
- `--agentic-correctness false`: Use subprocess mode (faster)

### Expected Outcomes

1. **If executor is generating code:**
   - Will see response length > 0
   - Will see "Has kernel definition: True"
   - Will see lines changed > 0

2. **If correctness is too strict:**
   - Will see "Triggered by indicator: X"
   - Can identify which indicator is causing false positives

3. **If executor is failing to generate:**
   - Will see exception traceback
   - Will see reason in execution_result

## Next Steps Based on Results

### Scenario A: Executor Generates Code, Correctness Fails
- Review which failure indicator is triggering
- Adjust failure indicators list if too broad
- Check if timeout is sufficient
- Review generated code for actual bugs

### Scenario B: Executor Fails to Generate Code
- Review LLM response in debug output
- Check if prompt is clear enough
- Consider adding explicit examples to system prompt
- Verify API key and quotas

### Scenario C: Code Generated and Passes Correctness
- Success! Executor is working
- Previous failure was temporary (API issue, rate limit)
- Can proceed with full pipeline

## Files Modified

1. `kernelopt/kernelopt/agents/executor.py`
2. `kernelopt/kernelopt/langgraph_pipeline.py`
3. `kernelopt/kernelopt/functions/correctness.py`
4. `kernelopt/kernelopt/agents/correctness.py`

## Files Created

1. `EXECUTOR_VERIFICATION.md` - Initial verification report
2. `EXECUTOR_DETAILED_ANALYSIS.md` - Comprehensive analysis
3. `pipeline_output/manual_plan1_implementation.py` - Reference implementation
4. `pipeline_output/executor_prompt.txt` - Sample executor prompt
5. `test_executor_mock.py` - Prompt analysis script
6. `DEBUGGING_SUMMARY.md` - This file

## Verification Checklist

- [x] Debug logging added to executor
- [x] Debug logging added to pipeline orchestrator
- [x] Debug logging added to correctness checker
- [x] Debug logging added to agentic correctness
- [x] Temperature adjusted from 0.0 to 0.2
- [x] Manual Plan 1 implementation created
- [x] Manual implementation passes correctness
- [ ] Test pipeline run with debug output (in progress)
- [ ] Root cause identified from debug logs
- [ ] Fix applied based on root cause
- [ ] Full pipeline test with fix

## Success Criteria

✓ Manual implementation passes correctness  
⏳ Debug output reveals failure point  
⏳ Root cause identified  
⏳ Fix implemented and tested  
⏳ Pipeline generates at least one candidate  

## Current Status

**Phase:** Testing with debug output enabled  
**Command:** Running in background  
**Expected completion:** ~2-5 minutes  
**Log file:** `pipeline_debug.log`  

Once the test completes, we will:
1. Review debug output
2. Identify exact failure point
3. Implement targeted fix
4. Verify fix with another run
