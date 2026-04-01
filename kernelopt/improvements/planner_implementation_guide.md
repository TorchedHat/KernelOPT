# How to Implement Enhanced Planner

## Step 1: Update System Prompt

**File**: `kernelopt/kernelopt/agents/planner.py`

**Changes**:

1. Insert the architecture-specific knowledge after line 62 (after roofline analysis section)
2. Add the advanced optimization patterns section
3. Add the enhanced reasoning framework

**Concrete additions** (insert after line 62):

```python
PLANNER_SYSTEM_PROMPT = """\
You are the Planner Agent in a GPU kernel optimization pipeline.

[... existing content ...]

Step 2 — Cross-reference with raw metrics

[... existing roofline analysis ...]

## Advanced Architecture-Specific Optimizations

### Hopper (H100/H200) Architecture Features

When the GPU is H100 or H200 (check gpu_specs.gpu_model), consider these Hopper-specific optimizations:

1. **Async Copy Pipeline (num_stages parameter)**
   - Available in: Triton kernels
   - When to use: Memory-bound kernels (Memory Throughput > SM Throughput)
   - How: Add num_stages parameter to kernel launch, try values [2, 3, 4]
   - Effect: Overlaps memory copy with compute, hides memory latency
   - Evidence: NCU shows high "No Eligible Warps %" (>30%)
   - Example: Change from `kernel[(grid,)](args, BLOCK_SIZE=1024, num_warps=8)` 
             to `kernel[(grid,)](args, BLOCK_SIZE=1024, num_warps=8, num_stages=4)`

2. **Increased num_warps for Latency Hiding**
   - H100 can support more concurrent warps effectively
   - When: Theoretical occupancy is 100% but achieved is <70%
   - How: Increase num_warps from 4→8 or 8→16
   - Constraint: Check Block Limit Warps is not the bottleneck

3. **Warp Specialization**
   - Use different warps for memory vs compute operations
   - When: High warp stall percentage (>40%)
   - How: Split kernel into producer/consumer roles using tl.program_id(1)

4. **L2 Cache Compression**
   - Hopper has hardware L2 compression
   - When: NCU rule "MemoryL2Compression" appears with >15% speedup
   - How: Use data types FP16/BF16, ensure aligned access patterns

### Memory Access Optimizations

**Pattern: Vectorized Loads (128-bit)**
- When: Memory-bound + non-coalesced access pattern
- Evidence: "UncoalescedGlobalAccess" NCU rule OR low L1 hit rate
- Triton: Use larger BLOCK_SIZE (power of 2), Triton auto-vectorizes
- CUDA: Use float4 instead of float

**Pattern: Async Barriers**
- When: Memory latency dominates (high "Warp Stall Long Scoreboard")
- Triton: Use tl.async_wait() between load and compute
- Structure: Load next tile → async_wait(0) → compute current → async_wait(1)

### Compute Optimizations

**Pattern: Instruction Fusion**
- When: Compute-bound (SM Throughput > Memory Throughput)
- Evidence: IPC Active < 1.0 or "IssueSlotUtilization" rule
- How: Fuse operations in single expressions
  - BAD:  `x = x - row_max; exp_x = tl.exp(x)`
  - GOOD: `exp_x = tl.exp(x - row_max)`
  - BAD:  `softmax = exp_x / denom`
  - GOOD: `inv_denom = 1.0 / denom; softmax = exp_x * inv_denom`

**Pattern: Fast Math Reciprocal**
- Replace division with reciprocal multiply
- `y = x / d` → `inv_d = 1.0 / d; y = x * inv_d`
- Saves cycles on FP32 units

### Occupancy Optimizations

**Pattern: Reduce Register Pressure**
- When: "Block Limit Registers" is the occupancy constraint
- Evidence: Registers Per Thread > 40 (CUDA) or > 32 (Triton)
- How: Reduce tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)
  - For softmax: BLOCK_SIZE from 2048 → 1024
  - For matmul: BLOCK_M from 128 → 64

**Pattern: Eliminate Partial Wave Tail**
- When: "LaunchConfiguration" NCU rule with "partial wave" in description
- Evidence: waves_per_sm is not an integer (e.g., 3.88)
- How: Adjust grid size to be multiple of SM count
  - H100 SXM5: 132 SMs → use grid_size ∈ {132, 264, 528, 1056, ...}
  - May require padding input tensors

### Anti-Patterns (Learn from Negative Memory Items)

If optimization_memory contains items with `rewrite_type: "negative"`, explicitly AVOID those directions.

Common anti-patterns:
- Doubling tile size when registers are already constrained → causes spilling
- Increasing num_warps when shared memory is the limiter → no improvement
- Adding async copy when kernel is compute-bound → overhead without benefit

---

Step 3 — Look at the kernel source

[... rest of existing prompt ...]
```

## Step 2: Add Multi-Step Reasoning

After the optimization patterns, add explicit reasoning steps:

```python
## Enhanced Reasoning Framework

Before generating a plan, walk through these steps explicitly in your `reasoning` field:

1. **Bottleneck Classification**
   - Compare Memory Throughput % vs SM Throughput %
   - Classify as: memory-bound, compute-bound, or balanced

2. **Quantify Headroom**
   - Memory-bound: 100% - Memory Throughput = available headroom
   - Compute-bound: 100% - SM Throughput = available headroom

3. **Root Cause Diagnosis**
   - Check DRAM Throughput (>30% = HBM bandwidth limit)
   - Check L1/L2 hit rates (low = cache thrashing)
   - Check Achieved Occupancy (<70% = under-utilized)
   - Check Warp Stall % (>40% = latency not hidden)

4. **Architecture-Specific Opportunities**
   - Does GPU have Hopper features? (num_stages, TMA)
   - Is framework Triton? (can use autotune, num_stages)
   - Is framework CUDA? (can use __ldg, texture cache)

5. **Pattern Selection**
   - Match bottleneck + root cause to optimization patterns above
   - Consult optimization_memory for similar profiling signals

6. **Impact Estimation**
   - Back-of-envelope: If we reduce memory traffic by X%, latency improves by Y%
   - Conservative estimate: Assume 50-70% of theoretical gain

7. **Constraints Check**
   - Correctness: Does this preserve semantics?
   - Resources: Does this fit in registers/SMEM?
   - Framework: Is the feature available?

8. **Generate Concrete Plan**
   - Exact code changes needed
   - Predicted metric changes
   - Success criteria

Example reasoning field output:
```json
{
  "reasoning": "Bottleneck: Memory-bound (Memory 54% > SM 49%). Headroom: 46% below peak. Root cause: DRAM Throughput 32% indicates HBM bandwidth limit, not cache misses. Architecture: H100 SXM5 has Tensor Memory Accelerator (TMA) for async copy. Pattern: Async pipeline with num_stages can hide memory latency. Impact: With 4-stage pipeline, can overlap 75% of memory fetch time with compute. Conservative estimate: 10-15% speedup. Constraints: Requires Triton ✓, Hopper ✓. Plan: Add num_stages=4 to kernel launch."
}
```
```

## Step 3: Enhance Output Schema

Add these fields to the plan JSON output:

```python
# In OUTPUT FORMAT section, add after "predicted_effect":

  "step": {
    "action": "...",
    "target": "...",
    "change": "...",
    "predicted_effect": "...",
    
    "implementation_hints": [
      "Specific code snippets or parameter values to guide executor"
    ],
    
    "success_criteria": {
      "primary_metric": "Memory Throughput",
      "expected_direction": "decrease|increase|stable",
      "expected_magnitude": "10-20%",
      "secondary_effects": "SM Throughput may increase"
    }
  },
```

## Step 4: Add Examples to Prompt

Add 2-3 concrete examples at the end of the system prompt:

```python
---

EXAMPLE PLANS

Example 1: Memory-Bound Softmax on H100

Profiling signals:
- Memory Throughput: 54.7%, SM Throughput: 49.4%
- DRAM Throughput: 31.6%
- Achieved Occupancy: 86.2%
- NCU rule: "CPIStall" 31.3% speedup

Plan:
{
  "plan_id": "async_pipeline_001",
  "direction": "enable_async_copy_pipeline",
  "framework": "triton",
  "evidence": {
    "ncu_rule": "CPIStall",
    "ncu_rule_speedup_pct": 31.3,
    "supporting_metrics": {
      "GPU Speed Of Light Throughput.Memory Throughput": 54.7,
      "GPU Speed Of Light Throughput.Compute (SM) Throughput": 49.4,
      "Occupancy.Achieved Occupancy": 86.2
    },
    "reasoning": "Memory-bound kernel (Memory 54.7% > SM 49.4%). H100 Hopper architecture supports async copy via num_stages parameter. CPIStall rule indicates cycles wasted waiting on memory. Async pipeline can hide memory latency by overlapping loads with compute."
  },
  "step": {
    "action": "add",
    "target": "kernel launch parameter",
    "change": "Add num_stages=4 to softmax_kernel launch configuration",
    "predicted_effect": "Reduces warp stall time by overlapping memory fetch with compute. Memory Throughput may remain ~55% but SM Throughput should increase to ~60-65%. Overall latency reduction: 10-15%",
    "implementation_hints": [
      "In run_triton_softmax function, modify kernel launch:",
      "OLD: softmax_kernel[(M,)](X, Y, M, N, ..., BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)",
      "NEW: softmax_kernel[(M,)](X, Y, M, N, ..., BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=4)",
      "Note: num_stages only works on Hopper GPUs (H100/H200)"
    ],
    "success_criteria": {
      "primary_metric": "Warp Stall Cycles",
      "expected_direction": "decrease",
      "expected_magnitude": "15-25%",
      "secondary_effects": "SM Busy percentage should increase"
    }
  },
  "correctness_constraints": [
    "num_stages is a performance hint, does not affect correctness",
    "Triton compiler validates GPU support automatically"
  ],
  "architecture_notes": "Hopper's Tensor Memory Accelerator (TMA) enables async copy with software pipelining via num_stages parameter",
  "memory_item_used": null,
  "confidence": "high",
  "confidence_reason": "NCU CPIStall rule directly points to memory latency hiding. num_stages is designed for this exact scenario on Hopper."
}

Example 2: Register-Bound Matmul

[Similar detailed example for register pressure reduction]
```

## Step 5: Temperature and Model Selection

Consider using a higher-tier model or higher temperature for the planner:

```python
# In run_planner function (line 193):
llm = build_llm(
    model=model,
    api_key=api_key,
    max_tokens=4096,
    temperature=0.3 + (plan_index * 0.1),  # ← Already good!
    # Consider: Use opus for complex reasoning
    # model = "claude-opus-4@20250514" for plan_index == 0
)
```

## Step 6: Add Profiling Context Enrichment

Before passing profiling_context to planner, compute derived metrics:

```python
def enrich_profiling_context(ctx: dict) -> dict:
    """Add derived metrics to help planner reasoning."""
    sol = ctx['sections'].get('GPU Speed Of Light Throughput', {})
    
    # Compute bottleneck classification
    mem_throughput = sol.get('Memory Throughput', {}).get('value', 0)
    sm_throughput = sol.get('Compute (SM) Throughput', {}).get('value', 0)
    
    ctx['derived'] = {
        'bottleneck': 'memory' if mem_throughput > sm_throughput else 'compute',
        'bottleneck_margin': abs(mem_throughput - sm_throughput),
        'memory_headroom_pct': 100 - mem_throughput,
        'compute_headroom_pct': 100 - sm_throughput,
        'is_balanced': abs(mem_throughput - sm_throughput) < 5,
    }
    
    # Compute roofline metrics
    launch = ctx['launch']
    gpu_specs = ctx['gpu_specs']
    
    # Estimate arithmetic intensity
    # (requires kernel-specific knowledge, this is placeholder)
    ctx['derived']['estimated_ai'] = 'N/A'  # TODO: compute from kernel analysis
    
    return ctx

# Use in run_planner:
profiling_context = enrich_profiling_context(profiling_context)
```

## Step 7: Few-Shot Learning from Memory

When optimization_memory is provided, extract patterns:

```python
def extract_memory_patterns(memory: list[dict]) -> str:
    """Summarize successful patterns from memory."""
    positive = [m for m in memory if m.get('rewrite_type') == 'positive']
    negative = [m for m in memory if m.get('rewrite_type') == 'negative']
    
    summary = "## Learned Patterns from Memory\n\n"
    
    if positive:
        summary += "### Successful Patterns (APPLY):\n"
        for item in positive[:3]:  # Top 3
            summary += f"- {item['direction']}: {item.get('strategy_description', 'N/A')}\n"
            summary += f"  Speedup: {item['speedup']:.2f}x\n"
    
    if negative:
        summary += "\n### Failed Patterns (AVOID):\n"
        for item in negative[:3]:
            summary += f"- {item['direction']}: {item.get('strategy_description', 'N/A')}\n"
            summary += f"  Caused regression: {item['speedup']:.2f}x\n"
    
    return summary

# Add to human_msg in run_planner:
if optimization_memory:
    memory_patterns = extract_memory_patterns(optimization_memory)
    human_msg += f"\n\n{memory_patterns}"
```

---

## Testing the Enhanced Planner

Run with a known bottleneck:

```bash
# Memory-bound kernel
kernelopt plan pipeline_output/baseline.csv test/softmax_simple.py -N 3

# Should generate sophisticated plans like:
# - Plan 1: Enable async copy pipeline (num_stages=4)
# - Plan 2: Vectorized loads with eviction policy
# - Plan 3: Reduce grid size for better cache utilization
```

Compare plan quality:
- Does it reference specific H100 features?
- Does it provide concrete implementation hints?
- Does it quantify expected impact?
- Does reasoning chain make sense?
