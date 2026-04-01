# Profiler → Planner Verification

## ✅ Test Result: PASSED

The profiler correctly generates `profiling_context` JSON that the planner consumes to generate `optimization_plan` JSON.

---

## Test Flow

```
CSV File (NCU output)
    ↓
parse_ncu_csv() → (sections, rules, launch)
    ↓
build_profiling_context() → profiling_context JSON
    ↓
run_planner(profiling_context) → optimization_plan JSON
```

---

## Test Command

```bash
python3 test_profiler_to_planner.py
```

**Result:** ✅ SUCCESS

---

## Profiler Output: profiling_context JSON

### Top-Level Structure ✅

```json
{
  "kernel_name": "distribution_elementwise_grid_stride_kernel",
  "kernel_name_full": "void at::<unnamed>::distribution_elementwise_grid_stride_kernel<...>",
  "kernel_id": 0,
  "framework_hint": "cuda",
  "device": "0",
  "compute_capability": "9.0",
  "launch": {...},
  "gpu_specs": {...},
  "sections": {...},
  "rules": [...]
}
```

### Launch Config ✅

```json
{
  "block_size": 256,
  "grid_size": 1056,
  "sm_count": 132,
  "waves_per_sm": 1.6,
  "registers_per_thread": 44,
  "dynamic_smem_per_block_bytes": 0,
  "static_smem_per_block_bytes": 0,
  "threads": 270336
}
```

### GPU Specs ✅

```json
{
  "gpu_model": "H100 SXM5",
  "gpu_model_detected": true,
  "peak_hbm_bandwidth_gbs": 3350,
  "peak_fp32_tflops": 67,
  "peak_fp32_tensor_tflops": 989,
  "peak_fp16_tensor_tflops": 1979,
  "peak_bf16_tensor_tflops": 1979,
  "l2_cache_mb": 50,
  "sm_count": 132,
  "max_warps_per_sm": 64
}
```

### Sections (11 total) ✅

- GPU Speed Of Light Throughput
- PM Sampling
- Compute Workload Analysis
- Memory Workload Analysis
- Scheduler Statistics
- Warp State Statistics
- Instruction Statistics
- Launch Statistics
- Source Counters
- GPU and Memory Workload Distribution
- Occupancy

Each section contains metrics with `{value, unit}` structure.

### Rules (10 total) ✅

Top 3 NCU rules by speedup:
1. **LaunchConfiguration**: 50.0% (global)
2. **TheoreticalOccupancy**: 37.5% (local)
3. **CPIStall**: 33.43% (local)

Each rule has: `{name, type, description, speedup_type, speedup_pct}`

---

## Planner Output: optimization_plan JSON

### Structure ✅

```json
{
  "plan_id": "lc-397",
  "direction": "adjust_grid_size_for_full_waves",
  "framework": "triton",
  "evidence": {
    "ncu_rule": "LaunchConfiguration",
    "ncu_rule_speedup_pct": 50.0,
    "supporting_metrics": {
      "Launch Statistics.Grid Size": 1056.0,
      "Launch Statistics.# SMs": 132.0,
      "Launch Statistics.Waves Per SM": 1.6
    },
    "reasoning": "NCU reports 1 full wave plus a partial wave of 397 blocks..."
  },
  "step": {
    "action": "modify",
    "target": "parameter",
    "change": "In run_triton_softmax function, change the grid launch...",
    "predicted_effect": "Grid size will align with SM boundaries..."
  },
  "correctness_constraints": [
    "Must ensure all M rows are still processed...",
    "Kernel logic must check pid < M..."
  ],
  "memory_item_used": null,
  "confidence": "high",
  "confidence_reason": "NCU rule directly points to launch configuration issue..."
}
```

---

## Verification Checklist

| Check | Status | Details |
|-------|--------|---------|
| CSV parsing | ✅ | Parsed 11 sections, 10 rules from baseline.csv |
| profiling_context schema | ✅ | All 10 required fields present |
| launch config | ✅ | All 8 metrics extracted correctly |
| gpu_specs | ✅ | H100 SXM5 detected with all specs |
| sections | ✅ | 11 sections with metric {value, unit} pairs |
| rules | ✅ | 10 rules with speedup percentages |
| Planner received context | ✅ | No errors, parsed successfully |
| Planner generated plan | ✅ | Valid optimization_plan JSON |
| Plan targets NCU rule | ✅ | Targets LaunchConfiguration (50% speedup) |
| Plan has evidence | ✅ | Supporting metrics from profiling_context |
| Plan has concrete step | ✅ | Specific action to modify grid size |
| Plan has constraints | ✅ | 2 correctness constraints specified |

---

## Claude's Reasoning (Planner)

**NCU Rule Identified:** LaunchConfiguration (50% global speedup potential)

**Evidence:**
- Grid size: 1056 blocks
- SM count: 132
- Waves per SM: 1.6 (meaning 1 full wave + 0.6 partial wave)
- Partial wave overhead: 397 blocks wasted

**Optimization Direction:** Adjust grid size to be multiple of 132 (SM count)

**Predicted Effect:** Eliminate partial wave overhead → 50% speedup

**Correctness Constraints:**
1. Must process all M rows
2. Must check `pid < M` to avoid out-of-bounds

---

## Data Flow Validation

### Input → Profiler
```
baseline.csv (3.2 MB NCU export)
```

### Profiler → Planner
```json
{
  "kernel_name": "...",
  "launch": { "grid_size": 1056, "sm_count": 132, ... },
  "gpu_specs": { "peak_hbm_bandwidth_gbs": 3350, ... },
  "sections": { "GPU Speed Of Light": {...}, ... },
  "rules": [ { "name": "LaunchConfiguration", "speedup_pct": 50.0 }, ... ]
}
```

### Planner → Executor (next step)
```json
{
  "plan_id": "lc-397",
  "direction": "adjust_grid_size_for_full_waves",
  "step": {
    "action": "modify",
    "target": "parameter",
    "change": "In run_triton_softmax function, change the grid launch..."
  }
}
```

---

## Files Generated

1. **profiling_context_test.json** (30 KB)
   - Full profiling context with all sections, rules, metrics
   - Ready for planner consumption

2. **optimization_plan_test.json** (1 KB)
   - Full optimization plan from Claude
   - Ready for executor consumption

---

## Conclusion

✅ **The profiler generates correct output for the planner:**

1. **CSV Parser** correctly extracts sections, rules, and launch config from NCU CSV
2. **build_profiling_context()** correctly assembles the profiling_context JSON with all required fields
3. **GPU specs** are automatically detected and included
4. **Planner** successfully receives profiling_context and generates a valid optimization_plan
5. **Claude** uses the profiling data to identify the top NCU rule and propose a concrete optimization

**The data flow is correct and working!** 🎯

---

## Next Steps

To test the full pipeline (Profiler → Planner → Executor → Profiler):

```bash
# Run full optimization with 2 iterations, 4 plans per iteration
kernelopt run test/softmaxtri.py \
  --kernel-name softmax_kernel \
  --kernel-args '--profile-only' \
  -T 2 -N 4 \
  -o results \
  --verbose
```

This will exercise the complete flow:
- Profiler generates profiling_context
- Planner generates optimization_plans (x4)
- Executor rewrites kernels (x4)
- Correctness validates each kernel
- Profiler profiles valid candidates
- Summarizer extracts experience
- Repeat for iteration 2
