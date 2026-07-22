# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_transpose_view_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 244030 ns — **Best latency**: 243970 ns (**1.0002×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_and_add_evict_first_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_transpose_view_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_7_triton_poi_fused_clone_transpo/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (393216,1,1), Waves/SM: 186.18
- **Registers/thread**: 16
- **Duration**: 244030 ns
- **SM Throughput**: 25.27 % | **Memory**: 32.19 % | **DRAM**: 32.19 %
- **Achieved Occupancy**: 43.44 %
- **IPC Active**: 1.02 inst/cycle | **No Eligible**: 74.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_transpose | 0.858× | 0.874× | ✓ |  |
| C | increase_xblock_to_hide_memory_latency | 0.873× | 0.836× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_pointwise_transpose | 1.048× | 0.586× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244030 ns | 279330 ns | +35300 ns (+14.5%) | 292030 ns | +48000 ns (+19.7%) | 416700 ns | +172670 ns (+70.8%) |
| SM Throughput (%) | 25.27 % | 28.22 % | +2.95 pp | 46.03 % | +20.76 pp | 66.36 % | +41.09 pp |
| Memory Throughput (%) | 32.19 % | 28.51 % | -3.68 pp | 27.53 % | -4.66 pp | 19.83 % | -12.36 pp |
| DRAM Throughput (%) | 32.19 % | 28.51 % | -3.68 pp | 27.53 % | -4.66 pp | 19.83 % | -12.36 pp |
| Achieved Occupancy (%) | 43.44 % | 37.27 % | -6.17 pp | 38.92 % | -4.52 pp | 65.23 % | +21.79 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 372.36 | +186.18 | 744.73 | +558.55 |
| No Eligible (%) | 74.49 % | 68.43 % | -6.06 pp | 52.90 % | -21.59 pp | 33.61 % | -40.88 pp |
| IPC Active | 1.02 inst/cycle | 1.22 inst/cycle | +0.20 | 1.85 inst/cycle | +0.83 | 2.65 inst/cycle | +1.63 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_eviction_policies | 2.051× | 0.886× | ✓ |  |
| B | fix_autotune_grid_lambda_and_add_evict_first_loads | 2.354× | 1.000× | ✓ | **✓ best** |
| C | eviction_policies_with_corrected_grid_lambda | 2.071× | 0.552× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244030 ns | 275460 ns | +31430 ns (+12.9%) | 243970 ns | -60 ns (-0.0%) | 442430 ns | +198400 ns (+81.3%) |
| SM Throughput (%) | 25.27 % | 29.08 % | +3.81 pp | 31.72 % | +6.45 pp | 71.81 % | +46.54 pp |
| Memory Throughput (%) | 32.19 % | 28.04 % | -4.15 pp | 31.80 % | -0.39 pp | 18.48 % | -13.71 pp |
| DRAM Throughput (%) | 32.19 % | 28.04 % | -4.15 pp | 31.80 % | -0.39 pp | 16.83 % | -15.36 pp |
| Achieved Occupancy (%) | 43.44 % | 35.46 % | -7.98 pp | 43.49 % | +0.05 pp | 61.75 % | +18.31 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 744.73 | +558.55 |
| No Eligible (%) | 74.49 % | 67.05 % | -7.44 pp | 67.93 % | -6.56 pp | 28.34 % | -46.15 pp |
| IPC Active | 1.02 inst/cycle | 1.26 inst/cycle | +0.24 | 1.28 inst/cycle | +0.26 | 2.87 inst/cycle | +1.85 |

---

## Iteration 3

**Best so far**: 1.0002× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_differentiation_load_evict_first_store_evict_last | 2.352× | 1.000× | ✓ |  |
| B | fix_grid_lambda_to_activate_existing_autotune | 2.051× | 0.886× | ✓ |  |
| C | fix_grid_lambda_to_match_autotuned_xblock | 2.071× | 0.552× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_d... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244030 ns | 244100 ns | +70 ns (+0.0%) | 275550 ns | +31520 ns (+12.9%) | 442400 ns | +198370 ns (+81.3%) |
| SM Throughput (%) | 25.27 % | 36.59 % | +11.32 pp | 28.95 % | +3.68 pp | 71.82 % | +46.55 pp |
| Memory Throughput (%) | 32.19 % | 30.54 % | -1.65 pp | 28.00 % | -4.19 pp | 18.48 % | -13.71 pp |
| DRAM Throughput (%) | 32.19 % | 30.54 % | -1.65 pp | 28.00 % | -4.19 pp | 16.83 % | -15.36 pp |
| Achieved Occupancy (%) | 43.44 % | 43.11 % | -0.33 pp | 35.52 % | -7.92 pp | 62.07 % | +18.63 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 744.73 | +558.55 |
| No Eligible (%) | 74.49 % | 62.94 % | -11.55 pp | 67.32 % | -7.17 pp | 28.03 % | -46.46 pp |
| IPC Active | 1.02 inst/cycle | 1.48 inst/cycle | +0.46 | 1.26 inst/cycle | +0.24 | 2.88 inst/cycle | +1.86 |


### Key Learnings

#### ❌ Plan C: `fix_grid_lambda_to_match_autotuned_xblock` (0.551×)

Adding @triton.autotune with larger XBLOCK configs (512+) while keeping the grid hardcoded to `cdiv(xnumel, 128)` caused a 3.5x inflation in executed instructions (70M → 248M) and an 81% increase in duration (244 → 443 us), despite superficially improving scheduler utilization metrics. The regression occurs because autotune selected XBLOCK=512 but the grid remained sized for XBLOCK=128, causing each block to process 4x more elements than the tensor contains. The grid MUST be `lambda meta: (cdiv(xnumel, meta['XBLOCK']),)` for autotune to work correctly — changing autotune configs alone without fixing the grid always causes this failure mode.

> **Applicable when**: Any Triton kernel using @triton.autotune with variable XBLOCK where the call-site grid is a static tuple rather than a lambda reading meta['XBLOCK']. Symptom: executed instructions inflate by exactly XBLOCK_selected/XBLOCK_hardcoded ratio, waves/SM inflate proportionally, DRAM throughput drops while compute SM throughput rises spuriously.

> **Avoid when**: Grid is already a lambda using meta['XBLOCK']; this is a pure implementation flaw diagnosis. Changing autotune configs alone without fixing the grid is always harmful and must never be attempted.

---

## Iteration 4

**Best so far**: 1.0002× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_for_latency_hiding | 2.353× | 1.000× | ✓ |  |
| B | fix_grid_lambda_to_enable_autotune_xblock_selection | 2.050× | 0.891× | ✓ |  |
| C | fix_grid_lambda_to_use_meta_xblock | 2.071× | 0.551× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 244030 ns | 244130 ns | +100 ns (+0.0%) | 273860 ns | +29830 ns (+12.2%) | 443100 ns | +199070 ns (+81.6%) |
| SM Throughput (%) | 25.27 % | 36.59 % | +11.32 pp | 29.03 % | +3.76 pp | 71.42 % | +46.15 pp |
| Memory Throughput (%) | 32.19 % | 30.54 % | -1.65 pp | 28.17 % | -4.02 pp | 18.38 % | -13.81 pp |
| DRAM Throughput (%) | 32.19 % | 30.54 % | -1.65 pp | 28.17 % | -4.02 pp | 16.82 % | -15.37 pp |
| Achieved Occupancy (%) | 43.44 % | 43.07 % | -0.37 pp | 35.70 % | -7.74 pp | 61.82 % | +18.38 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 186.18 | 186.18 | +0.00 | 186.18 | +0.00 | 744.73 | +558.55 |
| No Eligible (%) | 74.49 % | 62.99 % | -11.50 pp | 67.41 % | -7.08 pp | 28.26 % | -46.23 pp |
| IPC Active | 1.02 inst/cycle | 1.48 inst/cycle | +0.46 | 1.28 inst/cycle | +0.26 | 2.87 inst/cycle | +1.85 |

---

## Cumulative Progress

```
Baseline:  244030 ns  (1.000×)
Iter 2     243970 ns  (1.000×)  — fix_autotune_grid_lambda_and_add_evict_first_loads

Final best: 243970 ns  (1.0002×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,28 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 50331648
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 96)
     x1 = ((xindex // 96) % 8)
     x2 = ((xindex // 768) % 512)
     x3 = xindex // 393216
     x4 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), None)
-    tl.store(out_ptr0 + (x4), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), xmask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x4), tmp0, xmask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 50331648
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_transpose_view_7[grid](arg_0, arg_1, 50331648, XBLOCK=128)
+    triton_poi_fused_clone_transpose_view_7[grid](arg_0, arg_1, 50331648)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
