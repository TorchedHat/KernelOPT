# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_relu_view_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6460 ns — **Best latency**: 6110 ns (**1.0573×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_autotune_grid_lambda_to_eliminate_partial_wave_and_reduce_cta_count`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_relu_view_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_11_triton_poi_fused_addmm_relu_vi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6304,1,1), Waves/SM: 2.98
- **Registers/thread**: 16
- **Duration**: 6460 ns
- **SM Throughput**: 14.86 % | **Memory**: 14.07 % | **DRAM**: 10.26 %
- **Achieved Occupancy**: 45.51 %
- **IPC Active**: 0.95 inst/cycle | **No Eligible**: 75.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_l1_reuse_and_occupancy | 1.000× | 1.052× | ✓ |  |
| B | autotune_xblock_num_warps_for_pointwise_bias_relu | 1.028× | 1.057× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 1.002× | 0.885× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 1.024× | 0.734× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 6140 ns | -320 ns (-5.0%) | 6110 ns | -350 ns (-5.4%) | 7300 ns | +840 ns (+13.0%) | 8800 ns | +2340 ns (+36.2%) |
| SM Throughput (%) | 14.86 % | 20.80 % | +5.94 pp | 20.82 % | +5.96 pp | 55.05 % | +40.19 pp | 57.64 % | +42.78 pp |
| Memory Throughput (%) | 14.07 % | 13.71 % | -0.36 pp | 13.74 % | -0.33 pp | 26.58 % | +12.51 pp | 22.78 % | +8.71 pp |
| DRAM Throughput (%) | 10.26 % | 10.80 % | +0.54 pp | 10.82 % | +0.56 pp | 9.08 % | -1.18 pp | 7.54 % | -2.72 pp |
| Achieved Occupancy (%) | 45.51 % | 40.00 % | -5.51 pp | 39.77 % | -5.74 pp | 76.01 % | +30.50 pp | 67.68 % | +22.17 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 16.00 register/thread | +0 |
| Waves/SM | 2.98 | 2.98 | +0.00 | 2.98 | +0.00 | 5.97 | +2.99 | 11.94 | +8.96 |
| No Eligible (%) | 75.47 % | 57.95 % | -17.52 pp | 57.53 % | -17.94 pp | 26.35 % | -49.12 pp | 27.76 % | -47.71 pp |
| IPC Active | 0.95 inst/cycle | 1.58 inst/cycle | +0.63 | 1.59 inst/cycle | +0.64 | 2.95 inst/cycle | +2.00 | 2.98 inst/cycle | +2.03 |


### Key Learnings

#### ✅ Plan B: `autotune_xblock_num_warps_for_pointwise_bias_relu` (1.057×)

For pointwise Triton kernels with a fixed small tile size (e.g. XBLOCK=128), the scheduler idle rate (No Eligible) can be very high (>70%) due to L1TEX load stalls, and a large grid with a partial tail wave adds further overhead. Adding @triton.autotune over a range of XBLOCK values (512–4096) and num_warps (4–16) allows the runtime to select a tile size that reduces the tail-wave penalty and gives the scheduler more in-flight memory requests to hide latency. In this case, the change reduced Warp Cycles Per Issued Instruction from 27.5 to 15.0 and raised Issue Slots Busy from 24% to 41%, yielding a 5.7% speedup despite lower achieved occupancy.

> **Applicable when**: Triton pointwise kernel with fixed small XBLOCK (≤128); scheduler No Eligible >60%; Warp Cycles Per Issued Instruction >20; CPIStall rule triggered with L1TEX stalls >50% of stall cycles; LaunchConfiguration rule triggered with partial tail wave >20%; compute and memory SOL both <30%.

> **Avoid when**: Kernel is already autotuned or XBLOCK is already large (≥512); kernel has shared memory usage that limits block size; xnumel is small enough that a single wave already covers all SMs; the bottleneck is compute-bound rather than latency-bound.

#### ❌ Plan D: `autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency` (0.594×)

Reordering or modifying autotune configs (XBLOCK values) while keeping a static grid tuple causes the autotuner to select a larger XBLOCK, but the grid remains fixed at the original CTA count. In this case the grid stayed at 6304 CTAs while autotune selected XBLOCK=256, launching 1.6M threads for only 806K elements (2x overwork). Executed instructions ballooned from 1.0M to 5.3M, registers per thread doubled from 16 to 32, and duration regressed from 6.11µs to 10.88µs (0.59x). Despite dramatically improved per-cycle efficiency (compute SOL 21% → 63%, occupancy 41% → 82%), total wall-clock time worsened severely because total work multiplied proportionally to selected_XBLOCK / grid_denominator.

> **Applicable when**: Never — this is a pure implementation flaw. Any autotune over XBLOCK without a lambda grid will cause overwork proportional to (selected_XBLOCK / original_grid_denominator).

> **Avoid when**: Always avoid: never change autotune XBLOCK configs without simultaneously changing the grid from a static tuple to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. This anti-pattern is now confirmed across three iterations (iter 1: 0.73x, iter 2: 0.59x, iter 3: 0.59x) with consistent symptom signature: executed instructions multiply by XBLOCK/128, registers per thread increase, waves/SM double.

---

## Iteration 2

**Best so far**: 1.0573× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_match_xblock | 1.005× | 1.052× | ✓ |  |
| B | fix_autotune_grid_lambda_to_eliminate_partial_wave_and_reduce_cta_count | 1.021× | 1.057× | ✓ |  |
| C | fix_autotune_grid_lambda_to_match_xblock | 1.022× | 0.863× | ✓ |  |
| D | fix_grid_lambda_to_correct_overwork_regression_and_enable_fma | 1.000× | 0.594× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 6140 ns | -320 ns (-5.0%) | 6110 ns | -350 ns (-5.4%) | 7490 ns | +1030 ns (+15.9%) | 10880 ns | +4420 ns (+68.4%) |
| SM Throughput (%) | 14.86 % | 21.18 % | +6.32 pp | 21.10 % | +6.24 pp | 54.78 % | +39.92 pp | 63.30 % | +48.44 pp |
| Memory Throughput (%) | 14.07 % | 13.84 % | -0.23 pp | 13.82 % | -0.25 pp | 26.46 % | +12.39 pp | 26.99 % | +12.92 pp |
| DRAM Throughput (%) | 10.26 % | 10.82 % | +0.56 pp | 10.81 % | +0.55 pp | 8.81 % | -1.45 pp | 6.08 % | -4.18 pp |
| Achieved Occupancy (%) | 45.51 % | 42.21 % | -3.30 pp | 40.86 % | -4.65 pp | 75.98 % | +30.47 pp | 81.22 % | +35.71 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 32.00 register/thread | +16 |
| Waves/SM | 2.98 | 2.98 | +0.00 | 2.98 | +0.00 | 5.97 | +2.99 | 5.97 | +2.99 |
| No Eligible (%) | 75.47 % | 57.57 % | -17.90 pp | 58.07 % | -17.40 pp | 26.46 % | -49.01 pp | 24.25 % | -51.22 pp |
| IPC Active | 0.95 inst/cycle | 1.53 inst/cycle | +0.58 | 1.57 inst/cycle | +0.62 | 2.96 inst/cycle | +2.01 | 3.00 inst/cycle | +2.05 |

---

## Iteration 3

**Best so far**: 1.0573× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_fma_via_relu_rewrite_and_fix_grid_lambda | 1.022× | 1.052× | ✓ |  |
| B | fix_grid_lambda_to_enable_correct_xblock_scaling_and_fma_fusion | 1.010× | 1.052× | ✓ |  |
| C | fix_grid_lambda_to_match_autotuned_xblock_and_eliminate_instruction_bloat | 1.022× | 0.885× | ✓ |  |
| D | fix_static_grid_to_lambda_to_eliminate_autotune_overwork | 1.005× | 0.594× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_fma_via_re... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6460 ns | 6140 ns | -320 ns (-5.0%) | 6140 ns | -320 ns (-5.0%) | 7300 ns | +840 ns (+13.0%) | 10880 ns | +4420 ns (+68.4%) |
| SM Throughput (%) | 14.86 % | 21.13 % | +6.27 pp | 21.28 % | +6.42 pp | 55.02 % | +40.16 pp | 62.88 % | +48.02 pp |
| Memory Throughput (%) | 14.07 % | 13.82 % | -0.25 pp | 13.94 % | -0.13 pp | 26.55 % | +12.48 pp | 26.88 % | +12.81 pp |
| DRAM Throughput (%) | 10.26 % | 10.80 % | +0.54 pp | 10.80 % | +0.54 pp | 9.06 % | -1.20 pp | 6.08 % | -4.18 pp |
| Achieved Occupancy (%) | 45.51 % | 39.82 % | -5.69 pp | 38.44 % | -7.07 pp | 76.12 % | +30.61 pp | 81.81 % | +36.30 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 24.00 register/thread | +8 | 32.00 register/thread | +16 |
| Waves/SM | 2.98 | 2.98 | +0.00 | 2.98 | +0.00 | 5.97 | +2.99 | 5.97 | +2.99 |
| No Eligible (%) | 75.47 % | 57.53 % | -17.94 pp | 57.44 % | -18.03 pp | 25.41 % | -50.06 pp | 24.30 % | -51.17 pp |
| IPC Active | 0.95 inst/cycle | 1.59 inst/cycle | +0.64 | 1.62 inst/cycle | +0.67 | 2.95 inst/cycle | +2.00 | 2.94 inst/cycle | +1.99 |

---

## Cumulative Progress

```
Baseline:  6460 ns  (1.000×)
Iter 1     6110 ns  (1.057×)  — autotune_xblock_num_warps_for_pointwise_bias_relu

Final best: 6110 ns  (1.0573×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,29 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 4096}, num_warps=8),
+        triton.Config({'XBLOCK': 4096}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_addmm_relu_view_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 806912
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 2048)
     x2 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_out_ptr0 + (x2), None)
+    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
     tmp2 = tmp0 + tmp1
     tmp3 = tl.full([1], 0, tl.int32)
     tmp4 = tl.maximum(tmp3, tmp2)
-    tl.store(in_out_ptr0 + (x2), tmp4, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 806912
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_addmm_relu_view_11[grid](arg_0, arg_1, 806912, XBLOCK=128)
+    triton_poi_fused_addmm_relu_view_11[grid](arg_0, arg_1, 806912)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
