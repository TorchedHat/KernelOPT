# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3300 ns — **Best latency**: 3330 ns (**0.9910×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_4_triton_poi_fused__scaled_dot_p/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1576,1,1), Waves/SM: 0.75
- **Registers/thread**: 16
- **Duration**: 3300 ns
- **SM Throughput**: 10.72 % | **Memory**: 7.44 % | **DRAM**: 5.03 %
- **Achieved Occupancy**: 51.98 %
- **IPC Active**: 1.18 inst/cycle | **No Eligible**: 69.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 1.006× | 0.991× | ✓ | **✓ best** |
| B | increase_xblock_autotune_for_underutilized_pointwise_kernel | 1.005× | 1.051× | ✓ |  |
| C | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 0.979× | 1.065× | ✓ |  |
| D | increase_xblock_and_autotune_tile_size | 0.970× | 0.769× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | increase_xblock_a... | Δ | increase_xblock_w... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3330 ns | +30 ns (+0.9%) | 3140 ns | -160 ns (-4.8%) | 3100 ns | -200 ns (-6.1%) | 4290 ns | +990 ns (+30.0%) |
| SM Throughput (%) | 10.72 % | 6.84 % | -3.88 pp | 12.12 % | +1.40 pp | 12.81 % | +2.09 pp | 40.73 % | +30.01 pp |
| Memory Throughput (%) | 7.44 % | 7.30 % | -0.14 pp | 7.52 % | +0.08 pp | 7.57 % | +0.13 pp | 20.38 % | +12.94 pp |
| DRAM Throughput (%) | 5.03 % | 5.05 % | +0.02 pp | 5.33 % | +0.30 pp | 5.36 % | +0.33 pp | 3.90 % | -1.13 pp |
| Achieved Occupancy (%) | 51.98 % | 27.01 % | -24.97 pp | 45.14 % | -6.84 pp | 44.98 % | -7.00 pp | 69.65 % | +17.67 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 30.00 register/thread | +14 |
| Waves/SM | 0.75 | 0.37 | -0.38 | 0.75 | +0.00 | 0.75 | +0.00 | 1.49 | +0.74 |
| No Eligible (%) | 69.08 % | 76.98 % | +7.90 pp | 56.49 % | -12.59 pp | 57.23 % | -11.85 pp | 22.68 % | -46.40 pp |
| IPC Active | 1.18 inst/cycle | 0.80 inst/cycle | -0.38 | 1.55 inst/cycle | +0.37 | 1.53 inst/cycle | +0.35 | 2.99 inst/cycle | +1.81 |


### Key Learnings

#### ❌ Plan D: `increase_xblock_and_autotune_tile_size` (0.769×)

Adding @triton.autotune with larger XBLOCK configs caused the compiler to select a configuration (XBLOCK=256) that nearly doubled register usage per thread (16→30 registers/thread), which reduced the number of blocks that can be resident per SM and caused a 5.3x increase in executed instructions (277K→1.46M). Although scheduler utilization improved (No Eligible: 69%→23%) and achieved occupancy rose (52%→70%), the massive instruction count increase and partial-wave tail effect (50% of runtime per NCU LaunchConfiguration rule) caused overall kernel duration to increase from 3.3µs to 4.29µs. The autotune overhead and the selected config's register pressure outweighed the latency-hiding benefit for this small, low-register-pressure kernel.

> **Applicable when**: Kernel has low waves_per_sm (<1.0), high scheduler No Eligible % (>60%), and low achieved occupancy (<60%), suggesting insufficient warps to hide memory latency.

> **Avoid when**: Kernel already has very low register usage (≤16 registers/thread) and the autotune search space includes configs that cause register count to nearly double; the instruction count explosion from larger tiles can outweigh latency-hiding gains. Also avoid when the grid size is fixed in the call site and not updated to use lambda/meta, causing the autotune to run with a mismatched fixed grid. Do not apply when the kernel has complex index arithmetic that scales instruction count super-linearly with XBLOCK.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_for_correct_tile_scheduling | 0.965× | 1.051× | ✓ |  |
| B | persistent_tile_scheduling_grid_stride_loop | 0.965× | 0.982× | ✓ |  |
| C | fix_persistent_kernel_grid_to_sm_count | 0.962× | 1.051× | ✓ |  |
| D | persistent_grid_stride_to_eliminate_partial_wave_tail | 0.963× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | persistent_tile_s... | Δ | fix_persistent_ke... | Δ | persistent_grid_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3140 ns | -160 ns (-4.8%) | 3360 ns | +60 ns (+1.8%) | 3140 ns | -160 ns (-4.8%) | 3390 ns | +90 ns (+2.7%) |
| SM Throughput (%) | 10.72 % | 12.87 % | +2.15 pp | 13.30 % | +2.58 pp | 7.99 % | -2.73 pp | 13.26 % | +2.54 pp |
| Memory Throughput (%) | 7.44 % | 7.33 % | -0.11 pp | 7.31 % | -0.13 pp | 7.35 % | -0.09 pp | 7.19 % | -0.25 pp |
| DRAM Throughput (%) | 5.03 % | 5.32 % | +0.29 pp | 5.00 % | -0.03 pp | 5.34 % | +0.31 pp | 4.95 % | -0.08 pp |
| Achieved Occupancy (%) | 51.98 % | 45.82 % | -6.16 pp | 52.04 % | +0.06 pp | 29.11 % | -22.87 pp | 51.55 % | -0.43 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 22.00 register/thread | +6 | 16.00 register/thread | +0 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 |
| No Eligible (%) | 69.08 % | 57.25 % | -11.83 pp | 62.91 % | -6.17 pp | 83.36 % | +14.28 pp | 62.81 % | -6.27 pp |
| IPC Active | 1.18 inst/cycle | 1.46 inst/cycle | +0.28 | 1.35 inst/cycle | +0.17 | 0.61 inst/cycle | -0.57 | 1.39 inst/cycle | +0.21 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_bias_eviction_policy_and_autotune_grid_lambda | 0.998× | 1.051× | ✓ |  |
| B | revert_persistent_loop_to_single_pass_with_correct_autotune_grid | 0.975× | 1.031× | ✓ |  |
| C | streaming_eviction_policy_for_non_coalesced_large_tensor_load | 0.995× | 0.965× | ✓ |  |
| D | autotune_xblock_with_corrected_standard_grid_remove_broken_persistent_loop | 0.966× | 1.041× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_bias_eviction... | Δ | revert_persistent... | Δ | streaming_evictio... | Δ | autotune_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3140 ns | -160 ns (-4.8%) | 3200 ns | -100 ns (-3.0%) | 3420 ns | +120 ns (+3.6%) | 3170 ns | -130 ns (-3.9%) |
| SM Throughput (%) | 10.72 % | 13.01 % | +2.29 pp | 13.29 % | +2.57 pp | 14.60 % | +3.88 pp | 13.30 % | +2.58 pp |
| Memory Throughput (%) | 7.44 % | 7.34 % | -0.10 pp | 7.25 % | -0.19 pp | 7.14 % | -0.30 pp | 7.39 % | -0.05 pp |
| DRAM Throughput (%) | 5.03 % | 5.31 % | +0.28 pp | 5.18 % | +0.15 pp | 4.88 % | -0.15 pp | 5.25 % | +0.22 pp |
| Achieved Occupancy (%) | 51.98 % | 47.90 % | -4.08 pp | 50.38 % | -1.60 pp | 52.90 % | +0.92 pp | 44.59 % | -7.39 pp |
| Registers/Thread | 16.00 register/thread | 18.00 register/thread | +2 | 18.00 register/thread | +2 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 0.75 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 | 0.75 | +0.00 |
| No Eligible (%) | 69.08 % | 54.93 % | -14.15 pp | 56.19 % | -12.89 pp | 60.47 % | -8.61 pp | 55.68 % | -13.40 pp |
| IPC Active | 1.18 inst/cycle | 1.30 inst/cycle | +0.12 | 1.22 inst/cycle | +0.04 | 1.41 inst/cycle | +0.23 | 1.64 inst/cycle | +0.46 |

---

## Cumulative Progress

```
Baseline:  3300 ns  (1.000×)

Final best: 3330 ns  (0.9910×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=8),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 201728
@@ -20,15 +32,13 @@

     tmp2 = tmp0 + tmp1
     tl.store(out_ptr0 + (x3), tmp2, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     arg_2 = inputs[2]
     xnumel = 201728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4[grid](arg_0, arg_1, arg_2, 201728, XBLOCK=128)
+    triton_poi_fused__scaled_dot_product_efficient_attention_addmm_clone_select_squeeze_transpose_unsqueeze_view_4[grid](arg_0, arg_1, arg_2, 201728)
     return arg_2
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
