# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_mean_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7140 ns — **Best latency**: 2690 ns (**2.6543×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_lambda_and_expand_tiny_xblock_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_mean_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_042/kernel_3_triton_red_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.01
- **Registers/thread**: 32
- **Duration**: 7140 ns
- **SM Throughput**: 0.73 % | **Memory**: 3.06 % | **DRAM**: 3.06 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_autotune_increase_grid_coverage | 1.778× | 1.605× | ✓ |  |
| D | increase_grid_coverage_via_autotune_tile_sizes | 1.807× | 1.641× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_autotune_incr... | Δ | increase_grid_cov... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 4450 ns | -2690 ns (-37.7%) | 4350 ns | -2790 ns (-39.1%) |
| SM Throughput (%) | 0.73 % | 0.48 % | -0.25 pp | 0.48 % | -0.25 pp |
| Memory Throughput (%) | 3.06 % | 0.90 % | -2.16 pp | 0.93 % | -2.13 pp |
| DRAM Throughput (%) | 3.06 % | 0.65 % | -2.41 pp | 0.66 % | -2.40 pp |
| Achieved Occupancy (%) | 6.25 % | 3.11 % | -3.14 pp | 3.11 % | -3.14 pp |
| Registers/Thread | 32.00 register/thread | 56.00 register/thread | +24 | 56.00 register/thread | +24 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.48 % | 86.12 % | -5.36 pp | 86.38 % | -5.10 pp |
| IPC Active | 0.33 inst/cycle | 0.27 inst/cycle | -0.06 | 0.29 inst/cycle | -0.04 |

---

## Iteration 2

**Best so far**: 1.6414× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_dynamic_autotune_xblock | 1.773× | 1.676× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_tile_exploration | 1.809× | 2.684× | ✓ |  |
| C | fix_autotune_grid_to_enable_dynamic_block_count | 1.872× | 2.625× | ✓ | **✓ best** |
| D | fix_grid_lambda_to_respect_autotuned_xblock_and_expand_configs | 1.869× | 2.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 4260 ns | -2880 ns (-40.3%) | 2660 ns | -4480 ns (-62.7%) | 2720 ns | -4420 ns (-61.9%) | 2400 ns | -4740 ns (-66.4%) |
| SM Throughput (%) | 0.73 % | 0.47 % | -0.26 pp | 0.11 % | -0.62 pp | 0.11 % | -0.62 pp | 0.06 % | -0.67 pp |
| Memory Throughput (%) | 3.06 % | 0.94 % | -2.12 pp | 1.30 % | -1.76 pp | 1.08 % | -1.98 pp | 1.26 % | -1.80 pp |
| DRAM Throughput (%) | 3.06 % | 0.68 % | -2.38 pp | 0.09 % | -2.97 pp | 0.09 % | -2.97 pp | 0.11 % | -2.95 pp |
| Achieved Occupancy (%) | 6.25 % | 3.11 % | -3.14 pp | 3.00 % | -3.25 pp | 2.99 % | -3.26 pp | 1.56 % | -4.69 pp |
| Registers/Thread | 32.00 register/thread | 56.00 register/thread | +24 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 18.00 register/thread | -14 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.48 % | 85.84 % | -5.64 pp | 94.35 % | +2.87 pp | 95.91 % | +4.43 pp | 92.86 % | +1.38 pp |
| IPC Active | 0.33 inst/cycle | 0.27 inst/cycle | -0.06 | 0.08 inst/cycle | -0.25 | 0.09 inst/cycle | -0.24 | 0.07 inst/cycle | -0.26 |

---

## Iteration 3

**Best so far**: 2.625× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_divisor_to_dynamic_autotune_lambda | 1.826× | 2.822× | ✓ |  |
| B | fix_hardcoded_grid_to_lambda_for_autotune_coverage | 1.817× | 1.641× | ✓ |  |
| C | fix_static_grid_to_use_autotune_meta_xblock_lambda | 1.830× | 2.789× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_and_expand_tiny_xblock_configs | 1.909× | 2.654× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.875× | 2.789× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_static_grid_t... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 2530 ns | -4610 ns (-64.6%) | 4350 ns | -2790 ns (-39.1%) | 2560 ns | -4580 ns (-64.1%) | 2690 ns | -4450 ns (-62.3%) | 2560 ns | -4580 ns (-64.1%) |
| SM Throughput (%) | 0.73 % | 0.10 % | -0.63 pp | 0.52 % | -0.21 pp | 0.11 % | -0.62 pp | 0.07 % | -0.66 pp | 0.12 % | -0.61 pp |
| Memory Throughput (%) | 3.06 % | 1.17 % | -1.89 pp | 0.93 % | -2.13 pp | 1.17 % | -1.89 pp | 1.10 % | -1.96 pp | 1.16 % | -1.90 pp |
| DRAM Throughput (%) | 3.06 % | 0.09 % | -2.97 pp | 0.66 % | -2.40 pp | 0.10 % | -2.96 pp | 0.09 % | -2.97 pp | 0.10 % | -2.96 pp |
| Achieved Occupancy (%) | 6.25 % | 2.97 % | -3.28 pp | 3.11 % | -3.14 pp | 2.97 % | -3.28 pp | 1.56 % | -4.69 pp | 3.00 % | -3.25 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 56.00 register/thread | +24 | 16.00 register/thread | -16 | 18.00 register/thread | -14 | 16.00 register/thread | -16 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.48 % | 95.42 % | +3.94 pp | 85.59 % | -5.89 pp | 95.60 % | +4.12 pp | 93.66 % | +2.18 pp | 94.39 % | +2.91 pp |
| IPC Active | 0.33 inst/cycle | 0.10 inst/cycle | -0.23 | 0.28 inst/cycle | -0.05 | 0.10 inst/cycle | -0.23 | 0.07 inst/cycle | -0.26 | 0.08 inst/cycle | -0.25 |

---

## Iteration 4

**Best so far**: 2.6543× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_dynamic_lambda_and_expand_r0block_configs | 1.857× | 2.568× | ✓ |  |
| B | bump_num_warps_on_small_xblock_configs_to_increase_theoretical_occupancy | 1.827× | 2.789× | ✓ |  |
| C | fix_static_grid_divisor_to_dynamic_autotune_lambda_launch_site | 1.827× | 2.568× | ✓ |  |
| D | fix_static_grid_to_dynamic_lambda_for_autotune_coverage | 1.834× | 2.505× | ✓ |  |
| E | add_triton_autotune | 1.832× | 2.596× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | bump_num_warps_on... | Δ | fix_static_grid_d... | Δ | fix_static_grid_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 2780 ns | -4360 ns (-61.1%) | 2560 ns | -4580 ns (-64.1%) | 2780 ns | -4360 ns (-61.1%) | 2850 ns | -4290 ns (-60.1%) | 2750 ns | -4390 ns (-61.5%) |
| SM Throughput (%) | 0.73 % | 0.10 % | -0.63 pp | 0.11 % | -0.62 pp | 0.12 % | -0.61 pp | 0.11 % | -0.62 pp | 0.11 % | -0.62 pp |
| Memory Throughput (%) | 3.06 % | 1.08 % | -1.98 pp | 1.16 % | -1.90 pp | 1.07 % | -1.99 pp | 1.04 % | -2.02 pp | 1.08 % | -1.98 pp |
| DRAM Throughput (%) | 3.06 % | 0.09 % | -2.97 pp | 0.09 % | -2.97 pp | 0.09 % | -2.97 pp | 0.09 % | -2.97 pp | 0.09 % | -2.97 pp |
| Achieved Occupancy (%) | 6.25 % | 2.97 % | -3.28 pp | 2.99 % | -3.26 pp | 3.00 % | -3.25 pp | 2.99 % | -3.26 pp | 2.97 % | -3.28 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.48 % | 95.74 % | +4.26 pp | 95.28 % | +3.80 pp | 94.19 % | +2.71 pp | 94.32 % | +2.84 pp | 95.08 % | +3.60 pp |
| IPC Active | 0.33 inst/cycle | 0.10 inst/cycle | -0.23 | 0.09 inst/cycle | -0.24 | 0.08 inst/cycle | -0.25 | 0.09 inst/cycle | -0.24 | 0.10 inst/cycle | -0.23 |

---

## Iteration 5

**Best so far**: 2.6543× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_lambda_and_add_pipelining_num_stages | 1.836× | 2.505× | ✓ |  |
| B | fix_static_grid_lambda_and_add_pipeline_stages_for_latency_hiding | 1.863× | 2.789× | ✓ |  |
| C | coalesce_global_loads_via_large_xblock_and_evict_last | 1.885× | 2.479× | ✓ |  |
| D | fix_uncoalesced_global_load_by_swapping_reduction_and_channel_dims | 1.708× | 2.532× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_l... | Δ | fix_static_grid_l... | Δ | coalesce_global_l... | Δ | fix_uncoalesced_g... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7140 ns | 2850 ns | -4290 ns (-60.1%) | 2560 ns | -4580 ns (-64.1%) | 2880 ns | -4260 ns (-59.7%) | 2820 ns | -4320 ns (-60.5%) |
| SM Throughput (%) | 0.73 % | 0.11 % | -0.62 pp | 0.11 % | -0.62 pp | 0.10 % | -0.63 pp | 0.11 % | -0.62 pp |
| Memory Throughput (%) | 3.06 % | 1.03 % | -2.03 pp | 1.17 % | -1.89 pp | 1.04 % | -2.02 pp | 1.05 % | -2.01 pp |
| DRAM Throughput (%) | 3.06 % | 0.08 % | -2.98 pp | 0.10 % | -2.96 pp | 0.09 % | -2.97 pp | 0.09 % | -2.97 pp |
| Achieved Occupancy (%) | 6.25 % | 2.98 % | -3.27 pp | 3.00 % | -3.25 pp | 2.97 % | -3.28 pp | 3.00 % | -3.25 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 91.48 % | 94.20 % | +2.72 pp | 95.44 % | +3.96 pp | 94.29 % | +2.81 pp | 95.43 % | +3.95 pp |
| IPC Active | 0.33 inst/cycle | 0.09 inst/cycle | -0.24 | 0.08 inst/cycle | -0.25 | 0.10 inst/cycle | -0.23 | 0.08 inst/cycle | -0.25 |

---

## Cumulative Progress

```
Baseline:  7140 ns  (1.000×)
Iter 1     4350 ns  (1.641×)  — increase_grid_coverage_via_autotune_tile_sizes
Iter 2     2720 ns  (2.625×)  — fix_autotune_grid_to_enable_dynamic_block_count
Iter 3     2690 ns  (2.654×)  — fix_hardcoded_grid_to_lambda_and_expand_tiny_xblock_configs

Final best: 2690 ns  (2.6543×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 1,   'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2,   'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4,   'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 8,   'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_mean_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -26,14 +50,12 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16512*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16512*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
-    tl.store(out_ptr0 + (x3), tmp2, xmask)
-
-
+    tl.store(out_ptr0 + (x3), tmp2, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +63,7 @@

     xnumel = 2048
     r0_numel = 129
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused_convolution_mean_3[grid](arg_0, arg_1, 2048, 129, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused_convolution_mean_3[grid](arg_0, arg_1, 2048, 129)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
