# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_6` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5310 ns — **Best latency**: 2500 ns (**2.1240×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_static_grid_lambda_and_enrich_autotune_configs_with_num_warps_stages`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_6`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_6_triton_red_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 5310 ns
- **SM Throughput**: 0.09 % | **Memory**: 0.46 % | **DRAM**: 0.33 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_parallelism_via_smaller_xblock | 1.577× | 2.159× | ✓ |  |
| B | reduce_xblock_to_increase_grid_parallelism | 1.623× | 2.124× | ✓ | **✓ best** |
| C | add_autotune_with_vectorized_tile_sizes | 1.563× | 2.124× | ✓ |  |
| D | increase_grid_size_via_smaller_xblock | 1.551× | 1.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_grid_par... | Δ | reduce_xblock_to_... | Δ | add_autotune_with... | Δ | increase_grid_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5310 ns | 2460 ns | -2850 ns (-53.7%) | 2500 ns | -2810 ns (-52.9%) | 2500 ns | -2810 ns (-52.9%) | 2720 ns | -2590 ns (-48.8%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp | 0.03 % | -0.06 pp | 0.01 % | -0.08 pp | 0.03 % | -0.06 pp |
| Memory Throughput (%) | 0.46 % | 0.43 % | -0.03 pp | 0.40 % | -0.06 pp | 0.50 % | +0.04 pp | 0.38 % | -0.08 pp |
| DRAM Throughput (%) | 0.33 % | 0.08 % | -0.25 pp | 0.08 % | -0.25 pp | 0.09 % | -0.24 pp | 0.08 % | -0.25 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 6.14 % | -0.10 pp | 3.08 % | -3.16 pp | 6.14 % | -0.10 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.74 % | 95.56 % | +4.82 pp | 96.22 % | +5.48 pp | 95.62 % | +4.88 pp | 95.35 % | +4.61 pp |
| IPC Active | 0.34 inst/cycle | 0.04 inst/cycle | -0.30 | 0.16 inst/cycle | -0.18 | 0.09 inst/cycle | -0.25 | 0.16 inst/cycle | -0.18 |


### Key Learnings

#### ✅ Plan B: `reduce_xblock_to_increase_grid_parallelism` (2.124×)

When a reduction kernel has a very small output space (xnumel) and launches far fewer blocks than available SMs, the R0_BLOCK is often massively over-provisioned relative to r0_numel, causing wasted register pressure and shared memory bank conflicts. Adding @triton.autotune with smaller XBLOCK and R0_BLOCK configs allows the runtime to select tile sizes that eliminate unnecessary work: executed instructions dropped from 4024 to 496 (8x), registers per thread halved from 32 to 16, and shared memory bank conflicts were eliminated entirely. The duration dropped from 5.31us to 2.5us (2.12x speedup) primarily through instruction count reduction and register pressure relief, even though grid size remained at 2 blocks.

> **Applicable when**: Triton reduction kernel with hardcoded XBLOCK and R0_BLOCK where R0_BLOCK >> r0_numel (e.g., R0_BLOCK=512 vs r0_numel=98); SharedMemoryConflicts rule triggered with high bank conflict rates (>80% of wavefronts); executed instruction count is high relative to problem size; registers per thread are elevated (32+); kernel has existing xmask and r0_mask guards that handle partial tiles

> **Avoid when**: R0_BLOCK is already well-matched to r0_numel; kernel lacks mask guards for partial tiles (correctness risk); grid is already large enough to saturate SMs and the bottleneck is memory bandwidth or compute throughput rather than tile sizing

---

## Iteration 2

**Best so far**: 2.124× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_expand_autotune_configs | 1.613× | 1.974× | ✓ |  |
| B | fix_static_grid_to_use_autotune_xblock_lambda | 1.613× | 2.159× | ✓ |  |
| C | fix_static_grid_to_use_autotune_xblock | 1.610× | 2.074× | ✓ |  |
| D | expand_autotune_configs_with_larger_xblock_and_r0_block_for_cache_reuse | 1.548× | 2.099× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5310 ns | 2690 ns | -2620 ns (-49.3%) | 2460 ns | -2850 ns (-53.7%) | 2560 ns | -2750 ns (-51.8%) | 2530 ns | -2780 ns (-52.4%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp |
| Memory Throughput (%) | 0.46 % | 0.37 % | -0.09 pp | 0.41 % | -0.05 pp | 0.61 % | +0.15 pp | 0.48 % | +0.02 pp |
| DRAM Throughput (%) | 0.33 % | 0.06 % | -0.27 pp | 0.09 % | -0.24 pp | 0.08 % | -0.25 pp | 0.08 % | -0.25 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 3.08 % | -3.16 pp | 3.08 % | -3.16 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.74 % | 95.63 % | +4.89 pp | 94.46 % | +3.72 pp | 95.90 % | +5.16 pp | 94.98 % | +4.24 pp |
| IPC Active | 0.34 inst/cycle | 0.05 inst/cycle | -0.29 | 0.04 inst/cycle | -0.30 | 0.09 inst/cycle | -0.25 | 0.09 inst/cycle | -0.25 |

---

## Iteration 3

**Best so far**: 2.124× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_tile_sizes_with_dynamic_grid_lambda | 1.578× | 2.159× | ✓ |  |
| B | fix_static_grid_to_dynamic_lambda_using_autotuned_xblock | 1.607× | 2.099× | ✓ |  |
| C | fix_static_grid_lambda_and_enrich_autotune_configs_with_num_warps_stages | 1.551× | 2.124× | ✓ |  |
| D | fix_static_grid_lambda_and_expand_large_xblock_configs | 1.559× | 1.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ | fix_static_grid_t... | Δ | fix_static_grid_l... | Δ | fix_static_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5310 ns | 2460 ns | -2850 ns (-53.7%) | 2530 ns | -2780 ns (-52.4%) | 2500 ns | -2810 ns (-52.9%) | 2660 ns | -2650 ns (-49.9%) |
| SM Throughput (%) | 0.09 % | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp | 0.01 % | -0.08 pp |
| Memory Throughput (%) | 0.46 % | 0.58 % | +0.12 pp | 0.50 % | +0.04 pp | 0.50 % | +0.04 pp | 0.51 % | +0.05 pp |
| DRAM Throughput (%) | 0.33 % | 0.09 % | -0.24 pp | 0.08 % | -0.25 pp | 0.08 % | -0.25 pp | 0.08 % | -0.25 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 3.08 % | -3.16 pp | 3.08 % | -3.16 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 | 16.00 register/thread | -16 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 90.74 % | 94.37 % | +3.63 pp | 95.95 % | +5.21 pp | 94.56 % | +3.82 pp | 94.77 % | +4.03 pp |
| IPC Active | 0.34 inst/cycle | 0.05 inst/cycle | -0.29 | 0.09 inst/cycle | -0.25 | 0.09 inst/cycle | -0.25 | 0.04 inst/cycle | -0.30 |

---

## Cumulative Progress

```
Baseline:  5310 ns  (1.000×)
Iter 1     2500 ns  (2.124×)  — reduce_xblock_to_increase_grid_parallelism

Final best: 2500 ns  (2.1240×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_warps=1, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 192
@@ -26,7 +47,7 @@

         roffset = r0_offset
         rindex = r0_index
         r0_2 = r0_index
-        tmp0 = tl.load(in_ptr0 + (x0 + 96*r0_2 + 9408*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (x0 + 96*r0_2 + 9408*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
@@ -34,9 +55,7 @@

     tmp4 = tl.full([1, 1], 12544.0, tl.float32)
     tmp5 = (tmp2 / tmp4)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
-
-
+    tl.store(in_out_ptr0 + (x3), tmp5, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +63,7 @@

     xnumel = 192
     r0_numel = 98
     grid = (triton.cdiv(xnumel, 128),)
-    triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_6[grid](arg_0, arg_1, 192, 98, XBLOCK=128, R0_BLOCK=512)
+    triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_relu_6[grid](arg_0, arg_1, 192, 98)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
