# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6270 ns — **Best latency**: 6080 ns (**1.0312×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`enable_fma_fusion_by_restructuring_bn_arithmetic`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_10_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 18
- **Duration**: 6270 ns
- **SM Throughput**: 39.10 % | **Memory**: 14.65 % | **DRAM**: 9.83 %
- **Achieved Occupancy**: 59.91 %
- **IPC Active**: 2.48 inst/cycle | **No Eligible**: 37.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.973× | 0.288× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 0.866× | 0.834× | ✓ |  |
| D | add_autotune_increase_xblock_for_vectorized_loads | 0.950× | 0.289× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 21730 ns | +15460 ns (+246.6%) | 7520 ns | +1250 ns (+19.9%) | 21660 ns | +15390 ns (+245.5%) |
| SM Throughput (%) | 39.10 % | 79.30 % | +40.20 pp | 61.22 % | +22.12 pp | 79.21 % | +40.11 pp |
| Memory Throughput (%) | 14.65 % | 15.90 % | +1.25 pp | 17.13 % | +2.48 pp | 15.90 % | +1.25 pp |
| DRAM Throughput (%) | 9.83 % | 2.83 % | -7.00 pp | 8.20 % | -1.63 pp | 2.84 % | -6.99 pp |
| Achieved Occupancy (%) | 59.91 % | 84.29 % | +24.38 pp | 78.08 % | +18.17 pp | 84.71 % | +24.80 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 2.78 | 11.14 | +8.36 | 2.78 | +0.00 | 11.14 | +8.36 |
| No Eligible (%) | 37.76 % | 12.18 % | -25.58 pp | 19.35 % | -18.41 pp | 12.13 % | -25.63 pp |
| IPC Active | 2.48 inst/cycle | 3.50 inst/cycle | +1.02 | 3.28 inst/cycle | +0.80 | 3.47 inst/cycle | +0.99 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_to_eliminate_partial_wave` (0.289×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) while keeping the grid hardcoded to cdiv(xnumel, 128) caused a 3.45x regression: executed instructions exploded from 3.65M to 13.53M and duration increased from 7.52 µs to 21.66 µs. The autotuner selected XBLOCK=512, but the grid remained at 5,880 blocks (sized for XBLOCK=128), so each block processed 512 elements while the grid was not reduced — causing 4x the total work (3,010,560 elements processed vs 752,640 needed). The fix is mandatory: the grid must be a lambda `lambda META: (triton.cdiv(xnumel, META['XBLOCK']),)` so it scales inversely with XBLOCK, keeping total work constant regardless of which config autotune selects.

> **Applicable when**: Never applicable as implemented — this is a pure implementation flaw. The general strategy of autotuning XBLOCK is valid only when the grid lambda correctly uses META['XBLOCK'].

> **Avoid when**: Do not use @triton.autotune with variable XBLOCK configs if the grid is hardcoded to a fixed block size. The grid must always be a lambda using META['XBLOCK'] when autotuning XBLOCK, otherwise total work multiplies by (autotuned_XBLOCK / original_XBLOCK), causing proportional instruction count explosion.

#### ❌ Plan B: `autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy` (0.834×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) on a pointwise kernel with only 18 registers/thread increased occupancy from 59.9% to 78.1% and compute throughput from 39% to 61%, but caused a 20% regression in wall time. The root cause is that the autotuner selected a config that nearly doubled the executed instruction count (1.97M → 3.65M) and increased registers/thread from 18 to 32, which reduced the block limit from 21 to 16 blocks/SM and added overhead without proportional latency hiding benefit. For very short-duration kernels (~6 µs), the autotune overhead and increased instruction count from larger tiles can outweigh the partial-wave elimination benefit, especially when the kernel is already compute-bound at the warp level.

> **Applicable when**: Kernel has large partial wave (>20% of grid), low achieved occupancy (<60%), low register pressure (<20 registers/thread), and duration is long enough (>20 µs) that autotune overhead amortizes

> **Avoid when**: Kernel duration is very short (~6 µs); the selected autotune config increases register count (18→32), which reduces block occupancy and inflates instruction count without proportional latency hiding; compute throughput is already moderate (39%) and the kernel is not memory-bandwidth bound

#### ❌ Plan D: `add_autotune_increase_xblock_for_vectorized_loads` (0.289×)

Adding @triton.autotune with larger XBLOCK values (256–2048) to a small, memory-bound pointwise kernel caused a 3.45x regression. Although occupancy improved (59.9% → 84.7%) and compute throughput rose (39% → 79%), the total executed instruction count exploded 6.8x (1.97M → 13.5M) and kernel duration increased 3.45x (6.27 µs → 21.66 µs). The autotune selected XBLOCK=512 (not the largest config), but the grid lambda was not updated to use META['XBLOCK'] — the grid remained hardcoded to cdiv(xnumel, 128), causing each block to process 4x more elements than its tile size, leading to massive redundant computation. When the grid is not updated to match the new XBLOCK, each thread block iterates over far more work than intended, multiplying instruction count proportionally to the XBLOCK ratio.

> **Applicable when**: Never applicable as implemented — this is a pure implementation flaw. The general strategy of autotuning XBLOCK is valid when the grid lambda correctly uses META['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK if the grid is hardcoded to a fixed XBLOCK value. The grid must always be a lambda using META['XBLOCK'] when autotuning XBLOCK, otherwise total work multiplies by (autotuned_XBLOCK / original_XBLOCK).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.963× | 0.984× | ✓ |  |
| B | fix_broken_grid_lambda_and_enable_fma_fusion | 1.020× | 1.031× | ✓ | **✓ best** |
| C | fix_grid_lambda_for_correct_persistent_tile_scheduling | 0.950× | 0.289× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_broken_grid_l... | Δ | fix_grid_lambda_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 6370 ns | +100 ns (+1.6%) | 6080 ns | -190 ns (-3.0%) | 21660 ns | +15390 ns (+245.5%) |
| SM Throughput (%) | 39.10 % | 43.54 % | +4.44 pp | 24.13 % | -14.97 pp | 78.38 % | +39.28 pp |
| Memory Throughput (%) | 14.65 % | 14.21 % | -0.44 pp | 15.45 % | +0.80 pp | 15.72 % | +1.07 pp |
| DRAM Throughput (%) | 9.83 % | 9.71 % | -0.12 pp | 10.15 % | +0.32 pp | 2.84 % | -6.99 pp |
| Achieved Occupancy (%) | 59.91 % | 67.43 % | +7.52 pp | 48.75 % | -11.16 pp | 84.42 % | +24.51 pp |
| Registers/Thread | 18.00 register/thread | 30.00 register/thread | +12 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 11.14 | +8.36 |
| No Eligible (%) | 37.76 % | 34.96 % | -2.80 pp | 60.65 % | +22.89 pp | 12.19 % | -25.57 pp |
| IPC Active | 2.48 inst/cycle | 2.63 inst/cycle | +0.15 | 1.60 inst/cycle | -0.88 | 3.50 inst/cycle | +1.02 |

---

## Iteration 3

**Best so far**: 1.0312× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 0.992× | 1.036× | ✓ |  |
| B | eliminate_non_fused_fp32_instructions_via_arithmetic_simplification | 1.010× | 1.021× | ✓ |  |
| C | enable_fma_fusion_by_restructuring_bn_arithmetic | 0.998× | 1.031× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | eliminate_non_fus... | Δ | enable_fma_fusion... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 6050 ns | -220 ns (-3.5%) | 6140 ns | -130 ns (-2.1%) | 6080 ns | -190 ns (-3.0%) |
| SM Throughput (%) | 39.10 % | 24.25 % | -14.85 pp | 30.35 % | -8.75 pp | 23.78 % | -15.32 pp |
| Memory Throughput (%) | 14.65 % | 15.53 % | +0.88 pp | 14.91 % | +0.26 pp | 15.43 % | +0.78 pp |
| DRAM Throughput (%) | 9.83 % | 10.20 % | +0.37 pp | 10.05 % | +0.22 pp | 10.15 % | +0.32 pp |
| Achieved Occupancy (%) | 59.91 % | 48.22 % | -11.69 pp | 53.87 % | -6.04 pp | 48.21 % | -11.70 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 16.00 register/thread | -2 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 37.76 % | 61.21 % | +23.45 pp | 51.25 % | +13.49 pp | 61.87 % | +24.11 pp |
| IPC Active | 2.48 inst/cycle | 1.59 inst/cycle | -0.89 | 1.90 inst/cycle | -0.58 | 1.58 inst/cycle | -0.90 |

---

## Iteration 4

**Best so far**: 1.0312× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_large_tensor_to_improve_channel_cache_reuse | 0.993× | 1.036× | ✓ |  |
| B | streaming_eviction_policy_for_main_tensor_to_reduce_l1tex_stalls | 1.019× | 1.036× | ✓ |  |
| C | fma_fusion_via_algebraic_restructuring_of_bn_arithmetic | 1.008× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | streaming_evictio... | Δ | fma_fusion_via_al... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6270 ns | 6050 ns | -220 ns (-3.5%) | 6050 ns | -220 ns (-3.5%) | 6340 ns | +70 ns (+1.1%) |
| SM Throughput (%) | 39.10 % | 26.75 % | -12.35 pp | 25.69 % | -13.41 pp | 31.01 % | -8.09 pp |
| Memory Throughput (%) | 14.65 % | 15.53 % | +0.88 pp | 14.83 % | +0.18 pp | 15.07 % | +0.42 pp |
| DRAM Throughput (%) | 9.83 % | 10.20 % | +0.37 pp | 10.19 % | +0.36 pp | 9.73 % | -0.10 pp |
| Achieved Occupancy (%) | 59.91 % | 48.71 % | -11.20 pp | 49.64 % | -10.27 pp | 53.97 % | -5.94 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 37.76 % | 54.74 % | +16.98 pp | 54.90 % | +17.14 pp | 51.28 % | +13.52 pp |
| IPC Active | 2.48 inst/cycle | 1.73 inst/cycle | -0.75 | 1.79 inst/cycle | -0.69 | 1.87 inst/cycle | -0.61 |

---

## Cumulative Progress

```
Baseline:  6270 ns  (1.000×)
Iter 2     6080 ns  (1.031×)  — fix_broken_grid_lambda_and_enable_fma_fusion

Final best: 6080 ns  (1.0312×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -18,20 +19,11 @@

     tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
+    tmp_eps = tl.full([1], 1e-05, tl.float32)
+    tmp_inv_std = tl.math.rsqrt(tmp3 + tmp_eps)
+    tmp_norm = (tmp0 - tmp1) * tmp_inv_std
+    tmp_result = tl.math.fma(tmp_norm, tmp12, tmp14)
+    tl.store(in_out_ptr0 + (x2), tmp_result, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
